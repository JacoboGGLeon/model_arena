from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import TFTTemporalBackbone

TARGETS = ("saldo", "variacion")
BACKBONES = ("gru", "lstm", "tft", "mlp")


class PairCovariates(nn.Module):
    """Calendar-only additions. X0 is always the historical target baseline."""
    def __init__(self, bundle, window, currency_vocab=1, calendar_features=7):
        super().__init__()
        if bundle not in ("X1", "X2", "X3"):
            raise ValueError("unsupported calendar bundle")
        self.bundle = bundle
        if bundle in ("X1", "X3"):
            self.context = nn.Sequential(nn.Linear(window * calendar_features, 96), nn.GELU(), nn.Linear(96, 96))
        if bundle in ("X2", "X3"):
            self.future_calendar = nn.Linear(calendar_features, 96)

    def forward(self, batch):
        past = self.context(batch["history_calendar"].flatten(1))[:, None] if self.bundle in ("X1", "X3") else None
        future = self.future_calendar(batch["future_calendar"]) if self.bundle in ("X2", "X3") else None
        return future if past is None else past if future is None else past + future


class PairEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        window: int,
        horizon: int,
        dropout: float,
        input_bundle="X0",
        currency_vocab=1,
        calendar_features=7,
    ):
        super().__init__()
        self.backbone, self.horizon = backbone, horizon
        self.covariates = (
            PairCovariates(input_bundle, window, currency_vocab, calendar_features)
            if input_bundle != "X0"
            else None
        )
        self.position = nn.Embedding(horizon, 96)
        self.norm = nn.LayerNorm(96)
        self.dropout = nn.Dropout(dropout)
        if backbone == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(window * 2, 192),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(192, 96),
                nn.GELU(),
            )
        elif backbone == "tft":
            self.encoder = TFTTemporalBackbone(96, dropout)
            # Reuse the compact TFT mechanics, with an explicit two-channel input.
            self.encoder.input = nn.Linear(2, 96)
        else:
            recurrent = {"gru": nn.GRU, "lstm": nn.LSTM}[backbone]
            self.encoder = recurrent(2, 96, batch_first=True)

    def forward(self, history: torch.Tensor, batch=None) -> torch.Tensor:
        query = self.position.weight[None].expand(history.shape[0], -1, -1)
        if self.covariates is not None:
            query = query + self.covariates(batch)
        if self.backbone == "tft":
            _, tokens = self.encoder(history, query)
            return self.dropout(self.norm(tokens))
        if self.backbone == "mlp":
            state = self.encoder(history)
        else:
            _, state = self.encoder(history)
            state = state[0] if self.backbone == "lstm" else state
            state = state[-1]
        return self.dropout(self.norm(state[:, None] + query))


def head(dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(96, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1)
    )


class BalanceVariationModel(nn.Module):
    """A: two disjoint networks; B: shared representation and two output heads."""

    def __init__(
        self,
        experiment: str,
        backbone: str,
        window=25,
        horizon=20,
        dropout=0.1,
        input_bundle="X0",
        currency_vocab=1,
        calendar_features=7,
        coherence_weight=0.0,
    ):
        super().__init__()
        self.coherence_weight = coherence_weight
        if experiment not in ("A", "B") or backbone not in BACKBONES:
            raise ValueError("invalid paired experiment/backbone")
        self.experiment, self.window, self.horizon = experiment, window, horizon
        self.input_bundle = input_bundle
        self.encoders = nn.ModuleList(
            PairEncoder(
                backbone,
                window,
                horizon,
                dropout,
                input_bundle,
                currency_vocab,
                calendar_features,
            )
            for _ in range(2 if experiment == "A" else 1)
        )
        self.heads = nn.ModuleList(head(dropout) for _ in TARGETS)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history_y"]
        if history.ndim != 3 or history.shape[1:] != (self.window, 2):
            raise ValueError(f"paired history_y must be [B,{self.window},2]")
        representations = [encoder(history, batch) for encoder in self.encoders]
        if getattr(self,"decoder", "direct")=="residual":
            self.point_scaled=torch.cat([
                layer(representations[i if self.experiment=="A" else 0].detach())
                for i,layer in enumerate(self.point_heads)
            ],dim=-1)
            global_values=[]; residual_values=[]
            for i in range(2):
                rep=representations[i if self.experiment=="A" else 0]
                centre=self.global_heads[i](rep)
                local=self.residual_heads[i](torch.cat([rep.detach(),centre.detach()],dim=-1))
                global_values.append(centre.squeeze(-1));residual_values.append(local)
            self.last_global=torch.stack(global_values,dim=-1)
            residual=torch.stack(residual_values,dim=-2).sort(dim=-1).values
            return self.last_global.detach().unsqueeze(-1)+residual
        self.last_global=None
        if getattr(self, "quantile_mode", False):
            return (
                torch.stack(
                    [
                        output(representations[i if self.experiment == "A" else 0])
                        for i, output in enumerate(self.heads)
                    ],
                    dim=-2,
                )
                .sort(dim=-1)
                .values
            )
        return torch.cat(
            [
                output(representations[i if self.experiment == "A" else 0])
                for i, output in enumerate(self.heads)
            ],
            dim=-1,
        )


def coherence_loss(
    prediction: torch.Tensor, scales: torch.Tensor, last_balance: torch.Tensor
):
    """Differentiable consistency in the variation head's transformed units."""
    balance = torch.sinh(prediction[..., 0]) * scales[:, None, 0]
    previous = torch.cat([last_balance[:, None], balance[:, :-1]], dim=1)
    implied = torch.asinh((balance - previous) / scales[:, None, 1])
    return F.smooth_l1_loss(prediction[..., 1], implied)


QUANTILES = (0.025, 0.5, 0.975)


def objective(prediction, batch, experiment: str, coherence_weight=None, global_prediction=None):
    weight = (
        (0.1 if experiment == "B" else 0.0)
        if coherence_weight is None
        else coherence_weight
    )
    if prediction.ndim == 4:
        q = prediction.new_tensor(QUANTILES)
        error = batch["future_y"].unsqueeze(-1) - prediction
        pinball = torch.maximum(q * error, (q - 1) * error)
        saldo, variation = pinball[..., 0, :].mean(), pinball[..., 1, :].mean()
        coherent = coherence_loss(
            prediction[..., 1], batch["scales"], batch["last_balance"]
        )
        loss = 0.5 * saldo + 0.5 * variation + weight * coherent
        if global_prediction is not None:
            # The same translation defines the residual pinball objective; stop global gradients.
            loss=0.5*F.l1_loss(global_prediction,batch["future_y"])+0.5*loss
        return loss, {
            "saldo_loss": saldo,
            "variation_loss": variation,
            "coherence_loss": coherent,
        }

    saldo = F.l1_loss(prediction[..., 0], batch["future_y"][..., 0])
    variation = F.l1_loss(prediction[..., 1], batch["future_y"][..., 1])
    coherent = coherence_loss(prediction, batch["scales"], batch["last_balance"])
    loss = 0.5 * saldo + 0.5 * variation + weight * coherent
    return loss, {
        "saldo_loss": saldo,
        "variation_loss": variation,
        "coherence_loss": coherent,
    }
