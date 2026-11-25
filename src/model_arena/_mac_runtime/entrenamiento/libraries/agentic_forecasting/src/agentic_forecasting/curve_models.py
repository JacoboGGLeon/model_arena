"""TFT de doble cabeza para pronosticar las curvas móviles del Observatorio."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model_arena.encoders import ContextEncoder
from agentic_forecasting.data import batch
from agentic_forecasting.backbones import TFTTemporalBackbone

TARGETS = ("saldo", "variacion")

CURVE_PROBS = (.025, .5, .975)


def persistence(history, horizon):
    values = np.quantile(history, CURVE_PROBS, axis=1).transpose(1, 2, 0)
    return np.repeat(values[:, None], horizon, axis=1)


class CurveTFT(nn.Module):
    def __init__(self, spec, candidate, dropout):
        super().__init__()
        width = candidate.width
        activation = getattr(candidate, "activation", "ELU")
        activations = {"ReLU": nn.ReLU, "Mish": nn.Mish, "GELU": nn.GELU, "Tanh": nn.Tanh, "ELU": nn.ELU}
        self.input_kind = candidate.inputs
        calendar_size = spec.calendar_features
        self.past_calendar = nn.Linear(calendar_size, width) if candidate.inputs in ("X1", "X3") else None
        self.future_calendar = nn.Linear(calendar_size, width) if candidate.inputs in ("X2", "X3") else None
        self.encoder_kind = getattr(candidate, "encoder_kind", "tft_compact")
        self.encoders = nn.ModuleList()
        for _ in range(1 if candidate.encoder_design == "shared" else len(TARGETS)):
            if self.encoder_kind != "tft_compact":
                self.encoders.append(ContextEncoder(kind=self.encoder_kind, window=spec.window,
                    width=width, depth=candidate.depth, dropout=dropout, activation=activations[activation], targets=len(TARGETS)))
                continue
            encoder = TFTTemporalBackbone(width, dropout)
            encoder.input = nn.Linear(len(TARGETS), width)
            encoder.encoder = nn.LSTM(width, width, candidate.depth, batch_first=True)
            for block in (encoder.history_grn, encoder.state_grn, encoder.query_grn, encoder.output_grn):
                block.activation = activations[activation]()
            self.encoders.append(encoder)
        self.positions = nn.ModuleList(nn.Embedding(spec.horizon, width) for _ in self.encoders)
        heads = []
        for _ in TARGETS:
            layers = []
            for _ in range(getattr(candidate, "decoder_depth", 1)):
                layers.extend([nn.Linear(width, width), activations[activation](), nn.Dropout(dropout)])
            heads.append(nn.Sequential(*layers, nn.Linear(width, 3)))
        self.quantiles = nn.ModuleList(heads)

    def forward(self, features):
        history = features["history_y"]
        representations = []
        for enc, pos in zip(self.encoders, self.positions):
            projected = enc.input(history)
            if self.input_kind in ("X1", "X3"):
                projected = projected + self.past_calendar(features["history_calendar"])
            if self.encoder_kind != "tft_compact":
                queries = pos.weight.expand(len(history), -1, -1)
                if self.input_kind in ("X2", "X3"):
                    queries = queries + self.future_calendar(features["future_calendar"])
                representations.append(enc(projected, queries))
                continue
            encoded, (hidden, _) = enc.encoder(projected)
            encoded = enc.history_grn(encoded)
            state = enc.state_grn(hidden[-1])
            queries = pos.weight.expand(len(history), -1, -1)
            if self.input_kind in ("X2", "X3"):
                queries = queries + self.future_calendar(features["future_calendar"])
            queries = enc.query_grn(queries + state[:, None, :])
            attended, _ = enc.attention(queries, encoded, encoded, need_weights=False)
            value, control = enc.attention_gate(attended).chunk(2, dim=-1)
            representations.append(enc.output_grn(enc.attention_norm(queries + value * control.sigmoid())))
        outputs = []
        for i, rep in enumerate((representations[0], representations[-1])):
            mid, lower, upper = self.quantiles[i](rep).split(1, dim=-1)
            outputs.append(torch.cat((mid-F.softplus(lower), mid,
                                      mid+F.softplus(upper)), dim=-1))
        return torch.stack(outputs, dim=2)


def make_batch(series, refs, spec, candidate, device, calendar, observed):
    features = batch(series, refs, spec, candidate, device, {}, observed=False,
                     calendar_table=calendar)
    for key in ("history_calendar", "future_calendar"):
        if key in features:
            features[key] = torch.as_tensor(features[key], dtype=torch.float32, device=device)
    history = np.stack([series[i].values[end-spec.horizon:end] for i, end in refs])
    anchor_native = None if observed else persistence(history, spec.horizon)
    scale = features['scales'][:, None, :, None]

    truth = None
    if observed:
        future = np.stack([series[i].values[end:end+spec.horizon] for i, end in refs])
        truth = torch.as_tensor(future, dtype=torch.float32, device=device) / scale.squeeze(-1)
    return features, anchor_native, truth


def fit_curves(series, end, spec, candidate, device, calendar, config, on_epoch=None, monitor=None):
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    model = CurveTFT(spec, candidate, config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(candidate, "lr", spec.learning_rate),
                                  weight_decay=getattr(candidate, "weight_decay", .01))
    first_origin = max(spec.window, spec.horizon)
    last_origin = end - spec.horizon
    if first_origin > last_origin:
        raise ValueError("Historia insuficiente")
    samples = max(spec.samples_per_epoch, len(series))
    for epoch in range(spec.epochs):
        accounts = np.concatenate((np.arange(len(series)), rng.integers(len(series), size=samples-len(series))))
        schedule = np.column_stack((accounts, rng.integers(first_origin, last_origin+1, size=samples)))
        rng.shuffle(schedule)
        loss_total = 0.
        loss_by_target = np.zeros(2)
        for first in range(0, samples, spec.batch_size):
            features, _, truth = make_batch(
                series, schedule[first:first+spec.batch_size], spec, candidate, device, calendar, True
            )
            optimizer.zero_grad()
            prediction = model(features)
            residual = truth.unsqueeze(-1) - prediction
            probs = prediction.new_tensor(CURVE_PROBS)
            elements = torch.maximum(probs*residual, (probs-1)*residual)
            loss = elements.mean()
            loss_by_target += elements.detach().mean(dim=(0,1,3)).cpu().numpy() * len(truth)
            if not torch.isfinite(loss):
                raise ValueError('Pérdida de entrenamiento no finita')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_total += loss.detach().item() * len(truth)
        epoch_loss = loss_total/samples
        if on_epoch is not None:
            record=dict(epoch=epoch+1, train_loss=epoch_loss, validation_loss=None, train_loss_saldo=float(loss_by_target[0]/samples), train_loss_variacion=float(loss_by_target[1]/samples))
            if monitor is not None:record.update(monitor(model))
            on_epoch(record)
        print(f'Época {epoch+1}/{spec.epochs} · pérdida {epoch_loss:.6g}', flush=True)
    return model


def predict_curves(model, series, end, spec, candidate, device, calendar):
    predicted, reference = [], []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(series), spec.batch_size):
            refs = [(i, end) for i in range(first, min(len(series), first+spec.batch_size))]
            features, native, _ = make_batch(series, refs, spec, candidate, device, calendar, False)
            result = model(features).cpu().numpy().astype(np.float64)
            result *= features['scales'].cpu().numpy()[:, None, :, None]
            if not np.isfinite(result).all():
                raise ValueError('Pronóstico no finito')
            predicted.append(result)
            reference.append(native)
    return np.concatenate(predicted), np.concatenate(reference)
