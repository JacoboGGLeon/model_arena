"""Compact TFT, generalized target/feature dimensions; no domain column names."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pydantic import Field

from ..contracts import Contract
from ..data import scale_for
from ..encoders import ContextEncoder
from .engine import predict_batches, train_quantiles
from typing import Literal


class TFTParameters(Contract):
    encoder_kind: Literal["mlp", "gru", "tft_compact"] = "tft_compact"
    encoder_design: str = "shared"
    depth: int = Field(default=2, ge=1)
    decoder_depth: int = Field(default=1, ge=1)
    width: int = Field(default=64, ge=4)
    activation: str = "GELU"
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.00001, ge=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)


class GRN(nn.Module):
    def __init__(self, width, dropout, activation):
        super().__init__()
        self.activation = activation()
        self.dense_in = nn.Linear(width, width)
        self.dense_out = nn.Linear(width, width)
        self.gate = nn.Linear(width, 2 * width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, value):
        hidden = self.dropout(self.dense_out(self.activation(self.dense_in(value))))
        gated, control = self.gate(hidden).chunk(2, dim=-1)
        return self.norm(value + gated * control.sigmoid())


class Encoder(nn.Module):
    def __init__(self, targets, width, depth, dropout, activation):
        super().__init__()
        self.input = nn.Linear(targets, width)
        self.encoder = nn.LSTM(width, width, depth, batch_first=True)
        self.history_grn = GRN(width, dropout, activation)
        self.state_grn = GRN(width, dropout, activation)
        self.query_grn = GRN(width, dropout, activation)
        self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.attention_gate = nn.Linear(width, 2 * width)
        self.attention_norm = nn.LayerNorm(width)
        self.output_grn = GRN(width, dropout, activation)


class Network(nn.Module):
    def __init__(self, targets, h, historical, future, params, window=None, *, tft_encoder=Encoder):
        super().__init__()
        activations = {"ReLU": nn.ReLU, "Mish": nn.Mish, "GELU": nn.GELU, "Tanh": nn.Tanh, "ELU": nn.ELU}
        if params.activation not in activations or params.encoder_design not in ("shared", "independent"):
            raise ValueError("Unsupported activation or encoder design")
        if params.encoder_kind == "tft_compact" and params.width % 4:
            raise ValueError("TFT width must be divisible by four attention heads")
        width, act = params.width, activations[params.activation]
        self.past_calendar = nn.Linear(historical, width) if historical else None
        self.future_calendar = nn.Linear(future, width) if future else None
        count = 1 if params.encoder_design == "shared" else targets
        self.encoder_kind = params.encoder_kind
        self.encoders = nn.ModuleList(
            tft_encoder(targets, width, params.depth, params.dropout, act)
            if params.encoder_kind == "tft_compact" else ContextEncoder(
                kind=params.encoder_kind, window=window, width=width, depth=params.depth,
                dropout=params.dropout, activation=act, targets=targets) for _ in range(count))
        self.positions = nn.ModuleList(nn.Embedding(h, width) for _ in range(count))
        heads = []
        for _ in range(targets):
            layers = []
            for _ in range(params.decoder_depth):
                layers.extend([nn.Linear(width, width), act(), nn.Dropout(params.dropout)])
            heads.append(nn.Sequential(*layers, nn.Linear(width, 3)))
        self.quantiles = nn.ModuleList(heads)

    def forward(self, history, past, future):
        representations = []
        for enc, pos in zip(self.encoders, self.positions):
            projected = enc.input(history)
            if self.past_calendar is not None:
                projected = projected + self.past_calendar(past)
            if self.encoder_kind != "tft_compact":
                queries = pos.weight.expand(len(history), -1, -1)
                if self.future_calendar is not None:
                    queries = queries + self.future_calendar(future)
                representations.append(enc(projected, queries))
                continue
            encoded, (hidden, _) = enc.encoder(projected)
            encoded = enc.history_grn(encoded)
            state = enc.state_grn(hidden[-1])
            queries = pos.weight.expand(len(history), -1, -1)
            if self.future_calendar is not None:
                queries = queries + self.future_calendar(future)
            queries = enc.query_grn(queries + state[:, None])
            attended, _ = enc.attention(queries, encoded, encoded, need_weights=False)
            value, control = enc.attention_gate(attended).chunk(2, dim=-1)
            representations.append(enc.output_grn(enc.attention_norm(queries + value * control.sigmoid())))
        outputs = []
        for i, head in enumerate(self.quantiles):
            rep = representations[0] if len(representations) == 1 else representations[i]
            mid, lower, upper = head(rep).split(1, dim=-1)
            outputs.append(torch.cat((mid-F.softplus(lower), mid, mid+F.softplus(upper)), dim=-1))
        return torch.stack(outputs, dim=2)


class TFTCompact:
    def __init__(self, model_spec, arena_spec):
        self.model_spec, self.spec = model_spec, arena_spec
        self.params = TFTParameters.model_validate(model_spec.params)
        self._feature_cache = {}
        self.device = torch.device(arena_spec.training.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; choose device explicitly")
        torch.set_num_threads(arena_spec.training.cpu_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arena_spec.training.seed)
            self.network = Network(len(arena_spec.schema.target_cols), arena_spec.h,
                                   len(model_spec.historical_features), len(model_spec.future_features),
                                   self.params, window=arena_spec.window).to(self.device)

    def _batch(self, data, origins, future=None, truth=False):
        histories, scales, pasts, futures, targets = [], [], [], [], []
        features = tuple(dict.fromkeys(self.model_spec.historical_features + self.model_spec.future_features))
        cache_key = (data.signature, features)
        if cache_key not in self._feature_cache:
            self._feature_cache[cache_key] = {
                entity: group[list(features)].to_numpy(dtype=np.float32)
                for entity, group in data.frame.groupby(data.schema.id_col, sort=False)
            } if features else {}
        arrays = self._feature_cache[cache_key]
        for entity, origin in origins:
            history = data.values[entity][origin-self.spec.window:origin]
            scale = scale_for(history)
            histories.append(np.arcsinh(history/scale))
            scales.append(scale)
            if self.model_spec.historical_features:
                indices = [features.index(f) for f in self.model_spec.historical_features]
                pasts.append(arrays[entity][origin-self.spec.window:origin][:, indices])
            if self.model_spec.future_features:
                if origin+self.spec.h <= len(data.axis):
                    indices = [features.index(f) for f in self.model_spec.future_features]
                    futures.append(arrays[entity][origin:origin+self.spec.h][:, indices])
                else:
                    cov = future[future[data.schema.id_col] == entity].sort_values(data.schema.time_col)
                    futures.append(cov[self.model_spec.future_features].to_numpy())
            if truth:
                targets.append(data.values[entity][origin:origin+self.spec.h] / scale)
        def tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)
        return (tensor(histories), tensor(pasts) if pasts else None,
                tensor(futures) if futures else None, np.asarray(scales), tensor(targets) if truth else None)

    def _losses(self, data, origins):
        result = []
        previous = self.network.training
        self.network.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(origins), self.spec.training.batch_size):
                    refs = origins[start:start+self.spec.training.batch_size]
                    history, past, future, _, truth = self._batch(data, refs, truth=True)
                    residual = truth.unsqueeze(-1) - self.network(history, past, future)
                    q = truth.new_tensor(self.spec.quantiles)
                    loss = torch.maximum(q*residual, (q-1)*residual).double().mean(dim=(1, 3))
                    for (entity, _), values in zip(refs, loss.cpu().numpy()):
                        result.append((entity, values))
        finally:
            self.network.train(previous)
        return result

    def fit(self, data, *, end, epochs, fold, on_epoch):
        train = self.spec.training
        if end < self.spec.window + self.spec.h:
            raise ValueError("Insufficient causal training history")
        torch.manual_seed(train.seed)
        rng = np.random.default_rng(train.seed)
        monitor_rng = np.random.default_rng(train.seed + 901)
        ids = data.training_ids
        monitored = ids if train.monitor_accounts is None else sorted(
            monitor_rng.choice(ids, min(len(ids), train.monitor_accounts), replace=False).tolist())
        fixed = [(key, int(monitor_rng.integers(self.spec.window, end-self.spec.h+1))) for key in monitored]
        validation = [(key, end) for key in data.ids] if fold != "FUTURO" else []
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.params.lr,
                                      weight_decay=self.params.weight_decay)
        samples = max(train.samples_per_epoch, len(ids))
        def schedule():
            chosen = np.concatenate((np.arange(len(ids)), rng.integers(len(ids), size=samples-len(ids))))
            refs = [(ids[i], int(rng.integers(self.spec.window, end-self.spec.h+1))) for i in chosen]
            rng.shuffle(refs)
            return refs

        def batch(refs):
            history, past, future, _, truth = self._batch(data, refs, truth=True)
            return (history, past, future), truth

        def record(epoch, _, averages):
            rows = []
            for target, value in zip(self.spec.schema.target_cols, averages):
                rows.append(dict(epoch=epoch, target=target, split="train_sampled", value=float(value),
                                 scope="total", entity=None, n_observations=samples*self.spec.h*3))
            for split, refs in [("train_fixed", fixed), ("validation", validation)]:
                for entity, values in self._losses(data, refs):
                    for target, value in zip(self.spec.schema.target_cols, values):
                        rows.append(dict(epoch=epoch, target=target, split=split, value=float(value),
                                         scope="entity", entity=entity, n_observations=self.spec.h*3))
            on_epoch(rows)

        train_quantiles(self.network, optimizer, schedule, epochs=epochs, batch_size=train.batch_size,
                        quantiles=self.spec.quantiles, batch=batch, on_epoch=record)
        self.monitor = {"train_fixed": fixed, "validation": validation}
        return self

    def predict(self, data, *, origins, future=None):
        def batch(refs):
            history, past, futr, scales, _ = self._batch(data, refs, future=future)
            return (history, past, futr), scales[:, None, :, None], None

        result, _ = predict_batches(self.network, origins, batch_size=self.spec.training.batch_size, batch=batch)
        return result

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def restore(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self

