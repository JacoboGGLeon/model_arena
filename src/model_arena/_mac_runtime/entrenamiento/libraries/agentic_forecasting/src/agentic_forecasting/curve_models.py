"""TFT de doble cabeza para pronosticar las curvas móviles del Observatorio."""

import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from model_arena.mad import mad_envelopes, point_forecast
from model_arena.models.engine import predict_batches, train_quantiles
from model_arena.models.tft import Network, TFTParameters
from agentic_forecasting.data import batch
from agentic_forecasting.backbones import TFTTemporalBackbone

TARGETS = ("saldo", "variacion")

CURVE_PROBS = (.025, .5, .975)


def persistence(history, horizon):
    values = np.quantile(history, CURVE_PROBS, axis=1).transpose(1, 2, 0)
    return np.repeat(values[:, None], horizon, axis=1)


def legacy_tft_encoder(targets, width, depth, dropout, activation):
    """Preserve MAC's seeded initialization sequence."""
    encoder = TFTTemporalBackbone(width, dropout)
    encoder.input = nn.Linear(targets, width)
    encoder.encoder = nn.LSTM(width, width, depth, batch_first=True)
    for block in (encoder.history_grn, encoder.state_grn, encoder.query_grn, encoder.output_grn):
        block.activation = activation()
    return encoder


class CurveTFT(nn.Module):
    def __init__(self, spec, candidate, dropout):
        super().__init__()
        self.target_names = (candidate.target,) if getattr(candidate, "target", None) else TARGETS
        targets = len(self.target_names)
        declared = getattr(candidate, "input_features", None)
        past = len(declared["historical"]) if declared is not None else (
            spec.calendar_features if candidate.inputs in ("X1", "X3") else 0
        )
        future = len(declared["future"]) if declared is not None else (
            spec.calendar_features if candidate.inputs in ("X2", "X3") else 0
        )
        if getattr(candidate, "family", "encoder_decoder") != "encoder_decoder":
            from model_arena.models.complete import CompleteNetwork, CompleteParameters

            self.external = CompleteNetwork(
                family=candidate.family, targets=targets, window=spec.window,
                h=spec.horizon, historical=past, future=future,
                parameters=CompleteParameters.model_validate(candidate.family_params),
                point=point_forecast(candidate),
            )
        else:
            params = TFTParameters(
                encoder_kind=getattr(candidate, "encoder_kind", "tft_compact"),
                encoder_design="independent" if candidate.encoder_design == "independent" else "shared",
                depth=candidate.depth, decoder_depth=getattr(candidate, "decoder_depth", 1),
                width=candidate.width, activation=getattr(candidate, "activation", "ELU"),
                dropout=dropout,
            )
            self.network = Network(targets, spec.horizon, past, future, params, window=spec.window,
                                   tft_encoder=legacy_tft_encoder)

    @property
    def encoders(self):
        return self.network.encoders

    @property
    def quantiles(self):
        return self.network.quantiles

    def forward(self, features):
        model = self.external if hasattr(self, "external") else self.network
        return model(features["history_y"], features.get("history_calendar"), features.get("future_calendar"))


def make_batch(series, refs, spec, candidate, device, calendar, observed):
    declared = getattr(candidate, "input_features", None)
    if declared is None:
        batch_candidate = candidate
    else:
        legacy_input = {(False, False): "X0", (True, False): "X1",
                        (False, True): "X2", (True, True): "X3"}
        batch_candidate = SimpleNamespace(inputs=legacy_input[bool(declared["historical"]), bool(declared["future"])])
    features = batch(series, refs, spec, batch_candidate, device, {}, observed=False,
                     calendar_table=calendar)
    if declared is not None:
        for key, period in (("history_calendar", "historical"), ("future_calendar", "future")):
            if declared[period]:
                features[key] = features[key][..., [calendar.names.index(name) for name in declared[period]]]
    if getattr(candidate, "transform", None) == "linear_history" and getattr(candidate, "family", "encoder_decoder") == "encoder_decoder":
        features["history_y"] = features["history_y"].sinh()
    for key in ("history_calendar", "future_calendar"):
        if key in features:
            features[key] = torch.as_tensor(features[key], dtype=torch.float32, device=device)
    anchor_native = None
    if not observed and not point_forecast(candidate):
        history = np.stack([series[i].values[end-spec.horizon:end] for i, end in refs])
        anchor_native = persistence(history, spec.horizon)
    scale = features['scales'][:, None, :, None]

    truth = None
    if observed:
        future = np.stack([series[i].values[end:end+spec.horizon] for i, end in refs])
        truth = torch.as_tensor(future, dtype=torch.float32, device=device) / scale.squeeze(-1)
    if getattr(candidate, 'target', None):
        index = TARGETS.index(candidate.target)
        features['history_y'] = features['history_y'][..., index:index+1]
        features['scales'] = features['scales'][..., index:index+1]
        if truth is not None:truth = truth[..., index:index+1]
        if anchor_native is not None:anchor_native = anchor_native[..., index:index+1, :]
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
    def schedule():
        accounts = np.concatenate((np.arange(len(series)), rng.integers(len(series), size=samples-len(series))))
        refs = np.column_stack((accounts, rng.integers(first_origin, last_origin+1, size=samples)))
        rng.shuffle(refs)
        return refs

    def batch_training(refs):
        features, _, truth = make_batch(series, refs, spec, candidate, device, calendar, True)
        return (features,), truth

    def record(epoch, loss, averages):
        if on_epoch is not None:
            by_target = dict(zip(model.target_names, map(float, averages)))
            row = dict(epoch=epoch, train_loss=loss, validation_loss=None,
                       **{f"train_loss_{target}": by_target.get(target) for target in TARGETS})
            if monitor is not None:
                row.update(monitor(model))
            on_epoch(row)
        print(f'Época {epoch}/{spec.epochs} · pérdida {loss:.6g}', flush=True)

    train_quantiles(model, optimizer, schedule, epochs=spec.epochs, batch_size=spec.batch_size,
                    quantiles=CURVE_PROBS, batch=batch_training, on_epoch=record, accumulate_double=False,
                    point=point_forecast(candidate))
    return model


def predict_curves(model, series, end, spec, candidate, device, calendar):
    def batch_prediction(refs):
        features, native, _ = make_batch(series, refs, spec, candidate, device, calendar, False)
        return (features,), features["scales"].cpu().numpy()[:, None, :, None], native

    values, baseline = predict_batches(model, [(i, end) for i in range(len(series))],
                                       batch_size=spec.batch_size, batch=batch_prediction)
    if not point_forecast(candidate):
        return values, baseline
    interval = candidate.interval
    history = np.stack([s.values[max(0, end - interval['window']):end, 0] for s in series])
    baseline = np.zeros_like(values)
    if candidate.target == "saldo":
        baseline[:, :, 0, 0] = history[:, -1, None]
    arguments = dict(target=candidate.target, window=interval['window'], multiplier=interval['multiplier'])
    return mad_envelopes(values, history, **arguments), mad_envelopes(baseline, history, **arguments)
