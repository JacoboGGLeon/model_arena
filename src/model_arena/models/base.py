"""Operational interface for replaceable model implementations."""
from typing import Protocol


class ForecastModel(Protocol):
    def fit(self, data, *, end, epochs, fold, on_epoch): ...
    def predict(self, data, *, origins, future=None): ...
    def save(self, path): ...


class ModelRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, kind, factory, *, replace=False):
        if kind in self._factories and not replace:
            raise ValueError("Model already registered: " + kind)
        self._factories[kind] = factory

    def create(self, model_spec, arena_spec):
        kind = model_spec.kind
        if kind in ("tft_compact", "encoder_decoder") and kind not in self._factories:
            from .tft import TFTCompact
            return TFTCompact(model_spec, arena_spec)
        if kind in ("tide", "nhits", "nlinear", "dlinear", "nbeats", "rlinear") and kind not in self._factories:
            from .complete import CompleteModel
            return CompleteModel(model_spec, arena_spec)
        if kind == "historical_quantile" and kind not in self._factories:
            from .naive import HistoricalQuantile
            return HistoricalQuantile(model_spec, arena_spec)
        if kind not in self._factories:
            raise ValueError("No registered implementation: " + kind)
        return self._factories[kind](model_spec, arena_spec)

