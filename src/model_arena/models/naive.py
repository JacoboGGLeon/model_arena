"""A dependency-light baseline also used to verify domain agnosticism."""
import numpy as np
from ..storage import write_json


class HistoricalQuantile:
    def __init__(self, model_spec, arena_spec):
        if model_spec.params:
            raise ValueError("HistoricalQuantile has no tunable parameters")
        self.spec = arena_spec
        self.model_spec = model_spec

    def fit(self, data, *, end, epochs, fold, on_epoch):
        if end < self.spec.window:
            raise ValueError("Insufficient history")
        return self

    def predict(self, data, *, origins, future=None):
        result = []
        for entity, origin in origins:
            history = data.values[entity][origin-self.spec.window:origin]
            if len(history) != self.spec.window:
                raise ValueError("Insufficient history")
            q = np.quantile(history, self.spec.quantiles, axis=0).T
            result.append(np.repeat(q[None], self.spec.h, axis=0))
        return np.stack(result)

    def save(self, path):
        write_json(path, {"kind": "historical_quantile"})

    def restore(self, path):
        return self

