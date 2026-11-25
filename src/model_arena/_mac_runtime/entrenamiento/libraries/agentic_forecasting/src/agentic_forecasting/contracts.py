"""Explicit experiment axes and adapter capabilities; no accounting concepts."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product


@dataclass(frozen=True)
class Candidate:
    recipe: str
    architecture: str
    inputs: str
    encoder_design: str
    coherence_weight: float
    decoder: str = "direct"
    approach: str = "mixed"

    def __post_init__(self):
        if self.decoder not in ("direct","residual") or self.approach not in ("mixed","curriculum"):
            raise ValueError("Invalid decoder/approach")
        if self.recipe != "mac_modeling":
            raise ValueError("Unsupported recipe")
        if self.architecture not in ("gru", "tft_compact", "tft_full", "mlp"):
            raise ValueError("Unsupported architecture")
        if self.inputs not in ("X0", "X1", "X2", "X3") or self.encoder_design not in (
            "separate",
            "independent",
            "shared",
        ):
            raise ValueError("Invalid candidate")
        if (
            not isinstance(self.coherence_weight, (int, float))
            or not math.isfinite(self.coherence_weight)
            or self.coherence_weight < 0
        ):
            raise ValueError("Negative coherence weight")

    @property
    def id(self):
        return "__".join(
            (
                self.recipe,
                self.architecture,
                self.inputs,
                self.encoder_design,
                f"coh{self.coherence_weight:g}",
                self.decoder, self.approach,
            )
        )


@dataclass(frozen=True)
class ExperimentSpec:
    recipes: tuple = ("mac_modeling",)
    architectures: tuple = ("mlp", "gru", "tft_compact")
    inputs: tuple = ("X0", "X1", "X2", "X3")
    encoder_designs: tuple = ("separate", "shared")
    coherence_weights: tuple = (0.0,)
    decoders: tuple = ("direct","residual")
    approaches: tuple = ("mixed","curriculum")
    window: int = 20
    horizon: int = 20
    validation: int = 20
    calibration: int = 0
    test: int = 20
    cv_folds: int = 1
    backtest_cuts: int = 1
    backtest_stride: int = 20
    epochs: int = 2
    patience: int = 2
    samples_per_epoch: int = 64
    batch_size: int = 16
    mc_samples: int = 20
    calendar_features: int = 7
    cpu_threads: int = 4
    learning_rate: float = 0.001
    seed: int = 42
    coverages: tuple = (0.95,)

    @property
    def evaluation_span(self):
        return self.test+self.horizon+(self.backtest_cuts-1)*self.backtest_stride

    @property
    def minimum_history(self):
        return self.window+self.horizon+self.cv_folds*self.validation+self.evaluation_span

    def __post_init__(self):
        if self.calendar_features < 0:
            raise ValueError("Calendar feature count cannot be negative")
        if self.backtest_cuts<1 or self.backtest_stride<1:
            raise ValueError("Positive backtest sizes required")
        if (
            min(
                self.cpu_threads,
                self.window,
                self.horizon,
                self.validation,
                self.test,
                self.epochs,
                self.samples_per_epoch,
                self.batch_size,
            )
            < 1
        ):
            raise ValueError("Positive sizes required")
        if (
            self.validation < self.horizon
            or self.test != self.horizon
            or ("standard" in self.recipes and self.calibration < self.horizon)
        ):
            raise ValueError("Temporal block sizes incompatible")
        if self.patience < 1 or not self.learning_rate > 0:
            raise ValueError("Invalid optimizer or patience")
        if not self.coverages or len(set(self.coverages)) != len(self.coverages):
            raise ValueError("Empty or duplicate coverages")
        if "standard" in self.recipes and any(
            math.ceil((self.calibration - self.horizon + 2) * c)
            > self.calibration - self.horizon + 1
            for c in self.coverages
        ):
            raise ValueError("Not enough calibration origins for coverage")
        if self.mc_samples < 2:
            raise ValueError("At least two uncertainty samples")
        if set(self.coverages) - {0.8, 0.9, 0.95}:
            raise ValueError("Unsupported coverage")
        if self.recipes != ("mac_modeling",) and list(self.recipes) != ["mac_modeling"]:
            raise ValueError("0.4 requires learned quantiles only")
        if tuple(self.coverages) != (0.95,) or tuple(self.coherence_weights) != (0.0,):
            raise ValueError("0.4 profile requires 95% and coherence disabled")
        if self.calibration != 0 or len({self.window,self.horizon,self.validation,self.test}) != 1:
            raise ValueError("Require calibration=0 and window=validation=test=horizon")
        self.candidates()

    def candidates(self):
        axes = (
            self.recipes,
            self.architectures,
            self.inputs,
            self.encoder_designs,
            self.coherence_weights,
            self.decoders, self.approaches,
        )
        if any(not x or len(set(x)) != len(x) for x in axes):
            raise ValueError("Empty or duplicate axis")
        return [Candidate(*x) for x in product(*axes)]

    def to_dict(self):
        return asdict(self)


@dataclass
class Series:
    id: str
    dates: list
    values: object
    currency: str = "UNKNOWN"
    budget: int = 1
    cohort: str = "seen"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
