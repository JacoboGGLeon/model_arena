"""Serializable public contracts. Unknown parameters are rejected."""
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PanelSchema(Contract):
    id_col: str = "unique_id"
    time_col: str = "ds"
    target_cols: list[str] = Field(default_factory=lambda: ["y"])
    population_col: str | None = None
    training_label: str = "seen"
    group_cols: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct(self):
        names = [self.id_col, self.time_col, *self.target_cols]
        if not self.target_cols or len(names) != len(set(names)):
            raise ValueError("Identifiers, time and targets must be distinct; targets cannot be empty")
        return self


class TemporalCV(Contract):
    n_windows: int = Field(default=3, ge=1)
    step_size: int | None = Field(default=None, ge=1)
    refit: Literal[True] = True
    evidence_role: Literal["selection", "independent_test"] = "selection"


class ModelSpec(Contract):
    alias: str
    kind: str = "tft_compact"
    historical_features: list[str] = Field(default_factory=list)
    future_features: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingSpec(Contract):
    epochs: int = Field(default=20, ge=1)
    final_epochs: int = Field(default=75, ge=1)
    batch_size: int = Field(default=128, ge=1)
    samples_per_epoch: int = Field(default=4096, ge=1)
    monitor_accounts: int | None = Field(default=None, ge=1)
    device: str = "cpu"
    seed: int = 42
    cpu_threads: int = Field(default=4, ge=1)


class SearchSpec(Contract):
    n_trials: int = Field(default=15, ge=1)
    encoder_design: list[Literal["shared", "independent"]] = ["shared", "independent"]
    depth: list[int] = [2, 3]
    decoder_depth: list[int] = [1, 2]
    width: list[int] = [32, 64, 128]
    activation: list[str] = ["ReLU", "Mish", "GELU", "Tanh"]
    lr_range: tuple[float, float] = (1e-8, 1e-2)
    wd_range: tuple[float, float] = (1e-6, 1e-2)
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def valid_space(self):
        for values in (self.encoder_design, self.depth, self.decoder_depth, self.width, self.activation):
            if not values:
                raise ValueError("Search dimensions cannot be empty")
        for values in (self.depth, self.decoder_depth, self.width):
            if min(values) < 1:
                raise ValueError("Layer depths and widths must be positive")
        for low, high in (self.lr_range, self.wd_range):
            if not 0 < low < high:
                raise ValueError("Log search ranges require 0 < low < high")
        return self


class SelectionPolicy(Contract):
    by: str = "unique_id"
    min_gain: float = Field(default=0.05, ge=0, lt=1)
    min_accounts: int = Field(default=3, ge=1)
    group_cols: list[str] = Field(default_factory=list)


class ArenaSpec(Contract):
    schema_version: Literal["1"] = "1"
    panel_schema: PanelSchema = Field(default_factory=PanelSchema, alias="schema")
    models: list[ModelSpec]
    h: int = Field(default=20, ge=1)
    window: int = Field(default=20, ge=2)
    freq: str | None = None
    calendar: list[str] | None = None
    quantiles: list[float] = [0.025, 0.5, 0.975]
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    search: SearchSpec = Field(default_factory=SearchSpec)
    selection: SelectionPolicy = Field(default_factory=SelectionPolicy)
    cv: TemporalCV = Field(default_factory=TemporalCV)
    artifacts: str = "runs"

    @model_validator(mode="after")
    def compatible(self):
        if not self.models or len({m.alias for m in self.models}) != len(self.models):
            raise ValueError("At least one model and unique aliases are required")
        if self.quantiles != sorted(set(self.quantiles)) or not all(0 < q < 1 for q in self.quantiles):
            raise ValueError("Quantiles must be unique, increasing and between zero and one")
        if self.quantiles != [0.025, 0.5, 0.975] and any(m.kind == "tft_compact" for m in self.models):
            raise ValueError("This TFT implementation supports quantiles .025, .5, .975")
        if self.freq is None and self.calendar is None:
            raise ValueError("Declare a frequency or an authoritative calendar")
        return self

    @property
    def schema(self):
        return self.panel_schema

    def model_dump(self, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    @classmethod
    def from_yaml(cls, path):
        import yaml
        path = Path(path)
        return cls.model_validate(yaml.safe_load(path.read_text()))


class RunRef(Contract):
    run_id: str
    path: str
    signature: str


class StudyResult(RunRef):
    evidence_role: str = "selection"


class SelectionResult(RunRef):
    study_id: str


class ArtifactRef(Contract):
    path: str
    signature: str


class Publication(Contract):
    publication_id: str
    path: str
    manifest_sha256: str

