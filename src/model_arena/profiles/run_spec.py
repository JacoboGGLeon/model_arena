"""Declarative groups for independent MAC experiments."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, JsonValue, model_validator

from ..contracts import Contract

PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]


class GroupMember(Contract):
    model: Identifier
    inputs: list[Identifier] = Field(min_length=1)
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_inputs(self):
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Input names must be unique per model")
        return self


class ResidualSpec(Contract):
    baseline: Literal["last", "zero", "learned"]
    widths: list[PositiveInt] = Field(default_factory=list)
    base_weight: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_widths(self):
        if self.baseline == "learned" and (not self.widths or len(set(self.widths)) != len(self.widths)):
            raise ValueError("Declare unique residual widths for a learned baseline")
        return self


class TreeSearch(Contract):
    learning_rate: tuple[float, float] = (0.01, 0.15)
    num_leaves: list[PositiveInt] = Field(default_factory=lambda: [15, 31, 63])
    min_data_in_leaf: list[PositiveInt] = Field(default_factory=lambda: [32, 64, 128])
    lambda_l2: tuple[float, float] = (0.001, 10)
    patience: PositiveInt = 3
    threads: PositiveInt = 4
    max_rows: PositiveInt = 1_000_000

    @model_validator(mode="after")
    def validate_search(self):
        if any(not 0 < lower < upper for lower, upper in (self.learning_rate, self.lambda_l2)):
            raise ValueError("Declare positive increasing log-search limits")
        for values in (self.num_leaves, self.min_data_in_leaf):
            if not values or len(set(values)) != len(values):
                raise ValueError("Declare nonempty unique tree search choices")
        if min(self.num_leaves) < 2:
            raise ValueError("num_leaves must be at least two")
        return self


class RunSpec(Contract):
    name: Identifier
    models: list[GroupMember] = Field(min_length=1)
    objectives: dict[Identifier, PositiveInt] = Field(min_length=1)
    sampler_startup: PositiveInt = 3
    residuals: dict[Identifier, ResidualSpec] = Field(default_factory=dict)
    tree_search: TreeSearch | None = None

    @classmethod
    def from_yaml(cls, path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @model_validator(mode="after")
    def validate_allocation(self):
        if len({member.model for member in self.models}) != len(self.models):
            raise ValueError("A group must declare each model once")
        if self.residuals and set(self.residuals) != set(self.objectives):
            raise ValueError("Declare one baseline per objective when using residuals")
        for objective, budget in self.objectives.items():
            per_model, remainder = divmod(budget, len(self.models))
            if remainder or any(per_model == 0 or per_model % len(model.inputs) for model in self.models):
                raise ValueError(f"{objective}: budget must split equally across models and their inputs")
        return self

    def curves(self, base):
        """Compile a profile into the legacy execution plan without sharing run outputs."""
        models = {member.model: member.inputs for member in self.models}
        reference_budget = next(iter(self.objectives.values())) // len(models)
        return {
            **base,
            "objectives": list(self.objectives),
            "inputs": list(dict.fromkeys(alias for inputs in models.values() for alias in inputs)),
            "families": dict.fromkeys(models, reference_budget),
            "family_inputs": models,
            "objective_budgets": self.objectives,
            "model_options": {member.model: member.options for member in self.models},
            "input_search": {"mode": "balanced", "tpe_startup_trials": self.sampler_startup},
            "run_spec": self.model_dump(mode="json"),
        }
