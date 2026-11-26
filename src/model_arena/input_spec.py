"""Resolve named input configurations against the frozen calendar."""

import re
from typing import Literal

from pydantic import Field, model_validator

from .contracts import Contract


class InputSearchSpec(Contract):
    mode: Literal["per_input", "family_objective", "balanced"] = "per_input"
    startup_trials_per_input: int = Field(default=4, ge=1, strict=True)
    tpe_startup_trials: int | None = Field(default=None, ge=1, strict=True)


class InputSpec(Contract):
    historical: list[str]
    future: list[str]

    @model_validator(mode="after")
    def unique_features(self):
        for names in (self.historical, self.future):
            if len(names) != len(set(names)):
                raise ValueError("Cada tramo del input debe tener columnas únicas")
        return self


def resolve_inputs(names, declared, calendar_features):
    if not names or len(set(names)) != len(names):
        raise ValueError("Declara inputs únicos")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", name) for name in names):
        raise ValueError("Usa letras, números o guiones en el nombre del input; debe empezar con letra o número")
    defaults = {
        "X0": {"historical": [], "future": []},
        "X1": {"historical": calendar_features, "future": []},
        "X2": {"historical": [], "future": calendar_features},
        "X3": {"historical": calendar_features, "future": calendar_features},
    }
    declared = defaults if declared is None else declared
    result = {}
    for name in names:
        if name not in declared:
            raise ValueError(f"Falta la declaración del input {name}")
        resolved = InputSpec.model_validate(declared[name])
        missing = set(resolved.historical + resolved.future) - set(calendar_features)
        if missing:
            raise ValueError(f"Columnas ausentes del calendario para {name}: {sorted(missing)}")
        if declared is defaults and name in ("X1", "X2", "X3") and not calendar_features:
            raise ValueError(f"El input {name} requiere features en el calendario")
        result[name] = resolved.model_dump()
    return result


def study_plan(curves, epochs, *, folds=3):
    from .profiles.forecasters import model_definitions

    definitions_by_model = model_definitions()

    search = InputSearchSpec.model_validate(curves.get("input_search", {}))
    sampler_settings = ({"tpe_startup_trials": search.tpe_startup_trials}
                        if search.tpe_startup_trials is not None else {})
    definitions = curves.get("input_specs")
    families = curves["families"] if "families" in curves else {"encoder_decoder": curves["trials"]}
    requested = curves.get("family_inputs", {})
    for family, inputs in requested.items():
        if family not in families or not inputs or set(inputs) - set(curves["inputs"]):
            raise ValueError(f"Combinaciones inválidas para {family}: {inputs}")
    studies = []
    for name in curves["inputs"]:
        features = definitions[name] if definitions is not None else {
            "historical": ["*"] if name in ("X1", "X3") else [],
            "future": ["*"] if name in ("X2", "X3") else [],
        }
        for family, trials in families.items():
            if family in definitions_by_model:
                try:
                    definitions_by_model[family].supports(len(features["historical"]), len(features["future"]))
                except ValueError:
                    if "family_inputs" in curves and name in curves["family_inputs"].get(family, []):
                        raise
                    continue
            if family in curves.get("family_inputs", {}) and name not in curves["family_inputs"][family]:
                continue
            for target in curves.get("objectives", [None]):
                budget = curves["objective_budgets"][target] // len(families) if search.mode == "balanced" else trials
                studies.append(dict(input=name, family=family, objective=target, trials=budget, folds=folds,
                                    epochs=epochs, historical=features["historical"], future=features["future"],
                                    **sampler_settings))
    missing = set(families) - {item["family"] for item in studies}
    if missing:
        raise ValueError(f"Familias sin inputs compatibles: {sorted(missing)}")
    if search.mode == "per_input":
        return studies

    grouped = {}
    for item in studies:
        grouped.setdefault((item["family"], item["objective"]), []).append(item)
    pooled = []
    for (family, objective), options in grouped.items():
        if len(options) == 1:
            pooled.append(options[0])
            continue
        trials = options[0]["trials"]
        startup = trials // len(options) if search.mode == "balanced" else search.startup_trials_per_input
        if search.mode == "balanced" and trials % len(options):
            raise ValueError(f"{family}/{objective}: incompatible budget for {len(options)} inputs")
        if startup * len(options) > trials:
            raise ValueError(
                f"{family}/{objective}: {len(options)} inputs × {startup} trials iniciales "
                f"requieren al menos {len(options) * startup} trials; configurados: {trials}"
            )
        pooled.append(dict(
            family=family, objective=objective, trials=trials, folds=folds, epochs=epochs,
            inputs=[item["input"] for item in options], startup_trials_per_input=startup,
            **sampler_settings,
            input_features={
                item["input"]: {"historical": item["historical"], "future": item["future"]}
                for item in options
            },
        ))
    return pooled
