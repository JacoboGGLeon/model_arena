"""Resolve named input configurations against the frozen calendar."""

import re

from pydantic import model_validator

from .contracts import Contract


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
    from .models.complete import FAMILIES, capabilities

    definitions = curves.get("input_specs")
    families = curves.get("families", {"encoder_decoder": curves["trials"]})
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
            if family in FAMILIES:
                try:
                    capabilities(family, len(features["historical"]), len(features["future"]))
                except ValueError:
                    if "family_inputs" in curves and name in curves["family_inputs"].get(family, []):
                        raise
                    continue
            if family in curves.get("family_inputs", {}) and name not in curves["family_inputs"][family]:
                continue
            for target in curves.get("objectives", [None]):
                studies.append(dict(input=name, family=family, objective=target, trials=trials, folds=folds,
                                    epochs=epochs, historical=features["historical"], future=features["future"]))
    missing = set(families) - {item["family"] for item in studies}
    if missing:
        raise ValueError(f"Familias sin inputs compatibles: {sorted(missing)}")
    return studies
