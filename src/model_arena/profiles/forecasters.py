"""Model capabilities and extension points for the MAC runtime."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Protocol

from ..models.complete import FAMILIES, capabilities, suggest_parameters


class ForecasterDefinition(Protocol):
    encoder_search: bool

    def supports(self, historical: int, future: int) -> None: ...
    def validate_study(self, study, curves, spec) -> None: ...
    def suggest(self, trial, config, search, *, window): ...
    def network(self, spec, candidate, dropout, *, targets, past, future, legacy_encoder): ...
    def fit(self, **context): ...


@dataclass(frozen=True)
class TorchDefinition:
    name: str
    encoder_search: bool = False

    def supports(self, historical, future):
        if not self.encoder_search:
            capabilities(self.name, historical, future)

    def validate_study(self, study, curves, spec):
        if not self.encoder_search and spec.residuals:
            raise ValueError("Residual profiles require an encoder–decoder")
        if self.encoder_search:
            quota = study.get("startup_trials_per_input", study["trials"])
            if quota < len(curves.get("encoder_kinds", ["tft_compact"])):
                raise ValueError("Each input needs enough trials to cover all encoders")
        if spec.residuals and curves.get("interval", {}).get("method") != "mad":
            raise ValueError("Residual profiles require a point forecast")

    def suggest(self, trial, config, search, *, window):
        if self.encoder_search:
            params = dict(
                encoder_kind=trial.suggest_categorical("encoder_kind", config["encoder_kinds"])
                if "encoder_kinds" in config else "tft_compact",
                encoder_design="single" if config.get("target") else trial.suggest_categorical("encoder", config["encoders"]),
                depth=trial.suggest_categorical("encoder_depth", config["depths"]),
                decoder_depth=trial.suggest_categorical("decoder_depth", config["decoder_depths"]),
                width=trial.suggest_categorical("width", config["widths"]),
                activation=trial.suggest_categorical("activation", config["activations"]),
                lr=trial.suggest_float("lr", *config["lr_range"], log=True),
                weight_decay=trial.suggest_float("weight_decay", *config["wd_range"], log=True),
            )
            residual = (config.get("run_spec", {}).get("residuals") or {}).get(config.get("target"))
            if residual:
                params["residual"] = {"baseline": residual["baseline"], "base_weight": residual["base_weight"]}
                if residual["baseline"] == "learned":
                    params["residual"]["width"] = trial.suggest_categorical("residual_width", residual["widths"])
            return params
        params = suggest_parameters(
            self.name, trial, search, mac_point=config.get("interval", {}).get("method") == "mad",
            moving_avg_max=min(20, window),
        )
        return dict(family_params=params, encoder_kind=self.name, encoder_design="independent",
                    lr=params["lr"], weight_decay=params["weight_decay"])

    def network(self, spec, candidate, dropout, *, targets, past, future, legacy_encoder):
        from ..mad import point_forecast

        if self.encoder_search:
            from ..models.tft import Network, TFTParameters

            params = TFTParameters(
                encoder_kind=getattr(candidate, "encoder_kind", "tft_compact"),
                encoder_design="independent" if candidate.encoder_design == "independent" else "shared",
                depth=candidate.depth, decoder_depth=getattr(candidate, "decoder_depth", 1),
                width=candidate.width, activation=getattr(candidate, "activation", "ELU"),
                dropout=dropout,
            )
            return Network(targets, spec.horizon, past, future, params, window=spec.window,
                           tft_encoder=legacy_encoder, point=point_forecast(candidate),
                           residual=getattr(candidate, "residual", None),
                           linear_history=getattr(candidate, "transform", None) == "linear_history")
        from ..models.complete import CompleteNetwork, CompleteParameters

        return CompleteNetwork(
            family=self.name, targets=targets, window=spec.window, h=spec.horizon,
            historical=past, future=future, parameters=CompleteParameters.model_validate(candidate.family_params),
            point=point_forecast(candidate),
        )

    def fit(self, *, fit_neural, **context):
        return fit_neural(**context)


@lru_cache(maxsize=1)
def model_definitions() -> dict[str, ForecasterDefinition]:
    from .tree import LightGBMDefinition

    definitions = {name: TorchDefinition(name) for name in FAMILIES}
    definitions["encoder_decoder"] = TorchDefinition("encoder_decoder", encoder_search=True)
    definitions["lightgbm"] = LightGBMDefinition()
    for entry in entry_points(group="model_arena.mac_models"):
        if entry.name in definitions:
            raise ValueError(f"Duplicate MAC model definition: {entry.name}")
        definitions[entry.name] = entry.load()()
    return definitions
