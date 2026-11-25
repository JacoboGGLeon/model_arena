"""Typed policies resolved when a MAC project is created."""

from typing import Literal

from pydantic import model_validator

from .contracts import Contract


class MacPolicy(Contract):
    eligibility: Literal["complete_finite"] = "complete_finite"
    segmenter: Literal["saldo_entropy", "none"] = "saldo_entropy"
    transform: Literal["family_default", "linear_history"] = "family_default"
    selection_metric: Literal["pinball", "scaled_mae"] = "pinball"
    fallback: list[Literal["currency_budget", "budget", "global"]] = [
        "currency_budget", "budget", "global"
    ]

    @model_validator(mode="after")
    def validate_fallback(self):
        if not self.fallback or self.fallback[-1] != "global" or len(set(self.fallback)) != len(self.fallback):
            raise ValueError("El respaldo debe ser una secuencia única que termine en global")
        return self
