"""PyTorch forecasting primitives."""

from .calendar import CalendarTable
from .contracts import Candidate, ExperimentSpec, Series
from .engine import Experiment, ExperimentResult, select_device

__version__ = "0.11.0"
__all__ = [
    "CalendarTable",
    "Candidate",
    "Experiment",
    "ExperimentResult",
    "ExperimentSpec",
    "Series",
    "select_device",
]

from .api import fit, predict, tune
__all__ += ["fit","predict","tune"]
