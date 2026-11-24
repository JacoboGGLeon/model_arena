"""Model Arena. Training and UI dependencies are loaded only when requested."""
from .contracts import ArenaSpec, ModelSpec, PanelSchema, TemporalCV
from .arena import ModelArena

from ._version import __version__ as __version__
__all__ = ["ModelArena", "ArenaSpec", "ModelSpec", "PanelSchema", "TemporalCV"]

