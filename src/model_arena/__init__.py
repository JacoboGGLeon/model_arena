"""Model Arena. Training and UI dependencies are loaded only when requested."""
from .contracts import ArenaSpec, ModelSpec, PanelSchema, TemporalCV
from .arena import ModelArena

__version__ = "0.1.0rc1"
__all__ = ["ModelArena", "ArenaSpec", "ModelSpec", "PanelSchema", "TemporalCV"]

