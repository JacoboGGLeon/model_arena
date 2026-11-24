"""Local disk reserve for sequential ADA experiments; no remote side effects."""

from pathlib import Path
import json
import math
import shutil


class DiskSpaceError(OSError):
    """Stop before consuming the reserve used by Optuna and small state files."""


def check_space(path, *, extra_bytes=0, stage="write"):
    path = Path(path).resolve()
    anchor = path
    while not anchor.exists():
        anchor = anchor.parent
    policy = next(
        (p / "operations.json" for p in [path, *path.parents] if (p / "operations.json").is_file()), None
    )
    reserve = float(json.loads(policy.read_text()).get("min_free_gib", 2)) if policy else 2.0
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("min_free_gib must be finite and nonnegative")
    free = shutil.disk_usage(anchor).free
    needed = int(reserve * 2**30) + max(0, int(extra_bytes))
    if free < needed:
        raise DiskSpaceError(
            f"Espacio insuficiente antes de {stage}: {free / 2**30:.2f} GiB libres; "
            f"se requieren {needed / 2**30:.2f} GiB incluyendo reserva. "
            "Libera espacio o amplía el volumen y reanuda el mismo proyecto."
        )
    return {"free_gib": round(free / 2**30, 3), "reserve_gib": reserve, "checked_path": str(anchor)}
