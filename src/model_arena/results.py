"""Read-only queries. They never import a training framework."""
from pathlib import Path
import pandas as pd
from .storage import read_table, read_json, verify_manifest


class Results:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def open(cls, publication):
        path = Path(getattr(publication, "path", publication))
        if (path / "latest.json").is_file():
            path = path / read_json(path / "latest.json")["publication_id"]
        verify_manifest(path)
        return cls(path)

    def table(self, name, *, limit=None, offset=0, **filters):
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid table name")
        frame = read_table(self.path / (name + ".json"))
        if frame.empty and (self.path / name).is_dir():
            parts = [read_table(p) for p in sorted((self.path / name).glob("*.json"))]
            frame = pd.concat(parts, ignore_index=True) if parts else frame
        for column, selected in filters.items():
            if column not in frame:
                if frame.empty:
                    return frame
                raise ValueError("Unknown result column: " + column)
            values = selected if isinstance(selected, (list, tuple, set)) else [selected]
            frame = frame[frame[column].isin(values)]
        return frame.iloc[offset:None if limit is None else offset+limit].copy()

    def learning_curves(self, *, scope="entity", **filters):
        frame = self.table("learning_curves", **filters)
        if frame.empty:
            return frame
        if scope == "entity":
            return frame[frame["scope"] == "entity"].copy()
        columns = ["candidate", "input", "fold", "epoch", "target", "split", "metric", "scale"]
        if scope == "total":
            entity = frame[frame.scope == "entity"].copy()
        else:
            inventory = self.table("inventory")
            if scope not in inventory:
                raise ValueError("Unknown aggregation scope")
            entity = frame[frame.scope == "entity"].merge(inventory, on="entity", validate="m:1")
            columns.append(scope)
        if entity.empty:
            return entity
        entity["weighted_sum"] = entity.value*entity.n_observations
        grouped = entity.groupby(columns, dropna=False).agg(
            weighted_sum=("weighted_sum", "sum"), n_observations=("n_observations", "sum"),
            n_entities=("entity", "nunique")).reset_index()
        grouped["value"] = grouped.weighted_sum / grouped.n_observations
        grouped["scope"] = scope
        return grouped.drop(columns="weighted_sum")

