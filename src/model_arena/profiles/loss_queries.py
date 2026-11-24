"""Read-only loss queries for the local training dashboard."""

from pathlib import Path
import polars as pl
from model_arena.loss_store import detail_files


def curves(project, *, inp="X0", folds=None, scope="Total", value=None, sampled=False):
    root = Path(project) / "run/curvas"

    def accepted(stem):
        return (inp == "Todos" or stem.startswith(inp + "_")) and (
            not folds or any(stem.endswith("_" + f) for f in folds)
        )

    if scope == "Total":
        paths = [p for p in (root / "loss").glob("*.parquet") if accepted(p.stem)]
        if not paths:
            return pl.DataFrame()
        lazy = pl.scan_parquet(paths)
        blocks = []
        for target in ["saldo", "variacion"]:
            for split in ["train", "validation"]:
                column = (
                    ("train_loss_" + target if split == "train" else "validation_loss")
                    if sampled
                    else "fixed_" + split + "_" + target
                )
                if sampled and split == "validation":
                    continue
                blocks.append(
                    lazy.select(
                        "candidate",
                        "fold",
                        "epoch",
                        pl.lit(target).alias("target"),
                        pl.lit(split).alias("split"),
                        pl.col(column).alias("loss"),
                    )
                )
        return (
            pl.concat(blocks)
            .filter(pl.col("loss").is_not_null())
            .sort("candidate", "fold", "target", "split", "epoch")
            .collect(engine="streaming")
        )
    paths = [
        p
        for p in detail_files(root / "loss_entities")
        if p.suffix == ".parquet"
        and accepted(p.parent.name if p.parent != root / "loss_entities" else p.stem)
    ]
    if not paths:
        return pl.DataFrame()
    lazy = pl.scan_parquet(paths)
    if scope == "Budget":
        lazy = lazy.filter(pl.col("budget") == int(value))
    elif scope == "Cuenta":
        lazy = lazy.filter(pl.col("account_id") == str(value))
    else:
        raise ValueError("Scope desconocido")
    keys = ["candidate", "fold", "epoch", "target", "split"]
    return (
        lazy.group_by(keys)
        .agg(
            ((pl.col("loss") * pl.col("n_observations")).sum() / pl.col("n_observations").sum()).alias("loss")
        )
        .sort(keys)
        .collect(engine="streaming")
    )


def figure(frame, highlight=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    folds = [f for f in ["F1", "F2", "F3", "FUTURO"] if f in set(frame["fold"])]
    targets = ["saldo", "variacion"]
    fig = make_subplots(
        rows=2,
        cols=len(folds),
        subplot_titles=[t.capitalize() + " · " + f for t in targets for f in folds],
        vertical_spacing=0.18,
        horizontal_spacing=0.06,
    )
    legends = set()
    for row, target in enumerate(targets, 1):
        for col, fold in enumerate(folds, 1):
            block = frame.filter((pl.col("target") == target) & (pl.col("fold") == fold))
            for (candidate, split), part in block.group_by("candidate", "split", maintain_order=True):
                part = part.sort("epoch")
                active = highlight == candidate
                color = {"train": "#168b91", "validation": "#c64358"}[split]
                fig.add_trace(
                    go.Scatter(
                        x=part["epoch"].to_list(),
                        y=part["loss"].to_list(),
                        name={"train": "Entrenamiento", "validation": "Validación"}[split],
                        legendgroup=split,
                        showlegend=split not in legends,
                        line=dict(
                            color=color,
                            width=3 if active else 1.2,
                            dash="solid" if split == "train" else "dash",
                        ),
                        opacity=1 if active or frame["candidate"].n_unique() == 1 else 0.28,
                        text=[candidate] * part.height,
                        hovertemplate="%{text}<br>Época %{x}<br>Loss %{y:.8g}<extra>" + split + "</extra>",
                    ),
                    row=row,
                    col=col,
                )
                legends.add(split)
            fig.update_xaxes(title_text="Época" if row == 2 else None, row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="Pinball normalizado", row=row, col=col)
    fig.update_layout(
        height=690,
        template="plotly_white",
        margin=dict(t=60, l=50, r=20, b=50),
        legend=dict(orientation="h", y=1.12),
        hovermode="closest",
    )
    return fig
