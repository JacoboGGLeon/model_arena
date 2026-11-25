"""Isolated compatibility worker. No network publication except in publish stage."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from .._version import __version__


def execute(project, stage):
    project = Path(project).resolve()
    runtime = project / "runtime"
    training = runtime / "entrenamiento"
    sys.path[:0] = [
        str(training),
        str(training / "visor"),
        str(runtime),
        *[str(p) for p in (training / "libraries").glob("*/src")],
    ]
    from model_arena.loss_store import write_parquet
    import polars as pl
    import torch
    import arena_estudio as worker
    from entrenar_curvas import read, write, spec_for
    from observatorio_cuentas.calendar import read_calendar
    from model_arena.winner_selection import assignments
    from model_arena.storage import write_json

    config_path = project / "training.json"
    cfg = read(config_path)
    device = json.loads((project / "project.json").read_text())["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    run = project / "run"
    run.mkdir(exist_ok=True)
    if stage == "prepare":
        worker.execute(run, config_path, "ingesta", device)
    elif stage == "tune":
        worker.run_study(run, config_path, device, stage="tune")
    else:
        spec = spec_for(cfg)
        axis, seen, atypical, inventory = worker.population(run, cfg, spec)
        root = run / "curvas"
        calendar = read_calendar(training / "data/calendar_authoritative.csv")
        dates = axis + calendar.after(axis[-1], spec.horizon)
        candidates = [SimpleNamespace(**item) for item in read(root / "candidatos.json")]
        catalog = {item.id: item for item in candidates}
        if stage == "select":
            chosen = assignments(
                pl.read_parquet(root / "scores.parquet"), inventory, candidates, ["F1", "F2", "F3"]
            )
            write_parquet(chosen, root / "winners_FUTURO.parquet")
        elif stage in ("fit", "predict", "postprocess"):
            chosen = pl.read_parquet(root / "winners_FUTURO.parquet")

            def infer(fold, end):
                pieces = []
                for candidate_id in sorted(set(chosen["candidate"])):
                    candidate = catalog[candidate_id]
                    checkpoint = root / "modelos" / f"{candidate_id}_{fold}.pt"
                    if not checkpoint.exists():
                        raise ValueError(f"Missing fitted checkpoint {checkpoint}; inference never trains")
                    model = worker.fit_saved(
                        root,
                        seen,
                        dates,
                        end,
                        spec,
                        candidate,
                        fold,
                        torch.device(device),
                        calendar,
                        cfg["curvas"],
                    )
                    ids = set(chosen.filter(pl.col("candidate") == candidate_id)["account_id"])
                    group = [series for series in [*seen, *atypical] if series.id in ids]
                    frame = worker.emit(
                        model, group, dates, end, spec, candidate, fold, torch.device(device), calendar
                    )
                    pieces.append(
                        frame.with_columns(
                            pl.col("method").alias("candidate_model"),
                            pl.when(pl.col("method") == "Persistencia")
                            .then(pl.lit("Persistencia"))
                            .otherwise(pl.lit("Winner"))
                            .alias("method"),
                        )
                    )
                output = root / "predicciones" / f"Winner_{fold}.parquet"
                write_parquet(pl.concat(pieces), output)
                write_parquet(chosen, root / f"winners_{fold}.parquet")
                return dict(
                    modelo="Winner",
                    fold=fold,
                    origen=dates[end - 1],
                    desde=dates[end],
                    hasta=dates[end + spec.horizon - 1],
                    predicciones=str(output.relative_to(root)),
                )

            if stage == "fit":
                for key in sorted(set(chosen["candidate"])):
                    worker.fit_saved(
                        root,
                        seen,
                        dates,
                        len(axis),
                        spec,
                        catalog[key],
                        "FUTURO",
                        torch.device(device),
                        calendar,
                        cfg["curvas"],
                    )
            elif stage == "predict":
                infer("FUTURO", len(axis))
            else:
                records = [infer(f"F{i + 1}", len(axis) - (3 - i) * spec.horizon) for i in range(3)]
                future = infer("FUTURO", len(axis))
                history = worker.combine_history(root)
                write(
                    root / "resultados_x0.json",
                    dict(
                        version="0.11.0",
                        producer=f"model-arena-sdk {__version__}",
                        corrida=run.name,
                        horizonte=spec.horizon,
                        ajustes=records,
                        futuro=future,
                        muestra=dict(entrenadas=len(seen), atipicas_zero_shot=len(atypical)),
                        config=cfg,
                        training_history=str(history.relative_to(root)),
                        protocol="F1-F3 selection evidence; fixed winner configuration per entity; no independent test",
                    ),
                )
                from observatorio_athena.common import Config
                from observatorio_athena.prepare import prepare

                out = project / "release"
                if (out / "publication.json").exists():
                    print("Existing prepared release retained; publication is explicit", flush=True)
                else:
                    prepare(
                        run,
                        out,
                        Config.read(project / "publication.json"),
                        delivery_dirs=[root / "modelos", project / "calendar"],
                        training_path=history,
                    )
        elif stage == "publish":
            from observatorio_athena.aws import Publisher
            from observatorio_athena.common import Config

            Publisher(Config.read(project / "publication.json")).publish(project / "release")
        else:
            raise ValueError("Unknown MAC stage: " + stage)
    write_json(project / "status.json", {"status": stage.upper() + "_COMPLETE"})


if __name__ == "__main__":
    execute(sys.argv[1], sys.argv[2])
