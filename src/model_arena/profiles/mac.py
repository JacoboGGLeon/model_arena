"""Migration adapter preserving the MAC engines and Athena alert dashboard."""

from importlib.resources import files
from pathlib import Path
import shutil
import subprocess
import sys
import fcntl
import os
from contextlib import contextmanager
from ..disk import check_space

from ..storage import write_json, digest, read_json


class MacProject:
    """Execute the established MAC profile through explicit lifecycle stages.

    Configurations and the authoritative calendar are deployment inputs, never
    public package data. This compatibility profile is separate from the generic
    ModelArena core so migration does not silently change accounting rules.
    """

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.settings = read_json(self.path / "project.json")

    @classmethod
    def create(
        cls, path, *, training_config, publication_config, calendar_path=None, calendar=None, device="cuda"
    ):
        path = Path(path).expanduser().resolve()
        if path.exists():
            raise FileExistsError("Choose a new project directory or reopen it with MacProject(path)")
        if device not in ("cpu", "cuda"):
            raise ValueError("Select cpu or cuda explicitly")
        training = read_json(training_config)
        if training.get("requested_accounts") != 0 or training.get("aws", {}).get("account_limit") != 0:
            raise ValueError("The profile requires the complete eligible universe, account limits = 0")
        objectives = training['curvas'].get('objectives', ['saldo','variacion'])
        if objectives != ['saldo','variacion']:
            raise ValueError('Declare objectives=[saldo, variacion] for independent MAC studies')
        training['curvas']['objectives'] = objectives
        training['curvas'].pop('encoders', None)
        training['curvas']['selection_scope'] = 'entity_target'
        training["curvas"].setdefault("final_epochs", 75)
        training["curvas"].setdefault("loss_monitor_accounts", None)
        from ..contracts import PruningSpec

        PruningSpec.model_validate(training["curvas"].get("pruner") or {})
        families=training['curvas'].get('families',{})
        if ('families' in training['curvas'] and not families) or set(families)-{'encoder_decoder','tide','nhits','nlinear','dlinear','nbeats'} or any(not isinstance(n,int) or n<1 for n in families.values()):
            raise ValueError('Invalid family names or per-family trial budgets')
        kinds = training["curvas"].get("encoder_kinds", ["tft_compact"])
        from ..contracts import SearchSpec
        SearchSpec(encoder_kind=kinds)
        if len(set(kinds)) != len(kinds):
            raise ValueError("Declare distinct encoders")
        if "encoder_decoder" in families or "families" not in training["curvas"]:
            encoder_trials = families["encoder_decoder"] if families else training["curvas"]["trials"]
            if encoder_trials < len(kinds):
                raise ValueError("Encoder–Decoder family budget must cover all declared encoders")
        from ..calendar import prepare_snapshot

        snapshot = prepare_snapshot(
            calendar=calendar, calendar_path=calendar_path, n_features=training["model"].get("calendar_features")
        )
        from ..input_spec import InputSearchSpec, resolve_inputs, study_plan
        from ..mac_policy import MacPolicy

        curves = training["curvas"]
        training["model"]["calendar_features"] = snapshot.provenance["feature_count"]
        curves["input_specs"] = resolve_inputs(curves["inputs"], curves.get("input_specs"), snapshot.provenance["features"])
        curves["calendar_sha256"] = snapshot.provenance["calendar_sha256"]
        if type(curves["final_epochs"]) is not int or curves["final_epochs"] < 1:
            raise ValueError("final_epochs debe ser un entero positivo")
        curves["input_search"] = InputSearchSpec.model_validate(curves.get("input_search", {})).model_dump()
        curves["policies"] = MacPolicy.model_validate(curves.get("policies", {})).model_dump(mode="json")
        study_plan(curves, training["model"]["epochs"])
        check_space(path, stage="crear proyecto")
        path.mkdir(parents=True)
        source = Path(str(files("model_arena") / "_mac_runtime"))
        shutil.copytree(source, path / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        calendar = path / "runtime/entrenamiento/data/calendar_authoritative.csv"
        calendar.parent.mkdir(parents=True, exist_ok=True)
        calendar.write_bytes(snapshot.csv_bytes)
        calendar_manifest = snapshot.save(path)
        write_json(path / "training.json", training)
        shutil.copy2(publication_config, path / "publication.json")
        settings = dict(
            device=device,
            training_sha=digest(path / "training.json"),
            publication_sha=digest(path / "publication.json"),
            calendar_sha=digest(calendar),
            calendar_manifest_sha=digest(calendar_manifest),
            code_sha=digest(path / "runtime/BUILD_MONITOR.json"),
        )
        write_json(path / "project.json", settings)
        write_json(path / "operations.json", {"min_free_gib": 2})
        return cls(path)

    def verify(self):
        paths = {
            "training_sha": self.path / "training.json",
            "publication_sha": self.path / "publication.json",
            "calendar_sha": self.path / "runtime/entrenamiento/data/calendar_authoritative.csv",
            "code_sha": self.path / "runtime/BUILD_MONITOR.json",
        }
        if any(digest(path) != self.settings[key] for key, path in paths.items()):
            raise ValueError("Project identity changed; create a new project")
        if "calendar_manifest_sha" in self.settings:
            from ..calendar import verify_snapshot

            verify_snapshot(self.path, self.settings["calendar_manifest_sha"])
        root = self.path / "runtime"
        for name, expected in read_json(root / "BUILD_MONITOR.json")["files"].items():
            if digest(root / name) != expected:
                raise ValueError("Profile runtime changed: " + name)

    @contextmanager
    def exclusive(self):
        with (self.path / ".stage.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Este proyecto ya tiene una etapa activa") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _stage(self, stage):
        with self.exclusive():
            self.verify()
            check_space(self.path, stage=stage)
            # The child must use this bundle even if an older SDK is installed.
            env = dict(os.environ)
            source = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join([source, env.get("PYTHONPATH", "")])
            subprocess.run(
                [sys.executable, "-u", "-m", "model_arena.profiles.mac_worker", str(self.path), stage],
                check=True,
                env=env,
            )
        return self

    def plan(self):
        """Describe the frozen study budget before training."""
        from ..input_spec import InputSearchSpec, study_plan

        training = read_json(self.path / "training.json")
        studies = study_plan(training["curvas"], training["model"]["epochs"])
        return dict(
            studies=studies,
            input_search=InputSearchSpec.model_validate(training["curvas"].get("input_search", {})).model_dump(),
            candidate_configurations=sum(item["trials"] for item in studies),
            hpo_fits=sum(item["trials"] * item["folds"] for item in studies),
            final_epochs=training["curvas"].get("final_epochs", 75),
        )

    def prepare(self):
        return self._stage("prepare")

    def tune(self):
        return self._stage("tune")

    def select(self):
        return self._stage("select")

    def fit(self):
        return self._stage("fit")

    def predict(self):
        self._stage("predict")
        return self.path / "run/curvas/predicciones/Winner_FUTURO.parquet"

    def postprocess(self):
        self._stage("postprocess")
        return self.path / "release"

    def export_layout(self, destination, *, engine, fold):
        """Export a prepared Engine 2 or 3 segment as CSV or typed Parquet."""
        from .._mac_runtime.observatorio_athena.layout_export import export_prepared_layout, layout_schema

        layout_schema(engine, fold)
        if Path(destination).suffix.lower() not in (".csv", ".parquet"):
            raise ValueError("El destino debe terminar en .csv o .parquet")
        release = self.path / "release"
        if not (release / "publication.json").is_file():
            self.postprocess()
        return export_prepared_layout(release, destination, engine=engine, fold=fold)

    def publish(self):
        self._stage("publish")
        return read_json(self.path / "release/published.json")

    def calendar_provenance(self):
        """Read verified frozen calendar provenance; never regenerate it."""
        self.verify()
        path = self.path / "calendar/provenance.json"
        return (
            read_json(path)
            if path.exists()
            else {
                "generator": "legacy_snapshot",
                "calendar_sha256": self.settings["calendar_sha"],
                "source_metadata": None,
            }
        )

    def status(self):
        path = self.path / "status.json"
        return read_json(path) if path.exists() else {"status": "CREATED"}

    def resume(self):
        """Reuse completed verified artifacts; interrupted fits restart from scratch."""
        for stage in ["prepare", "tune", "select", "fit", "predict", "postprocess"]:
            self._stage(stage)
        return self

    def observe(self, *, publication_id=None):
        self.verify()
        if publication_id is None and (self.path / "release/published.json").exists():
            publication_id = read_json(self.path / "release/published.json")["publication_id"]
        return open_monitor(
            self.path / "publication.json",
            publication_id=publication_id,
            workspace=self.path / ".observatory",
        )


def open_monitor(publication_config, *, publication_id=None, workspace=".observatory"):
    """Open the Athena monitor without a local training run, calendar or GPU."""
    import importlib.util

    workspace = Path(workspace).expanduser().resolve()
    source = Path(str(files("model_arena") / "_mac_runtime"))
    root = workspace / ("runtime_" + digest(source / "BUILD_MONITOR.json")[:12])
    if not root.exists():
        shutil.copytree(source, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = read_json(publication_config)
    if publication_id:
        config["initial_publication_id"] = publication_id
    write_json(workspace / "monitor.json", config)
    spec = importlib.util.spec_from_file_location("_standalone_mac_monitor", root / "abrir_monitor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.launch(config=workspace / "monitor.json", python=sys.executable)
