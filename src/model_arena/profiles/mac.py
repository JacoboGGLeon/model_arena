"""Migration adapter preserving the MAC engines and Athena alert dashboard."""
from importlib.resources import files
from pathlib import Path
import shutil
import subprocess
import sys

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
    def create(cls, path, *, training_config, publication_config, calendar_path, device="cuda"):
        path = Path(path).expanduser().resolve()
        if path.exists():
            raise FileExistsError("Choose a new project directory or reopen it with MacProject(path)")
        if device not in ("cpu", "cuda"):
            raise ValueError("Select cpu or cuda explicitly")
        training = read_json(training_config)
        if training.get("requested_accounts") != 0 or training.get("aws", {}).get("account_limit") != 0:
            raise ValueError("The profile requires the complete eligible universe, account limits = 0")
        training["curvas"]["final_epochs"] = 75
        training["curvas"]["loss_monitor_accounts"] = None
        path.mkdir(parents=True)
        source = Path(str(files("model_arena") / "_mac_runtime"))
        shutil.copytree(source, path / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        calendar = path / "runtime/entrenamiento/data/calendar_authoritative.csv"
        calendar.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(calendar_path, calendar)
        write_json(path / "training.json", training)
        shutil.copy2(publication_config, path / "publication.json")
        settings = dict(device=device, training_sha=digest(path / "training.json"),
                        publication_sha=digest(path / "publication.json"), calendar_sha=digest(calendar),
                        code_sha=digest(path / "runtime/BUILD_MONITOR.json"))
        write_json(path / "project.json", settings)
        return cls(path)

    def verify(self):
        paths = {"training_sha": self.path / "training.json", "publication_sha": self.path / "publication.json",
                 "calendar_sha": self.path / "runtime/entrenamiento/data/calendar_authoritative.csv",
                 "code_sha": self.path / "runtime/BUILD_MONITOR.json"}
        if any(digest(path) != self.settings[key] for key, path in paths.items()):
            raise ValueError("Project identity changed; create a new project")
        root = self.path / "runtime"
        for name, expected in read_json(root / "BUILD_MONITOR.json")["files"].items():
            if digest(root / name) != expected:
                raise ValueError("Profile runtime changed: " + name)

    def _stage(self, stage):
        self.verify()
        subprocess.run([sys.executable, "-u", "-m", "model_arena.profiles.mac_worker",
                        str(self.path), stage], check=True)
        return self

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

    def publish(self):
        self._stage("publish")
        return read_json(self.path / "release/published.json")

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
        cfg = read_json(self.path / "publication.json")
        if publication_id is None and (self.path / "release/published.json").exists():
            publication_id = read_json(self.path / "release/published.json")["publication_id"]
        if publication_id:
            cfg["initial_publication_id"] = publication_id
        write_json(self.path / "monitor.json", cfg)
        # Existing dashboard and launcher are preserved; interpreter is explicit.
        import importlib.util
        runtime = self.path / "runtime"
        spec = importlib.util.spec_from_file_location("_mac_profile_monitor", runtime / "abrir_monitor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.launch(config=self.path / "monitor.json", python=sys.executable)


def open_monitor(publication_config, *, publication_id=None, workspace=".observatory"):
    """Open the Athena monitor without a local training run, calendar or GPU."""
    import importlib.util
    workspace = Path(workspace).expanduser().resolve()
    source = Path(str(files("model_arena") / "_mac_runtime"))
    root = workspace / ("runtime_" + digest(source / "BUILD_MONITOR.json")[:12])
    if not root.exists():
        shutil.copytree(source, root,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = read_json(publication_config)
    if publication_id:
        config["initial_publication_id"] = publication_id
    write_json(workspace / "monitor.json", config)
    spec = importlib.util.spec_from_file_location("_standalone_mac_monitor", root / "abrir_monitor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.launch(config=workspace / "monitor.json", python=sys.executable)

