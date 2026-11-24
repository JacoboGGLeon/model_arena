"""Read-only dashboard launcher, with an explicit Jupyter reverse proxy route."""
from dataclasses import dataclass
from pathlib import Path
import html
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid


@dataclass
class MonitorHandle:
    url: str
    pid: int
    log_path: str
    process: object

    def button(self):
        return (f'<a href="{html.escape(self.url, quote=True)}" target="_blank" '
                'style="display:inline-block;padding:14px 24px;background:#07264a;'
                'color:white;border-radius:10px;text-decoration:none;font:600 16px sans-serif">'
                'Abrir Observatory ↗</a>')

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class Observatory:
    def __init__(self, publication, *, proxy_prefix=None):
        self.publication = str(getattr(publication, "path", publication))
        self.proxy_prefix = proxy_prefix

    @classmethod
    def from_config(cls, path, *, publication=None):
        import yaml
        config = yaml.safe_load(Path(path).read_text())
        return cls(publication or config["publication"], proxy_prefix=config.get("proxy_prefix"))

    def launch(self, *, port=None, proxy_prefix=None):
        import importlib.util
        if importlib.util.find_spec("streamlit") is None:
            raise RuntimeError("Install the monitor extra in this interpreter")
        if port is None:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        prefix = proxy_prefix if proxy_prefix is not None else self.proxy_prefix
        if prefix is None:
            service = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
            prefix = service.rstrip("/")+f"/proxy/absolute/{port}"
        prefix = prefix.replace("{port}", str(port)).rstrip("/")
        if not prefix.startswith("/"):
            raise ValueError("proxy_prefix must be an absolute URL path")
        log = Path(tempfile.gettempdir()) / f"model_arena_monitor_{uuid.uuid4().hex}.log"
        program = Path(__file__).with_name("dashboard.py")
        command = [sys.executable, "-m", "streamlit", "run", str(program),
                   "--server.address", "127.0.0.1", "--server.port", str(port),
                   "--server.baseUrlPath", prefix.lstrip("/"), "--server.headless", "true",
                   "--browser.gatherUsageStats", "false", "--", self.publication]
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Monitor stopped. See log: " + str(log))
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}{prefix}/_stcore/health",
                                                timeout=1) as response:
                        if response.status == 200:
                            return MonitorHandle(prefix+"/", process.pid, str(log), process)
                except (OSError, TimeoutError):
                    pass
                time.sleep(.2)
            raise TimeoutError("Monitor health check timed out. See log: " + str(log))
        except BaseException:
            process.terminate()
            process.wait(timeout=5)
            raise

