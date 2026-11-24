"""Launch the bundled read-only Streamlit training dashboard through ADA proxy."""

from pathlib import Path
import os
import socket
import subprocess
import sys
import time
import urllib.request
import html

ROOT = Path(__file__).resolve().parent


def launch(project):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    prefix = (
        os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/jupyterlab/default/").rstrip("/")
        + f"/proxy/absolute/{port}"
    )
    logs = Path(project).expanduser().resolve() / "monitor_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"loss_monitor_{port}.log"
    env = dict(os.environ, OBS_LOSS_PROJECT=str(Path(project).resolve()))
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "loss_app.py"),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--server.baseUrlPath",
                prefix.lstrip("/"),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
                "--theme.base",
                "light",
            ],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(log.read_text()[-4000:])
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{prefix}/_stcore/health", timeout=1
            ) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.25)
    else:
        process.terminate()
        raise TimeoutError(log.read_text()[-4000:])
    from IPython.display import display, HTML

    display(
        HTML(
            f'<a href="{html.escape(prefix)}/" target="_blank" rel="noopener" style="display:inline-block;background:#072146;color:white;padding:16px 28px;border-radius:12px;text-decoration:none;font:600 16px Arial;border-left:5px solid #02a5a5">Ver entrenamiento ↗</a>'
        )
    )
    process.health_url = f"http://127.0.0.1:{port}{prefix}/_stcore/health"
    print("PID:", process.pid, "· Log:", log)
    return process
