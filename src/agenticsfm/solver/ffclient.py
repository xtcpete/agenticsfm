"""Client for the persistent feedforward worker (backends/ff_worker.py): one worker per (env, model, GPU)
per process, started on first use and kept for the life of the process."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agenticsfm.backends import SRC_DIR, BackendError, env_python
from agenticsfm.config import REPO_ROOT

_WORKERS: dict[tuple, "Worker"] = {}
_LOCK = threading.Lock()


class Worker:
    def __init__(self, cfg: dict[str, Any], env: str, model: str, gpu: int | None, workdir: Path):
        workdir.mkdir(parents=True, exist_ok=True)
        repos = workdir / "repos.json"
        repos.write_text(json.dumps(cfg.get("repos", {})))
        env_vars = dict(os.environ)
        env_vars["PYTHONPATH"] = str(SRC_DIR) + (":" + env_vars["PYTHONPATH"] if env_vars.get("PYTHONPATH") else "")
        env_vars["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        env_vars.setdefault("OMP_NUM_THREADS", "8")
        self.log = open(workdir / f"ff_worker_{model}_{gpu}.log", "a")
        self.proc = subprocess.Popen([env_python(cfg, env), "-m", "agenticsfm.backends.ff_worker", model, str(repos)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                     env=env_vars, cwd=str(REPO_ROOT))
        self.lock = threading.Lock()
        line = self.proc.stdout.readline()
        if not line.startswith("READY"):
            raise BackendError(f"feedforward worker did not start: {line!r}; log {self.log.name}")
        self.load_seconds = float(line.split()[1])

    def request(self, spec: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            Path(spec["result_path"]).unlink(missing_ok=True)
            t0 = time.time()
            self.proc.stdin.write(json.dumps(spec, default=str) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise BackendError(f"feedforward worker died; log {self.log.name}")
            if line.startswith("ERR"):
                raise BackendError(json.loads(line[4:]))
            res = json.loads(Path(line.split(" ", 1)[1].strip()).read_text())
            res.setdefault("cost", {})["wall"] = time.time() - t0
            return res

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def run_feedforward(cfg: dict[str, Any], env: str, model: str, spec: dict[str, Any], workdir: Path, gpu: int | None) -> dict[str, Any]:
    """Same contract as run_backend(cfg, "feedforward", env, spec, workdir, gpu), through a persistent worker."""
    key = (env, model, gpu)
    with _LOCK:
        w = _WORKERS.get(key)
        if w is None or w.proc.poll() is not None:
            w = _WORKERS[key] = Worker(cfg, env, model, gpu, REPO_ROOT / "outputs" / "workers")
    workdir.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    spec.setdefault("result_path", str(workdir / "feedforward.result.json"))
    return w.request(spec)


@atexit.register
def _shutdown():
    for w in _WORKERS.values():
        w.close()
