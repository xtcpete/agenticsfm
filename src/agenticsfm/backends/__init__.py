"""Backends: small CLIs run under the interpreter of their own conda env.

    <env python> -m agenticsfm.backends.<name> spec.json

The spec is a JSON dict; the backend writes a JSON result at ``spec["result_path"]`` and its
outputs where the spec says. ``run_backend`` launches one, pinned to a GPU, with this package on
PYTHONPATH, and returns the result dict. Heavy model weights are loaded per call for now; a
persistent worker per env is a later optimisation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agenticsfm.config import REPO_ROOT, load_config

SRC_DIR = REPO_ROOT / "src"


class BackendError(RuntimeError):
    pass


def env_python(cfg: dict[str, Any], env: str) -> str:
    try:
        return cfg["envs"][env]
    except KeyError:
        raise BackendError(f"no interpreter configured for env {env!r} (configs: envs)") from None


def run_backend(cfg: dict[str, Any], name: str, env: str, spec: dict[str, Any], workdir: Path,
                gpu: int | None = None, timeout: float | None = None) -> dict[str, Any]:
    """Run ``agenticsfm.backends.<name>`` in ``env`` on ``gpu``; returns its result dict."""
    workdir.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    spec.setdefault("result_path", str(workdir / f"{name}.result.json"))
    spec.setdefault("repos", cfg.get("repos", {}))
    spec_path = workdir / f"{name}.spec.json"
    spec_path.write_text(json.dumps(spec, indent=1, default=str))
    result_path = Path(spec["result_path"])
    result_path.unlink(missing_ok=True)
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = str(SRC_DIR) + (":" + env_vars["PYTHONPATH"] if env_vars.get("PYTHONPATH") else "")
    env_vars["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env_vars.setdefault("OMP_NUM_THREADS", "8")
    log_path = workdir / f"{name}.log"
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([env_python(cfg, env), "-m", f"agenticsfm.backends.{name}", str(spec_path)],
                              stdout=log, stderr=subprocess.STDOUT, env=env_vars, cwd=str(REPO_ROOT), timeout=timeout)
    wall = time.time() - start
    if proc.returncode != 0 or not result_path.is_file():
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-30:])
        raise BackendError(f"backend {name} ({env}, gpu {gpu}) failed with code {proc.returncode}; log {log_path}:\n{tail}")
    result = json.loads(result_path.read_text())
    result.setdefault("cost", {})
    result["cost"]["wall"] = wall
    if gpu is not None:
        result["cost"]["gpu_seconds"] = result["cost"].get("gpu_seconds", wall)
    return result


def read_spec(argv: list[str] | None = None) -> dict[str, Any]:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m agenticsfm.backends.<name> spec.json")
    return json.loads(Path(argv[0]).read_text())


def write_result(spec: dict[str, Any], result: dict[str, Any]) -> None:
    Path(spec["result_path"]).write_text(json.dumps(result, indent=1, default=float))


def add_repo_paths(spec: dict[str, Any], *names: str) -> None:
    for name in names:
        root = spec["repos"][name]
        if root not in sys.path:
            sys.path.insert(0, root)


__all__ = ["BackendError", "add_repo_paths", "load_config", "read_spec", "run_backend", "write_result"]
