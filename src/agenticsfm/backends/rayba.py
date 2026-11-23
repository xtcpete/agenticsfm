"""Backend: ray-anchored GPU bundle adjustment (solver/rayba.py) of a problem written by solver/rayba_build.py.

Spec: {problem_path (.npz), output_path (.npz), config: {...overrides of rayba.DEFAULTS}, result_path}
Result: {wall, iterations, observations, merged, device}

One-shot:   <env python> -m agenticsfm.backends.rayba <spec.json>
Persistent: <env python> -m agenticsfm.backends.rayba serve <repos.json>   (solver/ffclient.py's protocol: one JSON
            spec per line on stdin, "OK <result_path>" or "ERR <message>" per line on stdout). The growth loop bundle
            adjusts dozens of times per scene; the worker pays the torch import and CUDA start-up once.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np

from agenticsfm.backends import read_spec, write_result


def serve_one(spec: dict, dev: str) -> None:
    from agenticsfm.solver.rayba import solve

    z = dict(np.load(spec["problem_path"]))
    out = solve(z, dev=dev, **spec.get("config", {}))
    np.savez(spec["output_path"], **{k: v for k, v in out.items() if isinstance(v, np.ndarray)}, model=out["model"])
    write_result(spec, {k: out[k] for k in ("wall", "iterations", "observations", "merged")} | {"device": dev})


def serve() -> None:
    import os

    proto = os.fdopen(os.dup(1), "w", buffering=1)      # the protocol gets its own descriptor; prints go to the log
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    t0 = time.time()
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.zeros(1, device=dev)
    proto.write(f"READY {time.time() - t0:.1f}\n")
    proto.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            spec = json.loads(line)
            serve_one(spec, dev)
            proto.write(f"OK {spec['result_path']}\n")
        except Exception as e:  # report and keep serving
            proto.write("ERR " + json.dumps(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}") + "\n")
        finally:
            if dev == "cuda":
                torch.cuda.empty_cache()         # the GPU is shared with the feedforward worker
        proto.flush()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve()
        return
    spec = read_spec()
    import torch

    serve_one(spec, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
