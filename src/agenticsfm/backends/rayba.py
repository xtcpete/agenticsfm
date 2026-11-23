"""Backend: ray-anchored GPU bundle adjustment (solver/rayba.py) of a problem written by solver/rayba_build.py.

Spec: {problem_path (.npz), output_path (.npz), config: {...overrides of rayba.DEFAULTS}, result_path}
Result: {wall, iterations, observations, merged}
"""
from __future__ import annotations


import numpy as np

from agenticsfm.backends import read_spec, write_result


def main() -> None:
    spec = read_spec()
    import torch

    from agenticsfm.solver.rayba import solve

    z = dict(np.load(spec["problem_path"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = solve(z, dev=dev, **spec.get("config", {}))
    np.savez(spec["output_path"], **{k: v for k, v in out.items() if isinstance(v, np.ndarray)}, model=out["model"])
    write_result(spec, {k: out[k] for k in ("wall", "iterations", "observations", "merged")} | {"device": dev})


if __name__ == "__main__":
    main()
