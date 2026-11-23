"""optimize_gpu: `optimize` with the ray-anchored GPU bundle adjustment (third_party/rayba) instead of COLMAP's.

Same interface as `optimize` (a model in, a new model out). Per sub-model: the problem is built here from the model
and its match database (rayba.build: the model's tracks plus the verified matches that fit), solved by the `rayba`
worker on a GPU (rayba.solver.solve: hybrid observations, OPENCV with a principal point per intrinsics group, groups
merged when their focal lengths agree; Cauchy 1 match px, then 0.25 in the last round; dense or
block-sparse reduced camera system by size), and written back (rayba.colmap.write_back): refined poses, one OPENCV
camera per intrinsics group and image size, 3D points moved to their solved positions, points with a mean reprojection
error above 4 px removed. Offline comparison with
COLMAP on the p22 grown models: docs/RAYBA.md.
"""
from __future__ import annotations

import json
import shutil
import time
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.reconstruct import model_summary
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "principal_point": {"type": "boolean", "default": True, "description": "refine one principal point per intrinsics group"},
        "iters": {"type": "integer", "default": 20, "description": "Levenberg-Marquardt iterations per round"},
    },
    "required": ["model_id"],
}


@register("optimize_gpu", "Refine a model with the GPU bundle adjustment: the model's tracks plus the verified matches "
                          "that fit the refined geometry, one OPENCV camera with a principal point per image-size and "
                          "focal cluster. Faster than optimize and more accurate on the dev benchmarks. Returns a new model.",
          PARAMS)
def optimize_gpu(ws: Workspace, cfg: dict[str, Any], model_id: str, principal_point: bool = True, iters: int = 20) -> ToolResult:
    import pycolmap

    from rayba.build import build
    from rayba.client import save_problem
    from rayba.colmap import write_back

    from agenticsfm.solver.ffclient import run_rayba

    started = time.time()
    params = {"principal_point": bool(principal_point), "iters": int(iters), "v": 3}   # v3: one camera per group and image size
    id = ws.artifact_id("optimize_gpu", params, {"model": model_id})
    if (r := cached_result(ws, id)) is not None:
        return r
    if model_id not in ws.registry:
        raise KeyError(f"unknown artifact {model_id!r}: pass the model id returned by reconstruct")
    src = ws.abspath(model_id)
    db = pycolmap.Database.open(str(ws.abspath(ws.ancestor(model_id, "db").id)))
    out = ws.path("model", id)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    work = out / "_work"
    gpu = ws.gpus[0] if ws.gpus else None
    details = {}
    for sub in sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit()):
        rec = pycolmap.Reconstruction(str(sub))
        if rec.num_reg_images() >= 3:
            z = build(rec, db)
            prob = work / f"{sub.name}.problem.npz"
            work.mkdir(parents=True, exist_ok=True)
            save_problem(z, prob)
            res = run_rayba(cfg, {"problem_path": str(prob), "output_path": str(work / f"{sub.name}.out.npz"),
                                  "config": {"model": "opencv_pp" if principal_point else "opencv", "iters": int(iters)}},
                            work / sub.name, gpu)   # the persistent worker (the growth loop's, when it ran in this process)
            sol = dict(np.load(work / f"{sub.name}.out.npz"))
            before = rec.compute_mean_reprojection_error()
            rec = write_back(rec, z, sol)
            details[sub.name] = {"build_s": round(z["t_build"], 1), "solve_s": round(res["wall"], 1), "backend_s": round(res["cost"]["wall"], 1),
                                 "observations": res["observations"], "merged_groups": res["merged"], "stats": z["stats"],
                                 "reproj_before": round(before, 3), "reproj_after": round(rec.compute_mean_reprojection_error(), 3)}
            prob.unlink(missing_ok=True)
        (out / sub.name).mkdir()
        rec.write(str(out / sub.name))
    summary = {"backend": "rayba", "num_images_in_db": ws.get(model_id).summary.get("num_images_in_db"), **model_summary(out),
               "refinement": details}
    (out / "meta.json").write_text(json.dumps({"params": params, "model": model_id, "summary": summary}, indent=1, default=str))
    art = Artifact(id, "model", "optimize_gpu", params, {"model": model_id, "db": ws.ancestor(model_id, "db").id},
                   str(out.relative_to(ws.store)), summary)
    return finish(ws, art, started)
