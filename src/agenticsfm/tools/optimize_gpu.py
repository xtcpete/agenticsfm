"""optimize_gpu: `optimize` with the ray-anchored GPU bundle adjustment (solver/rayba.py) instead of COLMAP's.

Same interface as `optimize` (a model in, a new model out). Per sub-model: the problem is built here from the model
and its match database (solver/rayba_build.py: the model's tracks plus the verified matches that fit), solved by the
`rayba` backend on a GPU (rayba.solve: hybrid observations, OPENCV with a principal point per intrinsics group, groups
merged when their focal lengths agree; Cauchy 1 match px, then 0.25 in the last round; dense or
block-sparse reduced camera system by size), and written back: refined poses, one OPENCV camera per intrinsics group, 3D
points moved to their solved positions, points with a mean reprojection error above 4 px removed. Offline comparison with
COLMAP on the p22 grown models: docs/RAYBA.md.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
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


def write_back(rec, z, out):
    """A copy of rec with the solved poses, OPENCV cameras (through COLMAP's text format, as optimize.reassign_cameras)
    and the solved 3D points; points with mean reprojection error > 4 px removed. One camera per intrinsics group and
    image size: a merged group can hold images whose sizes differ by a few pixels (undistorted crops), which share the
    focal length and distortion but keep their own centre (the solver's per-image principal point plus the group's
    offset)."""
    import pycolmap

    model = str(out["model"])
    pp = model.endswith("_pp")
    I = out["intr"]
    ids = [int(i) for i in z["image_ids"]]
    group = out["group"]
    cams, key_of = {}, {}
    for n, iid in enumerate(ids):
        g = int(group[n])
        cam = rec.cameras[rec.images[iid].camera_id]
        key = (g, int(cam.width), int(cam.height), round(float(z["c"][n][0]), 6), round(float(z["c"][n][1]), 6))
        key_of[iid] = key
        if key in cams:
            continue
        cx, cy = float(z["c"][n][0]), float(z["c"][n][1])
        if pp:
            cx += float(I[g, -2]); cy += float(I[g, -1])
        if model.startswith("opencv"):
            fx, fy, k1, k2, p1, p2 = np.exp(I[g, 0]), np.exp(I[g, 1]), I[g, 2], I[g, 3], I[g, 4], I[g, 5]
        else:
            fx = fy = np.exp(I[g, 0]); k1, k2, p1, p2 = I[g, 1], 0.0, 0.0, 0.0
        cams[key] = (cam.width, cam.height, [fx, fy, cx, cy, k1, k2, p1, p2])
    for n, iid in enumerate(ids):
        im = rec.images[iid]
        rec.frame(im.frame_id).rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(out["R"][n]), out["t"][n])
    for k, pid in enumerate(z["pt_ids"]):
        if np.isfinite(out["pt_xyz"][k]).all():
            rec.points3D[int(pid)].xyz = out["pt_xyz"][k]
    d = Path(tempfile.mkdtemp())
    rec.write_text(str(d))
    for f in ("rigs.txt", "frames.txt"):
        (d / f).unlink(missing_ok=True)
    gid = {key: k + 1 for k, key in enumerate(sorted(cams))}
    with open(d / "cameras.txt", "w") as f:
        for key, (w, h, p) in cams.items():
            f.write(f"{gid[key]} OPENCV {w} {h} " + " ".join(f"{float(v):.12g}" for v in p) + "\n")
    image_cam = {iid: gid[key_of[iid]] for iid in ids}
    lines = (d / "images.txt").read_text().splitlines()
    outl, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            outl.append(line); i += 1; continue
        parts = line.split(); iid = int(parts[0])
        if iid in image_cam:
            parts[8] = str(image_cam[iid]); outl.append(" ".join(parts)); outl.append(lines[i + 1] if i + 1 < len(lines) else "")
        i += 2
    (d / "images.txt").write_text("\n".join(outl) + "\n")
    new = pycolmap.Reconstruction(str(d))
    shutil.rmtree(d)
    new.update_point_3d_errors()
    for pid in [pid for pid, p3 in new.points3D.items() if p3.error > 4.0]:   # as optimize between its rounds
        new.delete_point3D(pid)
    return new


@register("optimize_gpu", "Refine a model with the GPU bundle adjustment: the model's tracks plus the verified matches "
                          "that fit the refined geometry, one OPENCV camera with a principal point per image-size and "
                          "focal cluster. Faster than optimize and more accurate on the dev benchmarks. Returns a new model.",
          PARAMS)
def optimize_gpu(ws: Workspace, cfg: dict[str, Any], model_id: str, principal_point: bool = True, iters: int = 20) -> ToolResult:
    import pycolmap

    from agenticsfm.solver.ffclient import run_rayba
    from agenticsfm.solver.rayba_build import build

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
            np.savez(prob, **{k: v for k, v in z.items() if isinstance(v, np.ndarray)})
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
