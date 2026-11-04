"""optimize: refine a model's intrinsics and poses with bundle adjustment.

Mapping estimates one camera per image (no EXIF), so each camera's focal length and distortion
come from that image's own points only. Images taken by the same device at the same zoom share
intrinsics in reality; estimating them jointly is more precise. ``share_intrinsics`` groups the
images of each sub-model by image size ("size"), or by image size and clustered focal length
("cluster": per-image estimates within ``cluster_tol`` of each other, so zoom levels stay apart),
gives each group one camera initialised at the group's median parameters, optionally upgrades the
camera model (SIMPLE_RADIAL -> OPENCV), and runs bundle adjustment with a robust loss.
Every sub-model is refined; the output is a new model artifact.
"""

from __future__ import annotations

import json
import shutil
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
        "camera_model": {"type": "string", "enum": ["keep", "SIMPLE_RADIAL", "OPENCV"], "default": "OPENCV",
                         "description": "camera model after refinement; OPENCV adds tangential and a second radial term"},
        "share_intrinsics": {"type": "string", "enum": ["none", "size", "cluster"], "default": "cluster",
                             "description": "none: keep one camera per image; size: one camera per image size; "
                                            "cluster: one camera per image size and focal-length cluster (zoom level)"},
        "cluster_tol": {"type": "number", "default": 0.08, "description": "relative focal-length gap that separates clusters"},
        "rounds": {"type": "integer", "default": 2, "description": "bundle adjustment rounds (points filtered between rounds)"},
    },
    "required": ["model_id"],
}


def to_params(cam, target: str) -> tuple[str, list[float]]:
    """Camera parameters of ``cam`` expressed in the target model (fills missing terms with 0)."""
    name = cam.model.name
    p = list(cam.params)
    if name == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        full = dict(fx=f, fy=f, cx=cx, cy=cy, k1=k, k2=0.0, p1=0.0, p2=0.0)
    elif name == "RADIAL":
        f, cx, cy, k1, k2 = p
        full = dict(fx=f, fy=f, cx=cx, cy=cy, k1=k1, k2=k2, p1=0.0, p2=0.0)
    elif name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        full = dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2)
    elif name in ("PINHOLE", "SIMPLE_PINHOLE"):
        if name == "PINHOLE":
            fx, fy, cx, cy = p
        else:
            fx, cx, cy = p
            fy = fx
        full = dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0)
    else:
        return name, p  # unknown: keep as is
    target = name if target == "keep" else target
    if target == "SIMPLE_RADIAL":
        return target, [0.5 * (full["fx"] + full["fy"]), full["cx"], full["cy"], full["k1"]]
    if target == "OPENCV":
        return target, [full["fx"], full["fy"], full["cx"], full["cy"], full["k1"], full["k2"], full["p1"], full["p2"]]
    return name, p


def group_images(rec, mode: str, tol: float) -> list[list[int]]:
    """Groups of image ids that will share one camera."""
    imgs = [im for im in rec.images.values() if im.has_pose]
    if mode == "none":
        return [[im.image_id] for im in imgs]
    by_size: dict[tuple, list] = {}
    for im in imgs:
        cam = rec.cameras[im.camera_id]
        by_size.setdefault((cam.width, cam.height), []).append(im)
    groups = []
    for members in by_size.values():
        if mode == "size":
            groups.append([im.image_id for im in members])
            continue
        # cluster by focal length: sort, split where the relative gap exceeds tol
        members.sort(key=lambda im: rec.cameras[im.camera_id].mean_focal_length())
        cur = [members[0]]
        for prev, im in zip(members, members[1:]):
            f0, f1 = rec.cameras[prev.camera_id].mean_focal_length(), rec.cameras[im.camera_id].mean_focal_length()
            if (f1 - f0) / max(f0, 1e-9) > tol:
                groups.append([m.image_id for m in cur])
                cur = []
            cur.append(im)
        groups.append([m.image_id for m in cur])
    return groups


def reassign_cameras(rec, groups: list[list[int]], camera_model: str):
    """A copy of ``rec`` where each group of images shares one new camera (median parameters,
    target model). Done through COLMAP's text format: writing without rigs/frames files and reading
    back gives every camera a trivial rig, which pycolmap's rig-aware bundle adjuster needs."""
    import tempfile

    import pycolmap

    new_cams = {}  # new camera id -> (model, width, height, params)
    image_cam = {}
    for k, g in enumerate(groups, start=1):
        cams = [rec.cameras[rec.images[i].camera_id] for i in g]
        converted = [to_params(c, camera_model) for c in cams]
        params = np.median(np.array([p for _, p in converted]), axis=0).tolist()
        new_cams[k] = (converted[0][0], cams[0].width, cams[0].height, params)
        for i in g:
            image_cam[i] = k
    d = Path(tempfile.mkdtemp())
    rec.write_text(str(d))
    for f in ("rigs.txt", "frames.txt"):
        (d / f).unlink(missing_ok=True)
    with open(d / "cameras.txt", "w") as f:
        f.write("# Camera list\n")
        for cid, (model, w, h, params) in new_cams.items():
            f.write(f"{cid} {model} {w} {h} " + " ".join(f"{v:.10g}" for v in params) + "\n")
    lines = (d / "images.txt").read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            out.append(line)
            i += 1
            continue
        parts = line.split()
        image_id = int(parts[0])
        if image_id in image_cam:
            parts[8] = str(image_cam[image_id])
            out.append(" ".join(parts))
            out.append(lines[i + 1] if i + 1 < len(lines) else "")
        i += 2
    (d / "images.txt").write_text("\n".join(out) + "\n")
    new = pycolmap.Reconstruction(str(d))
    shutil.rmtree(d)
    return new


def refine(rec, camera_model: str, share: str, tol: float, rounds: int):
    import pycolmap

    groups = group_images(rec, share, tol)
    before = rec.compute_mean_reprojection_error()
    rec = reassign_cameras(rec, groups, camera_model)
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = True
    opts.refine_extra_params = True
    opts.refine_principal_point = False
    opts.print_summary = False
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = 1.0
    opts.ceres.solver_options.max_num_iterations = 100
    for r in range(rounds):
        pycolmap.bundle_adjustment(rec, opts)
        if r + 1 < rounds:
            bad = [pid for pid, p in rec.points3D.items() if p.error > 4.0]
            for pid in bad:
                rec.delete_point3D(pid)
    return rec, {"groups": len(groups), "reproj_before": round(before, 3), "reproj_after": round(rec.compute_mean_reprojection_error(), 3),
                 "group_sizes": sorted((len(g) for g in groups), reverse=True)[:10]}


@register("optimize", "Refine a model with bundle adjustment: optionally share intrinsics among images of the same size "
                      "and focal-length cluster (same device and zoom) and upgrade the camera model. Returns a new model. "
                      "Improves precision when many images come from few cameras.", PARAMS)
def optimize(ws: Workspace, cfg: dict[str, Any], model_id: str, camera_model: str = "OPENCV", share_intrinsics: str = "cluster",
             cluster_tol: float = 0.08, rounds: int = 2) -> ToolResult:
    import pycolmap

    started = time.time()
    params = {"camera_model": camera_model, "share_intrinsics": share_intrinsics, "cluster_tol": float(cluster_tol), "rounds": int(rounds)}
    id = ws.artifact_id("optimize", params, {"model": model_id})
    if (r := cached_result(ws, id)) is not None:
        return r
    if model_id not in ws.registry:
        raise KeyError(f"unknown artifact {model_id!r}: pass the model id returned by reconstruct")
    src = ws.abspath(model_id)
    out = ws.path("model", id)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    details = {}
    for sub in sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit()):
        rec = pycolmap.Reconstruction(str(sub))
        if rec.num_reg_images() >= 3:
            rec, details[sub.name] = refine(rec, camera_model, share_intrinsics, cluster_tol, rounds)
        (out / sub.name).mkdir()
        rec.write(str(out / sub.name))
    summary = {"backend": "bundle_adjustment", "num_images_in_db": ws.get(model_id).summary.get("num_images_in_db"),
               **model_summary(out), "refinement": details}
    (out / "meta.json").write_text(json.dumps({"params": params, "model": model_id, "summary": summary}, indent=1))
    art = Artifact(id, "model", "optimize", params, {"model": model_id, "db": ws.ancestor(model_id, "db").id}, str(out.relative_to(ws.store)), summary)
    return finish(ws, art, started)
