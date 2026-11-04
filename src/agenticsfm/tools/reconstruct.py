"""reconstruct: camera poses and points from a database (global, all images)."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string"},
        "backend": {"type": "string", "enum": ["colmap", "glomap", "vggt", "vggt_omega"], "default": "colmap",
                    "description": "colmap: incremental mapping (robust, slower, may split into several models); "
                                   "glomap: global mapping (fast, one model, needs a consistent graph); vggt / vggt_omega: "
                                   "feedforward local reconstruction of the listed `images` (up to ~40, seconds, no matches "
                                   "needed, poses only moderately precise; use to place a group the mapper cannot, then "
                                   "merge_models into the global model through shared images)"},
        "images": {"type": "array", "items": {"type": "string"},
                   "description": "restrict the reconstruction to these images: for colmap / glomap a local model of a partition "
                                  "(see partition_graph), for vggt / vggt_omega the group to reconstruct together (up to ~40); "
                                  "include 3 or more images that the global model already places well, so merge_models can align "
                                  "the result"},
        "min_num_matches": {"type": "integer", "default": 15},
        "seed": {"type": "integer", "default": 0},
    },
    "required": ["db_id"],
}


def model_summary(model_dir: Path) -> dict[str, Any]:
    """Compact statistics of a model directory with sub-models 0, 1, ..."""
    import pycolmap

    comps, registered = [], set()
    for sub in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        rec = pycolmap.Reconstruction(str(sub))
        registered |= {im.name for im in rec.images.values() if im.has_pose}
        comps.append({"id": int(sub.name), "num_images": rec.num_reg_images(), "num_points": rec.num_points3D(),
                      "mean_reproj_error_px": round(rec.compute_mean_reprojection_error(), 3),
                      "mean_track_length": round(rec.compute_mean_track_length(), 2),
                      "mean_obs_per_image": round(rec.compute_mean_observations_per_reg_image(), 1)})
    comps.sort(key=lambda c: -c["num_images"])
    # An image can sit in several COLMAP sub-models; count it once.
    return {"num_models": len(comps), "models": comps[:10], "registered_images": len(registered)}


def run_colmap(db_path: Path, images_dir: Path, out: Path, num_threads: int, min_num_matches: int, seed: int,
               images: list[str] | None = None) -> None:
    import pycolmap

    pycolmap.set_random_seed(seed)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = num_threads
    opts.min_num_matches = min_num_matches
    if images:
        opts.image_names = list(images)
    with pycolmap.ostream():
        pycolmap.incremental_mapping(db_path, images_dir, out, options=opts)


def run_glomap(db_path: Path, images_dir: Path, out: Path, num_threads: int, min_num_matches: int, seed: int,
               images: list[str] | None = None) -> None:
    import pycolmap

    pycolmap.set_random_seed(seed)
    opts = pycolmap.GlobalPipelineOptions()
    opts.num_threads = num_threads
    opts.min_num_matches = min_num_matches
    if images:
        opts.image_names = list(images)
    opts.random_seed = seed
    opts.mapper.global_positioning.use_gpu = False      # pycolmap without CUDA
    opts.mapper.bundle_adjustment.ceres.use_gpu = False
    with pycolmap.ostream():
        pycolmap.global_mapping(db_path, images_dir, out, options=opts)


def write_feedforward_model(ws: Workspace, db_path: Path, group: dict, out: Path) -> None:
    """A COLMAP model (one sub-model) from one feedforward group: PINHOLE cameras at the original
    resolution, poses, and the subsampled depth points, so examine / renders / merge work on it."""
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        sizes = {}
        for im in db.read_all_images():
            cam = db.read_camera(im.camera_id)
            sizes[im.name] = (cam.width, cam.height)
    d = out.parent / (out.name + "_txt_tmp")
    d.mkdir(parents=True, exist_ok=True)
    cams, imgs = [], []
    for i, (name, E, K, tr) in enumerate(zip(group["names"], group["extrinsics"], group["intrinsics"], group["transforms"]), start=1):
        w, h = sizes[name]
        sx, sy, ox, oy = tr
        fx, fy = K[0][0] / sx, K[1][1] / sy
        cx, cy = (K[0][2] - ox) / sx, (K[1][2] - oy) / sy
        cams.append(f"{i} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}")
        R = np.array(E)[:, :3]
        t = np.array(E)[:, 3]
        q = pycolmap.Rotation3d(R).quat  # x, y, z, w
        imgs.append(f"{i} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i} {name}")
        imgs.append("")
    (d / "cameras.txt").write_text("\n".join(cams) + "\n")
    (d / "images.txt").write_text("\n".join(imgs) + "\n")
    pts = group.get("points", [])
    lines = [f"{k + 1} {x:.6f} {y:.6f} {z:.6f} 128 128 128 0.0" for k, (x, y, z) in enumerate(pts)]
    (d / "points3D.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    rec = pycolmap.Reconstruction(str(d))
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))
    shutil.rmtree(d)


@register("reconstruct", "Reconstruct camera poses from a database with COLMAP (incremental) or GLOMAP (global). "
                         "Returns per-model statistics: registered images, points, reprojection error, track length.", PARAMS)
def reconstruct(ws: Workspace, cfg: dict[str, Any], db_id: str, backend: str = "colmap", min_num_matches: int = 15,
                seed: int = 0, num_threads: int = 16, images: list[str] | None = None) -> ToolResult:
    started = time.time()
    if backend in ("vggt", "vggt_omega"):
        from agenticsfm.backends import run_backend

        if not images or len(images) < 2:
            raise ValueError("feedforward reconstruction needs `images` (2 to ~40 names)")
        names = sorted(set(images))
        unknown = [n for n in names if n not in set(ws.image_names())]
        if unknown:
            raise ValueError(f"unknown images: {unknown[:5]}")
        params = {"backend": backend, "images": names}
        id = ws.artifact_id("reconstruct", params, {"db": db_id})
        if (r := cached_result(ws, id)) is not None:
            return r
        out = ws.path("model", id)
        shutil.rmtree(out, ignore_errors=True)
        res = run_backend(cfg, "feedforward", "vggt", {"images_dir": str(ws.images_dir), "groups": [names], "model": backend,
                                                        "points_per_image": 300}, ws.path("model", id, "_work"),
                          gpu=ws.gpus[0] if ws.gpus else None)
        write_feedforward_model(ws, ws.abspath(db_id), res["groups"][0], out / "0")
        summary = {"backend": backend, "num_images_in_db": ws.get(db_id).summary.get("num_images"), **model_summary(out),
                   "_doc": "a feedforward local model: poses in its own frame and scale; no matches were used, so examine's "
                           "pair test still applies; attach it to a global model with merge_models"}
        (out / "meta.json").write_text(json.dumps({"params": params, "db": db_id, "summary": summary}, indent=1))
        art = Artifact(id, "model", "reconstruct", params, {"db": db_id}, str(out.relative_to(ws.store)), summary,
                       {k: v for k, v in res["cost"].items() if isinstance(v, (int, float))})
        return finish(ws, art, started)
    params = {"backend": backend, "min_num_matches": int(min_num_matches), "seed": int(seed)}
    if images:
        params["images"] = sorted(set(images))
    id = ws.artifact_id("reconstruct", params, {"db": db_id})
    with ws.lock(id):
        ws.reload_registry()
        if (r := cached_result(ws, id)) is not None:
            return r
        out = ws.path("model", id)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        runner = {"colmap": run_colmap, "glomap": run_glomap}[backend]
        db_path = ws.abspath(db_id)
        if params.get("images"):
            # Concurrent mappings of several parts must not share one SQLite file: each gets its own copy.
            db_copy = out.with_name(out.name + "_db.db")
            shutil.copyfile(db_path, db_copy)
            db_path = db_copy
        try:
            runner(db_path, ws.images_dir, out, num_threads, min_num_matches, seed, params.get("images"))
        finally:
            if params.get("images"):
                db_path.unlink(missing_ok=True)
        summary = {"backend": backend, "num_images_in_db": ws.get(db_id).summary.get("num_images"), **model_summary(out)}
        (out / "meta.json").write_text(json.dumps({"params": params, "db": db_id, "summary": summary}, indent=1))
        art = Artifact(id, "model", "reconstruct", params, {"db": db_id}, str(out.relative_to(ws.store)), summary)
        return finish(ws, art, started)
