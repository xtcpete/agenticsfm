"""merge_models: attach model B to model A through their shared images, then re-triangulate and
bundle adjust on A's database.

B is aligned to A with a similarity estimated on the centres of the images both register (at least
3, preferably well spread); B's other images are added to A with the transformed poses and their
own cameras. The merged poses are then refined with COLMAP: triangulate tracks from the database's
matches for all images, bundle adjust (poses, intrinsics, points), and drop observations that do
not fit. The result is one sub-model; examine and verify_poses judge whether the attached images
hold. Typical use: B is a feedforward local reconstruction (reconstruct with backend vggt /
vggt_omega) of a group the mapper could not place, plus a few anchor images from A.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.frame import cameras_of, umeyama
from agenticsfm.tools.reconstruct import model_summary
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "model_a": {"type": "string", "description": "the global model (its database is used for triangulation)"},
        "model_b": {"type": "string", "description": "the model to attach (e.g. a feedforward local model)"},
        "more": {"type": "array", "items": {"type": "string"},
                 "description": "further models to attach after B, one by one (each needs >= 3 images shared with the growing model)"},
        "images": {"type": "array", "items": {"type": "string"},
                   "description": "which of B's images to add (default: all of B's images that A does not register)"},
        "refine": {"type": "boolean", "default": True, "description": "re-triangulate from the database and bundle adjust after merging"},
        "submodel_b": {"type": "integer", "description": "which sub-model of B to attach (default: its largest)"},
    },
    "required": ["model_a", "model_b"],
}


def submodels(ws: Workspace, model_id: str):
    """The model's sub-models in directory order (0, 1, 2 ...)."""
    import pycolmap

    dirs = sorted((p for p in ws.abspath(model_id).iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    return [pycolmap.Reconstruction(str(p)) for p in dirs]


def largest(ws: Workspace, model_id: str, submodel: int | None = None):
    subs = submodels(ws, model_id)
    return subs[submodel] if submodel is not None else max(subs, key=lambda r: r.num_reg_images())


def merged_text_model(rec_a, rec_b, s: float, R: np.ndarray, t: np.ndarray, add: list[str], out: Path, db_path: Path | None = None) -> None:
    """A's model plus B's listed images, transformed into A's frame, written as a text model with
    trivial rigs (through COLMAP's text format: new cameras and images need rigs, which the reader
    creates when the rig files are absent)."""
    import pycolmap
    import tempfile

    d = Path(tempfile.mkdtemp())
    rec_a.write_text(str(d))
    for f in ("rigs.txt", "frames.txt"):
        (d / f).unlink(missing_ok=True)
    # Image and camera ids must be the database's, so that triangulation can add the database's observations.
    db_ids, db_cams = {}, {}
    if db_path is not None:
        with pycolmap.Database.open(db_path) as dbh:
            db_ids = {im.name: (im.image_id, im.camera_id) for im in dbh.read_all_images()}
            db_cams = {c.camera_id: c for c in dbh.read_all_cameras()}
    next_cam = max(rec_a.cameras) + 1
    next_img = max(rec_a.images) + 1
    used_cams = set(rec_a.cameras)
    cam_lines, img_lines = [], []
    b_by_name = {im.name: im for im in rec_b.images.values()}
    for k, name in enumerate(add):
        im = b_by_name[name]
        cam = rec_b.cameras[im.camera_id]
        iid, cid = db_ids.get(name, (next_img + k, next_cam + k))
        if cid in used_cams:  # the database shares this camera with an image A already has: give the new image its own
            cid = next_cam + k
        used_cams.add(cid)
        # Triangulation requires the database's camera model for this camera id: convert B's intrinsics into it
        # (focal and principal point kept, distortion zero; bundle adjustment refines all of them).
        model_name, params = cam.model.name, list(cam.params)
        target = db_cams.get(cid)
        if target is not None and target.model.name != model_name:
            f = cam.mean_focal_length()
            px, py = cam.principal_point_x, cam.principal_point_y
            model_name = target.model.name
            params = {"SIMPLE_RADIAL": [f, px, py, 0.0], "RADIAL": [f, px, py, 0.0, 0.0], "PINHOLE": [f, f, px, py],
                      "SIMPLE_PINHOLE": [f, px, py], "OPENCV": [f, f, px, py, 0.0, 0.0, 0.0, 0.0]}.get(model_name, list(target.params))
        cam_lines.append(f"{cid} {model_name} {cam.width} {cam.height} " + " ".join(f"{v:.9g}" for v in params))
        # cam_from_world in A's frame: x_a = s R x_b + t  =>  cam_from_world_a = cam_from_world_b o (x_b = R^T (x_a - t) / s)
        Rb = im.cam_from_world().rotation.matrix()
        tb = np.asarray(im.cam_from_world().translation)
        Ra = Rb @ R.T
        ta = s * tb - Ra @ t
        q = pycolmap.Rotation3d(Ra).quat
        img_lines.append(f"{iid} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {ta[0]:.9f} {ta[1]:.9f} {ta[2]:.9f} {cid} {name}")
        img_lines.append("")
    with open(d / "cameras.txt", "a") as f:
        f.write("\n".join(cam_lines) + "\n")
    with open(d / "images.txt", "a") as f:
        f.write("\n".join(img_lines) + "\n")
    rec = pycolmap.Reconstruction(str(d))
    shutil.rmtree(d)
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))


def refine_with_database(ws: Workspace, db_path: Path, model_dir: Path, out: Path) -> dict:
    """COLMAP: point triangulation from the database's matches for the model's poses, then bundle adjustment."""
    import pycolmap

    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = 16
    out.mkdir(parents=True, exist_ok=True)
    with pycolmap.ostream():
        rec = pycolmap.triangulate_points(pycolmap.Reconstruction(str(model_dir)), db_path, ws.images_dir, out, options=opts)
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_focal_length = True
    ba.refine_extra_params = True
    ba.print_summary = False
    ba.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    for _ in range(2):
        pycolmap.bundle_adjustment(rec, ba)
        bad = [pid for pid, p in rec.points3D.items() if p.error > 4.0]
        for pid in bad:
            rec.delete_point3D(pid)
    rec.write(str(out))
    return {"num_points": rec.num_points3D(), "reproj": round(rec.compute_mean_reprojection_error(), 3), "registered": rec.num_reg_images()}


def attach(ws: Workspace, rec_a, rec_b, images: list[str] | None, db_path: Path | None, out: Path) -> dict:
    """B's images added to A (text model at ``out``); returns the alignment details."""
    ca, cb = cameras_of(rec_a), cameras_of(rec_b)
    common = sorted(set(ca) & set(cb))
    if len(common) < 3:
        raise ValueError(f"only {len(common)} shared registered images; need >= 3 to align (add anchor images)")
    s, R, t = umeyama(np.array([cb[n][0] for n in common]), np.array([ca[n][0] for n in common]))
    resid = np.array([np.linalg.norm(s * R @ cb[n][0] + t - ca[n][0]) for n in common])
    span = np.linalg.norm(np.ptp(np.array([c for c, _, _ in ca.values()]), axis=0))
    add = sorted(n for n in (images or cb) if n in cb and n not in ca)
    merged_text_model(rec_a, rec_b, s, R, t, add, out, db_path)
    return {"shared_images": len(common), "alignment_residual_frac_of_span": round(float(resid.mean() / max(span, 1e-9)), 4),
            "scale_b_to_a": round(float(s), 4), "added_images": add if len(add) <= 30 else add[:30] + [f"... {len(add)} in all"]}


@register("merge_models", "Attach model B to model A through the images both register (similarity alignment on >= 3 shared "
                          "images), then re-triangulate and bundle adjust on A's database. Returns one merged model; check "
                          "it with examine and verify_poses.", PARAMS)
def merge_models(ws: Workspace, cfg: dict[str, Any], model_a: str, model_b: str, images: list[str] | None = None,
                 refine: bool = True, more: list[str] | None = None, submodel_b: int | None = None) -> ToolResult:
    import pycolmap

    started = time.time()
    more = list(more or [])
    params = {"images": sorted(images) if images else None, "refine": bool(refine), "more": more,
              **({"submodel_b": int(submodel_b)} if submodel_b is not None else {})}
    id = ws.artifact_id("merge_models", params, {"model_a": model_a, "model_b": model_b})
    if (r := cached_result(ws, id)) is not None:
        return r
    out = ws.path("model", id)
    shutil.rmtree(out, ignore_errors=True)
    db = ws.ancestor(model_a, "db")
    db_path = ws.abspath(db.id) if db else None
    rec = largest(ws, model_a)
    steps = []
    for k, mb in enumerate([model_b] + more):
        tmp = out / f"merged{k}"
        steps.append({"model": mb, **attach(ws, rec, largest(ws, mb, submodel_b if k == 0 else None), images if k == 0 else None, db_path, tmp)})
        rec = pycolmap.Reconstruction(str(tmp))
    details = {"steps": steps, **steps[0]} if len(steps) == 1 else {"steps": steps}
    last = out / f"merged{len(steps) - 1}"
    if refine and db is not None:
        details["refine"] = refine_with_database(ws, db_path, last, out / "0")
    else:
        last.rename(out / "0")
    for k in range(len(steps)):
        shutil.rmtree(out / f"merged{k}", ignore_errors=True)
    summary = {"backend": "merge", "num_images_in_db": ws.get(model_a).summary.get("num_images_in_db"), **model_summary(out), "merge": details,
               "_doc": "alignment_residual_frac_of_span above ~0.05 means the shared images disagree between the two models; "
                       "added images keep B's relative poses until the refinement moves them"}
    (out / "meta.json").write_text(json.dumps({"params": params, "model_a": model_a, "model_b": model_b, "summary": summary}, indent=1))
    inputs = {"model_a": model_a, "model_b": model_b, **({"db": db.id} if db else {})}
    art = Artifact(id, "model", "merge_models", params, inputs, str(out.relative_to(ws.store)), summary)
    return finish(ws, art, started)
