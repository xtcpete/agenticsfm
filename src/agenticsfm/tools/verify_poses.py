"""verify_poses: check a model's poses against a feedforward reconstruction model (VGGT).

For every registered image (or the ones asked for), VGGT is run on the image and its strongest
neighbours in the model (by verified inliers, within the same sub-model). Its relative rotations
and translation directions are compared with the model's. An image whose model pose disagrees
with VGGT across its neighbours is placed wrongly (a folded doppelganger, a distractor, or a
drifted camera); a disagreement on one pair only is noise. The check is independent of the
feature matches, so it also catches poses that the matches cannot: GLOMAP registering an image
somewhere plausible but wrong, or an incremental mapper growing through a repeated facade.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.backends import run_backend
from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.examine import edge_consistency
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "images": {"type": "array", "items": {"type": "string"},
                   "description": "check only these images (default: every registered image, up to max_images)"},
        "neighbours": {"type": "integer", "default": 5, "description": "neighbours per image in each VGGT group"},
        "max_images": {"type": "integer", "default": 80, "description": "evenly spaced sample when the model has more"},
    },
    "required": ["model_id"],
}

SUSPECT_ROT_DEG = 15.0
MATCH_CONSISTENT = 0.5  # an image whose strong pairs are mostly consistent with its pose is supported by the matches


def match_support(db_path: Path, recs: list) -> dict[str, float]:
    """Per registered image: inlier-weighted fraction of its strong pairs' matches that agree with the model's poses."""
    rows = edge_consistency(db_path, recs, 4.0).get("_rows", [])
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for a, b, n, kind, frac in rows:
        if kind != "same" or n < 50:
            continue
        for x in (a, b):
            num[x] = num.get(x, 0.0) + n * frac
            den[x] = den.get(x, 0.0) + n
    return {x: num[x] / den[x] for x in den}


def rot_angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def vec_angle(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def strongest_neighbours(db_path: Path, where: dict[str, int], k: int) -> dict[str, list[str]]:
    """For each registered image, its k verified pairs with most inliers inside the same sub-model."""
    import pycolmap

    nb: dict[str, list[tuple[int, str]]] = {n: [] for n in where}
    with pycolmap.Database.open(db_path) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            n = int(g.inlier_matches.shape[0])
            if n == 0:
                continue
            i1, i2 = divmod(int(pid), 2147483647)
            a, b = names[i1], names[i2]
            if a in where and b in where and where[a] == where[b]:
                nb[a].append((n, b))
                nb[b].append((n, a))
    return {n: [b for _, b in sorted(v, reverse=True)[:k]] for n, v in nb.items()}


@register("verify_poses", "Check a model's poses with a feedforward reconstruction model (VGGT) on each image and its "
                          "neighbours: an independent test of whether every image is placed correctly, not only "
                          "whether it is registered. Returns per-image disagreement and the suspect images.", PARAMS)
def verify_poses(ws: Workspace, cfg: dict[str, Any], model_id: str, images: list[str] | None = None, neighbours: int = 5,
                 max_images: int = 80) -> ToolResult:
    import pycolmap

    started = time.time()
    params = {"images": sorted(images) if images else None, "neighbours": int(neighbours), "max_images": int(max_images), "v": 4}
    id = ws.artifact_id("verify_poses", params, {"model": model_id})
    if (r := cached_result(ws, id)) is not None:
        return r
    if model_id not in ws.registry:
        raise KeyError(f"unknown artifact {model_id!r}: pass the model id returned by reconstruct")
    model_dir = ws.abspath(model_id)
    db = ws.ancestor(model_id, "db")
    recs = [pycolmap.Reconstruction(str(p)) for p in sorted(model_dir.iterdir()) if p.is_dir() and p.name.isdigit()]
    recs.sort(key=lambda r: -r.num_reg_images())
    where, pose = {}, {}
    for ci, rec in enumerate(recs):  # largest first; an image in several sub-models is checked in the largest
        for im in rec.images.values():
            if im.has_pose and im.name not in where:
                where[im.name] = ci
                pose[im.name] = im.cam_from_world().matrix()  # 3x4
    nbs = strongest_neighbours(ws.abspath(db.id), where, neighbours)
    support = match_support(ws.abspath(db.id), recs)
    targets = [n for n in (images or sorted(where)) if n in where and nbs.get(n)]
    if not images and len(targets) > max_images:
        idx = np.unique(np.linspace(0, len(targets) - 1, max_images).round().astype(int))
        targets = [targets[i] for i in idx]
    unknown = [n for n in (images or []) if n not in where]
    groups = [[n] + nbs[n] for n in targets]
    ff_model = cfg.get("verify", {}).get("model", "vggt")  # "vggt" or "vggt_omega" (configs/default.yaml)
    out = run_backend(cfg, "feedforward", "vggt", {"images_dir": str(ws.images_dir), "groups": groups, "model": ff_model},
                      ws.path("examine", id, "_vggt"), gpu=ws.gpus[0] if ws.gpus else None)

    per_image = []
    for g in out["groups"]:
        names = g["names"]
        E = np.array(g["extrinsics"])  # (n, 3, 4) cam_from_world in VGGT's frame
        anchor = names[0]
        rots, trans = [], []
        for j in range(1, len(names)):
            # relative pose anchor -> j from VGGT and from the model
            R_ff = E[j, :, :3] @ E[0, :, :3].T
            t_ff = E[j, :, 3] - R_ff @ E[0, :, 3]
            Pa, Pj = pose[anchor], pose[names[j]]
            R_m = Pj[:, :3] @ Pa[:, :3].T
            t_m = Pj[:, 3] - R_m @ Pa[:, 3]
            rots.append(rot_angle(R_ff @ R_m.T))
            if np.linalg.norm(t_ff) > 1e-3 and np.linalg.norm(t_m) > 1e-6:
                trans.append(vec_angle(t_ff, t_m))
        per_image.append({"image": anchor, "neighbours": names[1:], "rot_err_deg": [round(r, 1) for r in rots],
                          "median_rot_err_deg": round(float(np.median(rots)), 1) if rots else None,
                          "median_trans_dir_err_deg": round(float(np.median(trans)), 1) if trans else None,
                          "match_support": round(support[anchor], 3) if anchor in support else None})
    med = np.array([p["median_rot_err_deg"] for p in per_image if p["median_rot_err_deg"] is not None])
    disagree = [p for p in per_image if p["median_rot_err_deg"] is not None and p["median_rot_err_deg"] > SUSPECT_ROT_DEG]
    # Misplaced: VGGT disagrees AND the image's own matches disagree with its pose. VGGT alone is wrong on
    # some images (extreme zoom, top-down views), and then the matches still support the pose.
    suspects = sorted([p for p in disagree if p["match_support"] is None or p["match_support"] < MATCH_CONSISTENT],
                      key=lambda p: -p["median_rot_err_deg"])
    vggt_only = sorted([p for p in disagree if p not in suspects], key=lambda p: -p["median_rot_err_deg"])
    # Ranking of likely misplaced images (no hard threshold): mean rank of rotation disagreement and low match
    # support. The translation direction is kept in per_image.json for research only: against GT on ETH3D (2.9k
    # checks) and WRIVA (10k checks) flagging it above 60 deg has precision 0.09 and 0.11 (scripts/eval_verify.py).
    scored = [p for p in per_image if p["median_rot_err_deg"] is not None]
    if scored:
        def ranks(vals):
            order = np.argsort(np.argsort(-np.array(vals, float)))
            return order / max(1, len(vals) - 1)
        r_rot = ranks([p["median_rot_err_deg"] for p in scored])
        r_sup = ranks([-(p["match_support"] if p["match_support"] is not None else 1.0) for p in scored])
        for p, a, c in zip(scored, r_rot, r_sup):
            p["suspicion_rank"] = round(float((a + c) / 2), 3)
    likely = sorted(scored, key=lambda p: p.get("suspicion_rank", 1.0))[:10]
    summary = {
        "checked_images": len(per_image), "registered_images": len(where), "not_in_model": unknown[:20],
        "most_suspicious_images": [{"image": p["image"], "median_rot_err_deg": p["median_rot_err_deg"],
                                    "match_support": p["match_support"]} for p in likely],
        "rotation_disagreement_deg": {"median": float(np.median(med)) if len(med) else None,
                                      "p90": float(np.percentile(med, 90)) if len(med) else None},
        "suspect_images": [{"image": p["image"], "median_rot_err_deg": p["median_rot_err_deg"],
                            "match_support": p["match_support"]} for p in suspects[:20]],
        "num_suspect": len(suspects), "suspect_frac": round(len(suspects) / max(1, len(per_image)), 3),
        "vggt_disagrees_but_matches_support_pose": [{"image": p["image"], "median_rot_err_deg": p["median_rot_err_deg"],
                                                     "match_support": p["match_support"]} for p in vggt_only[:20]],
        "_doc": f"most_suspicious_images: ranked by rotation disagreement with VGGT and low match support (a ranking, "
                f"not a verdict: the top entries are where to look with inspect / analyze, never a list to drop). "
                f"suspect: the model pose disagrees with VGGT by > {SUSPECT_ROT_DEG:.0f} deg (median over neighbours) AND fewer "
                f"than {MATCH_CONSISTENT:.0%} of the image's strong matches agree with its pose: placed wrongly, or a distractor; "
                f"may be dropped with filter_edges(drop_images=...). vggt_disagrees_but_matches_support_pose: VGGT is probably "
                f"wrong on these (extreme zoom, top-down views); keep them.",
    }
    out_dir = ws.path("examine", id)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "per_image.json").write_text(json.dumps(per_image, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    art = Artifact(id, "examine", "verify_poses", params, {"model": model_id}, str(out_dir.relative_to(ws.store)), summary,
                   {k: v for k, v in out["cost"].items() if isinstance(v, (int, float))})
    return finish(ws, art, started)
