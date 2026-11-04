"""match: local features + matches for the graph's pairs, verified into a COLMAP database.

Three artifacts: ``features`` (per detector), ``matches`` (per features x graph x matcher) and
``db`` (per matches x camera setup): the database holds cameras, images, keypoints, matches and
two-view geometries, and is what reconstruct / score_pairs / filter_edges consume.
"""

from __future__ import annotations

import json
import shutil
import time
from typing import Any

import numpy as np

from agenticsfm.backends import run_backend
from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.graph import read_pairs
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "graph_id": {"type": "string"},
        "detector": {"type": "string", "enum": ["aliked", "superpoint"], "default": "aliked",
                     "description": "keypoint detector; use the one the benchmark prescribes (the harness tells you) and keep it fixed"},
        "matcher": {"type": "string", "enum": ["flexglue"], "default": "flexglue",
                    "description": "flexglue: sparse matcher with a DINOv3 backbone, also gives a covisibility score per "
                                   "pair (the matcher is fixed; the reconstruction quality comes from how its pairs are "
                                   "examined, filtered and mapped)"},
        "num_keypoints": {"type": "integer", "default": 4096},
        "resize": {"type": "integer", "default": 1024, "description": "long side for detection"},
        "camera_mode": {"type": "string", "enum": ["auto", "single", "per_folder", "per_image"], "default": "auto",
                        "description": "how images share intrinsics; auto groups by EXIF and size"},
        "camera_model": {"type": "string", "default": "SIMPLE_RADIAL"},
    },
    "required": ["graph_id"],
}

CAMERA_MODES = {"auto": "AUTO", "single": "SINGLE", "per_folder": "PER_FOLDER", "per_image": "PER_IMAGE"}


def db_summary(db_path) -> dict[str, Any]:
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        ids, geoms = db.read_two_view_geometries()
        inliers = np.array([g.inlier_matches.shape[0] for g in geoms])
        num_images, num_cams = db.num_images(), db.num_cameras()
        cams = {c.camera_id: c for c in db.read_all_cameras()}
        models = sorted({cams[c].model.name for c in cams})
    verified = inliers[inliers > 0]
    return {"num_images": num_images, "num_cameras": num_cams, "camera_models": models,
            "num_verified_pairs": int(len(verified)),
            "inliers_per_pair": {"median": float(np.median(verified)) if len(verified) else 0.0,
                                 "p10": float(np.percentile(verified, 10)) if len(verified) else 0.0,
                                 "p90": float(np.percentile(verified, 90)) if len(verified) else 0.0}}


def build_database(ws: Workspace, db_path, pairs_path, features_path, matches_path, camera_mode: str, camera_model: str) -> None:
    import pycolmap
    from hloc.reconstruction import create_empty_db, get_image_ids, import_images
    from hloc.triangulation import estimation_and_geometric_verification, import_features, import_matches

    names = sorted({n for line in open(pairs_path) for n in line.split()})
    tmp = db_path.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    create_empty_db(tmp)
    import_images(ws.images_dir, tmp, getattr(pycolmap.CameraMode, CAMERA_MODES[camera_mode]), image_list=names,
                  options={"camera_model": camera_model})
    image_ids = get_image_ids(tmp)
    with pycolmap.Database.open(tmp) as db:
        import_features(image_ids, db, features_path)
        import_matches(image_ids, db, pairs_path, matches_path)
    estimation_and_geometric_verification(tmp, pairs_path)
    for suffix in ("-wal", "-shm", "-journal"):
        tmp.with_name(tmp.name + suffix).unlink(missing_ok=True)
    tmp.replace(db_path)


def run_match_sharded(ws: Workspace, cfg: dict[str, Any], pairs: list, feat_params: dict, match_params: dict,
                      features_path, matches_path, match_id: str) -> dict[str, Any]:
    """Features once (first shard, so the file exists), then the pairs split over the workspace GPUs in
    parallel backends; the per-shard matches files are concatenated into one."""
    import h5py
    from concurrent.futures import ThreadPoolExecutor

    gpus = ws.gpus or [None]
    work = ws.path("matches", match_id, "_work")
    base = {"images_dir": str(ws.images_dir), **feat_params, **match_params, "features_path": str(features_path), "batch_size": 8}
    n_shards = min(len(gpus), max(1, len(pairs) // 2000))
    if n_shards == 1 or not features_path.is_file():
        # Detection has to happen once before shards can run: do the first shard (or everything) alone.
        first = pairs if n_shards == 1 else pairs[: len(pairs) // n_shards]
        out = run_backend(cfg, "match", "pointglue", {**base, "pairs": [list(p) for p in first],
                                                      "matches_path": str(matches_path if n_shards == 1 else work / "shard0.h5")},
                          work, gpu=gpus[0])
        if n_shards == 1:
            return out
        results = [out]
        rest = pairs[len(pairs) // n_shards:]
    else:
        results, rest = [], pairs
    chunks = [rest[i::n_shards - (1 if results else 0)] for i in range(n_shards - (1 if results else 0))]
    chunks = [c for c in chunks if c]

    def run(k_and_chunk):
        k, chunk = k_and_chunk
        shard = work / f"shard{len(results) + k}.h5"
        return run_backend(cfg, "match", "pointglue", {**base, "pairs": [list(p) for p in chunk], "matches_path": str(shard)},
                           work / f"shard{len(results) + k}", gpu=gpus[(len(results) + k) % len(gpus)])

    with ThreadPoolExecutor(len(chunks)) as pool:
        results += list(pool.map(run, enumerate(chunks)))
    # concatenate shards
    tmp = matches_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as dst:
        for k in range(len(results)):
            with h5py.File(work / f"shard{k}.h5", "r") as src:
                for a in src:
                    for b in src[a]:
                        src.copy(src[a][b], dst.require_group(a), name=b)
    tmp.replace(matches_path)
    merged = {"num_pairs": sum(r["num_pairs"] for r in results), "num_images": results[0]["num_images"],
              "num_matches": {k: v for r in results for k, v in r["num_matches"].items()},
              "cost": {"detect_seconds": results[0]["cost"].get("detect_seconds", 0.0),
                       "match_seconds": max(r["cost"].get("match_seconds", 0.0) for r in results),
                       "gpu_seconds": sum(r["cost"].get("gpu_seconds", 0.0) for r in results), "shards": len(results)}}
    if any("covisibility" in r for r in results):
        merged["covisibility"] = {k: v for r in results for k, v in r.get("covisibility", {}).items()}
    return merged


@register("match", "Detect local features and match the graph's pairs, then geometrically verify them into a "
                   "COLMAP database. Returns the database id and match statistics.", PARAMS)
def match(ws: Workspace, cfg: dict[str, Any], graph_id: str, detector: str = "aliked", matcher: str = "flexglue",
          num_keypoints: int = 4096, resize: int = 1024, camera_mode: str = "auto",
          camera_model: str = "SIMPLE_RADIAL", matcher_params: dict | None = None) -> ToolResult:
    started = time.time()
    matcher_params = dict(matcher_params or {})
    feat_params = {"detector": detector, "num_keypoints": int(num_keypoints), "resize": int(resize)}
    feat_id = ws.artifact_id("features", feat_params, {})
    match_params = {"matcher": matcher, "matcher_params": matcher_params}
    match_id = ws.artifact_id("matches", match_params, {"features": feat_id, "graph": graph_id})
    db_params = {"camera_mode": camera_mode, "camera_model": camera_model}
    db_id = ws.artifact_id("db", db_params, {"matches": match_id})
    with ws.lock(db_id):
        ws.reload_registry()
        if (r := cached_result(ws, db_id)) is not None:
            return r
        return _match(ws, cfg, started, graph_id, detector, matcher, feat_params, feat_id, match_params, match_id, db_params, db_id,
                      camera_mode, camera_model)


def _match(ws, cfg, started, graph_id, detector, matcher, feat_params, feat_id, match_params, match_id, db_params, db_id,
           camera_mode, camera_model) -> ToolResult:
    features_path = ws.path("features", feat_id, ".h5")
    matches_path = ws.path("matches", match_id, ".h5")
    pairs = read_pairs(ws, graph_id)
    cost: dict[str, float] = {}
    if not ws.has(match_id):
        out = run_match_sharded(ws, cfg, pairs, feat_params, match_params, features_path, matches_path, match_id)
        cost = {k: v for k, v in out["cost"].items() if isinstance(v, (int, float))}
        if not ws.has(feat_id):
            ws.register(Artifact(feat_id, "features", "features", feat_params, {}, str(features_path.relative_to(ws.store)),
                                 {"num_images": out["num_images"]}, {"wall": out["cost"].get("detect_seconds", 0.0)}))
        n = np.array(list(out["num_matches"].values()))
        summary = {"num_pairs": out["num_pairs"], "matches_per_pair": {"median": float(np.median(n)), "p10": float(np.percentile(n, 10)),
                                                                      "p90": float(np.percentile(n, 90))}}
        if "covisibility" in out:
            c = np.array(list(out["covisibility"].values()))
            summary["covisibility"] = {"median": float(np.median(c)), "frac_below_0.5": float((c < 0.5).mean())}
        ws.register(Artifact(match_id, "matches", "match", match_params, {"features": feat_id, "graph": graph_id},
                             str(matches_path.relative_to(ws.store)), summary, cost))
    pairs_path = ws.path("matches", match_id, "_pairs.txt")
    shutil.copyfile(ws.abspath(graph_id), pairs_path)
    db_path = ws.path("db", db_id, ".db")
    t = time.time()
    build_database(ws, db_path, pairs_path, features_path, matches_path, camera_mode, camera_model)
    summary = {**db_summary(db_path), "detector": detector, "matcher": matcher,
               "matches": ws.get(match_id).summary, "features_id": feat_id, "matches_id": match_id}
    ws.path("db", db_id, ".json").write_text(json.dumps(summary, indent=1))
    art = Artifact(db_id, "db", "match", {**feat_params, **match_params, **db_params}, {"matches": match_id, "graph": graph_id},
                   str(db_path.relative_to(ws.store)), summary, {**cost, "verify_seconds": time.time() - t})
    return finish(ws, art, started)
