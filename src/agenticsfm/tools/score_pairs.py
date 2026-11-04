"""score_pairs: one score per verified pair of a database (higher = more likely the same surface)."""

from __future__ import annotations

import json
import time
from typing import Any

import h5py
import numpy as np

from agenticsfm.backends import run_backend
from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string"},
        "method": {"type": "string", "enum": ["xdg", "flexglue_covis", "cycle_consistency"],
                   "description": "xdg: learned doppelganger classifier on the two images (needs the GPU); "
                                  "flexglue_covis: FlexGlue's own covisibility score, free if the db was matched with flexglue; "
                                  "cycle_consistency: share of the pair's triangles (with common neighbours) whose relative "
                                  "rotations close; a wrong edge breaks its triangles; model-free, CPU, minutes"},
    },
    "required": ["db_id", "method"],
}


def verified_pairs(db_path) -> list[tuple[str, str]]:
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        ids, geoms = db.read_two_view_geometries()
        pairs = []
        for pid, g in zip(ids, geoms):
            if g.inlier_matches.shape[0] == 0:
                continue
            i1, i2 = divmod(int(pid), 2147483647)
            pairs.append((names[i1], names[i2]))
    return pairs


def load_scores(ws: Workspace, scores_id: str) -> dict[tuple[str, str], float]:
    z = np.load(ws.abspath(scores_id), allow_pickle=False)
    return {(str(a), str(b)): float(s) for a, b, s in zip(z["name0"], z["name1"], z["score"])}


def score_summary(scores: np.ndarray) -> dict[str, Any]:
    return {"num_pairs": int(len(scores)),
            "histogram": {f"<{t:.1f}": float((scores < t).mean()) if len(scores) else 0.0 for t in (0.2, 0.4, 0.5, 0.6, 0.8)},
            "median": float(np.median(scores)) if len(scores) else 0.0}


@register("score_pairs", "Score every verified pair of a database with a doppelganger / covisibility classifier. "
                         "Returns the distribution of scores; use filter_edges to remove low-scoring pairs.", PARAMS)
def score_pairs(ws: Workspace, cfg: dict[str, Any], db_id: str, method: str) -> ToolResult:
    started = time.time()
    params = {"method": method}
    id = ws.artifact_id("score_pairs", params, {"db": db_id})
    with ws.lock(id):
        ws.reload_registry()
        if (r := cached_result(ws, id)) is not None:
            return r
        return _score(ws, cfg, started, db_id, method, params, id)


def _score(ws, cfg, started, db_id, method, params, id) -> ToolResult:
    pairs = verified_pairs(ws.abspath(db_id))
    cost: dict[str, float] = {}
    if method == "xdg":
        out = run_backend(cfg, "xdg", "xdg", {"images_dir": str(ws.images_dir), "pairs": [list(p) for p in pairs],
                                              "out_dir": str(ws.path("scores", id, "_work")), "num_workers": 16},
                          ws.path("scores", id, "_work"), gpu=ws.gpus[-1] if ws.gpus else None)
        lookup = {tuple(k.split("/")): v for k, v in out["scores"].items()}
        cost = {k: v for k, v in out["cost"].items() if isinstance(v, (int, float))}
    elif method == "flexglue_covis":
        matches = ws.ancestor(db_id, "matches")
        if matches is None or matches.params.get("matcher") != "flexglue":
            raise ValueError("flexglue_covis needs a database matched with flexglue")
        with h5py.File(ws.abspath(matches.id), "r") as f:
            lookup = {}
            for a in f:
                for b in f[a]:
                    lookup[(a, b)] = float(f[a][b].attrs["covisibility"])
    elif method == "cycle_consistency":
        from agenticsfm.tools.relpose import cycle_consistency, relative_poses

        rel = relative_poses(ws.abspath(db_id), ws.path("scores", f"relpose_{db_id}", ".npz"))
        cyc = cycle_consistency(rel)
        # No triangles: no evidence either way; 0.5 so that a threshold at 0.5 keeps the edge.
        lookup = {k: (v[0] if v[1] > 0 else 0.5) for k, v in cyc.items()}
        for a, b in pairs:  # pairs whose relative pose could not be estimated: no evidence
            lookup.setdefault((a, b), 0.5)
        cost = {"num_triplet_edges": sum(1 for v in cyc.values() if v[1] > 0)}
    else:
        raise ValueError(f"unknown scoring method {method}")
    scores = np.array([lookup.get((a, b), lookup.get((b, a), np.nan)) for a, b in pairs], np.float32)
    if np.isnan(scores).any():
        raise RuntimeError(f"{int(np.isnan(scores).sum())} verified pairs have no score")
    path = ws.path("scores", id, ".npz")
    np.savez(path, name0=np.array([a for a, _ in pairs]), name1=np.array([b for _, b in pairs]), score=scores)
    summary = {"method": method, **score_summary(scores)}
    ws.path("scores", id, ".json").write_text(json.dumps(summary, indent=1))
    art = Artifact(id, "scores", "score_pairs", params, {"db": db_id}, str(path.relative_to(ws.store)), summary, cost)
    return finish(ws, art, started)
