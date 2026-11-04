"""partition_graph: split the view graph into overlapping parts for local reconstruction.

Recursive spectral bisection (Fiedler vector of the normalised Laplacian) of the graph weighted
by inliers (times a pair score when given) until every part has at most ``target_size`` images;
parts smaller than ``min_size`` are absorbed into their best-connected neighbour. Each part is
then extended with its strongest outside neighbours (``overlap`` fraction of the part), so that
neighbouring parts share images and ``merge_models`` can align them. The parts are an artifact
(JSON list of image lists) that reconstruct(images=...) consumes; thousands of images become a few
dozen mappings of a few hundred images each, which run in parallel and merge.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.score_pairs import load_scores
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string"},
        "scores_id": {"type": "string", "description": "optional pair scores: edges below 0.5 count less, so parts do not straddle doppelganger edges"},
        "target_size": {"type": "integer", "default": 150, "description": "largest part before the overlap is added"},
        "min_size": {"type": "integer", "default": 20},
        "overlap": {"type": "number", "default": 0.3, "description": "extra neighbouring images per part, as a fraction of its size "
                                                                   "(at least 10); parts must share several registered images to be merged"},
    },
    "required": ["db_id"],
}


def weighted_graph(ws: Workspace, db_id: str, scores_id: str | None):
    import pycolmap

    sc = load_scores(ws, scores_id) if scores_id else {}
    W: dict[tuple[str, str], float] = {}
    with pycolmap.Database.open(ws.abspath(db_id)) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            n = int(g.inlier_matches.shape[0])
            if n == 0:
                continue
            i1, i2 = divmod(int(pid), 2147483647)
            a, b = names[i1], names[i2]
            w = float(np.log1p(n))
            if sc:
                s = sc.get((a, b), sc.get((b, a), 0.5))
                w *= 0.2 + 0.8 * s
            W[(a, b)] = w
    return sorted(names.values()), W


def bisect(nodes: list[str], W: dict) -> tuple[list[str], list[str]]:
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for (a, b), w in W.items():
        if a in idx and b in idx:
            A[idx[a], idx[b]] += w
            A[idx[b], idx[a]] += w
    d = A.sum(1)
    Dm = np.diag(1 / np.sqrt(np.maximum(d, 1e-9)))
    L = np.eye(n) - Dm @ A @ Dm
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1] if n > 1 else np.zeros(n)
    order = np.argsort(f)
    half = n // 2
    # cut near the median of the Fiedler vector but at the sparsest point in the middle 40%
    best, best_cut = half, float("inf")
    for k in range(int(0.3 * n), int(0.7 * n) + 1):
        left, right = order[:k], order[k:]
        cut = A[np.ix_(left, right)].sum() / max(1e-9, min(d[left].sum(), d[right].sum()))
        if cut < best_cut:
            best, best_cut = k, cut
    return [nodes[i] for i in order[:best]], [nodes[i] for i in order[best:]]


def partition(nodes: list[str], W: dict, target: int, min_size: int) -> list[list[str]]:
    parts = [nodes]
    out = []
    while parts:
        p = parts.pop()
        if len(p) <= target:
            out.append(p)
            continue
        a, b = bisect(p, W)
        if min(len(a), len(b)) < max(2, min_size // 2):
            out.append(p)
            continue
        parts += [a, b]
    # absorb tiny parts into the neighbour they connect to most
    changed = True
    while changed and len(out) > 1:
        changed = False
        out.sort(key=len)
        small = out[0]
        if len(small) >= min_size:
            break
        best, best_w = None, -1.0
        sset = set(small)
        for j, other in enumerate(out[1:], start=1):
            oset = set(other)
            w = sum(v for (a, b), v in W.items() if (a in sset and b in oset) or (b in sset and a in oset))
            if w > best_w:
                best, best_w = j, w
        if best is not None:
            out[best] = out[best] + small
            out.pop(0)
            changed = True
    return out


def add_overlap(parts: list[list[str]], W: dict, overlap: float) -> list[list[str]]:
    nb: dict[str, dict[str, float]] = {}
    for (a, b), w in W.items():
        nb.setdefault(a, {})[b] = w
        nb.setdefault(b, {})[a] = w
    out = []
    for p in parts:
        pset = set(p)
        k = max(10, int(round(overlap * len(p))))
        cand: dict[str, float] = {}
        for n in p:
            for m, w in nb.get(n, {}).items():
                if m not in pset:
                    cand[m] = cand.get(m, 0.0) + w
        extra = [m for m, _ in sorted(cand.items(), key=lambda kv: -kv[1])[:k]]
        out.append(sorted(p) + extra)
    return out


@register("partition_graph", "Split the view graph into overlapping parts of at most target_size images (spectral bisection "
                             "weighted by inliers and, optionally, pair scores). Reconstruct each part with "
                             "reconstruct(images=part) and attach them with merge_models. For scenes of many hundreds or "
                             "thousands of images.", PARAMS)
def partition_graph(ws: Workspace, cfg: dict[str, Any], db_id: str, scores_id: str | None = None, target_size: int = 150,
                    min_size: int = 20, overlap: float = 0.3) -> ToolResult:
    started = time.time()
    params = {"target_size": int(target_size), "min_size": int(min_size), "overlap": float(overlap), "v": 2}
    inputs = {"db": db_id, **({"scores": scores_id} if scores_id else {})}
    id = ws.artifact_id("partition_graph", params, inputs)
    if (r := cached_result(ws, id)) is not None:
        return r
    names, W = weighted_graph(ws, db_id, scores_id)
    connected = sorted({x for p in W for x in p})
    parts = partition(connected, W, target_size, min_size)
    parts = add_overlap(parts, W, overlap)
    path = ws.path("graph", id, "_parts.json")
    path.write_text(json.dumps(parts))
    cross = {}
    for i, p in enumerate(parts):
        for j, q in enumerate(parts):
            if j > i:
                shared = len(set(p) & set(q))
                if shared:
                    cross[f"{i}-{j}"] = shared
    summary = {"num_parts": len(parts), "part_sizes": [len(p) for p in parts], "shared_images_between_parts": cross,
               "isolated_images": len(names) - len(connected),
               "parts_preview": [p[:4] + (["..."] if len(p) > 4 else []) for p in parts[:6]],
               "_doc": "parts overlap so that merge_models can align them; reconstruct each with images=part (the artifact holds the lists; "
                       "analyze can read it with json.load(open(ws.abspath('<id>'))))"}
    art = Artifact(id, "graph", "partition_graph", params, inputs, str(path.relative_to(ws.store)), summary)
    return finish(ws, art, started)
