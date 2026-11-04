"""Scene split: separate the input images into scenes BEFORE any feedforward call.

A feedforward model forces whatever it is given into one frame, so images of different sites must
never share a seed group or an anchor set. The split works on the verified view graph:

* edge weight = FlexGlue covisibility x min(inliers, cap) / cap: a spurious cross-scene pair is weak
  on at least one of the two (on the mixed M07 + terrace folder all 11 cross pairs have 15 to 26
  inliers, at covisibility 0.41 to 0.76; within-scene pairs sit at covisibility 1.0);
* connected components first (no verified pair at all between them);
* then recursive spectral bisection with a sweep cut on the Fiedler vector of the normalised
  Laplacian, accepted when the conductance of the best cut (crossing weight over the smaller side's
  volume) is below ``split_conductance`` and both sides have at least ``min_scene`` images. Two sites
  are joined by a few scattered edges, one site by many, which holds even when single spurious
  edges score high;
* cuts with conductance in the band up to ``ambiguous_conductance`` go to the VLM (thumbnail grids
  of both sides: same site?) when an advisor is given; otherwise they are kept together.

Small weakly attached groups (fewer than ``min_scene`` images) are never split off: they stay with
their scene and are handled by the hard-image stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_IMAGE_ID = 2147483647


@dataclass
class SplitConfig:
    min_inliers: int = 15
    inlier_cap: int = 100
    min_scene: int = 5
    split_conductance: float = 0.01
    ambiguous_conductance: float = 0.05
    max_depth: int = 8


def weighted_graph(db_path, scores: dict[tuple[str, str], float] | None, sc: SplitConfig):
    """names, symmetric weight matrix (dense), and per-pair (inliers, covis) for reporting."""
    import pycolmap

    with pycolmap.Database.open(str(db_path)) as db:
        names_by_id = {im.image_id: im.name for im in db.read_all_images()}
        ids, geoms = db.read_two_view_geometries()
    names = sorted(names_by_id.values())
    idx = {n: k for k, n in enumerate(names)}
    W = np.zeros((len(names), len(names)), dtype=np.float64)
    for pid, g in zip(ids, geoms):
        n = int(g.inlier_matches.shape[0])
        if n < sc.min_inliers:
            continue
        i, j = divmod(int(pid), MAX_IMAGE_ID)
        a, b = names_by_id[i], names_by_id[j]
        cov = 1.0
        if scores is not None:
            cov = scores.get((a, b), scores.get((b, a), 0.0))
        w = cov * min(n, sc.inlier_cap) / sc.inlier_cap
        W[idx[a], idx[b]] = W[idx[b], idx[a]] = max(W[idx[a], idx[b]], w)
    return names, W


def components(W: np.ndarray, nodes: list[int]) -> list[list[int]]:
    nodes_set, seen, out = set(nodes), set(), []
    for s in nodes:
        if s in seen:
            continue
        comp, todo = [], [s]
        seen.add(s)
        while todo:
            u = todo.pop()
            comp.append(u)
            for v in np.flatnonzero(W[u] > 0):
                v = int(v)
                if v in nodes_set and v not in seen:
                    seen.add(v)
                    todo.append(v)
        out.append(comp)
    return sorted(out, key=len, reverse=True)


def best_cut(W: np.ndarray, nodes: list[int], min_side: int) -> tuple[list[int], list[int], float] | None:
    """Sweep cut on the Fiedler vector of the normalised Laplacian; the cut of lowest conductance with both sides >= min_side."""
    A = W[np.ix_(nodes, nodes)]
    d = A.sum(1)
    if len(nodes) < 2 * min_side or (d <= 0).any():
        return None
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(len(nodes)) - (Dm[:, None] * A * Dm[None, :])
    if len(nodes) <= 2000:
        vals, vecs = np.linalg.eigh(L)
        f = vecs[:, 1] * Dm
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        vals, vecs = eigsh(csr_matrix(L), k=2, sigma=-1e-3, which="LM")
        f = vecs[:, np.argsort(vals)[1]] * Dm
    order = np.argsort(f)
    vol_total = d.sum()
    in_left = np.zeros(len(nodes), bool)
    cut = 0.0
    vol_left = 0.0
    best = None
    for k in range(len(nodes) - 1):
        u = order[k]
        # moving u to the left: edges from u to the left stop crossing, edges to the right start crossing
        to_left = A[u, in_left].sum()
        cut += d[u] - 2 * to_left
        in_left[u] = True
        vol_left += d[u]
        nl = k + 1
        if nl < min_side or len(nodes) - nl < min_side:
            continue
        cond = cut / max(min(vol_left, vol_total - vol_left), 1e-12)
        if best is None or cond < best[0]:
            best = (cond, nl)
    if best is None:
        return None
    cond, nl = best
    left = [nodes[i] for i in order[:nl]]
    right = [nodes[i] for i in order[nl:]]
    return left, right, float(cond)


def split_scenes(db_path, scores, sc: SplitConfig | None = None, advisor=None, sample_k: int = 6) -> dict[str, Any]:
    """Returns {"scenes": [[names]], "unassigned": [names], "decisions": [...]}."""
    sc = sc or SplitConfig()
    names, W = weighted_graph(db_path, scores, sc)
    decisions: list[dict[str, Any]] = []
    isolated = [i for i in range(len(names)) if W[i].sum() == 0]
    work = [c for c in components(W, [i for i in range(len(names)) if i not in set(isolated)])]
    decisions.append({"step": "components", "sizes": [len(c) for c in work], "isolated": len(isolated)})
    small = [c for c in work if len(c) < sc.min_scene]
    queue = [(c, 0) for c in work if len(c) >= sc.min_scene]
    scenes: list[list[int]] = []
    while queue:
        nodes, depth = queue.pop(0)
        res = best_cut(W, nodes, sc.min_scene) if depth < sc.max_depth else None
        if res is None:
            scenes.append(nodes)
            continue
        left, right, cond = res
        row = {"step": "cut", "depth": depth, "sizes": [len(left), len(right)], "conductance": round(cond, 5)}
        if cond < sc.split_conductance:
            row["decision"] = "split"
        elif cond < sc.ambiguous_conductance and advisor is not None and advisor.enabled:
            # the VLM sees samples of each side, preferring the images on each side most connected to the other
            cross_l = W[np.ix_(left, right)].sum(1)
            cross_r = W[np.ix_(right, left)].sum(1)
            sa = [names[left[i]] for i in np.argsort(-cross_l)[:sample_k]]
            sb = [names[right[i]] for i in np.argsort(-cross_r)[:sample_k]]
            v = advisor.same_scene(sa, sb, {"conductance": round(cond, 5), "sizes": [len(left), len(right)]})
            row.update({"vlm": v, "decision": "split" if v == "different" else "keep"})
        else:
            row["decision"] = "keep"
        decisions.append(row)
        if row["decision"] == "split":
            queue.append((left, depth + 1))
            queue.append((right, depth + 1))
        else:
            scenes.append(nodes)
    scenes.sort(key=len, reverse=True)
    # small components join the scene they share most weight with, else stay unassigned
    unassigned = [names[i] for i in isolated]
    for c in small:
        best, bw = None, 0.0
        for k, s in enumerate(scenes):
            w = W[np.ix_(c, s)].sum()
            if w > bw:
                best, bw = k, w
        if best is None:
            unassigned += [names[i] for i in c]
        else:
            scenes[best] = scenes[best] + c
    return {"scenes": [[names[i] for i in s] for s in scenes], "unassigned": sorted(unassigned), "decisions": decisions,
            "config": sc.__dict__}
