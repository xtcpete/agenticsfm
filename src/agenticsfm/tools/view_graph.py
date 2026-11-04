"""view_graph: the view graph of a database as a picture and as structure.

Nodes are images, placed at their camera positions in the scene frame when a model is given (then
the picture shows which edges cross the scene) or by a spectral layout otherwise. Edges are the
verified pairs, coloured by a pair score (covisibility or XDG from score_pairs, or examine's
consistency when a model is given); width grows with inliers. The summary lists the clusters
found by cutting the weakest edges, the bridges (edges whose removal disconnects the graph), the
articulation images, and the images with the fewest strong edges. A doppelganger appears as two
clusters joined by a few strong but low-scoring edges; a weak part of the scene as a cluster
attached by one bridge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, register
from agenticsfm.tools.filter_edges import components
from agenticsfm.tools.frame import cameras_of, scene_frame
from agenticsfm.tools.score_pairs import load_scores, verified_pairs
from agenticsfm.workspace import Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string"},
        "scores_id": {"type": "string", "description": "colour edges by these pair scores (from score_pairs)"},
        "model_id": {"type": "string", "description": "place nodes at this model's camera positions and colour edges by their consistency with it"},
        "cut_threshold": {"type": "number", "default": 0.5, "description": "edges with score below this are drawn dashed and cut for the cluster analysis"},
    },
    "required": ["db_id"],
}


def spectral_layout(names: list[str], edges: list[tuple[str, str, float]]) -> dict[str, np.ndarray]:
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    W = np.zeros((n, n))
    for a, b, w in edges:
        W[idx[a], idx[b]] += w
        W[idx[b], idx[a]] += w
    d = W.sum(1)
    L = np.diag(d) - W
    Dm = np.diag(1 / np.sqrt(np.maximum(d, 1e-9)))
    vals, vecs = np.linalg.eigh(Dm @ L @ Dm)
    coords = vecs[:, 1:3] if n > 2 else np.zeros((n, 2))
    coords = coords / (np.abs(coords).max() + 1e-9)
    return {name: coords[i] for name, i in idx.items()}


def bridges_and_articulations(names: list[str], edges: list[tuple[str, str]]):
    adj: dict[str, set] = {n: set() for n in names}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    disc, low, parent = {}, {}, {}
    bridges, arts = [], set()
    timer = [0]

    def dfs(u):
        stack = [(u, iter(adj[u]))]
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = {u: 0}
        while stack:
            v, it = stack[-1]
            advanced = False
            for w in it:
                if w not in disc:
                    parent[w] = v
                    children[v] = children.get(v, 0) + 1
                    children.setdefault(w, 0)
                    disc[w] = low[w] = timer[0]
                    timer[0] += 1
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
                elif w != parent.get(v):
                    low[v] = min(low[v], disc[w])
            if not advanced:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[v])
                    if low[v] > disc[p]:
                        bridges.append((p, v))
                    if parent.get(p) is not None and low[v] >= disc[p]:
                        arts.add(p)
                    if parent.get(p) is None and children.get(p, 0) > 1:
                        arts.add(p)

    for n in names:
        if n not in disc:
            parent[n] = None
            dfs(n)
    return bridges, sorted(arts)


@register("view_graph", "Draw the view graph (images as nodes, verified pairs as edges coloured by a pair score or by their "
                        "consistency with a model) and report its structure: clusters after cutting weak edges, bridges, "
                        "articulation images, weakly connected images.", PARAMS)
def view_graph(ws: Workspace, cfg: dict[str, Any], db_id: str, scores_id: str | None = None, model_id: str | None = None,
               cut_threshold: float = 0.5) -> ToolResult:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pycolmap

    started = time.time()
    pairs = verified_pairs(ws.abspath(db_id))
    inl = {}
    with pycolmap.Database.open(ws.abspath(db_id)) as db:
        names_by_id = {im.image_id: im.name for im in db.read_all_images()}
        ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            i1, i2 = divmod(int(pid), 2147483647)
            inl[(names_by_id[i1], names_by_id[i2])] = int(g.inlier_matches.shape[0])
    score = {}
    label = "inliers (normalised)"
    if scores_id:
        score = load_scores(ws, scores_id)
        label = ws.get(scores_id).params["method"]
    elif model_id:
        exams = [a for a in ws.list("examine") if a.tool == "examine" and a.inputs.get("model") == model_id]
        if exams:
            for line in (ws.abspath(exams[-1].id) / "edges.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["consistent_frac"] is not None:
                    score[tuple(r["pair"])] = r["consistent_frac"]
            label = "consistency with the model"
    if not score:
        mx = max(inl.values()) if inl else 1
        score = {p: min(1.0, inl[p] / (0.2 * mx)) for p in inl}
    names = sorted({x for p in pairs for x in p})
    sc = {p: score.get(p, score.get((p[1], p[0]), 0.5)) for p in pairs}
    strong = [p for p in pairs if inl.get(p, 0) >= 50]
    kept = [p for p in pairs if sc[p] >= cut_threshold]
    clusters = components(names, kept)
    br, arts = bridges_and_articulations(names, pairs)
    deg_strong = {n: 0 for n in names}
    for a, b in strong:
        deg_strong[a] += 1
        deg_strong[b] += 1
    cross = []
    if len(clusters) > 1:
        cl = {n: i for i, c in enumerate(clusters) for n in c}
        cross = sorted([(inl.get(p, 0), p) for p in pairs if cl[p[0]] != cl[p[1]]], reverse=True)
    # layout
    pos = None
    if model_id:
        subs = [pycolmap.Reconstruction(str(p)) for p in sorted(ws.abspath(model_id).iterdir()) if p.is_dir() and p.name.isdigit()]
        rec = max(subs, key=lambda r: r.num_reg_images())
        fr = scene_frame(ws.store, model_id, rec)
        cams = cameras_of(rec)
        pos = {n: fr.apply(cams[n][0][None])[0][:2] for n in cams}
        missing = [n for n in names if n not in pos]
        if missing:
            ring = np.linspace(0, 2 * np.pi, len(missing), endpoint=False)
            for n, t in zip(missing, ring):
                pos[n] = np.array([1.15 * fr.span * np.cos(t), 1.15 * fr.span * np.sin(t)])
    if pos is None:
        pos = spectral_layout(names, [(a, b, max(sc[(a, b)], 0.05) * inl.get((a, b), 1)) for a, b in pairs])
    out_dir = ws.run_dir / "graphs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{len(list(out_dir.iterdir())):03d}_{db_id}.png"
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.get_cmap("RdYlGn")
    order = sorted(pairs, key=lambda p: sc[p])
    for a, b in order:
        w = 0.3 + 2.0 * min(1.0, inl.get((a, b), 0) / 500)
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=cmap(sc[(a, b)]), lw=w, alpha=0.5 if sc[(a, b)] >= cut_threshold else 0.35,
                ls="-" if sc[(a, b)] >= cut_threshold else (0, (2, 2)), zorder=1)
    for a, b in br:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="black", lw=2.2, zorder=2)
    cl = {n: i for i, c in enumerate(clusters) for n in c}
    node_colors = [plt.get_cmap("tab10")(cl[n] % 10) for n in names]
    ax.scatter([pos[n][0] for n in names], [pos[n][1] for n in names], s=70, c=node_colors, edgecolors="black", linewidths=0.6, zorder=3)
    for n in names:
        import re

        m = re.findall(r"(\d+)", n)
        ax.annotate(str(int(m[-1])) if m else n[:5], pos[n], fontsize=6, xytext=(3, 3), textcoords="offset points", zorder=4)
    ax.set_aspect("equal")
    ax.set_title(f"view graph of {db_id}: {len(names)} images, {len(pairs)} verified pairs; edge colour = {label} (red low, green high), "
                 f"dashed below {cut_threshold}; black = bridge; node colour = cluster after the cut", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    summary = {
        "num_images": len(names), "num_pairs": len(pairs), "edge_score": label,
        "clusters_after_cut": [len(c) for c in clusters[:10]], "num_clusters": len(clusters),
        "cluster_members": [[c[0], "...", c[-1]] if len(c) > 6 else c for c in clusters[:5]],
        "strongest_cross_cluster_pairs": [{"pair": list(p), "inliers": n, "score": round(sc[p], 3)} for n, p in cross[:10]],
        "bridges": [list(p) for p in br[:15]], "num_bridges": len(br), "articulation_images": arts[:15],
        "images_with_fewest_strong_pairs": sorted(deg_strong.items(), key=lambda kv: kv[1])[:10],
        "layout": "camera positions of " + model_id if model_id else "spectral",
        "_doc": "cross-cluster pairs with many inliers but low score are the doppelganger candidates; a bridge is the only "
                "link of some images: dropping it cuts them off",
    }
    return ToolResult(None, summary, cost={"wall": time.time() - started}, images=[str(path)])
