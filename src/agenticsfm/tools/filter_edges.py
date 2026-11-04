"""filter_edges: a new database without the pairs below a score threshold (or in a list)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.match import db_summary
from agenticsfm.tools.score_pairs import load_scores
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string"},
        "scores_id": {"type": "string", "description": "from score_pairs"},
        "threshold": {"type": "number", "description": "drop verified pairs scored below this"},
        "drop_pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                       "description": "explicit [name0, name1] pairs to drop (instead of, or besides, the threshold)"},
        "drop_images": {"type": "array", "items": {"type": "string"},
                        "description": "images to leave out entirely (all their pairs are dropped): distractors or images "
                                       "verify_poses shows to be placed wrongly"},
        "keep_connected": {"type": "boolean", "default": True,
                           "description": "after thresholding, restore the highest-scoring dropped pairs that reconnect "
                                          "split components, so no image is cut off by the threshold alone"},
    },
    "required": ["db_id"],
}


def components(names: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=len, reverse=True)


@register("filter_edges", "Make a new database that drops verified pairs scored below a threshold, pairs listed "
                          "explicitly, and/or whole images. Returns the remaining graph's statistics (pairs, components).", PARAMS)
def filter_edges(ws: Workspace, cfg: dict[str, Any], db_id: str, scores_id: str | None = None,
                 threshold: float | None = None, drop_pairs: list | None = None, drop_images: list | None = None,
                 keep_connected: bool = True) -> ToolResult:
    started = time.time()
    drop_pairs = [tuple(p) for p in (drop_pairs or [])]
    drop_images = set(drop_images or [])
    params = {"threshold": threshold, "drop_pairs": sorted(drop_pairs), "drop_images": sorted(drop_images),
              "keep_connected": bool(keep_connected and threshold is not None)}
    inputs = {"db": db_id, **({"scores": scores_id} if scores_id else {})}
    id = ws.artifact_id("filter_edges", params, inputs)
    if (r := cached_result(ws, id)) is not None:
        return r
    drop = set(drop_pairs) | {(b, a) for a, b in drop_pairs}
    restored: list = []
    if threshold is not None:
        if scores_id is None:
            raise ValueError("threshold needs scores_id")
        scores = load_scores(ws, scores_id)
        below = [((a, b), s) for (a, b), s in scores.items() if s < threshold]
        for (a, b), s in below:
            drop.add((a, b))
            drop.add((b, a))
        if keep_connected:
            # Kruskal over the dropped edges, best score first: an edge that joins two components of the
            # kept graph is restored. Images cut off only by the threshold stay reachable.
            alive = [n for n in {x for p in scores for x in p} if n not in drop_images]
            kept = [(a, b) for (a, b), s in scores.items() if s >= threshold and (a, b) not in drop
                    and a not in drop_images and b not in drop_images]
            parent = {n: n for n in alive}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in kept:
                parent[find(a)] = find(b)
            for (a, b), s in sorted(below, key=lambda e: -e[1]):
                if a in drop_images or b in drop_images or (a, b) in set(drop_pairs) or (b, a) in set(drop_pairs):
                    continue
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
                    drop.discard((a, b))
                    drop.discard((b, a))
                    restored.append([a, b, round(s, 3)])
    src, dst = ws.abspath(db_id), ws.path("db", id, ".db")
    shutil.copyfile(src, dst)
    con = sqlite3.connect(dst)
    names = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
    all_names = sorted(names.values())
    kept, removed = [], []
    for (pid,) in con.execute("SELECT pair_id FROM two_view_geometries").fetchall():
        i1, i2 = divmod(int(pid), 2147483647)
        pair = (names[i1], names[i2])
        rows = con.execute("SELECT rows FROM two_view_geometries WHERE pair_id=?", (pid,)).fetchone()[0]
        if pair in drop or pair[0] in drop_images or pair[1] in drop_images:
            removed.append(pid)
        elif rows > 0:
            kept.append(pair)
    con.executemany("DELETE FROM matches WHERE pair_id=?", [(p,) for p in removed])
    con.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", [(p,) for p in removed])
    con.commit()
    con.close()
    comps = components([n for n in all_names if n not in drop_images], kept)
    summary = {**db_summary(dst), "removed_pairs": len(removed), "threshold": threshold, "dropped_images": sorted(drop_images),
               "restored_for_connectivity": restored[:20], "num_restored": len(restored),
               "graph_components": [len(c) for c in comps[:10]], "num_graph_components": len(comps)}
    ws.path("db", id, ".json").write_text(json.dumps(summary, indent=1))
    art = Artifact(id, "db", "filter_edges", params, inputs, str(dst.relative_to(ws.store)), summary)
    return finish(ws, art, started)
