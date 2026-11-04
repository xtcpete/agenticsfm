"""build_graph: which image pairs to match."""

from __future__ import annotations

import json
import time
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from agenticsfm.backends import run_backend
from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "enum": ["exhaustive", "netvlad", "sequential"],
                   "description": "exhaustive: all pairs (fine up to ~150 images); netvlad: top-k retrieval "
                                  "per image; sequential: each image with its k neighbours in name order"},
        "k": {"type": "integer", "default": 50, "description": "neighbours per image (netvlad, sequential)"},
    },
    "required": ["method"],
}


def graph_stats(names: list[str], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    deg = Counter()
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    degrees = np.array([deg[n] for n in names])
    return {"num_images": len(names), "num_pairs": len(pairs),
            "degree": {"min": int(degrees.min()) if len(degrees) else 0, "median": float(np.median(degrees)) if len(degrees) else 0,
                       "max": int(degrees.max()) if len(degrees) else 0},
            "isolated_images": [n for n in names if deg[n] == 0][:20]}


def read_pairs(ws: Workspace, graph_id: str) -> list[tuple[str, str]]:
    return [tuple(line.split()) for line in ws.abspath(graph_id).read_text().splitlines() if line]


@register("build_graph", "Choose candidate image pairs (the view graph) to match. Returns graph statistics.", PARAMS)
def build_graph(ws: Workspace, cfg: dict[str, Any], method: str, k: int = 50) -> ToolResult:
    started = time.time()
    params = {"method": method, "k": int(k)} if method != "exhaustive" else {"method": method}
    id = ws.artifact_id("build_graph", params, {})
    if (r := cached_result(ws, id)) is not None:
        return r
    names = ws.image_names()
    cost: dict[str, float] = {}
    if method == "exhaustive":
        pairs = list(combinations(names, 2))
    elif method == "sequential":
        pairs = sorted({(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, min(len(names), i + 1 + k))})
    elif method == "netvlad":
        out = run_backend(cfg, "retrieval", "pointglue",
                          {"images_dir": str(ws.images_dir), "image_names": names, "method": "netvlad", "k": int(k),
                           "out_dir": str(ws.path("graph", id, "_netvlad"))},
                          ws.path("graph", id, "_netvlad"), gpu=ws.gpus[0] if ws.gpus else None)
        pairs = [tuple(p) for p in out["pairs"]]
        cost = out["cost"]
    else:
        raise ValueError(f"unknown graph method {method}")
    path = ws.path("graph", id, ".txt")
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    summary = {"method": method, **graph_stats(names, pairs)}
    ws.path("graph", id, ".json").write_text(json.dumps(summary, indent=1))
    art = Artifact(id, "graph", "build_graph", params, {}, str(path.relative_to(ws.store)), summary,
                   {k: v for k, v in cost.items() if isinstance(v, (int, float))})
    return finish(ws, art, started)
