"""Fixed pipelines: a scripted sequence of tool calls, no diagnosis. These are the baselines, and
they reproduce the reference pipelines of the README from images.

A pipeline is a list of (tool, params) where a param value "$name" refers to the artifact of an
earlier step tagged ``as: name``. Graph selection: exhaustive up to ``retrieval_min_images``
images, NetVLAD top-k above (as pointglue's WRIVA protocol).
"""

from __future__ import annotations

from typing import Any

from agenticsfm.tools import ToolResult, call
from agenticsfm.workspace import Workspace

RETRIEVAL_MIN_IMAGES = 100
RETRIEVAL_K = 100


def _graph_step(ws: Workspace) -> dict[str, Any]:
    n = len(ws.image_names())
    return {"method": "exhaustive"} if n <= RETRIEVAL_MIN_IMAGES else {"method": "netvlad", "k": RETRIEVAL_K}


def _match(detector: str, matcher: str) -> dict[str, Any]:
    return {"graph_id": "$graph", "detector": detector, "matcher": matcher}


def pipeline(detector: str, matcher: str, mapper: str, filter: str | None = None, threshold: float = 0.6,
             num_keypoints: int = 4096, resize: int = 1024) -> list:
    steps = [("build_graph", "graph", None), ("match", "db", {**_match(detector, matcher), "num_keypoints": num_keypoints, "resize": resize})]
    db = "$db"
    if filter:
        steps.append(("score_pairs", "scores", {"db_id": "$db", "method": filter}))
        steps.append(("filter_edges", "db_f", {"db_id": "$db", "scores_id": "$scores", "threshold": threshold}))
        db = "$db_f"
    steps += [("reconstruct", "model", {"db_id": db, "backend": mapper}),
              ("examine", "exam", {"model_id": "$model"}),
              ("submit", None, {"model_id": "$model"})]
    return steps


PIPELINES: dict[str, list] = {
    "colmap_alikedlg": pipeline("aliked", "lightglue", "colmap"),
    "colmap_alikedfg": pipeline("aliked", "flexglue", "colmap"),
    "colmap_alikedfg_xdg": pipeline("aliked", "flexglue", "colmap", "xdg", 0.6),
    "colmap_alikedfg_covis": pipeline("aliked", "flexglue", "colmap", "flexglue_covis", 0.6),
    # FlexGlue paper, ETH3D (detector study setting): SuperPoint 4096 @ 1024, FlexGlue sparse, covisibility 0.9
    "colmap_spfg": pipeline("superpoint", "flexglue", "colmap"),
    "colmap_spfg_covis09": pipeline("superpoint", "flexglue", "colmap", "flexglue_covis", 0.9),
    "colmap_spfg_covis06": pipeline("superpoint", "flexglue", "colmap", "flexglue_covis", 0.6),
    "colmap_spfg1600_covis09": pipeline("superpoint", "flexglue", "colmap", "flexglue_covis", 0.9, 4096, 1600),  # detection resolution check
    "colmap_dadloma": pipeline("dad", "loma", "colmap"),
    "colmap_dadloma_xdg": pipeline("dad", "loma", "colmap", "xdg", 0.6),
    "glomap_alikedfg": pipeline("aliked", "flexglue", "glomap"),
    "glomap_alikedfg_covis": pipeline("aliked", "flexglue", "glomap", "flexglue_covis", 0.6),
    "glomap_alikedfg_xdg": pipeline("aliked", "flexglue", "glomap", "xdg", 0.6),
}


def run_pipeline(ws: Workspace, cfg: dict[str, Any], name: str, verbose: bool = True) -> dict[str, ToolResult]:
    refs: dict[str, str] = {}
    results: dict[str, ToolResult] = {}
    for tool, tag, params in PIPELINES[name]:
        if tool == "build_graph":
            params = _graph_step(ws)
        params = {k: (refs[v[1:]] if isinstance(v, str) and v.startswith("$") else v) for k, v in params.items()}
        result = call(ws, cfg, tool, params, source=f"fixed:{name}")
        if verbose:
            cached = " (cached)" if result.cached else ""
            print(f"[{ws.scene.name}] {tool}{cached} {result.cost.get('wall', 0):.1f}s -> {result.artifact} "
                  f"{_short(result.summary)}", flush=True)
        if result.error:
            raise RuntimeError(f"{tool} failed: {result.summary.get('error')}\n{result.error}")
        if tag:
            refs[tag] = result.artifact
        results[tool if tag is None else tag] = result
    return results


def _short(summary: dict[str, Any], limit: int = 300) -> str:
    import json

    s = json.dumps({k: v for k, v in summary.items() if k not in ("renders", "edges", "models", "matches")}, default=str)
    return s if len(s) <= limit else s[:limit] + "..."
