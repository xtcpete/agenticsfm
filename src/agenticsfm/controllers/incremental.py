"""Incremental controller: the open solver end to end, no COLMAP mapper.

graph -> match -> [covisibility filter] -> grow_model -> examine -> optimize (kept if the quality does
not drop) -> submit. The same tools and refinement policy as the scripted controller, with
``grow_model`` in place of ``reconstruct``; the comparison between the two is the comparison of the
mappers.
"""

from __future__ import annotations

from typing import Any

from agenticsfm.controllers.fixed import _graph_step
from agenticsfm.controllers.scripted import make_step, quality, recipe_of
from agenticsfm.workspace import Workspace


def run_incremental(ws: Workspace, cfg: dict[str, Any], verbose: bool = True, dataset: str | None = None,
                    feedforward: str | None = None, threshold: float | None = None, optimize: bool = True,
                    seed_size: int = 12, group_size: int = 8, min_support: float = 0.5, hard: bool = True, vlm: str = "decide", multi_model: bool = True, support_pairs: int = 1000) -> dict[str, Any]:
    n = len(ws.image_names())
    step = make_step(ws, cfg, "incremental", verbose)
    rc = recipe_of(cfg, dataset)
    ff = feedforward or cfg.get("feedforward", {}).get("model", "pi3x")
    graph = step("build_graph", **_graph_step(ws)).artifact
    db = step("match", graph_id=graph, detector=rc["detector"], matcher="flexglue", num_keypoints=rc["num_keypoints"], resize=rc["resize"]).artifact
    scores = step("score_pairs", db_id=db, method="flexglue_covis").artifact   # for the scene split (and the optional filter)
    if threshold is not None:
        db = step("filter_edges", db_id=db, scores_id=scores, threshold=float(threshold)).artifact
        scores = step("score_pairs", db_id=db, method="flexglue_covis").artifact
    m = step("grow_model", db_id=db, feedforward=ff, seed_size=seed_size, group_size=group_size, min_support=min_support, hard=hard, vlm=vlm,
             multi_model=multi_model, scores_id=scores, support_pairs=support_pairs)
    e = step("examine", model_id=m.artifact).summary
    best = m.artifact
    grow = m.summary.get("grow", {})
    decisions: dict[str, Any] = {
        "grow": {k: grow.get(k) for k in ("num_registered", "num_images", "num_models", "rounds", "hard_rounds", "hard_placed", "held", "rejected_total", "pnp_failed_total", "seed", "wall", "feedforward_calls", "vlm_calls", "extra_models")},
        "examine": {"largest": e["models"][0]["num_images"] if e.get("models") else 0,
                    "strong_inconsistent_frac": e.get("edges", {}).get("consistency", {}).get("strong_inconsistent_frac"),
                    "quality": quality(e, n)}}
    if optimize:
        opt = step("optimize", model_id=best, camera_model="OPENCV", share_intrinsics="cluster")
        if not opt.error:
            e_opt = step("examine", model_id=opt.artifact).summary
            ok = quality(e_opt, n) >= quality(e, n) - 0.01
            decisions["optimize"] = {"model": opt.artifact, "kept": ok, "quality": quality(e_opt, n), "quality_before": quality(e, n)}
            if ok:
                best = opt.artifact
        else:
            decisions["optimize"] = {"error": opt.summary.get("error")}
    step("submit", model_id=best)
    ws.log({"source": "incremental", "decisions": decisions, "submitted": best})
    return {"submitted": best, "decisions": decisions}
