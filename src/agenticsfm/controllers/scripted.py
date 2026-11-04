"""Scripted controller: the playbook as fixed diagnose-and-repair rules, with the same tools and
budget as the agent. The main ablation: the agent must beat this.

1. graph -> match (aliked + flexglue) -> colmap -> examine.
2. If the model is incomplete (unregistered images, several models) or inconsistent (pairs whose
   matches disagree with the poses), score pairs with FlexGlue's covisibility, filter at 0.6,
   map again, examine.
3. If the filtered colmap model is still incomplete, try glomap on the filtered database.
4. verify_poses on the best candidate; if it flags suspect images, drop them and map again.
5. Pick the candidate with the best GT-free quality (fraction of verified images in the largest
   model minus a penalty for inconsistent pairs), refine it with `optimize` (shared intrinsics per
   device and zoom level, OPENCV model, bundle adjustment) and submit.
"""

from __future__ import annotations

from typing import Any

from agenticsfm.controllers.fixed import _graph_step
from agenticsfm.tools import ToolResult, call
from agenticsfm.workspace import Workspace

INCOMPLETE_FRACTION = 0.05   # unregistered images beyond this count as incomplete
INCONSISTENT_FRACTION = 0.05  # strong pairs mostly disagreeing with the poses beyond this count as folded


LARGE_SCENE = 1000  # above this many images, map partitions concurrently and merge (reconstruct_parts); one COLMAP mapping of 600 images takes ~45 min, which is affordable, and the partition/merge path is not yet as accurate
# GLOMAP-derived models look best to every match-based statistic (GLOMAP fits those pairwise constraints directly) but
# score 1 to 9 points lower than COLMAP models of the same coverage on the dev split (catastrophic on PTZ scenes).
# Audited: a 0.10 penalty (about 4 images in 39) lowers the choice regret from 2.2 to 1.5 points.
GLOMAP_PENALTY = 0.10


def quality(exam: dict[str, Any], num_images: int, verify: dict[str, Any] | None = None, glomap: bool = False) -> float:
    largest = exam["models"][0]["num_images"] if exam.get("models") else 0
    inconsistent = exam.get("edges", {}).get("consistency", {}).get("strong_inconsistent_frac", 0.0)
    suspects = verify.get("num_suspect", 0) if verify else 0
    medc = exam.get("edges", {}).get("consistency", {}).get("median", 0.0)
    return (largest - suspects) / max(1, num_images) - 2.0 * inconsistent + 0.3 * medc - (GLOMAP_PENALTY if glomap else 0.0)


def is_glomap(ws: Workspace, model_id: str) -> bool:
    return any(a.tool == "reconstruct" and a.params.get("backend") == "glomap" for a in ws.provenance(model_id))


def incomplete(exam: dict[str, Any], num_images: int) -> bool:
    largest = exam["models"][0]["num_images"] if exam.get("models") else 0
    return largest < (1 - INCOMPLETE_FRACTION) * num_images


def inconsistent(exam: dict[str, Any]) -> bool:
    return exam.get("edges", {}).get("consistency", {}).get("strong_inconsistent_frac", 0.0) > INCONSISTENT_FRACTION


def make_step(ws: Workspace, cfg: dict[str, Any], source: str, verbose: bool):
    def step(tool: str, **params) -> ToolResult:
        r = call(ws, cfg, tool, params, source=source)
        if verbose:
            print(f"[{ws.scene.name}] {tool}{' (cached)' if r.cached else ''} {r.cost.get('wall', 0):.0f}s -> {r.artifact}", flush=True)
        if r.error:
            raise RuntimeError(f"{tool} failed: {r.summary.get('error')}")
        return r
    return step


DEFAULT_RECIPE = {"detector": "aliked", "num_keypoints": 4096, "resize": 1024, "thresholds": [0.6, 0.4, 0.3]}


def recipe_of(cfg: dict[str, Any], dataset: str | None) -> dict[str, Any]:
    """The benchmark's matching recipe (detector, keypoints, resolution, filter thresholds to sweep)."""
    r = dict(DEFAULT_RECIPE)
    if dataset and dataset in cfg.get("datasets", {}):
        r.update(cfg["datasets"][dataset].get("recipe", {}))
    return r


def candidates_and_best(ws: Workspace, cfg: dict[str, Any], step, dataset: str | None = None) -> tuple[list, str, dict]:
    """Steps 1-4 of the policy: the candidate models, the best one by quality (after the verify/drop
    round), and the decisions taken. Shared by the scripted controller and the agent's prelude."""
    n = len(ws.image_names())
    rc = recipe_of(cfg, dataset)
    graph = step("build_graph", **_graph_step(ws)).artifact
    db = step("match", graph_id=graph, detector=rc["detector"], matcher="flexglue", num_keypoints=rc["num_keypoints"], resize=rc["resize"]).artifact
    candidates: list[tuple[str, dict]] = []
    if n > LARGE_SCENE:
        # Scale: partition the view graph, map the parts concurrently, merge, refine; one global mapping
        # of thousands of images would take hours.
        parts = step("partition_graph", db_id=db, target_size=150).artifact
        m0 = step("reconstruct_parts", db_id=db, parts_id=parts, backend="colmap", workers=4).artifact
    else:
        m0 = step("reconstruct", db_id=db, backend="colmap").artifact
    e0 = step("examine", model_id=m0).summary
    candidates.append((m0, e0))
    decisions = {"unfiltered": {"incomplete": incomplete(e0, n), "inconsistent": inconsistent(e0), "quality": quality(e0, n)}}
    if incomplete(e0, n) or inconsistent(e0):
        scores = step("score_pairs", db_id=db, method="flexglue_covis").artifact
        # Threshold sweep: the fold disappears at some threshold; above it true low-overlap edges are
        # lost with their images. Map a few values and let the quality rule choose.
        sweep = {}
        first = rc["thresholds"][0]
        for t in rc["thresholds"]:
            dbt = step("filter_edges", db_id=db, scores_id=scores, threshold=t).artifact
            if n > LARGE_SCENE:
                pt = step("partition_graph", db_id=dbt, target_size=150).artifact
                mt = step("reconstruct_parts", db_id=dbt, parts_id=pt, backend="colmap", workers=4).artifact
            else:
                mt = step("reconstruct", db_id=dbt, backend="colmap").artifact
            et = step("examine", model_id=mt).summary
            candidates.append((mt, et))
            sweep[t] = {"model": mt, "incomplete": incomplete(et, n), "inconsistent": inconsistent(et), "quality": quality(et, n)}
            if t == first:
                db1, m1, e1 = dbt, mt, et
        decisions["filtered_colmap"] = sweep
        if incomplete(e1, n):
            m2 = step("reconstruct", db_id=db1, backend="glomap").artifact
            e2 = step("examine", model_id=m2).summary
            candidates.append((m2, e2))
            decisions["filtered_glomap"] = {"incomplete": incomplete(e2, n), "inconsistent": inconsistent(e2), "quality": quality(e2, n, glomap=True)}
    best, best_exam = max(candidates, key=lambda c: quality(c[1], n, glomap=is_glomap(ws, c[0])))
    verify = step("verify_poses", model_id=best).summary
    decisions["verify"] = {"model": best, "num_suspect": verify.get("num_suspect"), "suspects": [x["image"] for x in verify.get("suspect_images", [])]}
    if verify.get("num_suspect", 0) > 0:
        backend = ws.get(best).params["backend"]
        db_best = ws.ancestor(best, "db").id
        db2 = step("filter_edges", db_id=db_best, drop_images=[x["image"] for x in verify["suspect_images"]]).artifact
        m3 = step("reconstruct", db_id=db2, backend=backend).artifact
        e3 = step("examine", model_id=m3).summary
        v3 = step("verify_poses", model_id=m3).summary
        q_old, q_new = quality(best_exam, n, verify, is_glomap(ws, best)), quality(e3, n, v3, is_glomap(ws, m3))
        decisions["after_drop"] = {"model": m3, "num_suspect": v3.get("num_suspect"), "quality": q_new, "quality_before": q_old}
        if q_new > q_old:
            best = m3
            candidates.append((m3, e3))
    return candidates, best, decisions


def run_scripted(ws: Workspace, cfg: dict[str, Any], verbose: bool = True, dataset: str | None = None) -> dict[str, Any]:
    n = len(ws.image_names())
    step = make_step(ws, cfg, "scripted", verbose)
    candidates, best, decisions = candidates_and_best(ws, cfg, step, dataset)
    # Precision: share intrinsics among images of one device and zoom level, upgrade the camera model, bundle adjust.
    # Kept unless it introduces inconsistent pairs or loses images.
    opt = step("optimize", model_id=best, camera_model="OPENCV", share_intrinsics="cluster")
    if not opt.error:
        e_opt = step("examine", model_id=opt.artifact).summary
        e_best = next(c[1] for c in candidates if c[0] == best) if any(c[0] == best for c in candidates) else {}
        ok = (quality(e_opt, n) >= quality(e_best, n) - 0.01) if e_best else True
        decisions["optimize"] = {"model": opt.artifact, "kept": ok, "quality": quality(e_opt, n), "quality_before": quality(e_best, n) if e_best else None}
        if ok:
            best = opt.artifact
    step("submit", model_id=best)
    ws.log({"source": "scripted", "decisions": decisions, "submitted": best})
    return {"submitted": best, "decisions": decisions}
