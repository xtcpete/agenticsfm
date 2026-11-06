"""grow_model: the open incremental solver as a tool. From a database (verified matches) it builds one model by
seeding with a feedforward model on the densest cluster, growing by PnP groups with verification of every added
image, feedforward joins where PnP cannot reach, and bundle adjustment. Returns a model artifact like
``reconstruct``; ``examine`` / ``verify_poses`` / ``optimize`` / ``submit`` apply to it.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from dataclasses import replace
from agenticsfm.solver.grow import GrowConfig, Solver
from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.reconstruct import model_summary
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string", "description": "database with verified matches"},
        "feedforward": {"type": "string", "enum": ["pi3x", "vggt", "vggt_omega"], "default": "pi3x",
                        "description": "feedforward model for the seed and the joins"},
        "seed_size": {"type": "integer", "default": 12, "description": "images in the feedforward seed cluster (8 to 30)"},
        "group_size": {"type": "integer", "default": 8, "description": "PnP registrations per round before verification"},
        "min_support": {"type": "number", "default": 0.5,
                        "description": "an added image stays if at least this share of its strong pairs' matches agree with the poses"},
        "hard": {"type": "boolean", "default": True, "description": "place hard images (lookalike matches) with the feedforward model and two anchor sets"},
        "vlm": {"type": "string", "enum": ["off", "log", "decide"], "default": "decide",
                "description": "the VLM advisor: decide = arbitrate disagreeing placements and judge whether extra models are the same scene; log = ask but never act"},
        "multi_model": {"type": "boolean", "default": True, "description": "reconstruct leftover connected image sets as further models (separate scenes)"},
        "support_pairs": {"type": "integer", "default": 1000, "description": "match support from each image's strongest pairs only"},
        "support_norm": {"type": "boolean", "default": True, "description": "pair consistency relative to the pair's own best two-view fit"},
        "loop": {"type": "string", "enum": ["ff", "pnp"], "default": "ff", "description": "ff: feedforward placement every round, checked by PnP; pnp: PnP rounds"},
        "select": {"type": "string", "enum": ["graph", "vlm"], "default": "graph", "description": "who picks each round's images"},
        "ff_off": {"type": "string", "default": "", "description": "ff-loop features to switch off (ablations), comma separated: "
                   "evidence (inliers in the VLM selection prompt), fill (fill VLM picks up to the round size), points (point-based alignment), "
                   "settle (match support settles disagreements), sweep (final retry of unregistered images), anchors (matched anchors only), rotation, vlmplace, sky, coloc, rsrc, backface, cover, rgate (Phase 3b stability gate for VLM placements instead of the 3c render veto), ainl (no alignment-inlier requirement for 3c), retry (no re-placement with other anchors when the alignment is weak), extra (images of unmerged second models stay out of vlm_place), trusted (vlm_place references from all registered images, hard-stage ones included), vlmtemp (VLM at temperature 0.3 unseeded, as before cache v29)"},
        "scores_id": {"type": "string", "description": "FlexGlue covisibility scores of the database: the images are split into scenes first, "
                                                       "so feedforward calls never mix scenes; each scene gets its own solver run"},
    },
    "required": ["db_id"],
}


@register("grow_model", "Build a model with the open incremental solver: feedforward seed (Pi3X) on the densest cluster, "
                        "PnP growth in verified groups, feedforward placement of hard images (two anchor sets, VLM arbitration), bundle adjustment; leftover connected images become further models. Every added image is checked "
                        "against the matches and dropped when they disagree. Returns per-model statistics and the growth log.", PARAMS)
def grow_model(ws: Workspace, cfg: dict[str, Any], db_id: str, feedforward: str = "pi3x", seed_size: int = 12, group_size: int = 8,
               min_support: float = 0.5, hard: bool = True, vlm: str = "decide", multi_model: bool = True, scores_id: str | None = None,
               support_pairs: int = 1000, support_norm: bool = True, loop: str = "ff", select: str = "graph", ff_off: str = "") -> ToolResult:
    from agenticsfm.solver.scenes import split_scenes
    from agenticsfm.tools.score_pairs import load_scores

    started = time.time()
    params = {"feedforward": feedforward, "seed_size": int(seed_size), "group_size": int(group_size),
              "min_support": float(min_support), "hard": bool(hard), "vlm": vlm, "multi_model": bool(multi_model), "v": 29,
              "support_pairs": int(support_pairs), "support_norm": bool(support_norm), "loop": loop, "select": select,
              "ff_off": ",".join(sorted(x for x in ff_off.split(",") if x.strip()))}
    inputs = {"db": db_id, **({"scores": scores_id} if scores_id else {})}
    id = ws.artifact_id("grow_model", params, inputs)
    if (r := cached_result(ws, id)) is not None:
        return r
    out = ws.path("model", id)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    gc = GrowConfig(feedforward=feedforward, seed_size=int(seed_size), group_size=int(group_size), min_support=float(min_support), hard=bool(hard),
                    vlm=vlm, multi_model=bool(multi_model), support_pairs=int(support_pairs),
                    support_norm=bool(support_norm), loop=loop, select=select)
    off = {x.strip() for x in ff_off.split(",") if x.strip()}
    unknown = off - {"evidence", "fill", "points", "settle", "sweep", "anchors", "rotation", "vlmplace", "sky", "coloc", "rsrc", "backface", "cover", "rgate", "ainl", "retry", "extra", "trusted", "vlmtemp"}
    if unknown:
        raise ValueError(f"unknown ff_off features: {sorted(unknown)}")
    gc = replace(gc, select_evidence="evidence" not in off, select_fill="fill" not in off, ff_align_points="points" not in off,
                 ff_settle_support="settle" not in off, ff_sweep="sweep" not in off, ff_rotation="rotation" not in off,
                 ff_vlm_place="vlmplace" not in off, rac_min_conf=0.0 if "sky" in off else gc.rac_min_conf, ff_coloc="coloc" not in off, rac_sources_match="rsrc" not in off, rac_backface="backface" not in off,
                 rac_min_cover=0.0 if "cover" in off else gc.rac_min_cover,
                 vlm_place_render="rgate" not in off, vlm_place_stable="rgate" in off, vlm_place_min_inl=0.0 if "ainl" in off else gc.vlm_place_min_inl,
                 vlm_place_retry="retry" not in off, vlm_place_extra="extra" not in off, vlm_place_trusted="trusted" not in off,
                 vlm_temperature=0.3 if "vlmtemp" in off else 0.0, vlm_seed="vlmtemp" not in off,
                 **({"ff_anchors_match": 6, "ff_anchors_diverse": 4} if "anchors" in off else {}))
    gpu = ws.gpus[0] if ws.gpus else None
    split = None
    scenes = [None]
    advisor = None
    if vlm != "off":
        from agenticsfm.solver.advisor import Advisor
        advisor = Advisor(cfg, ws.images_dir, [], mode=vlm, temperature=gc.vlm_temperature, seed=gc.vlm_seed)
    if scores_id:
        split = split_scenes(ws.abspath(db_id), load_scores(ws, scores_id), advisor=advisor)
        scenes = [s for s in split["scenes"] if len(s) >= 3] or [None]
    runs, logs, k_model = [], [], 0
    for si, scene in enumerate(scenes):
        sub = out / f"scene{si}"
        solver = Solver(ws, cfg, db_id, gc, out / "_work" / f"scene{si}", gpu=gpu, image_subset=scene, advisor=advisor)
        try:
            grow = solver.run(sub / "0")
        except RuntimeError as e:          # a scene whose seed does not hold is left out, the others are returned
            logs.append({"kind": "scene", "scene": si, "images": len(scene or []), "error": str(e)[:300]})
            continue
        for d in sorted((x for x in sub.iterdir() if x.is_dir() and x.name.isdigit()), key=lambda x: int(x.name)):
            d.rename(out / str(k_model))
            k_model += 1
        sl = json.loads((sub / "grow_log.json").read_text())
        logs.extend({**r, "scene": si} for r in sl["log"])
        runs.append({"scene": si, "images": len(scene) if scene else grow.get("num_images"), **grow})
    if not runs:
        raise RuntimeError(f"no scene could be seeded: {logs[-1] if logs else ''}")
    grow = dict(runs[0])
    if len(runs) > 1:
        grow["scenes"] = [{k: r.get(k) for k in ("scene", "images", "num_registered", "num_models", "wall")} for r in runs]
        grow["num_models"] = k_model
    if split:
        grow["split"] = {"scenes": [len(s) for s in split["scenes"]], "unassigned": len(split["unassigned"]),
                         "cuts": [d for d in split["decisions"] if d.get("step") == "cut"]}
        logs.insert(0, {"round": 0, "kind": "split", **grow["split"]})
    for key in ("feedforward_seconds", "vlm_seconds", "wall"):
        grow[key] = round(sum(r.get(key, 0.0) or 0.0 for r in runs), 1)
    (out / "grow_log.json").write_text(json.dumps({"summary": grow, "log": logs}, indent=1, default=str))
    summary = {"backend": "grow", "num_images_in_db": ws.get(db_id).summary.get("num_images"), **model_summary(out), "grow": grow,
               "_doc": "grow: the solver's own account (scene split, seed, rounds, hard placements, VLM verdicts, unplaced images); the full "
                       "per-round log is grow_log.json in the model directory"}
    (out / "meta.json").write_text(json.dumps({"params": params, "inputs": inputs, "summary": summary}, indent=1, default=str))
    art = Artifact(id, "model", "grow_model", params, inputs, str(out.relative_to(ws.store)), summary,
                   {"feedforward_seconds": grow.get("feedforward_seconds", 0.0), "vlm_seconds": grow.get("vlm_seconds", 0.0)})
    return finish(ws, art, started)
