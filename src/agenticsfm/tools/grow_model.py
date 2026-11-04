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
    },
    "required": ["db_id"],
}


@register("grow_model", "Build a model with the open incremental solver: feedforward seed (Pi3X) on the densest cluster, "
                        "PnP growth in verified groups, feedforward placement of hard images (two anchor sets, VLM arbitration), bundle adjustment; leftover connected images become further models. Every added image is checked "
                        "against the matches and dropped when they disagree. Returns per-model statistics and the growth log.", PARAMS)
def grow_model(ws: Workspace, cfg: dict[str, Any], db_id: str, feedforward: str = "pi3x", seed_size: int = 12, group_size: int = 8,
               min_support: float = 0.5, hard: bool = True, vlm: str = "decide", multi_model: bool = True) -> ToolResult:
    started = time.time()
    params = {"feedforward": feedforward, "seed_size": int(seed_size), "group_size": int(group_size),
              "min_support": float(min_support), "hard": bool(hard), "vlm": vlm, "multi_model": bool(multi_model), "v": 9}
    id = ws.artifact_id("grow_model", params, {"db": db_id})
    if (r := cached_result(ws, id)) is not None:
        return r
    out = ws.path("model", id)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    gc = GrowConfig(feedforward=feedforward, seed_size=int(seed_size), group_size=int(group_size), min_support=float(min_support), hard=bool(hard),
                    vlm=vlm, multi_model=bool(multi_model))
    solver = Solver(ws, cfg, db_id, gc, out / "_work", gpu=ws.gpus[0] if ws.gpus else None)
    grow = solver.run(out / "0")
    summary = {"backend": "grow", "num_images_in_db": ws.get(db_id).summary.get("num_images"), **model_summary(out), "grow": grow,
               "_doc": "grow: the solver's own account (seed, rounds, joins, rejected images, unplaced images); the full "
                       "per-round log is grow_log.json in the model directory"}
    (out / "meta.json").write_text(json.dumps({"params": params, "db": db_id, "summary": summary}, indent=1, default=str))
    art = Artifact(id, "model", "grow_model", params, {"db": db_id}, str(out.relative_to(ws.store)), summary,
                   {"feedforward_seconds": grow.get("feedforward_seconds", 0.0), "vlm_seconds": grow.get("vlm_seconds", 0.0)})
    return finish(ws, art, started)
