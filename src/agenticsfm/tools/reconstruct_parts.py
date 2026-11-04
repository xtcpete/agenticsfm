"""reconstruct_parts: map every part of a partition in parallel and merge the results into one
model. The scalable mapping: thousands of images become a few dozen local mappings of a few hundred
images each (COLMAP or GLOMAP on the database restricted to the part), run concurrently, then
merged and refined on the full database (triangulation + bundle adjustment). The merge is greedy
over the overlap graph: every sub-model of every part is a unit, the largest unit anchors, and at
each step the unit sharing the most registered images with the growing model is attached, so the
model grows outward through the overlaps instead of in a fixed order (a fixed order fails as soon
as the next part is not adjacent to what has been merged). Units that never reach 3 shared images,
or whose alignment residual exceeds ``max_residual``, are reported, not merged.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agenticsfm.tools import ToolResult, cached_result, call, finish, register
from agenticsfm.workspace import Artifact, Workspace

MIN_UNIT = 5  # registered images a sub-model needs to count as a unit to merge

PARAMS = {
    "type": "object",
    "properties": {
        "db_id": {"type": "string"},
        "parts_id": {"type": "string", "description": "from partition_graph"},
        "backend": {"type": "string", "enum": ["colmap", "glomap"], "default": "colmap"},
        "workers": {"type": "integer", "default": 4, "description": "parts mapped concurrently"},
        "max_residual": {"type": "number", "default": 0.0, "description": "do not attach a unit whose alignment residual on the "
                                                                         "shared images exceeds this fraction of the model span (0: no limit)"},
    },
    "required": ["db_id", "parts_id"],
}


@register("reconstruct_parts", "Map every part of a partition (reconstruct with images=part) concurrently, then merge the "
                               "parts into one model through their shared images and refine on the full database. The "
                               "scalable alternative to one global mapping for scenes of many hundreds or thousands of images.", PARAMS)
def reconstruct_parts(ws: Workspace, cfg: dict[str, Any], db_id: str, parts_id: str, backend: str = "colmap", workers: int = 4,
                      max_residual: float = 0.0) -> ToolResult:
    started = time.time()
    params = {"backend": backend, "max_residual": float(max_residual), "v": 3}  # v bumps invalidate results computed with earlier, buggier code
    id = ws.artifact_id("reconstruct_parts", params, {"db": db_id, "parts": parts_id})
    if (r := cached_result(ws, id)) is not None:
        return r
    parts = json.loads(ws.abspath(parts_id).read_text())

    def map_part(part):
        return call(ws, cfg, "reconstruct", {"db_id": db_id, "backend": backend, "images": part, "num_threads": max(2, 16 // max(1, workers))},
                    source="reconstruct_parts")

    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        results = list(pool.map(map_part, parts))
    models = []
    for k, (part, r) in enumerate(zip(parts, results)):
        if r.error or not r.artifact:
            models.append({"part": k, "size": len(part), "error": r.summary.get("error")})
            continue
        models.append({"part": k, "size": len(part), "model": r.artifact, "registered": r.summary.get("registered_images"),
                       "largest": r.summary["models"][0]["num_images"] if r.summary.get("models") else 0, "num_models": r.summary.get("num_models")})
    ok = sorted([m for m in models if "model" in m], key=lambda m: -m["largest"])
    if not ok:
        raise RuntimeError("no part could be mapped")
    from agenticsfm.tools.frame import cameras_of
    from agenticsfm.tools.merge_models import submodels

    units = []  # every sub-model of every part: (part, sub-model index, registered image names)
    for m in ok:
        for k, rec in enumerate(submodels(ws, m["model"])):
            names = set(cameras_of(rec))
            if len(names) >= MIN_UNIT:
                units.append({"part": m["part"], "submodel": k, "model": m["model"], "names": names})
    units.sort(key=lambda u: -len(u["names"]))
    current, have = units[0]["model"], set(units[0]["names"])
    pending, merged, failed, redundant = units[1:], [], [], 0
    while pending:
        best = max(pending, key=lambda u: len(u["names"] & have))
        pending.remove(best)
        shared, new = len(best["names"] & have), len(best["names"] - have)
        if new == 0:
            redundant += 1
            continue
        if shared < 3:  # nothing left is adjacent to the model: the rest cannot be attached
            failed.append({"part": best["part"], "submodel": best["submodel"], "images": len(best["names"]), "shared": shared,
                           "error": "fewer than 3 registered images shared with the merged model"})
            failed.extend({"part": u["part"], "submodel": u["submodel"], "images": len(u["names"]), "shared": len(u["names"] & have),
                           "error": "fewer than 3 registered images shared with the merged model"} for u in pending if len(u["names"] - have))
            break
        r = call(ws, cfg, "merge_models", {"model_a": current, "model_b": best["model"], "submodel_b": best["submodel"], "refine": False},
                 source="reconstruct_parts")
        if r.error:
            failed.append({"part": best["part"], "submodel": best["submodel"], "images": len(best["names"]), "shared": shared, "error": r.summary.get("error")})
            continue
        st = r.summary.get("merge", {})
        resid = st.get("alignment_residual_frac_of_span")
        if max_residual and resid is not None and resid > max_residual:
            failed.append({"part": best["part"], "submodel": best["submodel"], "images": len(best["names"]), "shared": shared,
                           "error": f"alignment residual {resid} of the model span (limit {max_residual})"})
            continue
        current = r.artifact
        have |= best["names"]
        merged.append({"part": best["part"], "submodel": best["submodel"], "shared": shared, "added": new, "residual": resid,
                       "scale": st.get("scale_b_to_a"), "registered": len(have)})
    # one refinement (triangulation + bundle adjustment on the full database) of the merged poses
    final = call(ws, cfg, "merge_models", {"model_a": current, "model_b": units[0]["model"]}, source="reconstruct_parts") if merged else None
    final_id = final.artifact if final is not None and not final.error else current
    note = {"merge_units": len(units), "merge_steps": merged, "merge_failures": failed, "redundant_units": redundant,
            "_doc": "merge_steps: the units in the order they were attached (shared = registered images in common with the model "
                    "at that moment; residual = alignment error on them as a fraction of the model span, above ~0.05 is a disagreement); "
                    "merge_failures: units that stayed detached and why"}
    out = ws.path("model", id)
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ws.abspath(final_id), out)
    from agenticsfm.tools.reconstruct import model_summary

    summary = {"backend": f"parts+{backend}", "num_images_in_db": ws.get(db_id).summary.get("num_images"), **model_summary(out),
               "parts": models, **note, "merged_from": final_id}
    (out / "meta.json").write_text(json.dumps({"params": params, "db": db_id, "parts": parts_id, "summary": summary}, indent=1, default=str))
    art = Artifact(id, "model", "reconstruct_parts", params, {"db": db_id, "parts": parts_id}, str(out.relative_to(ws.store)), summary)
    return finish(ws, art, started)
