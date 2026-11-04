"""rank_models: every model of this run side by side, with the same GT-free statistics, and a
harness quality score. The score is coverage of the largest sub-model minus penalties for strong
inconsistent pairs and VGGT-corroborated suspects; on the dev split it picks within about one
point of the best candidate. It is a hint for the final choice, not a verdict: a model that has
not been examined (or verified) shows missing statistics.
"""

from __future__ import annotations

import time
from typing import Any

from agenticsfm.tools import ToolResult, register
from agenticsfm.workspace import Workspace

PARAMS = {"type": "object", "properties": {}, "required": []}


def quality(coverage: float, strong_incons: float | None, suspects: int | None, largest: int, num_models: int,
            median_consistency: float | None = None, glomap: bool = False) -> float:
    """Audited on 162 dev models against GT: regret 2.2 points vs the best candidate (the median
    pair consistency term and the GLOMAP penalty are the two additions that helped)."""
    q = coverage - 2.0 * (strong_incons or 0.0) - (suspects or 0) / max(1, largest) + 0.3 * (median_consistency or 0.0)
    return q - (0.10 if glomap else 0.0)  # GLOMAP looks consistent to match statistics yet is less precise against GT


def describe(ws: Workspace, model_id: str) -> str:
    chain = []
    for a in ws.provenance(model_id):
        if a.kind == "model":
            chain.append(a.params["backend"] if a.tool == "reconstruct" else (f"optimize({a.params.get('share_intrinsics')})" if a.tool == "optimize" else a.tool))
        elif a.tool == "filter_edges":
            t = a.params.get("threshold")
            chain.append(f"filter {t}" if t is not None else "filter(drop)")
        elif a.kind == "scores":
            chain.append(a.params["method"])
        elif a.kind == "db" and a.tool == "match":
            chain.append(f"{a.params['detector']}+{a.params['matcher']}")
    return " <- ".join(chain)


@register("rank_models", "List every model built in this run with the same statistics (coverage, sub-models, inconsistent "
                         "pairs, VGGT suspects, reprojection error) and a quality score. Use it before submit.", PARAMS)
def rank_models(ws: Workspace, cfg: dict[str, Any]) -> ToolResult:
    started = time.time()
    ws.reload_registry()
    n = len(ws.image_names())
    rows = []
    for m in ws.list("model"):
        if m.id not in ws.used:
            # Only models this run built or touched: the scene store is shared between runs and controllers, and
            # ranking another run's models would hand the agent work it never did (found on WRIVA M07, pass 11/12).
            continue
        exams = [a for a in ws.list("examine") if a.tool == "examine" and a.inputs.get("model") == m.id]
        vers = [a for a in ws.list("examine") if a.tool == "verify_poses" and a.inputs.get("model") == m.id and a.params.get("v", 1) >= 2]
        s = m.summary
        largest = s["models"][0]["num_images"] if s.get("models") else 0
        e = exams[-1].summary["edges"]["consistency"] if exams else {}
        v = vers[-1].summary if vers else {}
        glomap = any(a.tool == "reconstruct" and a.params.get("backend") == "glomap" for a in ws.provenance(m.id))
        row = {"model_id": m.id, "pipeline": describe(ws, m.id), "num_models": s.get("num_models"), "largest_model_images": largest,
               "coverage": round(largest / max(1, n), 3), "reproj_px": s["models"][0]["mean_reproj_error_px"] if s.get("models") else None,
               "strong_inconsistent_frac": round(e["strong_inconsistent_frac"], 3) if e else None,
               "median_pair_consistency": round(e["median"], 3) if e else None,
               "vggt_suspects": v.get("num_suspect"), "examined": bool(exams), "verified": bool(vers)}
        row["quality"] = round(quality(row["coverage"], row["strong_inconsistent_frac"], row["vggt_suspects"], largest, row["num_models"] or 1,
                                       row["median_pair_consistency"], glomap), 3)
        rows.append(row)
    rows.sort(key=lambda r: -r["quality"])
    return ToolResult(None, {"num_images": n, "models": rows,
                             "_doc": "quality = coverage - 2*strong_inconsistent_frac - suspects/largest + 0.3*median_pair_consistency "
                                     "- 0.10 if glomap-derived (GLOMAP fits the pairwise constraints the statistics measure, so they cannot see its "
                                     "lower precision; prefer the COLMAP model at equal coverage); higher is better; "
                                     "unexamined models have no inconsistency statistics and may hide folds"},
                      cost={"wall": time.time() - started})
