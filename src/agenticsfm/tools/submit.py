"""submit: the final model of the run (copied to <run>/final)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from agenticsfm.tools import ToolResult, register
from agenticsfm.workspace import Workspace

PARAMS = {"type": "object", "properties": {"model_id": {"type": "string"}}, "required": ["model_id"]}


@register("submit", "Submit a model as the final answer. Ends the episode.", PARAMS)
def submit(ws: Workspace, cfg: dict[str, Any], model_id: str) -> ToolResult:
    started = time.time()
    src = ws.abspath(model_id)
    dst = ws.run_dir / "final"
    shutil.rmtree(dst, ignore_errors=True)
    dropped = redundant_submodels(src)
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if Path(d) == src and n in dropped])
    (ws.run_dir / "submitted.txt").write_text(model_id + "\n")
    return ToolResult(None, {"submitted": model_id, "path": str(dst), "dropped_redundant_submodels": len(dropped),
                             **ws.get(model_id).summary}, cost={"wall": time.time() - started})


def redundant_submodels(model_dir: Path) -> set[str]:
    """Sub-model directories that add no image to the largest sub-model: COLMAP keeps an abandoned initialization
    next to the main model, and a set-based evaluation over all sub-models scores those images twice (the second
    time from the leftover's poses). Dropping them changes nothing else."""
    import pycolmap

    subs = sorted((p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    sets = {}
    for p in subs:
        try:
            sets[p.name] = {im.name for im in pycolmap.Reconstruction(str(p)).images.values() if im.has_pose}
        except Exception:
            continue
    if len(sets) < 2:
        return set()
    largest = max(sets, key=lambda k: len(sets[k]))
    return {k for k, names in sets.items() if k != largest and names <= sets[largest]}
