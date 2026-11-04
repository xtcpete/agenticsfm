"""analyze: run a short Python snippet over the workspace and return what it prints.

Pictures are vague; numbers are not. The snippet runs in the harness environment with these names
bound: ``ws`` (the workspace), ``model(id)`` -> list of pycolmap.Reconstruction (largest first),
``db(id)`` -> path of a database, ``pairs(db_id)`` -> {(name0, name1): inliers} of its verified
pairs, ``scores(id)`` -> {(name0, name1): score}, ``edges(model_id)`` -> rows of examine's
per-pair test (pair, inliers, kind, consistent_frac), ``images(model_id)`` -> examine's per-image
table, ``verify(model_id)`` -> verify_poses' per-image rows, ``np``, ``json``, ``pycolmap``.
Only the workspace is visible: there is no ground truth anywhere in it. Output is truncated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agenticsfm.backends import SRC_DIR
from agenticsfm.tools import ToolResult, register
from agenticsfm.workspace import Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": (
            "Python; print what you want to see. Bound names (no others; do not introspect): "
            "model(id) -> [pycolmap.Reconstruction, ...] largest first (rec.images.values(): .name, .has_pose, .cam_from_world(), "
            ".num_points3D; rec.cameras; rec.points3D); db(id) -> Path; pairs(db_id) -> {(name0, name1): inliers}; "
            "scores(scores_id) -> {(name0, name1): score}; edges(model_id) -> [{pair, inliers, kind, consistent_frac}] from examine; "
            "images(model_id) -> [{image, points3D, points3D_rel, match_support, weakly_constrained}] from examine; "
            "verify(model_id) -> [{image, neighbours, rot_err_deg, median_rot_err_deg, match_support}] from verify_poses; "
            "np, json, pycolmap. Example: P = pairs('<db_id>'); S = scores('<scores_id>'); "
            "for h in ['image_000013.JPG']: print(h, sorted([(n, b if a == h else a, S.get((a, b))) for (a, b), n in P.items() "
            "if h in (a, b)], reverse=True)[:5])")},
        "timeout": {"type": "integer", "default": 120},
    },
    "required": ["code"],
}

PRELUDE = '''
import json, sys
import numpy as np
import pycolmap
from pathlib import Path
sys.path.insert(0, {src!r})
from agenticsfm.config import load_config
from agenticsfm.datasets import Scene
from agenticsfm.workspace import Workspace
pycolmap.logging.minloglevel = 2
_scene = Scene({name!r}, Path({images_dir!r}))
ws = Workspace(_scene, {run_dir!r}, {cache_root!r})

def model(model_id):
    d = ws.abspath(model_id)
    recs = [pycolmap.Reconstruction(str(p)) for p in sorted(d.iterdir()) if p.is_dir() and p.name.isdigit()]
    return sorted(recs, key=lambda r: -r.num_reg_images())

def db(db_id):
    return ws.abspath(db_id)

def pairs(db_id):
    out = {{}}
    with pycolmap.Database.open(ws.abspath(db_id)) as d:
        names = {{im.image_id: im.name for im in d.read_all_images()}}
        ids, geoms = d.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            n = int(g.inlier_matches.shape[0])
            if n:
                i1, i2 = divmod(int(pid), 2147483647)
                out[(names[i1], names[i2])] = n
    return out

def scores(scores_id):
    z = np.load(ws.abspath(scores_id))
    return {{(str(a), str(b)): float(s) for a, b, s in zip(z["name0"], z["name1"], z["score"])}}

def _latest(tool, model_id):
    arts = [a for a in ws.list("examine") if a.tool == tool and a.inputs.get("model") == model_id]
    if not arts:
        raise KeyError(f"run {{tool}} on {{model_id}} first")
    return ws.abspath(arts[-1].id)

def edges(model_id):
    return [json.loads(l) for l in (_latest("examine", model_id) / "edges.jsonl").read_text().splitlines() if l]

def images(model_id):
    return json.loads((_latest("examine", model_id) / "images.json").read_text())

def verify(model_id):
    return json.loads((_latest("verify_poses", model_id) / "per_image.json").read_text())

models, databases, pair_scores, examine_edges, examine_images, verify_images = model, db, scores, edges, images, verify  # aliases
# ---- user code ----
'''

MAX_OUTPUT = 6000


@register("analyze", "Run a short Python snippet over the workspace and get its printed output. The bound functions are listed "
                     "in the `code` parameter description with an example; write the analysis directly, do not spend calls "
                     "exploring the namespace. Use it for exact numbers: which images share strong edges, score distributions "
                     "per image, pose differences, anything a picture leaves vague.", PARAMS)
def analyze(ws: Workspace, cfg: dict[str, Any], code: str, timeout: int = 120) -> ToolResult:
    started = time.time()
    script = PRELUDE.format(src=str(SRC_DIR), name=ws.scene.name, images_dir=str(ws.scene.images_dir), run_dir=str(ws.run_dir),
                            cache_root=str(ws.store.parent)) + code
    out_dir = ws.run_dir / "analyze"
    out_dir.mkdir(exist_ok=True)
    k = len(list(out_dir.iterdir()))
    path = out_dir / f"{k:03d}.py"
    path.write_text(script)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=int(timeout), cwd=str(out_dir), env=env)
        output = proc.stdout
        err = "\n".join(l for l in proc.stderr.splitlines() if not l.startswith(("I2026", "W2026", "E2026")))
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        output, err, status = "", "", f"timeout after {timeout}s"
    (out_dir / f"{k:03d}.out").write_text(output + ("\n--- stderr ---\n" + err if err else ""))
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + f"\n... [truncated, {len(output)} chars]"
    if len(err) > 2000:
        err = err[-2000:]
    return ToolResult(None, {"status": status, "stdout": output, "stderr": err}, cost={"wall": time.time() - started})
