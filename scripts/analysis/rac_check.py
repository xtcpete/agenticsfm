"""Offline check of render-and-compare settings on a finished run (benchmark layer: GT only for scoring).
Every registered camera with GT error < 3 deg is judged at its model pose and at a pose rotated 12 deg about its vertical
axis; old = nearest-pose sources, all depth; new = sources facing the camera, back-face culling, coverage test.
usage: python scripts/analysis/rac_check.py --run p4_vlm_dev --scenes ...
"""
import argparse, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ff_offline import find_grow, largest, gt_errors, align_gt  # noqa: E402
from agenticsfm.solver.render_compare import RenderCompare  # noqa: E402
from agenticsfm.config import load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--scenes", nargs="+", required=True)
ap.add_argument("--max", type=int, default=50); a = ap.parse_args()
cfg = load_config()
for scene in a.scenes:
    idir = next(x for x in load_benchmark(cfg, "wriva", "all") if x.name == scene).scene.images_dir
    d, c, g = find_grow(a.run, scene); rec = largest(g)
    err = align_gt(rec, gt_errors(scene))
    regs = [im for im in rec.images.values() if im.has_pose]
    good = [im for im in regs if err.get(im.name, 99) < 3][:: max(1, len([1 for im in regs if err.get(im.name, 99) < 3]) // a.max)]
    variants = {"old": dict(min_conf=0.1), "new": dict(min_conf=0.1, backface=True, min_cover=0.1)}
    res = {k: {"true": Counter(), "rot12": Counter(), "cover_true": []} for k in variants}
    for k, kw in variants.items():
        rc = RenderCompare(rec, Path(idir), **kw)
        for p in sorted(glob.glob(str(g / "_work/scene0/*/depth_*.npz"))):
            rc.add_depth(p)
        for im in good:
            P = im.cam_from_world(); dvec = im.viewing_direction()
            nbs = rc.neighbours(im.name, k=12)
            if k == "new":
                nbs = [n for n in nbs if float(np.dot(rc.image(n).viewing_direction(), dvec)) > 0]
            nbs = nbs[:4]
            j = rc.judge(im.name, P, nbs); res[k]["true"][j["verdict"]] += 1; res[k]["cover_true"].append(j.get("cover", 0))
            Pw = rc.perturbed(P, 12.0, 0.0)
            j2 = rc.judge(im.name, Pw, nbs); res[k]["rot12"][j2["verdict"]] += 1
    print(f"== {scene[-24:]} ({len(good)} correct cameras)")
    for k in variants:
        print(f"   {k}: correct pose {dict(res[k]['true'])}   rotated 12 deg {dict(res[k]['rot12'])}   cover median {np.median(res[k]['cover_true']):.2f}")
