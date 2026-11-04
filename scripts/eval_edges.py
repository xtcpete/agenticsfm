"""Benchmark-layer edge oracle: label every verified pair of a scene's main database by the GT
points the two images share in the GT-triangulated model (config datasets.wriva.gt_triangulated),
then score each pair signal at ranking true edges above wrong ones.

    python scripts/eval_edges.py --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import DEFAULT_CONFIG, REPO_ROOT, load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402
from agenticsfm.tools.relpose import cycle_consistency, relative_poses, rot_angle  # noqa: E402
from agenticsfm.workspace import Workspace, scene_id  # noqa: E402

TRUE_MIN_SHARED = 10
WRONG_MIN_INLIERS = 30


def average_precision(score, label):
    order = np.argsort(-score, kind="stable")
    label = label[order]
    tp = np.cumsum(label)
    prec = tp / (np.arange(len(label)) + 1)
    return float((prec * label).sum() / max(1, label.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()
    import pycolmap

    pycolmap.logging.minloglevel = 2
    cfg = load_config(args.config)
    totals = {}
    for b in load_benchmark(cfg, "wriva", args.split):
        if args.scenes and b.name not in args.scenes:
            continue
        gt = pycolmap.Reconstruction(cfg["datasets"]["wriva"]["gt_triangulated"].format(scene=b.name))
        gtP = {im.name: im.cam_from_world() for im in gt.images.values()}
        seen = {im.name: {p.point3D_id for p in im.points2D if p.has_point3D()} for im in gt.images.values()}
        store = REPO_ROOT / cfg["paths"]["cache"] / scene_id(b.scene)
        ws = Workspace(b.scene, store / "_analysis", REPO_ROOT / cfg["paths"]["cache"])
        dbs = [a for a in ws.list("db") if a.tool == "match" and a.params.get("matcher") == "flexglue" and a.params.get("camera_mode") == "auto"]
        dba = max(dbs, key=lambda a: a.summary.get("num_verified_pairs", 0))
        rel = relative_poses(ws.abspath(dba.id), ws.path("scores", f"relpose_{dba.id}", ".npz"))
        pairs = list(rel)
        n = np.array([rel[p][2] for p in pairs], float)
        shared = np.array([len(seen[a] & seen[b]) for a, b in pairs])
        # GT relative rotation error of the estimated two-view rotation (how good are the relposes themselves?)
        rot_err = np.array([rot_angle(rel[(a, b)][0] @ (gtP[b] * gtP[a].inverse()).rotation.matrix().T) for a, b in pairs])
        true = shared >= TRUE_MIN_SHARED
        wrong = (shared == 0) & (n >= WRONG_MIN_INLIERS)
        keep = true | wrong
        print(f"\n== {b.name[-30:]}: {len(pairs)} pairs with relpose; true {true.sum()}, wrong {wrong.sum()}, other {(~keep).sum()}")
        print(f"   two-view rotation vs GT: true edges median {np.median(rot_err[true]):.1f} deg (p90 {np.percentile(rot_err[true], 90):.1f}); "
              f"wrong edges median {np.median(rot_err[wrong]):.1f} deg; true edges with rot err > 30: {np.mean(rot_err[true] > 30):.2f}")
        signals = {"inliers": n}
        for sc in ws.list("scores"):
            if sc.inputs.get("db") == dba.id and sc.params["method"] != "cycle_consistency":
                z = np.load(ws.abspath(sc.id))
                lookup = {(str(x), str(y)): float(s) for x, y, s in zip(z["name0"], z["name1"], z["score"])}
                signals[sc.params["method"]] = np.array([lookup.get(p, lookup.get((p[1], p[0]), np.nan)) for p in pairs])
        for thr in (10, 20, 30, 45):
            cyc = cycle_consistency(rel, max_angle=thr)
            signals[f"cycle_frac<{thr}"] = np.array([cyc[p][0] if cyc[p][1] else 0.5 for p in pairs])
        cyc = cycle_consistency(rel, max_angle=10)
        signals["cycle_-median_err"] = np.array([-cyc[p][2] if cyc[p][1] else -45.0 for p in pairs])
        if "flexglue_covis" in signals and "cycle_frac<30" in signals:
            signals["covis*cycle30"] = signals["flexglue_covis"] * signals["cycle_frac<30"]
            signals["covis+cycle30"] = 0.5 * (signals["flexglue_covis"] + signals["cycle_frac<30"])
        if "xdg" in signals and "flexglue_covis" in signals:
            signals["xdg*covis"] = signals["xdg"] * signals["flexglue_covis"]
        print(f"   {'signal':18s} {'AP':>6s} {'AP strong':>9s}  {'at 0.6: wrong kept':>18s} {'true dropped':>12s}")
        for name, s in signals.items():
            ok = keep & ~np.isnan(s)
            strong = ok & (n >= 100)
            ap = average_precision(s[ok], true[ok].astype(float))
            aps = average_precision(s[strong], true[strong].astype(float)) if strong.sum() else float("nan")
            at = f"{int(((s >= 0.6) & wrong).sum()):18d} {int(((s < 0.6) & true).sum()):12d}" if name not in ("inliers", "cycle_-median_err") else ""
            print(f"   {name:18s} {ap:6.3f} {aps:9.3f}  {at}")
            totals.setdefault(name, []).append(ap)
    print("\nmean AP over scenes: " + ", ".join(f"{k} {np.nanmean(v):.3f}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
