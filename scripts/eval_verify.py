"""Benchmark-layer check of verify_poses: for every verify_poses artifact in the dev scenes, the
GT pose error of each checked image (median relative-rotation error against its 10 nearest GT
neighbours, in degrees) next to its VGGT disagreement. Are the suspects really misplaced?

    python scripts/eval_verify.py --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import DEFAULT_CONFIG, REPO_ROOT, load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402
from agenticsfm.workspace import Workspace, scene_id  # noqa: E402


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def gt_errors(gt, model_dir: Path) -> dict[str, float]:
    """Per image in the largest sub-model: median relative-rotation error vs GT over its 10 nearest GT cameras."""
    import pycolmap

    subs = sorted((p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()))
    recs = [pycolmap.Reconstruction(str(p)) for p in subs]
    errs: dict[str, float] = {}
    gtR = {im.name: im.cam_from_world().rotation.matrix() for im in gt.images.values()}
    gtC = {im.name: im.projection_center() for im in gt.images.values()}
    for rec in recs:
        R = {im.name: im.cam_from_world().rotation.matrix() for im in rec.images.values() if im.has_pose and im.name in gtR}
        names = list(R)
        for a in names:
            near = sorted((n for n in names if n != a), key=lambda n: np.linalg.norm(gtC[n] - gtC[a]))[:10]
            e = [rot_angle((R[b] @ R[a].T) @ (gtR[b] @ gtR[a].T).T) for b in near]
            if e:
                errs[a] = float(np.median(e))
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset", default="wriva")
    args = parser.parse_args()
    import pycolmap

    cfg = load_config(args.config)
    all_rows = []
    for b in load_benchmark(cfg, args.dataset, args.split):
        store = REPO_ROOT / cfg["paths"]["cache"] / scene_id(b.scene)
        if not (store / "artifacts.json").is_file():
            continue
        ws = Workspace(b.scene, store / "_analysis", REPO_ROOT / cfg["paths"]["cache"])
        gt = pycolmap.Reconstruction(str(b.gt_dir))
        cache = {}
        for v in ws.list("examine"):
            if v.tool != "verify_poses":
                continue
            m = v.inputs["model"]
            if m not in cache:
                cache[m] = gt_errors(gt, ws.abspath(m))
            per = json.loads((ws.abspath(v.id) / "per_image.json").read_text())
            flagged = {x["image"] for x in v.summary.get("suspect_images", [])}
            for p in per:
                if p["median_rot_err_deg"] is None or p["image"] not in cache[m]:
                    continue
                all_rows.append((b.name, m, p["image"], p["median_rot_err_deg"], cache[m][p["image"]], p["image"] in flagged,
                                 v.params.get("v", 1), p.get("median_trans_dir_err_deg"), p.get("match_support")))
    rows = np.array([(r[3], r[4]) for r in all_rows])
    print(f"{len(all_rows)} image checks in {len(set((r[0], r[1]) for r in all_rows))} models")
    for thr in (10, 15, 20, 30, 45):
        flagged = rows[:, 0] > thr
        bad = rows[:, 1] > 10
        tp = (flagged & bad).sum(); fp = (flagged & ~bad).sum(); fn = (~flagged & bad).sum()
        print(f"VGGT disagreement > {thr:2d} deg: flagged {flagged.sum():4d}, truly misplaced (GT err > 10 deg) among them {tp:3d}, "
              f"false alarms {fp:3d}, missed misplaced {fn:3d}  precision {tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f}")
    # translation-direction disagreement and match support as misplacement signals (v3 artifacts)
    v3 = [r for r in all_rows if r[7] is not None]
    if v3:
        t = np.array([r[7] for r in v3]); sup = np.array([np.nan if r[8] is None else r[8] for r in v3])
        rot = np.array([r[3] for r in v3]); bad = np.array([r[4] > 10 for r in v3])
        print(f"\n{len(v3)} v3 checks: translation-direction disagreement / match support as signals (misplaced = GT err > 10 deg)")
        for label, f in (("trans_dir > 30", t > 30), ("trans_dir > 60", t > 60), ("trans_dir > 90", t > 90),
                         ("support < 0.1", sup < 0.1), ("support < 0.3", sup < 0.3), ("support < 0.5", sup < 0.5),
                         ("rot > 15", rot > 15), ("rot > 15 & support < 0.5", (rot > 15) & (sup < 0.5)),
                         ("trans_dir > 60 & support < 0.1", (t > 60) & (sup < 0.1)),
                         ("rot > 10 & trans_dir > 60", (rot > 10) & (t > 60))):
            tp, fp, fn = (f & bad).sum(), (f & ~bad).sum(), (~f & bad).sum()
            print(f"   {label:32s} flagged {f.sum():4d}  precision {tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f}")
        print("   per scene, trans_dir > 60 false alarms (images GT says are fine):")
        from collections import Counter
        c = Counter(r[0] for r, fl, b_ in zip(v3, t > 60, bad) if fl and not b_)
        c2 = Counter(r[0] for r, fl, b_ in zip(v3, t > 60, bad) if fl and b_)
        for sc in sorted(set(c) | set(c2)):
            print(f"      {sc[-28:]:28s} false {c[sc]:3d}  true {c2[sc]:3d}")
    v2 = [r for r in all_rows if r[6] >= 2]
    if v2:
        f = np.array([r[5] for r in v2]); bad = np.array([r[4] > 10 for r in v2])
        tp, fp, fn = (f & bad).sum(), (f & ~bad).sum(), (~f & bad).sum()
        print(f"\nverify_poses v2 suspects (VGGT > 15 deg AND matches inconsistent), {len(v2)} checks: flagged {f.sum()}, "
              f"truly misplaced {tp}, false alarms {fp}, missed {fn}  precision {tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f}")
        for r in sorted(v2, key=lambda r: -r[3]):
            if r[5] or r[3] > 15:
                print(f"  {r[0][-22:]:22s} {r[1]} {r[2]:18s} vggt {r[3]:6.1f}  gt {r[4]:6.1f}  {'SUSPECT' if r[5] else 'kept'}")


if __name__ == "__main__":
    main()
