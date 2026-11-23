"""Offline Solv3R vs Pi3X test, step 1 (agenticsfm env): export per scene the GT cameras, the final p22 model poses and
the Pi3X groups that the run actually solved (names + cam_from_world + K from _work/**/depth_*.npz).
GT is read here only for offline scoring (benchmark layer).

usage: python scripts/analysis/solv3r_prep.py --run p22_vlm_dev --dataset wriva --out <dir>
       python scripts/analysis/solv3r_prep.py --run p22_eth3d --dataset eth3d --split dev --out <dir>
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff_offline import find_grow, largest  # noqa: E402


def cams(rec):
    out = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        T = im.cam_from_world()
        cam = rec.cameras[im.camera_id]
        out[im.name] = {"R": T.rotation.matrix().tolist(), "t": list(map(float, T.translation)),
                        "f": float(cam.mean_focal_length()), "w": int(cam.width), "h": int(cam.height)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-groups", type=int, default=40)
    a = ap.parse_args()
    from agenticsfm.config import load_config
    from agenticsfm.datasets import load_benchmark
    cfg = load_config()
    bench = {b.name: b for b in load_benchmark(cfg, a.dataset, a.split)}
    done = [json.loads(l)["scene"] for l in open(ROOT / "outputs/runs" / a.run / "results.jsonl")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for sc in done:
        if sc not in bench:
            continue
        b = bench[sc]
        _, _, gdir = find_grow(a.run, sc)
        gt = cams(pycolmap.Reconstruction(str(b.gt_dir)))
        model = cams(largest(gdir))
        groups = []
        files = sorted(glob.glob(f"{gdir}/_work/**/ff_*/depth_*.npz", recursive=True))
        for f in files:
            kind = Path(f).parent.name[3:]
            if not (kind.startswith("ff") or kind.startswith("seed")):
                continue                                   # the regular Pi3X rounds: anchors + batch, and the seed
            z = np.load(f)
            groups.append({"tag": kind, "names": [str(n) for n in z["names"]], "E": z["E"].tolist(),
                           "K": z["K"].tolist(), "transforms": z["transforms"].tolist()})
        rng = np.random.default_rng(0)
        if len(groups) > a.max_groups:
            keep = sorted(rng.choice(len(groups), a.max_groups, replace=False).tolist())
            groups = [groups[i] for i in keep]
        rec = {"scene": sc, "dataset": a.dataset, "images_dir": str(b.scene.images_dir), "gt": gt, "model": model,
               "groups": groups}
        json.dump(rec, open(out / f"{sc}.json", "w"))
        print(f"{sc}: {len(gt)} GT, {len(model)} in model, {len(groups)} Pi3X groups of {sorted(set(len(g['names']) for g in groups))}")


if __name__ == "__main__":
    main()
