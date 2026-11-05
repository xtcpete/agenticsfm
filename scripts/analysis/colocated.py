"""Co-located camera groups (GT centres within 2% of the camera spread) vs a run's submitted model (benchmark layer, GT only
for scoring). Per group: how many members are registered, and how far apart the model puts them (after a model->GT
similarity), in units of the GT camera spread. Co-located cameras should land on one point.

usage: python scripts/analysis/colocated.py --runs p3b_vlm_dev incremental_v13 [--scenes ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from ff_offline import gt_errors, largest, ROOT  # noqa: E402
from ff_stability import sim_to_gt  # noqa: E402

SCENES = ["t04_v05_s03_r02_VaryingAltitudes_A08", "t04_v09_s00_r04_VaryingAltitudes_BLH0001", "t09_v04_s00_r10_PTZ_A08",
          "t09_v04_s01_r10_PTZ_A08", "t09_v04_s02_r10_PTZ_A08"]


def groups(gt, frac=0.02):
    names = [im.name for im in gt.images.values()]
    C = np.array([im.projection_center() for im in gt.images.values()])
    spread = float(np.median(np.linalg.norm(C - np.median(C, 0), axis=1)))
    par = list(range(len(names)))
    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if D[i, j] < frac * spread:
                par[f(i)] = f(j)
    out = {}
    for i in range(len(names)):
        out.setdefault(f(i), []).append(names[i])
    return [g for g in out.values() if len(g) > 1], spread


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--scenes", nargs="*", default=SCENES)
    a = ap.parse_args()
    for scene in a.scenes:
        gt = gt_errors(scene); gs, spread = groups(gt)
        G = {im.name: im for im in gt.images.values()}
        print(f"== {scene}: {len(gs)} groups, {sum(map(len, gs))} images")
        for run in a.runs:
            fin = ROOT / "outputs/runs" / run / scene / "final"
            if not fin.is_dir():
                print(f"  {run}: no final"); continue
            rec = largest(fin)
            (s, R, t), _ = sim_to_gt(rec, gt)
            by = {im.name: im for im in rec.images.values() if im.has_pose}
            n_all = n_reg = full = partial = none = 0; spreads = []
            bad = []
            for g in gs:
                r = [n for n in g if n in by]
                n_all += len(g); n_reg += len(r)
                full += len(r) == len(g); none += not r; partial += 0 < len(r) < len(g)
                if len(r) >= 2:
                    C = np.array([s * R @ by[n].projection_center() + t for n in r])
                    sp = float(np.max(np.linalg.norm(C - C.mean(0), axis=1)) / spread)
                    spreads.append(sp)
                    if sp > 0.05:
                        bad.append((len(r), len(g), round(sp, 2)))
            spreads = np.array(spreads)
            print(f"  {run:18s} registered {n_reg}/{n_all} in groups; groups full {full}, partial {partial}, none {none}; "
                  f"model spread of registered members: median {np.median(spreads):.3f}, >0.05 in {int((spreads > 0.05).sum())}/{len(spreads)} groups "
                  f"(GT spread < 0.02 by construction)  worst {sorted(bad, key=lambda x: -x[2])[:5]}")


if __name__ == "__main__":
    main()
