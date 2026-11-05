"""Can Pi3X tell co-located cameras? For every saved Pi3X group of a run (depth_*.npz: E per image), pairs whose Pi3X centres
lie within tau x (median camera distance from the group's centre) are called co-located; compared with GT co-location
(centres within 2% of the GT camera spread). Benchmark layer: GT only for scoring.
usage: python scripts/analysis/coloc_pi3x.py --run p3b_vlm_dev
"""
import argparse, glob
import numpy as np
from colocated import groups, SCENES
from ff_offline import gt_errors, find_grow

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
taus = [0.01, 0.02, 0.03, 0.05, 0.08]
tot = {t: [0, 0] for t in taus}; gt_pairs_seen_tot = 0
for scene in SCENES:
    gt = gt_errors(scene); gs, spread = groups(gt)
    lab = {n: k for k, g in enumerate(gs) for n in g}
    d, c, g = find_grow(a.run, scene)
    seen = {}
    for p in glob.glob(str(g / "_work/scene*/*/depth_*.npz")):
        z = np.load(p); names = [str(n) for n in z["names"]]; E = z["E"].astype(np.float64)
        C = np.array([-e[:, :3].T @ e[:, 3] for e in E]); sc = np.median(np.linalg.norm(C - np.median(C, 0), axis=1))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = tuple(sorted((names[i], names[j])))
                r = np.linalg.norm(C[i] - C[j]) / max(sc, 1e-9)
                seen.setdefault(key, []).append(r)
    per = {}
    gtp = sum(1 for k in seen if lab.get(k[0], -1) == lab.get(k[1], -2))
    gt_pairs_seen_tot += gtp
    row = []
    for t in taus:
        called = [k for k, v in seen.items() if np.median(v) < t]
        tp = sum(1 for k in called if lab.get(k[0], -1) == lab.get(k[1], -2))
        tot[t][0] += tp; tot[t][1] += len(called)
        row.append(f"tau {t}: called {len(called)}, precision {tp / max(1, len(called)):.2f}, recall {tp / max(1, gtp):.2f}")
    print(f"== {scene[:30]}: pairs seen together {len(seen)}, GT co-located among them {gtp}\n   " + "\n   ".join(row))
print("ALL:", [(t, f"P {tot[t][0] / max(1, tot[t][1]):.2f}", f"R {tot[t][0] / max(1, gt_pairs_seen_tot):.2f}") for t in taus])
