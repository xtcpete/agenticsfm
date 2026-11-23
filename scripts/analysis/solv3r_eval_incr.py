"""Score solv3r_incr.py outputs against GT (agenticsfm env). Per scene, AUC@3/10/30 over pairs:
  all pairs for joint (from solv3r_run.py output joint32), anyorder, anyorder_polish;
  query vs map pairs for query (full cache) and query_nb (16 nearest map views), with joint32 on the same pairs.
usage: python scripts/analysis/solv3r_eval_incr.py --prep <dir> --run <solv3r_run out> --incr <dir>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solv3r_eval import auc, gt_arrays, pair_err, to_cfw  # noqa: E402


def errs(names, c2w, gt, idx):
    R, t = to_cfw(np.asarray(c2w))
    Rg, tg = gt_arrays(gt, names)
    return pair_err(R, t, Rg, tg, idx)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--run", required=True); ap.add_argument("--incr", required=True)
    a = ap.parse_args()
    pooled = {}
    for f in sorted(Path(a.incr).glob("*.json")):
        r = json.load(open(f)); gt = json.load(open(Path(a.prep) / f.name))["gt"]
        j = json.load(open(Path(a.run) / f.name))
        names = r["names"]; n = len(names)
        allp = np.array([(x, y) for x in range(n) for y in range(x + 1, n)])
        jn = j["joint32"]["names"]; assert jn == names
        row = {"joint32": errs(names, j["joint32"]["rays"]["c2w"], gt, allp),
               "anyorder": errs(names, r["anyorder"], gt, allp), "anyorder_polish": errs(names, r["anyorder_polish"], gt, allp)}
        mset = set(r["query"]["map"]); mi = [k for k in range(n) if names[k] in mset]; qi = [k for k in range(n) if names[k] not in mset]
        qp = np.array([(q, m) for q in qi for m in mi])
        row["joint32@qm"] = errs(names, j["joint32"]["rays"]["c2w"], gt, qp)
        row["query"] = errs(names, r["query"]["c2w"], gt, qp)
        row["query_nb"] = errs(names, r["query"]["c2w_nb"], gt, qp)
        print(f"{r['scene'][:40]:40s} n={n:3d} " + " | ".join(f"{k} {auc(e,3):5.1f} {auc(e,10):5.1f}" for k, e in row.items()))
        for k, e in row.items():
            pooled.setdefault(("wriva" if r["scene"].startswith("t0") else "eth3d", k), []).append((auc(e, 3), auc(e, 10), auc(e, 30)))
    for (ds, k), v in sorted(pooled.items()):
        m = np.mean(v, 0); print(f"{ds:6s} {k:16s} mean per-scene AUC {m[0]:5.1f} {m[1]:5.1f} {m[2]:5.1f} ({len(v)})")


if __name__ == "__main__":
    main()
