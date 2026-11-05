"""Benchmark-layer audit of a Pi3X-loop run: for each image, the path that accepted it (the ff row's reason, or the
hard / sweep step) and its GT rotation error in the submitted model; summary per path.

usage: python scripts/analysis/ff_paths.py --run p1_smoke_vlm [--scenes ...] [--show-bad 5]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ff_offline import find_grow, largest, gt_errors, align_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--scenes", nargs="*"); ap.add_argument("--show-bad", type=float, default=10.0)
    a = ap.parse_args()
    run_dir = ROOT / "outputs/runs" / a.run
    for d in sorted(run_dir.iterdir()):
        if not (d / "final").is_dir() or (a.scenes and not any(s in d.name for s in a.scenes)):
            continue
        rd, store, mdir = find_grow(a.run, d.name)
        log = json.loads((mdir / "grow_log.json").read_text())["log"]
        final = largest(rd / "final"); err = align_gt(final, gt_errors(d.name))
        path = {}
        for r in log:
            k = r.get("kind")
            if k == "seed" and not r.get("model"):
                for n in r["attempts"][r["chosen"]]["kept"]:
                    path.setdefault(n, "seed")
            elif k == "ff" and not r.get("model"):
                for n in r.get("registered", []):
                    path.setdefault(n, ("sweep: " if r.get("sweep") else "") + r["images"][n]["why"])
            elif k == "hard" and not r.get("model"):
                for n in r.get("registered", []):
                    path.setdefault(n, "hard: " + str((r.get("images") or {}).get(n, {}).get("why")))
            elif k == "component":
                pass
        by = defaultdict(list)
        for n, e in err.items():
            by[path.get(n, "other (second model merged / repair)")].append((e, n))
        print(f"== {d.name}  ({len(err)} images in the submitted model)")
        for p_, v in sorted(by.items(), key=lambda x: -len(x[1])):
            e = np.array([x[0] for x in v])
            bad = sorted([x for x in v if x[0] > a.show_bad], reverse=True)
            print(f"   {len(v):4d}  med {np.median(e):5.2f}°  >5° {int((e > 5).sum()):3d}  >10° {int((e > 10).sum()):3d}  {p_}" + (f"   bad: {', '.join(f'{n[-8:-4]}={x:.0f}°' for x, n in bad[:6])}" if bad else ""))


if __name__ == "__main__":
    main()
