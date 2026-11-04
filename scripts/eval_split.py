"""Benchmark-layer evaluation of the scene split on cached matches.

Mixed folders (outputs/data/mixed/<name>, site = filename prefix before the first underscore):
purity (share of images whose scene's majority site is their own site) and completeness (share of
images in the largest scene of their site). Single-site benchmark scenes: how many scenes the split
makes and how many images leave the main scene (each is an over-split).

    python scripts/eval_split.py [--mixed m07_terrace m07_blh ...] [--datasets wriva eth3d]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import load_config  # noqa: E402
from agenticsfm.datasets import folder, load_benchmark  # noqa: E402
from agenticsfm.solver.scenes import SplitConfig, split_scenes  # noqa: E402
from agenticsfm.tools.score_pairs import load_scores  # noqa: E402
from agenticsfm.workspace import make_workspace  # noqa: E402


def db_and_scores(ws):
    dbs = [a for a in ws.list("db") if a.tool == "match" and a.params.get("matcher") == "flexglue"]
    if not dbs:
        return None, None
    db = sorted(dbs, key=lambda a: (a.params.get("detector") != "aliked", a.params.get("resize", 1024) != 1024))[0]
    sc = [a for a in ws.list("scores") if a.tool == "score_pairs" and a.params.get("method") == "flexglue_covis" and a.inputs.get("db") == db.id]
    return db, (sc[0] if sc else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", nargs="*", default=["m07_terrace", "m07_blh", "three_sites", "ptz_a08_a06", "same_site_two_sessions"])
    ap.add_argument("--datasets", nargs="*", default=["wriva", "eth3d"])
    ap.add_argument("--split-conductance", type=float, default=0.01)
    ap.add_argument("--no-covis", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = load_config()
    scfg = SplitConfig(split_conductance=args.split_conductance)
    rows = []
    for name in args.mixed:
        d = Path(f"outputs/data/mixed/{name}/images")
        if not d.is_dir():
            continue
        ws = make_workspace(cfg, "eval_split", folder(str(d), name=name), [])
        db, sc = db_and_scores(ws)
        if db is None or (sc is None and not args.no_covis):
            print(f"{name}: not matched / scored yet")
            continue
        res = split_scenes(ws.abspath(db.id), None if args.no_covis else load_scores(ws, sc.id), scfg)
        site = lambda n: n.split("_")[0]  # noqa: E731
        sites = Counter(site(n) for s in res["scenes"] for n in s) + Counter(site(n) for n in res["unassigned"])
        expected = 1 if name == "same_site_two_sessions" else len(sites)
        pure = sum(Counter(site(n) for n in s).most_common(1)[0][1] for s in res["scenes"])
        total = sum(len(s) for s in res["scenes"])
        compl = {st: max((sum(site(n) == st for n in s) for s in res["scenes"]), default=0) / sites[st] for st in sites}
        if name == "same_site_two_sessions":
            compl = {"all": max(len(s) for s in res["scenes"]) / sum(sites.values())}
            pure = total
        row = {"case": name, "images": sum(sites.values()), "expected_scenes": expected, "scenes": [len(s) for s in res["scenes"]],
               "unassigned": len(res["unassigned"]), "purity": round(pure / max(total, 1), 3),
               "completeness": {k: round(v, 3) for k, v in compl.items()},
               "cuts": [(d["sizes"], d["conductance"], d["decision"]) for d in res["decisions"] if d["step"] == "cut"]}
        rows.append(row)
        print(json.dumps(row))
    for ds in args.datasets:
        for b in load_benchmark(cfg, ds, "all"):
            ws = make_workspace(cfg, "eval_split", b.scene, [])
            db, sc = db_and_scores(ws)
            if db is None or (sc is None and not args.no_covis):
                continue
            res = split_scenes(ws.abspath(db.id), None if args.no_covis else load_scores(ws, sc.id), scfg)
            n = sum(len(s) for s in res["scenes"]) + len(res["unassigned"])
            row = {"case": b.name, "dataset": ds, "images": n, "expected_scenes": 1, "scenes": [len(s) for s in res["scenes"]],
                   "unassigned": len(res["unassigned"]), "split_off": n - (len(res["scenes"][0]) if res["scenes"] else 0),
                   "cuts": [(d["sizes"], d["conductance"], d["decision"]) for d in res["decisions"] if d["step"] == "cut"][:4]}
            rows.append(row)
            print(json.dumps(row))
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
