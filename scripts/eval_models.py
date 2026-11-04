"""Benchmark-layer analysis: score every model artifact in a scene's store against GT, next to its
GT-free examine statistics. Tells whether `examine` separates good models from bad ones.

    python scripts/eval_models.py --scenes t04_v09_s00_r04_VaryingAltitudes_BLH0001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import DEFAULT_CONFIG, REPO_ROOT, load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402
from agenticsfm.evaluation import Evaluator, find_models  # noqa: E402
from agenticsfm.workspace import Workspace, scene_id  # noqa: E402


def describe(ws: Workspace, model) -> str:
    chain = []
    for a in ws.provenance(model.id):
        if a.kind == "model":
            chain.append(a.params["backend"] if a.tool == "reconstruct" else f"optimize({a.params.get('share_intrinsics')})")
        elif a.tool == "filter_edges":
            chain.append(f"filter {a.params.get('threshold')}")
        elif a.kind == "scores":
            chain.append(a.params["method"])
        elif a.kind == "db" and a.tool == "match":
            chain.append(f"{a.params['detector']}+{a.params['matcher']}")
        elif a.kind == "graph":
            chain.append(a.params["method"])
    return " <- ".join(chain)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="wriva")
    parser.add_argument("--split", default="all")
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ev = Evaluator.from_config(cfg)
    for b in load_benchmark(cfg, args.dataset, args.split):
        if args.scenes and b.name not in args.scenes:
            continue
        store = REPO_ROOT / cfg["paths"]["cache"] / scene_id(b.scene)
        if not (store / "artifacts.json").is_file():
            continue
        ws = Workspace(b.scene, store / "_analysis", REPO_ROOT / cfg["paths"]["cache"])
        n = len(b.scene.image_names())
        print(f"\n== {b.name} ({n} images)")
        print(f"{'model':11s} {'auc_all@3/10/30':>18s} {'largest':>7s} {'models':>6s} {'reproj':>6s} {'incons<0.5':>10s} {'<0.8':>6s}  pipeline")
        rows = []
        for m in ws.list("model"):
            exams = [a for a in ws.list("examine") if a.inputs.get("model") == m.id]
            e = exams[0].summary if exams else {}
            c = e.get("edges", {}).get("consistency", {})
            metrics = ev.evaluate(b.gt_dir, find_models(ws.abspath(m.id)))
            rows.append((metrics["auc_all@10"], m, e, c, metrics))
        for auc10, m, e, c, metrics in sorted(rows, key=lambda r: -r[0]):
            largest = e["models"][0]["num_images"] if e.get("models") else (m.summary["models"][0]["num_images"] if m.summary.get("models") else 0)
            reproj = e["models"][0]["mean_reproj_error_px"] if e.get("models") else float("nan")
            print(f"{m.id:11s} {100*metrics['auc_all@3']:5.1f} {100*auc10:5.1f} {100*metrics['auc_all@30']:5.1f} {largest:7d} {m.summary.get('num_models', 0):6d} {reproj:6.2f} "
                  f"{c.get('frac_pairs_below_0.5', float('nan')):10.3f} {c.get('frac_pairs_below_0.8', float('nan')):6.3f}  {describe(ws, m)}")


if __name__ == "__main__":
    main()
