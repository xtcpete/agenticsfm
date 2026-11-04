"""Evaluate reconstructions of benchmark scenes and print per-split averages.

Each run is ``name=pattern``, where ``pattern`` holds ``{scene}`` and points to a model directory
or to a directory of sub-models (COLMAP's ``sparse/<i>``). ``--baselines`` adds the reference runs
listed in the config. A scene without a model scores 0 everywhere (status ``missing``).

    python scripts/evaluate.py --baselines --out outputs/eval/baselines.csv
    python scripts/evaluate.py --run agent=outputs/runs/v1/{scene}/final --split dev

This is the only code that reads ground truth.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import DEFAULT_CONFIG, load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402
from agenticsfm.evaluation import Evaluator, find_models  # noqa: E402

TABLE_KEYS = ("auc_all", "auc", "colmap_auc")
TABLE_THRESHOLDS = (3, 10, 30)

_evaluator: Evaluator | None = None


def _evaluate(job: tuple[str, str, str, str, str, str]) -> dict:
    global _evaluator
    config_path, run, scene, split, gt_dir, pred = job
    if _evaluator is None:
        _evaluator = Evaluator.from_config(load_config(config_path))
    models = find_models(pred)
    metrics = _evaluator.evaluate(gt_dir, models)
    return {"run": run, "scene": scene, "split": split, "status": "ok" if models else "missing",
            "pred": pred, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="wriva")
    parser.add_argument("--run", action="append", default=[], help="name=pattern with {scene}")
    parser.add_argument("--baselines", nargs="*", help="add these baseline groups from the config (none listed: all)")
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--scenes", nargs="*", help="restrict to these scenes")
    parser.add_argument("--out", help="per-scene CSV")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = dict(r.split("=", 1) for r in args.run)
    if args.baselines is not None:
        for name in args.baselines or list(cfg["baselines"]):
            base = cfg["baselines"][name]
            runs.update({run: f"{base['root']}/{rel}" for run, rel in base["runs"].items()})
    if not runs:
        parser.error("nothing to evaluate: pass --run and/or --baselines")

    scenes = [s for s in load_benchmark(cfg, args.dataset, args.split)
              if not args.scenes or s.name in args.scenes]
    jobs = [(args.config, run, s.name, s.split, str(s.gt_dir), pattern.format(scene=s.name))
            for run, pattern in runs.items() for s in scenes]
    # Largest scenes first, so the pool does not end on one 600-image scene.
    sizes = {s.name: len(s.scene.image_names()) for s in scenes}
    jobs.sort(key=lambda j: -sizes[j[2]])
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(_evaluate, jobs))
    rows.sort(key=lambda r: (list(runs).index(r["run"]), r["scene"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    cols = [f"{k}@{t}" for k in TABLE_KEYS for t in TABLE_THRESHOLDS]
    splits = ["dev", "test", "all"] if args.split == "all" else [args.split]
    print(f"{'split':5s} {'run':22s} {'n':>3s} " + " ".join(f"{c:>13s}" for c in cols) + "   reg/gt   models")
    for split in splits:
        for run in runs:
            rs = [r for r in rows if r["run"] == run and (split == "all" or r["split"] == split)]
            if not rs:
                continue
            means = [100 * statistics.mean(r[c] for r in rs) for c in cols]
            reg = sum(r["num_images_reg"] for r in rs)
            gt = sum(r["num_images_gt"] for r in rs)
            models = sum(r["num_models"] for r in rs)
            print(f"{split:5s} {run:22s} {len(rs):3d} " + " ".join(f"{m:13.1f}" for m in means)
                  + f"  {reg:5d}/{gt:<5d} {models:4d}")


if __name__ == "__main__":
    main()
