"""Run a controller on benchmark scenes (or any image folder) and, optionally, evaluate.

    python scripts/run.py --controller fixed --pipeline colmap_alikedfg_covis --scenes <scene> --gpus 4 --evaluate
    python scripts/run.py --controller scripted --split dev --gpus 4 5 6 7 --workers 4 --evaluate
    python scripts/run.py --controller agent --split dev --gpus 4 5 6 7 --workers 4 --evaluate --max-steps 20
    python scripts/run.py --folder /path/to/images --pipeline colmap_alikedfg
    python scripts/run.py --controller incremental --scenes <scene> --gpus 4 --evaluate   # the open solver (no COLMAP mapper)

Scenes run in parallel (--workers), each pinned to one of --gpus. Outputs:
outputs/runs/<run>/<scene>/{trace.jsonl, artifacts.json, final/} and outputs/runs/<run>/results.jsonl.
Evaluation (GT) is the benchmark layer only; the harness itself never reads it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import DEFAULT_CONFIG, load_config  # noqa: E402
from agenticsfm.controllers.fixed import PIPELINES, run_pipeline  # noqa: E402
from agenticsfm.datasets import folder, load_benchmark  # noqa: E402
from agenticsfm.workspace import make_workspace  # noqa: E402


def run_scene(args, cfg, run, scene, gpu, gt_dir):
    import pycolmap

    pycolmap.logging.minloglevel = 2  # the mapper is chatty
    ws = make_workspace(cfg, run, scene, [gpu])
    start = time.time()
    print(f"== {scene.name}: {len(scene.image_names())} images, gpu {gpu}, run {ws.run_dir}", flush=True)
    row = {"scene": scene.name, "controller": args.controller}
    try:
        if args.controller == "fixed":
            results = run_pipeline(ws, cfg, args.pipeline)
            row.update({k: v for k, v in results["submit"].summary.items() if k not in ("models", "path")})
        elif args.controller == "scripted":
            from agenticsfm.controllers.scripted import run_scripted

            row.update(run_scripted(ws, cfg, dataset=None if args.folder else args.dataset))
        elif args.controller == "incremental":
            from agenticsfm.controllers.incremental import run_incremental

            row.update(run_incremental(ws, cfg, dataset=None if args.folder else args.dataset, feedforward=args.feedforward,
                                       threshold=args.threshold, optimize=not args.no_optimize, seed_size=args.seed_size,
                                       group_size=args.group_size, min_support=args.min_support, hard=not args.no_hard, vlm=args.vlm, multi_model=not args.no_multi_model,
                                       support_pairs=args.support_pairs, support_norm=not args.raw_support,
                                       loop=args.loop, select=args.select, ff_off=args.ff_off))
        elif args.controller == "agent":
            from agenticsfm.agent import Agent

            agent = Agent(cfg, max_steps=args.max_steps, max_minutes=args.max_minutes, prelude=not args.no_prelude,
                          dataset=None if args.folder else args.dataset)
            row.update(agent.run(ws, cfg))
        row["status"] = "ok"
    except Exception as e:
        row.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        traceback.print_exc()
    row["wall"] = round(time.time() - start, 1)
    final = ws.run_dir / "final"
    if args.evaluate and gt_dir is not None and final.is_dir():
        from agenticsfm.evaluation import Evaluator, find_models

        metrics = Evaluator.from_config(cfg).evaluate(gt_dir, find_models(final))
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
                    if k.startswith(("auc_all@", "auc@", "colmap_auc@")) or k.startswith("num_")})
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run", help="run name (default: <controller>[_<pipeline>])")
    parser.add_argument("--controller", default="fixed", choices=["fixed", "scripted", "agent", "incremental"])
    parser.add_argument("--pipeline", default="colmap_alikedfg_covis", choices=sorted(PIPELINES))
    parser.add_argument("--max-steps", type=int, default=25, help="agent: tool-call budget (after the prelude)")
    parser.add_argument("--max-minutes", type=float, default=60, help="agent: wall-time budget")
    parser.add_argument("--no-prelude", action="store_true", help="agent: start from scratch instead of the default recipe's candidates")
    parser.add_argument("--feedforward", help="incremental: feedforward model for seed and joins (default: config feedforward.model)")
    parser.add_argument("--threshold", type=float, help="incremental: covisibility filter threshold before growing (default: none)")
    parser.add_argument("--no-optimize", action="store_true", help="incremental: submit the grown model without the shared-intrinsics refinement")
    parser.add_argument("--seed-size", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--min-support", type=float, default=0.5)
    parser.add_argument("--support-pairs", type=int, default=1000, help="incremental: match support from each image's strongest pairs only")
    parser.add_argument("--loop", default="ff", choices=["ff", "pnp"], help="incremental: feedforward placement every round (checked by PnP), or PnP rounds")
    parser.add_argument("--select", default="graph", choices=["graph", "vlm"], help="incremental: who picks each round's images")
    parser.add_argument("--ff-off", default="", help="incremental ff loop: features to switch off, comma separated (evidence,fill,points,settle,sweep,anchors,rotation,vlmplace,sky,coloc,rsrc,backface,cover,rgate,ainl)")
    parser.add_argument("--raw-support", action="store_true", help="incremental: raw match support (fixed pixel threshold, not normalised by the pair's own best fit)")
    parser.add_argument("--no-hard", action="store_true", help="incremental: no feedforward placement of hard images")
    parser.add_argument("--vlm", default="decide", choices=["off", "log", "decide"], help="incremental: the VLM advisor mode")
    parser.add_argument("--no-multi-model", action="store_true", help="incremental: do not reconstruct leftover image sets as further models")
    parser.add_argument("--dataset", default="wriva")
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--folder", help="run on this image directory instead of a benchmark")
    parser.add_argument("--gpus", nargs="*", type=int, help="tool GPUs (default: config compute.tool_gpus)")
    parser.add_argument("--workers", type=int, default=1, help="scenes in parallel (at most one per GPU)")
    parser.add_argument("--evaluate", action="store_true", help="score the submitted model against GT (benchmark scenes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run = args.run or (f"fixed_{args.pipeline}" if args.controller == "fixed" else args.controller)
    gpus = args.gpus if args.gpus else cfg["compute"]["tool_gpus"]
    if args.folder:
        scenes, gts = [folder(args.folder)], {}
    else:
        bench = [s for s in load_benchmark(cfg, args.dataset, args.split) if not args.scenes or s.name in args.scenes]
        scenes = [b.scene for b in bench]
        gts = {b.name: b.gt_dir for b in bench}
    if not scenes:
        parser.error("no scenes selected")
    scenes.sort(key=lambda s: -len(s.image_names()))  # largest first
    workers = min(args.workers, len(gpus), len(scenes))
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda it: run_scene(args, cfg, run, it[1], gpus[it[0] % len(gpus)], gts.get(it[1].name)),
                             enumerate(scenes)))
    out = Path(cfg["paths"]["runs"]) / run / "results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    keys = ["auc_all@3", "auc_all@10", "auc_all@30", "num_images_reg", "num_models"]
    if args.evaluate and any("auc_all@10" in r for r in rows):
        print(f"\n{'scene':50s} " + " ".join(f"{k:>14s}" for k in keys) + "   wall")
        for r in rows:
            print(f"{r['scene']:50s} " + " ".join(f"{(100 * r[k] if 'auc' in k else r[k]) if k in r else float('nan'):14.1f}" for k in keys) + f" {r['wall']:6.0f}")
        ok = [r for r in rows if "auc_all@10" in r]
        print(f"{'mean of ' + str(len(ok)):50s} " + " ".join(f"{100 * sum(r[k] for r in ok) / len(ok):14.1f}" if "auc" in k else f"{'':14s}" for k in keys))


if __name__ == "__main__":
    main()
