"""Score rayba_run.py outputs against GT (agenticsfm env; benchmark layer). Per scene: the grown model (input of both
BAs), COLMAP's `optimize` output from the run, and each GPU-BA variant (poses and intrinsics groups written into a copy
of the grown model). Prints the evaluator's metrics and the wall time of each BA (ours: problem construction + BA, when the export recorded
its construction time).

usage: python scripts/analysis/rayba_eval.py --run p22_vlm_dev --dataset wriva --prep <dir> --out <dir>
"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agenticsfm.config import load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402
from agenticsfm.evaluation import Evaluator, find_models  # noqa: E402


def optimize_artifact(run, scene):
    recs = [json.loads(l) for l in (ROOT / "outputs/runs" / run / scene / "trace.jsonl").read_text().splitlines() if l.strip()]
    for a, b in zip(recs, recs[1:]):
        if b.get("tool") == "optimize" and b.get("artifact"):
            for c in (ROOT / "outputs/cache").iterdir():
                d = c / "model" / b["artifact"]
                if d.is_dir():
                    return d, b["time"] - a["time"]
    return None, None


def write_variant(model_dir, z, v, dst):
    rec = pycolmap.Reconstruction(str(model_dir))
    by = {im.name: im for im in rec.images.values()}
    for n, name in enumerate(map(str, z["names"])):       # poses only: the evaluator scores poses
        im = by[name]
        rec.frame(im.frame_id).rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(v["R"][n]), v["t"][n])
    dst.mkdir(parents=True, exist_ok=True)
    rec.write(str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--split", default="all")
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="*", default=None)
    a = ap.parse_args()
    cfg = load_config()
    ev = Evaluator.from_config(cfg)
    bench = {b.name: b for b in load_benchmark(cfg, a.dataset, a.split)}
    keys = a.keys or (["auc@3", "auc@10", "auc_all@10"] if a.dataset == "wriva" else ["colmap_auc@1", "colmap_auc@3", "colmap_auc@5"])
    tmp = Path(tempfile.mkdtemp())
    rows = {}
    for f in sorted(Path(a.prep).glob("*.npz")):
        sc = f.stem
        if sc not in bench:
            continue
        z = np.load(f)
        res = {}
        m = ev.evaluate(bench[sc].gt_dir, [Path(str(z["model_dir"]))])
        sc100 = lambda m: [100 * m[k] if (k.startswith("colmap_auc") or m[k] <= 1.0) else m[k] for k in keys]  # noqa: E731
        res["grown"] = (sc100(m), None)
        od, ot = optimize_artifact(a.run, sc)
        if od is not None:
            m = ev.evaluate(bench[sc].gt_dir, [max(find_models(od), key=lambda p: pycolmap.Reconstruction(str(p)).num_reg_images())])
            res["colmap_opt"] = (sc100(m), ot)
        for vf in sorted(Path(a.out).glob(f"{sc}.*.npz")):
            name = vf.name[len(sc) + 1:-4]
            v = np.load(vf)
            d = tmp / sc / name
            write_variant(Path(str(z["model_dir"])), z, v, d)
            m = ev.evaluate(bench[sc].gt_dir, [d])
            res[name] = (sc100(m), float(v["wall"]) + (float(z["t_build"]) if "t_build" in z else 0.0))
        rows[sc] = res
        print(f"{sc[:38]:38s} " + " | ".join(f"{n}: " + " ".join(f"{x:5.1f}" for x in vals) + (f" ({t:.0f}s)" if t is not None else "")
                                         for n, (vals, t) in res.items()), flush=True)
    names = sorted({n for r in rows.values() for n in r}, key=lambda n: ["grown", "colmap_opt"].index(n) if n in ("grown", "colmap_opt") else 9)
    print("mean over scenes (" + ", ".join(keys) + "):")
    for n in names:
        vals = [r[n][0] for r in rows.values() if n in r]
        ts = [r[n][1] for r in rows.values() if n in r and r[n][1] is not None]
        print(f"  {n:12s} " + " ".join(f"{x:5.1f}" for x in np.mean(vals, 0)) + f"   ({len(vals)} scenes, BA time total {sum(ts):.0f}s)")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
