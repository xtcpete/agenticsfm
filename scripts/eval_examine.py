"""Benchmark-layer audit of the examination tools, over every model artifact of the dev scenes.

Model level: does each GT-free statistic rank the candidate models of a scene the way GT does
(Spearman correlation with auc_all@10), and which model would each statistic pick?
Image level: do the per-image signals (match support, VGGT disagreement) find the images GT says
are misplaced (median relative-rotation error vs 10 nearest GT neighbours > 30 deg) or imprecise
(> 10 deg)?

    python scripts/eval_examine.py --split dev [--verify]   # --verify runs verify_poses where missing (GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenticsfm.config import DEFAULT_CONFIG, REPO_ROOT, load_config  # noqa: E402
from agenticsfm.datasets import load_benchmark  # noqa: E402
from agenticsfm.evaluation import Evaluator, find_models  # noqa: E402
from agenticsfm.workspace import make_workspace  # noqa: E402


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def image_errors(gt, rec):
    gtR = {im.name: im.cam_from_world().rotation.matrix() for im in gt.images.values()}
    gtC = {im.name: im.projection_center() for im in gt.images.values()}
    R = {im.name: im.cam_from_world().rotation.matrix() for im in rec.images.values() if im.has_pose and im.name in gtR}
    out = {}
    for a in R:
        near = sorted((n for n in R if n != a), key=lambda n: np.linalg.norm(gtC[n] - gtC[a]))[:10]
        e = [rot_angle((R[b] @ R[a].T) @ (gtR[b] @ gtR[a].T).T) for b in near]
        out[a] = float(np.median(e)) if e else 0.0
    return out


def per_image_support(edges_path: Path) -> dict[str, float]:
    num, den = {}, {}
    for line in edges_path.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] != "same" or r["inliers"] < 50 or r["consistent_frac"] is None:
            continue
        for x in r["pair"]:
            num[x] = num.get(x, 0.0) + r["inliers"] * r["consistent_frac"]
            den[x] = den.get(x, 0.0) + r["inliers"]
    return {x: num[x] / den[x] for x in den}


def spearman(a, b):
    from scipy.stats import spearmanr

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--dataset", default="wriva")
    args = parser.parse_args()
    import pycolmap

    import agenticsfm.tools  # noqa: F401
    from agenticsfm.tools import call

    pycolmap.logging.minloglevel = 2
    cfg = load_config(args.config)
    ev = Evaluator.from_config(cfg)
    rows = []       # model level
    img_rows = []   # image level
    for b in load_benchmark(cfg, args.dataset, args.split):
        ws = make_workspace(cfg, "_analysis", b.scene, [args.gpu])
        gt = pycolmap.Reconstruction(str(b.gt_dir))
        n = len(b.scene.image_names())
        for m in ws.list("model"):
            subs = find_models(ws.abspath(m.id))
            if not subs:
                continue
            metrics = ev.evaluate(b.gt_dir, subs)
            ex = call(ws, cfg, "examine", {"model_id": m.id}, source="audit")
            if ex.error:
                continue
            E = ex.summary["edges"]["consistency"]
            largest = ex.summary["models"][0] if ex.summary.get("models") else {}
            support = per_image_support(ws.abspath(ex.artifact) / "edges.jsonl")
            ver = None
            vs = [a for a in ws.list("examine") if a.tool == "verify_poses" and a.inputs.get("model") == m.id and a.params.get("v", 1) >= 2]
            if vs:
                ver = vs[0]
            elif args.verify:
                r = call(ws, cfg, "verify_poses", {"model_id": m.id}, source="audit")
                ver = ws.get(r.artifact) if r.artifact else None
            vggt = {}
            if ver is not None:
                for p in json.loads((ws.abspath(ver.id) / "per_image.json").read_text()):
                    vggt[p["image"]] = p["median_rot_err_deg"]
            rec_largest = max((pycolmap.Reconstruction(str(s)) for s in subs), key=lambda r: r.num_reg_images())
            ierr = image_errors(gt, rec_largest)
            misplaced = {k for k, v in ierr.items() if v > 30}
            imprecise = {k for k, v in ierr.items() if v > 10}
            rows.append({"scene": b.name, "model": m.id, "tool": m.tool, "backend": m.params.get("backend", m.tool),
                         "auc10": 100 * metrics["auc_all@10"], "auc3": 100 * metrics["auc_all@3"],
                         "largest": largest.get("num_images", 0), "coverage": largest.get("num_images", 0) / n,
                         "num_models": ex.summary["num_models"], "reproj": largest.get("mean_reproj_error_px", np.nan),
                         "track": largest.get("mean_track_length", np.nan),
                         "strong_incons": E.get("strong_inconsistent_frac", np.nan), "incons_w": E.get("inlier_weighted_inconsistency", np.nan),
                         "frac_below_0.5": E.get("frac_pairs_below_0.5", np.nan),
                         "low_support_imgs": sum(1 for v in support.values() if v < 0.5),
                         "vggt_suspects": ver.summary.get("num_suspect") if ver is not None else np.nan,
                         "vggt_p90": ver.summary.get("rotation_disagreement_deg", {}).get("p90") if ver is not None else np.nan,
                         "gt_misplaced": len(misplaced), "gt_imprecise": len(imprecise)})
            for name, err in ierr.items():
                img_rows.append({"scene": b.name, "model": m.id, "image": name, "gt_err": err,
                                 "support": support.get(name, np.nan), "vggt": vggt.get(name, np.nan)})
    import pandas as pd

    df = pd.DataFrame(rows)
    print(f"{len(df)} models over {df.scene.nunique()} scenes\n")
    stats = ["coverage", "reproj", "track", "strong_incons", "incons_w", "frac_below_0.5", "low_support_imgs", "vggt_suspects", "vggt_p90", "gt_misplaced"]
    print("MODEL RANKING: Spearman correlation with auc_all@10 within each scene (mean over scenes), and the mean regret")
    print("of picking the model each statistic prefers (best auc in scene minus picked auc), vs the oracle pick:")
    for s in stats:
        cors, regrets = [], []
        for scene, g in df.groupby("scene"):
            g = g.dropna(subset=[s])
            if len(g) < 3:
                continue
            sign = 1 if s in ("coverage", "track") else -1
            cors.append(spearman(sign * g[s].values, g.auc10.values))
            pick = g.iloc[np.argmax(sign * g[s].values)]
            regrets.append(g.auc10.max() - pick.auc10)
        print(f"   {s:18s} rho {np.nanmean(cors):+.2f}   regret {np.mean(regrets):5.1f} over {len(regrets)} scenes")
    # simple combined rules
    def regret_of(score_fn, label):
        regrets = []
        for scene, g in df.groupby("scene"):
            sc = score_fn(g)
            pick = g.iloc[int(np.nanargmax(sc))]
            regrets.append(g.auc10.max() - pick.auc10)
        print(f"   {label:40s} regret {np.mean(regrets):5.1f}")
    print("\nCOMBINED PICK RULES:")
    regret_of(lambda g: g.coverage - 2 * g.strong_incons, "coverage - 2*strong_incons (scripted v1)")
    regret_of(lambda g: g.coverage - 2 * g.strong_incons - 0.05 * g.reproj, "... - 0.05*reproj")
    regret_of(lambda g: g.coverage - 2 * g.strong_incons - g.low_support_imgs / g.largest.clip(lower=1), "... - low_support_imgs/largest")
    regret_of(lambda g: g.coverage - 2 * g.strong_incons - g.low_support_imgs / g.largest.clip(lower=1) - 0.05 * g.reproj, "... - low_support/largest - 0.05*reproj")
    regret_of(lambda g: (g.coverage - g.low_support_imgs / g.largest.clip(lower=1)) * (g.num_models == 1) - 0.05 * g.reproj, "single-model only, support, reproj")
    print("\nPER SCENE: best model by GT vs by the scripted rule:")
    for scene, g in df.groupby("scene"):
        best = g.iloc[g.auc10.values.argmax()]
        sc = g.coverage - 2 * g.strong_incons
        pick = g.iloc[int(np.nanargmax(sc))]
        print(f"   {scene[-24:]:24s} n={len(g):2d}  GT best {best.auc10:5.1f} ({best.backend}, {best.largest} img, incons {best.strong_incons:.3f}, reproj {best.reproj:.2f})"
              f"  rule picks {pick.auc10:5.1f} ({pick.backend}, {pick.largest} img, incons {pick.strong_incons:.3f}, reproj {pick.reproj:.2f})")
    di = pd.DataFrame(img_rows)
    print(f"\nIMAGE LEVEL: {len(di)} image placements; misplaced (>30 deg) {int((di.gt_err > 30).sum())}, imprecise (>10) {int((di.gt_err > 10).sum())}")
    for label, flag in (("support < 0.5", di.support < 0.5), ("support < 0.7", di.support < 0.7),
                        ("vggt > 15", di.vggt > 15), ("vggt > 15 & support < 0.5", (di.vggt > 15) & (di.support < 0.5)),
                        ("vggt > 30 | support < 0.3", (di.vggt > 30) | (di.support < 0.3))):
        for thr in (30, 10):
            bad = di.gt_err > thr
            ok = ~di.support.isna() if "support" in label else ~di.vggt.isna()
            if "vggt" in label:
                ok = ok & ~di.vggt.isna()
            f, bb = flag[ok].fillna(False), bad[ok]
            tp, fp, fn = int((f & bb).sum()), int((f & ~bb).sum()), int((~f & bb).sum())
            print(f"   {label:28s} vs gt_err>{thr:2d}: flagged {int(f.sum()):4d}  precision {tp/max(1,tp+fp):.2f}  recall {tp/max(1,tp+fn):.2f}  (n={int(ok.sum())})")
    out = REPO_ROOT / "outputs" / "eval" / f"examine_audit_{args.dataset}_{args.split}.csv"
    df.to_csv(out, index=False)
    di.to_csv(out.with_name("examine_audit_images.csv"), index=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
