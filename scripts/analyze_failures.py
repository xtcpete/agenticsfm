"""Benchmark-layer failure analysis on the dev split.

1. For the submitted model of each run: the distribution of pairwise relative-pose errors against
   GT (fraction of pairs < 3 / 10 / 30 deg, unregistered), and how many images are "wrong" (median
   relative-rotation error vs their 10 nearest GT neighbours > 10 deg). Few wrong images vs uniform
   noise are different failure modes.
2. For the main database of each scene: GT label of every verified pair (inliers consistent with the
   GT relative pose: true edge; strong but inconsistent: wrong edge), and the average precision of
   each pair signal (inliers, FlexGlue covisibility, XDG, and any sidecar in scores/) at ranking true
   above wrong edges. Also the bridges: wrong edges whose removal would disconnect an image.

    python scripts/analyze_failures.py --runs fixed_colmap_alikedfg_covis scripted_v2 agent_v2
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
from agenticsfm.tools.examine import sampson  # noqa: E402
from agenticsfm.workspace import Workspace, scene_id  # noqa: E402


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def vec_angle(a, b):
    a, b = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def pair_errors(gt, rec, min_baseline=0.02):
    """Pairwise max(rot, trans-dir) error of the model's registered images vs GT, plus unregistered count."""
    gtP = {im.name: im.cam_from_world() for im in gt.images.values()}
    P = {im.name: im.cam_from_world() for im in rec.images.values() if im.has_pose and im.name in gtP}
    names = list(P)
    errs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rel_m = P[b] * P[a].inverse()
            rel_g = gtP[b] * gtP[a].inverse()
            r = rot_angle(rel_m.rotation.matrix() @ rel_g.rotation.matrix().T)
            t = vec_angle(rel_m.translation, rel_g.translation) if np.linalg.norm(rel_g.translation) > min_baseline else 0.0
            errs.append(max(r, t))
    n_gt = len(gtP)
    n_pairs_all = n_gt * (n_gt - 1) // 2
    return np.array(errs), n_pairs_all, n_gt - len(names)


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


def label_edges(gt, db_path, px=4.0):
    """Per verified pair: (a, b, inliers, frac of inliers consistent with GT relative pose)."""
    import pycolmap

    gtI = {im.name: im for im in gt.images.values()}
    rows = []
    with pycolmap.Database.open(db_path) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        kps = {}
        ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            n = int(g.inlier_matches.shape[0])
            if n == 0:
                continue
            i1, i2 = divmod(int(pid), 2147483647)
            a, b = names[i1], names[i2]
            if a not in gtI or b not in gtI:
                continue
            for iid in (i1, i2):
                if iid not in kps:
                    kps[iid] = np.asarray(db.read_keypoints(iid))[:, :2]
            m = g.inlier_matches
            ia, ib = gtI[a], gtI[b]
            ca, cb = gt.cameras[ia.camera_id], gt.cameras[ib.camera_id]
            x0 = np.asarray(ca.cam_from_img(kps[i1][m[:, 0]]))
            x1 = np.asarray(cb.cam_from_img(kps[i2][m[:, 1]]))
            rel = ib.cam_from_world() * ia.cam_from_world().inverse()
            Rm, t = rel.rotation.matrix(), rel.translation
            if np.linalg.norm(t) < 1e-6:
                h = np.c_[x0, np.ones(len(x0))] @ Rm.T
                err = np.linalg.norm(h[:, :2] / h[:, 2:3] - x1, axis=1)
            else:
                t = t / np.linalg.norm(t)
                E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ Rm
                err = sampson(E, x0, x1)
            f = 0.5 * (ca.mean_focal_length() + cb.mean_focal_length())
            rows.append((a, b, n, float((err * f < px).mean())))
    return rows


def average_precision(score, label):
    order = np.argsort(-score)
    label = label[order]
    tp = np.cumsum(label)
    prec = tp / (np.arange(len(label)) + 1)
    return float((prec * label).sum() / max(1, label.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--runs", nargs="*", default=["fixed_colmap_alikedfg_covis", "scripted_v2", "agent_v2"])
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--edges", action="store_true", help="also label the edges of the main database")
    args = parser.parse_args()
    import pycolmap

    pycolmap.logging.minloglevel = 2
    cfg = load_config(args.config)
    for b in load_benchmark(cfg, "wriva", args.split):
        if args.scenes and b.name not in args.scenes:
            continue
        gt = pycolmap.Reconstruction(str(b.gt_dir))
        print(f"\n== {b.name} ({gt.num_images()} images)")
        print(f"   {'run':30s} {'auc@10':>6s} {'<3':>5s} {'<10':>5s} {'<30':>5s} {'unreg':>5s} {'wrong imgs (>10deg)':>20s}  worst images")
        for run in args.runs:
            final = REPO_ROOT / cfg["paths"]["runs"] / run / b.name / "final"
            subs = sorted((p for p in final.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: -pycolmap.Reconstruction(str(p)).num_reg_images()) if final.is_dir() else []
            if not subs:
                print(f"   {run:30s} missing")
                continue
            rec = pycolmap.Reconstruction(str(subs[0]))
            errs, n_all, unreg = pair_errors(gt, rec)
            full = np.r_[errs, np.full(n_all - len(errs), 180.0)]
            ie = image_errors(gt, rec)
            wrong = sorted(((v, k) for k, v in ie.items() if v > 10), reverse=True)
            auc10 = np.mean([np.mean(full < t) for t in np.linspace(0.5, 10, 20)])
            print(f"   {run:30s} {100*auc10:6.1f} {100*np.mean(full<3):5.1f} {100*np.mean(full<10):5.1f} {100*np.mean(full<30):5.1f} {unreg:5d} {len(wrong):20d}  "
                  + ", ".join(f"{k[-8:-4]}:{v:.0f}" for v, k in wrong[:6]))
        if not args.edges:
            continue
        store = REPO_ROOT / cfg["paths"]["cache"] / scene_id(b.scene)
        ws = Workspace(b.scene, store / "_analysis", REPO_ROOT / cfg["paths"]["cache"])
        dbs = [a for a in ws.list("db") if a.tool == "match" and a.params.get("matcher") == "flexglue" and a.params.get("camera_mode") == "auto"]
        if not dbs:
            continue
        dba = max(dbs, key=lambda a: a.summary.get("num_verified_pairs", 0))
        rows = label_edges(gt, ws.abspath(dba.id))
        n = np.array([r[2] for r in rows]); f = np.array([r[3] for r in rows])
        true_edge = f >= 0.5
        wrong_edge = (f < 0.2)
        strong = n >= 50
        print(f"   edges of db {dba.id}: {len(rows)} verified; true {true_edge.sum()}, wrong {wrong_edge.sum()} (strong wrong {(wrong_edge & strong).sum()}), ambiguous {(~true_edge & ~wrong_edge).sum()}")
        # signals
        signals = {"inliers": n.astype(float)}
        for sc in ws.list("scores"):
            if sc.inputs.get("db") == dba.id:
                z = np.load(ws.abspath(sc.id))
                lookup = {(str(a), str(bb)): float(s) for a, bb, s in zip(z["name0"], z["name1"], z["score"])}
                signals[sc.params["method"]] = np.array([lookup.get((r[0], r[1]), lookup.get((r[1], r[0]), np.nan)) for r in rows])
        keep = true_edge | wrong_edge
        for name, s in signals.items():
            ok = keep & ~np.isnan(s)
            ap = average_precision(s[ok], true_edge[ok].astype(float))
            ap_strong = average_precision(s[ok & strong], true_edge[ok & strong].astype(float))
            # best single threshold: max accuracy on wrong-vs-true
            ths = np.unique(np.round(s[ok], 2))
            best = max(((np.mean((s[ok] >= t) == true_edge[ok]), t) for t in ths[::max(1, len(ths)//50)]), default=(0, 0))
            print(f"     {name:16s} AP(true>wrong) {ap:.3f}  strong-only {ap_strong:.3f}  best thr {best[1]:.2f} acc {best[0]:.3f}  "
                  f"wrong kept at 0.6: {int(((s >= 0.6) & wrong_edge).sum())}  true dropped at 0.6: {int(((s < 0.6) & true_edge).sum())}")
        # which images are only reachable through wrong edges at the covis threshold
        wrong_pairs = [(r[0], r[1]) for r, w in zip(rows, wrong_edge) if w]
        per_img = {}
        for a, bb in wrong_pairs:
            per_img[a] = per_img.get(a, 0) + 1; per_img[bb] = per_img.get(bb, 0) + 1
        print("     images in most wrong edges: " + ", ".join(f"{k[-8:-4]}:{v}" for k, v in sorted(per_img.items(), key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    main()
