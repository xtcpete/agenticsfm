"""Offline tests on saved Pi3X-loop runs (benchmark layer: GT only for scoring).

Replays every Pi3X round of a finished run from the saved dense depth (_work/scene*/ff_ff<k>/depth_*.npz) and compares
alignment of the feedforward frame to the model:
  rot    centres + chordal-mean rotations of the anchors (the solver's align_rot)
  pts    robust similarity on 3D points: the anchors' keypoints with a model point, lifted through Pi3X depth
The model at round k is approximated by the final grown model restricted to points seen by >= 2 images registered by
round k. Error of an aligned batch pose = rotation/centre gap to the image's final pose, counted only on images whose
final pose is within 3 degrees of GT (so the final pose is a good stand-in for the truth).
Also: depth-consistency check of candidate poses (--depth-check) on images where Pi3X and PnP disagreed.

usage: python scripts/analysis/ff_offline.py --runs ff_vlm_dev ff_vlm_dev_s02 [--site <site dir for GT errors>]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agenticsfm.tools.frame import umeyama  # noqa: E402


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def find_grow(run, scene):
    d = ROOT / "outputs/runs" / run / scene
    recs = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines() if l.strip()]
    g = next(x["artifact"] for x in recs if x.get("tool") == "grow_model" and x.get("artifact"))
    for c in (ROOT / "outputs/cache").iterdir():
        if (c / "model" / g / "grow_log.json").is_file():
            return d, c, c / "model" / g


def gt_errors(scene):
    """Per-image rotation error of the submitted model after a robust similarity to GT (export_site's alignment)."""
    from agenticsfm.config import load_config
    from agenticsfm.datasets import load_benchmark
    sys.path.insert(0, str(ROOT / "scripts"))
    cfg = load_config()
    b = next(x for x in load_benchmark(cfg, "wriva", "all") if x.name == scene)
    return pycolmap.Reconstruction(str(b.gt_dir))


def align_gt(rec, gt):
    """Similarity model->GT on camera centres (RANSAC umeyama), then per-image rotation error in degrees."""
    names = [im.name for im in rec.images.values() if im.has_pose]
    G = {im.name: im for im in gt.images.values()}
    names = [n for n in names if n in G]
    by = {im.name: im for im in rec.images.values()}
    src = np.array([by[n].projection_center() for n in names]); dst = np.array([G[n].projection_center() for n in names])
    rng = np.random.default_rng(0); best = None
    scale = np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1))
    for _ in range(300):
        idx = rng.choice(len(names), 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except Exception:
            continue
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        inl = r < 0.05 * scale
        if best is None or inl.sum() > best[0].sum():
            best = (inl, s, R, t)
    s, R, t = umeyama(src[best[0]], dst[best[0]])
    err = {}
    for n in names:
        Rm = by[n].cam_from_world().rotation.matrix(); Rg = G[n].cam_from_world().rotation.matrix()
        err[n] = rot_angle(Rg @ (Rm @ R.T).T)
    return err


def largest(model_dir):
    subs = [d for d in Path(model_dir).iterdir() if d.is_dir() and d.name.isdigit()]
    recs = [pycolmap.Reconstruction(str(d)) for d in subs]
    return max(recs, key=lambda r: r.num_reg_images())


def lift(z, i, im, rec, order, k):
    """Anchor i of npz z: (Pi3X world points, model points) for keypoints whose 3D point is seen by >= 2 images
    registered by round k."""
    depth = z["depth"][i].astype(np.float32); conf = z["conf"][i].astype(np.float32)
    K = z["K"][i]; E = z["E"][i]; sx, sy, ox, oy = z["transforms"][i]
    Rinv = E[:, :3].T; c = -Rinv @ E[:, 3]
    src, dst = [], []
    H, W = depth.shape
    for p2 in im.points2D:
        if not p2.has_point3D():
            continue
        p3 = rec.points3D[p2.point3D_id]
        seen = sum(1 for el in p3.track.elements if order.get(rec.images[el.image_id].name, 1e9) <= k)
        if seen < 2:
            continue
        u, v = p2.xy
        mu, mv = u * sx + ox, v * sy + oy
        iu, iv = int(round(mu)), int(round(mv))
        if not (0 <= iu < W and 0 <= iv < H) or depth[iv, iu] <= 0 or conf[iv, iu] < 0.1:
            continue
        d = depth[iv, iu]
        xc = np.array([(mu - K[0, 2]) / K[0, 0] * d, (mv - K[1, 2]) / K[1, 1] * d, d])
        src.append(Rinv @ xc + c); dst.append(p3.xyz)
    return np.array(src).reshape(-1, 3), np.array(dst).reshape(-1, 3)


def align_points(src, dst, scale, rng, iters=400, thr=0.02):
    """RANSAC similarity on 3D-3D pairs; threshold relative to the model's scene scale. Returns (s, R, t, inlier frac)."""
    if len(src) < 20:
        return None
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 4, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except Exception:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        inl = r < thr * scale
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 10:
        return None
    s, R, t = umeyama(src[best], dst[best])
    for _ in range(2):   # re-estimate on the refined inlier set
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        inl = r < thr * scale
        s, R, t = umeyama(src[inl], dst[inl])
    return s, R, t, float(inl.mean())


def align_rot(z, idx_of, anchors, rec_by):
    """The solver's align_rot (chordal-mean rotation, then scale/translation from centres), without its trimming."""
    use = [a for a in anchors if a in idx_of and a in rec_by]
    if len(use) < 3:
        return None
    Ms = [rec_by[a].cam_from_world().rotation.matrix().T @ z["E"][idx_of[a]][:, :3] for a in use]
    U, _, Vt = np.linalg.svd(sum(Ms)); R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    src = np.array([-z["E"][idx_of[a]][:, :3].T @ z["E"][idx_of[a]][:, 3] for a in use])
    dst = np.array([rec_by[a].projection_center() for a in use])
    ms, md = src.mean(0), dst.mean(0); Rs = (R @ (src - ms).T).T
    s = float(np.sum((dst - md) * Rs) / max(np.sum(Rs * Rs), 1e-12)); t = md - s * R @ ms
    return s, R, t


def pose_err(E, s, R, t, im, scale):
    Rn = E[:, :3] @ R.T; tn = s * E[:, 3] - Rn @ t
    Rm = im.cam_from_world().rotation.matrix()
    cn = -Rn.T @ tn
    return rot_angle(Rm @ Rn.T), float(np.linalg.norm(cn - im.projection_center()) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    chosen = {}
    for run in args.runs:
        res = {json.loads(l)["scene"]: json.loads(l) for l in (ROOT / "outputs/runs" / run / "results.jsonl").read_text().splitlines() if l.strip()}
        for d in sorted((ROOT / "outputs/runs" / run).iterdir()):
            if d.is_dir() and res.get(d.name, {}).get("status") == "ok":
                chosen[d.name] = run
    rng = np.random.default_rng(0)
    report = {}
    for scene, run in chosen.items():
        rd, store, mdir = find_grow(run, scene)
        log = json.loads((mdir / "grow_log.json").read_text())["log"]
        rec = largest(mdir)
        final = largest(rd / "final")
        gerr = align_gt(final, gt_errors(scene))
        good = {n for n, e in gerr.items() if e < 3}
        by = {im.name: im for im in rec.images.values() if im.has_pose}
        cen = np.median([im.projection_center() for im in by.values()], 0)
        scale = float(np.median([np.linalg.norm(im.projection_center() - cen) for im in by.values()]))
        order = {}
        for r in log:
            if r.get("kind") == "seed" and not r.get("model"):
                for n in r["attempts"][r["chosen"]]["kept"]:
                    order.setdefault(n, 0)
            elif isinstance(r.get("registered"), list) and not r.get("model"):
                for n in r["registered"]:
                    order.setdefault(n, r.get("round", 0))
        rows = []
        for r in log:
            if r.get("kind") != "ff" or r.get("model"):
                continue
            k = r["round"]
            scene_dir = mdir / "_work" / f"scene{r.get('scene', 0)}"
            npz = scene_dir / f"ff_ff{k}" / "depth_0.npz"
            if not npz.is_file():
                continue
            z = np.load(npz)
            names = [str(n) for n in z["names"]]
            idx_of = {n: i for i, n in enumerate(names)}
            anchors = [a for a in r["anchors"] if a in by and order.get(a, 1e9) < k]
            src, dst = [], []
            for a in anchors:
                if a in idx_of:
                    s_, d_ = lift(z, idx_of[a], by[a], rec, order, k - 1)
                    src.append(s_); dst.append(d_)
            src = np.concatenate(src) if src else np.zeros((0, 3)); dst = np.concatenate(dst) if dst else np.zeros((0, 3))
            ap_ = align_points(src, dst, scale, rng)
            ar_ = align_rot(z, idx_of, anchors, by)
            for n in r["batch"]:
                if n not in idx_of or n not in by or n not in good:
                    continue
                e = {"round": k, "name": n, "accepted": n in r["registered"], "why": r["images"].get(n, {}).get("why"),
                     "align_ok": r["align"].get("ok"), "align_rot_dev": r["align"].get("rot_dev")}
                if ar_:
                    e["rot"] = pose_err(z["E"][idx_of[n]], *ar_, by[n], scale)
                if ap_:
                    e["pts"] = pose_err(z["E"][idx_of[n]], *ap_[:3], by[n], scale); e["pts_inl"] = ap_[3]; e["pts_n"] = len(src)
                rows.append(e)
        report[scene] = rows
        def summ(key):
            v = np.array([x[key][0] for x in rows if key in x])
            p = np.array([x[key][1] for x in rows if key in x])
            if not len(v):
                return "n=0"
            return f"n={len(v)} rot med {np.median(v):.2f} p90 {np.percentile(v, 90):.1f} >5deg {np.mean(v > 5) * 100:.0f}% >15deg {np.mean(v > 15) * 100:.0f}% | centre med {np.median(p):.3f}"
        print(f"== {scene} ({run})")
        print("   centres+rotations:", summ("rot"))
        print("   points          :", summ("pts"))
        bad = [x for x in rows if "rot" in x and x["rot"][0] > 5]
        if bad:
            fixed = sum(1 for x in bad if "pts" in x and x["pts"][0] <= 5)
            print(f"   images > 5deg under centres+rotations: {len(bad)}, of which <= 5deg under points: {fixed}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, default=float))


if __name__ == "__main__":
    main()
