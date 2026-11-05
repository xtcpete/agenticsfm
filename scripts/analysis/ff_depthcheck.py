"""Offline test of a depth-consistency pose check (P2) on saved Pi3X-loop runs (GT only for selecting test images).

For images whose final pose is within 3 deg of GT: take the image's Pi3X depth from the first round it was batched in,
project the model's points that this image does not observe into a candidate pose, fit the depth scale (median ratio)
and score the fraction of points whose depth agrees with Pi3X depth within 15% (log ratio). Candidates: the final pose
(stand-in for the truth) and the same centre rotated by 2..45 degrees about a random axis. Reports the mean score per
candidate and how often the true pose scores higher than each perturbation.

usage: python scripts/analysis/ff_depthcheck.py --runs ff_vlm_dev ff_vlm_dev_s02
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ff_offline import find_grow, largest, gt_errors, align_gt  # noqa: E402

PERT = [2, 5, 10, 20, 45]


def rodrigues(axis, deg):
    a = np.radians(deg); k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def score(R, t, cam, X, depth, conf, tr, tol=0.15):
    Xc = X @ R.T + t
    z = Xc[:, 2]; front = z > 1e-6
    if front.sum() < 30:
        return 0.0, 0
    uv = cam.img_from_cam(Xc[front])
    sx, sy, ox, oy = tr
    mu = np.round(uv[:, 0] * sx + ox).astype(int); mv = np.round(uv[:, 1] * sy + oy).astype(int)
    H, W = depth.shape
    ins = (mu >= 0) & (mu < W) & (mv >= 0) & (mv < H)
    if ins.sum() < 30:
        return 0.0, int(ins.sum())
    d = depth[mv[ins], mu[ins]]; c = conf[mv[ins], mu[ins]]; zz = z[front][ins]
    ok = (d > 0) & (c >= 0.1)
    if ok.sum() < 30:
        return 0.0, int(ok.sum())
    r = np.log(zz[ok] / d[ok]); s = np.median(r)
    return float(np.mean(np.abs(r - s) < tol)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--per-scene", type=int, default=60)
    args = ap.parse_args()
    chosen = {}
    for run in args.runs:
        res = {json.loads(l)["scene"]: json.loads(l) for l in (ROOT / "outputs/runs" / run / "results.jsonl").read_text().splitlines() if l.strip()}
        for d in sorted((ROOT / "outputs/runs" / run).iterdir()):
            if d.is_dir() and res.get(d.name, {}).get("status") == "ok":
                chosen[d.name] = run
    rng = np.random.default_rng(0)
    for scene, run in chosen.items():
        rd, store, mdir = find_grow(run, scene)
        log = json.loads((mdir / "grow_log.json").read_text())["log"]
        rec = largest(mdir); final = largest(rd / "final")
        gerr = align_gt(final, gt_errors(scene)); good = {n for n, e in gerr.items() if e < 3}
        by = {im.name: im for im in rec.images.values() if im.has_pose}
        pts = list(rec.points3D.values())
        X_all = np.array([p.xyz for p in pts]); owners = [set(el.image_id for el in p.track.elements) for p in pts]
        first = {}
        for r in log:
            if r.get("kind") == "ff" and not r.get("model"):
                npz = mdir / "_work" / f"scene{r.get('scene', 0)}" / f"ff_ff{r['round']}" / "depth_0.npz"
                for n in r["batch"]:
                    if n not in first and npz.is_file():
                        first[n] = npz
        test = [n for n in first if n in good and n in by]
        rng.shuffle(test); test = test[: args.per_scene]
        S = {p: [] for p in [0] + PERT}
        for n in test:
            z = np.load(first[n]); names = [str(x) for x in z["names"]]; i = names.index(n)
            depth = z["depth"][i].astype(np.float32); conf = z["conf"][i].astype(np.float32); tr = z["transforms"][i]
            im = by[n]; cam = rec.cameras[im.camera_id]
            keep = np.array([im.image_id not in o for o in owners])
            X = X_all[keep]
            if len(X) > 20000:
                X = X[rng.choice(len(X), 20000, replace=False)]
            P = im.cam_from_world(); R0, t0 = P.rotation.matrix(), P.translation; c0 = -R0.T @ t0
            S[0].append(score(R0, t0, cam, X, depth, conf, tr)[0])
            for deg in PERT:
                Rp = rodrigues(rng.normal(size=3), deg) @ R0; tp = -Rp @ c0
                S[deg].append(score(Rp, tp, cam, X, depth, conf, tr)[0])
        base = np.array(S[0])
        line = " ".join(f"{p}°:{np.mean(S[p]):.2f}(true wins {np.mean(base > np.array(S[p])) * 100:.0f}%)" for p in PERT)
        print(f"== {scene[-28:]} n={len(base)} true {base.mean():.2f} | {line}", flush=True)


if __name__ == "__main__":
    main()
