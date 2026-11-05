"""Offline test of the Phase 3b stability check (benchmark layer: GT only for scoring).

For every vlm_place group of a finished run, Pi3X is run three times on the leftover images: with the anchors the solver
used (full), and with each half of the VLM's picks (picks[0::2], picks[1::2]). Each run is aligned to the final grown
model (points first, as the solver does, then rotations/centres), and every leftover image gets
  err_full / err_h0 / err_h1   rotation error vs GT of the aligned Pi3X pose (model->GT similarity on camera centres)
  gap                          rotation gap between the two half placements (the online check's statistic)
so we can see whether a small gap goes with a correct placement.

usage: python scripts/analysis/ff_stability.py --run p123_vlm_dev --scenes <scene> ... --gpu 5
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from agenticsfm.tools.frame import umeyama  # noqa: E402
from ff_offline import find_grow, largest, gt_errors, lift, align_points, align_rot, rot_angle  # noqa: E402


def sim_to_gt(rec, gt):
    G = {im.name: im for im in gt.images.values()}
    names = [im.name for im in rec.images.values() if im.has_pose and im.name in G]
    by = {im.name: im for im in rec.images.values()}
    src = np.array([by[n].projection_center() for n in names]); dst = np.array([G[n].projection_center() for n in names])
    scale = np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1)); rng = np.random.default_rng(0); best = None
    for _ in range(300):
        idx = rng.choice(len(names), 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except Exception:
            continue
        inl = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1) < 0.05 * scale
        if best is None or inl.sum() > best.sum():
            best = inl
    return umeyama(src[best], dst[best]), G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=5); ap.add_argument("--out")
    a = ap.parse_args()
    from agenticsfm.config import load_config
    from agenticsfm.backends import run_backend
    cfg = load_config()
    out = []
    for scene in a.scenes:
        d, c, g = find_grow(a.run, scene)
        L = json.loads((g / "grow_log.json").read_text())
        rows = [r for r in L["log"] if r.get("kind") == "vlm_place" and r.get("picks")]
        if not rows:
            print(scene, "no vlm_place rows"); continue
        rec = largest(g)
        by = {im.name: im for im in rec.images.values() if im.has_pose}
        scale = float(np.median(np.linalg.norm(np.array([im.projection_center() for im in by.values()])
                                               - np.median(np.array([im.projection_center() for im in by.values()]), 0), axis=1)))
        (sg, Rg, tg), G = sim_to_gt(rec, gt_errors(scene))
        images_dir = json.loads((g / "meta.json").read_text()).get("images_dir") if (g / "meta.json").is_file() else None
        if not images_dir:
            from agenticsfm.datasets import load_benchmark
            images_dir = str(next(x for x in load_benchmark(cfg, "wriva", "all") if x.name == scene).scene.images_dir)
        for r in rows:
            pend = [n for n in r["images"] if n in G and n not in r.get("registered", [])]
            if not pend:
                continue
            picks = r["picks"]
            sets = {"full": [x for x in r["anchors"] if x in by], "h0": [x for x in picks[0::2] if x in by], "h1": [x for x in picks[1::2] if x in by]}
            res = {}
            for tag, A in sets.items():
                if len(A) < 2:
                    continue
                wd = ROOT / "outputs/analysis/ff_stability" / a.run / scene / f"r{r['round']}_{tag}"
                spec = {"images_dir": images_dir, "groups": [A + pend], "model": "pi3x", "points_per_image": 0, "save_depth": True,
                        "repos": cfg.get("repos", {})}
                gr = run_backend(cfg, "feedforward", "pi3x", spec, wd, gpu=a.gpu)["groups"][0]
                z = np.load(gr["depth_path"]); idx = {str(n): i for i, n in enumerate(z["names"])}
                al, how = None, None
                S, D = [], []
                for x in A:
                    s_, d_ = lift(z, idx[x], by[x], rec, {}, 0) if x in idx else (np.zeros((0, 3)), np.zeros((0, 3)))
                    S.append(s_); D.append(d_)
                S, D = np.concatenate(S), np.concatenate(D)
                ap_ = align_points(S, D, scale, np.random.default_rng(0)) if len(S) else None
                if ap_ and ap_[3] >= 0.3:
                    al, how = ap_[:3], "points"
                elif len(A) >= 3:
                    al, how = align_rot(z, idx, A, by), "rot"
                if al is None:
                    continue
                s, R, t = al
                for n in pend:
                    E = z["E"][idx[n]]
                    Rn = E[:, :3] @ R.T                         # cam_from_model rotation
                    Rgt_m = G[n].cam_from_world().rotation.matrix() @ Rg   # cam_from_model per GT (model->GT rotation Rg)
                    cn = -Rn.T @ (s * E[:, 3] - Rn @ t)
                    res.setdefault(n, {})[tag] = {"err": round(rot_angle(Rgt_m @ Rn.T), 1), "how": how, "R": Rn, "c": cn}
            for n, v in res.items():
                row = {"scene": scene[:30], "round": r["round"], "image": n, "status": r["images"][n].get("why"),
                       **{f"err_{k}": v[k]["err"] for k in v}}
                if "h0" in v and "h1" in v:
                    row["gap"] = round(rot_angle(v["h0"]["R"] @ v["h1"]["R"].T), 1)
                    row["gap_frac"] = round(float(np.linalg.norm(v["h0"]["c"] - v["h1"]["c"]) / scale), 3)
                row["how"] = {k: v[k]["how"] for k in v}
                print(json.dumps(row)); out.append(row)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
