"""P5 offline test: bundle adjustment with a Pi3X depth prior (benchmark layer: GT only for scoring).

Refines a submitted model (poses + points; intrinsics fixed) with scipy's sparse least squares:
  reprojection   (projection - keypoint) / sigma_px                         for every observation
  depth prior    (log z_cam - log d_pi3x(u, v) - b_image) / sigma_depth     for every observation with Pi3X depth
b_image is a free per-image log scale (Pi3X depth is relative). sigma_depth = inf is plain bundle adjustment with the
same solver (the control). Each variant is scored with the benchmark evaluator (auc_all@3/10) and GT rotation errors.
Pi3X depth: the latest saved depth map of each image in the run's Pi3X calls (_work/scene*/ff_*/depth_*.npz).

usage: python scripts/analysis/depth_ba.py --run ff_vlm_dev [--scenes ...] [--sigmas inf 0.2 0.1 0.05]
"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ff_offline import find_grow, largest, gt_errors, align_gt  # noqa: E402


def distort(model, params, x, y):
    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        return x, y
    r2 = x * x + y * y
    if model == "SIMPLE_RADIAL":
        k = params[3]; return x * (1 + k * r2), y * (1 + k * r2)
    if model == "RADIAL":
        k1, k2 = params[3], params[4]; f = 1 + k1 * r2 + k2 * r2 * r2; return x * f, y * f
    if model == "OPENCV":
        k1, k2, p1, p2 = params[4:8]; f = 1 + k1 * r2 + k2 * r2 * r2
        return x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    raise ValueError(model)


def intr(model, p):
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


def depth_maps(mdir):
    """name -> (depth, conf, transform) from the latest Pi3X call that included the image."""
    out = {}
    files = sorted(mdir.glob("_work/scene*/ff_*/depth_*.npz"), key=lambda p: p.stat().st_mtime)
    for f in files:
        z = np.load(f)
        for r, n in enumerate(z["names"]):
            out[str(n)] = (z["depth"][r].astype(np.float32), z["conf"][r].astype(np.float32), z["transforms"][r])
    return out


def refine(rec, dmaps, sigma_d, sigma_px=1.0, iters=30):
    imgs = [im for im in rec.images.values() if im.has_pose]
    iid = {im.image_id: k for k, im in enumerate(imgs)}
    pids = [p for p, P in rec.points3D.items() if P.track.length() >= 2]
    pidx = {p: k for k, p in enumerate(pids)}
    ci, pi, uv, dz, cf = [], [], [], [], []
    for im in imgs:
        dm = dmaps.get(im.name)
        for p2 in im.points2D:
            if p2.has_point3D() and p2.point3D_id in pidx:
                ci.append(iid[im.image_id]); pi.append(pidx[p2.point3D_id]); uv.append(p2.xy)
                d = np.nan
                if dm is not None:
                    depth, conf, (sx, sy, ox, oy) = dm
                    u, v = int(round(p2.xy[0] * sx + ox)), int(round(p2.xy[1] * sy + oy))
                    if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] and depth[v, u] > 0 and conf[v, u] >= 0.1:
                        d = float(depth[v, u])
                dz.append(d)
    ci, pi, uv, dz = np.array(ci), np.array(pi), np.array(uv), np.array(dz)
    nC, nP, nO = len(imgs), len(pids), len(ci)
    use_d = np.isfinite(sigma_d)
    hasd = np.isfinite(dz) & use_d
    di = np.where(hasd)[0]
    cams = [rec.cameras[im.camera_id] for im in imgs]
    models = [c.model.name for c in cams]; params = [np.array(c.params, float) for c in cams]
    x0_c = np.concatenate([np.r_[Rotation.from_matrix(im.cam_from_world().rotation.matrix()).as_rotvec(), im.cam_from_world().translation] for im in imgs])
    x0_p = np.concatenate([rec.points3D[p].xyz for p in pids])
    # initial per-image log scale: median log(z / d)
    b0 = np.zeros(nC)
    if use_d:
        Rs = np.stack([im.cam_from_world().rotation.matrix() for im in imgs]); ts = np.stack([im.cam_from_world().translation for im in imgs])
        X = x0_p.reshape(-1, 3)
        z = np.einsum("nj,nj->n", Rs[ci[di]][:, 2, :], X[pi[di]]) + ts[ci[di]][:, 2]
        lr = np.log(np.maximum(z, 1e-9)) - np.log(dz[di])
        for c in range(nC):
            m = ci[di] == c
            if m.any():
                b0[c] = np.median(lr[m])
    x0 = np.r_[x0_c, x0_p, b0]
    fx, fy, cx, cy = np.array([intr(m_, p_) for m_, p_ in zip(models, params)]).T
    groups = {}
    for c, m_ in enumerate(models):
        groups.setdefault(m_, []).append(c)

    def fun(x):
        C = x[:6 * nC].reshape(nC, 6); P = x[6 * nC:6 * nC + 3 * nP].reshape(nP, 3); b = x[6 * nC + 3 * nP:]
        R = Rotation.from_rotvec(C[:, :3]).as_matrix()
        Xc = np.einsum("nij,nj->ni", R[ci], P[pi]) + C[ci, 3:]
        z = Xc[:, 2]; zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        xn, yn = Xc[:, 0] / zs, Xc[:, 1] / zs
        u = np.empty(nO); v = np.empty(nO)
        for m_, cs in groups.items():
            sel = np.isin(ci, cs)
            for c in cs:
                pass
            # per-camera distortion params differ: loop cameras of this model in one pass
            for c in cs:
                s = ci == c
                xd, yd = distort(m_, params[c], xn[s], yn[s])
                u[s] = fx[c] * xd + cx[c]; v[s] = fy[c] * yd + cy[c]
        r = [((u - uv[:, 0]) / sigma_px), ((v - uv[:, 1]) / sigma_px)]
        if use_d:
            r.append((np.log(np.maximum(z[di], 1e-9)) - np.log(dz[di]) - b[ci[di]]) / sigma_d)
        return np.concatenate(r)

    nR = 2 * nO + (len(di) if use_d else 0)
    S = lil_matrix((nR, len(x0)), dtype=np.int8)
    rows = np.arange(nO)
    for k in range(6):
        S[rows, 6 * ci + k] = 1; S[nO + rows, 6 * ci + k] = 1
    for k in range(3):
        S[rows, 6 * nC + 3 * pi + k] = 1; S[nO + rows, 6 * nC + 3 * pi + k] = 1
    if use_d:
        rd = 2 * nO + np.arange(len(di))
        for k in range(6):
            S[rd, 6 * ci[di] + k] = 1
        for k in range(3):
            S[rd, 6 * nC + 3 * pi[di] + k] = 1
        S[rd, 6 * nC + 3 * nP + ci[di]] = 1
    res = least_squares(fun, x0, jac_sparsity=S.tocsr(), method="trf", loss="soft_l1", f_scale=2.0, max_nfev=iters, x_scale="jac", tr_solver="lsmr", verbose=0)
    x = res.x
    C = x[:6 * nC].reshape(nC, 6); P = x[6 * nC:6 * nC + 3 * nP].reshape(nP, 3)
    out = pycolmap.Reconstruction(rec)
    for k, im in enumerate(imgs):
        R = Rotation.from_rotvec(C[k, :3]).as_matrix()
        out.frame(out.images[im.image_id].frame_id).rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), C[k, 3:])
    for k, p in enumerate(pids):
        out.points3D[p].xyz = P[k]
    return out, {"obs": nO, "depth_obs": int(len(di)) if use_d else 0, "cost0": float(0.5 * np.sum(fun(x0) ** 2)), "cost": float(res.cost), "nfev": res.nfev}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--sigmas", nargs="*", type=float, default=[float("inf"), 0.2, 0.1, 0.05]); ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    from agenticsfm.config import load_config
    from agenticsfm.datasets import load_benchmark
    from agenticsfm.evaluation import Evaluator
    cfg = load_config(); B = {b.name: b for b in load_benchmark(cfg, "wriva", "all")}
    ev = Evaluator.from_config(cfg)
    res_all = {}
    for d in sorted((ROOT / "outputs/runs" / a.run).iterdir()):
        if not (d / "final").is_dir() or (a.scenes and not any(s in d.name for s in a.scenes)):
            continue
        rd, store, mdir = find_grow(a.run, d.name)
        dm = depth_maps(mdir)
        subs = sorted([s for s in (d / "final").iterdir() if s.is_dir() and any((s / f).exists() for f in ("images.bin", "images.txt"))], key=lambda s: -pycolmap.Reconstruction(str(s)).num_reg_images())
        gt = pycolmap.Reconstruction(str(B[d.name].gt_dir))
        rows = {}
        for sg in [None] + a.sigmas:
            tmp = Path(tempfile.mkdtemp(prefix="dba_"))
            info = {}
            for k, s in enumerate(subs):
                rec = pycolmap.Reconstruction(str(s))
                if sg is not None and k == 0:
                    rec, info = refine(rec, dm, sg, iters=a.iters)
                (tmp / str(k)).mkdir(parents=True); rec.write(str(tmp / str(k)))
            m = ev.evaluate(B[d.name].gt_dir, [tmp / str(k) for k in range(len(subs))])
            e = align_gt(pycolmap.Reconstruction(str(tmp / "0")), gt); ev_ = np.array(list(e.values()))
            key = "submitted" if sg is None else (f"BA plain" if not np.isfinite(sg) else f"BA + depth sigma {sg}")
            rows[key] = {"auc_all@3": round(100 * m["auc_all@3"], 1), "auc_all@10": round(100 * m["auc_all@10"], 1),
                         "rot_med": round(float(np.median(ev_)), 2), "over5": int((ev_ > 5).sum()), **info}
            shutil.rmtree(tmp, ignore_errors=True)
            print(d.name[-24:], key, rows[key], flush=True)
        res_all[d.name] = rows
    out = ROOT / "outputs/analysis" / f"depth_ba_{a.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res_all, indent=1))


if __name__ == "__main__":
    main()
