"""Export a ray-anchored BA problem from a finished run's grown model (agenticsfm env, pycolmap). Offline test of the
GPU bundle adjustment (third_party/rayba) against COLMAP's `optimize`.

Per scene (largest sub-model of the grow_model artifact):
  images     registered images: pose, SIMPLE_RADIAL intrinsics, size, intrinsics group (optimize's "cluster" grouping)
  matches    verified inlier matches of registered pairs with >= 15 inliers that agree with the current poses
             (Sampson error < --epi-px match pixels); each match gives two observations (both directions)
  depths     one variable per source keypoint: initial depth from the model's 3D point if the keypoint has one, else
             the Pi3X depth scaled to the model, else a two-view triangulation; Pi3X depth + confidence at the keypoint
             as the prior (the image's most confident saved Pi3X call)
  fixed      images without any 3D point in the model (admitted unverified; COLMAP BA leaves them alone too), and the
             image with the most observations (gauge)
No GT is read here.

usage: python scripts/analysis/rayba_prep.py --run p22_vlm_dev --out <dir> [--scenes ...]
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff_offline import find_grow, largest  # noqa: E402
from agenticsfm.tools.optimize import group_images  # noqa: E402

MAX_IMAGE_ID = 2 ** 31 - 1


def cam_params(cam):
    p = list(cam.params); n = cam.model.name
    if n == "SIMPLE_RADIAL":
        return p[0], p[1], p[2], p[3]
    if n == "SIMPLE_PINHOLE":
        return p[0], p[1], p[2], 0.0
    if n in ("PINHOLE",):
        return 0.5 * (p[0] + p[1]), p[2], p[3], 0.0
    if n in ("RADIAL", "OPENCV"):
        f = p[0] if n == "RADIAL" else 0.5 * (p[0] + p[1])
        return (f, p[1], p[2], p[3]) if n == "RADIAL" else (f, p[2], p[3], p[4])
    raise ValueError(n)


def pi3x_depth(gdir):
    """name -> (depth map, conf map, transform (sx, sy, ox, oy)) from the most confident saved Pi3X call."""
    best = {}
    for f in glob.glob(f"{gdir}/_work/**/depth_*.npz", recursive=True):
        z = np.load(f)
        for q, n in enumerate(map(str, z["names"])):
            mc = float(np.asarray(z["conf"][q], dtype=np.float32).mean())
            if n not in best or mc > best[n][0]:
                best[n] = (mc, f, q)
    out = {}
    for n, (_, f, q) in best.items():
        z = np.load(f)
        out[n] = (np.asarray(z["depth"][q], dtype=np.float32), np.asarray(z["conf"][q], dtype=np.float32), np.asarray(z["transforms"][q], dtype=np.float64))
    return out


def sample(dmap, cmap, tr, uv):
    sx, sy, ox, oy = tr
    x = np.round(uv[:, 0] * sx + ox).astype(int); y = np.round(uv[:, 1] * sy + oy).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < dmap.shape[1]) & (y < dmap.shape[0])
    d = np.full(len(uv), np.nan); c = np.zeros(len(uv))
    d[ok] = dmap[y[ok], x[ok]]; c[ok] = cmap[y[ok], x[ok]]
    d[~(d > 0)] = np.nan
    return d, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--epi-px", type=float, default=4.0, help="Sampson error gate, match pixels (long side 1024)")
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--no-prior", action="store_true", help="skip the Pi3X depth (no prior; faster export)")
    ap.add_argument("--obs", choices=["matches", "tracks", "hybrid"], default="matches",
                    help="matches: stars from direct verified matches; tracks: one star per model 3D point (source = its "
                         "first registered observation, residuals to every other observation of the track)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scenes = [json.loads(l)["scene"] for l in open(ROOT / "outputs/runs" / a.run / "results.jsonl")]
    for sc in scenes:
        if a.scenes and sc not in a.scenes:
            continue
        _, cache, gdir = find_grow(a.run, sc)
        t_start = time.time()
        meta = json.load(open(gdir / "meta.json"))
        db = pycolmap.Database.open(str(cache / "db" / f"{meta['inputs']['db']}.db"))
        subs = sorted([d for d in Path(gdir).iterdir() if d.is_dir() and d.name.isdigit()],
                      key=lambda d: -pycolmap.Reconstruction(str(d)).num_reg_images())
        rec = pycolmap.Reconstruction(str(subs[0]))
        reg = sorted(rec.reg_image_ids())
        dbn = {im.image_id: im.name for im in db.read_all_images()}
        assert all(dbn[i] == rec.images[i].name for i in reg), "image ids differ between model and database"
        N = len(reg); pos = {i: n for n, i in enumerate(reg)}
        names = [rec.images[i].name for i in reg]
        R = np.zeros((N, 3, 3)); t = np.zeros((N, 3)); c = np.zeros((N, 2)); WH = np.zeros((N, 2)); f0 = np.zeros(N); k0 = np.zeros(N)
        for n, i in enumerate(reg):
            im = rec.images[i]; T = im.cam_from_world(); cam = rec.cameras[im.camera_id]
            R[n] = T.rotation.matrix(); t[n] = T.translation
            f, cx, cy, k = cam_params(cam); c[n] = (cx, cy); f0[n] = f; k0[n] = k; WH[n] = (cam.width, cam.height)
        sc_px = 1024.0 / WH.max(1)
        groups = group_images(rec, "cluster", 0.08)
        group = np.zeros(N, dtype=np.int64)
        for g, members in enumerate(groups):
            for i in members:
                group[pos[i]] = g
        G = len(groups)
        logf = np.array([np.log(np.median(f0[group == g])) for g in range(G)])
        kk = np.array([np.median(k0[group == g]) for g in range(G)])
        kps = {i: np.asarray(db.read_keypoints(i))[:, :2] for i in reg}
        # model depth of keypoints with a 3D point
        zmod = {}
        for i in reg:
            im = rec.images[i]; T = im.cam_from_world()
            z = np.full(len(kps[i]), np.nan)
            for kidx, p2 in enumerate(im.points2D):
                if p2.has_point3D():
                    X = rec.points3D[p2.point3D_id].xyz
                    z[kidx] = (T.rotation.matrix() @ X + T.translation)[2]
            zmod[i] = z
        nobs3d = np.array([np.isfinite(zmod[i]).sum() for i in reg])
        # verified matches that agree with the current poses
        ids, geoms = db.read_two_view_geometries()
        src_img, src_kp, tgt_img, tgt_kp = [], [], [], []
        kept = tot = 0
        for pid, g in zip(ids, geoms):
            m = np.asarray(g.inlier_matches)
            if m.shape[0] < a.min_inliers:
                continue
            i, j = divmod(int(pid), MAX_IMAGE_ID)
            if i not in pos or j not in pos:
                continue
            tot += len(m)
            ci, cj = rec.cameras[rec.images[i].camera_id], rec.cameras[rec.images[j].camera_id]
            xi = np.asarray(ci.cam_from_img(kps[i][m[:, 0]])); xj = np.asarray(cj.cam_from_img(kps[j][m[:, 1]]))
            rel = rec.images[j].cam_from_world() * rec.images[i].cam_from_world().inverse()
            Rr, tr_ = rel.rotation.matrix(), rel.translation
            E = np.array([[0, -tr_[2], tr_[1]], [tr_[2], 0, -tr_[0]], [-tr_[1], tr_[0], 0]]) @ Rr
            Xi = np.c_[xi, np.ones(len(xi))]; Xj = np.c_[xj, np.ones(len(xj))]
            Ex = Xi @ E.T; Etx = Xj @ E
            num = np.sum(Xj * Ex, 1) ** 2
            den = Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2
            samp = np.sqrt(num / np.maximum(den, 1e-18))                    # normalised units
            fpx = 0.5 * (np.exp(logf[group[pos[i]]]) * sc_px[pos[i]] + np.exp(logf[group[pos[j]]]) * sc_px[pos[j]])
            ok = samp * fpx < a.epi_px
            m = m[ok]; kept += len(m)
            src_img += [pos[i]] * len(m) + [pos[j]] * len(m)
            src_kp += list(m[:, 0]) + list(m[:, 1])
            tgt_img += [pos[j]] * len(m) + [pos[i]] * len(m)
            tgt_kp += list(m[:, 1]) + list(m[:, 0])
        trk = [False] * len(src_img)
        if a.obs in ("tracks", "hybrid"):
            if a.obs == "tracks":
                src_img, src_kp, tgt_img, tgt_kp, trk = [], [], [], [], []
            for p3 in rec.points3D.values():
                els = [(pos[e.image_id], e.point2D_idx) for e in p3.track.elements if e.image_id in pos]
                if len(els) < 2:
                    continue
                (si, sk), rest = els[0], els[1:]
                for ti, tk in rest:
                    src_img.append(si); src_kp.append(sk); tgt_img.append(ti); tgt_kp.append(tk); trk.append(True)
        src_img = np.array(src_img); src_kp = np.array(src_kp); tgt_img = np.array(tgt_img); tgt_kp = np.array(tgt_kp)
        trk = np.array(trk, dtype=bool)
        if a.obs == "hybrid":          # one observation per (source kp, target kp); a track member wins over a plain match
            okey = ((src_img.astype(np.int64) * 100000 + src_kp) * 100000 + tgt_img) * 100000 + tgt_kp
            order = np.lexsort((~trk, okey))
            okey, src_img, src_kp, tgt_img, tgt_kp, trk = okey[order], src_img[order], src_kp[order], tgt_img[order], tgt_kp[order], trk[order]
            first = np.r_[True, okey[1:] != okey[:-1]]
            src_img, src_kp, tgt_img, tgt_kp, trk = src_img[first], src_kp[first], tgt_img[first], tgt_kp[first], trk[first]
        # depth variables: unique (source image, source keypoint)
        key = src_img.astype(np.int64) * 100000 + src_kp
        ukey, p_o = np.unique(key, return_inverse=True)
        img_p = (ukey // 100000).astype(np.int64); kp_p = (ukey % 100000).astype(np.int64)
        P = len(ukey)
        x_p = np.zeros((P, 2)); zinit = np.full(P, np.nan); prior = np.full(P, np.nan); conf = np.zeros(P)
        t_dep = time.time()
        dep = {} if a.no_prior else pi3x_depth(gdir)
        t_dep = time.time() - t_dep
        s0 = np.zeros(N)
        for n, i in enumerate(reg):
            sel = np.nonzero(img_p == n)[0]
            uv = kps[i][kp_p[sel]]
            x_p[sel] = uv
            zinit[sel] = zmod[i][kp_p[sel]]
            if names[n] in dep:
                d, cf = sample(*dep[names[n]], uv)
                prior[sel] = np.log(d); conf[sel] = cf
                both = np.isfinite(prior[sel]) & np.isfinite(zinit[sel]) & (zinit[sel] > 0)
                if both.sum() >= 10:
                    s0[n] = np.median(np.log(zinit[sel][both]) - prior[sel][both])
                    fill = ~(zinit[sel] > 0) & np.isfinite(prior[sel])
                    zinit[sel[fill]] = np.exp(prior[sel][fill] + s0[n])
                med = np.median(cf[cf > 0]) if (cf > 0).any() else 1.0
                conf[sel] = cf / max(med, 1e-6)
        # two-view midpoint triangulation for what is still missing, with the first observation of each depth (batched)
        miss = np.nonzero(~(zinit > 0))[0]
        if len(miss):
            first = np.zeros(P, dtype=np.int64); first[p_o[::-1]] = np.arange(len(p_o))[::-1]
            q = first[miss]
            si, ti_ = img_p[miss], tgt_img[q]
            ri = np.zeros((len(miss), 3)); rj = np.zeros((len(miss), 3))
            for n in np.unique(si):
                m = si == n; cam = rec.cameras[rec.images[reg[n]].camera_id]
                ri[m, :2] = np.asarray(cam.cam_from_img(x_p[miss[m]]))
            for n in np.unique(ti_):
                m = ti_ == n; cam = rec.cameras[rec.images[reg[n]].camera_id]
                rj[m, :2] = np.asarray(cam.cam_from_img(kps[reg[n]][tgt_kp[q[m]]]))
            ri[:, 2] = 1; rj[:, 2] = 1
            Rw = np.transpose(R, (0, 2, 1))                       # world_from_cam rotations
            Cc = -np.einsum("nij,nj->ni", Rw, t)                  # camera centres
            di = np.einsum("nij,nj->ni", Rw[si], ri); dj = np.einsum("nij,nj->ni", Rw[ti_], rj)
            b = Cc[ti_] - Cc[si]
            a11 = (di * di).sum(1); a12 = -(di * dj).sum(1); a22 = (dj * dj).sum(1)
            b1 = (di * b).sum(1); b2 = -(dj * b).sum(1)
            det = a11 * a22 - a12 * a12
            with np.errstate(divide="ignore", invalid="ignore"):
                s1 = (b1 * a22 - a12 * b2) / det; s2 = (a11 * b2 - a12 * b1) / det
            ok = (s1 > 0) & (s2 > 0) & np.isfinite(s1) & (np.abs(det) > 1e-12)
            zinit[miss[ok]] = s1[ok]                            # depth along ri with ri[2] = 1 is the z-depth
        good = zinit > 0
        keep_o = good[p_o]
        remap = -np.ones(P, dtype=np.int64); remap[good] = np.arange(good.sum())
        fixed = nobs3d == 0
        t_build = time.time() - t_start - t_dep
        cnt = np.bincount(src_img[keep_o], minlength=N)
        fixed[int(np.argmax(cnt))] = True
        np.savez_compressed(out / f"{sc}.npz", names=np.array(names), R=R, t=t, c=c, WH=WH, sc=sc_px, group=group,
                            logf=logf, k=kk, fixed=fixed, s0=s0,
                            img_p=img_p[good], kp_p=kp_p[good], x_p=x_p[good], logd=np.log(zinit[good]),
                            prior_logd=prior[good], prior_conf=conf[good],
                            p_o=remap[p_o[keep_o]], tgt=tgt_img[keep_o], xt=kps_concat(kps, reg, tgt_img[keep_o], tgt_kp[keep_o]),
                            trk=trk[keep_o],
                            model_dir=str(subs[0]), t_build=t_build)
        print(f"{sc}: {N} images ({int(fixed.sum())} fixed), {G} intrinsics groups, matches kept {kept}/{tot}, "
              f"{int(good.sum())} depths ({int(np.isfinite(prior[good]).sum())} with Pi3X prior), {int(keep_o.sum())} observations, "
              f"built in {t_build:.1f}s (+{t_dep:.1f}s Pi3X depth)", flush=True)


def kps_concat(kps, reg, tgt_img, tgt_kp):
    xt = np.zeros((len(tgt_img), 2))
    for n, i in enumerate(reg):
        sel = tgt_img == n
        xt[sel] = kps[i][tgt_kp[sel]]
    return xt


if __name__ == "__main__":
    main()
