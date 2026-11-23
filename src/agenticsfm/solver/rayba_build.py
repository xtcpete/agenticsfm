"""Build a ray-anchored BA problem (solver/rayba.py) from a COLMAP model and its match database (numpy + pycolmap;
no torch, runs in the harness env).

Observations ("hybrid", the default):
  tracks   for every 3D point of the model, its first registered observation is the source (one depth variable) and
           every other observation of the track a residual: COLMAP's clean multi-view observations;
  matches  every verified inlier match of a registered pair (>= min_inliers) within epi_px match pixels of the
           current epipolar geometry, both directions; flagged so the solver can add only those that fit the refined
           geometry (rayba.solve_hybrid).
One observation per (source keypoint, target keypoint); a track member wins over a plain match. Depth initialisation:
the model's 3D point when the source keypoint has one, else a two-view midpoint triangulation with its first
observation. Fixed: images without any 3D point (admitted unverified; COLMAP leaves them alone too) and the image with
the most observations (gauge). Intrinsics groups: optimize's "cluster" grouping (image size + focal clusters).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

MAX_IMAGE_ID = 2 ** 31 - 1


def cam_params(cam):
    p = list(cam.params); n = cam.model.name
    if n == "SIMPLE_RADIAL":
        return p[0], p[1], p[2], p[3]
    if n == "SIMPLE_PINHOLE":
        return p[0], p[1], p[2], 0.0
    if n == "PINHOLE":
        return 0.5 * (p[0] + p[1]), p[2], p[3], 0.0
    if n == "RADIAL":
        return p[0], p[1], p[2], p[3]
    if n == "OPENCV":
        return 0.5 * (p[0] + p[1]), p[2], p[3], p[4]
    raise ValueError(f"unsupported camera model {n}")


def track_arrays(rec):
    """All 3D points of rec as arrays, read from COLMAP's binary points3D format (a Python loop over the points' track
    objects takes ~1 min at 600k points): point ids (P,), xyz (P, 3), track lengths (P,), and the concatenated track
    elements in track order: image ids (E,), keypoint indices (E,)."""
    import shutil
    import struct
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    try:
        rec.write_binary(str(d))
        buf = (d / "points3D.bin").read_bytes()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    n = struct.unpack_from("<Q", buf, 0)[0]
    offs = np.empty(n, dtype=np.int64); lens = np.empty(n, dtype=np.int64)
    off = 8
    unpack = struct.Struct("<Q").unpack_from
    for k in range(n):                       # id u64, xyz 3 f64, rgb 3 u8, error f64, track length u64, elements
        offs[k] = off
        L = unpack(buf, off + 43)[0]
        lens[k] = L
        off += 51 + 8 * L
    raw = np.frombuffer(buf, dtype=np.uint8)
    take = lambda o, nb: raw[o[:, None] + np.arange(nb)]  # noqa: E731
    pid = take(offs, 8).copy().view("<u8")[:, 0].astype(np.int64)
    xyz = take(offs + 8, 24).copy().view("<f8")
    eo = np.repeat(offs + 51, lens) + 8 * (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    el = take(eo, 8).copy().view("<u4")
    return pid, xyz, lens, el[:, 0].astype(np.int64), el[:, 1].astype(np.int64)


def _by(idx, N):
    """Positions of idx grouped by value: order[bounds[n]:bounds[n + 1]] are the positions where idx == n (one sort
    instead of a scan per image, which is quadratic at thousands of images)."""
    order = np.argsort(idx, kind="stable")
    return order, np.searchsorted(idx[order], np.arange(N + 1))


def build(rec, database, obs: str = "hybrid", min_inliers: int = 15, epi_px: float = 4.0,
          group_tol: float = 0.08, grouping: str = "cluster", kps: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    """grouping: "cluster" (optimize's image size + focal clusters) or "camera" (one intrinsics group per camera of
    the model, as COLMAP's bundle adjustment refines them). kps: keypoints by image id, read from the database when
    missing (a caller that builds often passes its cache)."""
    from agenticsfm.tools.optimize import group_images

    t0 = time.time()
    reg = sorted(rec.reg_image_ids())
    N = len(reg); pos = {i: n for n, i in enumerate(reg)}
    names = [rec.images[i].name for i in reg]
    R = np.zeros((N, 3, 3)); t = np.zeros((N, 3)); c = np.zeros((N, 2)); WH = np.zeros((N, 2)); f0 = np.zeros(N); k0 = np.zeros(N)
    for n, i in enumerate(reg):
        im = rec.images[i]; T = im.cam_from_world(); cam = rec.cameras[im.camera_id]
        R[n] = T.rotation.matrix(); t[n] = T.translation
        f, cx, cy, k = cam_params(cam); c[n] = (cx, cy); f0[n] = f; k0[n] = k; WH[n] = (cam.width, cam.height)
    sc = 1024.0 / WH.max(1)
    if grouping == "camera":
        by_cam: dict[int, list[int]] = {}
        for i in reg:
            by_cam.setdefault(rec.images[i].camera_id, []).append(i)
        groups = list(by_cam.values())
    else:
        groups = group_images(rec, "cluster", group_tol)
    group = np.zeros(N, dtype=np.int64)
    for g, members in enumerate(groups):
        for i in members:
            group[pos[i]] = g
    G = len(groups)
    logf = np.array([np.log(np.median(f0[group == g])) for g in range(G)])
    kk = np.array([np.median(k0[group == g]) for g in range(G)])
    kps = {i: kps[i] if kps is not None and i in kps else np.asarray(database.read_keypoints(i))[:, :2] for i in reg}
    # tracks: model depth of every keypoint with a 3D point, and the track observations
    zmod = {i: np.full(len(kps[i]), np.nan) for i in reg}
    pid_a, xyz_a, lens, e_img, e_kp = track_arrays(rec)
    lut = np.full(int(max(max(reg), e_img.max(initial=0))) + 1, -1, dtype=np.int64)
    lut[np.array(reg)] = np.arange(N)
    e_pt = np.repeat(np.arange(len(pid_a)), lens)
    e_n = lut[e_img]
    keep = e_n >= 0
    e_pt, e_n, e_kp = e_pt[keep], e_n[keep], e_kp[keep]
    dep = np.einsum("ej,ej->e", R[e_n, 2], xyz_a[e_pt]) + t[e_n, 2]
    o_, b_ = _by(e_n, N)
    for n, i in enumerate(reg):
        m = o_[b_[n]:b_[n + 1]]
        zmod[i][e_kp[m]] = dep[m]
    cnt = np.bincount(e_pt, minlength=len(pid_a))
    first = np.r_[True, e_pt[1:] != e_pt[:-1]]                  # first registered element of each point = its source
    multi = cnt[e_pt] >= 2
    src_e = np.nonzero(first & multi)[0]
    pt_ids = pid_a[e_pt[src_e]]
    rest = np.nonzero(~first & multi)[0]
    srcpos = np.full(len(pid_a), -1, dtype=np.int64); srcpos[e_pt[src_e]] = src_e
    s_of = srcpos[e_pt[rest]]
    nobs3d = np.array([np.isfinite(zmod[i]).sum() for i in reg])
    src_l, skp_l, tgt_l, tkp_l = [e_n[s_of]], [e_kp[s_of]], [e_n[rest]], [e_kp[rest]]
    trk_l = [np.ones(len(rest), dtype=bool)]
    kept = tot = 0
    if obs in ("hybrid", "matches"):
        ids, geoms = database.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            m = np.asarray(g.inlier_matches)
            if m.shape[0] < min_inliers:
                continue
            i, j = divmod(int(pid), MAX_IMAGE_ID)
            if i not in pos or j not in pos:
                continue
            tot += len(m)
            ci, cj = rec.cameras[rec.images[i].camera_id], rec.cameras[rec.images[j].camera_id]
            xi = np.asarray(ci.cam_from_img(kps[i][m[:, 0]])); xj = np.asarray(cj.cam_from_img(kps[j][m[:, 1]]))
            ni, nj = pos[i], pos[j]
            Rr = R[nj] @ R[ni].T; tr = t[nj] - Rr @ t[ni]
            E = np.array([[0, -tr[2], tr[1]], [tr[2], 0, -tr[0]], [-tr[1], tr[0], 0]]) @ Rr
            Xi = np.c_[xi, np.ones(len(xi))]; Xj = np.c_[xj, np.ones(len(xj))]
            Ex = Xi @ E.T; Etx = Xj @ E
            samp = np.sqrt(np.sum(Xj * Ex, 1) ** 2 / np.maximum(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2, 1e-18))
            fpx = 0.5 * (np.exp(logf[group[ni]]) * sc[ni] + np.exp(logf[group[nj]]) * sc[nj])
            m = m[samp * fpx < epi_px]; kept += len(m)
            L = len(m)
            src_l += [np.full(L, ni), np.full(L, nj)]; skp_l += [m[:, 0], m[:, 1]]
            tgt_l += [np.full(L, nj), np.full(L, ni)]; tkp_l += [m[:, 1], m[:, 0]]
            trk_l += [np.zeros(2 * L, dtype=bool)]
    src_img = np.concatenate(src_l); src_kp = np.concatenate(skp_l).astype(np.int64)
    tgt_img = np.concatenate(tgt_l); tgt_kp = np.concatenate(tkp_l).astype(np.int64); trk = np.concatenate(trk_l)
    if obs == "hybrid":
        nk = int(max(len(v) for v in kps.values())) + 1
        okey = ((src_img * nk + src_kp) * N + tgt_img) * nk + tgt_kp      # < 2^63 up to ~10^5 images of 10^4 keypoints
        order = np.lexsort((~trk, okey))
        okey = okey[order]
        first = np.r_[True, okey[1:] != okey[:-1]]
        sel = order[first]
        src_img, src_kp, tgt_img, tgt_kp, trk = src_img[sel], src_kp[sel], tgt_img[sel], tgt_kp[sel], trk[sel]
    key = src_img * 100000 + src_kp
    ukey, p_o = np.unique(key, return_inverse=True)
    img_p = ukey // 100000; kp_p = ukey % 100000
    P = len(ukey)
    x_p = np.zeros((P, 2)); zinit = np.full(P, np.nan)
    o_, b_ = _by(img_p, N)
    for n, i in enumerate(reg):
        sel = o_[b_[n]:b_[n + 1]]
        x_p[sel] = kps[i][kp_p[sel]]
        zinit[sel] = zmod[i][kp_p[sel]]
    miss = np.nonzero(~(zinit > 0))[0]
    if len(miss):
        first = np.zeros(P, dtype=np.int64); first[p_o[::-1]] = np.arange(len(p_o))[::-1]
        q = first[miss]
        si, tj = img_p[miss], tgt_img[q]
        ri = np.ones((len(miss), 3)); rj = np.ones((len(miss), 3))
        o_, b_ = _by(si, N)
        for n in np.nonzero(np.diff(b_))[0]:
            m = o_[b_[n]:b_[n + 1]]
            ri[m, :2] = np.asarray(rec.cameras[rec.images[reg[n]].camera_id].cam_from_img(x_p[miss[m]]))
        o_, b_ = _by(tj, N)
        for n in np.nonzero(np.diff(b_))[0]:
            m = o_[b_[n]:b_[n + 1]]
            rj[m, :2] = np.asarray(rec.cameras[rec.images[reg[n]].camera_id].cam_from_img(kps[reg[n]][tgt_kp[q[m]]]))
        Rw = np.transpose(R, (0, 2, 1)); Cc = -np.einsum("nij,nj->ni", Rw, t)
        di = np.einsum("nij,nj->ni", Rw[si], ri); dj = np.einsum("nij,nj->ni", Rw[tj], rj)
        b = Cc[tj] - Cc[si]
        a11 = (di * di).sum(1); a12 = -(di * dj).sum(1); a22 = (dj * dj).sum(1)
        b1 = (di * b).sum(1); b2 = -(dj * b).sum(1)
        det = a11 * a22 - a12 * a12
        with np.errstate(divide="ignore", invalid="ignore"):
            s1 = (b1 * a22 - a12 * b2) / det; s2 = (a11 * b2 - a12 * b1) / det
        ok = (s1 > 0) & (s2 > 0) & np.isfinite(s1) & (np.abs(det) > 1e-12)
        zinit[miss[ok]] = s1[ok]
    good = zinit > 0
    keep_o = good[p_o]
    remap = -np.ones(P, dtype=np.int64); remap[good] = np.arange(good.sum())
    fixed = nobs3d == 0
    cnt = np.bincount(src_img[keep_o], minlength=N)
    fixed[int(np.argmax(cnt))] = True
    tg = tgt_img[keep_o]; tk = tgt_kp[keep_o]
    xt = np.zeros((len(tg), 2))
    o_, b_ = _by(tg, N)
    for n, i in enumerate(reg):
        s_ = o_[b_[n]:b_[n + 1]]
        xt[s_] = kps[i][tk[s_]]
    # depth variable of each track's source (-1 if dropped)
    sk_ = e_n[src_e] * 100000 + e_kp[src_e]                     # ukey is sorted (np.unique)
    at = np.minimum(np.searchsorted(ukey, sk_), max(len(ukey) - 1, 0))
    pt_p = np.where((len(ukey) > 0) & (ukey[at] == sk_), remap[at], -1).astype(np.int64)
    return dict(names=np.array(names), image_ids=np.array(reg), R=R, t=t, c=c, WH=WH, sc=sc, group=group, logf=logf, k=kk,
                fixed=fixed, img_p=img_p[good], x_p=x_p[good], logd=np.log(zinit[good]),
                prior_logd=np.full(int(good.sum()), np.nan), prior_conf=np.zeros(int(good.sum())), s0=np.zeros(N),
                p_o=remap[p_o[keep_o]], tgt=tg, xt=xt, trk=trk[keep_o], pt_ids=np.array(pt_ids, dtype=np.int64), pt_p=pt_p,
                t_build=time.time() - t0, stats=dict(images=N, groups=G, matches_kept=kept, matches_total=tot,
                                                       depths=int(good.sum()), observations=int(keep_o.sum())))
