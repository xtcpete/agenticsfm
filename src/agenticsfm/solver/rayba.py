"""Ray-anchored bundle adjustment on the GPU (torch), after DMap (ICLR 2027 submission) and MP-SfM's depth priors.

Every source keypoint k of image i owns one depth d_ik along its ray. Each verified match (i, k) -> (j, l) gives one
reprojection residual: the point d_ik * ray_i(k), moved from camera i to camera j, projected, minus the matched pixel
x_jl. No tracks are built: a keypoint matched in several images forms a star around its source ray, and both
directions of a match are separate observations with separate depths.

Unknowns: per image a pose (left perturbation of cam_from_world: rotation 3, translation 3) and a log-depth scale s_i
used only by the depth prior; per intrinsics group the camera parameters (principal point fixed):
  simple_radial  [log f, k1]
  opencv         [log fx, log fy, k1, k2, p1, p2]   (COLMAP's OPENCV model)
and per source keypoint a log depth. Each depth couples only with the cameras of its star, so the depth block of the
normal equations is diagonal and is eliminated by the Schur complement; the reduced camera system (7 per image + K per
intrinsics group) is solved by Cholesky (dense, small problems) or by preconditioned CG on the implicit Schur
complement (Options.solver; scales to thousands of images). Levenberg-Marquardt with damping of the whole system, Cauchy loss.

Optional depth prior (MP-SfM, DMap): (log d_ik - log D_ik - s_i) with weight lambda * conf_ik, Cauchy-robust, where D is
the feedforward depth at the keypoint and s_i a free per-image log scale (feedforward depth is relative).

Residuals are measured in "match pixels": pixels of the image resized to a long side of 1024, the resolution the
keypoints were matched at, so one unit means the same thing on every image size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch.func import jacfwd, vmap

BASE_PARAMS = {"simple_radial": 2, "opencv": 6}
# "<base>_pp" adds a principal-point offset (2, pixels) shared by the intrinsics group
CAMERA_PARAMS = {**BASE_PARAMS, **{f"{m}_pp": k + 2 for m, k in BASE_PARAMS.items()}}


def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)[..., None]
    K = hat(w / th[..., 0])
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


# ---- camera models: params p (K,), normalised coordinates <-> pixels (principal point c separate) ---------------------
def _distort(model, x, p):
    if model == "simple_radial":
        return x * (1 + p[1] * (x * x).sum())
    k1, k2, p1, p2 = p[2], p[3], p[4], p[5]
    u, v = x[0], x[1]
    r2 = u * u + v * v
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return torch.stack([u * rad + 2 * p1 * u * v + p2 * (r2 + 2 * u * u), v * rad + p1 * (r2 + 2 * v * v) + 2 * p2 * u * v])


def _focal(model, p):
    return torch.exp(p[0:1]).expand(2) if model == "simple_radial" else torch.exp(p[0:2])


def _focal_batch(model, P):
    return torch.exp(P[:, 0]) if model == "simple_radial" else torch.exp(P[:, 0:2]).mean(1)


def _undistort(model, xd, p, iters=10):
    xu = xd
    for _ in range(iters):
        xu = xu + (xd - _distort(model, xu, p))     # fixed point; converges for the mild distortions of SfM cameras
    return xu


def make_residual(model):
    K = CAMERA_PARAMS[model]
    pp = model.endswith("_pp")
    base = model[:-3] if pp else model

    def residual(delta, Ri, ti, Rj, tj, pi, ci, pj, cj, logd, xs, xt, sc):
        """One observation. delta (13 + 2K): rot_i, trans_i, rot_j, trans_j, intr_i (K), intr_j (K), dlogd.
        First order in the rotation perturbations: exact Jacobian at delta = 0."""
        pi_ = pi + delta[12:12 + K]
        pj_ = pj + delta[12 + K:12 + 2 * K]
        if pp:
            ci = ci + pi_[K - 2:K]; cj = cj + pj_[K - 2:K]
        model_ = base
        xu = _undistort(model_, (xs - ci) / _focal(model_, pi_), pi_)
        d = torch.exp(logd + delta[12 + 2 * K])
        Xci = d * torch.cat([xu, torch.ones_like(xu[:1])])
        Ri_ = Ri + hat(delta[0:3]) @ Ri
        Xw = Ri_.T @ (Xci - ti - delta[3:6])
        Rj_ = Rj + hat(delta[6:9]) @ Rj
        Xc = Rj_ @ Xw + tj + delta[9:12]
        z = Xc[2]
        xd = _distort(model_, Xc[:2] / z, pj_)
        r = (_focal(model_, pj_) * xd + cj - xt) * sc
        return r, torch.stack([r[0], r[1], z])

    jac = vmap(jacfwd(residual, argnums=0, has_aux=True), in_dims=(None,) + (0,) * 12)
    res = vmap(residual, in_dims=(None,) + (0,) * 12)
    return residual, jac, res


_FNS = {m: make_residual(m) for m in CAMERA_PARAMS}


def make_parts(model):
    """Pieces of the residual for the chain-rule Jacobian: the source ray (undistorted normalised coordinates) as a
    function of the source intrinsics, and the projection of a camera-frame point with the target intrinsics."""
    K = CAMERA_PARAMS[model]
    pp = model.endswith("_pp")
    base = model[:-3] if pp else model

    def ray(p, x, c):
        if pp:
            c = c + p[K - 2:K]
        return _undistort(base, (x - c) / _focal(base, p), p)

    def gap(xu, p, x, c):
        """distort(xu) - normalised pixel: zero at the undistorted ray (implicit function for its derivative)."""
        if pp:
            c = c + p[K - 2:K]
        return _distort(base, xu, p) - (x - c) / _focal(base, p)

    def proj(X, p, c, xt, sc):
        if pp:
            c = c + p[K - 2:K]
        xd = _distort(base, X[:2] / X[2], p)
        r = (_focal(base, p) * xd + c - xt) * sc
        return r, r

    gap_j = vmap(jacfwd(gap, argnums=(0, 1)))
    ray_v = vmap(ray)

    def ray_j(xu, p, x, c):
        """d xu / d p by the implicit function theorem: -(dG/dxu)^-1 dG/dp (no differentiation through iterations)."""
        Gx, Gp = gap_j(xu, p, x, c)
        return -torch.linalg.solve(Gx, Gp)

    proj_j = vmap(jacfwd(proj, argnums=(0, 1), has_aux=True))
    return ray_v, ray_j, proj_j


_PARTS = {m: make_parts(m) for m in CAMERA_PARAMS}


@dataclass
class Problem:
    """All tensors on one device. Images 0..N-1, intrinsics groups 0..G-1, depth variables 0..P-1, observations
    0..M-1 sorted by depth variable."""
    R: torch.Tensor          # (N, 3, 3) cam_from_world rotation
    t: torch.Tensor          # (N, 3)
    c: torch.Tensor          # (N, 2) principal point, fixed
    sc: torch.Tensor         # (N,) pixel -> match-pixel factor (1024 / long side)
    group: torch.Tensor      # (N,) intrinsics group
    intr: torch.Tensor       # (G, K) camera parameters of the model
    fixed: torch.Tensor      # (N,) bool: pose held constant
    img_p: torch.Tensor      # (P,) source image of each depth variable
    x_p: torch.Tensor        # (P, 2) source pixel
    logd: torch.Tensor       # (P,)
    prior_logd: torch.Tensor  # (P,) feedforward log depth, nan if none
    prior_conf: torch.Tensor  # (P,) feedforward confidence, normalised to median 1 per image
    p_o: torch.Tensor        # (M,) depth variable of each observation (sorted)
    tgt: torch.Tensor        # (M,) target image
    xt: torch.Tensor         # (M, 2) target pixel
    model: str = "simple_radial"
    s: torch.Tensor = None   # (N,) per-image log-depth scale of the prior

    def __post_init__(self):
        if self.s is None:
            self.s = torch.zeros(self.R.shape[0], dtype=self.R.dtype, device=self.R.device)

    @staticmethod
    def intr_from_simple_radial(logf, k, model):
        z = torch.zeros_like(logf)
        base = model[:-3] if model.endswith("_pp") else model
        cols = [logf, k] if base == "simple_radial" else [logf, logf, k, z, z, z]
        if model.endswith("_pp"):
            cols += [z, z]
        return torch.stack(cols, 1)


@dataclass
class Options:
    iters: int = 20
    cauchy: float = 1.0           # Cauchy scale, match pixels
    prior_weight: float = 0.0     # lambda; 0 = no depth prior
    prior_cauchy: float = 0.2     # Cauchy scale of the log-depth prior
    refine_intrinsics: bool = True
    chunk: int = 200_000
    lm_init: float = 1e-4
    min_rel_decrease: float = 1e-6
    chain: bool = True            # chain-rule Jacobian (fast); False: full forward-mode autodiff (reference)
    solver: str = "auto"          # "dense": reduced camera system formed and factored (Cholesky); "sparse": its blocks
                                  # kept per image pair + block-Jacobi preconditioned CG (no C x C matrix; many images);
                                  # "implicit": S v from the per-observation terms (least memory, slow);
                                  # "auto": dense while the reduced system has <= dense_max unknowns, else sparse
    dense_max: int = 8000
    pcg_tol: float = 1e-6         # relative residual of the reduced system
    pcg_max: int = 500
    precond: str = "auto"         # "chol32": float32 Cholesky of the reduced system (half the memory of the dense
                                  # float64 path, CG then needs a few iterations); "jacobi": its diagonal blocks;
                                  # "auto": chol32 while the reduced system has <= chol32_max unknowns
    chol32_max: int = 64000
    refactor_its: int = 20        # chol32: keep the factor of an earlier step (damping, linearisation) until CG needs
                                  # more iterations than this
    verbose: bool = False
    log: list = field(default_factory=list)


class RayBA:
    def __init__(self, pb: Problem, opt: Options):
        self.pb, self.opt = pb, opt
        self.K = CAMERA_PARAMS[pb.model]
        self.res_fn, self.jac_fn, self.res_v = _FNS[pb.model]
        self.N, self.G, self.P, self.M = pb.R.shape[0], pb.intr.shape[0], pb.logd.shape[0], pb.p_o.shape[0]
        K = self.K
        self.C = 7 * self.N + K * self.G
        self.D = 13 + 2 * K                                   # perturbation size of one observation
        self.nS, self.nT = 7 + K, 6 + K                       # source / target block sizes
        dev = pb.R.device
        self.count = torch.bincount(pb.p_o, minlength=self.P)
        self.start = torch.cumsum(self.count, 0) - self.count
        self.src_o = pb.img_p[pb.p_o]
        key = self.src_o * self.N + pb.tgt
        self.pair_keys, self.pair_o = torch.unique(key, return_inverse=True)
        self.has_prior = torch.isfinite(pb.prior_logd)
        self._pairs = None
        idx = torch.arange(self.N, device=dev)
        gk = 7 * self.N + K * pb.group[:, None] + torch.arange(K, device=dev)
        self.src_idx = torch.cat([7 * idx[:, None] + torch.arange(7, device=dev), gk], 1)   # (N, 7 + K)
        self.tgt_idx = torch.cat([7 * idx[:, None] + torch.arange(6, device=dev), gk], 1)   # (N, 6 + K)

    # ---------------------------------------------------------------------------------------------------------------
    def _consts(self, sl):
        pb = self.pb
        p = pb.p_o[sl]; i = self.src_o[sl]; j = pb.tgt[sl]
        return (pb.R[i], pb.t[i], pb.R[j], pb.t[j], pb.intr[pb.group[i]], pb.c[i], pb.intr[pb.group[j]], pb.c[j],
                pb.logd[p], pb.x_p[p], pb.xt[sl], pb.sc[j])

    def _rays(self):
        """Per depth variable: undistorted ray (P, 2) and its derivative w.r.t. the source intrinsics (P, 2, K)."""
        pb, o = self.pb, self.opt
        ray_v, ray_j, _ = _PARTS[pb.model]
        P = self.P
        xu = torch.empty(P, 2, dtype=pb.R.dtype, device=pb.R.device)
        dxu = torch.empty(P, 2, self.K, dtype=pb.R.dtype, device=pb.R.device)
        for a in range(0, P, o.chunk):
            sl = slice(a, min(P, a + o.chunk))
            pi = pb.intr[pb.group[pb.img_p[sl]]]
            xu[sl] = ray_v(pi, pb.x_p[sl], pb.c[pb.img_p[sl]])
            dxu[sl] = ray_j(xu[sl], pi, pb.x_p[sl], pb.c[pb.img_p[sl]])
        return xu, dxu

    def _jacobian(self, sl, xu, dxu):
        """Chain rule: analytic in the poses and the depth, autodiff only through the projection (3 + K directions).
        Same layout as the full forward-mode Jacobian: (B, 2, 13 + 2K), aux (B, 3) = residual, depth in target."""
        pb, K = self.pb, self.K
        _, _, proj_j = _PARTS[pb.model]
        p = pb.p_o[sl]; i = self.src_o[sl]; j = pb.tgt[sl]
        Ri, ti, Rj, tj = pb.R[i], pb.t[i], pb.R[j], pb.t[j]
        d = torch.exp(pb.logd[p])
        ray = torch.cat([xu[p], torch.ones_like(xu[p][:, :1])], 1)
        Xci = d[:, None] * ray
        Rij = Rj @ Ri.transpose(1, 2)
        a = Xci - ti
        Xw = (Ri.transpose(1, 2) @ a[:, :, None])[:, :, 0]
        RXw = (Rj @ Xw[:, :, None])[:, :, 0]
        Xc = RXw + tj
        (JX, Jp), r = proj_j(Xc, pb.intr[pb.group[j]], pb.c[j], pb.xt[sl], pb.sc[j])        # (B,2,3), (B,2,K)
        B = Xc.shape[0]
        J = torch.empty(B, 2, self.D, dtype=Xc.dtype, device=Xc.device)
        JR = JX @ Rij
        J[:, :, 0:3] = JR @ hat(a)                                   # rotation of camera i
        J[:, :, 3:6] = -JR                                           # translation of camera i
        J[:, :, 6:9] = -(JX @ hat(RXw))                              # rotation of camera j
        J[:, :, 9:12] = JX                                           # translation of camera j
        dX = torch.cat([dxu[p], torch.zeros_like(dxu[p][:, :1])], 1) * d[:, None, None]   # (B, 3, K)
        J[:, :, 12:12 + K] = JR @ dX                                 # source intrinsics (through the ray)
        J[:, :, 12 + K:12 + 2 * K] = Jp                              # target intrinsics
        J[:, :, 12 + 2 * K] = (JR @ Xci[:, :, None])[:, :, 0]        # log depth
        return J, torch.cat([r, Xc[:, 2:3]], 1)

    def _zero(self):
        return torch.zeros(self.D, dtype=self.pb.R.dtype, device=self.pb.R.device)

    def cost(self):
        pb, o = self.pb, self.opt
        zero = self._zero()
        total = torch.zeros((), dtype=torch.float64, device=pb.R.device)
        c2 = o.cauchy ** 2
        for a in range(0, self.M, o.chunk):
            sl = slice(a, min(self.M, a + o.chunk))
            r, aux = self.res_v(zero, *self._consts(sl))
            s2 = (r * r).sum(-1)
            s2 = torch.where((aux[:, 2] > 1e-9) & torch.isfinite(s2), s2, torch.full_like(s2, 1e4 * c2))
            total += (c2 * torch.log1p(s2 / c2)).sum()
        if o.prior_weight > 0 and self.has_prior.any():
            e = self._prior_e()
            w = o.prior_weight * pb.prior_conf[self.has_prior]
            pc2 = o.prior_cauchy ** 2
            total += (w * pc2 * torch.log1p(e * e / pc2)).sum()
        return float(total)

    def residual_norms(self):
        """|r| of every observation in match pixels (inf when the point is behind the target camera)."""
        pb, o = self.pb, self.opt
        zero = self._zero()
        out = torch.empty(self.M, dtype=pb.R.dtype, device=pb.R.device)
        for a in range(0, self.M, o.chunk):
            sl = slice(a, min(self.M, a + o.chunk))
            r, aux = self.res_v(zero, *self._consts(sl))
            n = r.norm(dim=-1)
            out[sl] = torch.where((aux[:, 2] > 1e-9) & torch.isfinite(n), n, torch.full_like(n, float("inf")))
        return out

    def sampson_px(self):
        """Depth-free check of every observation: Sampson distance of (source pixel, target pixel) to the epipolar
        geometry of the current poses and intrinsics, in original pixels of the target image."""
        pb, o = self.pb, self.opt
        base = pb.model[:-3] if pb.model.endswith("_pp") else pb.model
        K = self.K
        out = torch.empty(self.M, dtype=pb.R.dtype, device=pb.R.device)

        def norm(x, p, c):
            if pb.model.endswith("_pp"):
                c = c + p[K - 2:K]
            return _undistort(base, (x - c) / _focal(base, p), p)
        nv = vmap(norm)
        for a in range(0, self.M, o.chunk):
            sl = slice(a, min(self.M, a + o.chunk))
            p = pb.p_o[sl]; i = self.src_o[sl]; j = pb.tgt[sl]
            xi = nv(pb.x_p[p], pb.intr[pb.group[i]], pb.c[i]); xj = nv(pb.xt[sl], pb.intr[pb.group[j]], pb.c[j])
            Rr = pb.R[j] @ pb.R[i].transpose(1, 2)
            tr = pb.t[j] - (Rr @ pb.t[i][:, :, None])[:, :, 0]
            E = hat(tr) @ Rr
            Xi = torch.cat([xi, torch.ones_like(xi[:, :1])], 1); Xj = torch.cat([xj, torch.ones_like(xj[:, :1])], 1)
            Ex = (E @ Xi[:, :, None])[:, :, 0]; Etx = (E.transpose(1, 2) @ Xj[:, :, None])[:, :, 0]
            num = (Xj * Ex).sum(1) ** 2
            den = Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2
            fj = _focal_batch(base, pb.intr[pb.group[j]])
            out[sl] = torch.sqrt(num / den.clamp_min(1e-30)) * fj
        return out

    def _prior_e(self):
        pb = self.pb
        m = self.has_prior
        return pb.logd[m] - pb.prior_logd[m] - pb.s[pb.img_p[m]]

    # ---------------------------------------------------------------------------------------------------------------
    def _normal_equations(self):
        pb, o, N, P, C, K = self.pb, self.opt, self.N, self.P, self.C, self.K
        nS, nT = self.nS, self.nT
        dev, dt = pb.R.device, pb.R.dtype
        zero = self._zero()
        Q = self.pair_keys.shape[0]
        nB = nS + nT
        Hpair = torch.zeros(Q, nB, nB, dtype=dt, device=dev)
        bpair = torch.zeros(Q, nB, dtype=dt, device=dev)
        Hdd = torch.zeros(P, dtype=dt, device=dev)
        bd = torch.zeros(P, dtype=dt, device=dev)
        A = torch.zeros(P, nS, dtype=dt, device=dev)          # H_{src block, d}
        Ct = torch.empty(self.M, nT, dtype=dt, device=dev)    # H_{tgt block, d} per observation
        c2 = o.cauchy ** 2
        xu, dxu = self._rays() if o.chain else (None, None)
        for a in range(0, self.M, o.chunk):
            sl = slice(a, min(self.M, a + o.chunk))
            if o.chain:
                J, aux = self._jacobian(sl, xu, dxu)          # (B, 2, D), (B, 3)
            else:
                J, aux = self.jac_fn(zero, *self._consts(sl))
            r = aux[:, :2]
            valid = (aux[:, 2] > 1e-9) & torch.isfinite(r).all(-1) & torch.isfinite(J).all(-1).all(-1)
            r = torch.where(valid[:, None], r, torch.zeros_like(r))
            J = torch.where(valid[:, None, None], J, torch.zeros_like(J))
            s2 = (r * r).sum(-1)
            w = torch.where(valid, 1.0 / (1.0 + s2 / c2), torch.zeros_like(s2))
            if not o.refine_intrinsics:
                J[:, :, 12:12 + 2 * K] = 0
            Js = torch.cat([J[:, :, 0:6], torch.zeros_like(J[:, :, :1]), J[:, :, 12:12 + K]], 2)     # (B, 2, nS)
            Jt = torch.cat([J[:, :, 6:12], J[:, :, 12 + K:12 + 2 * K]], 2)                           # (B, 2, nT)
            Jd = J[:, :, 12 + 2 * K]                                                                  # (B, 2)
            Jc = torch.cat([Js, Jt], 2)
            wJc = Jc * w[:, None, None]
            q = self.pair_o[sl]
            Hpair.index_add_(0, q, torch.einsum("bki,bkj->bij", wJc, Jc))
            bpair.index_add_(0, q, torch.einsum("bki,bk->bi", wJc, r))
            p = pb.p_o[sl]
            wJd = Jd * w[:, None]
            Hdd.index_add_(0, p, (wJd * Jd).sum(-1))
            bd.index_add_(0, p, (wJd * r).sum(-1))
            A.index_add_(0, p, torch.einsum("bki,bk->bi", Js, wJd))
            Ct[sl] = torch.einsum("bki,bk->bi", Jt, wJd)
        src = torch.div(self.pair_keys, N, rounding_mode="floor"); tgt = self.pair_keys % N
        idx = torch.cat([self.src_idx[src], self.tgt_idx[tgt]], 1)               # (Q, nB) unknowns of each pair block
        pdiag = torch.zeros(C, dtype=dt, device=dev)                             # camera diagonal of the prior
        b = torch.zeros(C, dtype=dt, device=dev)
        b.index_put_((idx,), bpair, accumulate=True)
        if o.prior_weight > 0 and self.has_prior.any():
            m = self.has_prior
            pm = torch.nonzero(m)[:, 0]
            e = self._prior_e()
            pc2 = o.prior_cauchy ** 2
            wp = o.prior_weight * pb.prior_conf[m] / (1.0 + e * e / pc2)
            Hdd.index_add_(0, pm, wp)
            bd.index_add_(0, pm, wp * e)
            sidx = 7 * pb.img_p[pm] + 6
            pdiag.index_add_(0, sidx, wp)
            b.index_add_(0, sidx, -wp * e)
            A[pm, 6] += -wp
        return dict(Hpair=Hpair, idx=idx, pdiag=pdiag, b=b, Hdd=Hdd, bd=bd, A=A, Ct=Ct)

    def _dense(self, ne):
        C, nB = self.C, self.nS + self.nT
        H = torch.zeros(C, C, dtype=ne["b"].dtype, device=ne["b"].device)
        idx = ne["idx"]
        H.index_put_((idx[:, :, None].expand(-1, -1, nB), idx[:, None, :].expand(-1, nB, -1)), ne["Hpair"], accumulate=True)
        H.diagonal().add_(ne["pdiag"])
        return H, ne["b"].clone()

    def _schur(self, H, b, Hdd, bd, A, Ct):
        fams, b, inv = self._schur_blocks(b, Hdd, bd, A, Ct)
        for blocks, r, c in fams:
            H.index_put_((r[:, :, None].expand(-1, -1, c.shape[1]), c[:, None, :].expand(-1, r.shape[1], -1)), blocks, accumulate=True)
        return H, b, inv

    def _schur_blocks(self, b, Hdd, bd, A, Ct):
        """The Schur correction -Hcd Hdd^-1 Hdc as families (blocks (B, nr, nc), global rows (B, nr), global cols
        (B, nc)): per source image, per (source, target) image pair both ways, per co-observed (target, target)
        pair; and the reduced right-hand side (b updated in place)."""
        pb, N, nS, nT = self.pb, self.N, self.nS, self.nT
        dev, dt = b.device, b.dtype
        ok = Hdd > 1e-12
        inv = torch.where(ok, 1.0 / Hdd.clamp_min(1e-12), torch.zeros_like(Hdd))
        Ssrc = torch.zeros(N, nS, nS, dtype=dt, device=dev)
        Ssrc.index_add_(0, pb.img_p, A[:, :, None] * A[:, None, :] * inv[:, None, None])
        si = self.src_idx
        fams = [(-Ssrc, si, si)]
        bs = torch.zeros(N, nS, dtype=dt, device=dev)
        bs.index_add_(0, pb.img_p, A * (bd * inv)[:, None])
        b.index_put_((si,), -bs, accumulate=True)
        Q = self.pair_keys.shape[0]
        Sst = torch.zeros(Q, nS, nT, dtype=dt, device=dev)
        for a in range(0, self.M, self.opt.chunk):
            sl = slice(a, min(self.M, a + self.opt.chunk))
            po = pb.p_o[sl]
            Sst.index_add_(0, self.pair_o[sl], A[po][:, :, None] * Ct[sl][:, None, :] * inv[po][:, None, None])
        src = torch.div(self.pair_keys, N, rounding_mode="floor"); tgt = self.pair_keys % N
        ri, ci = self.src_idx[src], self.tgt_idx[tgt]
        fams += [(-Sst, ri, ci), (-Sst.transpose(1, 2), ci, ri)]
        bt = torch.zeros(N, nT, dtype=dt, device=dev)
        bt.index_add_(0, pb.tgt, Ct * (bd * inv)[pb.p_o][:, None])
        b.index_put_((self.tgt_idx,), -bt, accumulate=True)
        if self._pairs is None:                 # all ordered observation pairs that share a depth variable (fixed)
            cnt = self.count[pb.p_o]
            r = torch.repeat_interleave(torch.arange(self.M, device=dev), cnt)
            off = torch.arange(r.numel(), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            partner = self.start[pb.p_o[r]] + off
            ukey, kinv = torch.unique(pb.tgt[r] * N + pb.tgt[partner], return_inverse=True)
            self._pairs = (r, partner, kinv, ukey)
        r_all, pt_all, kinv_all, ukey = self._pairs
        Stt = torch.zeros(ukey.numel(), nT, nT, dtype=dt, device=dev)
        step = 4 * self.opt.chunk
        for a in range(0, r_all.numel(), step):
            r_, pt, key = r_all[a:a + step], pt_all[a:a + step], kinv_all[a:a + step]
            Stt.index_add_(0, key, Ct[r_][:, :, None] * Ct[pt][:, None, :] * inv[pb.p_o[r_]][:, None, None])
        ta, tb = torch.div(ukey, N, rounding_mode="floor"), ukey % N
        ia, ib = self.tgt_idx[ta], self.tgt_idx[tb]
        fams.append((-Stt, ia, ib))
        return fams, b, inv

    # ---------------------------------------------------------------------------------------------------------------
    # Block-sparse reduced camera system + PCG (solver "sparse"): the same blocks as the dense path, never scattered
    # into a C x C matrix; S v costs one batched matrix-vector product per block. Preconditioner: the exact diagonal
    # blocks of S (one per image, 7 unknowns; one per intrinsics group, K unknowns).
    def _pmap(self):
        if getattr(self, "_pm", None) is None:
            N, K, C = self.N, self.K, self.C
            g = torch.arange(C, device=self.pb.R.device)
            img = g < 7 * N
            blk = torch.where(img, torch.div(g, 7, rounding_mode="floor"), N + torch.div(g - 7 * N, K, rounding_mode="floor"))
            pos = torch.where(img, g % 7, (g - 7 * N) % K)
            self._pm = (blk, pos, max(7, K), N + self.G)
        return self._pm

    def _blockdiag(self, fams):
        blk, pos, W, NB = self._pmap()
        out = torch.zeros(NB * W * W, dtype=self.pb.R.dtype, device=self.pb.R.device)
        for vals, r, c in fams:
            B, nr, nc = vals.shape
            br, bc = blk[r][:, :, None].expand(B, nr, nc), blk[c][:, None, :].expand(B, nr, nc)
            m = br == bc
            flat = (br[m] * W + pos[r][:, :, None].expand(B, nr, nc)[m]) * W + pos[c][:, None, :].expand(B, nr, nc)[m]
            out.index_add_(0, flat, vals[m])
        return out.view(NB, W, W)

    def _sparse_solve(self, hcc, fams, b, lam, dcc, diag_extra):
        """Damped camera step from the block-sparse reduced system: hcc / fams = block families of Hcc and of the
        Schur correction, diag_extra = diagonal terms outside the blocks (prior), dcc = undamped diagonal of Hcc."""
        pb, N, C, o = self.pb, self.N, self.C, self.opt
        blk, pos, W, NB = self._pmap()
        dev, dt = b.device, b.dtype
        damp = lam * dcc.abs() + 1e-9 * (dcc.abs().max() + 1)
        free = torch.ones(C, dtype=torch.bool, device=dev)
        fixed = torch.nonzero(pb.fixed)[:, 0]
        if fixed.numel():
            free[(7 * fixed[:, None] + torch.arange(6, device=dev)).reshape(-1)] = False
        if not o.refine_intrinsics:
            free[7 * N:] = False
        allf = hcc + fams
        Pm = self._hcc_bd + self._blockdiag(fams)
        d = torch.zeros(NB, W, dtype=dt, device=dev)
        d[blk, pos] = damp + diag_extra
        used = torch.zeros(NB, W, dtype=torch.bool, device=dev)
        used[blk, pos] = free
        Pm = Pm + torch.diag_embed(d)
        um = used[:, :, None] & used[:, None, :]
        Pm = torch.where(um, Pm, torch.zeros_like(Pm)) + torch.diag_embed((~used).to(dt))
        fr = free.to(dt)
        dg = damp + diag_extra
        prec = None
        c32 = o.precond == "chol32" or (o.precond == "auto" and C <= o.chol32_max)
        fresh = True
        if c32:
            fresh = getattr(self, "_c32", None) is None
            if fresh:
                self._c32 = self._chol32(allf, dg, free)
                self.factorizations = getattr(self, "factorizations", 0) + 1
            prec = self._c32
        if prec is None:
            L, info = torch.linalg.cholesky_ex(Pm)
            if bool((info != 0).any()):
                return None

            def prec(r):
                x = torch.zeros(NB, W, dtype=dt, device=dev)
                x[blk, pos] = r
                return torch.cholesky_solve(x[:, :, None], L)[:, :, 0][blk, pos] * fr

        def mv(v):
            v = v * fr
            y = dg * v
            for vals, r, c in allf:
                y.index_add_(0, r.reshape(-1), torch.bmm(vals, v[c][:, :, None]).reshape(-1))
            return y * fr
        if c32 and self._c32 is not None and not fresh:
            # an earlier step's factor: give it 2 x refactor_its CG iterations, else refactor and solve again
            x = self._cg(mv, prec, -b * fr, max_its=2 * o.refactor_its)
            if self.pcg_iters[-1] <= 2 * o.refactor_its and self._cg_ok:
                if self.pcg_iters[-1] > o.refactor_its:
                    self._c32 = None
                return x
            prec = None; self._c32 = None             # drop every reference to the old factor first
            self._c32 = self._chol32(allf, dg, free)
            self.factorizations = getattr(self, "factorizations", 0) + 1
            if self._c32 is None:
                return None
            prec = self._c32
        x = self._cg(mv, prec, -b * fr)
        if c32 and self.pcg_iters[-1] > o.refactor_its:
            self._c32 = None                 # stale: refactor at the next solve
        return x

    def _chol32(self, allf, dg, free):
        """Preconditioner: the reduced system scattered into a float32 C x C matrix and factored in place (a small
        relative jitter on the diagonal keeps float32 positive definite; it only affects the preconditioner)."""
        C = self.C
        nf = torch.nonzero(~free)[:, 0]
        info = torch.zeros((), dtype=torch.int32, device=dg.device)
        for jit in (1e-6, 1e-4, 1e-2):
            H = torch.zeros(C, C, dtype=torch.float32, device=dg.device)
            for vals, r, c in allf:
                H.index_put_((r[:, :, None].expand(-1, -1, c.shape[1]), c[:, None, :].expand(-1, r.shape[1], -1)),
                             vals.float(), accumulate=True)
            H.diagonal().add_(dg.float())
            H[nf, :] = 0; H[:, nf] = 0; H[nf, nf] = 1.0
            H.diagonal().mul_(1 + jit)
            torch.linalg.cholesky_ex(H, out=(H, info))
            if int(info) == 0:
                fr = free.to(dg.dtype)
                return lambda r, L=H: torch.cholesky_solve(r.float()[:, None], L)[:, 0].to(r.dtype) * fr
            H = None
        return None

    def _cg(self, mv, prec, b, max_its=None):
        o = self.opt
        x = torch.zeros_like(b)
        bn = float(b.norm())
        self._cg_ok = True
        if bn == 0.0:
            self.pcg_iters.append(0)
            return x
        r = b.clone(); z = prec(r); p = z.clone(); rz = (r * z).sum()
        k = 0
        self._cg_ok = False
        for k in range(1, (max_its or o.pcg_max) + 1):
            Ap = mv(p)
            alpha = rz / (p * Ap).sum()
            x += alpha * p; r -= alpha * Ap
            if k % 5 == 0 and float(r.norm()) <= o.pcg_tol * bn:
                self._cg_ok = True
                break
            z = prec(r); rz_new = (r * z).sum()
            p = z + (rz_new / rz) * p; rz = rz_new
        self.pcg_iters.append(k)
        return x

    def _solve(self, H, b, lam, dcc):
        """H, b: reduced camera system (Schur complement taken with the damped depth block); dcc: the diagonal of the
        undamped camera block, for Marquardt damping."""
        pb, N = self.pb, self.N
        d = torch.diagonal(H)
        d += lam * dcc.abs() + 1e-9 * (dcc.abs().max() + 1)
        fixed = torch.nonzero(pb.fixed)[:, 0]
        if fixed.numel():
            fi = (7 * fixed[:, None] + torch.arange(6, device=H.device)).reshape(-1)
            H[fi, :] = 0; H[:, fi] = 0; H[fi, fi] = 1.0
            b = b.clone(); b[fi] = 0
        if not self.opt.refine_intrinsics:
            gi = torch.arange(7 * N, self.C, device=H.device)
            H[gi, :] = 0; H[:, gi] = 0; H[gi, gi] = 1.0
            b = b.clone(); b[gi] = 0
        L, info = torch.linalg.cholesky_ex(H)
        if int(info) != 0:
            return None
        return torch.cholesky_solve(-b[:, None], L)[:, 0]

    # ---------------------------------------------------------------------------------------------------------------
    # Implicit Schur complement (Agarwal et al., "Bundle Adjustment in the Large"; Ceres ITERATIVE_SCHUR): the reduced
    # camera system S = Hcc - Hcd Hdd^-1 Hdc is never formed; CG only needs S v, computed from the pair blocks of Hcc
    # (Q x nB x nB) and the depth couplings A (P x nS) and Ct (M x nT). Preconditioner: block Jacobi on S (one 7x7 block
    # per image, one K x K block per intrinsics group; cross terms between a pose and the intrinsics left out).
    def _pcg_prep(self, ne):
        """Per LM iteration: the undamped diagonal blocks of Hcc, its diagonal (Marquardt damping) and the depth
        couplings summed per (depth, intrinsics group)."""
        pb, N, G, K = self.pb, self.N, self.G, self.K
        dev, dt = ne["b"].device, ne["b"].dtype
        Hp = ne["Hpair"]
        src = torch.div(self.pair_keys, N, rounding_mode="floor"); tgt = self.pair_keys % N
        gs, gt = pb.group[src], pb.group[tgt]
        Pimg = torch.zeros(N, 7, 7, dtype=dt, device=dev)
        Pimg.index_add_(0, src, Hp[:, 0:7, 0:7])
        Tt = torch.zeros(N, 6, 6, dtype=dt, device=dev)
        Tt.index_add_(0, tgt, Hp[:, 7 + K:13 + K, 7 + K:13 + K])
        Pimg[:, :6, :6] += Tt
        Pg = torch.zeros(G, K, K, dtype=dt, device=dev)
        Pg.index_add_(0, gs, Hp[:, 7:7 + K, 7:7 + K])
        Pg.index_add_(0, gt, Hp[:, 13 + K:, 13 + K:])
        same = torch.nonzero(gs == gt)[:, 0]
        X = Hp[same, 7:7 + K, 13 + K:]
        Pg.index_add_(0, gs[same], X + X.transpose(1, 2))
        pd = ne["pdiag"]
        Pimg += torch.diag_embed(pd[:7 * N].reshape(N, 7))
        Pg += torch.diag_embed(pd[7 * N:].reshape(G, K))
        dcc = torch.cat([torch.diagonal(Pimg, dim1=1, dim2=2).reshape(-1), torch.diagonal(Pg, dim1=1, dim2=2).reshape(-1)])
        if getattr(self, "_ug", None) is None:     # (depth, group) keys of the intrinsics couplings (structure fixed)
            k1 = torch.arange(self.P, device=dev) * G + pb.group[pb.img_p]
            k2 = pb.p_o * G + pb.group[pb.tgt]
            u, inv_ = torch.unique(torch.cat([k1, k2]), return_inverse=True)
            self._ug = (inv_[:self.P], inv_[self.P:], torch.div(u, G, rounding_mode="floor"), u % G)
        i1, i2, up, ug = self._ug
        U = torch.zeros(up.numel(), K, dtype=dt, device=dev)
        U.index_add_(0, i1, ne["A"][:, 7:])
        U.index_add_(0, i2, ne["Ct"][:, 6:])
        free = torch.ones(self.C, dtype=torch.bool, device=dev)
        fixed = torch.nonzero(pb.fixed)[:, 0]
        if fixed.numel():
            free[(7 * fixed[:, None] + torch.arange(6, device=dev)).reshape(-1)] = False
        if not self.opt.refine_intrinsics:
            free[7 * N:] = False
        return dict(Pimg=Pimg, Pg=Pg, dcc=dcc, U=U, free=free)

    def _pcg_solve(self, ne, pp, inv, lam):
        """Damped step of the cameras from the implicit reduced system (None if the preconditioner fails)."""
        pb, N, G, K, C, o = self.pb, self.N, self.G, self.K, self.C, self.opt
        A, Ct, Hp, idx = ne["A"], ne["Ct"], ne["Hpair"], ne["idx"]
        dev, dt = A.device, A.dtype
        si = self.src_idx[pb.img_p]                     # (P, nS)
        step = 8 * o.chunk
        Ms = [slice(a, min(self.M, a + step)) for a in range(0, self.M, step)]
        dcc = pp["dcc"]
        damp = lam * dcc.abs() + 1e-9 * (dcc.abs().max() + 1)
        free = pp["free"].to(dt)

        def hcd(w, y):                                  # y += Hcd w   (w: one value per depth)
            y.index_add_(0, si.reshape(-1), (A * w[:, None]).reshape(-1))
            for sl in Ms:
                y.index_add_(0, self.tgt_idx[pb.tgt[sl]].reshape(-1), (Ct[sl] * w[pb.p_o[sl]][:, None]).reshape(-1))

        def hdc(v):                                     # Hdc v
            h = (A * v[si]).sum(-1)
            for sl in Ms:
                h.index_add_(0, pb.p_o[sl], (Ct[sl] * v[self.tgt_idx[pb.tgt[sl]]]).sum(-1))
            return h
        # right-hand side: b - Hcd Hdd^-1 bd
        b = ne["b"].clone()
        hcd(-ne["bd"] * inv, b)
        b = -b * free
        # preconditioner blocks: diag blocks of S, damped
        Pimg = pp["Pimg"].clone(); Pg = pp["Pg"].clone()
        Pimg.index_add_(0, pb.img_p, -(A[:, :7, None] * A[:, None, :7]) * inv[:, None, None])
        Tt = torch.zeros(N, 6, 6, dtype=dt, device=dev)
        for sl in Ms:
            c6 = Ct[sl, :6]
            Tt.index_add_(0, pb.tgt[sl], -(c6[:, :, None] * c6[:, None, :]) * inv[pb.p_o[sl]][:, None, None])
        Pimg[:, :6, :6] += Tt
        _, _, up, ug = self._ug
        U = pp["U"]
        Pg.index_add_(0, ug, -(U[:, :, None] * U[:, None, :]) * inv[up][:, None, None])
        Pimg += torch.diag_embed(damp[:7 * N].reshape(N, 7))
        Pg += torch.diag_embed(damp[7 * N:].reshape(G, K))
        fi = (~pp["free"][:7 * N]).reshape(N, 7)
        if bool(fi.any()):
            m = fi[:, :, None] | fi[:, None, :]
            Pimg = torch.where(m, torch.zeros_like(Pimg), Pimg) + torch.diag_embed(fi.to(dt))
        if not o.refine_intrinsics:
            Pg = torch.eye(K, dtype=dt, device=dev).expand(G, K, K).clone()
        Li, info1 = torch.linalg.cholesky_ex(Pimg)
        Lg, info2 = torch.linalg.cholesky_ex(Pg)
        if bool((info1 != 0).any()) or bool((info2 != 0).any()):
            return None

        def prec(r):
            zi = torch.cholesky_solve(r[:7 * N].reshape(N, 7, 1), Li).reshape(-1)
            zg = torch.cholesky_solve(r[7 * N:].reshape(G, K, 1), Lg).reshape(-1)
            return torch.cat([zi, zg]) * free

        def mv(v):
            v = v * free
            y = torch.zeros(C, dtype=dt, device=dev)
            y.index_add_(0, idx.reshape(-1), torch.bmm(Hp, v[idx][:, :, None]).reshape(-1))
            hcd(-hdc(v) * inv, y)
            y += (ne["pdiag"] + damp) * v
            return y * free

        x = torch.zeros(C, dtype=dt, device=dev)
        r = b.clone(); z = prec(r); p = z.clone(); rz = (r * z).sum()
        bn = float(b.norm())
        k = 0
        if bn == 0.0:
            self.pcg_iters.append(0)
            return x
        for k in range(1, o.pcg_max + 1):
            Ap = mv(p)
            alpha = rz / (p * Ap).sum()
            x += alpha * p; r -= alpha * Ap
            if k % 10 == 0 and float(r.norm()) <= o.pcg_tol * bn:
                break
            z = prec(r); rz_new = (r * z).sum()
            p = z + (rz_new / rz) * p; rz = rz_new
        self.pcg_iters.append(k)
        return x

    def _apply(self, dc, dd):
        pb, N = self.pb, self.N
        P6 = dc[: 7 * N].reshape(N, 7)
        Rn = so3_exp(P6[:, 0:3]) @ pb.R
        tn = pb.t + P6[:, 3:6]
        sn = pb.s + P6[:, 6]
        In = pb.intr + dc[7 * N:].reshape(self.G, self.K)
        return Rn, tn, sn, In, pb.logd + dd

    def _state(self):
        pb = self.pb
        return pb.R, pb.t, pb.s, pb.intr, pb.logd

    def _set(self, st):
        pb = self.pb
        pb.R, pb.t, pb.s, pb.intr, pb.logd = st

    def run(self):
        pb, o = self.pb, self.opt
        lam = o.lm_init
        t0 = time.time()
        cost = self.cost()
        o.log.append({"it": 0, "cost": cost, "lam": lam, "t": 0.0})
        mode = o.solver if o.solver != "auto" else ("dense" if self.C <= o.dense_max else "sparse")
        dense = mode == "dense"
        self.pcg_iters = []
        for it in range(1, o.iters + 1):
            ne = self._normal_equations()
            Hdd, bd, A, Ct = ne["Hdd"], ne["bd"], ne["A"], ne["Ct"]
            if dense:
                H0, b0 = self._dense(ne)
                dcc = torch.diagonal(H0).clone()
            elif mode == "sparse":
                hcc = [(ne["Hpair"], ne["idx"], ne["idx"])]
                self._hcc_bd = self._blockdiag(hcc)
                blk, pos, W, NB = self._pmap()
                dcc = torch.diagonal(self._hcc_bd, dim1=1, dim2=2)[blk, pos] + ne["pdiag"]
            else:
                pp = self._pcg_prep(ne)
            rel = 0.0
            while True:
                if dense:
                    H, b, inv = self._schur(H0.clone(), b0.clone(), Hdd * (1 + lam) + 1e-12, bd, A, Ct)
                    dc = self._solve(H, b, lam, dcc)
                elif mode == "sparse":
                    fams, b, inv = self._schur_blocks(ne["b"].clone(), Hdd * (1 + lam) + 1e-12, bd, A, Ct)
                    dc = self._sparse_solve(hcc, fams, b, lam, dcc, ne["pdiag"])
                else:
                    Hl = Hdd * (1 + lam) + 1e-12
                    inv = torch.where(Hl > 1e-12, 1.0 / Hl.clamp_min(1e-12), torch.zeros_like(Hl))
                    dc = self._pcg_solve(ne, pp, inv, lam)
                if dc is None:
                    lam *= 10
                    if lam > 1e8:
                        break
                    continue
                dsrc = (A * dc[self.src_idx[pb.img_p]]).sum(-1)
                for a in range(0, self.M, 8 * o.chunk):
                    sl = slice(a, min(self.M, a + 8 * o.chunk))
                    dsrc.index_add_(0, pb.p_o[sl], (Ct[sl] * dc[self.tgt_idx[pb.tgt[sl]]]).sum(-1))
                dd = -(bd + dsrc) * inv
                old = self._state()
                self._set(self._apply(dc, dd))
                new = self.cost()
                if new < cost:
                    rel = (cost - new) / max(cost, 1e-12)
                    cost = new
                    lam = max(lam / 3, 1e-8)
                    break
                self._set(old)
                lam *= 4
                if lam > 1e8:
                    break
            o.log.append({"it": it, "cost": cost, "lam": lam, "t": time.time() - t0})
            if o.verbose:
                print(f"  it {it}: cost {cost:.6g} lam {lam:.1e} {time.time() - t0:.1f}s", flush=True)
            if lam > 1e8 or rel < o.min_rel_decrease:
                break
        return pb


# ---------------------------------------------------------------------------------------------------------------------
# The full schedule used in place of COLMAP's final bundle adjustment (docs/RAYBA.md): hybrid observations, OPENCV with
# a principal point per intrinsics group, groups merged once their focal lengths agree.
DEFAULTS = dict(model="opencv_pp", cauchy=0.25, cauchy_wide=1.0, iters=20, hybrid_px=4.0, rounds=2, prune_px=0.5,
                merge_tol=0.005, merge_size_tol=0.01)
# cauchy_wide: Cauchy scale (match pixels) of every round but the last (the tracks round and round 1), cauchy that of
# the last; a wider loss first, then the tight one (docs/RAYBA.md step 17: ETH3D dev +2.0 AUC@1 over 0.25 throughout).


def problem_from_arrays(z, dev, model):
    import numpy as np
    order = np.argsort(z["p_o"], kind="stable")
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)  # noqa: E731
    L = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=dev)  # noqa: E731
    pb = Problem(R=T(z["R"]), t=T(z["t"]), c=T(z["c"]), sc=T(z["sc"]), group=L(z["group"]),
                 intr=Problem.intr_from_simple_radial(T(z["logf"]), T(z["k"]), model),
                 fixed=torch.as_tensor(np.asarray(z["fixed"]), device=dev), img_p=L(z["img_p"]), x_p=T(z["x_p"]),
                 logd=T(z["logd"]), prior_logd=T(z["prior_logd"]), prior_conf=T(z["prior_conf"]), p_o=L(z["p_o"][order]),
                 tgt=L(z["tgt"][order]), xt=T(z["xt"][order]), s=T(z["s0"]), model=model)
    trk = torch.as_tensor(np.asarray(z["trk"])[order], device=dev) if "trk" in z else None
    return pb, trk


def merge_groups(pb, WH, tol, size_tol=0.01):
    """Merge intrinsics groups whose refined focal lengths agree within tol (relative) and whose image sizes agree
    within size_tol (undistorted crops of one camera differ by a few pixels; principal points stay per image).
    Single linkage over groups sorted by focal length. GT-free. Returns (groups before, after)."""
    import numpy as np
    G = pb.intr.shape[0]
    grp = pb.group.cpu().numpy()
    present = [g for g in range(G) if (grp == g).any()]
    wh = {g: np.asarray(WH[grp == g][0], dtype=float) for g in present}
    f = pb.intr[:, 0].exp().cpu().numpy()
    clusters = []
    for g in sorted(present, key=lambda g: f[g]):
        for cl in clusters:
            if any(abs(f[g] - f[h]) / f[h] <= tol and np.all(np.abs(wh[g] - wh[h]) / wh[h] <= size_tol) for h in cl):
                cl.append(g); break
        else:
            clusters.append([g])
    new = {g: k for k, cl in enumerate(clusters) for g in cl}
    cnt = np.bincount(grp, minlength=G)
    intr = torch.zeros(len(clusters), pb.intr.shape[1], dtype=pb.intr.dtype, device=pb.intr.device)
    wsum = torch.zeros(len(clusters), dtype=pb.intr.dtype, device=pb.intr.device)
    for g, ng in new.items():
        intr[ng] += pb.intr[g] * float(cnt[g]); wsum[ng] += float(cnt[g])
    pb.intr = intr / wsum[:, None]
    pb.group = torch.as_tensor(np.array([new[g] for g in grp]), dtype=torch.long, device=pb.group.device)
    return len(present), len(clusters)


def solve(z, dev="cuda", **cfg):
    """Run the schedule on a problem built by rayba_build.build (dict of arrays). Returns numpy arrays: refined
    cam_from_world (R, t), intrinsics per group (intr, in the model's parameterisation) and the group of each image,
    and the world position of every track source (pt_xyz, for pt_ids; nan if its depth was dropped)."""
    import numpy as np
    c = {**DEFAULTS, **cfg}
    t0 = time.time()
    pb, trk = problem_from_arrays(z, dev, c["model"])
    sopt = {k: c[k] for k in ("solver", "dense_max", "pcg_tol", "pcg_max") if k in c}
    cw = c["cauchy_wide"] or c["cauchy"]
    log = []
    full = (pb.p_o, pb.tgt, pb.xt)
    if c["hybrid_px"] and trk is not None and bool((~trk).any()):
        pb.p_o, pb.tgt, pb.xt = full[0][trk], full[1][trk], full[2][trk]
        o = Options(iters=c["iters"], cauchy=cw, **sopt); RayBA(pb, o).run(); log += o.log
        pb.p_o, pb.tgt, pb.xt = full
        keep = trk | (RayBA(pb, Options(cauchy=c["cauchy"])).sampson_px() < c["hybrid_px"])
        pb.p_o, pb.tgt, pb.xt = full[0][keep], full[1][keep], full[2][keep]
    merged = None
    for r in range(c["rounds"]):
        o = Options(iters=c["iters"], cauchy=c["cauchy"] if r == c["rounds"] - 1 else cw, **sopt)
        ba = RayBA(pb, o); ba.run(); log += o.log
        if r + 1 < c["rounds"]:
            keep = ba.residual_norms() < c["prune_px"]
            pb.p_o, pb.tgt, pb.xt = pb.p_o[keep], pb.tgt[keep], pb.xt[keep]
            if c["merge_tol"]:
                merged = merge_groups(pb, np.asarray(z["WH"]), c["merge_tol"], c["merge_size_tol"])
    # 3D points of the tracks from their source depths, with the final intrinsics
    pt_xyz = np.full((len(z["pt_ids"]), 3), np.nan)
    pp = torch.as_tensor(np.asarray(z["pt_p"]), device=dev)
    okp = pp >= 0
    if bool(okp.any()):
        ray_v = _PARTS[pb.model][0]
        p = pp[okp]
        i = pb.img_p[p]
        xu = ray_v(pb.intr[pb.group[i]], pb.x_p[p], pb.c[i])
        Xc = torch.exp(pb.logd[p])[:, None] * torch.cat([xu, torch.ones_like(xu[:, :1])], 1)
        Xw = (pb.R[i].transpose(1, 2) @ (Xc - pb.t[i])[:, :, None])[:, :, 0]
        pt_xyz[okp.cpu().numpy()] = Xw.cpu().numpy()
    return dict(R=pb.R.cpu().numpy(), t=pb.t.cpu().numpy(), intr=pb.intr.cpu().numpy(), group=pb.group.cpu().numpy(),
                model=c["model"], pt_xyz=pt_xyz, wall=time.time() - t0, merged=merged, iterations=len(log),
                observations=int(pb.p_o.shape[0]))
