"""Render and compare: is photo i consistent with what the model says camera i sees? (docs/EXAMINE.md)

Neighbours of camera i are warped into i through dense feedforward depth (scaled to the model with the sparse
points each neighbour observes) and the model's poses; the photometric agreement (NCC on a coarse grid) of the
placed pose is compared with the agreement of perturbed poses. A correct pose is a clear maximum. No matches are
used, so the check judges the images whose matches are lookalike pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

LOW = 128           # width of the comparison grid
PERTURB = (("yaw5", 5.0, 0.0), ("yaw15", 15.0, 0.0), ("shift5", 0.0, 0.05), ("shift15", 0.0, 0.15))


class RenderCompare:
    def __init__(self, rec, images_dir: Path, min_conf: float = 0.0, backface: bool = False, min_cover: float = 0.0, v2: bool = False):
        self.rec = rec
        self.v2 = v2                 # dense occlusion-aware render, align-then-judge verdict, checkerboard figure (docs/EXAMINE.md, v2)
        self.cand_depth: dict[str, tuple[np.ndarray, Any, str]] = {}   # every image of a depth group -> (depth, transform, group file)
        self.group_members: dict[str, list[str]] = {}
        self.group_files: dict[str, Any] = {}
        self.group_scale: dict[str, float | None] = {}
        self.colour_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self.backface = backface     # drop source points whose surface faces away from the target camera
        self.min_cover = min_cover   # a render covering less of the photo than this is "no evidence", not "inconsistent"
        self.normals: dict[str, np.ndarray] = {}   # name -> unit surface normals in the source camera frame (H, W, 3); 0 = unknown
        self.min_conf = min_conf   # feedforward depth below this confidence is dropped (Pi3X: sky and far background)
        self.images_dir = images_dir
        self.gray_cache: dict[tuple[str, int], np.ndarray] = {}
        self.depth: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # name -> (depth in model pixels, transform original -> model pixels)
        self.scales: dict[str, float | None] = {}                   # name -> scale to the model (recomputed after refinements)

    # -- images and cameras ------------------------------------------------------------------
    def gray(self, name: str, w: int = LOW) -> np.ndarray:
        key = (name, w)
        if key not in self.gray_cache:
            from PIL import Image

            im = Image.open(self.images_dir / name).convert("L")
            h = max(1, round(im.height * w / im.width))
            self.gray_cache[key] = np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        return self.gray_cache[key]

    def image(self, name: str):
        return self.rec.find_image_with_name(name)

    def K(self, name: str) -> tuple[np.ndarray, int, int]:
        cam = self.rec.cameras[self.image(name).camera_id]
        f = cam.mean_focal_length()
        return np.array([[f, 0, cam.principal_point_x], [0, f, cam.principal_point_y], [0, 0, 1]]), cam.width, cam.height

    def span(self) -> float:
        C = np.array([im.projection_center() for im in self.rec.images.values() if im.has_pose])
        return float(np.linalg.norm(np.ptp(C, axis=0))) if len(C) > 1 else 1.0

    def neighbours(self, name: str, k: int = 4, exclude: set[str] = frozenset()) -> list[str]:
        im = self.image(name)
        span = self.span()
        c, d = im.projection_center(), im.viewing_direction()
        rows = []
        for other in self.rec.images.values():
            if not other.has_pose or other.name == name or other.name in exclude:
                continue
            rows.append((float(np.linalg.norm(other.projection_center() - c)) / span + 0.5 * (1 - float(np.dot(other.viewing_direction(), d))), other.name))
        return [n for _, n in sorted(rows)[:k]]

    # -- depth ---------------------------------------------------------------------------------
    def add_depth(self, npz_path: str) -> None:
        """Register dense depth of the images in a feedforward group (backend result with save_depth). Registered images
        become render sources; with v2 every member (a candidate too) keeps its depth for inverse rendering, scaled
        through the group's registered members (one feedforward frame)."""
        z = np.load(npz_path)
        if self.v2:
            names = [str(n) for n in z["names"]]
            self.group_members[npz_path] = names
            self.group_files[npz_path] = z
            for k, n in enumerate(names):
                if n in self.cand_depth:
                    continue
                d = z["depth"][k].astype(np.float32)
                if self.min_conf > 0 and "conf" in z.files:
                    d = np.where(z["conf"][k].astype(np.float32) >= self.min_conf, d, 0.0).astype(np.float32)
                self.cand_depth[n] = (d, z["transforms"][k], npz_path)
        for k, n in enumerate(z["names"]):
            n = str(n)
            if n in self.depth or not self.rec.find_image_with_name(n) or not self.image(n).has_pose:
                continue
            d = z["depth"][k].astype(np.float32)
            if self.min_conf > 0 and "conf" in z.files:
                d = np.where(z["conf"][k].astype(np.float32) >= self.min_conf, d, 0.0).astype(np.float32)
            self.depth[n] = (d, z["transforms"][k])

    def normal_map(self, name: str) -> np.ndarray:
        """Surface normals of a source's depth map in its camera frame (scale-free, so cached), oriented to the camera.
        Pixels at depth edges or without depth get a zero normal and count as facing."""
        if name not in self.normals:
            depth, tr = self.depth[name]
            K, _, _ = self.K(name)
            sx, sy, ox, oy = tr
            H, W = depth.shape
            vs, us = np.mgrid[0:H, 0:W]
            x = ((us - ox) / sx - K[0, 2]) / K[0, 0]
            y = ((vs - oy) / sy - K[1, 2]) / K[1, 1]
            X = np.stack([x * depth, y * depth, depth], -1)
            du = np.zeros_like(X); dv = np.zeros_like(X)
            du[:, 1:-1] = X[:, 2:] - X[:, :-2]
            dv[1:-1] = X[2:] - X[:-2]
            n = np.cross(du, dv)
            ln = np.linalg.norm(n, axis=-1, keepdims=True)
            n = np.where(ln > 0, n / np.maximum(ln, 1e-12), 0.0)
            n = np.where((np.sum(n * X, -1, keepdims=True) > 0), -n, n)       # towards the camera
            ok = depth > 0
            ok[:, 1:-1] &= (depth[:, 2:] > 0) & (depth[:, :-2] > 0)
            ok[1:-1] &= (depth[2:] > 0) & (depth[:-2] > 0)
            # depth edges: neighbouring depths differ by more than 20% -> normal unreliable
            edge = np.zeros_like(ok)
            edge[:, 1:-1] |= np.abs(depth[:, 2:] - depth[:, :-2]) > 0.2 * np.maximum(depth[:, 1:-1], 1e-6)
            edge[1:-1] |= np.abs(depth[2:] - depth[:-2]) > 0.2 * np.maximum(depth[1:-1], 1e-6)
            n[~ok | edge] = 0.0
            self.normals[name] = n.astype(np.float32)
        return self.normals[name]

    def facing(self, name: str, vs, us, Xw: np.ndarray, Rw: np.ndarray, c_t: np.ndarray) -> np.ndarray:
        """Which source points (pixels vs, us of source name; world points Xw) face the target camera centre c_t."""
        if not self.backface:
            return np.ones(len(Xw), dtype=bool)
        nw = self.normal_map(name)[vs, us] @ Rw.T
        return np.sum(nw * (c_t - Xw), -1) >= 0

    def invalidate(self) -> None:
        """After a refinement or normalisation the model's scale changed: recompute depth scales on demand."""
        self.scales = {}
        self.group_scale = {}

    def scale(self, name: str) -> float | None:
        if name not in self.scales:
            depth, tr = self.depth[name]
            self.scales[name] = self.depth_scale(name, depth, tr)[0]
        return self.scales[name]

    def usable(self, names: list[str]) -> list[str]:
        return [n for n in names if n in self.depth and self.image(n) is not None and self.image(n).has_pose and self.scale(n) is not None]

    def depth_scale(self, name: str, depth: np.ndarray, tr) -> tuple[float | None, int]:
        """Median ratio of the model's depth to the feedforward depth over the sparse points the image observes."""
        im = self.image(name)
        P = im.cam_from_world()
        sx, sy, ox, oy = tr
        ratios = []
        for p2 in im.points2D:
            if not p2.has_point3D():
                continue
            xc = P * self.rec.points3D[p2.point3D_id].xyz
            u, v = p2.xy
            mu, mv = int(round(u * sx + ox)), int(round(v * sy + oy))
            if 0 <= mv < depth.shape[0] and 0 <= mu < depth.shape[1] and depth[mv, mu] > 0 and xc[2] > 0:
                ratios.append(xc[2] / depth[mv, mu])
        return (float(np.median(ratios)) if len(ratios) >= 20 else None), len(ratios)

    # -- the score -----------------------------------------------------------------------------
    def score(self, target: str, P_t, nbs: list[str]) -> tuple[float | None, float]:
        """NCC between neighbours' pixels and the target photo where the model says they land; overlap fraction."""
        K_t, W_t, H_t = self.K(target)
        g_t = self.gray(target)
        s_t = LOW / W_t
        R_t, t_t = P_t.rotation.matrix(), P_t.translation
        c_t = -R_t.T @ t_t
        cells: set[int] = set()
        self.last_cover = 0.0
        va, vb, nvalid, ntotal = [], [], 0, 0
        for j in nbs:
            if j not in self.depth or self.scale(j) is None:
                continue
            depth_j, tr_j = self.depth[j]
            scale_j = self.scale(j)
            P_j = self.image(j).cam_from_world().inverse()
            Rw, tw = P_j.rotation.matrix(), P_j.translation
            H, W = depth_j.shape
            step = max(1, W // 96)
            vs, us = np.mgrid[0:H:step, 0:W:step]
            d = depth_j[vs, us] * scale_j
            ok = d > 0
            sx, sy, ox, oy = tr_j
            K_j, W_j, H_j = self.K(j)
            x = ((us - ox) / sx - K_j[0, 2]) / K_j[0, 0]
            y = ((vs - oy) / sy - K_j[1, 2]) / K_j[1, 1]
            Xc = np.stack([x * d, y * d, d], -1)[ok]
            Xw = Xc @ Rw.T + tw
            Xt = Xw @ R_t.T + t_t
            z = Xt[:, 2]
            front = (z > 1e-6) & self.facing(j, vs[ok], us[ok], Xw, Rw, c_t)
            zz = np.where(front, z, 1.0)
            u_t = K_t[0, 0] * Xt[:, 0] / zz + K_t[0, 2]
            v_t = K_t[1, 1] * Xt[:, 1] / zz + K_t[1, 2]
            inb = front & (u_t >= 0) & (u_t < W_t - 1) & (v_t >= 0) & (v_t < H_t - 1)
            cells.update(((v_t[inb] * s_t).astype(int) * LOW + (u_t[inb] * s_t).astype(int)).tolist())
            ntotal += len(z)
            nvalid += int(inb.sum())
            if inb.sum() < 50:
                continue
            g_j = self.gray(j)
            a = g_j[(vs[ok] * (LOW / W)).astype(int).clip(0, g_j.shape[0] - 1)[inb], (us[ok] * (LOW / W)).astype(int).clip(0, LOW - 1)[inb]]
            b = g_t[(v_t[inb] * s_t).astype(int).clip(0, g_t.shape[0] - 1), (u_t[inb] * s_t).astype(int).clip(0, LOW - 1)]
            va.append(a)
            vb.append(b)
        self.last_cover = len(cells) / float(LOW * max(1, int(round(H_t * s_t))))
        if not va:
            return None, 0.0
        a = np.concatenate(va)
        b = np.concatenate(vb)
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), nvalid / max(1, ntotal)

    def perturbed(self, P, yaw_deg: float, shift_frac: float):
        import pycolmap

        R = P.rotation.matrix()
        c = -R.T @ P.translation
        a = np.radians(yaw_deg)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        R2 = Ry @ R
        c2 = c + shift_frac * self.span() * R.T[:, 0]
        return pycolmap.Rigid3d(pycolmap.Rotation3d(R2), -R2 @ c2)

    def judge(self, name: str, P=None, nbs: list[str] | None = None) -> dict[str, Any]:
        """Score of the pose against its perturbations. margin = score - best perturbed score; verdict by margin."""
        P = P or self.image(name).cam_from_world()
        nbs = self.usable(self.neighbours(name) if nbs is None else nbs)   # an explicit empty list = no sources: "no evidence"
        s0, ov = self.score(name, P, nbs)
        cover = self.last_cover
        if s0 is None:
            return {"image": name, "score": None, "verdict": "no evidence", "neighbours": nbs, "cover": round(cover, 3)}
        pert = {lab: self.score(name, self.perturbed(P, yaw, sh), nbs)[0] for lab, yaw, sh in PERTURB}
        best = max(v for v in pert.values() if v is not None) if any(v is not None for v in pert.values()) else None
        margin = None if best is None else s0 - best
        if margin is None or cover < self.min_cover:   # the sources show too little of what this photo sees
            verdict = "no evidence"
        elif margin > 0.05:          # calibrated on 5 scenes (docs/EXAMINE.md): keeps 88% of GT-correct poses, passes 2-4% of 10-20 deg errors
            verdict = "consistent"
        elif margin < -0.05 or s0 < 0.2:   # 74-79% of 10-20 deg errors fall here, 2% of correct poses
            verdict = "inconsistent"
        else:
            verdict = "ambiguous"
        return {"image": name, "score": round(s0, 3), "overlap": round(ov, 3), "perturbed": {k: (None if v is None else round(v, 3)) for k, v in pert.items()},
                "margin": None if margin is None else round(margin, 3), "verdict": verdict, "neighbours": nbs, "cover": round(cover, 3)}

    # -- the figure the VLM reads ----------------------------------------------------------------
    def figure(self, name: str, P, nbs: list[str], out_path: Path, width: int = 480) -> str | None:
        """Photo | neighbours' pixels warped into the camera | overlay. Returns the path, or None without depth."""
        from PIL import Image

        nbs = self.usable(nbs)
        if not nbs:
            return None
        K_t, W_t, H_t = self.K(name)
        h = max(1, round(H_t * width / W_t))
        photo = Image.open(self.images_dir / name).convert("RGB").resize((width, h), Image.BILINEAR)
        canvas = np.zeros((h, width, 3), dtype=np.float32)
        zbuf = np.full((h, width), np.inf, dtype=np.float32)
        R_t, t_t = P.rotation.matrix(), P.translation
        s = width / W_t
        for j in nbs:
            depth_j, tr_j = self.depth[j]
            scale_j = self.scale(j)
            img_j = np.asarray(Image.open(self.images_dir / j).convert("RGB").resize((depth_j.shape[1], depth_j.shape[0]), Image.BILINEAR), dtype=np.float32)
            P_j = self.image(j).cam_from_world().inverse()
            Rw, tw = P_j.rotation.matrix(), P_j.translation
            H, W = depth_j.shape
            vs, us = np.mgrid[0:H, 0:W]
            d = depth_j * scale_j
            ok = d > 0
            sx, sy, ox, oy = tr_j
            K_j, _, _ = self.K(j)
            x = ((us - ox) / sx - K_j[0, 2]) / K_j[0, 0]
            y = ((vs - oy) / sy - K_j[1, 2]) / K_j[1, 1]
            Xc = np.stack([x * d, y * d, d], -1)[ok]
            Xw = Xc @ Rw.T + tw
            Xt = Xw @ R_t.T + t_t
            z = Xt[:, 2]
            front = (z > 1e-6) & self.facing(j, vs[ok], us[ok], Xw, Rw, -R_t.T @ t_t)
            zz = np.where(front, z, 1.0)
            u = ((K_t[0, 0] * Xt[:, 0] / zz + K_t[0, 2]) * s).astype(int)
            v = ((K_t[1, 1] * Xt[:, 1] / zz + K_t[1, 2]) * s).astype(int)
            inb = front & (u >= 0) & (u < width) & (v >= 0) & (v < h)
            col = img_j[ok][inb]
            u, v, z = u[inb], v[inb], z[inb]
            order = np.argsort(-z)   # far first, near last wins
            canvas[v[order], u[order]] = col[order]
            zbuf[v[order], u[order]] = z[order]
        filled = np.isfinite(zbuf)
        render = canvas.copy()
        render[~filled] = 40
        ph = np.asarray(photo, dtype=np.float32)
        overlay = ph.copy()
        overlay[filled] = 0.45 * ph[filled] + 0.55 * canvas[filled]
        out = np.concatenate([ph, render, overlay], axis=1).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(out_path, quality=85)
        return str(out_path)

    # -- v2: dense rendering, align-then-judge, checkerboard figure ----------------------------------------
    def target_scale(self, target: str) -> float | None:
        """Scale of a candidate's group depth to the model: median of the scales of the group's registered members
        (the candidate's own sparse points excluded)."""
        if target not in self.cand_depth:
            return None
        gp = self.cand_depth[target][2]
        key = gp + "|" + target
        if key not in self.group_scale:
            z = self.group_files[gp]
            sc = []
            for k, n in enumerate(self.group_members[gp]):
                im = self.rec.find_image_with_name(n)
                if n == target or im is None or not im.has_pose:
                    continue
                s_, _ = self.depth_scale(n, z["depth"][k].astype(np.float32), z["transforms"][k])
                if s_ is not None:
                    sc.append(s_)
            self.group_scale[key] = float(np.median(sc)) if sc else None
        return self.group_scale[key]

    def colour(self, name: str, w: int, h: int) -> np.ndarray:
        key = (name, w, h)
        if key not in self.colour_cache:
            from PIL import Image

            self.colour_cache[key] = np.asarray(Image.open(self.images_dir / name).convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        return self.colour_cache[key]

    def render(self, target: str, P_t, nbs: list[str], width: int = 96) -> dict[str, Any]:
        """The model's view from pose P_t: RGB, valid mask. Inverse when the candidate has feedforward depth (each photo
        pixel lifted through it and looked up in the sources where their own depth agrees: hole-free, occlusion-aware),
        otherwise forward splatting into a z-buffer with each source sample's footprint."""
        K_t, W_t, H_t = self.K(target)
        h = max(1, round(H_t * width / W_t))
        s = width / W_t
        R_t, t_t = P_t.rotation.matrix(), P_t.translation
        nbs = self.usable(nbs)
        st = self.target_scale(target)
        if st is not None and nbs:
            # feedforward depth scale is not shared exactly within a group (PTZ s00: members 0.5-2x the group median), and a
            # candidate has no sparse points to measure its own: take the scale at which its depth agrees with the most source
            # pixels (low resolution). No forward fallback: GT shows it calls correct PTZ poses inconsistent.
            w0 = 48
            h0 = max(1, round(H_t * w0 / W_t))
            best = max((self._render_inverse(target, R_t, t_t, K_t, nbs, w0, h0, w0 / W_t, st * f)["cover"], f)
                       for f in (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0))
            r = self._render_inverse(target, R_t, t_t, K_t, nbs, width, h, s, st * best[1])
            r["scale_factor"] = best[1]
            return r
        return self._render_forward(R_t, t_t, K_t, nbs, width, h, s)

    def _render_inverse(self, target, R_t, t_t, K_t, nbs, width, h, s, st, tol: float = 0.1):
        d_t, tr_t, _ = self.cand_depth[target]
        sx, sy, ox, oy = tr_t
        vs, us = np.mgrid[0:h, 0:width]
        u0, v0 = (us + 0.5) / s, (vs + 0.5) / s
        mu = np.clip(np.round(u0 * sx + ox).astype(int), 0, d_t.shape[1] - 1)
        mv = np.clip(np.round(v0 * sy + oy).astype(int), 0, d_t.shape[0] - 1)
        z = d_t[mv, mu] * st
        has = z > 0
        x = (u0 - K_t[0, 2]) / K_t[0, 0]
        y = (v0 - K_t[1, 2]) / K_t[1, 1]
        Xw = (np.stack([x * z, y * z, z], -1) - t_t) @ R_t
        rgb = np.zeros((h, width, 3), np.float32)
        best = np.full((h, width), np.inf, np.float32)
        for j in nbs:
            depth_j, tr_j = self.depth[j]
            sj = self.scale(j)
            Pj = self.image(j).cam_from_world()
            Rj, tj = Pj.rotation.matrix(), Pj.translation
            K_j, W_j, H_j = self.K(j)
            Xj = Xw @ Rj.T + tj
            zj = Xj[..., 2]
            ok = has & (zj > 1e-6)
            zz = np.where(ok, zj, 1.0)
            uj = K_j[0, 0] * Xj[..., 0] / zz + K_j[0, 2]
            vj = K_j[1, 1] * Xj[..., 1] / zz + K_j[1, 2]
            ok &= (uj >= 0) & (uj < W_j - 1) & (vj >= 0) & (vj < H_j - 1)
            sxj, syj, oxj, oyj = tr_j
            dj = depth_j[np.clip(np.round(vj * syj + oyj).astype(int), 0, depth_j.shape[0] - 1),
                         np.clip(np.round(uj * sxj + oxj).astype(int), 0, depth_j.shape[1] - 1)] * sj
            rel = np.where(ok & (dj > 0), np.abs(zj - dj) / np.maximum(dj, 1e-9), np.inf)
            take = (rel < tol) & (rel < best)
            if take.any():
                cw = 256
                chh = max(1, round(H_j * cw / W_j))
                col = self.colour(j, cw, chh)
                rgb[take] = col[np.clip((vj * chh / H_j).astype(int), 0, chh - 1)[take], np.clip((uj * cw / W_j).astype(int), 0, cw - 1)[take]]
                best[take] = rel[take]
        valid = np.isfinite(best)
        return {"rgb": rgb, "valid": valid, "mode": "inverse", "cover": float(valid.mean())}

    def _render_forward(self, R_t, t_t, K_t, nbs, width, h, s):
        rgb = np.zeros((h, width, 3), np.float32)
        zbuf = np.full((h, width), np.inf, np.float32)
        c_t = -R_t.T @ t_t
        for j in nbs:
            depth_j, tr_j = self.depth[j]
            sj = self.scale(j)
            Pjw = self.image(j).cam_from_world().inverse()
            Rw, tw = Pjw.rotation.matrix(), Pjw.translation
            H, W = depth_j.shape
            step = max(1, W // 192)
            vs, us = np.mgrid[0:H:step, 0:W:step]
            d = depth_j[vs, us] * sj
            ok = d > 0
            sx, sy, ox, oy = tr_j
            K_j, W_j, H_j = self.K(j)
            x = ((us - ox) / sx - K_j[0, 2]) / K_j[0, 0]
            y = ((vs - oy) / sy - K_j[1, 2]) / K_j[1, 1]
            Xw = np.stack([x * d, y * d, d], -1)[ok] @ Rw.T + tw
            Xt = Xw @ R_t.T + t_t
            z = Xt[:, 2]
            front = (z > 1e-6) & self.facing(j, vs[ok], us[ok], Xw, Rw, c_t)
            zz = np.where(front, z, 1.0)
            u = (K_t[0, 0] * Xt[:, 0] / zz + K_t[0, 2]) * s
            v = (K_t[1, 1] * Xt[:, 1] / zz + K_t[1, 2]) * s
            r = np.clip(np.ceil((step / sx) * (K_t[0, 0] / K_j[0, 0]) * (d[ok] / zz) * s / 2).astype(int), 0, 3)
            inb = front & (u >= 0) & (u < width) & (v >= 0) & (v < h)
            cw = 256
            chh = max(1, round(H_j * cw / W_j))
            col = self.colour(j, cw, chh)
            C = col[np.clip(((vs[ok] - oy) / sy * chh / H_j).astype(int), 0, chh - 1), np.clip(((us[ok] - ox) / sx * cw / W_j).astype(int), 0, cw - 1)]
            idx = np.where(inb)[0]
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    sel = idx[(abs(dx) <= r[idx]) & (abs(dy) <= r[idx])]
                    if not len(sel):
                        continue
                    uu = (u[sel] + dx).astype(int)
                    vv = (v[sel] + dy).astype(int)
                    m = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < h)
                    uu, vv, zs, cs = uu[m], vv[m], z[sel][m], C[sel][m]
                    o = np.argsort(-zs)
                    uu, vv, zs, cs = uu[o], vv[o], zs[o], cs[o]
                    closer = zs < zbuf[vv, uu]
                    zbuf[vv[closer], uu[closer]] = zs[closer]
                    rgb[vv[closer], uu[closer]] = cs[closer]
        valid = np.isfinite(zbuf)
        return {"rgb": rgb, "valid": valid, "mode": "forward", "cover": float(valid.mean())}

    @staticmethod
    def _grad(g: np.ndarray) -> np.ndarray:
        import cv2

        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    @staticmethod
    def _ncc(a, b, m):
        if m.sum() < 50:
            return None
        x = a[m] - a[m].mean()
        y = b[m] - b[m].mean()
        return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

    def judge_v2(self, name: str, P, nbs: list[str], width: int = 96, max_shift: float = 0.3) -> dict[str, Any]:
        """Align, then judge: the render-to-photo shift that best lines up their gradients (masked NCC), the agreement
        after that shift, and the agreement without it. A correct pose needs no shift; a coarse one a small shift; a pose
        rotated 10 deg about 10-20% of the width; a wrong place does not line up at any shift (GT calibration on six
        scenes, docs/EXAMINE.md). Verdict: consistent = shift <= 0.08 and NCC >= 0.4; inconsistent = shift > 0.15 or
        NCC < 0.3; no evidence below min_cover."""
        import cv2

        nbs = self.usable(nbs)
        if not nbs:
            return {"image": name, "score": None, "verdict": "no evidence", "neighbours": nbs, "cover": 0.0}
        r = self.render(name, P, nbs, width)
        h = r["valid"].shape[0]
        out = {"image": name, "neighbours": nbs, "cover": round(r["cover"], 3), "mode": r["mode"]}
        if r["cover"] < max(self.min_cover, 0.05):
            return {**out, "score": None, "verdict": "no evidence"}
        gp = self._grad(self.colour(name, width, h).mean(-1))
        gr = self._grad(r["rgb"].mean(-1))
        vm = cv2.erode(r["valid"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        n0 = vm.sum()
        z0 = self._ncc(gp, gr, vm)
        R = int(max_shift * width)
        best = (-2.0, 0, 0)
        for dy in range(-R, R + 1):
            ys0, ys1 = max(0, -dy), min(h, h - dy)
            if ys1 <= ys0:
                continue
            for dx in range(-R, R + 1):
                xs0, xs1 = max(0, -dx), min(width, width - dx)
                m = vm[ys0:ys1, xs0:xs1]
                if m.sum() < 0.4 * n0:
                    continue
                v = self._ncc(gp[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx], gr[ys0:ys1, xs0:xs1], m)
                if v is not None and v > best[0]:
                    best = (v, dx, dy)
        if z0 is None or best[0] < -1:
            return {**out, "score": None, "verdict": "no evidence"}
        shift = float(np.hypot(best[1], best[2])) / width
        ncc = best[0]
        if shift <= 0.08 and ncc >= 0.4:
            verdict = "consistent"
        elif shift > 0.15 or ncc < 0.3:
            verdict = "inconsistent"
        else:
            verdict = "ambiguous"
        return {**out, "score": round(ncc, 3), "ncc0": round(z0, 3), "shift": round(shift, 3), "dx": round(best[1] / width, 3), "dy": round(best[2] / width, 3),
                "margin": None, "verdict": verdict}

    def figure_v2(self, name: str, P, nbs: list[str], out_path: Path, width: int = 448) -> str | None:
        """Photo | render | checkerboard (tiles alternate photo and render: on a correct pose structures continue across
        the tile borders, on a wrong one they break or the render tiles show something else)."""
        from PIL import Image

        nbs = self.usable(nbs)
        if not nbs:
            return None
        r = self.render(name, P, nbs, width=width)
        h = r["valid"].shape[0]
        ph = self.colour(name, width, h)
        ren = np.where(r["valid"][..., None], r["rgb"], 0.16)
        tile = max(16, width // 8)
        yy, xx = np.mgrid[0:h, 0:width]
        board = np.where((((yy // tile + xx // tile) % 2 == 1) & r["valid"])[..., None], ren, ph)
        sep = np.full((h, 6, 3), 1.0)
        out = (np.concatenate([ph, sep, ren, sep, board], 1) * 255).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(out_path, quality=88)
        return str(out_path)

