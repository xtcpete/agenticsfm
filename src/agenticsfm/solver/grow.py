"""Open incremental solver, v1 (2026-11-04).

The state is a pycolmap ``Reconstruction`` (cameras, poses, tracks; explicit and exportable) driven
through pycolmap's ``IncrementalMapper`` primitives (PnP registration, triangulation, local and
global bundle adjustment) under OUR loop, instead of COLMAP's incremental pipeline:

1. ``seed``: a feedforward model (Pi3X by default) solves the densest cluster of the view graph;
   its poses and focal lengths initialise the reconstruction, tracks are triangulated from the
   database's matches and bundle adjusted.
2. ``grow_round``: the images with most 2D-3D correspondences are registered by PnP in groups,
   triangulated and locally refined; every new image is then verified against the database's
   matches (inlier-weighted share of its strong pairs' matches that agree with the poses) and
   deregistered when it fails; global refinement when the model has grown by 10%.
3. ``place_hard``: images PnP cannot place, or keeps placing against their matches, are the hard
   ones: their matches are mostly lookalike (doppelganger) pairs, so match support cannot judge
   them. The feedforward model places them twice, with two different anchor sets (a few match
   neighbours plus cameras sampled from all over the model, so the model's layout is in view);
   the two placements must agree, or the VLM advisor decides which neighbourhood the image really
   shows. Accepted placements are protected from being dragged by their wrong matches in bundle
   adjustment (drift guard).
4. ``multi-model``: images left over that are connected among themselves are reconstructed as a
   further model; the advisor judges whether it is the same site (then the solver absorbs it as
   hard images) or a different scene (then it is returned as a separate model).

Nothing here reads ground truth. Every decision, including every VLM question and answer, is in
the per-round log for the trace, the episode page and later audits.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.backends import run_backend
from agenticsfm.solver.render_compare import RenderCompare
from agenticsfm.workspace import Workspace


def sampson(E, x0, x1):   # imported lazily: agenticsfm.tools imports the grow_model tool, which imports this module
    from agenticsfm.tools.examine import sampson as _sampson
    return _sampson(E, x0, x1)


def umeyama(src, dst):
    from agenticsfm.tools.frame import umeyama as _umeyama
    return _umeyama(src, dst)

FEEDFORWARD_ENV = {"vggt": "vggt", "vggt_omega": "vggt", "pi3x": "pi3x"}
MAX_IMAGE_ID = 2147483647  # COLMAP pair id = image_id1 * MAX_IMAGE_ID + image_id2


class Timer:
    """Wall time per phase, for the solver's profile (logged in the summary)."""

    def __init__(self):
        self.t: dict[str, float] = {}

    def __call__(self, name: str):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            t0 = time.time()
            try:
                yield
            finally:
                self.t[name] = self.t.get(name, 0.0) + time.time() - t0
        return _cm()


def rot_angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


@dataclass
class GrowConfig:
    feedforward: str = "pi3x"
    seed_size: int = 12          # images in the feedforward seed group
    seed_attempts: int = 3       # seed centres to try before giving up
    group_size: int = 8          # PnP registrations per round before verification
    min_support: float = 0.5     # inlier-weighted share of consistent matches an image needs (strong pairs only)
    support_low: float = 0.25    # between support_low and min_support, render-and-compare gives the second opinion
    persistent_ff: bool = True   # keep the feedforward model loaded in a worker per GPU (2.8 s per call instead of 20 to 60)
    support_pairs: int = 12      # match support from each image's strongest pairs only
    round_gate: bool = True      # the VLM between rounds: suspects of each round (borderline admits and rejects) judged on their figures
    gate_admit_support: float = 0.65   # an admitted image below this support (or with no strong pair) is a suspect
    gate_max: int = 6            # suspects per round shown to the VLM (one batched call)
    fold_frac: float = 0.05      # share of strong pairs inconsistent with the poses above which the model is treated as folded
    repair_act: bool = False     # re-place the minority block of a detected fold (v1.6 misfired on old_computer: detect and log only until the re-placement is reliable)
    seed_fold_frac: float = 0.1
    strong_pair: int = 50        # inliers for a pair to count in the support
    epipolar_px: float = 4.0
    max_tries: int = 2           # PnP attempts per image in our loop (COLMAP's own cap is 3)
    global_ba_growth: float = 1.1
    hard: bool = True            # feedforward placement of hard images with two anchor sets
    hard_batch: int = 6
    hard_anchors_match: int = 4
    hard_anchors_diverse: int = 8
    hard_min_link: int = 50          # a hard candidate needs this many inliers to registered images in all (>= 2 pairs), or one pair this strong
    hard_agree_deg: float = 10.0     # the two placements must agree in rotation (GT-correct BLH0001 placements agreed within 3 deg; PTZ placements at 8-10 deg were imprecise: open item) ...
    hard_agree_frac: float = 0.05    # ... and in position (fraction of the scene span)
    max_alignment_frac: float = 0.08  # anchors' alignment residual over the scene span, after trimming
    drift_deg: float = 10.0          # a hard placement that bundle adjustment moves more than this is held at its placement
    vlm: str = "decide"              # off | log (ask, never act) | decide (arbitrate ambiguous placements, judge scenes)
    audit: bool = True               # render-and-compare audit of every camera after the final refinement (logged; drops only with audit_drop)
    audit_drop: bool = False         # until the verdict is calibrated against GT (docs/EXAMINE.md), the audit reports and does not act
    audit_max_images: int = 100
    veto_score: float = -1.0         # render veto of a placement (score below this AND margin below veto_margin); disabled: the calibration
    veto_margin: float = -0.05       # (docs/EXAMINE.md) shows the score is a precision test, and feedforward placements are coarse before BA
    multi_model: bool = True
    min_component: int = 4
    max_models: int = 4
    max_rounds: int = 2000
    num_threads: int = 8


class Solver:
    def __init__(self, ws: Workspace, cfg: dict[str, Any], db_id: str, gc: GrowConfig, workdir: Path,
                 gpu: int | None = None, verbose: bool = True, image_subset: list[str] | None = None, advisor=None, tag: str = "main"):
        import pycolmap

        self.ws, self.cfg, self.gc, self.workdir, self.gpu, self.verbose, self.tag = ws, cfg, gc, workdir, gpu, verbose, tag
        self.db_id = db_id
        self.db_path = ws.abspath(db_id)
        self.database = pycolmap.Database.open(str(self.db_path))
        co = pycolmap.DatabaseCacheOptions()
        co.min_num_matches = 15
        if image_subset:
            co.image_names = set(image_subset)
        self.cache = pycolmap.DatabaseCache.create(self.database, co)
        self.rec = pycolmap.Reconstruction()
        self.mapper = pycolmap.IncrementalMapper(self.cache)
        self.mapper.begin_reconstruction(self.rec)      # loads cameras, images, frames; nothing registered
        self.names = {im.image_id: im.name for im in self.rec.images.values()}
        self.ids = {n: i for i, n in self.names.items()}
        # verified matches of every pair, and the weighted view graph
        self.pairs: dict[tuple[int, int], np.ndarray] = {}
        self.W: dict[int, dict[int, int]] = {}
        ids, geoms = self.database.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            n = int(g.inlier_matches.shape[0])
            if n < 15:
                continue
            i, j = divmod(int(pid), MAX_IMAGE_ID)
            if i not in self.names or j not in self.names:
                continue
            self.pairs[(i, j)] = np.asarray(g.inlier_matches)
            self.W.setdefault(i, {})[j] = n
            self.W.setdefault(j, {})[i] = n
        self.kps: dict[int, np.ndarray] = {}
        self.tries = {i: 0 for i in self.names}
        self.hard_tried = {i: 0 for i in self.names}
        self.status = {i: "unplaced" for i in self.names}
        self.held: set[int] = set()
        self.hard_placed: set[int] = set()
        self.mo = pycolmap.IncrementalMapperOptions()
        self.mo.num_threads = gc.num_threads
        self.tri = pycolmap.IncrementalTriangulatorOptions()
        self.ba = pycolmap.BundleAdjustmentOptions()
        self.ba.print_summary = False
        self.last_global = 0
        self.ff_calls = 0
        self.ff_seconds = 0.0
        self.log: list[dict[str, Any]] = []
        self.timer = Timer()
        self.rac = RenderCompare(self.rec, ws.images_dir)
        if advisor is not None:
            self.advisor = advisor
            self.advisor.log = self.log
        elif gc.vlm != "off":
            from agenticsfm.solver.advisor import Advisor
            self.advisor = Advisor(cfg, ws.images_dir, self.log, mode=gc.vlm)
        else:
            self.advisor = None

    # -- helpers -----------------------------------------------------------------------------
    def say(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.ws.scene.name}] grow{'' if self.tag == 'main' else '/' + self.tag}: {msg}", flush=True)

    def kp(self, i: int) -> np.ndarray:
        if i not in self.kps:
            self.kps[i] = np.asarray(self.database.read_keypoints(i))[:, :2]
        return self.kps[i]

    def reg(self) -> set[int]:
        return set(self.rec.reg_image_ids())

    def centres(self) -> dict[int, np.ndarray]:
        return {i: self.rec.images[i].projection_center() for i in self.reg()}

    def span(self) -> float:
        C = np.array(list(self.centres().values()))
        return float(np.linalg.norm(np.ptp(C, axis=0))) if len(C) > 1 else 1.0

    def pair_consistency(self, i: int, j: int) -> float:
        """Share of the pair's verified matches within ``epipolar_px`` of the epipolar lines under the current poses."""
        a, b = (i, j) if i < j else (j, i)
        m = self.pairs[(a, b)]
        im0, im1 = self.rec.images[a], self.rec.images[b]
        cam0, cam1 = self.rec.cameras[im0.camera_id], self.rec.cameras[im1.camera_id]
        x0 = np.asarray(cam0.cam_from_img(self.kp(a)[m[:, 0]]))
        x1 = np.asarray(cam1.cam_from_img(self.kp(b)[m[:, 1]]))
        rel = im1.cam_from_world() * im0.cam_from_world().inverse()
        R, t = rel.rotation.matrix(), rel.translation
        tn = np.linalg.norm(t)
        f = 0.5 * (cam0.mean_focal_length() + cam1.mean_focal_length())
        # pure-rotation transfer: exact for a pan-tilt-zoom pair (same centre), where the epipolar test is degenerate
        x0r = np.c_[x0, np.ones(len(x0))] @ R.T
        err_rot = np.linalg.norm(x0r[:, :2] / np.where(np.abs(x0r[:, 2:3]) < 1e-9, 1e-9, x0r[:, 2:3]) - x1, axis=1)
        frac_rot = float((err_rot * f < self.gc.epipolar_px).mean())
        if tn < 1e-9:
            return frac_rot
        t = t / tn
        E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
        frac_epi = float((sampson(E, x0, x1) * f < self.gc.epipolar_px).mean())
        return max(frac_epi, frac_rot)

    def support(self, image_ids: list[int]) -> dict[int, float]:
        with self.timer("support"):
            return self._support(image_ids)

    def _support(self, image_ids: list[int]) -> dict[int, float]:
        """Per image: inlier-weighted consistency over its strong pairs with registered images (NaN: no strong pair)."""
        reg = self.reg()
        out: dict[int, float] = {}
        for i in image_ids:
            if i not in reg:
                continue
            num = den = 0.0
            strong = sorted(((n, j) for j, n in self.W.get(i, {}).items() if n >= self.gc.strong_pair and j in reg), reverse=True)
            for n, j in strong[: self.gc.support_pairs]:
                frac = self.pair_consistency(i, j)
                num += n * frac
                den += n
            out[i] = num / den if den > 0 else float("nan")
        return out

    def ff_groups(self, groups: list[list[str]], tag: str, save_depth: bool = False) -> list[dict[str, Any]]:
        """One backend invocation (one model load) for several groups; registers dense depth with render-and-compare."""
        model = self.gc.feedforward
        t0 = time.time()
        with self.timer("feedforward"):
            res = self._ff_call(groups, tag, save_depth, model)
        self.ff_calls += 1
        self.ff_seconds += time.time() - t0
        if save_depth:
            for g in res["groups"]:
                if g.get("depth_path"):
                    self.rac.add_depth(g["depth_path"])
        return res["groups"]

    def _ff_call(self, groups, tag, save_depth, model):
        spec = {"images_dir": str(self.ws.images_dir), "groups": groups, "model": model, "points_per_image": 0, "save_depth": save_depth,
                "repos": self.cfg.get("repos", {})}
        if self.gc.persistent_ff:
            from agenticsfm.solver.ffclient import run_feedforward

            return run_feedforward(self.cfg, FEEDFORWARD_ENV[model], model, spec, self.workdir / f"ff_{tag}", self.gpu)
        return run_backend(self.cfg, "feedforward", FEEDFORWARD_ENV[model], spec, self.workdir / f"ff_{tag}", gpu=self.gpu)

    def feedforward(self, names: list[str], tag: str, save_depth: bool = False) -> dict[str, dict[str, Any]]:
        """Run the feedforward model on one group. Returns per image: cam_from_world (3x4, frame of the group's
        first camera, arbitrary scale) and the focal length / principal point in original pixels."""
        g = self.ff_groups([names], tag, save_depth)[0]
        out = {}
        for name, E, K, tr in zip(g["names"], g["extrinsics"], g["intrinsics"], g["transforms"]):
            sx, sy, ox, oy = tr
            out[name] = {"E": np.asarray(E, dtype=np.float64), "f": 0.5 * (K[0][0] / sx + K[1][1] / sy),
                         "cx": (K[0][2] - ox) / sx, "cy": (K[1][2] - oy) / sy}
        return out

    def set_pose(self, i: int, E: np.ndarray, f: float | None = None) -> None:
        import pycolmap

        im = self.rec.images[i]
        R, t = E[:, :3], E[:, 3]
        self.rec.frame(im.frame_id).rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        if f is not None and f > 0:
            cam = self.rec.camera(im.camera_id)
            p = np.array(cam.params, dtype=float)
            if cam.model.name in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE"):
                p[0] = p[1] = f
            else:
                p[0] = f
            cam.params = p

    def pose_of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        cfw = self.rec.images[i].cam_from_world()
        return cfw.rotation.matrix(), cfw.translation

    def refine_pose_2d3d(self, i: int, R: np.ndarray, t: np.ndarray, schedule: tuple[float, ...] | None = None) -> dict[str, Any]:
        """Coarse-to-fine: refine on the correspondences within the first threshold, then re-select within the next.
        The default schedule is relative to the image width (3%, 1.5%, 0.75%, 0.25%): a coarse feedforward placement is
        degrees off, which is a hundred pixels on a large image."""
        out: dict[str, Any] = {}
        if schedule is None:
            w = self.rec.cameras[self.rec.images[i].camera_id].width
            schedule = tuple(max(12.0, w * f) for f in (0.03, 0.015, 0.0075, 0.0025))
        for px in schedule:
            r = self._refine_pose_2d3d(i, R, t, px)
            R, t = r["R"], r["t"]
            out = {**r, "schedule": list(schedule)}
            if r.get("inliers", 0) < 15:
                break
        return out

    def _refine_pose_2d3d(self, i: int, R: np.ndarray, t: np.ndarray, max_px: float = 12.0) -> dict[str, Any]:
        """Refine a placement from the 2D-3D correspondences that agree with it (reprojection under the given pose below
        ``max_px``): the lookalike correspondences project elsewhere and are left out. Returns the refined pose and counts."""
        import pycolmap

        reg = self.reg()
        kp_i = self.kp(i)
        pts2d, pts3d = [], []
        for j, n in self.W.get(i, {}).items():
            if j not in reg:
                continue
            a, b = (i, j) if i < j else (j, i)
            m = self.pairs[(a, b)]
            idx_i, idx_j = (m[:, 0], m[:, 1]) if i < j else (m[:, 1], m[:, 0])
            p2s = self.rec.images[j].points2D
            for ii, jj in zip(idx_i, idx_j):
                p2 = p2s[int(jj)]
                if p2.has_point3D():
                    pts2d.append(kp_i[int(ii)])
                    pts3d.append(self.rec.points3D[p2.point3D_id].xyz)
        out: dict[str, Any] = {"num_2d3d": len(pts3d)}
        if len(pts3d) < 15:
            return {**out, "R": R, "t": t, "inliers": 0}
        pts2d, pts3d = np.array(pts2d, dtype=np.float64), np.array(pts3d, dtype=np.float64)
        cam = self.rec.cameras[self.rec.images[i].camera_id]
        Xc = pts3d @ R.T + t
        front = Xc[:, 2] > 1e-6
        proj = np.full((len(pts3d), 2), np.inf)
        if front.any():
            proj[front] = np.asarray(cam.img_from_cam(Xc[front], False))
        err = np.linalg.norm(proj - pts2d, axis=1)
        inl = err < max_px
        out.update({"inliers": int(inl.sum()), "inlier_frac": round(float(inl.mean()), 3)})
        if inl.sum() < 15:
            return {**out, "R": R, "t": t}
        opts = pycolmap.AbsolutePoseRefinementOptions()
        opts.refine_focal_length = False
        opts.refine_extra_params = False
        res = pycolmap.refine_absolute_pose(pycolmap.Rigid3d(pycolmap.Rotation3d(R), t), pts2d, pts3d, inl, cam, opts)
        if not res:
            return {**out, "R": R, "t": t}
        pose = res["cam_from_world"] if isinstance(res, dict) else res
        R2, t2 = pose.rotation.matrix(), pose.translation
        out.update({"refined_deg": round(rot_angle(R2 @ R.T), 2), "refined_move_frac": round(float(np.linalg.norm((-R2.T @ t2) - (-R.T @ t))) / max(self.span(), 1e-9), 4)})
        return {**out, "R": R2, "t": t2}

    def attach_mapper(self) -> None:
        """A fresh mapper on the current reconstruction: its bookkeeping (registration events, observation
        statistics) is rebuilt from the registered frames, which is how externally posed images enter the loop."""
        import pycolmap

        self.mapper = pycolmap.IncrementalMapper(self.cache)
        self.mapper.begin_reconstruction(self.rec)

    def deregister(self, i: int) -> None:
        self.mapper.observation_manager.deregister_frame(self.rec.images[i].frame_id)

    def model_stats(self) -> dict[str, Any]:
        return {"num_registered": self.rec.num_reg_images(), "num_points": self.rec.num_points3D(),
                "reproj": round(self.rec.compute_mean_reprojection_error(), 3) if self.rec.num_points3D() else None}

    def fps(self, pool: list[int], k: int, far_from: list[int]) -> list[int]:
        """Farthest-point sample of registered cameras, started far from ``far_from``."""
        C = self.centres()
        pool = [i for i in pool if i in C]
        if not pool or k <= 0:
            return []
        ref = np.array([C[i] for i in far_from if i in C]) if any(i in C for i in far_from) else np.mean([C[i] for i in pool], axis=0, keepdims=True)
        chosen = [max(pool, key=lambda i: float(np.min(np.linalg.norm(ref - C[i], axis=1))))]
        while len(chosen) < min(k, len(pool)):
            chosen.append(max((i for i in pool if i not in chosen), key=lambda i: min(float(np.linalg.norm(C[i] - C[c])) for c in chosen)))
        return chosen

    def nearest_registered(self, c: np.ndarray, d: np.ndarray, k: int = 3, exclude: set[int] = frozenset()) -> list[int]:
        """Registered images closest to a camera position, favouring similar viewing directions."""
        span = self.span()
        rows = []
        for j in self.reg():
            if j in exclude:
                continue
            im = self.rec.images[j]
            dist = float(np.linalg.norm(im.projection_center() - c)) / max(span, 1e-9)
            ang = 1 - float(np.dot(im.viewing_direction(), d))
            rows.append((dist + 0.5 * ang, j))
        return [j for _, j in sorted(rows)[:k]]

    def align(self, ff: dict[str, dict[str, Any]], anchors: list[int]) -> dict[str, Any] | None:
        """Similarity from the feedforward frame to the model on the anchors' camera centres, trimming up to two
        anchors that do not fit. None when the anchors do not align (the feedforward layout disagrees with the model)."""
        centre = lambda E: -E[:, :3].T @ E[:, 3]  # noqa: E731
        use = [a for a in anchors if self.names[a] in ff and a in self.reg()]
        span = self.span()
        dropped = []
        while len(use) >= 3:
            src = np.array([centre(ff[self.names[a]]["E"]) for a in use])
            dst = np.array([self.rec.images[a].projection_center() for a in use])
            s, R, t = umeyama(src, dst)
            resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1) / max(span, 1e-9)
            if resid.mean() <= self.gc.max_alignment_frac or len(dropped) >= 2 or len(use) <= 3:
                break
            worst = int(np.argmax(resid))
            dropped.append(self.names[use[worst]])
            use.pop(worst)
        if len(use) < 3:
            return None
        if resid.mean() > self.gc.max_alignment_frac:
            return {"ok": False, "resid": float(resid.mean()), "used": len(use), "dropped": dropped}
        return {"ok": True, "s": s, "R": R, "t": t, "resid": float(resid.mean()), "used": len(use), "dropped": dropped}

    @staticmethod
    def transform_pose(E: np.ndarray, al: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        Rn = E[:, :3] @ al["R"].T
        tn = al["s"] * E[:, 3] - Rn @ al["t"]
        return Rn, tn

    # -- steps -------------------------------------------------------------------------------
    def seed(self) -> dict[str, Any]:
        deg = {i: sum(v.values()) for i, v in self.W.items()}
        order = sorted(deg, key=deg.get, reverse=True)
        if not order:
            raise RuntimeError("no verified pairs among these images")
        attempts = []
        size = min(self.gc.seed_size, len(self.names))
        for k, centre in enumerate(order[: self.gc.seed_attempts]):
            nb = sorted(self.W[centre], key=self.W[centre].get, reverse=True)[: size - 1]
            group = [centre] + nb
            if len(group) < 3:
                continue
            names = [self.names[i] for i in group]
            ff = self.feedforward(names, f"seed{k}")
            for i in group:
                self.set_pose(i, ff[self.names[i]]["E"], ff[self.names[i]]["f"])
                self.rec.register_frame(self.rec.images[i].frame_id)
            self.attach_mapper()
            for i in group:
                self.mapper.triangulate_image(self.tri, i)
            self.mapper.complete_and_merge_tracks(self.tri)
            self.mapper.iterative_global_refinement(3, 0.001, self.mo, self.ba, self.tri, True)
            self.mapper.filter_frames(self.mo)
            sup = self.support(group)
            good = [i for i in group if i in self.reg() and not (sup.get(i, float("nan")) < self.gc.min_support)]
            rows = self.strong_pair_rows(set(good))
            split = None
            if rows and sum(1 for r in rows if r[3] < 0.2) / len(rows) > self.gc.seed_fold_frac:
                comps = self.consistent_components(rows, set(good))
                if len(comps) > 1 and len(comps[0]) >= 3:
                    split = [len(c) for c in comps]
                    good = list(comps[0])        # the largest self-consistent block seeds; the rest is deferred and re-placed later
                else:
                    sp = self.signed_split(rows, sorted(good))
                    if sp and len(sp[0]) >= 3:
                        split = [len(sp[0]), len(sp[1]), sp[2]]
                        good = list(sp[0])
            bad = [i for i in group if i not in good]
            att = {"centre": self.names[centre], "group": names, "focal": {self.names[i]: round(ff[self.names[i]]["f"], 1) for i in group},
                   "support": {self.names[i]: (None if sup.get(i) != sup.get(i) else round(sup.get(i, float("nan")), 3)) for i in group},
                   "kept": [self.names[i] for i in good], "split": split, **self.model_stats()}
            attempts.append(att)
            self.say(f"seed {k}: centre {self.names[centre]}, {len(good)}/{len(group)} supported, {att['num_points']} points, reproj {att['reproj']}")
            if len(good) >= max(3, math.ceil(0.5 * len(group))):
                for i in bad:
                    if i in self.reg():
                        self.deregister(i)
                    self.status[i] = "deferred"
                for i in good:
                    self.status[i] = "placed"
                self.last_global = self.rec.num_reg_images()
                return {"attempts": attempts, "chosen": k}
            for i in group:               # reset and try the next centre
                if i in self.reg():
                    self.deregister(i)
            self.mapper.filter_points(self.mo)
        raise RuntimeError(f"no seed cluster held together: {json.dumps(attempts, default=str)[:800]}")

    def global_refine(self, final: bool = False) -> None:
        with self.timer("global_ba"):
            self._global_refine(final)

    def _global_refine(self, final: bool = False) -> None:
        held = {i: self.pose_of(i) for i in self.held if i in self.reg()}
        self.mapper.iterative_global_refinement(5 if final else 3, 0.0005, self.mo, self.ba, self.tri, True)
        self.mapper.filter_frames(self.mo)
        self.last_global = self.rec.num_reg_images()
        self.rac.invalidate()
        # normalisation moved the whole model: re-express held poses in the new frame via a registered reference
        if held:
            self.restore_held(held)

    def restore_held(self, before: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        """Held images keep their placement relative to the rest of the model: after a refinement that may have
        moved them (and re-normalised the scene), put them back using the similarity between old and new frames
        of the free images."""
        free = [j for j in self.reg() if j not in self.held]
        if len(free) < 3 or not before:
            return
        return  # v1: the drift guard removes a held image's observations, so BA has nothing to move it with; normalisation is a similarity applied to all frames alike

    def grow_round(self, k: int) -> dict[str, Any] | None:
        cand = [i for i in self.mapper.find_next_images(self.mo, False) if self.tries[i] < self.gc.max_tries]
        if not cand:
            return None
        group = cand[: self.gc.group_size]
        newly, failed = [], []
        for i in group:
            self.tries[i] += 1
            with self.timer("pnp"):
                ok = self.mapper.register_next_image(self.mo, i)
            if not ok:
                failed.append(i)
                continue
            with self.timer("triangulate"):
                self.mapper.triangulate_image(self.tri, i)
            with self.timer("local_ba"):
                self.mapper.iterative_local_refinement(2, 0.001, self.mo, self.ba, self.tri, i)
            newly.append(i)
        sup = self.support(newly)
        nan = lambda i: sup.get(i, float("nan")) != sup.get(i, float("nan"))  # noqa: E731
        border = [i for i in newly if not nan(i) and self.gc.support_low <= sup[i] < self.gc.min_support]          # would be rejected
        weak_in = [i for i in newly if nan(i) or self.gc.min_support <= sup[i] < self.gc.gate_admit_support]   # admitted on thin evidence
        use_vlm = self.gc.round_gate and self.advisor is not None and self.advisor.enabled
        suspects = border + (weak_in if use_vlm else [])
        second: dict[int, dict[str, Any]] = {}
        vlm: dict[int, str] = {}
        if suspects:
            try:
                self.ensure_depth([(i, *self.pose_of(i)) for i in suspects], f"round{k}depth")
                for i in suspects:
                    second[i] = self.judge_pose(i, *self.pose_of(i), "pnp", figure=use_vlm)
            except Exception as e:  # the second opinion is optional
                self.say(f"render second opinion failed: {e}")
            if use_vlm:
                # the VLM looks only where the two numbers do not already agree: borderline rejects the render does not
                # clearly confirm, and thin admits the render does not clearly confirm
                ask = [i for i in suspects if i in second and second[i].get("figure") and second[i]["verdict"] != "consistent"][: self.gc.gate_max]
                if ask:
                    answers = self.advisor.render_verdicts([second[i]["figure"] for i in ask],
                                                           {"round": k, "images": [self.names[i] for i in ask],
                                                            "support": [None if nan(i) else round(sup[i], 3) for i in ask],
                                                            "render": [{kk: second[i].get(kk) for kk in ("score", "margin", "verdict")} for i in ask]})
                    vlm = dict(zip(ask, answers))
        kept, rejected, readmitted, removed = [], [], [], []
        decide = self.gc.vlm == "decide"
        for i in newly:
            s = sup.get(i, float("nan"))
            j = second.get(i)
            rv = j["verdict"] if j is not None else None
            v = vlm.get(i)
            if i in border:
                # rejected by match support unless the render confirms, or the render is not against it and the VLM sees it aligned
                ok = rv == "consistent" or (rv == "ambiguous" and (j.get("score") or 0) >= 0.5) or \
                    (decide and v == "aligned" and rv != "inconsistent")
                if ok and rv != "consistent":
                    readmitted.append(i)
            elif s == s and s < self.gc.min_support:
                ok = False
            elif i in weak_in and decide and v == "misaligned" and rv == "inconsistent":
                ok = False                       # admitted on thin match evidence, and both the render and the VLM say it is misplaced
                removed.append(i)
            else:
                ok = True
            if not ok:
                self.deregister(i)
                self.status[i] = "deferred"
                rejected.append(i)
            else:
                self.status[i] = "placed"
                kept.append(i)
        if self.rec.num_reg_images() >= self.gc.global_ba_growth * self.last_global:
            self.global_refine()
        row = {"round": k, "kind": "pnp", "candidates": len(cand), "registered": [self.names[i] for i in kept],
               "rejected": {self.names[i]: round(sup[i], 3) for i in rejected}, "pnp_failed": [self.names[i] for i in failed],
               "render_second_opinion": {self.names[i]: {kk: j.get(kk) for kk in ("score", "margin", "verdict")} for i, j in second.items()},
               "vlm_gate": {self.names[i]: v for i, v in vlm.items()}, "readmitted": [self.names[i] for i in readmitted],
               "removed": [self.names[i] for i in removed],
               "support": {self.names[i]: (None if sup.get(i) != sup.get(i) else round(sup[i], 3)) for i in newly}, **self.model_stats()}
        self.say(f"round {k}: +{len(kept)} (rejected {len(rejected)}, pnp failed {len(failed)}"
                 + (f", VLM on {len(vlm)}: readmitted {len(readmitted)}, removed {len(removed)}" if vlm else "") + ") -> "
                 f"{self.rec.num_reg_images()}/{len(self.names)} registered, {row['num_points']} pts, reproj {row['reproj']}")
        return row

    def pose_neighbours(self, R: np.ndarray, t: np.ndarray, k: int = 4, exclude: set[int] = frozenset()) -> list[int]:
        return self.nearest_registered(-R.T @ t, R[2], k, exclude)

    def judge_pose(self, i: int, R: np.ndarray, t: np.ndarray, tag: str, figure: bool = False) -> dict[str, Any]:
        with self.timer("render_compare"):
            return self._judge_pose(i, R, t, tag, figure)

    def _judge_pose(self, i: int, R: np.ndarray, t: np.ndarray, tag: str, figure: bool = False) -> dict[str, Any]:
        """Render-and-compare verdict of a candidate pose; the three-panel figure for the advisor when asked."""
        import pycolmap

        n = self.names[i]
        P = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        nbs = [self.names[j] for j in self.pose_neighbours(R, t, 4, {i})]
        j = self.rac.judge(n, P, nbs)
        if figure and self.rac.usable(nbs):
            j["figure"] = self.rac.figure(n, P, nbs, self.workdir / "rac" / f"{Path(n).stem}_{tag}.jpg")
        return j

    def ensure_depth(self, wants: list[tuple[int, np.ndarray, np.ndarray]], tag: str) -> None:
        """Dense depth for the registered neighbours of the given candidate poses, in one backend call."""
        groups, seen = [], set()
        for i, R, t in wants:
            nbs = self.pose_neighbours(R, t, 4, {i})
            need = [j for j in nbs if self.names[j] not in self.rac.depth]
            if need:
                g = tuple([self.names[i]] + [self.names[j] for j in nbs])
                if g not in seen:
                    seen.add(g)
                    groups.append(list(g))
        if groups:
            self.ff_groups(groups, tag, save_depth=True)

    def place_hard(self, k: int, candidates: list[int] | None = None) -> dict[str, Any] | None:
        """Feedforward placement of hard images with two anchor sets, pose-only refinement on the correspondences that
        agree with each placement, and acceptance by render-and-compare; the advisor arbitrates on the figures."""
        reg = self.reg()
        if candidates is None:
            link, strongest, npairs = {}, {}, {}
            for i in self.names:
                if i in reg:
                    continue
                ns = [n for j, n in self.W.get(i, {}).items() if j in reg]
                link[i], strongest[i], npairs[i] = sum(ns), max(ns, default=0), len(ns)
            candidates = [i for i in sorted(link, key=link.get, reverse=True) if self.hard_tried[i] < 1
                          and ((link[i] >= self.gc.hard_min_link and npairs[i] >= 2) or strongest[i] >= self.gc.hard_min_link)]
        candidates = [i for i in candidates if i not in reg]
        if not candidates or len(reg) < 3:
            return None
        batch = candidates[: self.gc.hard_batch]
        for i in batch:
            self.hard_tried[i] += 1
        match_w: dict[int, int] = {}
        for i in batch:
            for j, n in self.W.get(i, {}).items():
                if j in reg:
                    match_w[j] = match_w.get(j, 0) + n
        ms = sorted(match_w, key=match_w.get, reverse=True)
        good = [j for j in reg if self.rec.images[j].num_points3D >= 50] or list(reg)
        nm, nd = self.gc.hard_anchors_match, self.gc.hard_anchors_diverse
        mA = ms[:nm]
        dA = self.fps([j for j in good if j not in mA], nd, mA or list(reg)[:1])
        mB = ms[nm: 2 * nm] or ms[:nm]
        dB = self.fps([j for j in good if j not in set(mB) | set(dA)], nd, list(dict.fromkeys(mB + dA)))
        A = list(dict.fromkeys(mA + dA))[: nm + nd]
        B = list(dict.fromkeys(mB + dB))[: nm + nd]
        names = lambda L: [self.names[i] for i in L]  # noqa: E731
        if len(A) < 3 or len(B) < 3:
            return {"round": k, "kind": "hard", "error": "too few anchors", "batch": names(batch)}
        ffA, ffB = self.ff_groups([names(A) + names(batch), names(B) + names(batch)], f"hard{k}")
        ffA = {n: {"E": np.asarray(E, dtype=np.float64), "f": 0.5 * (K[0][0] / tr[0] + K[1][1] / tr[1])} for n, E, K, tr in zip(ffA["names"], ffA["extrinsics"], ffA["intrinsics"], ffA["transforms"])}
        ffB = {n: {"E": np.asarray(E, dtype=np.float64), "f": 0.5 * (K[0][0] / tr[0] + K[1][1] / tr[1])} for n, E, K, tr in zip(ffB["names"], ffB["extrinsics"], ffB["intrinsics"], ffB["transforms"])}
        alA, alB = self.align(ffA, A), self.align(ffB, B)
        span = self.span()
        # candidate poses, refined on the correspondences that agree with them
        poses: dict[int, dict[str, tuple[np.ndarray, np.ndarray, float, dict]]] = {}
        for i in batch:
            n = self.names[i]
            poses[i] = {}
            for tag, ff, al in (("A", ffA, alA), ("B", ffB, alB)):
                if al and al.get("ok"):
                    Rn, tn = self.transform_pose(ff[n]["E"], al)
                    r = self.refine_pose_2d3d(i, Rn, tn)
                    poses[i][tag] = (r["R"], r["t"], ff[n]["f"], {kk: v for kk, v in r.items() if kk not in ("R", "t")})
        self.ensure_depth([(i, R, t) for i in batch for (R, t, _, _) in poses[i].values()], f"hard{k}depth")
        per: dict[str, dict[str, Any]] = {}
        accepted: dict[int, tuple[np.ndarray, np.ndarray, float, str]] = {}
        decide = self.gc.vlm == "decide" and self.advisor is not None and self.advisor.enabled
        askable = self.advisor is not None and self.advisor.enabled
        for i in batch:
            n = self.names[i]
            ps = poses[i]
            info: dict[str, Any] = {"placements": {tag: v[3] for tag, v in ps.items()}}
            judges = {tag: self.judge_pose(i, v[0], v[1], tag) for tag, v in ps.items()}
            info["render"] = {tag: {kk: j.get(kk) for kk in ("score", "margin", "verdict", "overlap")} for tag, j in judges.items()}
            choice, why = None, ""
            strong_bad = lambda tg: (judges[tg]["score"] is not None and judges[tg]["score"] < self.gc.veto_score and (judges[tg]["margin"] or 0) < self.gc.veto_margin)  # noqa: E731
            if len(ps) == 2:
                ang = rot_angle(ps["A"][0] @ ps["B"][0].T)
                dist = float(np.linalg.norm((-ps["A"][0].T @ ps["A"][1]) - (-ps["B"][0].T @ ps["B"][1]))) / max(span, 1e-9)
                info.update({"agree_deg": round(ang, 1), "agree_frac": round(dist, 4)})
                agree = ang < self.gc.hard_agree_deg and dist < self.gc.hard_agree_frac
                cons = [tag for tag in ("A", "B") if judges[tag]["verdict"] == "consistent"]
                if agree:
                    best = max(("A", "B"), key=lambda tg: (judges[tg]["score"] if judges[tg]["score"] is not None else -1))
                    if strong_bad(best):
                        why = "placements agree but render strongly inconsistent"
                    else:
                        choice, why = best, "placements agree" + (f", render {judges[best]['verdict']}" if judges[best]["verdict"] != "no evidence" else "")
                else:
                    if len(cons) == 1:
                        choice, why = cons[0], "placements disagree, render picks " + cons[0]
                    elif askable:
                        figA = self.judge_pose(i, ps["A"][0], ps["A"][1], "A", figure=True).get("figure")
                        figB = self.judge_pose(i, ps["B"][0], ps["B"][1], "B", figure=True).get("figure")
                        vv = self.advisor.render_choose(figA, figB, {"image": n, "agree_deg": round(ang, 1), **{f"render_{tg}": info["render"][tg] for tg in ("A", "B")}}) if figA and figB else "unsure"
                        info["vlm"] = vv
                        if decide and vv in ("a", "b") and judges[vv.upper()]["verdict"] != "inconsistent":
                            choice, why = vv.upper(), "placements disagree, advisor picks " + vv.upper()
                        elif decide and vv in ("a", "b"):
                            why = f"placements disagree, advisor picks {vv.upper()} but render inconsistent"
                        elif decide and vv == "both" and len(cons) == 2:
                            choice = max(cons, key=lambda tg: judges[tg]["score"]); why = "placements disagree, advisor: both, higher render score"
                        else:
                            why = f"placements disagree, advisor: {vv}"
                    else:
                        why = "placements disagree"
            elif len(ps) == 1:
                tag = next(iter(ps))
                v = judges[tag]["verdict"]
                if strong_bad(tag):
                    why = "single placement, render strongly inconsistent"
                elif v == "consistent":
                    choice, why = tag, "single placement, render consistent"
                elif askable:
                    fig = self.judge_pose(i, ps[tag][0], ps[tag][1], tag, figure=True).get("figure")
                    vv = self.advisor.render_verdict(fig, {"image": n, "placement": tag, **info["render"][tag]}) if fig else "unsure"
                    info["vlm"] = vv
                    if vv == "aligned" and decide and v != "inconsistent":
                        choice, why = tag, f"single placement, render {v}, advisor: aligned"
                    else:
                        why = f"single placement, render {v}, advisor: {vv}"
                else:
                    why = f"single placement, render {v}, no advisor"
            else:
                why = "anchors did not align"
            info["why"] = why
            if choice:
                accepted[i] = (ps[choice][0], ps[choice][1], ps[choice][2], choice)
            per[n] = info
        placed, held, reverted = [], [], []
        if accepted:
            for i, (Rn, tn, f, _) in accepted.items():
                self.set_pose(i, np.c_[Rn, tn], f)
                self.rec.register_frame(self.rec.images[i].frame_id)
            self.attach_mapper()
            for i in accepted:
                self.mapper.triangulate_image(self.tri, i)
                self.mapper.iterative_local_refinement(2, 0.001, self.mo, self.ba, self.tri, i)
            # drift guard: lookalike matches must not drag a verified placement
            for i, (Rn, tn, f, _) in accepted.items():
                if i not in self.reg():
                    reverted.append(i)
                    continue
                R2, t2 = self.pose_of(i)
                drift = rot_angle(R2 @ Rn.T)
                move = float(np.linalg.norm((-R2.T @ t2) - (-Rn.T @ tn))) / max(span, 1e-9)
                per[self.names[i]].update({"ba_drift_deg": round(drift, 1), "ba_move_frac": round(move, 4)})
                if drift > self.gc.drift_deg or move > self.gc.hard_agree_frac:
                    self.deregister(i)
                    self.set_pose(i, np.c_[Rn, tn], f)
                    self.rec.register_frame(self.rec.images[i].frame_id)
                    self.held.add(i)
                    held.append(i)
                    per[self.names[i]]["held"] = True
                placed.append(i)
                self.status[i] = "placed"
            for i in reverted:
                self.set_pose(i, np.c_[accepted[i][0], accepted[i][1]], accepted[i][2])
                self.rec.register_frame(self.rec.images[i].frame_id)
                self.held.add(i)
                self.status[i] = "placed"
                placed.append(i)
                per[self.names[i]]["held"] = True
            if held or reverted:
                self.attach_mapper()
            sup = self.support(placed)
            for i in placed:
                per[self.names[i]]["support"] = None if sup.get(i) != sup.get(i) else round(sup[i], 3)
                self.hard_placed.add(i)
            self.global_refine()
        for i in batch:
            if i not in placed:
                self.status[i] = "deferred"
        row = {"round": k, "kind": "hard", "batch": names(batch), "anchors_A": names(A), "anchors_B": names(B),
               "align_A": None if alA is None else {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in alA.items() if kk in ("ok", "resid", "used", "dropped")},
               "align_B": None if alB is None else {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in alB.items() if kk in ("ok", "resid", "used", "dropped")},
               "images": per, "registered": names(placed), "held": names(held + reverted), **self.model_stats()}
        self.say(f"hard {k}: +{len(placed)} of {len(batch)} (held {len(held) + len(reverted)}) -> reg {self.rec.num_reg_images()}/{len(self.names)}; "
                 + "; ".join(f"{Path(n).stem}:{v.get('why')}" for n, v in per.items()))
        return row

    def rerefine_held(self) -> dict[str, Any]:
        """Hard-placed images (held ones keep no observations, so bundle adjustment never touched them; the others may
        have been pulled by lookalike correspondences): re-refine their poses in the final model on the correspondences
        that agree with them, coarse to fine (48, 24, 12 px)."""
        out = {}
        for i in sorted(self.held | self.hard_placed):
            if i not in self.reg():
                continue
            R, t = self.pose_of(i)
            r = self.refine_pose_2d3d(i, R, t)
            if r.get("inliers", 0) >= 15:
                cam_f = None
                self.set_pose(i, np.c_[r["R"], r["t"]], cam_f)
            out[self.names[i]] = {kk: v for kk, v in r.items() if kk not in ("R", "t")}
        if out:
            self.attach_mapper()
        return out

    def audit(self) -> dict[str, Any] | None:
        """Render-and-compare every camera of the model; drop the ones whose photo disagrees with what the model shows."""
        reg = sorted(self.reg())
        if len(reg) > self.gc.audit_max_images or len(reg) < 4:
            return None
        nb_of = {}
        wants = []
        for i in reg:
            R, t = self.pose_of(i)
            nb_of[i] = self.pose_neighbours(R, t, 4, {i})
            wants.append((i, R, t))
        self.ensure_depth(wants, "audit")
        rows = {}
        for i in reg:
            rows[self.names[i]] = self.rac.judge(self.names[i], None, [self.names[j] for j in nb_of[i]])
        counts = {}
        for r in rows.values():
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        bad = [i for i in reg if rows[self.names[i]]["verdict"] == "inconsistent"] if self.gc.audit_drop else []
        for i in bad:
            self.deregister(i)
            self.status[i] = "dropped"
        if bad:
            self.global_refine(final=True)
        row = {"round": self.rounds, "kind": "audit", "verdicts": counts, "dropped": [self.names[i] for i in bad],
               "flagged": [self.names[i] for i in reg if rows[self.names[i]]["verdict"] == "inconsistent"],
               "images": {n: {kk: r.get(kk) for kk in ("score", "margin", "verdict", "overlap")} for n, r in rows.items()}, **self.model_stats()}
        self.say(f"audit: {counts}; dropped {row['dropped']}")
        return row

    def strong_pair_rows(self, among: set[int] | None = None) -> list[tuple[int, int, int, float]]:
        """(i, j, inliers, consistency) for strong pairs between registered images (optionally within ``among``)."""
        reg = self.reg() if among is None else (self.reg() & among)
        rows = []
        for (i, j), m in self.pairs.items():
            if i in reg and j in reg and len(m) >= self.gc.strong_pair:
                rows.append((i, j, len(m), self.pair_consistency(i, j)))
        return rows

    def consistent_components(self, rows, nodes: set[int]) -> list[list[int]]:
        """Connected components of the graph of strong pairs that agree with the poses (consistency >= 0.5)."""
        adj: dict[int, set[int]] = {n: set() for n in nodes}
        for i, j, n, f in rows:
            if f >= 0.5:
                adj[i].add(j)
                adj[j].add(i)
        seen, comps = set(), []
        for s0 in nodes:
            if s0 in seen:
                continue
            comp, todo = [], [s0]
            seen.add(s0)
            while todo:
                u = todo.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        todo.append(v)
            comps.append(comp)
        return sorted(comps, key=len, reverse=True)

    def signed_split(self, rows, nodes: list[int]) -> tuple[list[int], list[int], dict[str, float]] | None:
        """Two-way partition of the registered images that keeps strong consistent pairs together and separates
        strong inconsistent pairs (signed spectral clustering). Returns (majority, minority, cut statistics), or None
        when the split does not explain the inconsistency."""
        if len(nodes) < 6 or len(nodes) > 600:
            return None
        idx = {n: a for a, n in enumerate(nodes)}
        A = np.zeros((len(nodes), len(nodes)))
        for i, j, n, f in rows:
            if i not in idx or j not in idx:
                continue
            w = n if f >= 0.5 else (-n if f < 0.2 else 0.0)
            A[idx[i], idx[j]] += w
            A[idx[j], idx[i]] += w
        L = np.diag(np.abs(A).sum(1)) - A
        vals, vecs = np.linalg.eigh(L)
        v = vecs[:, 0]
        side = v >= np.median(v) if np.all(v >= 0) or np.all(v <= 0) else v >= 0
        a = [nodes[x] for x in range(len(nodes)) if side[x]]
        b = [nodes[x] for x in range(len(nodes)) if not side[x]]
        if len(a) < 2 or len(b) < 2:
            return None
        sa = set(a)
        neg_cut = neg_all = pos_cut = pos_all = 0.0
        for i, j, n, f in rows:
            if i not in idx or j not in idx:
                continue
            across = (i in sa) != (j in sa)
            if f < 0.2:
                neg_all += n
                neg_cut += n if across else 0
            elif f >= 0.5:
                pos_all += n
                pos_cut += n if across else 0
        stats = {"neg_cut_frac": round(neg_cut / max(neg_all, 1e-9), 3), "pos_cut_frac": round(pos_cut / max(pos_all, 1e-9), 3)}
        if stats["neg_cut_frac"] < 0.6 or stats["pos_cut_frac"] > 0.4:
            return None
        maj, mino = (a, b) if len(a) >= len(b) else (b, a)
        return maj, mino, stats

    def repair_fold(self, k: int) -> list[dict[str, Any]]:
        """A fold (two self-consistent blocks whose mutual strong pairs disagree with the poses, as on repeated or
        mirrored structure): deregister the smaller block and place its images again as hard images, with anchors
        from the larger block in view."""
        rows = self.strong_pair_rows()
        if not rows:
            return []
        bad = sum(1 for r in rows if r[3] < 0.2)
        frac = bad / len(rows)
        if frac <= self.gc.fold_frac:
            return []
        comps = self.consistent_components(rows, self.reg())
        suspects = [i for c in comps[1:] for i in c]
        stats = {}
        if not suspects or len(comps[0]) < 3:
            sp = self.signed_split(rows, sorted(self.reg()))
            if sp:
                suspects, stats = sp[1], sp[2]
        row = {"round": k, "kind": "repair", "strong_pairs": len(rows), "inconsistent_frac": round(frac, 3), "components": [len(c) for c in comps[:6]],
               "signed_split": stats, "suspects": [self.names[i] for i in suspects]}
        if not self.gc.repair_act:
            row["action"] = "detected only (repair_act is off)"
            self.say(f"repair: inconsistent strong pairs {frac:.3f}, components {row['components']}, split {stats}: detected, not acted on")
            return [row]
        if not suspects or len(suspects) > 0.5 * len(self.reg()):
            row["action"] = "none: no minority block to re-place"
            self.say(f"repair: inconsistent strong pairs {frac:.3f}, components {row['components']}, no action")
            return [row]
        for i in suspects:
            self.deregister(i)
            self.status[i] = "deferred"
            self.hard_tried[i] = 0
        self.global_refine()
        row["action"] = f"re-placing {len(suspects)} images of the minority block(s)"
        self.say(f"repair: inconsistent strong pairs {frac:.3f}, components {row['components']}: re-placing {len(suspects)} images")
        out = [row]
        cands = list(suspects)
        kk = k
        while cands:
            kk += 1
            r = self.place_hard(kk, candidates=cands[: self.gc.hard_batch])
            if r is None:
                break
            out.append(r)
            cands = cands[self.gc.hard_batch:]
        return out

    # -- the loop ----------------------------------------------------------------------------
    def build(self) -> dict[str, Any]:
        """Seed, grow, place hard images, final refinement. Returns the seed record."""
        seed_info = self.seed()
        self.log.append({"round": 0, "kind": "seed", **seed_info})
        k, stall = 0, 0
        while k < self.gc.max_rounds:
            k += 1
            row = self.grow_round(k)
            if row is None:
                row = self.place_hard(k) if self.gc.hard else None
                if row is None:
                    break
            self.log.append(row)
            stall = stall + 1 if not row.get("registered") else 0
            if stall >= 6:
                break
        self.rounds = k
        for r in self.repair_fold(k):
            self.log.append(r)
        self.global_refine(final=True)
        if self.held or self.hard_placed:
            self.log.append({"round": k, "kind": "held", "images": self.rerefine_held()})
        if self.gc.audit:
            row = self.audit()
            if row:
                self.log.append(row)
        return seed_info

    def components(self, nodes: list[int]) -> list[list[int]]:
        nodes_set, seen, comps = set(nodes), set(), []
        for s in nodes:
            if s in seen:
                continue
            comp, todo = [], [s]
            seen.add(s)
            while todo:
                u = todo.pop()
                comp.append(u)
                for v in self.W.get(u, {}):
                    if v in nodes_set and v not in seen:
                        seen.add(v)
                        todo.append(v)
            comps.append(comp)
        return sorted(comps, key=len, reverse=True)

    def sample_names(self, image_ids, k: int = 6) -> list[str]:
        ids = sorted(image_ids)
        if len(ids) <= k:
            return [self.names[i] for i in ids]
        step = len(ids) / k
        return [self.names[ids[int(j * step)]] for j in range(k)]

    def absorb(self, sub: "Solver", k0: int) -> tuple[int, list[dict[str, Any]]]:
        """Place another model's registered images into this one as hard images (feedforward with this model's anchors)."""
        cands = [self.ids[n] for n in (sub.names[i] for i in sub.reg()) if n in self.ids]
        rows, placed = [], 0
        k = k0
        while cands:
            k += 1
            row = self.place_hard(k, candidates=cands[: self.gc.hard_batch])
            if row is None:
                break
            rows.append(row)
            placed += len(row.get("registered", []))
            cands = cands[self.gc.hard_batch:]
        return placed, rows

    def run(self, out_dir: Path) -> dict[str, Any]:
        t0 = time.time()
        seed_info = self.build()
        models_out = [out_dir]
        extra: list[dict[str, Any]] = []
        if self.gc.multi_model:
            remaining = [i for i in self.names if i not in self.reg()]
            for comp in self.components(remaining):
                if len(comp) < self.gc.min_component or len(models_out) >= self.gc.max_models:
                    break
                tag = f"model{len(models_out)}"
                sub_cfg = replace(self.gc, multi_model=False, seed_size=min(self.gc.seed_size, len(comp)))
                sub = Solver(self.ws, self.cfg, self.db_id, sub_cfg, self.workdir / tag, self.gpu, self.verbose,
                             image_subset=[self.names[i] for i in comp], advisor=self.advisor, tag=tag)
                try:
                    sub.build()
                except RuntimeError as e:
                    self.log.append({"round": self.rounds, "kind": "component", "component": [self.names[i] for i in comp], "error": str(e)[:300]})
                    continue
                finally:
                    if self.advisor is not None:
                        self.advisor.log = self.log
                self.log.extend({**r, "model": tag} for r in sub.log)
                if sub.rec.num_reg_images() < 3:
                    continue
                verdict = "unsure"
                if self.advisor is not None and self.advisor.enabled:
                    verdict = self.advisor.same_scene(self.sample_names(self.reg()), self.sample_names(sub.reg()), {"model": tag, "images": sub.rec.num_reg_images()})
                merged, nplaced = False, 0
                if verdict != "different":
                    nplaced, rows = self.absorb(sub, self.rounds)
                    self.log.extend(rows)
                    merged = nplaced >= 0.5 * sub.rec.num_reg_images()
                rec_ = {"round": self.rounds, "kind": "component", "model": tag, "component": len(comp), "registered": sub.rec.num_reg_images(),
                        "same_scene": verdict, "absorbed": nplaced, "merged": merged}
                self.log.append(rec_)
                self.say(f"component {tag}: {sub.rec.num_reg_images()} of {len(comp)} images; same scene: {verdict}; absorbed {nplaced}; {'merged' if merged else 'kept as a separate model'}")
                if not merged:
                    d = out_dir.parent / str(len(models_out))
                    d.mkdir(parents=True, exist_ok=True)
                    sub.rec.write(str(d))
                    models_out.append(d)
                    extra.append({"model": tag, **sub.model_stats(), "component": len(comp), "same_scene": verdict})
                else:
                    self.global_refine(final=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.rec.write(str(out_dir))
        reg = self.reg()
        for i in self.names:
            if i in reg:
                self.status[i] = "placed"
            elif self.status[i] == "placed":
                self.status[i] = "filtered"
        unplaced = [self.names[i] for i in self.names if i not in reg]
        hard_rows = [r for r in self.log if r.get("kind") == "hard" and "model" not in r]
        summary = {"feedforward": self.gc.feedforward, "num_images": len(self.names), **self.model_stats(),
                   "seed": {"centre": seed_info["attempts"][seed_info["chosen"]]["centre"], "attempts": len(seed_info["attempts"]),
                            "kept": len(seed_info["attempts"][seed_info["chosen"]]["kept"]), "size": self.gc.seed_size},
                   "rounds": self.rounds, "hard_rounds": len(hard_rows), "hard_placed": sum(len(r.get("registered", [])) for r in hard_rows),
                   "audit": next(({"verdicts": r["verdicts"], "dropped": r["dropped"]} for r in self.log if r.get("kind") == "audit" and "model" not in r), None),
                   "held": len(self.held), "rejected_total": sum(len(r.get("rejected", {})) for r in self.log if r.get("kind") == "pnp"),
                   "pnp_failed_total": sum(len(r.get("pnp_failed", [])) for r in self.log if r.get("kind") == "pnp"),
                   "extra_models": extra, "num_models": len(models_out),
                   "unplaced": unplaced[:30] + ([f"... {len(unplaced)} in all"] if len(unplaced) > 30 else []),
                   "status_counts": {s: sum(1 for v in self.status.values() if v == s) for s in sorted(set(self.status.values()))},
                   "feedforward_calls": self.ff_calls, "feedforward_seconds": round(self.ff_seconds, 1),
                   "vlm_calls": self.advisor.calls if self.advisor else 0, "vlm_seconds": round(self.advisor.seconds, 1) if self.advisor else 0.0,
                   "wall": round(time.time() - t0, 1),
                   "profile": {k: round(v, 1) for k, v in sorted(self.timer.t.items(), key=lambda kv: -kv[1])},
                   "config": asdict(self.gc)}
        (out_dir.parent / "grow_log.json").write_text(json.dumps({"summary": summary, "log": self.log}, indent=1, default=str))
        return summary
