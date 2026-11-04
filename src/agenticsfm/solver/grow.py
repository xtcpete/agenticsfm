"""Open incremental solver, v0 (2026-11-03).

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
3. ``join``: when PnP finds nothing, the feedforward model solves a group of unplaced images
   together with anchor images of the model; the group is aligned to the model on the anchors,
   registered, triangulated, verified like any other addition.

Nothing here reads ground truth. Every decision is logged per round for the trace / the agent.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.backends import run_backend
from agenticsfm.tools.examine import sampson
from agenticsfm.tools.frame import umeyama
from agenticsfm.workspace import Workspace

FEEDFORWARD_ENV = {"vggt": "vggt", "vggt_omega": "vggt", "pi3x": "pi3x"}
MAX_IMAGE_ID = 2147483647  # COLMAP pair id = image_id1 * MAX_IMAGE_ID + image_id2


@dataclass
class GrowConfig:
    feedforward: str = "pi3x"
    seed_size: int = 12          # images in the feedforward seed group
    seed_attempts: int = 3       # seed centres to try before giving up
    group_size: int = 8          # PnP registrations per round before verification
    min_support: float = 0.5     # inlier-weighted share of consistent matches an image needs (strong pairs only)
    strong_pair: int = 50        # inliers for a pair to count in the support
    epipolar_px: float = 4.0
    max_tries: int = 2           # PnP attempts per image in our loop (COLMAP's own cap is 3)
    global_ba_growth: float = 1.1
    join: bool = True
    join_group: int = 12
    join_anchors: int = 6
    join_tries: int = 1
    max_alignment_frac: float = 0.05   # anchors' alignment residual over the scene span
    max_rounds: int = 2000
    num_threads: int = 8


class Solver:
    def __init__(self, ws: Workspace, cfg: dict[str, Any], db_id: str, gc: GrowConfig, workdir: Path,
                 gpu: int | None = None, verbose: bool = True):
        import pycolmap

        self.ws, self.cfg, self.gc, self.workdir, self.gpu, self.verbose = ws, cfg, gc, workdir, gpu, verbose
        self.db_id = db_id
        self.db_path = ws.abspath(db_id)
        self.database = pycolmap.Database.open(str(self.db_path))
        co = pycolmap.DatabaseCacheOptions()
        co.min_num_matches = 15
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
        self.join_tried = {i: 0 for i in self.names}
        self.status = {i: "unplaced" for i in self.names}
        self.mo = pycolmap.IncrementalMapperOptions()
        self.mo.num_threads = gc.num_threads
        self.tri = pycolmap.IncrementalTriangulatorOptions()
        self.ba = pycolmap.BundleAdjustmentOptions()
        self.ba.print_summary = False
        self.last_global = 0
        self.ff_calls = 0
        self.ff_seconds = 0.0
        self.log: list[dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------------------------
    def say(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.ws.scene.name}] grow: {msg}", flush=True)

    def kp(self, i: int) -> np.ndarray:
        if i not in self.kps:
            self.kps[i] = np.asarray(self.database.read_keypoints(i))[:, :2]
        return self.kps[i]

    def reg(self) -> set[int]:
        return set(self.rec.reg_image_ids())

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
        if tn < 1e-9:
            x0r = np.c_[x0, np.ones(len(x0))] @ R.T
            err = np.linalg.norm(x0r[:, :2] / x0r[:, 2:3] - x1, axis=1)
        else:
            t = t / tn
            E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
            err = sampson(E, x0, x1)
        f = 0.5 * (cam0.mean_focal_length() + cam1.mean_focal_length())
        return float((err * f < self.gc.epipolar_px).mean())

    def support(self, image_ids: list[int]) -> dict[int, float]:
        """Per image: inlier-weighted consistency over its strong pairs with registered images (NaN: no strong pair)."""
        reg = self.reg()
        out: dict[int, float] = {}
        for i in image_ids:
            if i not in reg:
                continue
            num = den = 0.0
            for j, n in self.W.get(i, {}).items():
                if n < self.gc.strong_pair or j not in reg:
                    continue
                frac = self.pair_consistency(i, j)
                num += n * frac
                den += n
            out[i] = num / den if den > 0 else float("nan")
        return out

    def feedforward(self, names: list[str], tag: str) -> dict[str, dict[str, Any]]:
        """Run the feedforward model on one group. Returns per image: cam_from_world (3x4, frame of the group's
        first camera, arbitrary scale) and the focal length / principal point in original pixels."""
        model = self.gc.feedforward
        t0 = time.time()
        res = run_backend(self.cfg, "feedforward", FEEDFORWARD_ENV[model],
                          {"images_dir": str(self.ws.images_dir), "groups": [names], "model": model, "points_per_image": 0},
                          self.workdir / f"ff_{tag}", gpu=self.gpu)
        self.ff_calls += 1
        self.ff_seconds += time.time() - t0
        g = res["groups"][0]
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

    # -- steps -------------------------------------------------------------------------------
    def seed(self) -> dict[str, Any]:
        deg = {i: sum(v.values()) for i, v in self.W.items()}
        order = sorted(deg, key=deg.get, reverse=True)
        attempts = []
        for k, centre in enumerate(order[: self.gc.seed_attempts]):
            nb = sorted(self.W[centre], key=self.W[centre].get, reverse=True)[: self.gc.seed_size - 1]
            group = [centre] + nb
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
            bad = [i for i in group if i not in good]
            att = {"centre": self.names[centre], "group": names, "focal": {self.names[i]: round(ff[self.names[i]]["f"], 1) for i in group},
                   "support": {self.names[i]: (None if sup.get(i) != sup.get(i) else round(sup.get(i, float("nan")), 3)) for i in group},
                   "kept": [self.names[i] for i in good], **self.model_stats()}
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
        self.mapper.iterative_global_refinement(5 if final else 3, 0.0005, self.mo, self.ba, self.tri, True)
        self.mapper.filter_frames(self.mo)
        self.last_global = self.rec.num_reg_images()

    def grow_round(self, k: int) -> dict[str, Any] | None:
        cand = [i for i in self.mapper.find_next_images(self.mo, False) if self.tries[i] < self.gc.max_tries]
        if not cand:
            return None
        group = cand[: self.gc.group_size]
        newly, failed = [], []
        for i in group:
            self.tries[i] += 1
            if not self.mapper.register_next_image(self.mo, i):
                failed.append(i)
                continue
            self.mapper.triangulate_image(self.tri, i)
            self.mapper.iterative_local_refinement(2, 0.001, self.mo, self.ba, self.tri, i)
            newly.append(i)
        sup = self.support(newly)
        kept, rejected = [], []
        for i in newly:
            s = sup.get(i, float("nan"))
            if s == s and s < self.gc.min_support:
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
               "support": {self.names[i]: (None if sup.get(i) != sup.get(i) else round(sup[i], 3)) for i in newly}, **self.model_stats()}
        self.say(f"round {k}: +{len(kept)} (rejected {len(rejected)}, pnp failed {len(failed)}) -> "
                 f"{self.rec.num_reg_images()}/{len(self.names)} registered, {row['num_points']} pts, reproj {row['reproj']}")
        return row

    def join(self, k: int) -> dict[str, Any] | None:
        """Feedforward placement of a group the PnP path cannot reach, aligned to the model on anchor images."""
        reg = self.reg()
        link = {i: sum(n for j, n in self.W.get(i, {}).items() if j in reg) for i in self.names if i not in reg}
        cand = [i for i in sorted(link, key=link.get, reverse=True) if link[i] > 0 and self.join_tried[i] < self.gc.join_tries]
        if not cand:
            return None
        group = cand[: self.gc.join_group]
        anchor_w: dict[int, int] = {}
        for i in group:
            for j, n in self.W.get(i, {}).items():
                if j in reg:
                    anchor_w[j] = anchor_w.get(j, 0) + n
        anchors = sorted(anchor_w, key=anchor_w.get, reverse=True)[: self.gc.join_anchors]
        for i in group:
            self.join_tried[i] += 1
        if len(anchors) < 3:
            return {"round": k, "kind": "join", "error": f"only {len(anchors)} anchors", "group": [self.names[i] for i in group]}
        names = [self.names[i] for i in anchors + group]
        ff = self.feedforward(names, f"join{k}")
        centre = lambda E: -E[:, :3].T @ E[:, 3]  # noqa: E731
        src = np.array([centre(ff[self.names[j]]["E"]) for j in anchors])
        dst = np.array([self.rec.images[j].projection_center() for j in anchors])
        s, R, t = umeyama(src, dst)
        resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        all_c = np.array([self.rec.images[j].projection_center() for j in reg])
        span = float(np.linalg.norm(np.ptp(all_c, axis=0))) if len(all_c) > 1 else 1.0
        frac = float(resid.mean() / max(span, 1e-9))
        row: dict[str, Any] = {"round": k, "kind": "join", "anchors": [self.names[j] for j in anchors], "group": [self.names[i] for i in group],
                               "alignment_residual_frac_of_span": round(frac, 4), "scale": round(float(s), 4)}
        if frac > self.gc.max_alignment_frac:
            row["error"] = "anchors do not align with the model"
            self.say(f"join {k}: alignment residual {frac:.3f} of span, skipped")
            return row
        for i in group:
            E = ff[self.names[i]]["E"]
            Rn = E[:, :3] @ R.T
            tn = s * E[:, 3] - Rn @ t
            self.set_pose(i, np.c_[Rn, tn], ff[self.names[i]]["f"])
            self.rec.register_frame(self.rec.images[i].frame_id)
        self.attach_mapper()
        for i in group:
            self.mapper.triangulate_image(self.tri, i)
        for i in group:
            self.mapper.iterative_local_refinement(2, 0.001, self.mo, self.ba, self.tri, i)
        sup = self.support(group)
        kept, rejected = [], []
        for i in group:
            s_i = sup.get(i, float("nan"))
            if i not in self.reg():
                rejected.append(i)
            elif s_i == s_i and s_i < self.gc.min_support:
                self.deregister(i)
                rejected.append(i)
                self.status[i] = "deferred"
            else:
                kept.append(i)
                self.status[i] = "placed"
        if kept:
            self.global_refine()
        row.update({"registered": [self.names[i] for i in kept], "rejected": {self.names[i]: (None if sup.get(i) != sup.get(i) else round(sup[i], 3)) for i in rejected},
                    **self.model_stats()})
        self.say(f"join {k}: +{len(kept)} of {len(group)} (anchors {len(anchors)}, residual {frac:.3f}) -> reg {self.rec.num_reg_images()}")
        return row

    def run(self, out_dir: Path) -> dict[str, Any]:
        t0 = time.time()
        seed_info = self.seed()
        self.log.append({"round": 0, "kind": "seed", **seed_info})
        k, joins, stall = 0, 0, 0
        while k < self.gc.max_rounds:
            k += 1
            row = self.grow_round(k)
            if row is None:
                row = self.join(k) if self.gc.join else None
                if row is None:
                    break
                joins += 1
            self.log.append(row)
            stall = stall + 1 if not row.get("registered") else 0
            if stall >= 6:
                break
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
        summary = {"feedforward": self.gc.feedforward, "num_images": len(self.names), **self.model_stats(),
                   "seed": {"centre": seed_info["attempts"][seed_info["chosen"]]["centre"], "attempts": len(seed_info["attempts"]),
                            "kept": len(seed_info["attempts"][seed_info["chosen"]]["kept"]), "size": self.gc.seed_size},
                   "rounds": k, "joins": joins, "rejected_total": sum(len(r.get("rejected", {})) for r in self.log),
                   "pnp_failed_total": sum(len(r.get("pnp_failed", [])) for r in self.log),
                   "unplaced": unplaced[:30] + ([f"... {len(unplaced)} in all"] if len(unplaced) > 30 else []),
                   "status_counts": {s: sum(1 for v in self.status.values() if v == s) for s in sorted(set(self.status.values()))},
                   "feedforward_calls": self.ff_calls, "feedforward_seconds": round(self.ff_seconds, 1), "wall": round(time.time() - t0, 1),
                   "config": asdict(self.gc)}
        (out_dir.parent / "grow_log.json").write_text(json.dumps({"summary": summary, "log": self.log}, indent=1, default=str))
        return summary
