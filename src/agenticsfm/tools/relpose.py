"""Relative poses of the verified pairs of a database, and rotation cycle consistency.

COLMAP's verification stores inlier matches but no relative pose. Here each verified pair gets
``cam2_from_cam1`` re-estimated from its inliers with the database's camera priors (pycolmap's
two-view estimator: essential matrix, or homography for planar / rotation-only pairs). A wrong
edge (doppelganger) carries a relative rotation that does not close the triangles it forms with
true edges, so the share of consistent triplets through an edge is a structural, model-free signal.
Cached per database in ``scores/relpose_<db_id>.npz``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def rot_angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def relative_poses(db_path: Path, cache: Path | None = None) -> dict:
    """{(name_a, name_b): (R, t, inliers, config)} with R, t = cam_b_from_cam_a for every verified pair."""
    import pycolmap

    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=False)
        return {(str(a), str(b)): (R, t, int(n), int(c)) for a, b, R, t, n, c in
                zip(z["name0"], z["name1"], z["R"], z["t"], z["inliers"], z["config"])}
    opts = pycolmap.TwoViewGeometryOptions()
    opts.compute_relative_pose = True
    opts.ransac.max_error = 4.0
    out = {}
    with pycolmap.Database.open(db_path) as db:
        images = {im.image_id: im for im in db.read_all_images()}
        cams = {c.camera_id: c for c in db.read_all_cameras()}
        kps = {}
        ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            m = np.asarray(g.inlier_matches)
            if len(m) < 8:
                continue
            i1, i2 = divmod(int(pid), 2147483647)
            for i in (i1, i2):
                if i not in kps:
                    kps[i] = np.asarray(db.read_keypoints(i))[:, :2].astype(np.float64)
            c1, c2 = cams[images[i1].camera_id], cams[images[i2].camera_id]
            tv = pycolmap.estimate_two_view_geometry(c1, kps[i1], c2, kps[i2], m.astype(np.uint32), opts)
            if tv is None or tv.cam2_from_cam1 is None or int(tv.config) in (0, 1):  # UNDEFINED, DEGENERATE
                continue
            out[(images[i1].name, images[i2].name)] = (tv.cam2_from_cam1.rotation.matrix(), np.asarray(tv.cam2_from_cam1.translation),
                                                       int(tv.inlier_matches.shape[0]), int(tv.config))
    if cache is not None and out:
        keys = list(out)
        np.savez(cache, name0=np.array([k[0] for k in keys]), name1=np.array([k[1] for k in keys]),
                 R=np.stack([out[k][0] for k in keys]), t=np.stack([out[k][1] for k in keys]),
                 inliers=np.array([out[k][2] for k in keys]), config=np.array([out[k][3] for k in keys]))
    return out


def cycle_consistency(relposes: dict, max_angle: float = 10.0, max_triplets: int = 30) -> dict:
    """Per edge: (fraction of its triplets whose rotations close within max_angle, number of triplets,
    median cycle error). Edges without triplets get (nan, 0, nan)."""
    R: dict[tuple, np.ndarray] = {}
    nb: dict[str, set] = {}
    for (a, b), (Rab, *_) in relposes.items():
        R[(a, b)] = Rab
        R[(b, a)] = Rab.T
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    rng = np.random.default_rng(0)
    out = {}
    for (a, b) in relposes:
        common = sorted(nb[a] & nb[b])
        if len(common) > max_triplets:
            common = list(rng.choice(common, max_triplets, replace=False))
        errs = [rot_angle(R[(c, a)] @ R[(b, c)] @ R[(a, b)]) for c in common]
        if errs:
            e = np.array(errs)
            out[(a, b)] = (float((e < max_angle).mean()), len(e), float(np.median(e)))
        else:
            out[(a, b)] = (float("nan"), 0, float("nan"))
    return out
