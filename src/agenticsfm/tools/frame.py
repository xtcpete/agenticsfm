"""A shared drawing frame per scene, so renders of different models can be compared.

The first examined model of a scene becomes the reference; later models are aligned to it with a
similarity (Umeyama on the centres of common registered images, when at least 3 exist). The frame
is gravity-aligned: "up" is minus the mean camera down-axis (COLMAP's image y axis in world
coordinates), which holds for hand-held, PTZ and most drone footage; the horizontal axes are the
principal directions of the camera centres in the plane perpendicular to up. Axis limits are fixed
per scene from the reference model, so a model that drifts or folds shows as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity s, R, t with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


def cameras_of(rec) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """name -> (centre, viewing direction, down axis) in the model's world frame."""
    out = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        R = im.cam_from_world().rotation.matrix()  # world -> cam
        out[im.name] = (np.asarray(im.projection_center()), R[2], R[1])  # cam z and y axes in world
    return out


class Frame:
    def __init__(self, s: float, R: np.ndarray, t: np.ndarray, axes: np.ndarray, centre: np.ndarray, span: float,
                 reference: str):
        self.s, self.R, self.t, self.axes, self.centre, self.span, self.reference = s, R, t, axes, centre, span, reference

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Points (n, 3) of the model -> frame coordinates (horizontal 1, horizontal 2, up)."""
        w = (self.s * (x @ self.R.T) + self.t)
        return (w - self.centre) @ self.axes.T

    def apply_dir(self, d: np.ndarray) -> np.ndarray:
        return (d @ self.R.T) @ self.axes.T

    @staticmethod
    def build(cams: dict, s=1.0, R=None, t=None, reference="self") -> "Frame":
        R = np.eye(3) if R is None else R
        t = np.zeros(3) if t is None else t
        C = np.array([s * R @ c + t for c, _, _ in cams.values()])
        down = np.array([R @ d for _, _, d in cams.values()]).mean(0)
        up = -down / (np.linalg.norm(down) + 1e-12)
        centre = C.mean(0)
        H = C - centre
        H = H - np.outer(H @ up, up)  # horizontal components
        _, _, vt = np.linalg.svd(H, full_matrices=False) if len(H) >= 2 else (None, None, np.eye(3))
        h1 = vt[0] - (vt[0] @ up) * up
        h1 /= np.linalg.norm(h1) + 1e-12
        h2 = np.cross(up, h1)
        axes = np.stack([h1, h2, up])
        local = (C - centre) @ axes.T
        span = float(np.percentile(np.abs(local), 98) * 1.3 + 1e-6)
        return Frame(s, R, t, axes, centre, span, reference)

    def to_json(self) -> dict:
        return {"s": self.s, "R": self.R.tolist(), "t": self.t.tolist(), "axes": self.axes.tolist(), "centre": self.centre.tolist(),
                "span": self.span, "reference": self.reference}


def scene_frame(store: Path, model_id: str, rec) -> Frame:
    """The frame for ``rec``: aligned to the scene's reference model when they share >= 3 images."""
    ref_path = store / "examine" / "reference.json"
    cams = cameras_of(rec)
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text())
        ref_c = {k: np.array(v) for k, v in ref["centres"].items()}
        common = [n for n in cams if n in ref_c]
        if ref["model"] == model_id or len(common) < 3:
            fr = Frame.build(cams) if len(common) < 3 else None
            if fr is not None:
                return fr
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            src = np.array([cams[n][0] for n in common])
            dst = np.array([ref_c[n] for n in common])
            s, R, t = umeyama(src, dst)
        fr = Frame(s, R, t, np.array(ref["axes"]), np.array(ref["centre"]), ref["span"], ref["model"])
        return fr
    fr = Frame.build(cams, reference=model_id)
    ref_path.parent.mkdir(exist_ok=True)
    ref_path.write_text(json.dumps({"model": model_id, "centres": {n: c.tolist() for n, (c, _, _) in cams.items()},
                                    "axes": fr.axes.tolist(), "centre": fr.centre.tolist(), "span": fr.span}))
    return fr
