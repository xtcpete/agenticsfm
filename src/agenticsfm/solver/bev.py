"""Bird's-eye view (top-down map) of a model, for the VLM: reconstructed points as grey density, every camera as a
triangle pointing along its viewing direction, optional highlighted and numbered camera sets.

Up: the mean of the cameras' up axes (photos are mostly upright); when that is weak (nadir-heavy captures), the normal
of the camera centres' best-fit plane, oriented to agree with it. The map's horizontal axis is the main direction of
the camera centres, so long sites fill the frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

COLORS = {"red": (220, 40, 40), "blue": (40, 90, 230), "green": (30, 160, 60), "orange": (240, 140, 0),
          "magenta": (200, 40, 200), "cyan": (0, 170, 200), "grey": (120, 120, 120)}


def up_vector(rec) -> np.ndarray:
    Rs = [im.cam_from_world().rotation.matrix() for im in rec.images.values() if im.has_pose]
    ups = np.array([-R[1] for R in Rs])                      # camera y points down in the image: world up = -row 1
    u = ups.mean(0)
    C = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    if np.linalg.norm(u) < 0.5 and len(C) >= 3:              # cameras disagree on up (nadir views): plane of centres
        _, _, Vt = np.linalg.svd(C - C.mean(0))
        n = Vt[2]
        u = n if n @ u >= 0 else -n
    return u / max(np.linalg.norm(u), 1e-12)


def frame(rec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, e1, e2): e1 the main horizontal direction of the camera centres, e2 = up x e1."""
    up = up_vector(rec)
    C = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    P = C - C.mean(0)
    P = P - np.outer(P @ up, up)
    if len(P) >= 2 and np.linalg.norm(P) > 0:
        _, _, Vt = np.linalg.svd(P)
        e1 = Vt[0] - (Vt[0] @ up) * up
    else:
        e1 = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return up, e1, e2


def render(rec, out: str | Path, highlight: dict[str, list[int]] | None = None, labels: dict[int, str] | None = None,
           size: int = 1280, max_points: int = 300_000, cam_px: float | None = None, title: str | None = None,
           dim_others: bool = False, edges: list[tuple[int, int, str, int]] | None = None,
           poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None, point_images: set[int] | None = None) -> dict[str, Any]:
    """Draw the map to `out` (jpg/png). highlight: color name -> image ids drawn bold in that color (later entries on
    top); labels: image id -> text drawn next to its camera; edges: (image id, image id, color name, width) lines between
    cameras, drawn under them; poses: image id -> (R, t) cam_from_world overriding the model's (an alternative placement);
    point_images: draw only points seen by at least two of these images. Returns the projection."""
    from PIL import Image, ImageDraw, ImageFont

    up, e1, e2 = frame(rec)
    ims = {i: im for i, im in rec.images.items() if im.has_pose}
    poses = poses or {}

    def pose(i):
        if i in poses:
            return poses[i]
        T = ims[i].cam_from_world()
        return T.rotation.matrix(), np.asarray(T.translation)

    def centre(i):
        R, t = pose(i)
        return -R.T @ t

    C = np.array([centre(i) for i in ims])
    if point_images is not None:
        X = np.array([p.xyz for p in rec.points3D.values() if sum(e.image_id in point_images for e in p.track.elements) >= 2] or np.zeros((0, 3)))
    else:
        X = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3))
    X = X.reshape(-1, 3)
    if len(X) > max_points:
        X = X[np.random.default_rng(0).choice(len(X), max_points, replace=False)]
    cu, cv = C @ e1, C @ e2
    xu, xv = X @ e1, X @ e2
    # extent: all cameras, and the 1-99 % range of the points (stray points do not shrink the map)
    lo_u = min(cu.min(), np.percentile(xu, 1) if len(xu) else cu.min()); hi_u = max(cu.max(), np.percentile(xu, 99) if len(xu) else cu.max())
    lo_v = min(cv.min(), np.percentile(xv, 1) if len(xv) else cv.min()); hi_v = max(cv.max(), np.percentile(xv, 99) if len(xv) else cv.max())
    span = max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    pad = 0.04 * span
    lo_u, hi_u, lo_v, hi_v = lo_u - pad, hi_u + pad, lo_v - pad, hi_v + pad
    W = size
    H = max(64, int(round(size * (hi_v - lo_v) / (hi_u - lo_u))))
    if H > size:                                             # tall site: fit the long side
        H, W = size, max(64, int(round(size * (hi_u - lo_u) / (hi_v - lo_v))))
    sx = (W - 1) / (hi_u - lo_u)
    to_px = lambda u, v: ((u - lo_u) * sx, (hi_v - v) * sx)  # noqa: E731  (image y grows downwards)
    img = np.full((H, W), 255.0)
    if len(X):
        px, py = to_px(xu, xv)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        hist = np.zeros((H, W))
        np.add.at(hist, (py[ok].astype(int), px[ok].astype(int)), 1.0)
        d = np.log1p(hist)
        if d.max() > 0:
            img = 255.0 - 200.0 * np.clip(d / np.percentile(d[d > 0], 99), 0, 1)
    canvas = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, size // 80))
    except OSError:
        font = ImageFont.load_default()
    r = cam_px or max(5.0, size / 110)
    colour_of = {}
    for name, ids in (highlight or {}).items():
        for i in ids:
            colour_of[i] = name

    def xy(i):
        c = centre(i)
        x, y = to_px(np.array(c @ e1), np.array(c @ e2))
        return float(x), float(y)

    for a_, b_, colname, width in edges or []:
        if a_ in ims and b_ in ims:
            draw.line([xy(a_), xy(b_)], fill=COLORS.get(colname, COLORS["grey"]), width=int(width))

    def tri(i, col, width):
        R = pose(i)[0]
        z = R[2]                                             # viewing direction in world
        du, dv = z @ e1, z @ e2
        n = np.hypot(du, dv)
        x, y = xy(i)
        x, y = float(x), float(y)
        if n < 0.25:                                         # looking (nearly) straight down: a square
            draw.rectangle([x - r * 0.7, y - r * 0.7, x + r * 0.7, y + r * 0.7], outline=col, width=width)
            return x, y
        du, dv = du / n, -dv / n                             # image coordinates (v flipped)
        tip = (x + du * r * 1.3, y + dv * r * 1.3)
        left = (x - du * r * 0.6 - dv * r * 0.75, y - dv * r * 0.6 + du * r * 0.75)
        right = (x - du * r * 0.6 + dv * r * 0.75, y - dv * r * 0.6 - du * r * 0.75)
        draw.polygon([tip, left, right], outline=col, width=width)
        return x, y

    base = COLORS["grey"] if dim_others else COLORS["red"]
    for i in ims:
        if i not in colour_of:
            tri(i, base, 2)
    for name, ids in (highlight or {}).items():
        for i in ids:
            if i in ims:
                tri(i, COLORS.get(name, COLORS["blue"]), 4)
    for i, text in (labels or {}).items():
        if i in ims:
            x, y = xy(i)
            col = COLORS.get(colour_of.get(i, "red"), COLORS["red"])
            draw.text((x + r * 1.2, y - r * 1.6), str(text), fill=col, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    if title:
        draw.text((8, 6), title, fill=(0, 0, 0), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    return {"up": up, "e1": e1, "e2": e2, "bounds": (lo_u, hi_u, lo_v, hi_v), "size": (W, H), "scale": sx}
