"""inspect: look at images. Thumbnail grids of chosen images, or one pair with its verified matches."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, register
from agenticsfm.workspace import Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "images": {"type": "array", "items": {"type": "string"},
                   "description": "image names to show as a labelled thumbnail grid (up to 24); omit for an evenly spaced sample"},
        "pair": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
                 "description": "[name0, name1]: show the two images side by side with their verified matches drawn (needs db_id)"},
        "db_id": {"type": "string", "description": "database whose matches to draw for `pair`"},
    },
}


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    s = max_side / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return im, (1.0 if s >= 1 else s)


def grid(ws: Workspace, names: list[str], out: Path, thumb: int = 320, cols: int = 4) -> None:
    import cv2

    tiles = []
    for n in names:
        im, _ = load_rgb(ws.images_dir / n, thumb)
        canvas = np.full((thumb + 24, thumb, 3), 255, np.uint8)
        y, x = (thumb - im.shape[0]) // 2, (thumb - im.shape[1]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, n[:28], (4, thumb + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def draw_pair(ws: Workspace, db_id: str, a: str, b: str, out: Path, max_side: int = 640) -> dict[str, Any]:
    import cv2
    import pycolmap

    with pycolmap.Database.open(ws.abspath(db_id)) as db:
        ia, ib = db.read_image_with_name(a).image_id, db.read_image_with_name(b).image_id
        ka, kb = np.asarray(db.read_keypoints(ia))[:, :2], np.asarray(db.read_keypoints(ib))[:, :2]
        swap = ia > ib
        g = db.read_two_view_geometry(min(ia, ib), max(ia, ib)) if db.exists_two_view_geometry(min(ia, ib), max(ia, ib)) else None
    m = np.asarray(g.inlier_matches) if g is not None else np.zeros((0, 2), int)
    if swap:
        m = m[:, ::-1]
    im0, s0 = load_rgb(ws.images_dir / a, max_side)
    im1, s1 = load_rgb(ws.images_dir / b, max_side)
    h = max(im0.shape[0], im1.shape[0])
    canvas = np.full((h, im0.shape[1] + im1.shape[1] + 10, 3), 255, np.uint8)
    canvas[:im0.shape[0], :im0.shape[1]] = im0
    off = im0.shape[1] + 10
    canvas[:im1.shape[0], off:off + im1.shape[1]] = im1
    rng = np.random.default_rng(0)
    for i, j in m[rng.permutation(len(m))[:150]]:
        p = (int(ka[i, 0] * s0), int(ka[i, 1] * s0))
        q = (int(kb[j, 0] * s1) + off, int(kb[j, 1] * s1))
        cv2.line(canvas, p, q, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, a, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, b, (off + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"pair": [a, b], "verified_inliers": int(len(m)), "drawn": int(min(len(m), 150))}


@register("inspect", "Look at images: a labelled thumbnail grid of chosen images (or an evenly spaced sample), or one "
                     "pair side by side with its verified matches drawn.", PARAMS)
def inspect(ws: Workspace, cfg: dict[str, Any], images: list[str] | None = None, pair: list[str] | None = None,
            db_id: str | None = None) -> ToolResult:
    started = time.time()
    out_dir = ws.run_dir / "inspect"
    out_dir.mkdir(exist_ok=True)
    k = len(list(out_dir.iterdir()))
    if pair:
        if not db_id:
            raise ValueError("pair needs db_id")
        out = out_dir / f"{k:03d}_pair.jpg"
        info = draw_pair(ws, db_id, pair[0], pair[1], out)
        return ToolResult(None, info, cost={"wall": time.time() - started}, images=[str(out)])
    names = ws.image_names()
    if images:
        missing = [n for n in images if n not in set(names)]
        if missing:
            raise ValueError(f"unknown images: {missing[:5]}")
        chosen = list(images)[:24]
    else:
        idx = np.unique(np.linspace(0, len(names) - 1, min(24, len(names))).round().astype(int))
        chosen = [names[i] for i in idx]
    out = out_dir / f"{k:03d}_grid.jpg"
    grid(ws, chosen, out)
    return ToolResult(None, {"shown": chosen, "num_images_total": len(names)}, cost={"wall": time.time() - started},
                      images=[str(out)])
