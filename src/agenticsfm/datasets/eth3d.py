"""ETH3D high-resolution multi-view (25 DSLR scenes, 898 undistorted images), the SfM benchmark
of the FlexGlue paper and of pointglue's ``experiments/sfm.md``.

Layout: ``<root>/<scene>/{images/dslr_images/*.JPG, dslr_calibration_undistorted/{cameras,images,points3D}.txt}``
(COLMAP's benchmark layout, PINHOLE GT). The harness wants a flat image directory and GT image
names that match it, so ``prepare`` links the images into ``<out>/<scene>/images/`` and writes a
GT model with the ``dslr_images/`` prefix stripped from the names (poses and cameras unchanged;
the GT points are kept for the edge oracle).

Protocol notes for a fair comparison with the paper (``configs/default.yaml`` ``eth3d``): 2048
keypoints per image at long side 1600, exhaustive pairs, one SIMPLE_RADIAL camera per image
estimated by the mapper (no GT priors), COLMAP's reconstruction benchmark metric. The GT
covisibility filter of COLMAP's benchmark is a benchmark-side step that uses GT; the harness
cannot apply it, so the comparison is to runs without it or the filter is applied in the
evaluator only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agenticsfm.datasets import BenchmarkScene, Scene, register


def prepare(root: Path, out: Path, scene: str) -> Path:
    scene_dir = out / scene
    images_dir, gt_dir = scene_dir / "images", scene_dir / "gt" / "0"
    if (gt_dir / "images.bin").is_file() and images_dir.is_dir():
        return scene_dir
    import pycolmap

    src = root / scene
    images_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "images").rglob("*")):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            link = images_dir / p.name
            if not link.exists():
                os.symlink(p.resolve(), link)
    rec = pycolmap.Reconstruction(str(src / "dslr_calibration_undistorted"))
    for im in rec.images.values():
        im.name = Path(im.name).name
    gt_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(gt_dir))
    return scene_dir


@register("eth3d")
def load(cfg: dict[str, Any], split: str) -> list[BenchmarkScene]:
    root = Path(os.path.expanduser(cfg["root"]))
    out = Path(cfg["out"])
    if not out.is_absolute():
        from agenticsfm.config import REPO_ROOT

        out = REPO_ROOT / out
    train = set(cfg.get("train_scenes", []))
    scenes = []
    for p in sorted(root.iterdir()):
        if not (p / "images").is_dir():
            continue
        d = prepare(root, out, p.name)
        scenes.append(BenchmarkScene(Scene(p.name, d / "images"), d / "gt" / "0", "eth3d", "dev" if p.name in train else "test"))
    return scenes
