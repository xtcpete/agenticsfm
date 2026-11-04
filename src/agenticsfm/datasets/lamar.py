"""LaMAR (CAB, HGE, LIN): the iOS phone images of a location's map sessions, thousands per location.

Layout (CAPTURE format): ``<root>/<loc>/sessions/map/{sensors.txt, images.txt, trajectories.txt,
raw_data/<session>/images/<ts>.jpg}``. Each phone frame has its own PINHOLE sensor and a pose in
``trajectories.txt`` (world -> camera, as Kapture). The harness needs a flat image directory, so
``prepare`` links the images into ``<out>/<loc>_<subset>/images/<session>__<ts>.jpg`` and writes
the GT as a COLMAP model in ``<out>/<loc>_<subset>/gt/0``. Scenes: ``<loc>_ios`` (all phone
images) and, for smoke tests, ``<loc>_<session>`` (one session). Config::

    datasets:
      lamar:
        root: ~/gluemap/datasets/lamar
        out: outputs/data/lamar
        scenes: {CAB_ios: {location: CAB}, CAB_s1: {location: CAB, sessions: [ios_2021-06-02_14.21.49_000]}}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.datasets import BenchmarkScene, Scene, register


def read_table(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append([x.strip() for x in line.split(",")])
    return rows


def prepare(root: Path, out: Path, location: str, name: str, sessions: list[str] | None, world_to_cam: bool = True) -> Path:
    """Link the images and write the GT model; returns the scene directory. Idempotent."""
    scene_dir = out / name
    images_dir, gt_dir = scene_dir / "images", scene_dir / "gt" / "0"
    if (gt_dir / "images.bin").is_file() and images_dir.is_dir():
        return scene_dir
    import pycolmap

    cap = root / location / "sessions" / "map"
    sensors = {r[0]: r for r in read_table(cap / "sensors.txt")}
    traj = {r[1]: [float(x) for x in r[2:9]] for r in read_table(cap / "trajectories.txt")}
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction()
    next_id = 1
    for ts, sensor_id, rel in read_table(cap / "images.txt"):
        session = sensor_id.split("/")[0]
        if not session.startswith("ios") or (sessions and session not in sessions) or sensor_id not in traj:
            continue
        s = sensors[sensor_id]
        if s[2] != "camera" or s[3] != "PINHOLE":
            continue
        w, h, fx, fy, cx, cy = (float(v) for v in s[4:10])
        fname = f"{session}__{Path(rel).name}"
        link = images_dir / fname
        if not link.exists():
            os.symlink((cap / "raw_data" / rel).resolve(), link)
        cam = pycolmap.Camera(model="PINHOLE", width=int(w), height=int(h), params=[fx, fy, cx, cy])
        cam.camera_id = next_id
        rec.add_camera_with_trivial_rig(cam) if hasattr(rec, "add_camera_with_trivial_rig") else rec.add_camera(cam)
        qw, qx, qy, qz, tx, ty, tz = traj[sensor_id]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([qx, qy, qz, qw])), np.array([tx, ty, tz]))
        if not world_to_cam:
            pose = pose.inverse()
        im = pycolmap.Image(name=fname, camera_id=cam.camera_id, image_id=next_id)
        if hasattr(rec, "add_image_with_trivial_frame"):
            rec.add_image_with_trivial_frame(im)
            rec.frames[rec.images[next_id].frame_id].rig_from_world = pose
            rec.register_frame(rec.images[next_id].frame_id)
        else:
            im.cam_from_world = pose
            rec.add_image(im)
            rec.register_image(next_id)
        next_id += 1
    rec.write(str(gt_dir))
    return scene_dir


@register("lamar")
def load(cfg: dict[str, Any], split: str) -> list[BenchmarkScene]:
    root = Path(os.path.expanduser(cfg["root"]))
    out = Path(cfg["out"])
    if not out.is_absolute():
        from agenticsfm.config import REPO_ROOT

        out = REPO_ROOT / out
    scenes = []
    for name, spec in cfg.get("scenes", {}).items():
        d = prepare(root, out, spec["location"], name, spec.get("sessions"), spec.get("world_to_cam", True))
        scenes.append(BenchmarkScene(Scene(name, d / "images"), d / "gt" / "0", "lamar", spec.get("split", "test")))
    return scenes
