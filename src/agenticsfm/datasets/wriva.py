"""WRIVA: 34 scenes, 6,804 images, mixed cameras, with doppelgangers (PTZ and multi-zone sites).

Layout: ``<root>/<scene>/{images, sparse/0}``. Scene names look like ``t09_v04_s00_r10_PTZ_A08``:
task, version, sequence, revision, task name, then a site code (A01, A05, A06, A08, S04, M07,
BLH0001). The site code recurs across tasks (``t04_v05_s03_r02_VaryingAltitudes_A08`` and
``t09_v04_*_PTZ_A08``), so the split holds out whole sites.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agenticsfm.datasets import BenchmarkScene, Scene, register

SITE_PATTERN = re.compile(r"_(A\d\d|S\d\d|M\d\d|BLH\d+)(?:_|$)")


def site(scene_name: str) -> str:
    match = SITE_PATTERN.search(scene_name)
    if match is None:
        raise ValueError(f"no site code in scene name {scene_name}")
    return match.group(1)


@register("wriva")
def load(cfg: dict[str, Any], split: str) -> list[BenchmarkScene]:
    root = Path(cfg["root"])
    dev_sites = set(cfg.get("dev_sites", []))
    scenes = []
    for p in sorted(root.iterdir()):
        if not (p / "images").is_dir():
            continue
        scene = Scene(p.name, p / "images")
        scenes.append(BenchmarkScene(scene, p / "sparse" / "0", "wriva",
                                     "dev" if site(p.name) in dev_sites else "test"))
    return scenes
