"""Scenes, and the benchmark adapters that pair them with ground truth.

The harness sees a ``Scene``: a directory of images plus optional hints. It has no ground-truth
field, so nothing the agent or a tool can reach contains GT. Benchmarks wrap a scene in a
``BenchmarkScene`` that adds the GT model; only ``scripts/evaluate.py`` reads it.

Adapters list the scenes of a dataset. ``folder`` makes a scene from any image directory, with no
GT, which is how the harness runs on new data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Scene:
    name: str
    images_dir: Path
    # Hints the harness may use, never GT: e.g. {"sequential": True}, {"intrinsics": "<path>"}.
    hints: dict[str, Any] = field(default_factory=dict)

    def image_names(self) -> list[str]:
        return sorted(p.name for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass(frozen=True)
class BenchmarkScene:
    scene: Scene
    gt_dir: Path                 # COLMAP model with GT poses; read by the evaluator only
    dataset: str
    split: str                   # "dev": prompts and policies may be tuned on it; "test": held out

    @property
    def name(self) -> str:
        return self.scene.name


def list_images(path: str | Path) -> list[Path]:
    return sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def folder(path: str | Path, name: str | None = None, **hints: Any) -> Scene:
    """A scene from any directory of images (no GT)."""
    path = Path(path)
    return Scene(name or path.name, path, dict(hints))


Adapter = Callable[[dict[str, Any], str], list[BenchmarkScene]]
ADAPTERS: dict[str, Adapter] = {}


def register(name: str) -> Callable[[Adapter], Adapter]:
    def deco(fn: Adapter) -> Adapter:
        ADAPTERS[name] = fn
        return fn
    return deco


def load_benchmark(cfg: dict[str, Any], dataset: str, split: str = "all") -> list[BenchmarkScene]:
    """Scenes of a benchmark dataset ("dev", "test" or "all"), from ``cfg["datasets"][dataset]``."""
    if split not in ("dev", "test", "all"):
        raise ValueError(f"unknown split {split}")
    if dataset not in ADAPTERS:
        raise ValueError(f"unknown dataset {dataset}; available: {sorted(ADAPTERS)}")
    scenes = ADAPTERS[dataset](cfg["datasets"][dataset], split)
    return [s for s in scenes if split == "all" or s.split == split]


from agenticsfm.datasets import eth3d, lamar, wriva  # noqa: E402,F401  (registers the adapters)

__all__ = ["ADAPTERS", "BenchmarkScene", "Scene", "folder", "list_images", "load_benchmark", "register"]
