"""Pose evaluation of a predicted reconstruction (one or several sub-models) against WRIVA GT.

Three AUCs of the pairwise relative pose error max(rotation, translation direction), in degrees:

- ``auc@t`` (GLUEMAP's ``benchmark/evaluate.py``, as in GLUEMAP's own WRIVA results): the largest
  sub-model only, over the N x N ordered pairs of its images that are in GT, self-pairs included.
  Images it does not register are ignored, so dropping hard images *raises* this score.
- ``auc_all@t`` (primary): the same errors on the largest sub-model, but over the distinct ordered
  pairs of *all* GT images; a pair with an image outside the largest sub-model counts as 180 deg.
  An agent cannot gain by discarding or splitting off images.
- ``colmap_auc@t``: COLMAP's set-based metric (``compute_grouped_rel_errors``) over all sub-models;
  a pair not in a common sub-model counts as 180 deg, translation ignored for GT baselines below
  ``min_proj_center_dist``.

The GLUEMAP and COLMAP implementations are imported by path (configs/default.yaml), so the
numbers match ~/gluemap and ~/pointglue's WRIVA tables exactly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

MODEL_FILES = ("cameras", "images", "points3D")


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and all(
        any((path / f"{stem}.{ext}").is_file() for ext in ("bin", "txt")) for stem in MODEL_FILES
    )


def find_models(path: str | Path) -> list[Path]:
    """A model directory itself, or the sub-model directories inside it (COLMAP's sparse/<i>)."""
    path = Path(path)
    if is_model_dir(path):
        return [path]
    if not path.is_dir():
        return []
    return [p for p in sorted(path.iterdir()) if is_model_dir(p)]


def _load_module(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Evaluator:
    def __init__(self, gluemap_evaluate: str, colmap_benchmark: str, thresholds: list[float],
                 min_proj_center_dist: float):
        self.gluemap = _load_module("gluemap_evaluate", Path(gluemap_evaluate))
        if colmap_benchmark not in sys.path:
            sys.path.insert(0, colmap_benchmark)
        from evaluation import utils as colmap_utils  # COLMAP's benchmark/reconstruction package

        self.colmap = colmap_utils
        self.thresholds = list(thresholds)
        self.min_proj_center_dist = min_proj_center_dist

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Evaluator":
        return cls(**cfg["evaluation"])

    def evaluate(self, gt_dir: str | Path, model_dirs: list[Path]) -> dict[str, Any]:
        import pycolmap

        gt = pycolmap.Reconstruction(str(gt_dir))
        models = [pycolmap.Reconstruction(str(p)) for p in model_dirs]
        gt_names = {image.name for image in gt.images.values()}
        num_gt = len(gt_names)

        errors = self.colmap.compute_grouped_rel_errors(
            gt, models, {name: 0 for name in gt_names}, self.min_proj_center_dist)
        # min_error as COLMAP's evaluate.py: errors below it count as 0, for every error type.
        aucs = self.colmap.compute_auc(errors, np.array(self.thresholds, dtype=float),
                                       min_error=self.min_proj_center_dist)
        metrics: dict[str, Any] = {f"colmap_auc@{t}": float(a) / 100 for t, a in zip(self.thresholds, aucs)}
        registered = {im.name for m in models for im in m.images.values()} & gt_names
        metrics.update(num_images_gt=num_gt, num_images_reg=len(registered), num_models=len(models))

        if not models:
            return {**metrics, **{f"{k}@{t}": 0.0 for k in ("auc", "auc_all") for t in self.thresholds},
                    "num_images_largest": 0}

        largest = max(models, key=lambda m: m.num_reg_images())
        gluemap = self.gluemap.compare_reconstructions(largest, gt, pose_thresholds=self.thresholds,
                                                       error_metric="pose")
        metrics.update({k: v for k, v in gluemap.items() if k.startswith("auc@")})
        metrics["num_images_largest"] = gluemap["num_images_common"]

        pred_R, pred_C = self.gluemap.extract_poses(largest)
        gt_R, gt_C = self.gluemap.extract_poses(gt)
        _, _, pair_errors = self.gluemap.compute_pairwise_pose_errors(pred_R, pred_C, gt_R, gt_C)
        # GLUEMAP's errors cover the N x N ordered pairs of the common images (self-pairs, error 0,
        # included); auc_all keeps the distinct ordered pairs and adds the missing ones at 180 deg.
        n = int(round(np.sqrt(len(pair_errors))))
        pair_errors = pair_errors.reshape(n, n)[~np.eye(n, dtype=bool)] if n > 1 else np.zeros(0)
        pair_errors = np.concatenate([pair_errors, np.full(num_gt * (num_gt - 1) - len(pair_errors), 180.0)])
        for key, value in self.gluemap.compute_auc_at_thresholds(pair_errors, self.thresholds).items():
            metrics[key.replace("auc@", "auc_all@")] = value
        return metrics
