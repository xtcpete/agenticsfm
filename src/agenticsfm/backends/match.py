"""Backend (pointglue env): keypoints + verified-free matches for a list of pairs.

Reuses pointglue's adapters (``benchmarks/relpose``: detectors and the LightGlue / FlexGlue / LoMa
matchers), which read keypoints from an in-memory cache and return matched 0.5-based pixel
coordinates. This backend writes hloc's files, which the harness imports into a COLMAP database:

  features_path   per image: keypoints (0-based, hloc), image_size (W, H), scores, descriptors
  matches_path    per pair: matches0 (int16), matching_scores0; attrs num_matches, covisibility

Spec: {images_dir, pairs: [[n0, n1], ...], detector, num_keypoints, resize, matcher, matcher_params,
       features_path (reused if present), matches_path, batch_size, result_path}
Result: {num_pairs, num_matches: {pair: n}, covisibility: {pair: s} (FlexGlue), cost}
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from agenticsfm.backends import add_repo_paths, read_spec, write_result

MAX_INDEX_ERROR = 0.01  # px, matched coordinate -> cached keypoint


def pair_key(n0: str, n1: str) -> str:
    return f"{n0}/{n1}"


class MemoryCache:
    """pointglue's KeypointCache interface (``get(pair, view)``) over an in-memory dict."""

    def __init__(self, entries: dict[str, dict[str, np.ndarray]]):
        self.entries = entries

    def get(self, pair, view: int) -> dict[str, np.ndarray]:
        return self.entries[str(pair.path0 if view == 0 else pair.path1)]


def retry_read(fn, path, attempts: int = 5, wait: float = 3.0):
    """Call fn(path) again after a failed image read: network mounts drop reads now and then, and one
    unreadable image must not end a match over thousands."""
    for k in range(attempts):
        try:
            return fn(path)
        except (FileNotFoundError, OSError) as e:
            if k == attempts - 1:
                raise
            print(f"[match] read of {path} failed ({e.__class__.__name__}), retry {k + 1}/{attempts - 1}", flush=True)
            time.sleep(wait)


def detect(spec: dict, paths: list[Path], device) -> dict[str, dict[str, np.ndarray]]:
    from benchmarks.relpose.keypoints import Detector
    from benchmarks.relpose.methods import read_rgb

    features_path = Path(spec["features_path"])
    entries: dict[str, dict[str, np.ndarray]] = {}
    if features_path.is_file():
        with h5py.File(features_path, "r") as f:
            for p in paths:
                g = f[p.name]
                entries[str(p)] = {k: np.asarray(g[k]) for k in g}
                entries[str(p)]["keypoints"] = entries[str(p)]["keypoints"] + 0.5  # hloc 0-based -> 0.5-based
        return entries
    detector = Detector(spec["detector"], "wriva", device, num_keypoints=int(spec["num_keypoints"]))
    detector.resize = int(spec["resize"]) if spec.get("resize") else None
    start = time.time()
    tmp = features_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        for p in paths:
            image = retry_read(read_rgb, p)
            out = detector(image)
            out["image_size"] = np.asarray(image.shape[1::-1], np.int32)
            entries[str(p)] = out
            g = f.create_group(p.name)
            g.create_dataset("keypoints", data=(out["keypoints"].astype(np.float64) - 0.5).astype(np.float32))
            for k in ("image_size", "scores", "descriptors"):
                if k in out:
                    g.create_dataset(k, data=out[k])
        f.attrs.update({"detector": spec["detector"], "num_keypoints": int(spec["num_keypoints"]),
                        "resize": int(spec["resize"] or -1), "seconds": time.time() - start})
    tmp.replace(features_path)
    return entries


def build_matcher(spec: dict, device):
    from benchmarks.relpose import methods

    params = dict(spec.get("matcher_params") or {})
    name = spec["matcher"]
    if name == "lightglue":
        return methods.LightGlueMethod("wriva", device, features=spec["detector"],
                                       threshold=params.get("threshold", 0.1), mp=params.get("mp", False))
    if name == "flexglue":
        return methods.FlexGlueMethod("wriva", device, spec["detector"], describe_resize=params.get("describe_resize"),
                                      threshold=params.get("threshold"), min_covisibility=0.0)
    if name == "loma":
        return methods.LoMaMethod("wriva", device, variant=params.get("variant", "G"), threshold=params.get("threshold"))
    if name == "superglue":
        return methods.SuperGlueMethod("wriva", device, **params)
    raise ValueError(f"unknown matcher {name}")


class IndexLookup:
    def __init__(self) -> None:
        self.trees: dict = {}

    def __call__(self, key: str, matched: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        if len(matched) == 0:
            return np.zeros(0, np.int64)
        if key not in self.trees:
            self.trees[key] = cKDTree(keypoints.astype(np.float64))
        error, index = self.trees[key].query(matched.astype(np.float64))
        if len(error) and error.max() > MAX_INDEX_ERROR:
            raise ValueError(f"{key}: matched point {error.max():.4f} px from any cached keypoint")
        return index


def main() -> None:
    spec = read_spec()
    add_repo_paths(spec, "pointglue", "hloc")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(spec["images_dir"])
    names = sorted({n for p in spec["pairs"] for n in p})
    paths = [images_dir / n for n in names]
    t0 = time.time()
    entries = detect(spec, paths, device)
    detect_seconds = time.time() - t0

    matcher = build_matcher(spec, device)
    matcher.cache = MemoryCache(entries)
    pairs = [SimpleNamespace(name=pair_key(a, b), path0=images_dir / a, path1=images_dir / b) for a, b in spec["pairs"]]
    if spec["matcher"] == "flexglue":
        for p in paths:
            retry_read(matcher.describe, p)
    if spec["matcher"] == "loma":
        # LoMa's DeDoDe descriptions once per image (pointglue's describe_loma), instead of per pair
        descriptor = matcher.model._descriptor
        memo = {}
        for p in paths:
            f = entries[str(p)]
            width, height = (int(v) for v in f["image_size"])
            kpts = matcher.tensor(2 * f["keypoints"] / np.array([width, height], np.float32) - 1)
            with torch.inference_mode():
                memo[str(p)] = descriptor.describe_keypoints(descriptor.read_image(str(p)), kpts)
        descriptor.describe_keypoints_from_path = lambda path, keypoints, *a, **k: memo[str(path)]
    batch_size = int(spec.get("batch_size", 8))
    groups: dict[tuple, list] = {}
    for pair in pairs:
        groups.setdefault(matcher.batch_key(pair), []).append(pair)
    batches = [g[i:i + batch_size] for g in groups.values() for i in range(0, len(g), batch_size)]

    lookup = IndexLookup()
    num_matches: dict[str, int] = {}
    covisibility: dict[str, float] = {}
    matches_path = Path(spec["matches_path"])
    tmp = matches_path.with_suffix(".h5.tmp")
    t1 = time.time()
    with h5py.File(tmp, "w") as f:
        for batch in batches:
            results = matcher.match_batch(batch)
            covis = list(getattr(matcher, "last_covisibility", [])) if spec["matcher"] == "flexglue" else [None] * len(batch)
            for pair, (k0, k1), c in zip(batch, results, covis):
                e0, e1 = entries[str(pair.path0)], entries[str(pair.path1)]
                idx0 = lookup(str(pair.path0), np.asarray(k0).reshape(-1, 2), e0["keypoints"])
                idx1 = lookup(str(pair.path1), np.asarray(k1).reshape(-1, 2), e1["keypoints"])
                matches0 = np.full(len(e0["keypoints"]), -1, np.int16)
                matches0[idx0] = idx1
                g = f.create_group(pair.name)
                g.create_dataset("matches0", data=matches0, compression="gzip", compression_opts=1)
                g.create_dataset("matching_scores0", data=(matches0 > -1).astype(np.float16),
                                 compression="gzip", compression_opts=1)
                g.attrs["num_matches"] = int(len(idx0))
                num_matches[pair.name] = int(len(idx0))
                if c is not None:
                    g.attrs["covisibility"] = float(c)
                    covisibility[pair.name] = float(c)
    tmp.replace(matches_path)
    match_seconds = time.time() - t1
    result = {"num_pairs": len(pairs), "num_images": len(names), "num_matches": num_matches,
              "cost": {"detect_seconds": detect_seconds, "match_seconds": match_seconds,
                       "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}}
    if covisibility:
        result["covisibility"] = covisibility
    write_result(spec, result)


if __name__ == "__main__":
    main()
