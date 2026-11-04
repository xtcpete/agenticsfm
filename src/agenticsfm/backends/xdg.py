"""Backend (xdg env): XDG's doppelganger probability for a list of image pairs.

Runs XDG's own ``doppelgangers_classifier`` (released checkpoint, its config, batch size and
autocast) on the pairs; it reads only the two images, so the score is independent of the matcher.

Spec: {images_dir, pairs: [[n0, n1], ...], out_dir, num_workers, result_path}
Result: {scores: {"n0/n1": p_same_surface}, cost}
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from agenticsfm.backends import read_spec, write_result


def main() -> None:
    spec = read_spec()
    root = Path(spec["repos"]["xdg"])
    import sys

    sys.path.insert(0, str(root))
    os.chdir(root)  # XDG resolves configs/ and weights/ relative to its root
    import torch
    from remove_doppelgangers import _build_args, doppelgangers_classifier
    from src.utils.checkpoints import resolve_checkpoint
    from src.utils.config import load_model_config

    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair_probability_list.npy").unlink(missing_ok=True)  # XDG skips inference if it exists
    pairs = [tuple(p) for p in spec["pairs"]]
    pair_path = out / "pairs_list.npy"
    np.save(pair_path, np.array([[a, b, 0, 0] for a, b in pairs], dtype=object).reshape(-1, 4))
    config = root / "configs" / "model_configs" / "xdg.yaml"
    cfg = load_model_config(str(config))
    args = _build_args(config=str(config), input_image_path=spec["images_dir"], output_path=str(out),
                       ckpt=resolve_checkpoint(None), num_workers=int(spec.get("num_workers", 16)))
    start = time.time()
    doppelgangers_classifier(args, cfg, str(pair_path))
    torch.cuda.synchronize()
    seconds = time.time() - start
    prob = np.load(out / "pair_probability_list.npy", allow_pickle=True).item()["prob"]
    if len(prob) != len(pairs):
        raise ValueError(f"{len(prob)} probabilities for {len(pairs)} pairs")
    scores = {f"{a}/{b}": float(p) for (a, b), p in zip(pairs, prob[:, 1])}
    write_result(spec, {"scores": scores, "cost": {"gpu_seconds": seconds, "ms_per_pair": 1000 * seconds / max(1, len(pairs))}})


if __name__ == "__main__":
    main()
