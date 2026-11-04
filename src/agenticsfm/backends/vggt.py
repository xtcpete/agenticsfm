"""Backend (agentic-dg env): VGGT camera poses for small groups of images.

For each group, one forward pass of VGGT-1B on the images padded to a square (518 px): the
extrinsics (cam_from_world, 3x4, in the frame of the group's first camera, arbitrary scale) and
intrinsics in model pixels. Independent of feature matches, so it is evidence of a different kind
than the two-view geometries.

Spec: {images_dir, groups: [[name, ...], ...], size, result_path}
Result: {groups: [{names, extrinsics: [[3x4], ...]}], cost}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from agenticsfm.backends import read_spec, write_result


def pad_to_square(image, size: int, patch: int = 14):
    from PIL import Image

    w, h = image.size
    s = size / max(w, h)
    nw, nh = max(patch, round(w * s / patch) * patch), max(patch, round(h * s / patch) * patch)
    image = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas.paste(image, (x0, y0))
    return canvas


def main() -> None:
    spec = read_spec()
    repo = spec["repos"]["vggt"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    import torchvision.transforms.functional as TF
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    size = int(spec.get("size", 518))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = VGGT(merging=None)
    path = hf_hub_download("facebook/VGGT_tracker_fixed", "model_tracker_fixed_e20.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model.update_patch_dimensions(size // 14, size // 14)
    model = model.to(device).eval().to(torch.bfloat16)
    load_seconds = time.time() - t0
    images_dir = Path(spec["images_dir"])
    cache: dict[str, torch.Tensor] = {}

    def tensor(name: str) -> torch.Tensor:
        if name not in cache:
            with Image.open(images_dir / name) as im:
                cache[name] = TF.to_tensor(pad_to_square(im.convert("RGB"), size))
        return cache[name]

    out = []
    t1 = time.time()
    for names in spec["groups"]:
        x = torch.stack([tensor(n) for n in names])[None].to(device, torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (size, size))
        out.append({"names": names, "extrinsics": extr[0].float().cpu().numpy().tolist()})
    write_result(spec, {"groups": out, "cost": {"load_seconds": load_seconds, "gpu_seconds": time.time() - t1,
                                                "num_groups": len(out)}})


if __name__ == "__main__":
    main()
