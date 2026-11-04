"""Backend (agentic-dg env): feedforward reconstruction of image groups with VGGT-1B or VGGT-Omega.

For each group, one forward pass gives per image: extrinsics (cam_from_world, 3x4, in the frame of
the group's first camera, arbitrary scale), intrinsics in the model's input pixels, the resize /
crop that maps original pixels to model pixels, and a subsample of confident depth points in 3D.
Used by ``verify_poses`` (poses only) and by ``reconstruct(backend="vggt"|"vggt_omega")`` for
local reconstructions that ``merge_models`` can attach to a global model.

Spec: {images_dir, groups: [[name, ...], ...], model: "vggt"|"vggt_omega", size, points_per_image, result_path}
Result: {groups: [{names, extrinsics, intrinsics, transforms: [[sx, sy, ox, oy], ...], points: [[x,y,z], ...],
         point_image: [i, ...]}], cost}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from agenticsfm.backends import read_spec, write_result


def load_vggt(repo: str, size: int, device: str):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    from huggingface_hub import hf_hub_download
    from vggt.models.vggt import VGGT

    model = VGGT(merging=None)
    path = hf_hub_download("facebook/VGGT_tracker_fixed", "model_tracker_fixed_e20.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model.update_patch_dimensions(size // 14, size // 14)
    return model.to(device).eval().to(torch.bfloat16)


def load_omega(repo: str, checkpoint: str, device: str):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().to(device).eval()
    sd = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd, strict=False)
    return model


def vggt_inputs(images_dir: Path, names: list[str], size: int):
    """VGGT-1B: pad to a square of ``size``; transform original -> model pixels is (s, s, ox, oy)."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    tensors, transforms = [], []
    for n in names:
        with Image.open(images_dir / n) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = size / max(w, h)
            nw, nh = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            x0, y0 = (size - nw) // 2, (size - nh) // 2
            canvas.paste(im.resize((nw, nh), Image.BICUBIC), (x0, y0))
            tensors.append(TF.to_tensor(canvas))
            transforms.append([nw / w, nh / h, x0, y0])
    return torch.stack(tensors), transforms


def omega_inputs(images_dir: Path, names: list[str], size: int):
    """VGGT-Omega: its own loader (balanced token budget, aspect ratio clipped to [0.5, 2])."""
    from PIL import Image
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    x = load_and_preprocess_images([str(images_dir / n) for n in names], image_resolution=size)
    transforms = []
    H, W = x.shape[-2:]
    for n in names:
        with Image.open(images_dir / n) as im:
            w, h = im.size
        # the loader centre-crops extreme aspect ratios; WRIVA images are within [0.5, 2], so a pure resize
        transforms.append([W / w, H / h, 0.0, 0.0])
    return x, transforms


def unproject(depth: np.ndarray, conf: np.ndarray, K: np.ndarray, E: np.ndarray, n: int, rng) -> np.ndarray:
    """n confident depth pixels of one image as 3D points in the group frame."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    ok = (conf > np.percentile(conf, 60)) & (depth > 0)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return np.zeros((0, 3))
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    u, v, z = xs.ravel()[idx], ys.ravel()[idx], depth.ravel()[idx]
    cam = np.linalg.inv(K) @ np.stack([u, v, np.ones_like(u)]).astype(np.float64) * z
    R, t = E[:, :3], E[:, 3]
    return (R.T @ (cam - t[:, None])).T


def main() -> None:
    spec = read_spec()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    which = spec.get("model", "vggt")
    size = int(spec.get("size", 518 if which == "vggt" else 512))
    t0 = time.time()
    if which == "vggt":
        model = load_vggt(spec["repos"]["vggt"], size, device)
    elif which == "vggt_omega":
        model = load_omega(spec["repos"]["vggt_omega"], spec["repos"]["vggt_omega_checkpoint"], device)
    else:
        raise ValueError(f"unknown feedforward model {which}")
    load_seconds = time.time() - t0
    images_dir = Path(spec["images_dir"])
    ppi = int(spec.get("points_per_image", 0))
    rng = np.random.default_rng(0)
    out = []
    t1 = time.time()
    for names in spec["groups"]:
        if which == "vggt":
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            x, transforms = vggt_inputs(images_dir, names, size)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x[None].to(device, torch.bfloat16))
            extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (size, size))
            depth = pred["depth"][0, ..., 0].float().cpu().numpy() if "depth" in pred else None
            conf = pred["depth_conf"][0].float().cpu().numpy() if "depth_conf" in pred else None
        else:
            from vggt_omega.utils.pose_enc import encoding_to_camera

            x, transforms = omega_inputs(images_dir, names, size)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x.to(device))
            extr, intr = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
            depth = pred["depth"][0, ..., 0].float().cpu().numpy() if "depth" in pred else None
            conf = pred["depth_conf"][0].float().cpu().numpy() if "depth_conf" in pred else None
        E = extr[0].float().cpu().numpy()
        K = intr[0].float().cpu().numpy()
        g = {"names": names, "extrinsics": E.tolist(), "intrinsics": K.tolist(), "transforms": transforms}
        if ppi and depth is not None:
            pts, owner = [], []
            for i in range(len(names)):
                p = unproject(depth[i], conf[i] if conf is not None else np.ones_like(depth[i]), K[i], E[i], ppi, rng)
                pts.append(p)
                owner += [i] * len(p)
            g["points"] = np.concatenate(pts).round(5).tolist() if pts else []
            g["point_image"] = owner
        out.append(g)
    write_result(spec, {"groups": out, "model": which, "cost": {"load_seconds": load_seconds, "gpu_seconds": time.time() - t1,
                                                                 "num_groups": len(out)}})


if __name__ == "__main__":
    main()
