"""Offline Solv3R vs Pi3X test, step 2 (solv3r env): run Solv3R on the inputs exported by solv3r_prep.py.

Per scene:
  groups   the exact Pi3X groups of the run (same names, same first view), 16 calls
  joint    all GT images in one joint solve (16 and 32 calls)
  anyorder 8 images spread over the scene, then the rest in random groups of 8 joining the solve (add_views), 8 calls
           per stage: the incremental mode, compared with the joint solve
Saves camera-to-world (from the ray head, the benchmark protocol, and from the camera head) and intrinsics in original
pixels. Images are read without EXIF rotation, as COLMAP does, and letterboxed (not stretched) on one canvas per solve.

usage: python scripts/analysis/solv3r_run.py --prep <dir> --out <dir> [--scenes ...]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from solv3r import Solv3R, cameras_from_pose_enc, cameras_from_rays
from solv3r.images import model_size


def canvas_for(paths, long_side=504):
    """One canvas per solve: the median aspect ratio of its images, long side 504, sides multiples of 14 (Solv3R's
    size). Images are letterboxed onto it isotropically, as our Pi3X backend does, so mixed portrait / landscape
    sets are not stretched."""
    a = float(np.median([w / h for w, h in (Image.open(p).size for p in paths)]))
    return model_size(1000, 1000 * a, long_side)


def load(paths, size=None):
    """-> images (S, 3, H, W) letterboxed on the canvas `size` (default: canvas_for(paths)), the canvas, and per image
    the transform original -> model pixels (s, ox, oy)."""
    size = size or canvas_for(paths)
    H, W = size
    out, tr = [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(W / w, H / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        ox, oy = (W - nw) // 2, (H - nh) // 2
        t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
        t = F.interpolate(t[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
        c = torch.ones(1, 3, H, W)                   # white padding, as the Pi3X backend
        c[:, :, oy:oy + nh, ox:ox + nw] = t
        out.append(c)
        tr.append((nw / w, nh / h, ox, oy))
    return torch.cat(out), size, tr


def readout(model, z, ctx, orig):
    pred = model.decode(z, ctx)
    c2w_r, K_r = cameras_from_rays(pred["rays"], pred["depth_conf"])
    c2w_p, K_p = cameras_from_pose_enc(pred["pose_enc"], (ctx.H, ctx.W))
    res = {}
    for key, c2w, K in (("rays", c2w_r, K_r), ("cam", c2w_p, K_p)):
        K = K.double().cpu().numpy().copy()
        for i, (sx, sy, ox, oy) in enumerate(orig):  # model pixels -> original pixels (letterbox transform)
            K[i, 0, 0] /= sx; K[i, 0, 2] = (K[i, 0, 2] - ox) / sx
            K[i, 1, 1] /= sy; K[i, 1, 2] = (K[i, 1, 2] - oy) / sy
        res[key] = {"c2w": c2w.double().cpu().numpy().tolist(), "K": K.tolist()}
    del pred
    return res


def solve(model, paths, calls):
    imgs, size, orig = load(paths)
    ctx = model.encode(imgs)
    z = model.solve(ctx, calls)
    return z, ctx, orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--max-joint", type=int, default=260)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = Solv3R.from_pretrained()
    for f in sorted(Path(a.prep).glob("*.json")):
        sc = json.load(open(f))
        if a.scenes and sc["scene"] not in a.scenes:
            continue
        if (out / f.name).exists():
            continue
        d = Path(sc["images_dir"])
        res = {"scene": sc["scene"], "groups": [], "timing": {}}
        t0 = time.time()
        for g in sc["groups"]:
            z, ctx, orig = solve(model, [d / n for n in g["names"]], 16)
            res["groups"].append({"tag": g["tag"], "names": g["names"], **readout(model, z, ctx, orig)})
        res["timing"]["groups"] = time.time() - t0
        names = sorted(sc["gt"])
        if len(names) <= a.max_joint:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            imgs, size, orig = load([d / n for n in names])
            ctx = model.encode(imgs)
            z = model.start(ctx)
            for calls_done, calls in ((0, 16), (16, 32)):
                z = model.solve(ctx, calls - calls_done, z)
                res[f"joint{calls}"] = {"names": names, **readout(model, z, ctx, orig)}
                res["timing"][f"joint{calls}"] = time.time() - t0
            res["timing"]["joint_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            del z, ctx, imgs
            # any-order incremental: 8 spread views, then random groups of 8, 8 calls per stage
            t0 = time.time()
            n = len(names)
            first = sorted(set(np.linspace(0, n - 1, min(8, n)).round().astype(int).tolist()))
            rest = np.random.default_rng(0).permutation([i for i in range(n) if i not in first]).tolist()
            stages = [first] + [rest[s:s + 8] for s in range(0, len(rest), 8)]
            order = []
            for k, idx in enumerate(stages):
                imgs, _, orig_k = load([d / names[i] for i in idx], size)     # the whole scene's canvas
                if k == 0:
                    ctx = model.encode(imgs)
                    z = model.start(ctx)
                else:
                    z, ctx = model.add_views(z, ctx, imgs)
                order += idx
                z = model.solve(ctx, 8, z)
            orig_o = [orig[i] for i in order]
            res["anyorder"] = {"names": [names[i] for i in order], **readout(model, z, ctx, orig_o)}
            res["timing"]["anyorder"] = time.time() - t0
            del z, ctx
        torch.cuda.empty_cache()
        json.dump(res, open(out / f.name, "w"))
        print(sc["scene"], {k: round(v, 1) for k, v in res["timing"].items()}, flush=True)


if __name__ == "__main__":
    main()
