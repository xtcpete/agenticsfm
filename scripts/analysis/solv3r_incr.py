"""Offline Solv3R incremental modes (solv3r env), on the inputs of solv3r_prep.py. Per scene, all GT images:
  joint      one joint solve, 32 calls (reference)
  anyorder+  8 spread views, random groups of 8 join (8 calls per stage), then 16 more joint calls (polish)
  query      a map of a random 70% of the images solved jointly (32 calls) and frozen; each other image is solved on
             its own against the map's key/value cache (16 calls, step_cached): Solv3R's relocalization mode, the
             way a candidate would be placed against the verified model without changing it
  query_nb   the same, but the cache holds only the reference view and 16 map views nearest to the query in GT
             (stand-in for its match-graph neighbours; GT used only to pick them in this offline test)
Output: c2w (ray head) of every image in the map frame.

usage: python scripts/analysis/solv3r_incr.py --prep <dir> --out <dir> [--scenes ...]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from solv3r import Solv3R, cameras_from_rays
from solv3r.model import Context, autocast
from solv3r_run import load


def rays_c2w(model, z, ctx):
    pred = model.decode(z, ctx)
    c2w, _ = cameras_from_rays(pred["rays"], pred["depth_conf"])
    return c2w.double().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = Solv3R.from_pretrained()
    for f in sorted(Path(a.prep).glob("*.json")):
        sc = json.load(open(f))
        if (a.scenes and sc["scene"] not in a.scenes) or (out / f.name).exists():
            continue
        d = Path(sc["images_dir"]); gt = sc["gt"]
        names = sorted(gt)
        n = len(names)
        imgs, size, orig = load([d / x for x in names])
        res = {"scene": sc["scene"], "names": names, "timing": {}}
        rng = np.random.default_rng(0)
        # any-order + polish
        t0 = time.time()
        first = sorted(set(np.linspace(0, n - 1, min(8, n)).round().astype(int).tolist()))
        rest = rng.permutation([i for i in range(n) if i not in first]).tolist()
        stages = [first] + [rest[s:s + 8] for s in range(0, len(rest), 8)]
        order = []
        for k, idx in enumerate(stages):
            if k == 0:
                ctx = model.encode(imgs[idx]); z = model.start(ctx)
            else:
                z, ctx = model.add_views(z, ctx, imgs[idx])
            order += idx
            z = model.solve(ctx, 8, z)
        c = rays_c2w(model, z, ctx)
        z = model.solve(ctx, 16, z)
        c2 = rays_c2w(model, z, ctx)
        inv = np.argsort(order)
        res["anyorder"] = c[inv].tolist(); res["anyorder_polish"] = c2[inv].tolist()
        res["timing"]["anyorder"] = time.time() - t0
        del z, ctx
        # query mode
        t0 = time.time()
        perm = rng.permutation(n)
        nm = max(3, int(round(0.7 * n)))
        mi = sorted(perm[:nm].tolist()); qi = sorted(perm[nm:].tolist())
        if 0 not in mi:                                  # keep image 0 as the map's reference (any view would do)
            mi = [0] + mi[:-1]; qi = sorted(set(range(n)) - set(mi))
        ctx_m = model.encode(imgs[mi]); z_m = model.solve(ctx_m, 32)
        c2w = np.zeros((n, 4, 4)); c2w[mi] = rays_c2w(model, z_m, ctx_m)
        c2w_nb = c2w.copy()
        with autocast():
            full = model.cache(z_m, ctx_m)
        C = np.array([-np.asarray(gt[names[i]]["R"]).T @ np.asarray(gt[names[i]]["t"]) for i in range(n)])
        for q in qi:
            ctx_q = model.encode(imgs[[q]], reference_first=False)
            for key, cache in (("all", full), ("nb", None)):
                if cache is None:
                    dist = np.linalg.norm(C[mi] - C[q], axis=1)
                    keep = sorted(set([0] + np.argsort(dist)[:16].tolist()))
                    sub = ctx_m.subset(keep)
                    with autocast():
                        cache = model.cache(z_m[keep], sub)
                zq = model.start(ctx_q)
                with autocast():
                    for _ in range(16):
                        zq = model.step_cached(zq, ctx_q, cache)
                cq = rays_c2w(model, zq, ctx_q)[0]
                (c2w if key == "all" else c2w_nb)[q] = cq
        res["query"] = {"map": [names[i] for i in mi], "c2w": c2w.tolist(), "c2w_nb": c2w_nb.tolist()}
        res["timing"]["query"] = time.time() - t0
        del z_m, ctx_m, full
        torch.cuda.empty_cache()
        json.dump(res, open(out / f.name, "w"))
        print(sc["scene"], {k: round(v, 1) for k, v in res["timing"].items()}, flush=True)


if __name__ == "__main__":
    main()
