"""Reference: COLMAP `optimize` (same grouping, camera model, Cauchy 1 px, 2 rounds with 4 px point filtering) run with
a chosen bundle-adjustment backend on the grown models of rayba_prep.py problems; saves poses in rayba_run.py's output
format (variant name colmap_<backend>_<model>) so rayba_eval.py scores them. Needs a pycolmap with that backend
(pycolmap-cuda12 4.2.1 for CASPAR / Ceres-GPU).

usage: python scripts/analysis/colmap_ba_ref.py --prep <dir> --out <dir> --backend caspar|ceres [--model OPENCV]
"""
import argparse, shutil, tempfile, time
from pathlib import Path
import numpy as np
import pycolmap


def to_params(cam, target):
    name = cam.model.name; p = list(cam.params)
    if name == "SIMPLE_RADIAL":
        f, cx, cy, k = p; full = dict(fx=f, fy=f, cx=cx, cy=cy, k1=k, k2=0.0, p1=0.0, p2=0.0)
    elif name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p; full = dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2)
    else:
        raise ValueError(name)
    if target == "OPENCV":
        return "OPENCV", [full[k] for k in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")]
    return "SIMPLE_RADIAL", [0.5 * (full["fx"] + full["fy"]), full["cx"], full["cy"], full["k1"]]


def group_images(rec, tol=0.08):
    imgs = [im for im in rec.images.values() if im.has_pose]
    by = {}
    for im in imgs:
        c = rec.cameras[im.camera_id]; by.setdefault((c.width, c.height), []).append(im)
    groups = []
    for members in by.values():
        members.sort(key=lambda im: rec.cameras[im.camera_id].mean_focal_length())
        cur = [members[0]]
        for prev, im in zip(members, members[1:]):
            f0, f1 = rec.cameras[prev.camera_id].mean_focal_length(), rec.cameras[im.camera_id].mean_focal_length()
            if (f1 - f0) / max(f0, 1e-9) > tol:
                groups.append([m.image_id for m in cur]); cur = []
            cur.append(im)
        groups.append([m.image_id for m in cur])
    return groups


def reassign(rec, groups, model):
    new_cams, image_cam = {}, {}
    for k, g in enumerate(groups, start=1):
        cams = [rec.cameras[rec.images[i].camera_id] for i in g]
        conv = [to_params(c, model) for c in cams]
        new_cams[k] = (conv[0][0], cams[0].width, cams[0].height, np.median(np.array([p for _, p in conv]), 0).tolist())
        for i in g:
            image_cam[i] = k
    d = Path(tempfile.mkdtemp()); rec.write_text(str(d))
    for f in ("rigs.txt", "frames.txt"):
        (d / f).unlink(missing_ok=True)
    with open(d / "cameras.txt", "w") as f:
        for cid, (m, w, h, params) in new_cams.items():
            f.write(f"{cid} {m} {w} {h} " + " ".join(f"{v:.10g}" for v in params) + "\n")
    lines = (d / "images.txt").read_text().splitlines(); out = []; i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            out.append(line); i += 1; continue
        parts = line.split(); iid = int(parts[0])
        if iid in image_cam:
            parts[8] = str(image_cam[iid]); out.append(" ".join(parts)); out.append(lines[i + 1] if i + 1 < len(lines) else "")
        i += 2
    (d / "images.txt").write_text("\n".join(out) + "\n")
    new = pycolmap.Reconstruction(str(d)); shutil.rmtree(d)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="caspar"); ap.add_argument("--model", default="OPENCV")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--pp", action="store_true", help="also refine the principal point (off in optimize)")
    ap.add_argument("--tol", type=float, default=0.08, help="focal clustering tolerance of the intrinsics groups (0: per image)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = f"colmap_{a.backend}_{a.model.lower()}" + ("_pp" if a.pp else "") + (f"_tol{a.tol:g}" if a.tol != 0.08 else "")
    for f in sorted(Path(a.prep).glob("*.npz")):
        sc = f.stem
        if (a.scenes and sc not in a.scenes) or (out / f"{sc}.{name}.npz").exists():
            continue
        z = np.load(f)
        rec = pycolmap.Reconstruction(str(z["model_dir"]))
        t0 = time.time()
        rec = reassign(rec, group_images(rec, a.tol), a.model)
        o = pycolmap.BundleAdjustmentOptions()
        o.refine_focal_length = True; o.refine_extra_params = True; o.refine_principal_point = a.pp; o.print_summary = False
        o.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY; o.ceres.loss_function_scale = 1.0
        o.ceres.solver_options.max_num_iterations = 100
        if a.backend == "caspar":
            o.backend = pycolmap.BundleAdjustmentBackend.CASPAR
        elif a.backend == "ceresgpu":
            o.ceres.use_gpu = True
        for r in range(2):
            pycolmap.bundle_adjustment(rec, o)
            if r == 0:
                for pid in [pid for pid, p in rec.points3D.items() if p.error > 4.0]:
                    rec.delete_point3D(pid)
        wall = time.time() - t0
        by = {im.name: im for im in rec.images.values()}
        R = np.stack([by[n].cam_from_world().rotation.matrix() for n in map(str, z["names"])])
        t = np.stack([by[n].cam_from_world().translation for n in map(str, z["names"])])
        np.savez(out / f"{sc}.{name}.npz", R=R, t=t, wall=wall, model=a.model, log="[]")
        print(f"{sc} {name}: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
