"""examine: ground-truth-free diagnostics of a model, with renders.

Per sub-model: registration, points, reprojection error, track length. Across the database's
verified pairs: which pairs' two-view inliers agree with the model's relative pose (epipolar
consistency), which do not (candidate doppelgangers the mapper may have folded), and which pairs
cross sub-models (candidate bridges). Renders: camera centres and points of the largest
sub-models from two principal views.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, cached_result, finish, register
from agenticsfm.tools.frame import cameras_of, scene_frame
from agenticsfm.tools.reconstruct import model_summary
from agenticsfm.workspace import Artifact, Workspace

PARAMS = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "epipolar_px": {"type": "number", "default": 4.0, "description": "pixel threshold of the pair-vs-model consistency test"},
        "render": {"type": "boolean", "default": True},
    },
    "required": ["model_id"],
}


def sampson(E: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Sampson distance of normalized correspondences under essential matrix E (x1^T E x0 = 0)."""
    h0 = np.c_[x0, np.ones(len(x0))]
    h1 = np.c_[x1, np.ones(len(x1))]
    Ex0 = h0 @ E.T
    Etx1 = h1 @ E
    num = np.einsum("ij,ij->i", h1, Ex0)
    den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return np.abs(num) / np.sqrt(np.maximum(den, 1e-12))


def edge_consistency(db_path: Path, recs: list, epipolar_px: float) -> dict[str, Any]:
    import pycolmap

    # image name -> (component index, Image). recs are sorted largest first and an image registered in several
    # sub-models (COLMAP keeps an abandoned initialization next to the main model) belongs to the largest one,
    # so its pairs are judged against the poses that will be submitted.
    where: dict[str, tuple[int, Any]] = {}
    for ci, rec in enumerate(recs):
        for im in rec.images.values():
            if im.has_pose and im.name not in where:
                where[im.name] = (ci, im)
    rows = []
    with pycolmap.Database.open(db_path) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        kps: dict[int, np.ndarray] = {}
        ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(ids, geoms):
            n_in = int(g.inlier_matches.shape[0])
            if n_in == 0:
                continue
            i1, i2 = divmod(int(pid), 2147483647)
            a, b = names[i1], names[i2]
            ra, rb = where.get(a), where.get(b)
            if ra is None or rb is None:
                rows.append((a, b, n_in, "unregistered", np.nan))
                continue
            if ra[0] != rb[0]:
                rows.append((a, b, n_in, "cross", np.nan))
                continue
            rec = recs[ra[0]]
            im0, im1 = ra[1], rb[1]
            for iid in (i1, i2):
                if iid not in kps:
                    kps[iid] = np.asarray(db.read_keypoints(iid))[:, :2]
            m = g.inlier_matches
            p0 = kps[i1][m[:, 0]]
            p1 = kps[i2][m[:, 1]]
            cam0, cam1 = rec.cameras[im0.camera_id], rec.cameras[im1.camera_id]
            x0 = np.asarray(cam0.cam_from_img(p0))
            x1 = np.asarray(cam1.cam_from_img(p1))
            rel = im1.cam_from_world() * im0.cam_from_world().inverse()
            R, t = rel.rotation.matrix(), rel.translation
            tn = np.linalg.norm(t)
            if tn < 1e-9:  # pure rotation: use the rotation only
                x0r = (np.c_[x0, np.ones(len(x0))] @ R.T)
                x0r = x0r[:, :2] / x0r[:, 2:3]
                err = np.linalg.norm(x0r - x1, axis=1)
            else:
                t = t / tn
                E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
                err = sampson(E, x0, x1)
            f = 0.5 * (cam0.mean_focal_length() + cam1.mean_focal_length())
            frac = float((err * f < epipolar_px).mean())
            rows.append((a, b, n_in, "same", frac))
    same = [r for r in rows if r[3] == "same"]
    fracs = np.array([r[4] for r in same]) if same else np.zeros(0)
    inliers = np.array([r[2] for r in same]) if same else np.zeros(0)
    # Strong pairs (>= 50 inliers) whose matches mostly disagree with the poses: the fold signal.
    # Weak pairs fail the test from pose noise alone, so they are reported separately.
    strong = inliers >= 50
    strong_bad = strong & (fracs < 0.2)
    inconsistent = sorted([r for r in same if r[4] < 0.2 and r[2] >= 50], key=lambda r: -r[2])
    per_image: dict[str, int] = {}
    for a, b, *_ in inconsistent:
        per_image[a] = per_image.get(a, 0) + 1
        per_image[b] = per_image.get(b, 0) + 1
    cross = sorted([r for r in rows if r[3] == "cross"], key=lambda r: -r[2])
    main = {im.name for im in recs[0].images.values() if im.has_pose} if recs else set()
    redundant = [ci for ci, rec in enumerate(recs[1:], 1)
                 if all(im.name in main for im in rec.images.values() if im.has_pose)]
    return {
        "num_verified_pairs": len(rows),
        "redundant_submodels": len(redundant),
        "_doc": "redundant_submodels: sub-models whose images are all registered in the largest model too (an "
                "initialization the mapper abandoned before re-registering those images in the main model): not a "
                "fold and not a split; the largest model already contains everything",
        "pairs_in_one_model": len(same),
        "pairs_across_models": len(cross),
        "pairs_with_unregistered_image": sum(r[3] == "unregistered" for r in rows),
        "consistency": {
            "strong_pairs": int(strong.sum()),
            "strong_inconsistent_frac": float(strong_bad.sum() / max(1, strong.sum())),
            "_doc": "strong_inconsistent_frac: share of pairs with >= 50 inliers whose matches mostly (>80%) disagree with "
                    "the model's poses; ~0.01 is normal, >0.05 means wrongly placed images (a fold); the other "
                    "numbers include weak pairs, which also fail from pose noise",
            "median": float(np.median(fracs)) if len(fracs) else 0.0,
            "frac_pairs_below_0.5": float((fracs < 0.5).mean()) if len(fracs) else 0.0,
            "inlier_weighted_inconsistency": float((inliers * (1 - fracs)).sum() / max(1, inliers.sum())) if len(fracs) else 0.0},
        "inconsistent_pairs": [{"pair": [a, b], "inliers": n, "consistent_frac": round(float(fr), 3)}
                               for a, b, n, _, fr in inconsistent[:15]],
        "images_in_most_inconsistent_pairs": sorted(per_image.items(), key=lambda kv: -kv[1])[:10],
        "strongest_cross_model_pairs": [{"pair": [a, b], "inliers": n} for a, b, n, *_ in cross[:10]],
        "_rows": rows,
    }


def per_image_table(recs: list, rows: list) -> list[dict]:
    """Per registered image: match support (from the edge rows) and triangulated points relative to the median."""
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for a, b, n, kind, frac in rows:
        if kind != "same" or n < 50 or frac is None or (isinstance(frac, float) and np.isnan(frac)):
            continue
        for x in (a, b):
            num[x] = num.get(x, 0.0) + n * frac
            den[x] = den.get(x, 0.0) + n
    table = []
    seen: set[str] = set()
    for ci, rec in enumerate(recs):
        counts = {im.name: im.num_points3D for im in rec.images.values() if im.has_pose and im.name not in seen}
        seen |= set(counts)
        med = float(np.median(list(counts.values()))) if counts else 1.0
        for name, c in counts.items():
            support = num[name] / den[name] if name in den else None
            rel = c / max(1.0, med)
            table.append({"image": name, "model": ci, "points3D": int(c), "points3D_rel": round(rel, 2),
                          "match_support": round(support, 3) if support is not None else None,
                          "weakly_constrained": bool((support is not None and support < 0.7) or rel < 0.3)})
    table.sort(key=lambda r: (r["match_support"] if r["match_support"] is not None else 1.0, r["points3D_rel"]))
    return table


def short(name: str) -> str:
    """A short label for an image: its number when the name has one."""
    import re

    m = re.findall(r"(\d+)", name)
    return str(int(m[-1])) if m else name[:6]


def draw_model(ax_top, ax_side, fr, rec, flagged: dict[str, str], title: str, points_color="0.65") -> None:
    cams = cameras_of(rec)
    names = list(cams)
    C = fr.apply(np.array([cams[n][0] for n in names]))
    D = fr.apply_dir(np.array([cams[n][1] for n in names]))
    pts = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3))
    if len(pts) > 50000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)]
    P = fr.apply(pts) if len(pts) else pts
    colors = {"ok": "tab:red", "weak": "tab:orange", "suspect": "tab:purple"}
    span = fr.span
    for ax, (i, j), label in ((ax_top, (0, 1), "top view (looking down; x, y horizontal)"),
                              (ax_side, (0, 2), "side view (up is up)")):
        if len(P):
            ax.scatter(P[:, i], P[:, j], s=0.3, c=points_color, linewidths=0, rasterized=True)
        for n, c, d in zip(names, C, D):
            kind = flagged.get(n, "ok")
            ax.scatter(c[i], c[j], s=18, c=colors[kind], zorder=3, edgecolors="black", linewidths=0.3)
            ax.annotate(short(n), (c[i], c[j]), fontsize=5.5, xytext=(2, 2), textcoords="offset points", color="0.2")
            ax.plot([c[i], c[i] + 0.06 * span * d[i]], [c[j], c[j] + 0.06 * span * d[j]], color=colors[kind], lw=0.8, zorder=3)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_title(f"{title}; {label}", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)


def render_models(recs: list, out_dir: Path, fr, flagged: dict[str, str], max_models: int = 2) -> list[str]:
    """One figure per sub-model: top and side views in the scene frame. Red: camera, orange: weakly
    constrained, purple: suspect (from the caller); labels are image numbers; arrows are viewing directions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for ci, rec in enumerate(recs[:max_models]):
        if rec.num_reg_images() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        title = f"model {ci}: {rec.num_reg_images()} cameras, {rec.num_points3D()} points"
        draw_model(axes[0], axes[1], fr, rec, flagged, title)
        fig.text(0.01, 0.01, f"frame: scene reference {fr.reference}; red = camera, orange = weakly constrained; number = image index; "
                             f"arrow = viewing direction", fontsize=7, color="0.3")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        path = out_dir / f"render_model{ci}.png"
        fig.savefig(path, dpi=115)
        plt.close(fig)
        paths.append(str(path))
    return paths


@register("examine", "Diagnose a model without ground truth: per-model statistics, the agreement of every verified pair "
                     "with the model's poses (inconsistent pairs are candidate doppelgangers), pairs across models "
                     "(candidate bridges), and renders of cameras and points.", PARAMS)
def examine(ws: Workspace, cfg: dict[str, Any], model_id: str, epipolar_px: float = 4.0,
            render: bool = True) -> ToolResult:
    import pycolmap

    started = time.time()
    params = {"epipolar_px": float(epipolar_px), "v": 5}
    id = ws.artifact_id("examine", params, {"model": model_id})
    if (r := cached_result(ws, id)) is not None:
        art = ws.get(id)
        return ToolResult(id, dict(art.summary), cached=True, images=art.summary.get("renders", []))
    if model_id not in ws.registry:
        raise KeyError(f"unknown artifact {model_id!r}: pass the 10-character model id returned by reconstruct "
                       f"(sub-models are examined together, not addressed separately)")
    model_dir = ws.abspath(model_id)
    db = ws.ancestor(model_id, "db")
    subs = sorted((p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    recs = [pycolmap.Reconstruction(str(p)) for p in subs]
    recs.sort(key=lambda r: -r.num_reg_images())
    out = ws.path("examine", id)
    out.mkdir(exist_ok=True)
    stats = model_summary(model_dir)
    num_images = len(ws.image_names())
    registered = {im.name for rec in recs for im in rec.images.values() if im.has_pose}
    edges = edge_consistency(ws.abspath(db.id), recs, epipolar_px) if db else {}
    rows = edges.pop("_rows", [])
    with open(out / "edges.jsonl", "w") as f:
        for a, b, n, kind, frac in rows:
            f.write(json.dumps({"pair": [a, b], "inliers": n, "kind": kind, "consistent_frac": None if np.isnan(frac) else round(float(frac), 4)}) + "\n")
    images_table = per_image_table(recs, rows)
    weak = [r for r in images_table if r["weakly_constrained"]]
    renders = []
    if render and recs:
        fr = scene_frame(ws.store, model_id, recs[0])
        flagged = {r["image"]: "weak" for r in weak}
        renders = render_models(recs, out, fr, flagged)
    summary = {**stats, "num_images_total": num_images, "unregistered_images": sorted(set(ws.image_names()) - registered)[:20],
               "num_unregistered": num_images - len(registered), "edges": edges,
               "images": {"num_weakly_constrained": len(weak),
                          "weakly_constrained": [{k: r[k] for k in ("image", "match_support", "points3D_rel")} for r in weak[:15]],
                          "_doc": "match_support: inlier-weighted share of the image's strong pairs' matches that agree with its pose "
                                  "(< 0.5 with a VGGT disagreement means misplaced; 0.5-0.7 means imprecise); points3D_rel: its "
                                  "triangulated points relative to the model's median (< 0.3 means weakly constrained). Imprecise "
                                  "images need more constraints (more or denser matches, optimize), not removal."},
               "renders": renders}
    (out / "images.json").write_text(json.dumps(images_table, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    art = Artifact(id, "examine", "examine", params, {"model": model_id}, str(out.relative_to(ws.store)), summary)
    return finish(ws, art, started, images=renders)
