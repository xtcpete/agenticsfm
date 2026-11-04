"""compare_models: two models of the scene side by side in the same frame, and which images moved.

Model B is aligned to model A with a similarity over their common registered images. The render
overlays A (grey) and B (red) with a line between the two placements of every camera; the summary
lists the images whose viewing direction changed by more than 10 deg or whose centre moved by more
than 5% of the scene span (those are the images the two pipelines disagree on, worth inspecting),
and the images registered in only one of the two.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from agenticsfm.tools import ToolResult, register
from agenticsfm.tools.frame import Frame, cameras_of, scene_frame, umeyama
from agenticsfm.workspace import Workspace

PARAMS = {
    "type": "object",
    "properties": {"model_a": {"type": "string", "description": "reference model"}, "model_b": {"type": "string", "description": "model to compare"}},
    "required": ["model_a", "model_b"],
}


def largest(ws: Workspace, model_id: str):
    import pycolmap

    subs = [pycolmap.Reconstruction(str(p)) for p in sorted(ws.abspath(model_id).iterdir()) if p.is_dir() and p.name.isdigit()]
    return max(subs, key=lambda r: r.num_reg_images())


@register("compare_models", "Overlay two models in one frame (B aligned to A on their common images) and list the images whose "
                            "pose differs between them, plus the images only one of them registers. Use it to see what a "
                            "filter, a remap or glomap changed.", PARAMS)
def compare_models(ws: Workspace, cfg: dict[str, Any], model_a: str, model_b: str) -> ToolResult:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.time()
    ra, rb = largest(ws, model_a), largest(ws, model_b)
    ca, cb = cameras_of(ra), cameras_of(rb)
    common = sorted(set(ca) & set(cb))
    fr = scene_frame(ws.store, model_a, ra)
    if len(common) >= 3:
        s, R, t = umeyama(np.array([cb[n][0] for n in common]), np.array([ca[n][0] for n in common]))
    else:
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    frb = Frame(fr.s * s, fr.R @ R, fr.s * (fr.R @ t) + fr.t, fr.axes, fr.centre, fr.span, fr.reference)
    A = {n: (fr.apply(ca[n][0][None])[0], fr.apply_dir(ca[n][1][None])[0]) for n in ca}
    B = {n: (frb.apply(cb[n][0][None])[0], frb.apply_dir(cb[n][1][None])[0]) for n in cb}
    moved = []
    for n in common:
        dpos = float(np.linalg.norm(A[n][0] - B[n][0]) / fr.span)
        ddir = float(np.degrees(np.arccos(np.clip(A[n][1] @ B[n][1], -1, 1))))
        if ddir > 10 or dpos > 0.05:
            moved.append({"image": n, "direction_change_deg": round(ddir, 1), "centre_move_frac_of_span": round(dpos, 3)})
    moved.sort(key=lambda m: -m["direction_change_deg"])
    out_dir = ws.run_dir / "compare"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{len(list(out_dir.iterdir())):03d}_{model_a}_{model_b}.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, (i, j), label in ((axes[0], (0, 1), "top view"), (axes[1], (0, 2), "side view, up is up")):
        for n, (c, d) in A.items():
            ax.scatter(c[i], c[j], s=16, c="0.5", zorder=2)
        for n, (c, d) in B.items():
            ax.scatter(c[i], c[j], s=14, c="tab:red", zorder=3, edgecolors="black", linewidths=0.3)
            if n in A:
                a = A[n][0]
                if np.linalg.norm(a - c) > 0.02 * fr.span:
                    ax.plot([a[i], c[i]], [a[j], c[j]], color="tab:red", lw=0.7, alpha=0.8)
        for n in moved[:30]:
            c = B[n["image"]][0]
            ax.annotate(n["image"].split(".")[0][-3:], (c[i], c[j]), fontsize=6, xytext=(2, 2), textcoords="offset points", color="tab:red")
        ax.set_xlim(-fr.span, fr.span)
        ax.set_ylim(-fr.span, fr.span)
        ax.set_aspect("equal")
        ax.set_title(f"grey: {model_a} ({ra.num_reg_images()} cams)  red: {model_b} ({rb.num_reg_images()} cams); {label}", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)
    fig.text(0.01, 0.01, "lines join the two placements of a camera; labelled cameras moved (> 10 deg or > 5% of the span)", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=115)
    plt.close(fig)
    summary = {"common_images": len(common), "only_in_a": sorted(set(ca) - set(cb))[:20], "only_in_b": sorted(set(cb) - set(ca))[:20],
               "num_moved": len(moved), "moved": moved[:25], "alignment_scale_b_to_a": round(float(s), 4),
               "_doc": "moved: images whose pose differs between the models after alignment; many large moves with a scale far "
                       "from 1 mean the models disagree globally (one is folded or drifted)"}
    return ToolResult(None, summary, cost={"wall": time.time() - started}, images=[str(path)])
