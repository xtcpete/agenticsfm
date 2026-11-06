"""Full rebuild of the showcase with every Pi3X-loop episode (WebP figures, models inside the episode files). Usage: python scripts/showcase/build_all.py <out_dir> [group ids]"""
import json, subprocess, sys, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(sys.argv[1])
PY = "/home/gochen/miniconda3/envs/agenticsfm/bin/python"
def mean_of(*runs, key="auc_all@10"):
    v = {}
    for run in runs:
        for l in (ROOT / "outputs/runs" / run / "results.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l); x = (r.get("metrics") or r).get(key)
                if x is not None: v[r["scene"]] = x
    return f"{100 * statistics.mean(v.values()):.1f}" + ("" if len(v) == 6 else f" over {len(v)} scenes")
reps = lambda p, ids: [f"{p}_blh_rep{i}=rep{i}" for i in ids]
DESC_V2 = ("Pi3X loop v2: the VLM sees each candidate's match inliers when it picks the next batch; Pi3X is aligned to the model on lifted 3D points; "
           "when Pi3X and PnP disagree, match support decides; a final sweep retries skipped images; leftover groups with no usable matches go to the VLM, "
           "which picks overlapping photos as Pi3X anchors, and such poses are registered unverified (no triangulation, no BA).")
G = [  # id, suffix, runs, controller, title, desc
 ("solver-ffvlm", "ffvlm", ["ff_vlm_dev", "ff_vlm_dev_s02", "ff_graph_dev=graph", "ff_graph_dev_s01=graph"], "Pi3X loop, VLM selection",
  "Pi3X loop v1 (Pi3X every round, VLM picks the next images, PnP cross-check, local BA per round); mean {m} (graph-selection control {g})", None),
 ("solver-p1", "p1", ["p1_vlm_dev"], "Pi3X loop v2 (phase 1)", "Pi3X loop v2, phase 1 (inliers in the VLM prompt, point alignment, match-support settling); mean {m}", DESC_V2),
 ("solver-p123", "p123", ["p123_vlm_dev"] + reps("p123", [995, 996, 997, 998, 999]), "Pi3X loop v2 (phases 1-3)",
  "Pi3X loop v2, phases 1-3 (+ VLM placement of leftovers); best run, mean {m}; BLH reruns of the same code (rep995-999)", DESC_V2),
 ("solver-p3b", "p3b", ["p3b_vlm_dev", "p3b_vlm_rep2=rep2", "p3b_graph_dev=graph"], "Pi3X loop v2 (phases 1-3b)",
  "Pi3X loop v2, phases 1-3b (VLM placements must be stable across anchor halves); mean {m}; repeat (rep2) and graph-selection control (graph)",
  DESC_V2 + " Phase 3b: an unverified pose is kept only if Pi3X places the image in the same spot with each half of the VLM's picks."),
 ("solver-p4", "p4", ["p4_vlm_dev"], "Pi3X loop (p4)", "p4: sky masking in render-and-compare, co-location guard; mean {m}", None),
 ("solver-p5", "p5", ["p5_vlm_dev"], "Pi3X loop (p5)", "p5: render fixes (renders from the right side of the buildings); mean {m}", None),
 ("solver-p6", "p6", ["p6_vlm_dev", "p6_vlm_rep2=rep2"] + reps("p6", [998, 999]), "Pi3X loop (p6)", "p6: phase 3c, the render judge can veto VLM placements; mean {m}; repeat and BLH reruns", None),
 ("solver-p7", "p7", ["p7_vlm_dev"] + reps("p7", [998, 999]), "Pi3X loop (p7)", "p7: phase 3c with an alignment requirement (point inliers >= 0.9); mean {m}", None),
 ("solver-p8", "p8", ["p8_vlm_dev"] + reps("p8", [997, 998, 999]), "Pi3X loop (p8)", "p8: retries with other anchor sets, second models handed to VLM placement; mean {m}", None),
 ("solver-p9", "p9", ["p9_vlm_dev"] + reps("p9", [997, 998, 999]), "Pi3X loop (p9)", "p9: reference photos and anchors from match-registered images only; mean {m}", None),
 ("solver-p10", "p10", ["p10_vlm_dev"] + reps("p10", [997, 998, 999]), "Pi3X loop (p10)", "p10: VLM at temperature 0, seeded per question; mean {m}", None),
 ("solver-p11", "p11", ["p11_vlm_dev"] + reps("p11", [997, 998, 999]) + ["p11_m07_rep998=rep998", "p11_m07_rep999=rep999"], "Pi3X loop (p11)",
  "p11: a group is re-placed without its render-inconsistent members; mean {m}", None),
 ("solver-p12", "p12", ["p12_vlm_dev"] + reps("p12", [997, 998, 999]), "Pi3X loop (p12)", "p12: hard stage drops 'agree, render inconsistent'; re-placement needs two survivors; mean {m}", None),
 ("solver-p13", "p13", ["p13_vlm_dev"] + reps("p13", [997, 998, 999]), "Pi3X loop (p13)", "p13: re-placed survivors must stay within 0.2 camera spreads and 3 deg of the joint placement; mean {m}", None),
 ("solver-p14", "p14", ["p14_vlm_dev"] + reps("p14", [997, 998, 999]), "Pi3X loop (p14, current code)",
  "p14 (current code): survivors that move under re-placement are admitted unverified with their joint Pi3X pose; mean {m}", None),
]
COMMON = ["--inline-images", "--inline-models", "--webp", "--jpeg-quality", "58"]
def run(args):
    print(">>", " ".join(args[:8]), flush=True)
    subprocess.run([PY, "scripts/export_site.py", "--out", str(OUT)] + args, cwd=ROOT, check=True)
only = sys.argv[2:]
if not only or "solver" in only:
    run(COMMON + ["--img-side", "900", "--max-points", "25000"])          # open solver v1.3 (default runs and title)
for gid, sfx, runs, ctl, title, desc in G:
    if only and gid not in only: continue
    base = [r.split("=")[0] for r in runs if "=" not in r]
    m = mean_of(*base); g = mean_of("ff_graph_dev", "ff_graph_dev_s01") if gid == "solver-ffvlm" else None
    args = COMMON + ["--img-side", "640", "--max-points", "15000", "--final-only", "--solver-runs", *runs, "--solver-group", gid,
                     "--solver-title", title.format(m=m, g=g), f"--ep-suffix=-{sfx}", "--controller", ctl, "--compare-run", "incremental_v13",
                     "--merge-into", str(OUT / "data/index.json")]
    if desc: args += ["--solver-desc", desc]
    run(args)
