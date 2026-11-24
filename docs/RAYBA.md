# GPU bundle adjustment (rayba): replacing COLMAP's BA

Goal (user, 2026-11-23): at least match COLMAP's bundle adjustment in accuracy, faster, so the final `optimize` and
later the in-growth global BA can drop COLMAP; feedforward depth as an optional prior (MP-SfM, DMap).

Code: the `rayba` package, a git submodule at `third_party/rayba` (github.com/xtcpete/rayba; until step 19 it lived in
`src/agenticsfm/solver/rayba.py`, `rayba_build.py` and `backends/rayba.py`). Offline test scripts in `scripts/analysis/`:
`rayba_prep.py` (agenticsfm env: export a problem from a run's grown model), `rayba_run.py` (any torch env: run
variants), `rayba_eval.py` (agenticsfm env: GT scoring next to the grown model and COLMAP `optimize`),
`colmap_ba_ref.py` (COLMAP's BA with a chosen camera model / backend, same output format).

## Protocol
- Input: the p22 grown models (output of `grow_model`, input of `optimize`), so both BAs start from the same state.
- Reference: COLMAP `optimize` as the run used it: one camera per image-size + focal cluster (tol 0.08), OPENCV,
  principal point fixed, Cauchy 1 px, 2 rounds of up to 100 iterations, points > 4 px removed between rounds (CPU Ceres).
- Iterate on dev only: WRIVA dev (6 scenes, `auc@3/10`, `auc_all@3/10`) and ETH3D dev (13 scenes, `colmap_auc@1/3/5`).
  Confirm on the held-out test splits (WRIVA test 28, ETH3D test 12) at the end. Every change judged on all scenes.
- Times: wall seconds of the BA alone (ours on one A100, COLMAP on CPU). COLMAP's GPU paths could not be tested:
  the `pycolmap-cuda12` 4.2.1 wheel has neither CASPAR (`built without CASPAR_ENABLED`) nor Ceres CUDA (falls back to CPU).

## Formulation
DMap-style ray-anchored BA: one log-depth per source keypoint along its ray, a residual per observation into the target
image; the depth block is diagonal and eliminated by the Schur complement; the reduced camera system (7 per image + K
per intrinsics group) is dense and solved by Cholesky; Levenberg-Marquardt with damping of the whole system (depths
too: undamped depth steps stalled LM), Cauchy loss, residuals in match pixels (long side 1024). Camera models
SIMPLE_RADIAL and OPENCV (COLMAP's definitions), optional per-group principal point (`_pp`). Optional log-depth prior
from Pi3X with a free per-image scale. Jacobians by forward-mode autodiff (checked against finite differences to 1e-10).
Synthetic check (12 cameras, 3k points, 10% gross outliers, 3.7 deg / 10% focal perturbation): rotations to 0.006 deg,
focal and k1 recovered, 0.3 s.

## Log
(newest last)

1. **Matches, SIMPLE_RADIAL** (stars from every verified match within 4 match px of the epipolar geometry, both
   directions). WRIVA dev: parity with `optimize` (auc@3 24.7 vs 25.0, auc@10 62.5 vs 62.6, auc_all@10 58.5 vs 58.6),
   52 s vs 350 s. ETH3D dev: 59.1 / 82.7 / 88.7 vs 64.3 / 85.3 / 90.3, 42 s vs 456 s.
   Pi3X depth prior (lambda 10 / 100): no gain on either dataset (as P5 in PI3X_LOOP.md).
   Tighter Cauchy (0.25), 50 iterations, two rounds with outlier removal: no change.
2. **Tracks instead of matches** (one star per model 3D point): ETH3D 57.6. COLMAP's own BA with SIMPLE_RADIAL on the
   same models: 57.7. So the whole ETH3D gap to `optimize` is the camera model (COLMAP OPENCV 64.3).
3. **OPENCV on matches**: erratic per scene (terrace 58.5 -> 74.0, delivery_area 58.2 -> 45.8), mean 57.8. Refined
   intrinsics look right (delivery_area fx 3402.8 / fy 3405.5 vs GT 3408.6), so the extra parameters fit noise of
   the loosely gated matches.
4. **OPENCV on tracks** (COLMAP's clean observations): ETH3D 64.0 / 85.0 / 90.1 (Cauchy 0.25, 50 its) vs 64.3 / 85.3 /
   90.3, in 29 s vs 456 s. Implementation verified; observations are what matter.
5. **Tight epipolar gate on matches** (1 match px instead of 4): WRIVA SIMPLE_RADIAL 25.0 / 62.4 (parity), OPENCV 24.5;
   ETH3D OPENCV 60.1. Still below tracks on ETH3D: two-view consistency is not enough.
6. **Tracks + OPENCV on WRIVA dev**: 25.1 / 62.4 / 22.9 / 58.4 vs `optimize` 25.0 / 62.6 / 22.9 / 58.6, 26 s vs 350 s.
7. **Hybrid**: round 1 on track observations; then add every direct match whose Sampson error under the round-1
   geometry is below X original px; optionally prune observations above 0.5 match px after round 2 and refine again
   (a multi-view 3D check of each star). One observation per (source keypoint, target keypoint), track members first.

   | dev | ETH3D colmap_auc@1/3/5 | WRIVA auc@3/10, auc_all@3/10 |
   |---|---|---|
   | `optimize` (COLMAP, CPU) | 64.3 / 85.3 / 90.3 (456 s) | 25.0 / 62.6 / 22.9 / 58.6 (350 s) |
   | hybrid, Sampson 1 px | 64.9 / 85.6 / 90.5 (132 s) | 25.2 / 62.5 / 23.0 / 58.6 (152 s) |
   | hybrid, Sampson 2 px | 66.2 / 86.2 / 90.9 (196 s) | 25.4 / 62.7 / 23.2 / 58.7 (226 s) |
   | hybrid, Sampson 4 px + prune 0.5 | **67.4 / 87.0 / 91.5** (333 s) | 25.3 / 62.7 / 23.1 / 58.7 (319 s) |

   First configuration that beats COLMAP on both splits; the speed advantage shrank (about 1M extra observations,
   100+ iterations).
8. **Speed**: profile of a 1.4M-observation WRIVA problem: 80% in building the normal equations (forward-mode
   Jacobian, 25 tangents per observation). Chain-rule Jacobian (analytic in poses and depth, autodiff only through
   the projection; source-ray derivative by the implicit function theorem, once per depth variable): identical to
   1e-9, normal equations 3x faster. Target-target Schur term vectorised (pairs enumerated once). 10 iterations on
   that problem: 9.4 s -> 4.2 s.
9. **Iteration budget** (fast code, hybrid Sampson 4 px + prune 0.5): 50 / 20 / 10 iterations per round give ETH3D
   67.4 / 67.4 / 67.4 at @1 and WRIVA auc@3 25.3 / 25.3 / 25.2: 20 is enough.
10. **Problem construction timed** (the export without the Pi3X prior, which gave nothing, and without its 34 s of
    npz decompression): 6.4 s on the largest dev scene (PTZ s00, 196 images, 2.5M observations). Midpoint
    triangulation of depths without a 3D point batched. Totals below are construction + BA.

    | dev (no prior) | ETH3D @1/3/5 | time | WRIVA auc@3/10, auc_all@3/10 | time |
    |---|---|---|---|---|
    | `optimize` | 64.3 / 85.3 / 90.3 | 456 s | 25.0 / 62.6 / 22.9 / 58.6 | 350 s |
    | hybrid, 20 it | 67.5 / 87.0 / 91.5 | 75 s | 25.3 / 62.7 / 23.0 / 58.7 | 81 s |
    | hybrid, 10 it | 67.4 / 86.9 / 91.4 | 47 s | 25.3 / 62.5 / 23.1 / 58.6 | 53 s |
11. **Intrinsics options** (hybrid, 20 it): merging groups whose refined focal lengths agree within 0.5% and whose
    sizes agree within 1% (undistorted ETH3D crops of one camera were 2-6 groups): ETH3D 67.6 (+0.2), WRIVA same.
    Principal point per group (`opencv_pp`): **ETH3D 76.1 / 90.3 / 93.5**, WRIVA 25.3 / 62.7 / 23.1 / 58.7 (same).
    ETH3D GT principal points sit about 13 px off the image centre, where both BAs had held them. Because COLMAP can
    also refine the principal point, COLMAP with `refine_principal_point=True` is added as a second reference.
12. **Fair reference for the principal point**: COLMAP's BA with `refine_principal_point=True` (same grouping, OPENCV,
    2 rounds). It gains as much on ETH3D but loses on WRIVA, where ours stays neutral:

    | dev | ETH3D @1/3/5 | time | WRIVA auc@3/10, auc_all@3/10 | time |
    |---|---|---|---|---|
    | COLMAP `optimize` (pp fixed) | 64.3 / 85.3 / 90.3 | 456 s | 25.0 / 62.6 / 22.9 / 58.6 | 350 s |
    | COLMAP + principal point | 76.2 / 90.0 / 93.3 | 534 s | 23.2 / 62.1 / 21.0 / 58.1 | 446 s |
    | **ours: hybrid + OPENCV + pp + group merge** | **77.2 / 90.7 / 93.7** | 86 s | **25.3 / 62.7 / 23.1 / 58.7** | 90 s |

    Times for ours include problem construction. Per scene vs `optimize`: ahead or tied everywhere (two WRIVA scenes
    -0.1 auc@3). Vs COLMAP + pp on ETH3D: 9 of 13 ahead, behind on meadow -6.5, courtyard -3.9, kicker -2.5,
    relief -2.4 (mean +1.0); on WRIVA ahead or tied except PTZ s01 (-0.5).
13. **Tool**: `optimize_gpu` (src/agenticsfm/tools/optimize_gpu.py), same interface as `optimize`. Builds the problem
    in the harness env (solver/rayba_build.py), solves in the `rayba` env (backends/rayba.py, `rayba.solve` with the
    defaults above), writes refined poses, one OPENCV camera per group with its principal point, 3D points at their
    solved positions (from each track's source depth), and removes points with mean error > 4 px. Not wired into any
    controller yet. End-to-end on p22 grown models: terrace 88.4 vs 66.7 (9 s vs 16 s), M07 43.1 vs 43.2 (16 s vs
    50 s), PTZ s02 17.9 vs 17.3 (33 s vs 58 s), tool time including the backend start (about 3 s).
    Note: the mean reprojection error after our BA is higher (e.g. 1.24 -> 1.54 px vs COLMAP's lower value) because
    points are anchored on their source keypoint and the loss is tighter; poses are what is scored and what is better.
14. **Held-out test confirmation** (configuration fixed on dev, run once; WRIVA test 28 scenes, ETH3D test 12):

    | test | ETH3D @1/3/5 | time | WRIVA auc@3/10, auc_all@3/10 | time |
    |---|---|---|---|---|
    | grown model (no final BA) | 55.1 / 79.3 / 85.9 | - | 37.8 / 65.6 / 33.4 / 58.6 | - |
    | COLMAP `optimize` (pp fixed) | 65.5 / 85.3 / 89.7 | 490 s | 42.9 / 68.1 / 38.0 / 60.9 | 4348 s |
    | COLMAP + principal point | 73.7 / 88.3 / 91.5 | 542 s | 41.2 / 67.3 / 36.2 / 60.0 | 4551 s |
    | ours, OPENCV (pp fixed) | 69.5 / 87.1 / 90.9 | 65 s | 43.7 / 68.7 / 38.6 / 61.2 | 903 s |
    | **ours, OPENCV + pp + group merge** | **76.3 / 89.5 / 92.5** | 74 s | **43.8 / 68.7 / 38.7 / 61.3** | 998 s |

    6.6x (ETH3D) and 4.4x (WRIVA) faster than `optimize`, more accurate on both. Times as before (ours: construction +
    BA on one A100, three GPU jobs and two COLMAP CPU jobs sharing the machine). Largest problem: t04_v10_s01 (13.3M
    observations, 1.9M depths), 84 s BA, 14 GB.
    Per scene, below `optimize` at auc@3: t01_v01_s08 (tiny, 58.5 vs 65.4; COLMAP+pp collapses there to 38.8),
    t04_v10_s01 (30.6 vs 32.6), the four PTZ SingleZone scenes (-0.7 to -1.0; the principal point costs 0.5 there,
    without it -0.3 to -0.5). ETH3D vs COLMAP + pp: behind on statue (76.1 vs 88.4), observatory (57.4 vs 64.0),
    door (85.2 vs 88.7), terrace_2 (86.4 vs 87.2); on door and terrace_2 the grown model (90.5, 88.0) beats every BA.
    Not tuned on these (test split).
15. **Linear solver for many images** (`Options.solver`). The dense path keeps three C x C float64 matrices
    (C = 7 per image + K per group): fine to ~1-2k images, 51 GB at LaMAR's 6.5k (40 GB A100s here). Also its
    target-target buffer was N x N blocks; now only co-observed pairs. Tried, all on the final dev configuration:
    - *implicit Schur + block-Jacobi PCG* (Agarwal et al. 2010; Ceres ITERATIVE_SCHUR): same solution (1e-6), but
      S v runs over every observation: 50 CG iterations 2.5 s vs 0.63 s for the whole dense Schur + Cholesky on a
      6.9M-observation WRIVA problem. Kept as `implicit` (least memory), not used.
    - *explicit block-sparse S* (the dense path's blocks, never scattered into a matrix) *+ block-Jacobi PCG*: dev
      parity (ETH3D 77.1-77.2, WRIVA 25.2 / 62.6 / 23.0 / 58.6), but 150-500 CG iterations per step on WRIVA
      (aerial, PTZ): 13M-observation problem 152 s vs 81 s dense. CG tolerance 1e-3 / 1e-2: same accuracy, little
      saving.
    - *"arrow" preconditioner* (pose blocks + the exact intrinsics rows / columns): not positive definite in general;
      made SPD (scaled pose blocks) it needs 1355 vs 1415 iterations to 1e-12: the ill-conditioning is pose-pose.
      Dropped.
    - **float32 Cholesky of the explicit S as the preconditioner** (factored in place, relative jitter 1e-6, reused
      across LM steps until CG needs > 20 iterations): CG converges in 5-10 iterations; one solve matches dense float64
      to 3e-12; dev identical to dense (ETH3D 77.2 / 90.7 / 93.7, 89 s vs 86 s; WRIVA 25.3 / 62.7 / 23.1 / 58.7, 95 s
      vs 90 s); 13M-observation problem 85 s vs 81 s. Memory: one C x C float32 (8.5 GB at 6.5k images).
    Default `solver="auto"`: dense up to 8000 unknowns (~1100 images), else block-sparse + float32-Cholesky
    preconditioner up to 64000 (~9000 images), block Jacobi beyond.
16. **Scale test: LaMAR CAB, 6,587 iPhone images** (scripts in the session scratchpad `rayba/lamar/`). GLOMAP on the
    harness's match database (6,587 images, 296k verified pairs) registers everything but is globally wrong (median
    pair error 110 deg, 82 deg within a session; no focal priors), so it only times the BAs: COLMAP `optimize`-style BA
    78 min on CPU, GLOMAP's own BA stages 28 min, ours (block-sparse + float32 Cholesky, 100 LM its) 8-11 min.
    For accuracy: the database's matches triangulated with LaMAR's GT poses (COLMAP point_triangulator, 2.65M points),
    then poses perturbed (rotation per axis, centre per axis, focal) and points re-triangulated from the perturbed
    poses, points above a reprojection cap dropped (as the mapper does). Both BAs start from the same model; tracks
    only (the hybrid problem has 62M observations, beyond one 40 GB GPU in float64). Score: GLUEMAP's pair error on a
    fixed sample of 1M image pairs (the full evaluator builds all 43M pairs). Note LaMAR's phone frames have per-frame
    intrinsics (focal +-1.4 %, principal point spread 31 px), which one camera per image size cannot fit.

    | 6,587 images | auc@3 / @10 | median err | time |
    |---|---|---|---|
    | *small perturbation (0.1 deg, 1 cm, 0.5 % focal; points <= 4 px): the final-BA regime* | | | |
    | start | 91.9 / 97.5 | 0.23 | |
    | COLMAP, `optimize` grouping (one OPENCV camera per size) | 82.5 / 94.1 | 0.40 | 3741 s |
    | ours, same grouping (OPENCV + pp, 20 its) | **89.6 / 96.9** | 0.27 | **129 s** |
    | COLMAP, SIMPLE_RADIAL per image | 84.0 / 94.6 | 0.32 | 1774 s |
    | ours, SIMPLE_RADIAL per image (20 its) | **90.8 / 97.2** | 0.23 | **217 s** |
    | *large perturbation (1 deg, 10 cm, 2 % focal; points <= 40 px): a stress test* | | | |
    | start | 26.1 / 75.9 | 2.27 | |
    | COLMAP, `optimize` grouping | **78.6 / 92.6** | 0.45 | 2344 s |
    | ours, final config's loss (Cauchy 0.25 match px, 100 its) | 66.2 / 89.6 | 0.79 | 1042 s |
    | ours, Cauchy 1 match px, 2 rounds, prune 2 match px (~ COLMAP's 4 px) | 75.8 / 92.4 | 0.49 | 559 s |

    From the final-BA regime both BAs move slightly away from GT (match noise; GT itself), ours less, 14-29x faster.
    From a rough start, our tight loss (tuned for polishing) loses; a COLMAP-like wider loss closes most of the gap.
17. **Wider loss before the final round** (graduated robustness; suggested by the LaMAR stress test, chosen on dev):
    Cauchy 1 match px for the tracks round and round 1, 0.25 for the last round (`DEFAULTS["cauchy_wide"]`).

    | dev | ETH3D @1/3/5 | WRIVA auc@3/10, auc_all@3/10 |
    |---|---|---|
    | 0.25 throughout (step 12) | 77.2 / 90.7 / 93.7 | 25.3 / 62.7 / 23.1 / 58.7 |
    | wide 0.5, then 0.25 | 78.6 / 91.2 / 94.1 | 25.4 / 62.9 / 23.2 / 58.8 |
    | **wide 1, then 0.25** | **79.2 / 91.4 / 94.2** | 25.4 / 62.8 / 23.2 / 58.8 |
    | wide 2, then 0.25 | 78.8 / 91.2 / 94.0 | 25.4 / 62.8 / 23.2 / 58.8 |

    All 13 ETH3D dev scenes improve (+0.3 to +5.3 @1); WRIVA dev per scene -0.3 to +0.5. Insensitive to the wide
    scale (0.5-2). Same time (84 s / 90 s).
18. **Held-out test with the step-17 default** (one run; references as in step 14):

    | test | ETH3D @1/3/5 | time | WRIVA auc@3/10, auc_all@3/10 | time |
    |---|---|---|---|---|
    | COLMAP `optimize` | 65.5 / 85.3 / 89.7 | 490 s | 42.9 / 68.1 / 38.0 / 60.9 | 4348 s |
    | COLMAP + principal point | 73.7 / 88.3 / 91.5 | 542 s | 41.2 / 67.3 / 36.2 / 60.0 | 4551 s |
    | ours, step 12 config (Cauchy 0.25 throughout) | 76.3 / 89.5 / 92.5 | 74 s | 43.8 / 68.7 / 38.7 / 61.3 | 998 s |
    | **ours, default (Cauchy 1, then 0.25)** | **77.0 / 89.7 / 92.5** | 71 s | **44.1 / 68.8 / 39.0 / 61.4** | 993 s |

    Per scene vs `optimize` (first metric, ties within 0.2): WRIVA 20 ahead, 7 behind (t01_v01_s08 -3.3, the four PTZ
    SingleZone -0.7 to -1.0, t04_v10_s01 -0.7), 1 tied; ETH3D 11 ahead, 1 behind (observatory -0.8).
    Tool check (`optimize_gpu`, v2) on p22 grown models: terrace 89.7 vs 66.7 (8 s vs 16 s), M07 43.0 vs 43.2 (14 s vs
    50 s), PTZ s02 18.2 vs 17.3 (28 s vs 58 s), tool time including the backend start.

19. **In the pipeline** (commit 80e3fa5; full runs of the incremental controller, `--vlm decide --select vlm`, as p22):
    - Growth (`grow_model ba=gpu`): the global refinement keeps COLMAP's structure (complete and merge tracks,
      retriangulate; then adjust, normalise, complete and merge tracks, filter points until < 0.05% of observations
      change) with the adjustment on the GPU: the model's tracks only, one intrinsics group per camera (focal + k1,
      principal point fixed, as COLMAP's global BA), Cauchy 1 match px, 20 iterations. Images without 3D points
      (held, unverified) stay fixed. Persistent worker (`backends/rayba.py serve`), so CUDA starts once per run.
      Local BA and the 12-image seed refinement stay COLMAP.
    - End: `optimize_gpu` replaces `optimize`, with the same keep rule (examine quality may not drop by > 0.01).
    - Bug found on the way: `optimize_gpu` v2 wrote one camera per merged group with the first image's size and
      centre; undistorted ETH3D crops of a few pixels' difference got the wrong centre (poses were right, so the
      offline scores stand, but points were filtered against wrong cameras and examine's pair consistency dropped,
      and the keep rule rejected the refinement on courtyard, electro, playground). v3: one camera per group and size.

    | dev, all scenes (one run each) | ETH3D @1/3/5 | WRIVA auc@3/10, auc_all@3/10 |
    |---|---|---|
    | p22: COLMAP BA in growth + `optimize` | 63.7 / 84.9 / 90.1 | 24.7 / 62.4 / 22.5 / 58.4 |
    | p22 grown + `optimize_gpu` offline (step 17) | 79.2 / 91.4 / 94.2 | 25.4 / 62.8 / 23.2 / 58.8 |
    | **GPU BA in growth + `optimize_gpu`** | **78.6 / 91.4 / 94.3** | **26.1 / 64.5 / 23.5 / 60.1** |
    | grown models only (no final BA): p22 / ours | 53.3 / 79.3 / 86.3 vs 60.3 / 83.4 / 89.1 | 20.9 / 60.2 / - / 56.3 vs 22.7 / 62.9 / - / 58.6 (auc@3/10, auc_all@10) |

    - Per scene vs p22: ETH3D 13/13 ahead (+5 to +25 @1); WRIVA 5/6 ahead, M07 -0.9 auc@3 / -0.3 auc@10.
    - Time (WRIVA dev): global BA in growth 1764 s -> 202 s; final BA 349 s -> 101 s; whole runs 7203 s -> 4974 s
      (-31 %). ETH3D dev: global BA 464 s -> 83 s, runs 3165 s -> 2271 s. On PTZ s01, 40 s of in-growth global BA =
      5 s build + 13 s solve + 6 s write-back + ~16 s of COLMAP track upkeep.
    - The keep rule still rejects the final BA on A08 VaryingAltitudes (strong-inconsistent pairs 11.9% -> 12.8%;
      GT: +1.4 auc@10 had it been kept). One scene: the rule is left as it is.
    - Grown models are better with the GPU BA in growth (ETH3D +7 @1), but the final BA brings both to the same
      level on ETH3D; on WRIVA the full runs end above the offline number (growth variance is large on WRIVA: BLH
      +8.7 auc@10).

20. **Held-out test of the pipeline** (g1b: GPU BA in growth + `optimize_gpu` v3; one run each, as p22):

    | test (one run each) | WRIVA auc@3/10, auc_all@3/10 | time | ETH3D @1/3/5 | time |
    |---|---|---|---|---|
    | p22: COLMAP BA in growth + `optimize` | 42.4 / 67.9 / 37.4 / 60.6 | 69,594 s | 65.5 / 85.3 / 89.7 | 3,182 s |
    | **g1b** | **44.2 / 68.5 / 38.8 / 60.6** | **48,669 s** | **75.5 / 87.9 / 90.7** | **2,649 s** |
    | (offline: p22 grown + `optimize_gpu`, step 18) | 44.1 / 68.8 / 39.0 / 61.4 | - | 77.0 / 89.7 / 92.5 | - |
    | (scripted COLMAP pipeline, reference) | 42.6 / 70.6 / - / 65.1 | - | - | - |

    - Time: global BA in growth 19,743 s -> 2,303 s (8.6x), final BA 4,347 s -> 1,039 s (WRIVA); whole runs -30 %
      (WRIVA), -17 % (ETH3D; tiny scenes pay ~15 s of worker start-up).
    - Grown models (before the final BA, largest model): WRIVA auc@3/10/auc_all@10 37.8 / 65.6 / 58.6 -> 41.1 / 67.0 /
      59.3; ETH3D 55.1 / 79.3 / 85.9 -> 59.9 / 80.7 / 86.1.
    - Per scene, WRIVA auc@3: 18 ahead, 1 tied, 9 behind. Two large drops are growth outcomes, already in the grown
      models: ReconstructedArea A06 (46.5 -> 33.2; strong-inconsistent pairs 27 % vs 22 %) and MultiZone s00 (48.5 ->
      34.5; 2 models, 275/300 registered vs 299/300). Large gains likewise: MultiZone s01 +13.5, TrailerPark s02 +11.6.
      Run-to-run swings of this size are known with COLMAP BA too (MultiZone s00 auc@10 34.8 in p21, 78.9 in p22);
      a repeat is needed to tell whether the GPU BA in growth changes the fold rate. PTZ SingleZone -1.0 to -1.2 (as in
      the offline test, step 14).
    - ETH3D @1: 10 of 12 ahead (+4 to +24), observatory and botanical_garden flat; botanical_garden loses @3/@5
      (87.5 -> 73.2) because growth left 4 of 30 images in a second model (p22: 30/30 in one).
    - auc_all@10 flat (60.6): the two drops cancel the gains; the gap to the scripted COLMAP pipeline (65.1) is still
      the A01 folds (step 42 of PI3X_LOOP.md).
21. **Package**: the code moved to the `rayba` repository (github.com/xtcpete/rayba, submodule third_party/rayba,
    v0.2): torch and pycolmap required; `rayba.refine` (final BA) and `rayba.iterative_global_refinement` (drop-in
    for the mapper's) with COLMAP's own BA as the CPU fallback (no GPU: growth uses COLMAP's refinement,
    `optimize_gpu` runs `optimize`'s BA; terrace 67.4 / 89.1 / 93.5 vs 91.3 on the GPU). Behaviour on GPU unchanged
    (problems, solutions and written models identical to the in-tree code on terrace and electro).

## Status (2026-11-23 morning)
- `rayba.solve` defaults = step 17; `optimize_gpu` tool = it, on dev and held-out test more accurate than COLMAP's
  `optimize` at 4-7x less time (14-29x at 6.5k images). Not wired into a controller; nothing committed.
- Open: the in-growth global BA (30 % of a WRIVA run) still uses COLMAP; ray anchoring trusts each track's source
  keypoint (a full 3D-point variant would remove that); the hybrid observations of a 6.5k-image model (62M) need
  float32 storage or chunked normal equations to fit one 40 GB GPU; per-frame intrinsics (phones) vs grouping.
