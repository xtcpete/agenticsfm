# Pi3X loop: experiment log

The open solver with Pi3X run every round (`--loop ff`), started 2026-11-04 on branch `pi3x-loop`.

Each round:
- the next images are chosen, by the match graph or by the VLM;
- Pi3X places them, with anchors from the model;
- each image's Pi3X pose is checked against the COLMAP mapper's PnP pose;
- accepted images are triangulated and locally refined.

Global bundle adjustment runs whenever the model has grown by 25%.

Rules:
- WRIVA dev scenes only (A08, M07, BLH0001, PTZ_A08 s00/s01/s02);
- GT only in the benchmark layer;
- no source edits while a run is going;
- one GPU per scene, on GPUs 4–6 (7 is reserved for the dtwin project);
- a change is judged on all six scenes.

Primary metric: auc_all@10 (an unregistered image counts as wrong). Also reported: auc_all@3, registered images, and cameras more than 10° from GT after alignment.

## Baselines (auc_all@10)

| Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
|---|---|---|---|---|---|---|---|
| Scripted COLMAP controller | 44.3 | 81.0 | 49.0 | 61.6 | 61.5 | 60.9 | 59.7 |
| Open solver v1.3 (PnP growth) | 43.1 | 81.6 | 50.7 | 56.1 | 55.3 | 52.3 | 56.5 |
| Open solver v2, no support check (best open solver) | 43.8 | 82.0 | 39.8 | 59.6 | 60.5 | 58.8 | 57.4 |
| Pi3X loop, graph selection (`ff_graph_dev` + `_s01`) | 23.4 | 82.1 | 46.1 | 48.1 | 49.4 | 50.4 | 49.9 |
| Pi3X loop, VLM selection (`ff_vlm_dev` + `_s02`) | 27.7 | 82.1 | **60.2** | 48.5 | 52.0 | 45.3 | 52.6 |

The Pi3X-loop VLM-selection run is in the showcase (group "Pi3X loop").

## Diagnosis of the baseline (2026-11-04/05)

**VLM selection ignores match strength.** The per-candidate inlier counts were logged but never put into the prompt. Over 122 rounds:
- 49% of the VLM's picks were also in the match graph's top N; picking at random would give 33%;
- 39% of skipped candidates came from the bottom third of the inlier ranking; at random it would be 33%.

**Disagreements end in deferral.** Every image gets a Pi3X pose and a PnP pose.
- Most accepted poses are PnP poses: 1,273 "agree", plus 55 + 29 + 16 on other PnP paths, against 45 from Pi3X.
- When the two disagree, PnP is kept only if render-and-compare confirms it. 599 images ended at "render does not settle it".
- The loop never uses v1.3's match-support check.

**The coverage loss is mostly recoverable.** Images v1.3 registered that the Pi3X loop did not:

| Scene | Only v1.3 | Of which < 5° from GT |
|---|---|---|
| A08 | 29 | 21 |
| PTZ s00 | 17 | 16 |
| PTZ s01 | 13 | 10 |
| PTZ s02 | 16 | 13 |

A08 loses 21 images to "anchors did not align". From round 7, the centre-based alignment drifts: anchor rotation spread up to 18°, because the far-spread anchors pull in other altitudes.

**Pi3X poses alone are not precise enough.** On images whose final pose is within 3° of GT, the median gap between the Pi3X pose and that final pose is:
- 1.1° on BLH, 2.1–2.6° on PTZ, 3.2° on A08 and 4.6° on M07;
- 15–42% of these images are more than 5° off.

So PnP stays as the pose source, and Pi3X checks it.

**BLH back group (13, 28, 33).**
- The three are taken from one spot outside the perimeter wall, 4–19° apart in rotation.
- They match each other with 1,300–2,964 inliers.
- Their 60–130 inliers to inside images are 0% consistent with GT: false matches on repeated brick.
- v1.3 placed 28 from two agreeing Pi3X runs with 0 PnP inliers. GT says it is 9.7° off.

## Plan

Phase 0: commit the baseline.

Phase 1 (one dev pass):
1. Inlier counts in the VLM selection prompt.
2. Disagreements settled by match support (v1.3's normalised check). Pi3X vetoes only when it aligned well that round and its own refined pose has match support.
3. Anchors near the batch only.
4. A final sweep over unregistered images after the last global bundle adjustment.

Offline before Phase 1, from the saved Pi3X depth (`_work/scene*/ff_*/depth_*.npz`):
- point-based alignment, compared with the centre fit;
- a depth-consistency check of candidate poses.

Phase 2: rotation groups.
- Detect co-located images that match each other strongly and fit a homography.
- Place them as one unit: a shared centre, with rotations from their matches.
- Lift matches to 3D with Pi3X depth for PnP.

Phase 3: VLM placement of images without correct matches.
- The VLM picks the overlapping registered images, which become Pi3X anchors, with targeted rematching against them.
- The VLM checks the layout on a top-down map.
- Matches are filtered by 3D consistency.
- These placements are flagged and kept out of bundle adjustment.

## Using Pi3X's dense point map

Every Pi3X round saves depth, per-pixel confidence, intrinsics and extrinsics for each image at 616×406 (`_work/scene*/ff_*/depth_*.npz`). Until now, only render-and-compare used the depth. Five uses, each tested offline on the saved runs before it goes into the solver:

| # | Use | Addresses | Phase |
|---|---|---|---|
| P1 | Align Pi3X to the model on 3D points: anchor keypoints with a model point, lifted through Pi3X depth, then a RANSAC similarity. This replaces the fit on about 10 camera centres. | A08 alignment drift; false Pi3X–PnP disagreements | 1 |
| P2 | Depth-consistency check of a candidate pose: project the model's points into the image and compare their depths with Pi3X depth, up to scale. | the 599 images where render did not settle a disagreement | 1 |
| P3 | Lift matches to 3D with Pi3X depth (scaled to the model), so PnP works where triangulation is thin. | rotation-only groups (PTZ, BLH back), images seeing few 3D points | 2 |
| P4 | Reject matches whose lifted 3D points disagree in a joint Pi3X frame. | lookalike / repeated-structure matches (BLH back group: 0% correct) | 3 |
| P5 | Pi3X depth as a prior in bundle adjustment. | weakly constrained cameras | later (pycolmap BA has no depth term) |

## Log

### Step 0 (2026-11-05): baseline committed
Commit `3e386b6` on branch `pi3x-loop`: the Pi3X loop, VLM next-image selection, the showcase group and this log.

### Step 1 (2026-11-05): offline test of P1, point-based alignment
Script: `scripts/analysis/ff_offline.py`. It replays every Pi3X round of `ff_vlm_dev` (s02 from `ff_vlm_dev_s02`) from the saved depth.

Setup:
- The model at round k is approximated by the final grown model. Its points count only when seen by at least 2 images registered by round k.
- Error is the gap between an aligned Pi3X batch pose and the image's final pose. Only images whose final pose is within 3° of GT are counted.

Methods compared:
- **rot:** the solver's centres + rotations fit.
- **pts:** RANSAC similarity on lifted anchor keypoints, threshold 2% of the scene scale.

Rotation error of aligned Pi3X poses (median / share > 5° / share > 15°) and centre error (fraction of scene scale):

| Scene | n | rot | pts | centre rot → pts |
|---|---|---|---|---|
| A08 | 107 | 3.81° / 43% / 9% | 2.28° / 28% / 15% | 0.233 → 0.075 |
| M07 | 38 | 5.65° / 63% / 0% | 5.60° / 55% / 0% | 0.131 → 0.220 |
| BLH | 31 | 2.21° / 29% / 0% | 1.96° / 6% / 0% | 0.153 → 0.350 |
| PTZ s00 | 141 | 2.99° / 33% / 14% | 2.06° / 25% / 13% | 0.146 → 0.034 |
| PTZ s01 | 133 | 3.48° / 33% / 4% | 2.05° / 33% / 5% | 0.126 → 0.030 |
| PTZ s02 | 113 | 3.52° / 35% / 4% | 1.98° / 29% / 7% | 0.106 → 0.040 |

Findings:
- On A08 and PTZ, point alignment cuts the median rotation error by 1–1.5° and the centre error by 3–4×.
- On BLH it removes most of the 5° tail.
- On M07 it changes little: only 5% of points are inliers, so Pi3X's depth fits this close-up door poorly at a 2% threshold.
- The gross errors (> 15°) are mostly images where both methods fail: 9 of 16 on A08, 16 of 18 on s00. That is Pi3X's own pose for the image being wrong, which no alignment can fix; PnP and verification have to catch those.
- Point alignment sometimes locks on wrongly: 7 extra > 15° cases on A08, and those rounds have a low point-inlier fraction (median 0.18 vs 0.28).
- A gate, use points when their inlier fraction ≥ 0.3 and otherwise the rot fit, gives:
  - A08: 2.56° median, 31% > 5°;
  - PTZ: about 2.0–2.6° median, about 4 points fewer > 5°;
  - M07 and BLH unchanged.

  Picking the better of the two per round with hindsight would reach 1.6–1.9°.

**Decision:**
- P1 goes into Phase 1, with the 0.3 inlier gate and a fallback to the rot fit. The tighter alignment should cut false disagreements at the 10° agreement test.
- The remaining > 15° cases must be caught by PnP and its match support, which is Phase 1 item 2.

### Step 2 (2026-11-05): offline test of P2, the depth-consistency check
Script: `scripts/analysis/ff_depthcheck.py`.

Setup:
- Images whose final pose is within 3° of GT; up to 60 per scene.
- Each image's Pi3X depth comes from the first round it was batched in.
- The model points this image does not observe are projected into the candidate pose. The score is the share of points whose depth agrees with Pi3X depth within 15%, after fitting the scale.
- Candidates: the final pose, and the same centre rotated by 2–45°.

Mean score of the true pose, and how often it beats a pose rotated by 10° and by 45°:

| Scene | true | 10°: wins | 45°: wins |
|---|---|---|---|
| A08 | 0.71 | 72% | 92% |
| M07 | 0.86 | 95% | 100% |
| BLH | 0.63 | 61% | 74% |
| PTZ s00 | 0.36 | 57% | 85% |
| PTZ s01 | 0.32 | 55% | 62% |
| PTZ s02 | 0.28 | 58% | 65% |

Findings:
- The check works on M07 (close range, strong depth structure) and partly on A08.
- On the PTZ scenes it is close to chance, even at 45°. Distant scenes seen by a long lens have little depth variation, and many projected points are occluded.
- It cannot be a general arbiter.

**Decision:** P2 stays out of Phase 1. Disagreements will be settled by match support (Phase 1 item 2), which uses the raw matches and does not depend on depth structure.

### Step 3 (2026-11-05): Phase 1 implemented
All new features are on by default. `--ff-off <list>` switches them off for ablations; `grow_model` cache version 18.

| Feature (`--ff-off` name) | Change | Code |
|---|---|---|
| `evidence` | The VLM selection prompt lists each candidate's verified inliers to the model and the number of model images it matches. The VLM is told to prefer many matches and skip only lookalike places or images with no usable content. | `Advisor.select_next(evidence=...)`, `Solver.vlm_select` |
| `fill` | The VLM's picks are topped up to the round size with unskipped candidates in match order. The VLM vetoes; it no longer shrinks the round. | `Solver.vlm_select` |
| `anchors` | Anchors are the 8 registered images most matched to the batch. Far-point anchors are added only when fewer than 4 matched ones exist; previously 6 matched + 4 far. | `Solver.ff_anchors` |
| `points` | P1: align Pi3X to the model on lifted anchor keypoints. RANSAC similarity with a threshold of 2% of the scene scale. Used when ≥ 30% of points fit; otherwise centres + rotations, then centres. | `Solver.align_points` |
| `settle` | When the Pi3X and PnP poses disagree after refinement, compute the match support of each pose: v1.3's normalised check on all strong pairs, without registering the image (`Solver.support_pose`). The rule follows the table below. | `Solver.ff_round` |
| `sweep` | After the final global bundle adjustment, every unregistered image gets one more try in match-ranked rounds with no VLM, until a round adds nothing; then a final global bundle adjustment. | `Solver.sweep` |

How `settle` decides:
- **PnP supported:** accept PnP unless Pi3X aligned well that round and Pi3X's own refined pose is also supported.
  - Aligned well means on points, or on rotations that agree within 5°.
  - If both poses are supported under a good alignment, it is a real conflict and render-and-compare decides as before.
- **Only Pi3X supported:** accept the Pi3X pose.
- **Anchors did not align:** PnP alone is accepted when its pose is supported.
- **No PnP pose:** the refined Pi3X pose is accepted when it is supported and has ≥ 15 inliers.

Each image's row now records the support of both poses, `align_good`, and the point-alignment diagnostics.

### Step 4 (2026-11-05): how common are co-located (rotation-only) camera groups? (GT, offline)
A group is a set of cameras whose GT centres lie within 2% of the median camera spread of each other.

| Scene | Images | Groups | Images in groups (largest groups) | Missed by Pi3X loop (in a group) | Missed by v1.3 (in a group) |
|---|---|---|---|---|---|
| A08 | 150 | 11 | 22 (all pairs) | 30 (4) | 1 (0) |
| M07 | 50 | 0 | 0 | 0 | 2 |
| BLH | 39 | 9 | 30 (6, 5, 4, 4, 3) | 3 (3) | 3 (3) |
| PTZ s00 | 198 | 38 | 108 (8, 8, 8, 8, 4) | 21 (6) | 4 (1) |
| PTZ s01 | 182 | 38 | 92 (4, 4, 4, 4, 4) | 14 (6) | 6 (0) |
| PTZ s02 | 174 | 39 | 87 (4, 3, 3, …) | 21 (8) | 9 (2) |

COLMAP's two-view label PLANAR_OR_PANORAMIC marks only a few of these pairs: 56 on s00, 9 on s01, 1 on s02. Almost every co-located pair is labelled UNCALIBRATED, so the solver needs its own test, such as homography against fundamental-matrix inliers.

Phase 2 could address about 27 of the 89 images the Pi3X loop misses. Phase 1 targets the rest.

### Step 5 (2026-11-05): offline test of a pure-rotation (co-located pair) test
For a pair with at least 50 inliers:
- fit a homography between the two images' undistorted, normalised coordinates (RANSAC, 4 px);
- factor it as diag(a, a, 1)·R, where a is the focal ratio;
- accept when the homography fits ≥ 60% of the matches, the two focal estimates agree within 10%, and the factored matrix is orthogonal within 0.1.

Checked against GT: co-located means centres within 2% of the camera spread; distant means beyond 10%, 400 sampled pairs per scene.

| Scene | co-located pairs | pass | rotation error of passing pairs (median / p90) | distant pairs passing |
|---|---|---|---|---|
| A08 | 10 | 30% | 0.81° / 1.58° | 0 / 400 |
| BLH | 44 | 36% | 0.21° / 0.46° | 0 / 400 |
| PTZ s00 | 127 | 59% | 0.58° / 2.78° | 0 / 400 |
| PTZ s01 | 63 | 22% | 0.51° / 1.90° | 0 / 400 |
| PTZ s02 | 52 | 4% | 0.85° / 1.26° | 0 / 400 |

Findings:
- The test is precise: no false positives in 2,000 distant pairs, and passing pairs give the relative rotation to within about 1°.
- Recall is low on s01/s02, where the homography fits only about a quarter of the matches of co-located pairs. Near structure gives parallax even within 2% of the spread.
- Without undistortion, BLH passes only 5%, so the solver must undistort with each camera's current parameters.

**Decision:** Phase 2 attaches an image to a registered co-located partner (shared centre, rotation from the homography). It accepts only through the match-support check, which already handles pure-rotation pairs through the rotation-transfer term.

### Step 6 (2026-11-05): smoke runs and a focal-length fix
Smoke runs: `p1_smoke_graph` (M07, graph selection, no VLM) and `p1_smoke_vlm` (BLH, VLM selection).

- **M07:** 82.5, all 50 images (baseline 82.1). The new paths ran: "disagree; matches support PnP" once, and the sweep (+0, nothing left).
- **BLH:** 41.3 (baseline 60.2). 35 images in the main model, two of them > 10° from GT (32 at 11.8°, 39 at 10.7°). `scripts/analysis/ff_paths.py` traces both to the "feedforward and PnP agree" path.
  - Image 32: PnP had 39/328 inliers and estimated a focal of 2,431 px, where Pi3X said 4,932 px. The rotations agreed (0.9°, view gap 9.7°), so the pose was accepted with half the focal length.
  - In the baseline, PnP failed on this image and it was never accepted this way.

**Fix:**
- Pi3X and PnP agree only if PnP's own focal estimate is within ×1.25 of Pi3X's. The test applies only when the mapper actually changed the focal (`focal_estimated`).
- When the focals disagree, the image goes to the match-support path.
- Cache version 19.

Commit `007bc30`. The full VLM dev pass `p1_vlm_dev` was launched on it: one process per GPU, scenes one after another (GPU 4: s00, A08; GPU 5: s01, BLH; GPU 6: s02, M07).

### Step 7 (2026-11-05): P5, Pi3X depth in bundle adjustment (offline test, running)
Requested on 2026-11-05: use Pi3X's predicted 3D points in bundle adjustment.

pycolmap 4.2.1's bundle adjuster has no hook for custom costs, and pyceres is not installed. So `scripts/analysis/depth_ba.py` implements the test as its own sparse least-squares bundle adjustment (scipy `least_squares`, trf + lsmr, soft-L1 loss) over poses and points, with intrinsics fixed.

Residuals:
- reprojection error, divided by 1 px;
- for every observation with Pi3X depth: (log z − log d_Pi3X(u, v) − b_image) / σ_d. b_image is a free per-image log scale, since Pi3X depth is relative.

Variants: the submitted model, plain BA with the same solver (control), and BA + depth at σ_d = 0.2 / 0.1 / 0.05. Each is scored with the benchmark evaluator and GT rotation errors.

**BLH** (ff_vlm_dev model, 20 iterations):

| Variant | auc@3 | auc@10 | GT rot median | > 5° |
|---|---|---|---|---|
| submitted | 29.6 | 60.2 | 1.78° | 0 |
| plain BA | 29.6 | 60.2 | 1.81° | 0 |
| + depth σ 0.2 | 29.5 | 60.2 | 1.81° | 0 |
| + depth σ 0.1 | 29.4 | 60.1 | 1.82° | 0 |
| + depth σ 0.05 | 29.2 | 60.1 | 1.84° | 0 |

On a well-constrained model the depth term changes nothing, and a stronger prior costs slightly. A08, M07 and PTZ are running.

### Step 8 (2026-11-05): Phase 1 dev pass `p1_vlm_dev` (VLM selection), first scenes
| Scene | auc@3 | auc@10 | registered | baseline ff_vlm auc@10 | v1.3 | best open solver (v2, no support) |
|---|---|---|---|---|---|---|
| PTZ s00 | 19.4 | **59.0** | 195/198 | 48.5 | 56.1 | 59.6 |
| PTZ s01 | 17.2 | **56.6** | 179/182 | 52.0 | 55.3 | 60.5 |
| PTZ s02 | 16.3 | **55.8** | 170/174 | 45.3 | 52.3 | 58.8 |
| M07 | 40.0 | 82.0 | 50/50 | 82.1 | 81.6 | 82.0 |

Wall times: 1,350–1,760 s per PTZ scene, 455 s on M07.

GT audit of the acceptance paths (`ff_paths.py`), count / median error / count > 10°:

| Path | s00 | s01 | s02 |
|---|---|---|---|
| Pi3X and PnP agree | 151 / 1.59° / 1 | 144 / 1.83° / 0 | 124 / 1.97° / 0 |
| **disagree; matches support PnP** (new) | 26 / 1.55° / 1 | 14 / 1.83° / 2 | 26 / 2.27° / 2 |
| sweep (new) | 6 images / 2 > 10° | 1 / 0 | 3 / 0 |

Settling disagreements by match support recovers most of the deferred PTZ images at the same precision as the agree path. This is the main source of the +4.6 to +10.5 gain.

**P5 results on all dev scenes** (ff_vlm_dev models, 20 iterations; auc@10, with cameras > 5° in brackets):

| Scene | submitted | plain BA | depth σ 0.2 | σ 0.1 | σ 0.05 |
|---|---|---|---|---|---|
| A08 | 27.7 (6) | 27.7 (6) | 27.7 (6) | 27.7 (7) | 27.7 (7) |
| M07 | 82.1 (0) | 82.1 (0) | 82.1 (0) | 82.1 (0) | 82.1 (0) |
| BLH | 60.2 (0) | 60.2 (0) | 60.2 (0) | 60.1 (0) | 60.1 (0) |
| PTZ s00 | 48.5 (21) | 48.5 (21) | 48.5 (21) | 48.5 (21) | 48.4 (21) |
| PTZ s01 | 52.0 (20) | 52.0 (20) | 52.0 (20) | 52.0 (20) | 52.0 (20) |
| PTZ s02 | 45.3 (20) | 45.3 (20) | 45.3 (21) | 45.3 (21) | 45.4 (21) |

The depth prior changes nothing measurable: ±0.1 auc, median rotation within 0.05°.
- Cameras that are well constrained by matches do not need it.
- Cameras that are wrong (> 5°) are wrong placements, not weakly constrained ones, and a depth term cannot move them back.
- Cameras with no observations at all (Phase 3's unverified ones) get no depth residuals either.

**Decision:** P5 is not added to the solver. Pi3X's 3D points earn their place in alignment (P1) and in filtering matches for PnP (P3/P4, Phase 3), not in bundle adjustment.

### Step 9 (2026-11-05): Phase 1 dev pass complete
| Scene | auc@3 | auc@10 | registered | before Phase 1 | v1.3 | best open solver |
|---|---|---|---|---|---|---|
| A08 | 18.3 | **44.8** | 149/150 | 27.7 | 43.1 | 43.8 |
| M07 | 40.0 | 82.0 | 50/50 | 82.1 | 81.6 | 82.0 |
| BLH | 20.0 | 39.9 | 33 (+6 in a second model) | **60.2** | 50.7 | 39.8 |
| PTZ s00 | 19.4 | 59.0 | 195/198 | 48.5 | 56.1 | 59.6 |
| PTZ s01 | 17.2 | 56.6 | 179/182 | 52.0 | 55.3 | 60.5 |
| PTZ s02 | 16.3 | 55.8 | 170/174 | 45.3 | 52.3 | 58.8 |
| **mean** | 21.9 | **56.4** | | 52.6 | 56.5 | 57.4 |

**A08:**
- 149 of 150 images, median GT error 0.75°.
- 39 images came in through "disagree; matches support PnP" (median 0.86°).
- Point alignment and matched anchors removed the "anchors did not align" stall.
- 2 cameras > 10° (agree path 0137 and 0097; sweep 0147).

**BLH regression** (60.2 → 39.9). The cause is the hard-image fallback (`place_hard`, unchanged), running on a different model:
- Hard round 7 accepted 12, 22 (a co-located pair) and 34 on "placements agree, render inconsistent". 12 and 22 ended 8–9° off; in the baseline they were within 2°.
- Hard round 8 accepted 2, 32, 38 and 36. 2, 32 and 38 were later filtered out; 36 ended 19° off.
- The leftover component {2, 13, 28, 32, 33, 38} became a second model with no 3D points, which was not absorbed (the baseline absorbed 5 of its images).

The Phase 1 changes did their job on BLH (16 agree, 0 > 10°). The loss is in the fallback's acceptance of "placements agree" without match support. That is the same weakness that placed v1.3's image 28 at 9.7°.

Phase 2 and Phase 3 target exactly these leftover groups (co-located pairs, the back group). Patches applied: `rotation_fit`/`rotation_attach` (Phase 2) and `vlm_place` with `lifted_pnp` and `layout_figure` (Phase 3). Cache version 20. Smoke runs `p3_smoke` on BLH and PTZ s02.

### Step 10 (2026-11-05): Phase 2–3 smoke run on BLH (`p3_smoke`)
**BLH: auc@3 30.0, auc@10 68.5** (best ever on this scene; was 60.2 in the baseline and 39.9 after Phase 1). **39/39 images, none more than 5° from GT.**

What `vlm_place` did:
- **Leftover group:** {13, 28, 33, 32}.
- **The VLM's overlap pick:** 6 placed photos (2, 38, 36, 29, 19, 18) showing "brick building with white tower and the white container/modular structure".
- **Alignment:** Pi3X on the group plus these anchors aligned on points with 98% inliers.
- **Match check:** of the group's 28–231 matches to anchor model points, **0** survived the depth-consistency filter. This agrees with the GT finding that these matches are all false. So no match-confirmed pose.
- **Layout check:** the VLM judged the map plausible (confidence 0.7): "red 4 lies just southeast of the container row and looks northwest, matching the container-and-road composition seen in the reference photos".
- **Result:** all 4 were registered unverified, with no points and no bundle adjustment. All 4 are within 5° of GT.

The hard fallback also behaved better on this model: 7 "placements agree, render inconsistent" images, median 1.91°, none > 5°. Phase 2's rotation attach was not needed here.

### Step 11 (2026-11-05): Phase 2–3 smoke run on PTZ s02, and a precision problem with unverified placements
**PTZ s02: auc@10 57.0** (Phase 1: 55.8). 174/174 images, 25 > 5° and 6 > 10° from GT.

Results:
- **Sweep:** +2 images.
- **`vlm_place`:** one leftover group {145, 155, 139, 156, 13}. The VLM picked 6 overlapping photos and judged the layout plausible (0.7): "each proposed camera sits on the surrounding embankment and looks inward across the flooded pond".
- **Unverified poses vs GT:** 0013 12.7°, 0139 25.2°, 0145 38.6°, 0155 5.5°, 0156 29.5°. **4 of 5 are wrong.**
- The score still rose, because an unregistered image already counts as wrong in auc_all. But this breaks the correctness-over-coverage rule. On BLH the same check, with the same confidence of 0.7, admitted 4 correct poses, so the layout check cannot separate the two cases.
- **Phase 2:** `rotation_attach` was never triggered on s02 (0 attempts). No deferred image had a registered co-located partner that passed the homography test. Offline, the test passed only 4% of s02's co-located pairs.

**Fix (Phase 3b, applied after the running passes finish):** an unverified placement must be *stable*.
- Pi3X runs again with each half of the VLM's picks as the only anchors: two disjoint sets of at least 3.
- Each image must land within 5° and 5% of the scene scale in both runs.
- A guess that depends on which anchors Pi3X sees is left out.
- Cost: two extra Pi3X calls per leftover group.

Passes in flight on commit `3facb08`:
- `p123_vlm_dev`: GPU 4 (s00, A08), GPU 6 (s01, M07); GPU 5 reused the smoke results for s02 and BLH.
- `p123_graph_dev`: GPU 5, all six scenes with match-graph selection, as a control for the VLM selection.

### Step 12 (2026-11-05): full pass with Phases 1–3 (`p123_vlm_dev`, commit `3facb08`)
| Scene | auc@3 | auc@10 | registered | Phase 1 | baseline ff_vlm | v1.3 | scripted |
|---|---|---|---|---|---|---|---|
| A08 | 18.3 | 44.7 | 149/150 | 44.8 | 27.7 | 43.1 | 44.3 |
| M07 | 40.6 | 82.2 | 50/50 | 82.0 | 82.1 | 81.6 | 81.0 |
| BLH | 30.0 | **68.5** | 39/39 | 39.9 | 60.2 | 50.7 | 49.0 |
| PTZ s00 | 15.6 | 52.8 | 197/198 (2 models) | 59.0 | 48.5 | 56.1 | 61.6 |
| PTZ s01 | 17.2 | 57.2 | 182/182 | 56.6 | 52.0 | 55.3 | 61.5 |
| PTZ s02 | 17.2 | 57.0 | 174/174 | 55.8 | 45.3 | 52.3 | 60.9 |
| **mean** | 23.2 | **60.4** | | 56.4 | 52.6 | 56.5 | 59.7 |

The first open-solver mean above the scripted COLMAP controller (59.7) and the best earlier open solver (57.4).

**Run-to-run variation from VLM selection:**
- PTZ s00 fell from 59.0 (Phase 1) to 52.8. Phases 2–3 act only at the end, and rotation attach never triggered, so both runs used the same code up to the leftovers.
- The difference is the VLM's choices. They are not deterministic, so a different batch order led to a different growth history: 8 cameras > 10° instead of 3, including one from the new path "disagree; matches support feedforward" at 24°.
- The match-graph control (`p123_graph_dev`, deterministic order) scored **56.4** on s00 with 198/198 images. It was stopped after s00 to apply Phase 3b; the remaining scenes rerun in `p3b_graph_dev`.
- Single-run differences of a few points on PTZ are therefore within noise. Later comparisons use the graph-selection control and, time permitting, repeated VLM runs.

On s00 the VLM layout check rejected the one leftover (149) as "implausible": "the photo depicts an open field with a parked white car … not the building edge". That was correct; the pose was not registered.

### Step 13 (2026-11-05, 02:40): Phase 3b passes launched (commit `8ec1f66`, cache version 21)
- `p3b_vlm_dev`: GPU 4 (BLH, s00, A08), GPU 6 (s02, s01, M07).
- `p3b_graph_dev`: GPU 5, all six scenes, match-graph selection (the control).

### Step 14 (2026-11-05, 03:35): Phase 3b VLM pass (`p3b_vlm_dev`)
| Scene | auc@3 | auc@10 | registered | Phases 1–3 (`p123`) | Phase 1 (`p1`) |
|---|---|---|---|---|---|
| A08 | 18.2 | 45.0 | 149/150 | 44.7 | 44.8 |
| M07 | 41.2 | 82.4 | 50/50 | 82.2 | 82.0 |
| BLH | 21.6 | 44.5 | 34/39 | 68.5 | 39.9 |
| PTZ s00 | 17.4 | 56.3 | 196/198 (2 models) | 52.8 | 59.0 |
| PTZ s01 | 17.8 | 59.9 | 182/182 | 57.2 | 56.6 |
| PTZ s02 | 16.1 | 55.4 | 174/174 | 57.0 | 55.8 |

| **mean** | | **57.3** | | 60.4 | 56.4 |

- **PTZ s00 crashed at start:** `grow_model failed: RuntimeError: No registered database factory succeeded` (pycolmap could not open the database; 0.6 s in). This is transient, not a code error; the scene reran on GPU 4 (`p3b_vlm_dev_g4b.log`) and scored 56.3.
- **Phase 3 added nothing on any scene this time:**
  - s01 and s02 had no leftover group with matches to the model (s02's 5 leftovers formed a separate model and are excluded).
  - On A08 the one leftover (149) was rejected by the stability check: the halves gave placements 14.3° apart.
  - On BLH the leftover group was {32, 2, 38, 28, 33}. The VLM picked only 2 overlapping photos (36, 13), but the stability check needs at least 3 per half, so all 5 were left out. The depth filter again kept 0 of 30–224 matches (the back of the building is only seen from the back).
- **BLH fell back to 44.5 because growth took a different path, not because of Phase 3b:**
  - In this run 13, 22, 12, 34 and 32 entered through the hard fallback "placements agree, render inconsistent".
  - 28 and 33 also entered that way, but were then filtered out in bundle adjustment.
  - The 4 hard-path images have a median error of 7.1° (22 at 11°).
  - BLH scores across three runs of nearly the same code: 39.9, 68.5, 44.5. With 39 images, a few cameras move auc@10 by 10–20 points.
- **Path audit (ff_paths.py, GT only for scoring):**
  - "feedforward and PnP agree" stays precise: median errors 0.7–1.9°, and only 2 of 92 A08 images > 10°.
  - "disagree; matches support PnP" is about as good.
  - The weak paths are the hard fallback "placements agree, render inconsistent" (BLH) and "disagree; render confirms PnP" on s02 (2 images, one at 20°).

### Step 15 (2026-11-05, morning): match-graph control and repeat runs (run-to-run variation)
`p3b_graph_dev` is the same code with match-graph selection; `p3b_vlm_rep2` repeats the VLM pass (run with `--support-pairs 999`, which only changes the cache key: no image has more than 197 pairs).

| Scene | `p1_vlm` | `p123_vlm` | `p3b_vlm` | `p3b_vlm_rep2` | `p123_graph` | `p3b_graph` | v1.3 | scripted |
|---|---|---|---|---|---|---|---|---|
| A08 | 44.8 | 44.7 | 45.0 | running | | 43.8 | 43.1 | 44.3 |
| M07 | 82.0 | 82.2 | 82.4 | 82.9 | | 82.1 | 81.6 | 81.0 |
| BLH | 39.9 | 68.5 | 44.5 | running | | 40.7 | 50.7 | 49.0 |
| PTZ s00 | 59.0 | 52.8 | 56.3 | running | 56.4 | 59.0 | 56.1 | 61.6 |
| PTZ s01 | 56.6 | 57.2 | 59.9 | 59.1 | | 58.0 | 55.3 | 61.5 |
| PTZ s02 | 55.8 | 57.0 | 55.4 | 53.8 | | 50.4 | 52.3 | 60.9 |
| **mean** | 56.4 | 60.4 | 57.3 | | | 55.6 | 56.5 | 59.7 |

- **The match-graph control is not deterministic either.** s00 scored 56.4 in `p123_graph` and 59.0 in `p3b_graph`, and the code differs only after the last refinement. Other sources of variation remain: render-and-compare VLM answers, RANSAC, and Pi3X on GPU.
- **The spread per scene across runs of near-identical code:**
  - PTZ: 3–6 points (s00 52.8–59.0, s02 50.4–57.0).
  - BLH: 39.9–68.5. With 39 images, the back-of-building group (13/28/33, plus 2/32/38) is all or nothing, and the hard fallback "placements agree, render inconsistent" sometimes admits 8–13° poses there (0012 = 13° in `p3b_graph`, 0022 = 11° in `p3b_vlm`).
  - A08 and M07 are stable (±1).
- **What is robust:**
  - Phase 1 lifted A08 from 27.7 to 44–45 in every run.
  - Every Pi3X-loop run of PTZ s01 and s02 is at or above v1.3, except one graph s02 run.
  - The 60.4 of `p123_vlm_dev` was the top of the distribution, not the expected value. The fair summary of the current code is about 57 ± 2 mean auc@10 with VLM selection, against 56.5 for v1.3 and 59.7 for the scripted COLMAP controller.
- **Path audit of `p3b_graph` (GT only for scoring):**
  - "feedforward and PnP agree" stays the precise workhorse (median 0.7–1.7°).
  - The errors > 10° are spread thinly over the weak paths: "disagree; render confirms PnP/feedforward", hard placements, and "agree after refining" (0066 = 45° on s02).
  - Isolated large failures also occur inside "agree": 0160 = 164° on s00, and 0102/0073/0105 = 24–30° on s02.

### Step 16 (2026-11-05, morning): offline test of the Phase 3b stability check (`scripts/analysis/ff_stability.py`)
- **Method:** for the two `p123_vlm_dev` leftover groups whose unverified poses we know are right (BLH) or wrong (s02), rerun Pi3X three times: with the solver's anchors, and with each half of the VLM's picks. Align each run to the final model and score it against GT. Alignment fell back to rotations and centres for all runs, because no point alignment passed 30%. Output: `outputs/analysis/ff_stability_p123.json`.

| Scene | Image | Error, all anchors | Error, half 0 / half 1 | Rotation gap between halves | Centre gap / scene scale |
|---|---|---|---|---|---|
| BLH | 0013 | 2.9° | 2.9° / 4.1° | 6.1° | 7.6 |
| BLH | 0028 | 4.1° | 0.9° / 4.9° | 4.6° | 7.9 |
| BLH | 0032 | 2.3° | 2.9° / 1.6° | 1.6° | 0.65 |
| BLH | 0033 | 3.4° | 1.2° / 4.0° | 4.7° | 7.9 |
| s02 | 0013 | 21.0° | 46.1° / 18.7° | 28.9° | 0.50 |
| s02 | 0139 | 14.4° | 11.1° / 11.9° | 21.9° | 0.50 |
| s02 | 0145 | 28.3° | 58.7° / 26.1° | 34.7° | 0.39 |
| s02 | 0155 | 15.0° | 50.6° / 14.0° | 46.1° | 0.37 |
| s02 | 0156 | 20.1° | 7.0° / 15.5° | 14.7° | 0.77 |

- **The rotation gap separates the two groups:**
  - The correct BLH poses have gaps of 1.6–6.1°.
  - The wrong s02 poses have gaps of 14.7–46°.
  - So the check has the right signal, and it would have stopped all of s02's wrong admissions.
- **The centre test is useless here.** With only 3 anchors, aligning by rotations and centres puts the camera centres several scene scales apart even when the rotation is right. The online rule (5° and 5% of scene scale) would therefore also reject the correct BLH poses.
- **Rotation alone at 5° would reject BLH 0013 (6.1°).** The sample is 9 images on 2 scenes, too small to set a threshold. Any change (rotation gap only, a threshold near 10°) needs more leftover groups first, per the no-overfitting rule.
- **Phase 3b as run also needs 6 picks.** On `p3b_vlm` BLH the VLM picked only 2 overlapping photos, so the check could not run. Splitting all of the solver's anchors (picks plus matched anchors) into halves would make it applicable.
- **Status:** Phase 3b stays as committed: conservative, admitting nothing wrong but also nothing right. The proposed change (rotation-gap-only test on halves of all anchors) is listed under next steps and not applied.

### Where things stand (2026-11-05 morning)
- **Implemented:** every item of the plan.
  - Phase 1: inliers in the prompt, matched anchors, point alignment P1, match support settling disagreements, focal check, final sweep.
  - Phase 2: rotation attach.
  - Phase 3: VLM overlap pick, Pi3X with the picks, depth-filtered PnP, layout check, unverified registration.
  - Phase 3b: the stability check.
  - Point-map tests: P1 kept, P2 dropped, P5 (depth in bundle adjustment) dropped.
- **Measured:** about 57 ± 2 mean auc@10, against 52.6 for the Pi3X-loop baseline and 56.5 for v1.3. The best single run was 60.4.
- **Kept:**
  - P1: A08 +17 every run, PTZ +3..+10.
  - Settle by match support and the focal check: removed the 11.8° BLH image 32.
- **Inert so far:**
  - Rotation attach: 0 triggers.
  - VLM placement: correct on BLH once; after Phase 3b it admits nothing.
- **Next, ranked:**
  1. Halves of all anchors, with a rotation-gap-only stability test, checked on more leftover groups.
  2. Tighten the BLH hard fallback "placements agree, render inconsistent" (8–13° admissions).
  3. Multi-seed runs: 3 per scene before quoting any change smaller than about 3 points.
  4. Isolated "agree" failures (0160 at 164°, s02 at 24–30°), which come from Pi3X and PnP agreeing on the same wrong pose.

### Step 17 (2026-11-05): GT check of the BLH render_choose figures in `p123_vlm_dev` (68.5 run)
The user noted that in both render-choose figures (28 in hard round 7, 33 in hard round 9) both candidate renders look wrong.

- **Method:** recompute both candidate Pi3X placements from the saved results (`_work/scene0/ff_hard{7,9}`), align them to the model on their anchors, and score them against GT (`scripts/analysis/hard_candidates_blh.py`).
- **Units:** centre error is a fraction of the GT scene scale. These cameras are 3.4–4.5 scene scales from the scene centre.

| Image | Candidate A | Candidate B | VLM's pick | Final (VLM placement, unverified) |
|---|---|---|---|---|
| 28 | 5.5° / 0.39 | **3.2° / 0.075** | B (right) | 3.3° / 0.055 |
| 33 | 4.5° / 0.35 | **1.2° / 0.074** | A (wrong) | 2.6° / 0.053 |

- **B was close to right both times, and A was off in position by about 10% of the camera's distance.** Both figures still look wrong, for two reasons:
  - The camera stands outside the perimeter wall. The model has no points on the outside of the wall, so nothing occludes the towers and building behind it; the render shows them over the wall.
  - Pi3X sky depth adds blurred sky "points".
- **What the figures show is the model's coverage, not pose error.** Render-and-compare cannot verify a viewpoint whose visible surface is not in the model; the VLM can only guess (one of two right here).
- **Effect on this run:** none. In `--vlm log` mode the answer was not used; the solver rejected both images ("placements disagree"), and `vlm_place` registered them later, correctly (centre error 1.6% of distance).
- **Far cameras of the final model (GT):**
  - Rotations are fine, but several cameras about 4.5 scene scales out are misplaced along their viewing axis:

    | Image | Centre error (scene scale) | Fraction of the camera's distance |
    |---|---|---|
    | 002 | 1.22 | 27% |
    | 032 | 0.82 | 18% |
    | 038 | 0.54 | |
    | 036 | 0.47 | |
    | 034 | 0.47 | |

  - Images 22 and 12 (0.27) came in through the hard path.
  - Of the VLM placements, 13, 28 and 33 are good. Image 32 has a good rotation (3.2°) but a depth error along its viewing axis. Narrow-view distant cameras are poorly constrained in depth by Pi3X and by few matches.

### Step 18 (2026-11-05): repeat VLM pass complete; co-located cameras checked against GT
- **`p3b_vlm_rep2`:**

  | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|
  | 45.0 | 82.9 | 41.9 (30/39) | 57.3 | 59.1 | 53.8 | **56.7** |

  - Phase 3b code, three VLM runs: 57.3, 56.7, and the graph control at 55.6.
- **Co-located groups (GT centres within 2% of the camera spread; `scripts/analysis/colocated.py`):**
  - On PTZ, almost every group is fully registered and lands on one point (median spread 0.005).
  - Each Pi3X-loop run splits 0–3 groups per scene: one member sits far from its partners (0.25–2.2 camera spreads). Scripted COLMAP splits none; v1.3 splits one.
  - **The split member is always one of the run's large errors (13–43°), and it enters through a weak path:**
    - "disagree; render confirms PnP" (s01 118 at 25°, s02 144 at 24°);
    - the hard path (s02 168 at 43°, BLH 12 at 13°);
    - unverified VLM placements in `p123` (s02 13 and 156, s01 93);
    - and twice through "agree" (s01 108, s02 073).
- **Can Pi3X detect co-location at runtime? (`scripts/analysis/coloc_pi3x.py`, all saved Pi3X groups of `p3b_vlm_dev`):** a pair counts as co-located when Pi3X's centres are within τ × the group's camera spread.

  | τ | Precision | Recall |
  |---|---|---|
  | 0.02 | 0.96 | 0.09 |
  | 0.03 | 0.96 | 0.19 |
  | 0.05 | 0.92 | 0.34 |
  | 0.08 | 0.81 | 0.55 |

  - A08 is the weak scene (0.71 at τ 0.03).
  - For 4 of the 7 split pairs, Pi3X put the two cameras on one spot (gap 0.006–0.037), including in the round where the wrong pose was accepted.
- **Correctly placed co-located pairs in the final models are up to 0.06–0.08 of the scene depth apart; split pairs are 0.25 or more.**

### Step 19 (2026-11-05): Phase 4, sky masking and the co-location guard (commit `8285d3d`, cache v22)
- **Sky:** Pi3X's confidence already separates sky.
  - `conf ≥ 0.1` removes the sky and keeps all structure, including a flagpole; 0.3 and above cut walls.
  - Point alignment and depth-filtered PnP already used 0.1. Render-and-compare ignored confidence, so sky "points" were drawn into the figures and counted in the NCC score.
  - It now drops depth below 0.1 (`rac_min_conf`; `--ff-off sky` reverts).
  - Effect at the near-correct final poses of BLH 28 and 33: the sky streaks are gone, and the render score goes from −0.11 to +0.19 (inconsistent → positive).
  - Structure behind the unmodelled perimeter wall still shows through: no occluder, no fix.
- **Co-location guard (`ff_coloc`; `--ff-off coloc` reverts):** every Pi3X round, a batch image whose Pi3X centre is within 0.03 × the group's camera spread of a registered anchor is called co-located with it.
  - If the accepted pose's centre is more than 0.10 × the scene depth from the partner's, the guard tries fixed-centre poses: the partner's centre, with the rotation from Pi3X, from the chosen pose, or from the pair's homography.
  - The best of these is taken when its match support is ≥ the minimum and not clearly below the chosen pose's support. Otherwise the image is deferred.
  - Images that would be rejected get the same fixed-centre candidates.
  - Not yet in the hard path (BLH 12/22 were placed there).
- **Smoke run (PTZ s01, graph selection):** 58.9, 179/182, no crash. The guard detected 6 co-located pairs, all already within 0.015, so it never had to act. No group split.
- **Launched:** `p4_vlm_dev`, all six scenes on GPUs 4–6.

### Step 20 (2026-11-05, 10:20): Phase 4 VLM pass (`p4_vlm_dev`)
| Scene | `p4_vlm` | `p3b_vlm` | `p3b_vlm_rep2` | `p3b_graph` | v1.3 | scripted |
|---|---|---|---|---|---|---|
| A08 | 45.1 | 45.0 | 45.0 | 43.8 | 43.1 | 44.3 |
| M07 | 82.6 | 82.4 | 82.9 | 82.1 | 81.6 | 81.0 |
| BLH | 42.0 (31/39) | 44.5 | 41.9 | 40.7 | 50.7 | 49.0 |
| PTZ s00 | 57.4 | 56.3 | 57.3 | 59.0 | 56.1 | 61.6 |
| PTZ s01 | 58.9 | 59.9 | 59.1 | 58.0 | 55.3 | 61.5 |
| PTZ s02 | 56.8 | 55.4 | 53.8 | 50.4 | 52.3 | 60.9 |
| **mean** | **57.1** | 57.3 | 56.7 | 55.6 | 56.5 | 59.7 |

- **Co-located groups:** no split group on any scene; `p3b_vlm` had one per PTZ scene, `p3b_vlm_rep2` none.
- **The guard acted twice**, deferring 2 images whose pose was away from their Pi3X partner (118 and 127); both are later registered.
- **Cameras > 10°** (A08 / BLH / s00 / s01 / s02):

  | Run | Counts |
  |---|---|
  | `p4` | 3 / 1 / 5 / 2 / 3 |
  | `p3b_vlm` | 3 / 1 / 4 / 2 / 3 |
  | `p3b_rep2` | 3 / 0 / 4 / 2 / 3 |
  | `p3b_graph` | 4 / 1 / 3 / 4 / 9 |

- **No measurable change, positive or negative:** splits were already rare in the VLM runs, and one pass cannot separate a one-image effect from run noise.
  - The sky mask changes every render score. Since the mean did not drop, it does not hurt.
  - Kept both: they are correct by construction.
- **BLH is the scene that keeps losing** (31/39). The back-of-building cameras are judged by renders made from front-side images (see step 21).

### Step 21 (2026-11-05): why the BLH renders compare the photo to the front of the buildings
- **How render-and-compare picks its sources:** it warps the 4 registered images *nearest the candidate pose* (camera distance + 0.5 × (1 − cos of the viewing-direction angle); `grow.py` `nearest_registered`, `render_compare.py` `neighbours`), whatever they see.
- **What that gives for the back cameras (28, 33):**
  - The nearest registered images are inside the compound, looking at the inner faces of the brick buildings.
  - Unverified neighbours (13, 33) have no sparse points, so their depth cannot be scaled to the model and they are skipped.
  - With no visibility or back-face test, those inner faces are drawn through the unmodelled perimeter wall.
  - The figure compares the photo's wall with the far side of the buildings, so even the correct pose looks "inconsistent".
- **Proposed fixes:**
  1. Choose render sources by shared matches with the candidate, not by pose distance.
  2. Drop back-facing points, using normals from the Pi3X depth.
  3. Report "no evidence" instead of "inconsistent" when the sources share little content with the photo.

### Step 22 (2026-11-05): render fixes (commit `65cdcc0`, cache v23)
- **Sources** (`render_sources`): registered images sharing at least 15 verified matches with the candidate come first, strongest first, so every candidate pose of an image is rendered from the same images. Nearby cameras fill up to 4. A source whose viewing direction is 90° or more from the candidate's is never used. `--ff-off rsrc` reverts.
- **Back-face culling:** surface normals come from each source's Pi3X depth, in the camera frame and cached. Points whose surface faces away from the target camera are dropped from the score and the figure. Depth edges (more than 20% jump) get no normal and are kept. `--ff-off backface` reverts.
- **Coverage:** the fraction of the target's comparison grid that receives a rendered point. Below 0.10 the verdict is "no evidence", not "inconsistent". `--ff-off cover` reverts.
- **Offline GT check** (`scripts/analysis/rac_check.py`: correct cameras < 3° judged at their pose and rotated 12°):
  - Verdicts on correct poses are unchanged on BLH and PTZ s01. A08 improves, with wrongly inconsistent going 5 → 4.
  - Rotated poses are caught as before.
  - The median coverage of a correct pose is 0.33–0.51.
- **BLH back cameras 28, 33, 13 at their correct poses:**
  - Before: "inconsistent", rendered from 4 front-side images (coverage 0.12–0.14, nearly all of it the far side of the buildings).
  - After: one source faces the same way, coverage 0.02–0.08, verdict "no evidence". The figure shows only the sliver of real overlap.
- **Launched:** `p5_vlm_dev` (GPU 6: BLH, s02, A08; GPU 5: s00, s01, M07).

### Step 23 (2026-11-05): reproducing the 68.5 BLH run (exact code `3facb08`, cache v20; git worktree `../agenticsfm_p123`; 5 reruns made distinct with `--support-pairs 995–999`)

| Run | auc@10 | Registered | VLM placement | Cameras > 10° |
|---|---|---|---|---|
| `p123_vlm_dev` (original) | 68.5 | 39/39 | group {33, 28, 13, 32}, 6 picks, layout plausible 0.7 → 4 unverified | none |
| rep998 | 63.5 | 39/39 | the same group and picks → the same 4 unverified | 12 (12°) |
| rep996 | 66.6 | 39/39 | group {2, 32, 38, 36, 28, 33}, 3 picks, plausible 0.7 → 6 unverified | 13 (16°) |
| rep997 | 44.3 | 34/39 | group {13} only → 1 unverified | 32 (22°) |
| rep995 | 42.5 | 34/39 | 5 images, 3 picks, layout **implausible** 0.6 | 36 (21°) |
| rep999 | 42.0 | 32/39 | 7 images, **1 pick**, layout **implausible** 0.6 | none |

- **68.5 reproduces about 2 runs in 5.** The outcome is decided by whether the VLM placement step admits the back group.
- **Every unverified placement admitted on BLH in any run is correct:** 15 placements at 1.1–3.6°.
- **The failures come from the VLM layout check, plus one hard-path admission at 21–22°:**
  - The check rejects correct layouts with confidence 0.6, giving reasons about the wall standing between camera and yard.
  - On s02 it accepted 4 wrong placements at 0.7.
  - **So the VLM layout check is not a usable gate in either direction.**
- **Next:** test the geometric alternative from step 16 on all of these groups: halves of all anchors, rotation gap only (`scripts/analysis/ff_stability.py --halves anchors`).

### Step 24 (2026-11-05): which gate can admit VLM placements? (offline, GT only for scoring)
Test set: every leftover-group placement with known GT, 19 images.
- **BLH, correct:** 11 images (1–4°), from `p123_vlm_dev` and rep996.
- **PTZ, wrong:** 8 images, from `p123_vlm_dev` (s02: 13, 139, 145, 156; s01: 85, 93, 145, 174) at 9–41°.
- **One borderline image:** s02 155 at 5°.

| Gate | Correct BLH admitted | Wrong PTZ admitted |
|---|---|---|
| VLM layout check ("plausible" ≥ 0.6) | 11 in the runs where it said plausible; it said implausible in 2 of 5 BLH reruns | 8/8 (all "plausible 0.7") |
| Stability, halves of the VLM picks (Phase 3b, 5° and 5%) | 0 (centre test) | 0 |
| Stability, halves of all anchors, rotation gap < 10° | 4/11 (rep996: all 6 at 18–24°, because one half's anchors pull the group off) | 4/8 (s02 139, 156, 13 and s01 93 are consistently wrong: gap 4.5–7.4°) |
| New render (step 22), veto only "inconsistent" | 10/11 (all "no evidence"; 32 vetoed, whose centre is 18% off along its view) | 3/8 (s02 13, 145 and s01 93 are "no evidence") |

- **None separates cleanly. The best is the new render as a veto: about 77% precision on admissions (10/13).**
- Wrong placements that are consistently wrong (Pi3X makes the same mistake with any anchors) defeat any self-consistency test.
- Views of unmodelled surfaces defeat any render test: there is nothing to compare against.
- **Decision for the user:** the policy for unverified VLM placements.
  - (a) Keep Phase 3b: admits nothing, 100% precision; BLH stays at 42–45.
  - (b) Render veto instead of stability and layout: BLH back group in most runs, but about 1 in 4 admissions wrong on PTZ, flagged unverified.
  - (c) (b) restricted to groups where Pi3X aligns on points and the render says "no evidence" for the whole group.

**User's choice: (c).**

### Step 25 (2026-11-05): p5 pass (render fixes) and Phase 3c, render veto only
- **p5_vlm_dev** (commit 65cdcc0, still Phase 3b): A08 43.6, M07 82.8, s00 55.6, s01 58.8, s02 55.0. All within run noise of p4 (45.1, 82.6, 57.4, 58.9, 56.8). No co-located group was split (`colocated.py`).
  - BLH crashed: when no registered image faced the candidate, `render_sources` returned an empty list. `RenderCompare.judge` then fell back to the neighbours of the (unregistered) candidate itself ("Frame does not have a valid pose").
  - Fix: an explicit empty source list now gives "no evidence".
- **Phase 3c, first version** (eb9366f):
  - Every pending (not match-confirmed) image of a VLM-placed group is judged by the fixed render.
  - If at least half the pending members are "inconsistent", the whole group is left out (Pi3X places the group jointly).
  - Otherwise every member that isn't "inconsistent" registers unverified.
  - The layout check is advisory only. `--ff-off rgate` restores 3b.
- **Results:**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p6_vlm_dev | 44.6 | 82.1 | 51.1 | 56.9 | 57.2 | 59.4 | **58.6** |
  | p6_vlm_rep2 | 42.8 | 81.5 | 59.9 | 58.2 | 59.0 | 56.1 | **59.6** |

  - BLH repeats: 43.8 (2 models) and 65.1.
- **GT check of every pending image** (`scripts/analysis/vlm_place_gt.py`; the Pi3X pose of each pending image is now logged as `ff_pose`, 1570227):
  - BLH admissions: all rotations 1–3°.
  - rep2 also admitted 5 wrong placements:
    - A08: a group of 3 at 11–19°, all "no evidence", layout "plausible";
    - s01 63: 61°, a single image, "no evidence";
    - s02 139: 11°, render "consistent".
  - The render-only rule cannot separate a BLH back group from A08's wrong group: both are all "no evidence".
- **BLH back cameras:** their rotations are right, but their centres sit 0.2–1.0 camera-spreads off along the viewing direction (Pi3X misjudges distance from a far side). The step-24 "correct" label was rotation only. They still raise auc@10 (relative rotations, and translation directions from far away).

### Step 26 (2026-11-05): Phase 3c with an alignment requirement
- **What separates the groups:** Pi3X's point alignment to the model on the group's anchors (`point_inl` = share of lifted anchor points within 2% of scene scale after the similarity). Over every vlm_place group since p123:
  - groups whose admitted placements are correct: 0.92–0.98 (residual ≤ 0.08), 9 groups, all BLH;
  - groups with wrong placements: ≤ 0.85, typically 0.4–0.7.
  - Two older PTZ groups (p3b runs) aligned at 0.95 and 0.99, but their poses were not logged. The offline rerun (`ff_stability.py`) aligns them only on rotations, so it is not conclusive.
- **Rule** (1bf88ce): unverified placements need `by == "points"` and `point_inl >= 0.9`. Then the step-25 render rule applies. `--ff-off ainl` drops the requirement. Cache v26.
- **Results:**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p7_vlm_dev | 43.9 | 82.7 | 41.7 | 59.9 | 57.6 | 56.7 | **57.1** |

  - BLH repeats: rep999 69.1 (38 registered) and rep998 39.9 (37, 2 models; the back group never became a leftover group, only image 39, match-confirmed).
- **GT check:**
  - **Admitted:** 6 images, all BLH rep999 at 2.4–3.6° rotation (5 of 6 within 0.5 camera-spread).
  - **Left out:** 17, all wrong or degraded:
    - PTZ and A08: 8–23°;
    - the p7 BLH group: aligned at 0.70, centres 1.2–1.5 spreads off, 3 of 8 above 5°.
  - **No wrong admission on any scene**, and the layout check (advisory) said "implausible" on the correct rep999 group.
- **Reading:**
  - 3c with alignment keeps 3b's precision and admits the BLH back group whenever Pi3X places it cleanly. In 4 BLH runs since step 25 the clean case happened twice (69.1, 65.1), against 0 of 4 for 3b.
  - The mean is within noise of the render-only version, which bought +1–2 points with wrong admissions. We keep the alignment requirement (correctness over coverage).
  - BLH still ranges 40–69, set upstream: whether the back group stays together as one leftover group and how cleanly Pi3X places it.
  - The 0.9 threshold sits in the gap 0.85–0.92 of about 25 groups. It is a general "the frame agrees with the model" test, but it was chosen on this data.

### Step 27 (2026-11-05): retries with other anchor sets, and second models handed to vlm_place (commit 4f2ded9, cache v27)
- **User's request:** stabilise BLH with (1) images of an unmerged second model going to vlm_place, the model merging when at least half its images are placed; (2) when Pi3X aligns below 0.9, retrying the joint placement with other anchor sets ("picks only", "most matched only", "without previously removed members"), keeping the best alignment under the same gate. `--ff-off extra`, `--ff-off retry`.
- **Results (p8):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p8_vlm_dev | 44.8 | 82.2 | 53.9 | 57.7 | 57.7 | 54.8 | **58.5** |

  - BLH repeats: 58.8, 48.2, 53.5 (four runs 48–59, against 40–69 before).
- **What fired:**
  - Retries never reached 0.9 anywhere (BLH rep998 0.76/0.81/0.73/0.77; PTZ groups 0.45–0.78, correctly left out).
  - Change 1 fired only on s00 and s02 (second models of 4 and 6 images); their alignments failed, so the models stayed separate. No BLH run produced a second model.
  - No wrong admission. BLH admissions had rotations < 5° but centres 0.03–1.6 spreads off.
  - The narrower BLH spread is the luck of route, not the changes.
- **Why the retries failed (GT, offline):**
  - Failed BLH alignments (p7 0.70, p8 0.81) all had hard-stage anchors (images placed by the hard fallback) with right rotations but centres 0.9–2.6 spreads off, e.g. 36 at 2.56, 12 at 0.94, 02 at 1.78.
  - Clean alignments (0.94–0.98, 5 groups): every anchor within 0.08 spreads, none from the hard stage.
  - The retry variants draw from the same registered set, so they keep the bad anchors.

### Step 28 (2026-11-05): reference photos and anchors from match-registered images only (commit 56a53c6, cache v28)
- **Rule:** vlm_place shows the VLM, and anchors Pi3X on, only images registered through matches (not hard-stage, held or unverified), if at least 3 of them share matches with the group; otherwise all registered images. `--ff-off trusted`.
- **Results (p9):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p9_vlm_dev | 42.1 | 82.9 | 52.6 | 59.3 | 56.0 | 55.6 | **58.1** |

  - BLH repeats: rep997 48.1, rep998 51.6, rep999 61.6. Eight BLH runs since step 27: 48–62, mean 53.6 (p8) and 53.5 (p9).
  - Other scenes within run noise of p8. A08's −2.7 has no vlm_place admission behind it (its only group aligned at 0.47 and was left out).
- **What fired:**
  - Every BLH group that got picks now aligns cleanly: 0.91 (dev), 0.967 (rep998), 0.931 (rep999); no retry was needed. Trusted anchors fixed the alignment failure.
  - Change 1 fired on BLH for the first time (dev): a 4-image second model, 3 placed, merged by vlm_place.
  - rep997: the VLM chose no reference photo for the 7-image back group ("none"), so nothing was placed (48.1).
- **GT check of admissions** (correct = rotation < 5° and centre < 0.1 spread): 9 admitted, all BLH, all rotations 1.7–3.1°. Centres:
  - 4 within 0.14 spreads (28, 33 twice, 13 in rep999);
  - 5 off by 0.44–1.09 (38 twice, 13 and 33 in rep998).
  - The render left out 2, 32 and 36 (centres 0.75–1.26 off) and found "no evidence" for all of the admitted.
- **Reading:**
  - With clean anchors and alignments ≥ 0.9, Pi3X still puts back-of-building cameras up to about one camera-spread off along the view. Alignment inliers say the frame agrees with the model; they don't bound the depth of cameras looking at an unmodelled side.
  - BLH no longer fails by a broken alignment. What remains:
    - whether the VLM finds reference photos for the back group (rep997);
    - Pi3X centre errors on the back side, which no render test can check ("no evidence");
    - the route before Phase 3 (the hard stage registering the back group gave the 63–69 runs).
  - The pass mean is within noise of p6–p8 (57–59.6). The run-to-run VLM noise (temperature 0.3, no seed) is the same size as the effects being measured.

### Step 29 (2026-11-05): VLM at temperature 0 (commit 864f944, cache v29)
- **User's request:** lower the temperature to make BLH stable. The advisor now runs at temperature 0 (was 0.3, unseeded) with a per-question seed (hash of the question and image names). `--ff-off vlmtemp` restores 0.3.
- **Results (p10):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p10_vlm_dev | 44.2 | 74.5 | 46.9 | 56.5 | 56.3 | 57.9 | **56.0** |

  - BLH repeats: 46.7, 33.5, 44.8. Three of four BLH runs took the same route (32/39 after the hard stage, the same 7-image back group, alignment 0.92–0.95).
  - Not fully deterministic: the same hard-stage question got different answers across runs, because Pi3X/GPU nondeterminism changes what the VLM is shown.
  - M07 74.5 had no VLM placement: it took a different growth order and every image came out about 0.5° worse. The p11 M07 repeats at temperature 0 gave 82.4 and 82.8, so it was a one-off.
- **Why BLH stayed low: the group veto.** The render was right about every member of the back group:
  - "inconsistent": 2, 32, 36, 38, at 1.0–1.6 spreads off;
  - "no evidence": 13, 28, 33, at 0.1–0.34 spreads.
  - At 4 of 7 inconsistent the whole group was left out. p9's 61.6 run had 3 of 7 and admitted the rest, so a one-image difference decided BLH.

### Step 30 (2026-11-05): re-place without the inconsistent members (commit 31e8c2e, cache v30)
- **Rule:** members the render calls inconsistent are dropped and Pi3X re-places the rest without them. The survivors must pass the ≥ 0.9 alignment and the render check of the new placement. `--ff-off replace` restores the veto.
- **Results (p11):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p11_vlm_dev | 44.2 | 81.6 | 39.6 | 57.2 | 59.0 | 55.1 | **56.1** |

  - BLH repeats: 60.7, 61.2, 56.1. M07 repeats: 82.4, 82.8.
- **GT check:**
  - The BLH re-placements admitted 13, 28 and 33 at 0.02–0.15 spreads in every repeat, plus 22 (0.11, 0.32) and 38 (0.75, 5.6°) once each.
  - **One wrong admission, on s00.** A 2-image group aligned at 0.956; 45 was inconsistent, and the lone survivor 172 re-placed with cover 0 and was admitted at 176° (a flip). The old veto would have left both out.
  - BLH dev (39.6) took another route: the hard stage registered 2, 32, 36 and 38 with "placements agree, render inconsistent", at 2.9–3.4 spreads off. The 5-image leftover group then aligned at only 0.88.
- **Hard-stage audit (GT, p6–p11, all scenes):** "placements agree, render inconsistent" covers 54 decisions. 47 of those images ended registered, and 37 of the 47 were off by ≥ 5° or ≥ 0.25 spreads (16 by ≥ 10° or ≥ 1 spread). That is the worst rate of any large bucket: plain "placements agree" was 18 of 24, "agree, render consistent" 4 of 11. Most cases were on BLH (36, 32 bad); s00/s01/M07 had 11 (5 bad).

### Step 31 (2026-11-05): hard stage drops "agree, render inconsistent"; re-placement needs two survivors (commit f91c5e6, cache v31)
- **Results (p12):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p12_vlm_dev | 43.9 | 82.0 | 55.1 | 57.5 | 59.0 | 55.7 | **58.9** |

  - BLH repeats: 59.1, 43.4, 59.1. rep997 and rep999 came out identical (same route, same GT errors).
- **GT check:** 20 admissions, all BLH.
  - Most had the right rotation (1–5°) but centres 0.55–1.14 spreads off.
  - rep998 admitted 28 and 33 at about 10 spreads (re-placement cover 0, "no evidence").
  - The AUC gain came from imprecise admissions.

### Step 32 (2026-11-05): re-placed survivors must stay near their joint placement (commit a578a62, cache v32)
- **Rule:** a re-placed survivor must stay within 0.2 camera spreads and 3° of its joint placement.
  - GT p11–p12: survivors within 0.14 spreads of GT moved ≤ 0.18 spreads / 0.9°; every survivor ≥ 0.55 off moved 0.24–9.7 spreads or 55°.
  - Replayed offline: it keeps 13 (11 within 0.25 of GT) and drops 16, all wrong.
  - The threshold was chosen in a narrow gap on this data.
- **Results (p13):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p13_vlm_dev | 43.2 | 82.0 | 38.6 | 57.8 | 54.7 | 57.2 | **55.6** |

  - BLH repeats: 51.5, 47.4, 50.0. BLH dev aligned at 0.877 (< 0.9) and left the whole back group out.
- **GT check:** 5 admissions (33, 28 twice; 13 once), all 0.09–0.15 spreads, rotations 2–4°. No imprecise admission. 33 images left out.
- **Reading (p9–p13):**

  | Pass | Mean | BLH mean of 4 | Admissions | Within 0.2 spreads |
  |---|---|---|---|---|
  | p9 (temperature 0.3) | 58.1 | 53.5 | 9 | 4 |
  | p10 (temperature 0) | 56.0 | 43.0 | 0 | 0 |
  | p11 (+ re-place) | 56.1 | 54.4 | 13 | 10 (one 176° flip on s00) |
  | p12 (+ hard fix) | 58.9 | 54.2 | 20 | 3 |
  | p13 (+ stability) | 55.6 | 46.9 | 5 | 5 |

  - Pi3X places BLH back images 2, 32, 36 and 38 (and side images 12, 34) about one camera-spread off in every run. Only 13, 28 and 33 come out precise.
  - BLH scores in the high 50s and 60s needed the imprecise placements. With precise admissions only, BLH is about 47–52.
  - Remaining run-to-run variance is no longer the VLM. It comes from Pi3X/GPU nondeterminism near thresholds (alignment 0.877 vs 0.9) and from the route through the hard stage.

### Step 33 (2026-11-06, offline): can the BLH back cameras be corrected? No
- **The back cameras are co-located clusters** (GT): 2/32/36/38 share one position, 13/28/33 another, 12/22 another, 7.7 camera spreads apart.
- **Match triangulation + BA** of the unverified poses (everything else fixed, Cauchy loss, 3 rounds):
  - 2/32/38 went from about 1.0 to 3.0 spreads (10°);
  - accurate admissions degraded (0.09 → 0.54);
  - 13/28/33 improved only where they started about 1 spread off.
- **Their matches to model points are lookalikes.** BLH's back looks like its front: PnP on them lands 7–10 spreads off, mostly flipped about 180°, per image and pooled per co-located cluster.
- **PnP on Pi3X-lifted points** of the other group members reproduces the Pi3X pose (circular). Using only the accurate cluster's points fails too: 2/32/36/38 share 22–70 matches with it.
- **3D–3D ICP** of Pi3X depth points to the model's points made every camera worse. Even with the cameras at their GT poses, only 0.6% of 2/32/36/38's Pi3X points lie within 0.25 spreads of a model point (0.0% within 0.05). The model has no reconstruction of what they see.
- **Conclusion:** no data in the scene can correct 2/32/36/38. Pi3X (about 1 spread off) is the only source.

### Step 34 (2026-11-06): admit moved survivors with their joint Pi3X pose, unverified (commit 5b395b8, cache v33)
- **User's choice:** admit them with the Pi3X pose, flagged unverified, no BA.
  - Survivors that move under re-placement keep the joint pose. GT p11–p13 (22 moved survivors): joint mean 0.87 spreads off vs 1.81 re-placed; > 2 spreads off: joint 1 vs re-placed 4.
  - All other gates are unchanged. `--ff-off moved` restores accurate-only.
- **Results (p14):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p14_vlm_dev | 44.7 | 82.7 | 46.2 | 57.7 | 55.5 | 57.5 | **57.4** |

  - BLH repeats: 56.3, 50.9, 58.7 (mean of 4: 53.0; p12 54.2, p13 46.9).
- **GT check:** 20 admissions, all BLH; none elsewhere (no PTZ or A08 group passed the alignment).
  - All rotations ≤ 6.2°. Centres: 13/28/33/22 at 0.02–0.31 spreads; 2/32/38 at 0.92–1.36.
  - In BLH dev, 2/32/38 were re-placed stably (they did not move) at 2.0–2.2 spreads off, so the stability test can't catch them.
- **Reading:** BLH now admits the back group in every run. BLH's remaining variance is how far off Pi3X puts the 2/32/36/38 cluster (0.9–2.2 spreads).

### Step 35 (2026-11-06, offline): why PTZ keeps clearly wrong cameras
- **GT, p14 grown models:** each PTZ scene keeps 4–6 cameras more than 10° off, most about 1 camera spread off too (s00 23/99/139, s01 87/90/118/128/135, s02 40/73).
- **They entered through normal ff rounds** with match support ≥ 0.5, while the model was small. Nothing looked at them again.
- **Against the final model their support is 0.06–0.38;** correct cameras are mostly ≥ 0.6. Support below 0.4 flags 13 wrong and 14 correct PTZ cameras, so a flag alone must not drop a camera.
- **A fresh PnP against points no flagged camera made** (scratchpad `fix_proto.py`) fixes most of them: s00 23 33° → 1.9° (support 0.15 → 0.57), 99 19° → 0.5°, s02 40 104° → 6°, 73 24° → 6°. Correct flagged cameras stay correct.
- **Not fixable this way:**
  - s01's co-located cluster 87/90/118/128 (their points come mostly from each other; rotations 23° → 4–7°, centres stay 0.2–0.6 spreads off);
  - cameras with high support that are still wrong (s00 160 at 164°, support 0.90; A08 147/137/97 at 11–16° in every run).

### Step 36 (2026-11-06): examine after every round (commit de35e97, cache v34 for p15)
- **After each round that adds cameras, and after the final refinement and the sweep:**
  - Every match-verified camera's support is re-checked. Hard-stage, held and unverified cameras are skipped: matches could not place them.
  - Below 0.4: a fresh PnP from points no flagged camera made. The camera moves when the new pose's support, against unflagged cameras, is ≥ 0.45 and ≥ 0.1 above the old pose's.
  - With no better pose and support below 0.2, the camera leaves the model and returns to the pool. Each camera is examined at most twice.
  - `--ff-off examine` reverts.
- **p15 (examine only):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p15_vlm_dev | 44.8 | 82.8 | 45.2 | 58.9 | 57.5 | 56.9 | **57.7** |

  - BLH repeats 49.0, 58.2, 59.1 (mean of 4: 52.9).
  - GT: PTZ cameras > 10° 5/6/4 → 2/2/3. s01 135 was moved 24° and ended 6.4° off; s02 73 was moved and ended 5.8°.
  - Early removals of correct cameras (M07 round 4, s00 44) came back later.

### Step 37 (2026-11-06, offline): render-and-compare v2 and what the VLM can judge
- **Calibration harness** (scratchpad `eval_rac2.py`, p14 models, 25 GT-correct cameras per scene): each camera scored at its true pose, a coarse pose (3° and 3% of span), 10° and 20° rotations, and 15% and 30% shifts.
- **v1** (forward splat of 4 sources, one global NCC, margin over 5°/5% perturbations):
  - calls 50–60% of coarse-but-correct poses inconsistent;
  - passes 14% of 10° errors ("consistent");
  - AUC vs 10° 0.61–0.87.
- **v2** (in `render_compare.py`):
  - Inverse warping through the candidate's own Pi3X depth: every photo pixel is lifted, placed by the pose and looked up in the sources where their own depth agrees. The render is dense and occlusion-aware.
  - Pi3X depth scale is not shared within a group (PTZ s00: 0.5–2× the group median), so the candidate's scale is searched (0.5–2×, best coverage).
  - Score: search the shift of the render onto the photo (masked NCC of gradient magnitude). The shift says how far off the pose is; the NCC after it says whether it is the same place.
  - AUC of the shift vs 10°/20° errors: 0.88–0.97 (M07 0.97, A08 0.88, PTZ 0.89–0.96, BLH 0.92).
  - Verdict "consistent" = shift ≤ 8% and NCC ≥ 0.4. It passes 2 of 90 10° errors and none of 90 shifted poses; about half of true poses pass, the rest are mostly "no evidence".
  - A forward fallback when the inverse render is empty made correct PTZ poses "inconsistent", so it was removed.
- **Photometric rotation refinement** (correct yaw/pitch from the shift, roll by search):
  - fixes 10° on M07 (to 0.9°) and 6° on BLH (to 0.6°);
  - does nothing on PTZ, and its acceptance rule was unsound;
  - not used.
- **VLM on render figures** (`vlm_eval.py`, 150 poses × 2 figure styles):
  - **Yes/no "aligned?":** "aligned" for 29/30 true but also 25/30 10° errors on the old figure, and near chance on the checkerboard. Never acted on.
  - **Pairwise, true vs wrong:** on the checkerboard right 22, wrong 3, no pick 5 (vs 10°); old figure 20 / 8. The numeric v2 check is never wrong when it decides, but is undecided on 40–50% of pairs. On exactly those, the checkerboard VLM picks right 15 of 17.
- **Policy:** numeric verdict first; the VLM only as a pairwise tie-breaker, on checkerboard figures.

### Step 38 (2026-11-06): v2 in the loop, pairwise VLM acting (commit de35e97, cache v35)
- **Ff-round arbitration and the hard stage use v2.** The `vlm_place` gate keeps v1, because its thresholds are calibrated on it.
- **Hard stage:**
  - when both placements are consistent but disagree, the smaller shift wins;
  - a single placement is accepted only on the render verdict (the VLM's yes/no is logged);
  - two disagreeing placements without a numeric decision go to the pairwise VLM (acts with `--vlm decide`).
- **Examine:** a suspect with only a slightly better fresh pose goes to the pairwise question (numeric verdicts first). No VLM before a removal.
- `--ff-off racv2` / `exvlm` revert.
- **p16 (all on, `--vlm decide`):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p14_vlm_dev | 44.7 | 82.7 | 46.2 | 57.7 | 55.5 | 57.5 | 57.4 |
  | p15_vlm_dev (examine) | 44.8 | 82.8 | 45.2 | 58.9 | 57.5 | 56.9 | 57.7 |
  | p16_vlm_dev (all) | 44.3 | 82.4 | 58.9 | 58.9 | 58.1 | 57.0 | **59.9** |

  - BLH repeats 56.4, 47.5, 50.5 (mean of 4: 53.3; the 58.9 dev run is partly luck). PTZ mean 56.9 (p14) → 57.8 (p15) → 58.0.
- **GT (p16):**
  - PTZ cameras > 10°: 3/1/1 (p14 5/6/4). BLH 39 in model, none > 10°.
  - Every examine move ended within 6.5°, most within 2°.
  - Hard-stage VLM acts: s00 44 (1.3°), s00 29 (4.4°), s01 91 (0.6°), s01 159 (later filtered).
  - s02 102, accepted wrongly at 69° in p15 by v1 ("render picks B"), entered on a v2 "consistent" render at 5.4°.
  - Every v2 "render confirms" arbitration ended within 5.5° (p14: two at 24° and 104°).
- **Still wrong:** high-support cameras examine cannot see (A08 147/137/97, s00 160). The BLH back cameras still enter the hard stage as "placements agree, render no evidence" at 0.3–3 spreads off (existing behaviour, not v2).
- **Repeat (`p16_vlm_rep`, same code, `--support-pairs 996`):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p16_vlm_rep | 45.0 | 81.7 | 54.3 | 59.4 | 56.8 | 58.2 | **59.2** |

  - PTZ mean 58.1 (p16 58.0, p14 56.9). PTZ cameras > 10°: 3/3/2, mostly 10–14°.
  - **Hard-stage VLM picks over both runs:** 8 acts; 2 ended > 10° off (BLH 13 at 12°, s01 174 at 13.9°). In both, both renders were "no evidence": the VLM chose between two near-empty figures. Picks on figures with content all ended within 5°.
- **Fix (cache v36):** the hard stage asks the pairwise VLM only when at least one render has evidence. Confirmation pass `p17_vlm_dev` with BLH repeats launched.
- **p17 (the evidence rule on):**

  | Run | A08 | M07 | BLH | s00 | s01 | s02 | Mean |
  |---|---|---|---|---|---|---|---|
  | p17_vlm_dev | 43.7 | 82.8 | 48.7 | 57.0 | 59.0 | 57.6 | **58.1** |

  - BLH repeats 50.5, 50.2, 48.9 (mean of 4: 49.6, against 53.5 over 5 p16 runs). PTZ mean 57.9 (p16 58.0/58.1).
  - GT: PTZ cameras > 10° 2/1/1 (the fewest so far). BLH has 35 cameras: the back cluster 2/32/36/38 was left out at `vlm_place` ("render inconsistent") in both runs checked, and 13 ended 45° and 17° off.
  - The rule changes BLH's route more than its precision.
- **Decision (user asked for p16 on all sequences):** the rule is now `hard_vlm_needs_evidence`, off by default (p16 behaviour, cache v35, commit 871c817).
- **Full runs launched 2026-11-06 17:17 on p16 code:** `p16_wriva_test` (28 WRIVA test sequences) then `p16_eth3d` (25 scenes), one process per GPU (4/5/6), balanced by image count.
  - `run.py --workers N` assigns scene k to GPU k mod N regardless of which GPU is free, which can put two Pi3X jobs on one GPU; hence one process per GPU.

### Step 39 (2026-11-07): full runs on p16 code, and re-aligning folded blocks (cache v38)

- **Full p16 results:**
  - WRIVA test (28 sequences), against COLMAP `fixed_wriva_v12`: auc@10 67.2 vs 70.6, auc_all@10 60.0 vs 65.1, auc@3 42.2 vs 42.6.
  - ETH3D (25 scenes): colmap_auc@1/3/5 64.8/84.9/89.6, against incremental_v13 65.7/85.6/90.2 and scripted 68.5/88.1/92.7.
- **Main failure mode (GT): a two-zone fold.** A self-consistent block of cameras sits turned about 180° about the vertical relative to the rest. It forms early (MultiZone's second zone attached at rounds 4–9).
  - Scenes: A01 s00/s01, MultiZone s00, TwoZones, ETH3D old_computer (26 of 54 cameras at 180°).
  - TwoZones has almost no genuine cross matches; COLMAP splits it.
- **Tool: `realign_blocks`**, run after growth.
  1. **Seam detection:** spectral bisection of the explained-match graph (per pair, the share of verified matches the poses explain). A cut whose crossing matches are explained below 0.35 is a seam.
  2. **Per-camera placement:** each block camera is placed by PnP on points that only outside cameras observe.
  3. **One similarity for the whole block:** hypotheses come from single cameras and from pairs (for scale), scored by the pooled outside-point inliers of all block cameras, then refined on the cameras that agree.
  4. **The move:** block-only points move with the block, mixed points are dropped, then re-triangulation and global BA.
- **Acting rule:**
  - The turn is at least 2°.
  - Inliers at least 2× + 30.
  - At most 20% of cameras lose inliers.
  - At least 40% of cameras gain (or 75% of their own PnP poses agree).
  - **A turn over 10° must explain at least 6% of the block's outside correspondences.**
    - A real fold, once corrected, explains a large share of the matches to the rest of the model. A lookalike (doppelganger) flip explains only the repeated structure.
    - GT on p16 and p18 models: correct flips 0.08–0.16; wrong ones (ETH3D facade at 180°, courtyard at 90°, two PTZ blocks) 0.016–0.029.
- **The VLM check was tried and is off (`realign_vlm`).**
  - **p18:** pairwise renders from outside-block cameras at the current vs the moved pose.
  - **Correct move refused.** On MultiZone the correct 179° move of 115 cameras was refused: the moved-pose render was empty even with GT-correct pose and sources. The block shares only far background (60–90 m away) with outside cameras, and their Pi3X depth drops it.
  - **Wrong move approved.** The VLM then approved a wrong 22.6° move of 19 cameras (GT 67.7° off afterwards; auc@10 41.2 → 34.9).
  - **Renders don't settle lookalikes.** Offline, on facade the wrong 180° pose renders "consistent"; on old_computer both poses do.
- **Offline (p16 models, the v38 rule, BA after each move, against a control with BA only):**
  - Over all 59 models (28 WRIVA test, 6 dev, 25 ETH3D), the rule acts on 6:

    | Model | Control | Realigned | Score |
    |---|---|---|---|
    | MultiZone s00 | 41.1 | 82.0 | auc@10 |
    | old_computer | 35.2 | 48.0 | colmap_auc@3 |
    | TrailerPark s00 | 62.9 | 73.3 | auc@10 |
    | A01 s00 | 27.8 | 28.2 | auc@10 |
    | TwoZones | 37.2 | 37.4 | auc@10 |
    | A01 s01 | 19.7 | 19.7 | auc@10 |

  - It refuses facade (0.023), courtyard (0.029) and PTZ s01's 145° block (0.016).
- **Live (p19, 12 scenes):**
  - old_computer: one 13-camera block moved 180°. Cameras > 90° off went from 26 to 13; colmap_auc@3 41.0 → 52.5.
  - MultiZone: the fold did not form this run (78.0).
  - TrailerPark s00 grew a new 18-camera fold that the rule could not fit (58.0).
  - No other moves; dev unchanged within run noise.
  - Fold formation varies between runs, so the offline test on fixed models is the cleaner measure.
- **Open:**
  - A01: PnP agrees with the fold, and focal is off by up to 29% in the folded half.
  - Unmerged sub-models (TrailerPark, SmartCampus).

### Step 40 (2026-11-07): full runs on v38 (p20)

- **Whole sets (p20 = v38 with realign on, against p16 on the same sets):**

  | Set | Score | p20 | p16 | Reference |
  |---|---|---|---|---|
  | WRIVA test (28) | auc@10 | 67.9 | 67.2 | COLMAP 70.6 |
  | WRIVA test (28) | auc_all@10 | 59.7 | 60.0 | COLMAP 65.1 |
  | WRIVA test (28) | auc@3 | 42.9 | 42.2 | COLMAP 42.6 |
  | ETH3D (25) | colmap_auc@1 | 64.3 | 64.8 | incremental_v13 65.7, scripted 68.5 |
  | ETH3D (25) | colmap_auc@3 | 85.0 | 84.9 | incremental_v13 85.6, scripted 88.1 |
  | ETH3D (25) | colmap_auc@5 | 89.9 | 89.6 | incremental_v13 90.2, scripted 92.7 |

- **Realign moved blocks in two scenes live:**
  - old_computer: a 13-camera block turned 179.8° (explained share 0.159); colmap_auc@3 41.0 → 52.5. The second 13-camera half was refused again (worse 0.23 > 0.2), as in p19.
  - TwoZones: 94 cameras by 3.8°; flat (37.8 vs 37.9).
- **The rest of the scene-level change is run-to-run growth variance, not realign.** With no move, realign does not touch the model (`_rac_exclude` stays empty, no BA).
  - MultiZone s00 41.2 → 78.0: the fold did not form this run (as in p19).
  - Drops in scenes with no realign row (CameraModels A01_new s00 85.0 → 65.0, A06 78.1 → 71.4, botanical_garden 89.7 → 83.3, lecture_room, meadow) are variance.
  - Without MultiZone s00, WRIVA auc@10 is 67.5 vs 68.2.
- **GT check on the refused large turns (offline only):**
  - TrailerPark s00, 15 cameras, proposed 151.3°: a real fold (GT error in the final model: median 146.6°, 15/15 > 90°). Refused: outside inliers 25 → 44, short of 2·old + 30.
  - TrailerPark s01, 19 cameras, proposed 146.2°: a real fold (median 142.9°, 19/19 > 90°). Refused: 17 → 37. Explained share 0.041.
  - TrailerPark s03, 39 cameras, proposed 177.8°: the block was right (median 1.2°). Correctly refused by the explained-share guard (0.041).
  - The explained share does not separate these PTZ cases: a true fold and a wrong flip both score 0.041. Small PTZ blocks have too few outside correspondences for the pooled-inlier evidence to decide.
- **Open:**
  - Small folded PTZ blocks (TrailerPark s00/s01: 18 and 38 cameras > 90° off) need evidence beyond pooled outside inliers.
  - old_computer's second half (worse 0.23).
  - A01 fold and focal error; unmerged sub-models (TrailerPark, SmartCampus).

### Step 41 (2026-11-08): VLM tools for folds; landmark anchoring (cache v39, v40)

- **Question:** COLMAP rarely forms the folds we lose on. Can a better VLM tool tell a true fold from a lookalike flip, given only the photos?
- **Offline case set (GT used offline only):**
  - 61 proposed block moves were harvested from the p16–p20 final models (`fold_cases.py`, scratchpad).
  - Each move is labelled by its GT error, with the GT similarity fitted on the cameras outside the block: move / stay / unclear.
  - 29 cases turn ≥ 10°: 9 true folds, 20 wrong moves.
  - The geometric statistics (explained share, inliers, improved, agree) overlap completely between the two classes.
- **Tools tried on the 29 large-turn cases:**

  | Tool | What the VLM sees and answers | Folds found | Wrong moves approved |
  |---|---|---|---|
  | 1 | block photo with the outside photos of each pose, A/B "same place", both orders | 5/9 | 4/20 |
  | 1b | absolute "same place" per supporter group | answers "same" almost always | |
  | 2 | match lines of one pair: same object or lookalike | 4/9 | 9/20 |
  | 1c | contrastive A/B, differences listed before the answer (medium effort) | 3/9 | 2/20 |
  | 1c, xhigh | same | all "unsure" (token cap) | |
  | **3, anchoring** | six distinctive details of the photo alone, then a count of the details found in each pose's photos (separate calls) | **3/9 (pro/con rule), 5/9 (sum ≥ 2)** | **0/20** |

  - Tool 1c's two orders agree on only 49 of 110 cameras: A/B judgements are position-biased.
  - Anchoring avoids A/B entirely. When it commits on a camera it is right 44 times and wrong 9.
- **Ceiling check (me, blinded sheets):**
  - Distinctive scenes (MultiZone 64°, courtyard, old_computer, facade): 12/12 cameras right.
  - TrailerPark trailers: 2/6, chance.
  - The PTZ trailer folds are not decidable from these photo groups, even for a stronger model. Distinctive structure is.
- **Rule (v39):**
  - A turn ≥ `realign_flip_deg` must pass anchoring instead of the 6% explained-share guard.
  - For up to 4 block cameras the score is Σ(details found at moved − at current). Supporters are outside photos whose verified matches the pose explains, with other suspect blocks excluded. The move needs score ≥ 2.
  - Offline, geometric gates plus anchoring: 5/9 folds found, 0/20 wrong (v38: 5/9, 2/20).
  - The two v38 errors are MultiZone's 64° block in p19 and p20. Anchoring vetoes them at −16 and −13.
- **Stability (same 29 cases, geometric gates plus anchoring):**

  | Run | Folds found | Wrong moves approved |
  |---|---|---|
  | medium effort | 5/9 | 0/20 |
  | xhigh effort | 4/9 | 1/20 |
  | temperature 0.7 | 3/9 | 0/20 |

  - The xhigh false move is the symmetric facade (+3; −3 to +1 elsewhere).
  - True MultiZone folds score only 1–4, so the margin is thin.
  - TrailerPark true folds score −4 to 0: confidently wrong, consistent with the ceiling check.
- **p21 live (v39), 9 scenes:**
  - old_computer: the anchor path moved the 13-camera 179.8° block (score 4; details such as the wall clock, glass doors and exit sign); colmap@3 52.5.
  - MultiZone s00 auc@10 34.8 (p20 78.0): growth formed a 179° fold this run.
    - Its 139-camera block was refused by the geometric support gate (improved 0.39 < 0.4) before anchoring ran.
    - GT (offline harvest): a true fold, median 179.7° → 0.7° if moved. Anchoring scores it +5 / +3 / +4 over three runs.
    - A 13.6° block beside it was anchor-vetoed live (−15). GT: wrong either way (89° → 76°).
  - Dev scenes flat against p16–p19; BLH auc@10 70.9.
- **v40:**
  - With anchoring as the veto, anchored turns need improved ≥ 0.3 (`realign_anchor_improved`) instead of 0.4.
  - Offline, the only cases newly admitted are two TrailerPark 143° lookalikes. Both are vetoed in all three anchoring runs (−4 to −1).
  - p22 = full WRIVA test, ETH3D and dev runs on v40.
- **Open:**
  - TrailerPark small PTZ folds: lookalike trailers; the tool ceiling is at chance with these photo groups.
  - A01: geometry never proposes the ~178° flip, because the outside matches support the fold.
  - The anchoring margin is thin (true folds score +1 to +7).

### Step 42 (2026-11-08): full runs on v40 (p22)

- **Whole sets:**

  | Set | Score | p22 (v40) | p20 (v38) | p16 | Reference |
  |---|---|---|---|---|---|
  | WRIVA test (28) | auc@10 | 67.9 | 67.9 | 67.2 | COLMAP 70.6 |
  | WRIVA test (28) | auc_all@10 | 60.6 | 59.7 | 60.0 | COLMAP 65.1 |
  | WRIVA test (28) | auc@3 | 42.4 | 42.9 | 42.2 | COLMAP 42.6 |
  | ETH3D (25) | colmap_auc@1 / @3 / @5 | 64.5 / 85.1 / 89.9 | 64.3 / 85.0 / 89.9 | 64.8 / 84.9 / 89.6 | scripted 68.5 / 88.1 / 92.7 |
  | vlm_dev (6) | auc@10 | 62.4 | p19 62.0 | 62.1 | COLMAP 62.0 |

  - Against COLMAP, auc@10: p22 wins 8 scenes, loses 11, ties 9 (±1).
  - Without the three A01 fold scenes (VaryingAltitudes s00/s01, TwoZones), p22 leads 72.6 vs 70.6.
- **Every live anchoring decision agrees with GT** (5 calls, GT from an offline harvest of the final models):

  | Scene | Block | Turn | Score | Decision | GT |
  |---|---|---|---|---|---|
  | MultiZone s00 | 99 cams | 179.6° | +13 | moved (passed the v40 support gate at improved 0.37) | true fold; auc@10 78.9 (p21 with the fold unrepaired: 34.8) |
  | old_computer | 13 cams | 179.6° | +9 | moved | true fold |
  | old_computer, second half | 13 cams | 180° | −3 | vetoed (now passes worse at 0.15) | correct: the block is right (GT 0.1°) |
  | courtyard | 8 cams | 90° | −8 | vetoed | correct (0.2° → 90.5° if moved) |
  | TrailerPark s02 | 47 cams | 133.8° | −3 | vetoed (admitted by the v40 gate at improved 0.30) | correct: part of a right block (1.2° → 142° if moved) |

- **Scene swings are still growth variance.**
  - Down: TrailerPark s02 78.0 → 62.4, MultiZone s01 76.9 → 72.4, SmartCampus s03.
  - Up: CameraModels A01_new s00 65.0 → 84.2, A06 71.4 → 78.4, TrailerPark s00/s01 +5/+6.
  - TrailerPark s02 grew an 18-camera PTZ fold (GT 145° → 5° if moved). Geometry refused it (improved 0.17, agree 0.42), so anchoring never ran.
- **Could anchoring alone judge weakly supported turns? Not yet.**
  - Offline, the TrailerPark s02 fold scores +4 / +4 / +5 over three runs. This is the first TrailerPark fold anchoring gets right.
  - The partial TrailerPark s01 fold (144° → 34°) scores −5 / −1 / −5.
  - Dropping the support gate would also admit the MultiZone 22.3° wrong move (improved 0.29), which scores +3 / +1 / +4.
  - No threshold separates the two, so the support gate stays at 0.3.
- **Open:**
  - The A01 fold and focal error account for the whole gap to COLMAP; geometry never proposes the flip.
  - Unmerged sub-models (auc_all@10 on PTZ scenes).
  - Small PTZ folds below the support gate.

### Step 43 (2026-11-23): VLM speed, and can the VLM review growth on a bird's-eye map?

User request: optimise VLM speed, use the VLM as scheduler with a bird's-eye (BEV) render after growth to avoid folds,
better tools for the VLM; try everything, keep only what works.

- **Where VLM time goes** (GPU-BA runs: WRIVA test, ETH3D test, WRIVA dev): `select_next` 988 calls x 24 s = 23,970 s,
  44 % of all growth time. A call is ~3.2k prompt tokens and ~1,460 thinking tokens; decoding is the time.
- **MTP speculative decoding** (`scripts/serve_vlm.sh`, the vLLM recipe's `{"method": "mtp", "num_speculative_tokens": 3}`;
  acceptance 73 %): 69 -> 132 tok/s, a thinking call 21.1 -> 11.2 s, same picks on 17/20 (replay noise: 54/59). **Kept.**
- **`select_next` does what match ranking does.** Thinking picks = the top by inliers on 78 % of 985 calls (99 % on dev
  replays); with thinking off a call takes 0.9 s but agrees with the thinking picks only 26/59 (noise, not judgement).
  GT outcome of the 559 top-ranked candidates the VLM skipped: 10.6 % ended misplaced vs 12.8 % of those it picked. The
  veto buys nothing measurable: **use `--select graph`** (match ranking, the CLI default).
- **BEV map** (`src/agenticsfm/solver/bev.py`): points as grey density, cameras as triangles along their viewing
  direction, highlighted / numbered sets, optional lines between cameras, alternative poses. Kept as a tool.
- **BEV review benchmark** (offline; GT from the p22 / g1 / g1b WRIVA runs, 2,071 growth rounds harvested with GT
  rotation and position errors; a camera is misplaced at > 20 deg or > 10 % of the scene radius). Cases: 103 fold
  onsets (a round's cameras mostly misplaced while their nearest earlier cameras are mostly right) and 40-41 clean
  rounds (all < 5 deg and < 3 %). The map is the state after the round (earlier cameras and their points).

  | Variant (low effort unless noted) | Onsets flagged | False alarms |
  |---|---|---|
  | map + photos of the new and nearest earlier cameras | 29 % | 10 % |
  | + lines of strong match pairs (red: contradict the placement, green: agree) | 52 % | 28 % |
  | map + photos, medium effort | 27 % | 15 % |
  | geometry only, no VLM: >= 50 % of the round's strong pairs contradict | **39 %** | **7 %** |
  | geometry only: >= 25 % contradict | 77 % | 28 % |
  | two maps (actual vs GT placement for onsets, actual vs block turned 180 deg for clean rounds), which is right? | right 29 % (onsets), 47 % (clean) | |

  - Rounds that extend an already misplaced block look locally consistent and are flagged 35 % of the time; the onset
    is the only point a review can act on.
  - The VLM cannot tell from the map and photos whether a block is in the right place; offered the GT placement it
    prefers the current one (68/103). None of the VLM variants is kept, and the scheduler built on such reviews is not
    built. The geometric contradiction share is the best signal found, but it was measured on final poses: it needs a
    test at round time before it can trigger anything.
- Scripts (session scratchpad, `vlm/`): `replay_select.py`, `harvest_rounds.py`, `bev_bench.py`, `bev_bench2.py`.
- **Confirmation run `g3`** (dev, GPU BA + `--select graph` + MTP; one run, vs the GPU-BA runs with VLM selection):

  | dev | VLM selection (g1 / g1b) | match ranking + MTP (g3) |
  |---|---|---|
  | WRIVA auc@3 / auc@10 / auc_all@10 | 26.1 / 64.5 / 60.1 | 25.1 / 62.8 / 60.9 |
  | WRIVA time | 4,974 s | 2,798 s (-44 %) |
  | ETH3D @1 / @3 / @5 | 78.6 / 91.4 / 94.3 | 79.0 / 91.6 / 94.4 |
  | ETH3D time | 2,271 s | 1,609 s (-29 %) |

  Per scene within the known run-to-run spread (PTZ s00 -2.2 auc@10; BLH -8.9 auc@10 but +6.0 auc_all@10). Runs use
  `--select graph` from now on; the VLM stays in its other roles (render choices, anchoring veto), now ~2x faster.
- **Two-site question** ("which of two places do the new photos show?": earlier photos around the actual placement
  vs around the GT placement for onsets, vs a look-alike decoy for clean rounds; 41 onsets with a reconstructed true
  place (62 of 103 onsets have none yet), 38 clean rounds): Qwen low 41 % / 26 % right, medium 49 % / 32 %. A blind
  check with Claude as the stronger model on 20 of these cases: 10/20 (Qwen 9/20 on the same cases). Both sites usually
  show the same objects from different positions; appearance does not say where the camera stood. Not kept.
- **VLM off vs on** (`g4` vs `g3`, dev, both GPU BA + `--select graph`; one run each): WRIVA auc@3 / auc@10 /
  auc_all@10 25.3 / 63.3 / 59.5 (off) vs 25.1 / 62.8 / 60.9 (on), 2,491 s vs 2,798 s; ETH3D identical on every scene
  (79.0 / 91.6 / 94.4), 1,520 s vs 1,609 s. The VLM made 45 calls on WRIVA dev (36 render checks) and 13 on ETH3D (12
  anchoring checks, no effect). The only scene it changed is BLH: 38 vs 35 registered (6 vs 2 hard placements),
  auc_all@10 63.3 vs 55.4 but auc@10 67.6 vs 69.9. Dev has few folds; the anchoring veto acted on test scenes (p22,
  step 42), so its value needs a test-split pair (on / off).

### Step 44 (2026-11-24): VLM on vs off on the test splits

`g5on_*_test` vs `g5off_*_test` (`outputs/logs/gpuba/test_g5.sh`): WRIVA test (28) + ETH3D test (12), GPU BA in growth +
`optimize_gpu`, `--select graph`, MTP server; each scene ran on then off on the same GPU. One run each.

| test | VLM on | VLM off |
|---|---|---|
| WRIVA auc@3 / auc@10 / auc_all@3 / auc_all@10 | 44.2 / 68.9 / 39.4 / 61.9 | 44.2 / 68.8 / 39.0 / 61.3 |
| WRIVA time | 25,019 s (VLM 1,237 s) | 23,309 s |
| ETH3D @1 / @3 / @5 | 75.4 / 87.8 / 90.6 | 75.3 / 87.6 / 90.4 |
| ETH3D time | 1,877 s | 1,632 s |

vs `g1b` (VLM selection, no MTP): WRIVA 44.2 / 68.5 / 38.8 / 60.6 in 48,669 s, ETH3D 75.5 / 87.9 / 90.7 in 2,649 s; match
ranking + MTP keeps the accuracy at about half the WRIVA time.

Where the on and off grow logs first diverge (script `vlm/onoff_diverge.py`, session scratchpad):

| WRIVA scenes | n | off - on auc_all@10, mean | mean abs |
|---|---|---|---|
| identical runs | 5 | 0.0 | 0.0 |
| diverge after a VLM decision | 6 | -6.3 | 6.3 |
| diverge before any VLM call (run-to-run noise) | 17 | +1.2 | 2.8 |

- **Run-to-run noise.** 17 WRIVA scenes diverge in rounds 2-6 before any VLM call, the 4 SingleZone scenes with no VLM
  call at all: point counts differ by a few (10,643 vs 10,639 after round 2), then batch order and anchors differ.
  Per-scene spread up to 19.6 auc_all@10 (FullTrailerPark s01: 52.3 on, 71.9 off). ETH3D is identical on every scene
  without a VLM decision. The split means are inside this noise.
- **Fold repair is the VLM's one large effect.** A 180 deg block turn found by realign is applied only when the
  anchoring check confirms it; with the VLM off it is never applied ("no VLM to confirm the turn"). On test the turn
  was proposed on 3 scenes:
  - MultiZone s00 (87 cameras): confirmed and applied. auc_all@10 70.8 (on) vs 35.8 (off).
  - exhibition_hall (20 cameras) and old_computer (13): declined, so the same result as off.
- **VLM placement of hard images** (render checks, `vlm_place`): more images registered, less precise, auc_all
  about flat. CameraModels A01_new s00: 25 vs 22 registered, auc@10 68.3 vs 87.0, auc_all@10 67.0 vs 66.5. Same
  pattern as BLH on dev.
- Without MultiZone s00 the VLM-off run is ahead on every WRIVA mean (auc_all@10 62.2 vs 61.6), inside the noise.
- **Were the anchoring decisions right?** Offline GT check (script `vlm/turn_check.py`): for each proposed turn, are
  the block's cameras misplaced in the final grown model (robust similarity to GT)? It covers every VLM-gated turn in
  the GPU-BA and p22 runs, dev and test: 14 decisions.

  | | turn was right (fold) | turn was wrong (block already correct) |
  |---|---|---|
  | VLM confirmed, turn applied | 4 (MultiZone s00 x2 runs, old_computer x2 runs) | 0 |
  | VLM declined | 3 (old_computer in g5, MultiZone s00 in g1b x2 events) | 7 (exhibition_hall, facade, courtyard, FullTrailerPark x2, old_computer turned back x2) |

  - Every confirmation was right. The misses are folds left in place: old_computer auc@1 38.1 (declined, g5) vs 47.1
    (confirmed, g1b); MultiZone s00 auc_all@10 45.7 (g1b) vs 70.8 (g5on).
  - Geometry alone does not separate the two columns. The wrong turns on exhibition_hall, facade and old_computer (turned
    back) have agree 0.57-0.80 and 8-27x inlier gains, the same as the real folds (agree 0.46-0.85). These are
    near-symmetric structures where both orientations fit the matches. A geometric rule that applies the real folds
    would also apply these 4.
  - Without the VLM no turn is applied: 7 of 14 right. With it: 11 of 14, with no wrongly applied turn. This is the
    VLM's one measured contribution so far: one task, and few cases. The same fold was confirmed in one run and
    declined in another (anchor score 10 vs 1), so the check's recall is the part to improve.

### Step 45 (2026-11-24/25): repeatable runs; a benchmark of the turn decisions

**Repeatable GPU BA (rayba 0.3, `third_party/rayba` ac946c1; grow_model v41, optimize_gpu v4).**

- The same saved problem (MultiZone s00, 282 images, 533k observations), solved three times, gave three answers.
  They differ in the last bits (1e-14). Cause: `index_add_` on CUDA sums in whatever order its atomics land. These
  differences change which points pass a filter (10,643 vs 10,639 points after round 2 of SingleZone s01). After that
  the rounds pick different images.
- The fix: every scatter-add (37 sites) is a segment sum over an index sorted once per solve, then one add into
  distinct rows (`RayBA._add`, `Options.deterministic`, default on).
  - Bitwise repeatable on the dense, sparse and implicit paths and the default schedule.
  - Same solution as before to 1e-10.
  - Cost: dense solve +21 %, final schedule +7 %. Torch's global deterministic mode costs 2.1x.
  - New GPU test in rayba (the old code fails it).
- The end-to-end check (`outputs/logs/gpuba/repeat_r{1,2,3}.sh`): SingleZone s01 and FullTrailerPark s01, each run
  twice with independent caches, VLM off. The two copies run at the same time, one on GPU 4 and one on GPU 5.

**Repeatable growth (grow_model v42: Ceres local BA on one thread).**

- r1 (rayba 0.3 only), SingleZone s01: both copies match in every logged field through round 4. In round 5 the
  same 8 images get the same poses but 29,096 vs 29,111 points; from round 7 the batches differ. Final: 80.18 vs
  80.19 AUC@10, poses up to 7e-3 apart. FullTrailerPark ran out of GPU memory when the other user's job came back.
- Cause: COLMAP's local BA (Ceres). The same BA on the same model, run twice: with 8 threads the results differ by
  1e-13 (poses) and 1e-9 (points); with 1 thread they are bitwise equal. Ceres only goes multithreaded above
  50,000 residuals, which this model reaches around round 5. `GrowConfig.ba_threads = 1` sets the local BA to one
  thread (it was all cores).
- r2, SingleZone s01, grown model: the two copies (GPU 4 and GPU 5) match in all 25 log events and in every field.
  All 175 poses and all 105,338 points are bitwise equal. So Pi3X is repeatable too, at least across these two
  GPUs. Local BA took 182 s vs 217 s with all cores (the machine is shared, so this is noise, not a speed-up); the
  grow took 576 s vs 600 s.
- Not yet checked end to end: the final refinement (bitwise repeatable offline) and FullTrailerPark. Both ran out
  of GPU memory in r2 when the other user's job came back. r3 reruns them.

**Turn benchmark** (session scratchpad `vlm/harvest_turns.py`, `vlm/replay_turns.py`).

- Every VLM-gated block turn in every run since anchoring (p21, p22, g1, g1b, g3, g5), one per grow artifact.
- Label: is the block misplaced in the final grown model (GT, offline)? One 13.5 deg proposal on a block 89 deg off
  is left out. That leaves 16 decisions: 8 turns that were right, 8 that were wrong.
- Replays reuse the logged photos and details (stage 2 of the check).

| score = sum over cameras of (details found at the turned pose - at the current pose) | right turns (8) | wrong turns (8) |
|---|---|---|
| logged in the runs | 4, 1, 1, -1, 10, 4, 9, 13 | -3, -4, -1, -3, -4, -8, -3, -3 |
| replay, same question (greedy, medium effort) | 4, 1, 3, 1, 10, 5, 9, 13 | -5, -5, -1, -3, -1, -10, -2, -3 |
| xhigh effort | 2, 1, 2, 2, 7, 3, 2, 9 | -7, -5, 1, 0, -5, -9, -1, -2 |
| 5 sampled answers (T = 1), mean | 3.4, 1.6, -0.2, 2.0, 8.6, 3.6, 6.8, 14.6 | -5, -2, 0.4, -3.4, -2, -9, -1.6, -2.2 |
| block photo shown next to its details | | broken: the VLM marks details as found in the reference photo |

- None of the variants separates the two columns better than the question as it is. The threshold (moved - current
  >= 2) applies no wrong turn in any variant.
- The misses (MultiZone s00 in g1b, old_computer in g5) score -1 to 1 everywhere. At both poses the supporting
  photos show few of the listed details (0/0, 0/1 per camera). That is a problem of which photos are shown, not of
  the question. A replay moves scores by 1-3, so a case near the threshold flips between runs.
- Not changed. The next lever would be the photo choice (more cameras when the score is near 0). It can only be
  tested in full runs, and those are only meaningful once runs repeat.
