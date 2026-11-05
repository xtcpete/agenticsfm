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
