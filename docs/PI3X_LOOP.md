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

## Log
