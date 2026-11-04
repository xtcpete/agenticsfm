# Agentic SfM: an open incremental solver with a VLM in the loop

**Input:** a directory of unordered images. **Output:** camera poses, as one COLMAP model per scene found in the
images. No ground truth, no scene-specific setup. Scenes may contain doppelgangers (repeated or symmetric
structure) whose matches are wrong and self-consistent, mixed cameras, and images of a different place.

The solver (`grow_model`, [solver/grow.py](src/agenticsfm/solver/grow.py)) replaces COLMAP's incremental mapper
with a loop over an explicit state, built on pycolmap's geometry primitives and a feedforward model (Pi3X):

```
 seed:  Pi3X on the densest cluster of the view graph -> poses + focal lengths -> triangulate -> bundle adjust
 grow:  rounds of PnP registrations, each verified against the matches (epipolar support), local BA, global BA at +10%
 hard:  images PnP cannot place or keeps placing against lookalike matches: Pi3X places them twice, with two anchor
        sets (match neighbours + cameras from all over the model); pose-only refinement on the correspondences that
        agree; accepted when the two placements agree, otherwise the VLM reads the render-and-compare figures and
        picks; a drift guard holds a placement that bundle adjustment would drag
 audit: every camera rendered and compared with its photo (docs/EXAMINE.md); extra connected image sets become
        further models, the VLM judges whether they are the same site
```

The VLM (Qwen3.8-27B, local vLLM) never outputs geometry. It answers questions a person answers at a glance,
with evidence it can read: *which of these two renderings is aligned with the photo?*, *are these two image sets
the same site?* Every question, image and answer is logged and audited against ground truth in the benchmark
layer. The earlier system, a VLM choosing among complete reconstruction tools, is archived in
[docs/ARCHIVE_agent_results.md](docs/ARCHIVE_agent_results.md): it tracked a scripted controller with the same
tools, which is why the mapper itself changed ([docs/DIRECTION.md](docs/DIRECTION.md)).

**Status (2026-11-04).** v0 of the solver matches the scripted COLMAP controller on five dev scenes in one pass
and beats every fixed pipeline. v1 places the hard doppelganger images of BLH0001 correctly (36 of 39 cameras,
all within 11 degrees of ground truth, against 32 to 34 for every COLMAP pipeline); their precision and the
calibration of the render-and-compare verdict are the open items. Results log below; lessons in
[docs/LESSONS.md](docs/LESSONS.md); the showcase site with every episode's reasoning and 3D model is at
https://claude.ai/artifact/2hoxdMsajWP6VjoEFWnZbB (private link).

## Design principles

1. **The agent never outputs geometry.** Every pose comes from a tool; the VLM gates which placement is kept.
2. **No ground truth at runtime, by construction.** The harness's `Scene` type has no GT field; GT lives in the
   benchmark layer ([scripts/evaluate.py](scripts/evaluate.py) and the `eval_*.py` audits), which also scores every
   VLM verdict after the fact.
3. **The examine is the product.** A mapper that cannot tell a right model from a wrong one grows the wrong one.
   Match-based checks are blind exactly where it matters (lookalike matches are wrong and self-consistent), so the
   solver's acceptance tests are the feedforward model's agreement across anchor sets and render-and-compare,
   with the VLM reading the figures ([docs/EXAMINE.md](docs/EXAMINE.md)).
4. **Correctness over coverage, then coverage.** A wrong camera also drags its neighbours. But every image of the
   site should end up in the model, and images of another site in another model.
5. **Explicit state, pure tools, versioned artifacts.** The reconstruction is a pycolmap `Reconstruction` the
   loop and the agent can read and write; every tool call is cached by content hash and traced with its cost.
6. **Judge every change on all dev scenes against ground truth, never on one scene.**

## Harness

**Canonical state.** The COLMAP database is the shared format: cameras, images, keypoints,
matches, two-view geometries. hloc writes it, COLMAP and GLOMAP read it, XDG filters it, GLUEMAP's
refinement reads it. Pair scores from disambiguators are sidecar tables keyed by image pair.
Models are directories in COLMAP format, possibly with several components. A filter produces a
*new* database, so the unfiltered one is still an artifact.

```
workspace/<scene>/
  images/                  -> symlink to the input
  db/<id>.db               matches + two-view geometries (one per view graph × matcher × filter)
  scores/<id>.npz          pair scores (xdg, dgpp, flexglue_covis, cycle_consistency, ...)
  models/<id>/{0,1,...}/   COLMAP models + meta.json (provenance, cost)
  models/<id>/examine/     diagnostics JSON + renders
  trace.jsonl              every tool call, result summary, cost, and VLM message
```

**Backends in isolated environments.** The tools span incompatible stacks (torch 2.4 in
`gluemap`, 2.7 in `xdg`, 2.3 in `vggt`). A backend is a small CLI, `python -m
agenticsfm.backends.<name> spec.json`, run under its env's interpreter; it reads and writes the
workspace. Adding a backend is adding one wrapper. Heavy models get a persistent worker per env
so weights load once. The harness, the agent and the evaluation run in `agenticsfm` (pycolmap
4.2.1 with GLOMAP, hloc, openai).

**What the agent sees.** Compact JSON summaries plus images on request: thumbnail grids, the
view graph, a top-down and a side render of cameras and points per component, a pair with its
matches drawn. Qwen3.8 is a VLM; a folded point cloud or two copies of one building is visible in
a render before any statistic flags it.

## Tools

Each tool has one schema and several backends; the agent chooses the backend as a parameter.
"Have" means the code and weights are on this machine.

| Tool | What it does | Backends |
|---|---|---|
| `build_graph` | candidate pairs from retrieval, exhaustive, or a sequence hint; graph stats (degree, components) | SALAD (GLUEMAP), NetVLAD (hloc), DINOv2, exhaustive, sequential · *have all* |
| `match` | features and verified matches for the graph's pairs, into a database | ALIKED / SuperPoint / DaD + LightGlue, FlexGlue (sparse and dense), LoMa / RoMa, XFeat · *have all* |
| `score_pairs` | one score per edge from a covisibility / doppelganger classifier or a geometric test | XDG, DG++ (MASt3R), FlexGlue covisibility · *have*; rotation cycle consistency, missing-correspondence test · *to write* |
| `filter_edges` | new database keeping edges above a threshold, or minus a list; keeps the graph connected if asked | – |
| `reconstruct` | poses from a database (global) or from a chosen image subset (local) | COLMAP incremental, GLOMAP (pycolmap 4.2), GLUEMAP coarse · *have*; VGGT, Pi3, MapAnything, VGGSfM on subsets · *have weights; wrap* |
| `grow_model` | the open incremental solver: Pi3X seed on the densest cluster, PnP growth in verified groups, Pi3X joins, bundle adjustment; no COLMAP mapper (branch `open-solver`) | pycolmap primitives + Pi3X / VGGT · *have* |
| `examine` | GT-free diagnostics of a model: registration, components, reprojection error, track length, triangulation angles, per-edge residual between each two-view geometry and the model's relative pose, cycle inconsistency, pairs with many inliers but inconsistent poses; renders | *to write* |
| `inspect` | thumbnails of chosen images; a pair with its matches; a crop | agentic_dg tools · *have* |
| `split_model` / `merge_models` / `register_images` / `remove_images` | repair at image and component level; merge via shared images or via a feedforward model on images spanning both components | pycolmap · *to write* |
| `optimize` | bundle adjustment with a chosen camera model; GLUEMAP track snapping + augmented BA; re-triangulation | pycolmap, GLUEMAP refinement · *have* |
| `verify_poses` | VGGT (or VGGT-Omega) on each image with its strongest neighbours; an image whose pose disagrees with the feedforward model AND whose own matches disagree with its pose is a suspect | VGGT-1B · *have*; VGGT-Omega · *code + weights in third_party, not yet allowed to run* |
| `optimize` | share intrinsics per image size and focal-length cluster, upgrade to OPENCV, bundle adjust | pycolmap |
| `rank_models` | every model with the same statistics and the harness quality score | – |
| `compare_models` | two models overlaid in the scene frame after similarity alignment; which images moved, which are only in one | – |
| `view_graph` | the view graph drawn at the model's camera positions (or a spectral layout), edges coloured by a pair score; clusters after cutting weak edges, bridges, articulation images | – |
| `analyze` | a Python snippet over the workspace (models, databases, scores, examine and verify tables bound as functions) returning its printed output: exact facts instead of pictures | – |
| `reconstruct(backend=vggt)` + `merge_models` | feedforward local reconstruction of a listed group (with anchor images), attached to the global model by similarity alignment on the shared images, then re-triangulated and bundle adjusted on the database | VGGT-1B · *have*; VGGT-Omega · *pending* |
| `submit` | the final model | – |

### Failure modes and the tools that address them

This table is the knowledge the agent needs, and the playbook of the scripted controller.

| GT-free symptom | Likely cause | Tools |
|---|---|---|
| many components, low registration | sparse graph or weak matches (low overlap, big viewpoint change) | larger retrieval k; dense matcher; feedforward model on images across the gap, then `merge_models` |
| one component but bimodal edge residuals, cycle inconsistency, duplicated structure in renders | doppelgangers | `score_pairs`, `filter_edges`, re-map; or `split_model` and re-check the cut |
| high reprojection error, short tracks, drift along the sequence | wrong camera model or shared intrinsics across different cameras | per-camera grouping from EXIF, richer camera model, `optimize` |
| tiny baselines, few points, near-pure rotation (PTZ) | rotation-only motion | global SfM (rotation averaging), feedforward priors; do not expect triangulation |
| incremental mapping stops at a few images | bad initial pair | choose the initial pair; use GLOMAP |
| few verified matches everywhere | textureless or repetitive imagery | dense matcher at higher resolution; feedforward model |

### Controllers

- **Incremental** (the system): graph, match, `grow_model`, examine, optimize, submit
  ([controllers/incremental.py](src/agenticsfm/controllers/incremental.py); `scripts/run.py --controller incremental`).
- **Fixed pipelines** (baselines): COLMAP / GLOMAP with and without covisibility or XDG filtering.
- **Scripted** (baseline): COLMAP with a threshold sweep, VGGT verification and shared-intrinsics refinement; the
  strongest COLMAP-based controller we have, and what the old agent tracked.
- **Agent** (archived): the VLM choosing among the tools above with a playbook.

## Benchmarks

The harness runs on any image directory. GT enters only through
[evaluate.py](scripts/evaluate.py), which scores a model against a COLMAP GT model.

**Metrics** ([evaluation.py](src/agenticsfm/evaluation.py)): AUCs of the pairwise relative pose
error, max(rotation, translation direction), at 1/3/5/10/30°. Tables show 3/10/30.

| Metric | Pairs scored | Why |
|---|---|---|
| `auc_all` (**primary**) | all ordered GT pairs; a pair with an image outside the largest component scores 180° | dropping or splitting off hard images cannot help |
| `auc` (GLUEMAP's `benchmark/evaluate.py`) | pairs inside the largest component only | comparable with GLUEMAP's tables; dropping images **raises** it |
| `colmap_auc` (COLMAP's benchmark) | all components, set-based | credits correct splits partially; pointglue's WRIVA metric |

GLUEMAP's and COLMAP's code is imported from the reference repos, so numbers match their tables
exactly. Cost is reported alongside: wall time, GPU seconds, tool calls, VLM tokens.

**Datasets** ([datasets/](src/agenticsfm/datasets)): one adapter each, listing scenes and GT.

- **WRIVA** (have): 34 scenes, 6,804 images, 5–600 per scene, mixed cameras, PTZ and multi-zone
  sites with doppelgangers. Dev = sites A08, M07, BLH0001 (6 scenes, 793 images), the rest test.
  Prompts and scripted rules are tuned on dev only.
- **ETH3D** (have, `data/eth3d`), **LaMAR** (have, `~/gluemap/datasets/lamar`): to add as
  adapters; no doppelgangers, so they test that the agent does no harm on ordinary scenes.
- **IMC 2024/2025** or the Doppelgangers++ SfM scenes: to download; more scenes with ambiguity.

```bash
conda activate agenticsfm
python scripts/evaluate.py --baselines --out outputs/eval/baselines.csv
python scripts/evaluate.py --dataset wriva --run mine=/path/to/{scene}/sparse --split dev
```

## Results

### 2026-11-04: open solver v0 (branch `open-solver`): no COLMAP mapper, Pi3X seed, verified growth

The first step of [docs/DIRECTION.md](docs/DIRECTION.md), built and run in one night. `grow_model`
([solver/grow.py](src/agenticsfm/solver/grow.py)) replaces COLMAP's incremental pipeline with our own
loop over an explicit state (a pycolmap `Reconstruction`: cameras, poses, tracks) and pycolmap's
primitives (PnP registration, triangulation, local and global bundle adjustment):

1. **Seed.** Pi3X ([backends/feedforward.py](src/agenticsfm/backends/feedforward.py), `model: pi3x`;
   the focal length is recovered from its local point map as GLUEMAP does) solves the 12 images
   around the image with the largest inlier-weighted degree; its poses and focal lengths initialise
   the reconstruction, tracks are triangulated from the database's matches and bundle adjusted.
   Pi3X on 12 images: 2 s forward, 13 s model load.
2. **Grow.** Rounds of 8 PnP registrations (the images with most 2D-3D correspondences), each
   triangulated and locally refined; then every new image is verified: the inlier-weighted share of
   its strong pairs' matches (>= 50 inliers, partner registered) within 4 px of the epipolar lines
   must be >= 0.5, otherwise the image is deregistered and deferred. Global refinement whenever the
   model has grown by 10%.
3. **Join.** When PnP finds nothing, Pi3X solves up to 12 unplaced images together with the 6
   registered images they share most inliers with; the group is aligned to the model on those
   anchors (similarity on the camera centres; rejected above 5% of the scene span), registered,
   triangulated, refined and verified like any other addition.
4. Final global refinement; the model goes through `examine`, `optimize` and `submit` as in the
   scripted controller ([controllers/incremental.py](src/agenticsfm/controllers/incremental.py);
   `scripts/run.py --controller incremental`).

Six runs on five dev scenes, one run each, unfiltered databases unless stated, same matches as every other controller. "Grown"
is the solver's model before `optimize`; "incremental" is the submitted model. The scripted and
fixed numbers are pass 12 on the same databases (the scripted controller runs COLMAP with the
threshold sweep, verify and `optimize`).

| Scene (images) | metric | fixed pipeline | scripted (COLMAP) | **grown** | **incremental (grown + optimize)** |
|---|---|---|---|---|---|
| WRIVA M07 (50) | `auc_all` @3 / 10 | 23.2 / 76.4 | 36.8 / 81.0 | 26.5 / 77.8 | **38.8 / 81.7** |
| WRIVA BLH0001 (39) | `auc_all` @3 / 10 | 18.4 / 49.6 (34 img, filter 0.6) | 21.2 / 49.0 (36 img) | 19.3 / 46.8 (32 img) | **24.6** / 49.5 (32 img) |
| WRIVA BLH0001, filter 0.6 (39) | `auc_all` @3 / 10 | 18.4 / 49.6 (34 img) | 21.2 / 49.0 (36 img) | – | **24.5** / 49.6 (33 img, no fold: 0.003) |
| WRIVA A08 (150) | `auc_all` @3 / 10 | 17.8 / 44.3 | 17.8 / 44.3 | – | **18.4** / 44.2 (148 img) |
| ETH3D terrace (23) | `colmap_auc` @1 / 3 / 5 | 53.9 / 81.0 / 88.1 | 65.6 / 88.4 / 93.0 | 57.3 / 84.0 / 90.3 | **67.7 / 89.2 / 93.5** |
| ETH3D delivery_area (44) | `colmap_auc` @1 / 3 / 5 | 58.2 / 84.1 / 90.3 | 76.9 / 91.9 / 95.1 | 58.7 / 83.8 / 89.9 | **77.8 / 92.2 / 95.3** |

- **The mapper replacement costs nothing.** The grown models alone are at the level of COLMAP's
  mapper on the same database (terrace 57.3 vs 53.9, delivery_area 58.7 vs 58.2 at 1 deg; M07
  77.8 vs 76.4 at 10 deg), with all images registered in one model on three scenes, and the whole
  pipeline is at or above the scripted controller on all four, in one pass instead of a threshold
  sweep over several COLMAP mappings. One seed per scene; the ETH3D margins are within COLMAP's
  run-to-run noise, so this is parity, not a win.
- **BLH0001 is still the doppelganger scene.** Unfiltered, the solver registers 32 of 39: 18 PnP
  failures, 9 verified rejections, one join adding 3 of 10; its `strong_inconsistent_frac` is 0.088
  (a fold survives), and the result ties the fixed and scripted numbers at 10 deg with fewer images
  while leading at 3 deg. The verification rule removes images; it does not yet remove the wrong
  edges that fold the model (the pass 5 lesson), which is the first repair step to add to the loop.
- **Pi3X vs VGGT as the seed and joiner** (M07, same loop): Pi3X 50 of 50 images, 77.8 at 10 deg,
  2 verified rejections; VGGT 47 of 50, 66.9, 13 rejections and 3 images left unplaced. Pi3X is the
  default (`feedforward.model` in the config).
- **Scale check.** A08 (150 images): 148 registered in 21 rounds, one join refused (anchors did not align),
  44.2 at 10 deg against 44.3 for COLMAP, 389 s for the solver. With the 0.6 covisibility filter BLH0001
  grows without a fold (`strong_inconsistent_frac` 0.003) to 33 images and 49.6 / 24.5.
- **Cost.** M07 237 s (40 s of it Pi3X including two model loads), BLH0001 116 s, terrace 40 s,
  delivery_area 86 s, A08 389 s for the solver; the global refinement after every round is most of it and is
  the first thing to make cheaper. `optimize` adds 20 to 65 s.

What the loop does not do yet, in the order it will be added: cut edges instead of dropping images
when a verified rejection points at a fold; the feedforward focal for every image (today only the
seed and joined images get it; PnP-registered images start from the 1.2 x max(w, h) prior);
feedforward relative poses as priors in the bundle adjustment; the agent as the scheduler over
`seed` / `grow_round` / `join` with the per-round log as its evidence.

### 2026-11-04: v1 to v1.3, the VLM in the loop, and what the ground truth says

Starting point, from GT in the benchmark layer: every camera v0 placed on BLH0001, A08 and M07 is correct. The
BLH0001 loss is coverage: seven images whose matches are almost all lookalike pairs. Match support cannot judge
them (it rejected correct and wrong placements alike), and one accepted join was wrong by 60 degrees.

**Prototype.** Pi3X with four match anchors plus eight cameras sampled from all over the model places all seven
within 1 to 3 degrees of GT; a second anchor set agrees within 2 to 7 degrees. So the loop gained `place_hard`
(two anchor sets, agreement test), the VLM advisor, a drift guard, and multi-model output.

| Version | what changed | BLH0001 `auc_all` @3 / 10 (cameras; GT-wrong) | M07 | A08 |
|---|---|---|---|---|
| v0 | joins on match anchors | 24.6 / 49.5 (32; 0) | 38.8 / 81.7 (50; 0) | 18.4 / 44.2 (148; 0) |
| v1 | two anchor sets, VLM `same_place` arbitration, hold on drift | 20.1 / 47.3 (36; 0 above 11 deg) | 39.4 / 81.8 (50; 0) | 17.1 / 42.7 (149; 1 at 13 deg) |
| v1.1 | render-and-compare decisive, audit drops | 27 cameras (correct ones rejected) | 37.4 / 78.4 (49) | 148 |
| v1.2 | render veto on strong evidence only, pose refinement | 20.4 / 44.5 (32; 0) | 39.0 / 81.7 (50; 0) | – |
| v1.3 | render advisory, coarse-to-fine re-refinement of hard images | see the broad run below | | |

What the audits say:

- **The VLM is right when the question is image alignment.** On v1.2's BLH0001 run it chose between the two
  render-and-compare figures of a hard image six times: five picks were the GT-correct placement and the sixth
  was "neither" on a figure with no rendered content. Its earlier scene-identity verdicts (`same_place` on photos)
  were also right on all seven BLH0001 images but accepted a 13-degree placement on A08: it can judge the place,
  not the pose.
- **Render-and-compare is a precision instrument.** Calibrated on 302 GT-correct cameras of five scenes against
  10 and 20 degree yaw, 10% shifts and a far camera's pose (1,510 scored poses): `margin > 0.05` keeps 88% of the
  correct poses and passes 2 to 4% of the wrong ones; `margin < -0.05` catches 74 to 79% of the wrong ones at 2%
  false alarms. But a feedforward placement that is 2 to 3 degrees off before bundle adjustment scores like a
  wrong one, so using the verdict to veto placements (v1.1) rejected correct images. It now reports; the
  decision is the anchor-set agreement, the figures go to the VLM when the sets disagree.
- **Precision of hard placements is the open item.** Holding a placement at its feedforward pose costs 2 to 6
  degrees per camera (v1: BLH0001 gains four cameras and loses two points); letting bundle adjustment move it
  risks the lookalike correspondences dragging it (one camera moved 30% of the scene span). v1.3 re-refines every
  hard camera pose-only, coarse to fine (48, 24, 12 px), on the correspondences that agree with it in the final
  model.
- **Different scenes become different models.** A folder of M07's 50 images plus terrace's 23 (two sites, 11
  spurious cross-scene pairs of 15 to 26 inliers among 1,420): the solver returns two models, 50 and 23 images,
  each pure, and the advisor's `same_scene` verdict for the second component was "different". Mid-run it had
  accepted three terrace images into the M07 model on the advisor's say-so over a garbage rendering (they were
  filtered out again, by luck); v1.4 therefore gates hard candidates on real links (at least two verified pairs
  and 50 inliers to the model) and never lets an advisor pick override a render verdict of inconsistent.
- **Pi3X as examiner.** 1.9 s for 12 images; the model load (13 s per backend call) is now most of the solver's
  time (M07: 4 to 7 calls). A persistent worker is the next engineering step.

### 2026-11-04: the broad runs (v1.3 on 31 scenes, v1.4 on WRIVA dev) and what they changed

ETH3D, all 25 scenes, v1.3 against the baselines (`colmap_auc` @1 / 3 / 5; every image registered in every scene):

| | fixed pipeline | scripted (COLMAP) | open solver v1.3 |
|---|---|---|---|
| 13 dev scenes | 60.4 / 83.5 / 89.5 | 66.7 / 87.1 / 91.9 | 66.1 / 86.8 / 91.8 |
| 12 test scenes | 63.5 / 85.7 / 91.2 | 70.5 / 89.3 / 93.5 | 65.2 / 84.3 / 88.6 |
| all 25 | 61.9 / 84.6 / 90.3 | 68.5 / 88.2 / 92.7 | 65.7 / 85.6 / 90.2 |

Parity on dev; the test loss is four scenes, and GT says what they are. **old_computer** (30.9 vs 56.8 at 1 deg)
is a fold: 17 of 54 cameras sit at 180 degrees. Four of the twelve seed images were already mirrored by Pi3X, and
the PnP rounds extended the block with match support of 0.5 to 0.9, because the lookalike matches are wrong and
self-consistent. The render-and-compare audit rated 13 of the 17 consistent: it compares a camera with its
nearest neighbours, and a mirrored block is consistent with itself. The signal that sees it is the one a person
uses on pairs: the strong pairs between the two blocks disagree with the poses. **lounge** (41.6 vs 66.5, 10
images) is diffuse precision (median error 0.27 vs 0.14 degrees), as are terrains and botanical_garden.

WRIVA dev (`auc_all` @3 / 10):

| scene | fixed | scripted | v0 | v1.3 | v1.4 |
|---|---|---|---|---|---|
| A08 (150) | 17.8 / 44.3 | 17.8 / 44.3 | 18.4 / 44.2 | 17.2 / 43.1 | 17.2 / 42.6 |
| M07 (50) | 23.2 / 76.4 | 36.8 / 81.0 | 38.8 / 81.7 | 38.5 / 81.6 | 39.8 / 82.0 |
| BLH0001 (39) | 18.4 / 49.6 | 21.2 / 49.0 | 24.6 / 49.5 | 25.0 / 50.7 (36 img) | 21.8 / 46.1 (32 img) |
| PTZ s00 (198) | 19.4 / 61.1 | 20.3 / 61.6 | – | 16.8 / 56.1 (194) | 15.1 / 49.8 (179) |
| PTZ s01 (182) | 19.5 / 61.4 | 19.9 / 61.5 | – | 16.8 / 55.3 (176) | 15.3 / 51.6 (178) |
| PTZ s02 (174) | 18.8 / 60.8 | 19.2 / 60.9 | – | 14.8 / 52.3 (171) | 15.6 / 53.0 (164) |
| mean | 19.5 / 58.9 | 22.5 / 59.7 | | 21.5 / 56.5 | 20.8 / 54.2 |

The PTZ scenes are the loss, and GT explains it twice over. In v1.3 the hard stage accepted placements that
agreed within 8 to 10 degrees and advisor picks over inconsistent renderings: a dozen cameras per scene 5 to 17
degrees off. v1.4's gates (5-degree agreement, no pick over an inconsistent render, real links required) stop
that, and the registered count drops instead: on PTZ s00, 21 images were rejected by match support at 0.24 to
0.43 in the PnP rounds and never readmitted, although COLMAP places all of them. The hard stage cannot give
them the precision the PnP path would have. The PTZ runs also take 35 to 60 minutes each, almost all of it Pi3X
model loads (13 s per call) and VLM calls in the hard rounds.

v1.5, running as this is written: the seed is split into its largest self-consistent block when its strong pairs
disagree; a fold repair after growth deregisters the minority block of the consistent-pair graph and re-places
it as hard images with anchors from the majority block in view; and an image with support between 0.25 and 0.5
gets render-and-compare as a second opinion before it is rejected.

## Running the harness

```bash
conda activate agenticsfm
# one fixed pipeline on one benchmark scene, scored against GT afterwards (benchmark layer only)
python scripts/run.py --pipeline colmap_alikedfg_covis --scenes t04_v09_s00_r04_VaryingAltitudes_BLH0001 --gpus 4 --evaluate
# the dev split, any image folder
python scripts/run.py --pipeline glomap_alikedfg_covis --split dev --gpus 4 5
python scripts/run.py --folder /path/to/images --pipeline colmap_alikedfg
```

Layout ([workspace.py](src/agenticsfm/workspace.py)): the artifact store `outputs/cache/<scene_id>/`
is content-addressed (an artifact's ID hashes the tool, its parameters and its input artifacts;
`scene_id` hashes the image names and sizes) and shared by every run on the scene, so a second
controller re-uses matches and models. A run writes `outputs/runs/<run>/<scene>/{trace.jsonl,
artifacts.json, final/}`. Backends ([backends/](src/agenticsfm/backends)) are CLIs run under their
env's interpreter with a JSON spec: `retrieval` and `match` in `pointglue` (hloc NetVLAD;
pointglue's detector and matcher adapters), `xdg` in `xdg`. Tools ([tools/](src/agenticsfm/tools))
carry a JSON schema (what the agent sees) and return a compact summary plus renders. Fixed
pipelines are in [controllers/fixed.py](src/agenticsfm/controllers/fixed.py).

### Setup

In this repository: the harness (`src/agenticsfm`: workspace, tools, backends, controllers,
datasets, agent, prompts), the scripts, `configs/default.yaml` and `docs/`. Not in it: `outputs/`
(artifact cache, runs, logs; 86 GB on 2026-11-03), the `data` symlink to the benchmark images,
model weights, and the reference repositories (GLUEMAP, COLMAP's benchmark, FlexGlue, XDG). Every
absolute path is in `configs/default.yaml` (dataset roots, reference evaluators, baseline
reconstructions, backend environments) and has to be edited for another machine.

Conda environments (each backend is a CLI run under its own interpreter, see
[backends/](src/agenticsfm/backends)):

| env | used for |
|---|---|
| `agenticsfm` | harness, controllers, evaluation: pycolmap 4.2.1 (with GLOMAP), hloc 1.5, openai; `pip install -e .` |
| `pointglue` | retrieval (NetVLAD) and matching (ALIKED / SuperPoint + FlexGlue, DaD + LoMa) |
| `xdg` | XDG pair scoring |
| `agentic-dg` | VGGT-1B for `verify_poses` and `reconstruct(backend=vggt)`; VGGT-Omega |
| `vllm-q38` | the VLM server: `scripts/serve_vlm.sh`, Qwen3.8-27B on GPUs 0 to 3 ([docs/VLM_SERVICE.md](docs/VLM_SERVICE.md)) |

GT is read by `scripts/evaluate.py` only. GT audits of the tools (benchmark layer):
`scripts/eval_verify.py`, `eval_examine.py`, `eval_edges.py`, `eval_models.py`,
`analyze_failures.py`, all with `--dataset`. Episode pages with every turn's thinking and
evidence: `scripts/trace_site.py`; a trace as text: `scripts/trace.py`. The showcase site (every episode's VLM turns, tool steps and evidence, the open solver's growth rounds,
and the models in a 3D viewer): `scripts/export_site.py --out outputs/site` writes the data and `scripts/showcase/index.html` is the
page; the published copy of 2026-11-04 is at https://claude.ai/artifact/2hoxdMsajWP6VjoEFWnZbB (private link).

## Plan

| Step | Content |
|---|---|
| done | Harness, tools, benchmark adapters (WRIVA, ETH3D, LaMAR), evaluation identical to GLUEMAP's and COLMAP's; the tool-selection agent and its ceiling (archive). |
| done | Open solver v0: Pi3X seed, verified PnP growth, no COLMAP mapper; parity with the scripted COLMAP controller on five dev scenes. |
| done | v1: hard-image placement with two anchor sets, VLM arbitration on render-and-compare figures, drift guard, multi-model output; render-and-compare calibrated against GT. |
| next | Precision of hard placements (photometric pose refinement without matches); examine v2 as the acceptance test once calibrated for coarse placements; persistent Pi3X worker; the LaMAR 448-frame session and the WRIVA test split; the VLM as scheduler over seed / grow / hard with the per-round log as its evidence. |
