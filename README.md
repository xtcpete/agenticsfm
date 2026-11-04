# Agentic SfM: a harness of reconstruction tools, driven by a VLM

**Input:** a directory of unordered images. **Output:** camera poses (COLMAP model, one or several
components). No ground truth, no scene-specific setup. Scenes may contain doppelgangers
(repeated or symmetric structure) that fold distinct surfaces into one wrong model.

Coding agents already build plausible 3D scenes in Blender from photos. This project asks the
reverse question: can a local VLM (Qwen3.8-27B) run a *metric* reconstruction well, if it never
computes geometry itself? The VLM inspects intermediate reconstructions, diagnoses what went wrong,
and chooses the next tool and its parameters. Every pose comes from a tool.

```
 images ──▶ harness ─────────────────────────────────────────────────────────────────────┐
            │ workspace: COLMAP database(s), models, pair scores, renders, trace          │
            │                                                                             │
            │   ┌────────────── agent loop (budgeted) ─────────────────────────────┐      │
            │   │ VLM reads summary + renders ─▶ picks tool + params ─▶ artifact   │      │
            │   │        ▲                                                 │       │      │
            │   │        └──────────── examine / inspect ◀─────────────────┘       │      │
            │   └──────────────────────────────────────────────────────────────────┘      │
            │   tools: graph · match · disambiguate · reconstruct · examine · repair ·   │
            │          optimize · inspect   (each with several backends, own conda env)  │
            └───────────────────────────────────────────────────────────────────▶ submit(model)
                                                                                       │
                                                           benchmark layer (GT): evaluate
```

**Status (2026-11-03).** The harness, three controllers (fixed pipelines, scripted, agent) and
three benchmark adapters (WRIVA, ETH3D, LaMAR) are built and audited against GT. Clean result
(pass 13): the agent and the scripted controller are the same system. Both beat every fixed
pipeline (ETH3D, 25 scenes, `colmap_auc` @1 / 3 / 5: 68.7 / 88.3 / 92.8 against 61.9 / 84.6 / 90.3;
the FlexGlue paper's own pipeline reports 68.9 / 86.8 / 92.3), and the agent does not beat the
script. All reconstruction runs were stopped on 2026-11-03 evening to change direction: an open
incremental solver whose loop the agent schedules, with COLMAP's mapper replaced and its bundle
adjustment kept ([docs/DIRECTION.md](docs/DIRECTION.md)). What the experiments taught:
[docs/LESSONS.md](docs/LESSONS.md). The first step, the open solver v0 (`grow_model`, Pi3X seed, verified PnP growth, no COLMAP mapper), is on branch `open-solver` and matches the scripted controller on four dev scenes (2026-11-04 section below). The dated results log is under [Results so far](#results-so-far).

## Design principles

1. **The agent never outputs geometry.** Its only outputs are tool calls with parameters, and
   `submit(model_id)`.
2. **No ground truth at runtime, by construction.** The harness's `Scene` type
   ([datasets/](src/agenticsfm/datasets/__init__.py)) has no GT field. GT lives only in
   `BenchmarkScene`, read by [evaluate.py](scripts/evaluate.py). A new dataset is a directory
   of images.
3. **One pipeline is one path through the tool space.** COLMAP, GLOMAP, GLUEMAP, hloc, and
   feedforward models (VGGT, Pi3, MapAnything, VGGSfM) are *backends* of the same typed tools.
   The harness can reproduce each of them as a scripted controller, and it can combine them:
   GLOMAP for a global initialization, a feedforward model to bridge two components, COLMAP's
   bundle adjustment to finish. The best reconstruction a fixed pipeline can produce is the floor.
4. **Classifiers make pair-level calls; the agent sets policy.** In
   [agentic_dg](/home/gochen/agentic_dg/README.md), Qwen3.8 judging single image pairs reached
   accuracy 0.86 against XDG's 0.92, and overturned XDG's "no" wrongly 20 times in 27. So the
   agent works at scene and cluster level (which disambiguator, which threshold, which subgraph to
   split, re-check or merge) and looks at a pair only when a consistency check flags it.
5. **Cheap inner loop, expensive outer steps.** Matching, pair scoring, local feedforward
   inference and refinement cost minutes and are cached by content hash (images + parameters).
   Global mapping from cached matches costs seconds. The agent iterates filter → map → examine,
   and refines once, on the model it submits.
6. **Artifacts are versioned; tools are pure functions.** Every artifact has an ID and provenance
   (tool, parameters, input IDs). Repeated calls hit the cache; every call is in the trace with
   its cost (wall time, GPU seconds, tokens).
7. **The agent must beat a script.** The main ablation is a scripted controller with the same
   tools and a fixed diagnose-and-repair policy. Beating the default pipelines is not enough.

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

Same tools, different policies. Each is a trajectory through the tool space.

- **Fixed pipelines**: COLMAP, GLOMAP, GLUEMAP, each with no filter / XDG / FlexGlue covisibility.
- **Scripted**: fixed diagnose-and-repair rules on `examine` output (the table above).
- **Agent**: Qwen3.8-27B with the tools, a budget, and the playbook in its system prompt.
- **Incremental** (branch `open-solver`): graph, match, `grow_model`, examine, optimize, submit: the open solver end to end, the mapper comparison against the scripted controller.
- **Oracle**: best fixed pipeline per scene, using GT. An upper bound on *pipeline selection*.

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

## Results so far

### 2026-11-02: WRIVA reference pipelines under the three metrics

Pre-existing reconstructions, re-scored with [evaluate.py](scripts/evaluate.py). GLUEMAP:
`~/gluemap_mod/results/wriva` (Pi3, SIMPLE_PINHOLE, SHARED intrinsics, DG threshold 0.8). COLMAP:
pointglue's hloc → COLMAP incremental mapping (SIMPLE_RADIAL per image; XDG and FlexGlue
covisibility thresholds 0.6). Per-scene CSVs: `outputs/eval/{gluemap,colmap}_baselines.csv`.

| Method | all 34: `auc_all` @3 / 10 / 30 | test 28: @10 | dev 6: @10 | GLUEMAP's `auc@10`, all 34 | Registered (of 6,804) |
|---|---|---|---|---|---|
| GLUEMAP, no disambiguation | 11.8 / 28.7 / 47.1 | 31.3 | 16.4 | 29.5 | 6804 |
| GLUEMAP + DG++ | 20.8 / 44.5 / 61.8 | 47.7 | 29.6 | 45.1 | 6804 |
| GLUEMAP + XDG | 19.9 / 44.6 / 63.2 | 46.4 | 36.1 | 45.4 | 6789 |
| COLMAP, DaD + LoMa-G | 31.3 / 57.2 / 71.0 | 56.5 | **60.1** | 59.4 | 6771 |
| COLMAP, DaD + LoMa-G + XDG | 33.1 / 59.8 / 73.9 | 61.4 | 52.4 | 67.5 | 6648 |
| COLMAP, ALIKED + FlexGlue | 29.3 / 54.3 / 68.3 | 53.3 | 58.7 | 55.7 | 6787 |
| COLMAP, ALIKED + FlexGlue + XDG | 33.6 / 60.0 / 74.1 | 62.1 | 50.4 | 66.9 | 6670 |
| COLMAP, ALIKED + FlexGlue + FlexGlue covis. | **35.9 / 66.1 / 81.2** | **67.7** | 58.8 | **71.0** | 6740 |
| *Oracle: best of these 8 per scene* | *38.6 / 68.2 / 83.0* | *69.8* | *60.5* | – | – |

Checks: GLUEMAP's `auc` matches GLUEMAP's `results.csv` exactly, and `colmap_auc` matches
pointglue's table, e.g. DaD + LoMa-G 31.5 / 57.3 / 71.2 in both.

Takeaways:
- **Picking a pipeline per scene is worth about 2 points** (oracle 68.2 vs 66.1 at 10°). The
  winner varies (FlexGlue covis. 16 scenes, DaD + LoMa-G 7, ALIKED + FlexGlue + XDG 6), but the
  margins are small. The agent's value, if any, is in fixing failures *inside* a scene: which
  edges, components or splits are wrong. Hence the emphasis on `examine` and repair tools.
- **Filters drop images, and GLUEMAP's metric rewards that.** On test, XDG lifts COLMAP
  (DaD + LoMa-G) by 11 points of GLUEMAP's `auc@10` but 5 points of `auc_all@10`. Hence
  `auc_all` is primary.
- **Disambiguation is not uniformly good.** On the dev sites XDG lowers COLMAP from 60.1 to
  52.4 but raises GLUEMAP from 16.4 to 36.1. How much to filter depends on scene and pipeline.
- **GLUEMAP is far behind COLMAP here** (44.6 vs 57–66). Untested guess: one shared pinhole
  camera fits WRIVA's mixed, distorted cameras poorly (GT is FULL_OPENCV).

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
evidence: `scripts/trace_site.py`; a trace as text: `scripts/trace.py`.

### 2026-11-02: harness reproduces the reference pipelines from images (M1)

Two dev scenes, run end to end by [scripts/run.py](scripts/run.py) (exhaustive pairs for BLH0001,
NetVLAD top-100 for A08; ALIKED 4096 @ 1024; one SIMPLE_RADIAL camera per image, estimated by the
mapper). "Ref." is pointglue's run of the same pipeline (table above); the mapper is seeded but
multi-threaded, so small differences are expected.

| Scene (images) | Pipeline | `auc_all` @3 / 10 / 30 | Ref. @3 / 10 / 30 | Registered | Models | Wall |
|---|---|---|---|---|---|---|
| BLH0001 (39) | COLMAP, ALIKED + FlexGlue + covis. 0.6 | 20.9 / 51.9 / 64.4 | 16.4 / 47.8 / 62.8 | 34 (ref. 36) | 1 | 2.5 min |
| BLH0001 (39) | COLMAP, ALIKED + FlexGlue + XDG 0.6 | 17.3 / 40.2 / 58.0 | 16.0 / 40.2 / 55.8 | 34 (ref. 35) | 1 | 3 min (XDG 1.9 min) |
| BLH0001 (39) | GLOMAP, ALIKED + FlexGlue + covis. 0.6 | 18.6 / 46.5 / 65.2 | – | 39 | 2 (36 + 3) | 18 s after matching |
| A08 (150) | COLMAP, ALIKED + FlexGlue + covis. 0.6 | 18.3 / 44.5 / 78.9 | 17.9 / 44.4 / 78.8 | 150 | 2 | 12.7 min (match 6, map 5) |

`examine` on the BLH0001 COLMAP model: 370 of 381 verified pairs lie inside the model, 3 of them
inconsistent with its poses (fewer than 50% of their inliers within 4 px of the epipolar lines),
11 touch the 5 unregistered images. GLOMAP registers everything but splits 3 images off and has a
higher reprojection error (2.1 vs 1.5 px). The next step (M2) puts these summaries and the
renders in front of the VLM.

### 2026-11-02: first agent and scripted runs on the dev split (M2)

Six dev scenes, `auc_all@10`. Every controller uses the same tools and the cached matches; the
agent is Qwen3.8-27B (thinking, `reasoning_effort: medium`) with a budget of 20 tool calls and
45 minutes. Pass 1: tools up to `examine`. Pass 2: `verify_poses` (VGGT on each image and its
neighbours, below) and a playbook that judges correctness before coverage and allows dropping
images. Trace pages with every turn's thinking and evidence: `scripts/trace_site.py`.

| Scene (images) | Best fixed (COLMAP + FlexGlue covis. 0.6) | Scripted, pass 1 | Scripted, pass 2 | Agent, pass 1 | Agent, pass 2 |
|---|---|---|---|---|---|
| A08 (150) | 44.5 | 44.5 | 44.4 | 43.7 | 43.5 |
| M07 (50) | 77.0 | 76.6 | 76.6 | 70.4 | 76.6 |
| BLH0001 (39) | 51.9 | 51.9 | 46.2 | 41.3 | 37.2 |
| PTZ s00 (198) | 61.4 | 61.4 | 61.5 | 61.6 | 61.4 |
| PTZ s01 (182) | 61.3 | 61.3 | 61.3 | 61.4 | 60.6 |
| PTZ s02 (174) | 60.7 | 60.7 | 60.7 | 60.6 | 56.3 |
| **mean** | **59.5** | 59.4 | 58.4 | 56.5 | 55.9 |

Passes 3 and 4 (`auc_all@10`, with `auc_all@3` in brackets; pass 3: corroborated `verify_poses`;
pass 4: `optimize`, `rank_models`, per-image constraint table in `examine`):

| Scene | Best fixed | Scripted, pass 3 | Scripted, pass 4 | Agent, pass 3 | Agent, pass 4 |
|---|---|---|---|---|---|
| A08 | 44.5 (18.3) | 44.4 | 44.4 (17.9) | 27.5 | **45.3** (18.8) |
| M07 | 77.0 (24.5) | 76.6 | 81.0 (36.8) | 76.6 | **82.7** (42.8) |
| BLH0001 | **51.9** (20.9) | 46.2 | 47.6 (20.2) | 42.4 | 39.3 (16.2) |
| PTZ s00 | 61.4 (19.9) | 61.5 | **61.7** (20.5) | 61.5 | 61.3 (19.9) |
| PTZ s01 | 61.3 (19.3) | 61.3 | **61.6** (19.9) | 59.6 | 61.0 (19.4) |
| PTZ s02 | 60.7 (18.6) | 60.7 | **60.9** (19.1) | 60.7 | **60.9** (19.1) |
| **mean** | 59.5 (20.2) | 58.4 | **59.5 (22.4)** | 54.7 | 58.4 (**22.7**) |

Pass 5 (GLOMAP penalised in the quality rule; playbook: a folded model needs edge removal, not
image dropping), `auc_all@10` (`@3`):

| Scene | Best fixed | Scripted, pass 5 | Agent, pass 5 |
|---|---|---|---|
| A08 | 44.5 (18.3) | 44.4 (17.9) | **45.6** (19.1) |
| M07 | 77.0 (24.5) | 81.0 (36.8) | **82.7** (42.8) |
| BLH0001 | 51.9 (20.9) | 48.5 (21.3) | **57.6** (24.6) |
| PTZ s00 | 61.4 (19.9) | **61.7** (20.5) | 61.3 (19.9) |
| PTZ s01 | 61.3 (19.3) | **61.6** (19.9) | 61.0 (19.4) |
| PTZ s02 | 60.7 (18.6) | **60.9** (19.1) | 60.7 (18.5) |
| **mean** | 59.5 (20.2) | 59.7 (22.6) | **61.5 (24.1)** |

The agent beats every fixed pipeline for the first time, by 2 points at 10 deg and 4 at 3 deg.
On BLH0001 it lowered the covisibility threshold until the graph stayed in one component (0.3),
mapped, saw the fold gone (strong inconsistent fraction 0.109 -> 0.005), verified (0 suspects),
optimised, and submitted 36 images at 57.6; the fixed pipeline's 0.6 keeps 34 images at 51.9. One
run; the mapper's variance on this scene is several points, so pass 6 repeats the comparison with
the threshold search also in the scripted candidates and the default recipe run as a prelude of
every agent episode. Per the user's direction the matcher stays ALIKED + FlexGlue (DaD + LoMa was
wired as a backend and scores 50.9 unfiltered on BLH0001, but it is not offered to the agent).

Pass 6 (threshold sweep 0.6 / 0.4 / 0.3 in the scripted candidates; agent with the default recipe
as a prelude): scripted 60.7 (BLH0001 54.8 with all 39 images, one model), its best so far. The
agent half of pass 6 is void on three scenes: the tool code was being edited during the run and
every reconstruct call failed on a temporary bug, after which one episode invented model ids.
Lesson recorded: never edit the source while a pass runs (backends and analysis snippets are
subprocesses that read the files at call time). Pass 7 repeats the agent run on the fixed code
with the complete tool set.

Pass 7 (agent with the prelude and the complete tool set, valid run): 59.6 / 22.2, below the
script (60.7) and below pass 5 (61.5). With the script's best candidate handed to it, the agent
accepts it after 3 calls on four scenes; on BLH0001 it took the 39-image candidate (54.8) and
submitted it after `optimize` had degraded it to 48.3, which neither examine nor verify flagged.
Two consequences: the prelude is dropped (the agent explores more when it builds its own
candidates), and `optimize` was audited: over 19 before/after pairs on the dev scenes it raises the
GT score in 18 (M07 +4 to +12, BLH0001 +1 to +2, A08 +1) and loses 1 point once, with a consistent
GT-free signature (inlier-weighted inconsistency down, match support up); the BLH0001 drop in
pass 7 came from the agent re-optimising a different, worse candidate. Pass 8: pass-5
configuration (no prelude, 25 calls, 60 min) with the complete tool set.

Pass 8 (no prelude, complete tool set): 59.3 / 21.9. The agent's `merge_models` attempt on BLH0001
failed on a mis-registered tool (now fixed, with an import-time check that every tool is bound to
the function of its name). Three valid agent runs so far: 61.5, 59.6, 59.3 against the script's
60.7 and the fixed pipeline's 59.5, so the agent is at parity with the fixed pipelines and the
pass-5 win is within run-to-run variance. Choice audit over 162 dev models: the quality rule's
regret vs the GT-best candidate is 2.5 points on average and 11 on BLH0001, where the 39-image
models beat the 36-image one on coverage while GT prefers the latter; adding 0.3 x median pair
consistency lowers the regret to 2.2 (max 9.2) and is adopted. Per image, VGGT's translation
direction disagreement ranks misplaced images with AUC 0.97 but makes a poor hard rule (precision
0.32), so `verify_poses` now reports a suspicion ranking (rotation, translation direction, match
support) next to the hard suspects. Pass 9 runs with these.

Pass 9 (merge fixed, quality term, suspicion ranking): 59.5 / 21.9. Four valid agent runs: 61.5,
59.6, 59.3, 59.5 (mean 60.0) against the script's 60.7 and the fixed pipeline's 59.5. The agent is
at parity with the fixed pipelines and about one point below the improved script; the pass-5 win is
not reproducible. The recurring loss is the same final choice on BLH0001 (a 39-image model at 48.3
over the script's 39-image model at 54.8 and the 36-image one at 57.6). The comparison of all 18
BLH0001 candidates explains it: the agent's choice is GLOMAP-derived and looks best of all to every
match-based statistic (strong inconsistent fraction 0.000, median pair consistency 0.971, match
support 0.905), while the COLMAP models with 36 images, which GT scores at 55 to 58, look slightly
worse. GLOMAP fits exactly the pairwise constraints those statistics measure, so they cannot see
its lower precision; on the dev split GLOMAP-derived models score lower than COLMAP ones on four of
six scenes (A08 39 vs 43, PTZ 8 to 17 vs 60). A GLOMAP penalty of 0.10 in the quality rule (about
4 images in 39) lowers the audited choice regret from 2.2 to 1.5 points (max 5.1) and is adopted
in `rank_models`, the scripted rule and the playbook. Pass 10 runs with it.

Pass 10 (GLOMAP penalty 0.10): 60.2 / 22.6. On BLH0001 the agent picked the VGGT-merged 37-image
model (52.5) over the GLOMAP ones. Five valid agent runs: 61.5, 59.6, 59.3, 59.5, 60.2 (mean 60.0,
all at or above the fixed pipeline's 59.5; script 60.7).

Pass 4: the scripted controller ties the best fixed pipeline at 10 deg and beats it by 2.2 at 3 deg;
the agent beats the fixed pipeline on A08 and M07 (its M07 model is the best of any run) and ties
on the PTZ scenes, but loses 12 points on BLH0001, where it kept dropping VGGT-flagged images
(down to 33) instead of removing the wrong edges that fold the model; the playbook now says so.
The scripted controller on BLH0001 prefers a 36-image GLOMAP model (47.6) over the 34-image COLMAP
one (51.9); GLOMAP's precision is lower and varies between runs (46.2 to 52.2 on the same input),
so the quality rule now penalises GLOMAP models.

Findings:
- **The agent diagnoses well and decides badly.** In every episode it found the right problem
  (the folded facades of BLH0001 and which images were misplaced; the doppelganger edges of the
  PTZ scenes) and chose sensible tools. It lost points on the final choice: on M07 (pass 1) it
  preferred GLOMAP's single 50-image model over COLMAP's more accurate split one; on BLH0001 it
  filtered at 0.5 instead of 0.6, keeping one misplaced image, then in pass 2 dropped too many.
- **GLOMAP is a trap on this data.** It registers every image, including wrong placements, and on
  the PTZ scenes (near-pure rotation) it is catastrophic: 7.5 / 13.6 / 17.0 against 60 for COLMAP
  on the same filtered database. Nothing GT-free in `examine` flags that, so the playbook now says
  to verify GLOMAP models before trusting them.
- **Match consistency separates folds from noise once weak pairs are excluded.** The headline
  statistic of `examine` is the share of strong pairs (>= 50 inliers) whose matches mostly disagree
  with the model's poses: 0.16 to 0.20 on the folded BLH0001 models, <= 0.016 on good ones, 0.045 on
  A08 where pose noise dominates. Counting all pairs gave 0.30 on A08's best model, useless.
- **VGGT as a pose check (`verify_poses`) is right when it fires hard and wrong on specific
  images.** Against GT, every image it flagged above 50 deg on BLH0001 was misplaced (GT error 42 to
  177 deg); but it also flagged one zoomed-in PTZ frame per scene and BLH0001's image 1 at 40 to 118
  deg while their GT error was below 3 deg, in every model, so the failure is VGGT's, not the
  model's. Dropping those cost 1 to 4 points per scene in pass 2. A suspect now needs both signals:
  VGGT disagreement and inconsistent matches of its own (inlier-weighted share of consistent
  matches over its strong pairs below 0.5). On 876 image checks in 15 dev models that gives
  precision 1.00 (10 of 10 suspects have GT error 20 to 177 deg) and recall 0.48; the misses are
  12 to 20 deg errors, mostly GLOMAP imprecision (`scripts/eval_verify.py`).
- **The scripted controller is as good as the best fixed pipeline** (59.4 vs 59.5), because its
  rule (filter when strong pairs are inconsistent) fires on exactly the scenes where filtering
  helps. The agent must beat it to justify itself.
- Cost: an agent episode is 6 to 19 tool calls, 2 to 40 minutes, 30k to 150k prompt tokens.

### 2026-11-02: examination tools, round 3

Renders now share one frame per scene (gravity up, estimated from the cameras; the first examined
model is the reference and later models are similarity-aligned to it on common images; fixed axis
range), with image numbers as labels and weakly constrained cameras in orange. `compare_models`,
`view_graph` and `analyze` (run code over the workspace) give the agent exact facts instead of
pictures. Local reconstruction with a feedforward model plus `merge_models` attaches a group the
mapper cannot place: on BLH0001 the three images cut off by every filter (13, 28, 33: their only
links outward are 90-inlier edges with covisibility 0.02 to 0.12) are placed by VGGT together with
7 anchor images and merged into the 34-image model; the merged model passes examine (inconsistent
fraction 0.011) and verify (0 suspects) and scores 52.5 with 37 images against 51.9 with 34.

### 2026-11-02: auditing the examination tools with GT (benchmark layer only)

Scripts: `analyze_failures.py` (per-model error distributions, per-image GT error),
`eval_edges.py` (edge oracle: shared points in the GT-triangulated model), `eval_examine.py`
(every model artifact of the dev scenes against its GT-free statistics), `eval_verify.py`.

- **Where the dev points go.** Five of the six submitted dev models have no misplaced image
  (median relative-rotation error vs the 10 nearest GT cameras > 30 deg); the loss is diffuse
  precision: A08 has 74% of pairs within 10 deg but 34% within 3 deg. Only BLH0001 is
  topology-limited. Substituting GT intrinsics and re-bundling lifts M07 from 77 to 87 and A08 from
  44.5 to 46.2: intrinsics are the main precision lever. The scenes carry no EXIF; GT has 1 camera
  for M07's 50 images and 2 for BLH0001's 39, 21 to 38 for the PTZ and A08 sets (zoom levels).
- **`optimize`** (new): share intrinsics among images of one size and focal-length cluster (zoom
  level), upgrade to OPENCV, bundle adjust. On the fixed pipeline's models, `auc_all@10`: A08 44.5
  -> 45.7, M07 77.0 -> 81.3, BLH0001 51.9 -> 50.9, PTZ 61.4 / 61.3 / 60.7 -> 61.6 / 61.6 / 60.9; dev mean
  59.5 -> 60.3 at 10 deg and 20.3 -> 23.0 at 3 deg. Sharing by image size alone diverges where one
  size spans many zoom levels (A08 41.0, PTZ 57.5; dev mean 57.5); one camera per image with the
  OPENCV model alone gives 59.9. Now the last step of the scripted controller and of the playbook.
- **Edge oracle** (12k true, 12k wrong edges over the six main databases): rotation cycle
  consistency at 30 deg from re-estimated two-view poses is the best single signal on the large
  scenes (A08: AP 0.93 vs 0.83 for FlexGlue covisibility; mean AP combined with covisibility 0.93 vs
  0.91), but on BLH0001 nothing beats the inlier count: its doppelganger edges form self-consistent
  cycles. Two-view rotations are off by 12 to 16 deg median under EXIF priors, 4 to 5 deg with the
  model's intrinsics, and the cycle AP barely moves, so the signal is as good as two-view geometry
  allows. No drop rule dominates: fewer wrong edges kept always costs more true edges dropped.
  `score_pairs(method="cycle_consistency")` is available; it is not in the default recipe yet.
- **Model choice is not the bottleneck.** Over 69 dev model artifacts, the scripted rule
  (coverage - 2 x strong inconsistent fraction) picks within 1.0 point of the GT-best candidate per
  scene on average; the candidates are what limit the result. `rank_models` (new) shows the agent
  all candidates with the same statistics and this score. Reprojection error does not predict pose
  accuracy (rank correlation -0.09 across models; imprecise images even fit their few points better).
- **Per-image signals.** Misplaced images (> 30 deg): VGGT disagreement > 15 deg together with match
  support < 0.5 finds them with precision 0.90 and recall 0.82 over 1,284 verified placements;
  either signal alone has precision 0.36 or 0.05. Imprecise images (10 to 30 deg, 262 of 9,046
  placements): match support ranks them with AUC 0.85 and the image's triangulated-point count with
  0.80; VGGT is blind to them (0.37). `examine` now lists weakly constrained images by those two.

### 2026-11-03: ETH3D in FlexGlue's setting (M4 started early)

Adapter [datasets/eth3d.py](src/agenticsfm/datasets/eth3d.py): the 25 DSLR scenes (898 undistorted
images) with the 13 training scenes as dev. Pipeline as the FlexGlue paper
([arXiv 2610.32976](https://arxiv.org/abs/2610.32976v1), Table 4 / Table 9 and App. B.2, E.2):
SuperPoint, 4,096 keypoints at a 1,024 px long side, FlexGlue sparse-to-sparse at 512 px, exhaustive
pairs, hloc defaults (one SIMPLE_RADIAL camera per image, no priors), COLMAP; its own
covisibility filter at 0.9; AUC of the relative pose error at 1 / 3 / 5 deg per scene, averaged.
The paper reports 68.9 / 86.8 / 92.3 (95.3 at 10) for SP + FlexGlue + filter over 25 scenes, three
COLMAP seeds; the ALIKED detector is not run for SfM in the paper, so ETH3D uses SuperPoint.

| 13 dev scenes, `colmap_auc` @1 / 3 / 5 | no filter | covisibility 0.9 | covisibility 0.6 |
|---|---|---|---|
| mean | 51.1 / 76.3 / 83.4 | 54.4 / 79.5 / 86.2 | 54.6 / 79.4 / 86.2 |
| meadow (15 images) | 15.2 / 35.8 / 43.5 | 40.5 / 68.2 / 75.4 | 31.7 / 59.6 / 69.2 |
| playground (38) | 67.2 / 88.3 / 92.9 | 64.9 / 85.5 / 90.0 (2 models) | 70.5 / 89.2 / 93.2 (2 models) |
| terrace (23) | 55.3 / 83.0 / 89.5 | 52.6 / 81.8 / 88.8 | 60.8 / 85.7 / 91.3 |

All 25 scenes, one seed (`colmap_auc` @1 / 3 / 5 / 10): no filter 50.3 / 73.4 / 79.6 / 85.1, covisibility
0.9 55.9 / 80.9 / 87.4 / 93.0; the paper reports 60.1 / 78.9 / 83.5 and 68.9 / 86.8 / 92.3 / 95.3. Our
numbers are about 10 points lower at 1 deg and 3 to 5 lower at 3 and 5 deg for both rows, so the gap
is a precision offset of the setup rather than the filter. Detection resolution is most of it: at
a 1,600 px long side (pointglue's ETH3D protocol; the paper's text says 1,024) the 13 dev scenes give
60.4 / 83.5 / 89.5 / 94.5 instead of 54.4 / 79.5 / 86.2 / 92.2, with meadow 40.5 -> 67.8 and terrains
39.4 -> 63.5 at 1 deg. The ETH3D experiments use 1,600 (stated deviation); the remaining gap to the
paper is about 5 points at 1 deg and may be its three-seed average or full-resolution detection.
Recipes are per benchmark in the config (`datasets.<name>.recipe`: detector, keypoints, resolution,
thresholds to sweep) and both controllers follow them.

Scripted controller on the 13 dev scenes with that recipe (`colmap_auc` @1 / 3 / 5): **66.7 / 87.1 /
91.9** against 60.4 / 83.5 / 89.5 for the fixed pipeline, a gain on 11 of 13 scenes (delivery_area
58.1 -> 76.9 and terrace 53.9 -> 65.6 at 1 deg; meadow and pipes unchanged). The threshold sweep
rarely changes the pick here (0.9 wins or ties on every scene); the gain is the shared-intrinsics
refinement, since each ETH3D scene is one DSLR camera, plus the connectivity-preserving filter on
the scenes the 0.9 filter had split. With the same matcher and recipe this is at the level the paper
reports over all 25 scenes (68.9 / 86.8 / 92.3).

Agent on the same 13 scenes (no prelude, 25 calls, 9 to 18 used, 2 to 19 minutes per scene):
**66.1 / 87.0 / 92.0**, tying the script (66.7 / 87.1 / 91.9) and beating the fixed pipeline with
the same matcher and recipe by 5.7 points at 1 deg and 3.5 at 3 deg. It beats the script on office
(34.6 / 65.4 / 77.4 vs 32.8 / 61.8 / 73.8) and loses on terrains (63.5 vs 71.4 at 1 deg), where it
submitted without the refinement. On ETH3D the agent therefore clears the paper's own pipeline
with the paper's matcher and settings, by the margin of what it adds on top: verified filtering and
shared-intrinsics refinement.

| 13 dev scenes, `colmap_auc` @1 / 3 / 5 | fixed pipeline (paper's recipe) | scripted controller | agent |
|---|---|---|---|
| mean | 60.4 / 83.5 / 89.5 | 66.7 / 87.1 / 91.9 | 66.1 / 87.0 / 92.0 | The filter's effect is scene-specific here too (meadow +32 at 3 deg,
playground and terrace split or lose), which is the room the threshold search and the merge tools
have to exploit.

### 2026-11-03: ETH3D held-out test (one shot) and what it exposed

The 12 scenes the recipe and prompts never saw (botanical_garden, boulders, bridge, door,
exhibition_hall, lecture_room, living_room, lounge, observatory, old_computer, statue, terrace_2),
run once with no tuning after the dev results above:

| 12 test scenes, `colmap_auc` @1 / 3 / 5 | fixed pipeline (paper's recipe) | scripted controller | agent (first shot) |
|---|---|---|---|
| mean | 63.5 / 85.7 / 91.2 | **70.5 / 89.3 / 93.5** | 67.9 / 86.1 / 90.1 |
| all 25 scenes (paper: 68.9 / 86.8 / 92.3) | 61.9 / 84.6 / 90.3 | **68.5 / 88.1 / 92.7** | 66.9 / 86.6 / 91.1 |

The scripted controller transfers: it beats the fixed pipeline on 10 of 12 test scenes (door and
terrace_2 lose about 3 points at 1 deg, both tiny scenes where the refinement overfits) and, over
all 25 scenes, matches the paper's own pipeline at 1 deg and beats it at 3 and 5 deg with the
paper's matcher and recipe. The agent ties or beats the script on 11 of 12 scenes (living_room
76.7 vs 75.5, lecture_room 73.6 vs 73.0 at 1 deg) and loses the split on one: botanical_garden,
41.6 / 52.0 / 54.1 against 75.0 / 91.5 / 94.8, where it dropped 7 of 30 images that were placed
correctly. Its trace is a clean piece of reasoning on wrong evidence: `examine` reported a fold
(`strong_inconsistent_frac` 0.18 on the unfiltered model, 0.16 at threshold 0.6), `verify_poses`
ranked exactly those 7 images as most suspicious by translation-direction disagreement (67 to
161 deg), the agent confirmed on the images that the greenhouse has repeated signs and palms,
concluded "true doppelganger that no edge score can separate", dropped the images, and the
suspects vanished. GT says every 30-image model of that scene scores 94.5 to 97.4 `auc_all@10`:
there was no fold.

Two tool defects, found by auditing against GT over all 25 ETH3D scenes and the 6 WRIVA dev scenes
(`scripts/eval_verify.py --dataset ...`, `scripts/eval_examine.py --dataset ...`), not by patching the scene:

- **Sub-model ownership.** COLMAP keeps an abandoned initialization next to the main model: 52
  of the 58 multi-model artifacts in the cache have a small sub-model whose images are all in the
  largest model too. `examine` and `verify_poses` let the last sub-model claim those images, so
  their pairs to the main model became "cross-model", their match support was computed against
  the leftover's poses (0.02 to 0.07 instead of 0.8) and VGGT was compared with the wrong poses.
  Fixed: an image belongs to the largest model that registers it; `examine` reports
  `redundant_submodels` and the playbook says what they are. On botanical_garden the unfiltered
  model now reads 0.006 (was 0.18) and verify finds 0 suspects (was 1 suspect and 7 "most
  suspicious").
- **Translation-direction disagreement is noise.** As a misplacement signal (GT error > 10 deg)
  flagging images above 60 deg has precision 0.09 on ETH3D (2,923 checks) and 0.11 on WRIVA
  (10,184 checks); above 30 deg, 0.19 and 0.08. The corroborated suspect rule (rotation > 15 deg
  AND match support < 0.5) keeps precision 0.71 and 0.79. The ranking no longer uses the
  translation direction, the summaries no longer show it, and the playbook says to drop images
  only on the suspect verdict (a dropped correct image costs exactly as many pairs as a kept
  misplaced one).

Also from the audit over 194 ETH3D models: the inlier-weighted inconsistency ranks candidate
models best (Spearman 0.70 with `auc_all@10`, regret 0.2 points against the oracle pick, vs 2.2
for `strong_inconsistent_frac` and 11.7 for reprojection error); it is a candidate for the
scripted rule in a later pass. Pass 11 re-runs both controllers on both dev splits with the fixes
and then, as a second shot, on the ETH3D test scenes; both numbers are reported.

### 2026-11-03: pass 11 (tool fixes from the held-out audit), dev and second-shot test

Both dev splits with the fixed `examine` / `verify_poses` and the playbook on dropping:

| Dev split | scripted (pass 10 -> 11) | agent pass 10 | agent pass 11 |
|---|---|---|---|
| ETH3D 13 scenes, `colmap_auc` @1 / 3 / 5 | 66.7 / 87.1 / 91.9 -> same | 66.1 / 87.0 / 92.0 | **66.8 / 87.4 / 92.2** |
| WRIVA 6 scenes, `auc_all` @10 / @3 | 60.7 / 22.6 -> 59.7 / 22.5 | 60.2 / 22.5 | **60.3 / 22.7** |

The agent now beats the script on both dev splits; it recovered terrains and delivery_area (pass 10
skipped the refinement on one and dropped images on the other). The script moved only on BLH0001
(54.8 -> 49.0): its quality rule rated the 0.3 and 0.4 filters within 0.002 of each other and the
corrected statistics flipped that coin. A candidate replacement was checked on both benchmarks
before touching the rule: the inlier-weighted inconsistency ranks ETH3D candidates best (Spearman
0.73, regret 0.2 points vs the oracle) but is among the worst on WRIVA (regret 13.6), because
GLOMAP and merged models fit the pairwise constraints exactly while being less precise. The rule
stays.

ETH3D held-out test, second shot (same 12 scenes, after the fixes; the first shot above stands as
the one-shot number):

| 12 test scenes, `colmap_auc` @1 / 3 / 5 | fixed | scripted pass 10 / 11 | agent pass 10 / 11 |
|---|---|---|---|
| mean | 63.5 / 85.7 / 91.2 | 70.5 / 89.3 / 93.5 -> 70.2 / 89.0 / 93.1 | 67.9 / 86.1 / 90.1 -> **70.4 / 89.0 / 93.1** |
| all 25 scenes (paper 68.9 / 86.8 / 92.3) | 61.9 / 84.6 / 90.3 | 68.5 / 88.1 / 92.7 -> 68.4 / 88.0 / 92.5 | 66.9 / 86.6 / 91.1 -> **68.5 / 88.2 / 92.7** |

The agent's botanical_garden went from 41.6 to 71.3 at 1 deg (30 of 30 images, same model as the
script); it ties the script on 9 of 12 test scenes, wins boulders and living_room, loses bridge by
0.3. Both controllers lost 3.7 points on botanical_garden against the pass 10 script (75.0), which
exposed a third, harness-level defect: both submitted the unfiltered model plus refinement, whose
`auc_all@10` is identical to the 0.9-filtered one (97.4), but COLMAP's set-based `colmap_auc`
scores *every* sub-model, and the 7-image leftover initialization beside the 30-image model costs
3.5 points at 1 deg (74.8 with the largest model alone vs 71.3 with both). `submit` now drops
sub-models that add no image to the largest one (nothing else changes under any metric); pass 12
re-runs all three controllers with it.

### 2026-11-03: pass 12 (submit drops redundant sub-models) and a leak in rank_models

Pass 12, all controllers re-run with the `submit` fix (ETH3D `colmap_auc` @1 / 3 / 5; WRIVA `auc_all` @10 / @3):

| | fixed | scripted | agent |
|---|---|---|---|
| ETH3D 13 dev | 60.4 / 83.5 / 89.5 | 66.7 / 87.1 / 91.9 | 66.9 / 87.4 / 92.2 |
| ETH3D 12 test | 63.5 / 85.7 / 91.2 | 70.5 / 89.3 / 93.5 | 70.7 / 89.4 / 93.5 |
| ETH3D all 25 (paper 68.9 / 86.8 / 92.3) | 61.9 / 84.6 / 90.3 | 68.5 / 88.2 / 92.7 | 68.7 / 88.4 / 92.9 |
| WRIVA 6 dev | 58.9 / 19.5 | 59.7 / 22.5 | 59.7 / 21.3 |

The submit fix returns botanical_garden to 74.8 (script) and 75.0 (agent) at 1 deg and adds a few
tenths elsewhere (exhibition_hall, delivery_area). **However, these agent numbers and all earlier
ones are not clean:** `rank_models` listed every model in the scene's cache, which is shared between
runs and controllers, so the agent could pick a model built by the scripted controller or an earlier
pass. It did so on WRIVA M07 in passes 11 and 12 (submitted models its episode never built, 81.3 and
78.8 against its own 81.0). On ETH3D both controllers follow the same recipe and build the same
artifact ids, so the exposure is small there; on WRIVA, where up to 48 models per scene accumulated
over ten passes, it is not. `rank_models` now lists only models the run built or touched; pass 13
re-runs the agent on both dev splits and the test split with that, and only pass 13 and later count
as agent results.

### 2026-11-03: pass 13 (agent with its own models only): the clean comparison

Agent re-run on both dev splits and the ETH3D test split with `rank_models` restricted to the
run's own models (scripted and fixed numbers are pass 12, since neither uses `rank_models`). Every
submitted model was checked against the episode's trace: all 31 were built inside their own episode.

| | fixed | scripted | agent (pass 13) |
|---|---|---|---|
| ETH3D 13 dev | 60.4 / 83.5 / 89.5 | 66.7 / 87.1 / 91.9 | 66.9 / 87.4 / 92.2 |
| ETH3D 12 test | 63.5 / 85.7 / 91.2 | 70.5 / 89.3 / 93.5 | 70.6 / 89.3 / 93.4 |
| ETH3D all 25 (paper 68.9 / 86.8 / 92.3) | 61.9 / 84.6 / 90.3 | 68.5 / 88.2 / 92.7 | 68.7 / 88.3 / 92.8 |
| WRIVA 6 dev | 58.9 / 19.5 | 59.7 / 22.5 | 59.7 / 22.5 |

What the leak was worth: on WRIVA the agent's BLH0001 drops from 52.0 (a foreign model) to 49.0
(the script's number) and M07 returns to 81.0, so the pass 12 WRIVA tie stands but the pass 11
agent lead (60.3) was the leak. On ETH3D nothing changes beyond tenths (facade 77.1 -> 76.1, bridge
78.6 -> 77.5 and pipes 69.5 -> 70.1, lecture_room 73.0 -> 73.6, all from the agent choosing a
different threshold of the same sweep). Per scene the agent now beats the script on 7 ETH3D scenes
(delivery_area +1.1, office +1.8, living_room +1.2, boulders +0.8, meadow, terrace, pipes,
lecture_room by a few tenths), loses on 3 (facade -1.0, bridge -1.1, kicker -0.1) and ties the rest;
on WRIVA it ties on every scene. All episodes finished on their own (no forced submit, no fallback)
in 8 to 25 tool calls.

So the honest state is: the agent and the scripted controller are the same system on these scenes.
Both beat the best fixed pipeline by 6 to 7 points at 1 deg on ETH3D and both are above FlexGlue's
paper numbers at 3 and 5 deg. The agent's tenths over the script come from picking a different
threshold out of the same sweep, which is within the noise of COLMAP's randomness. The scripted
controller is a ceiling the agent is tracking, not exceeding, because the playbook encodes the
script and the tools leave the agent nothing the script cannot also do. Beating the script needs
either a new capability the script lacks (the LaMAR partition / merge path is the first, and there
the gap between controllers will be real) or a second seed per scene to measure what is noise.

### 2026-11-03: LaMAR at scale (first run)

Adapter [datasets/lamar.py](src/agenticsfm/datasets/lamar.py): the CAB location's 6,587 iPhone
frames (22 sessions, per-frame pinhole intrinsics and poses) as one scene, and one 448-frame
session as a smoke test. The trajectories are camera-to-world (inverting them makes our model agree
with GT to 0.24 deg median between consecutive frames). The fixed pipeline on the 448-frame session
(NetVLAD k=100, 28k pairs, one GPU): 21 min matching, 36 min COLMAP, 447 of 448 images in one model,
but `auc_all@10` 14.5: the relative rotation error is 0.6 deg median at 10 frames apart and 107 deg
at 200 frames apart, with a p90 of 39 deg already at 10 frames, so the sequence contains at least
one wrong join (repetitive corridors), not drift. `optimize` lifts it to 16.6. This is the
doppelganger problem at scale and the target of the partition / merge tools; the first partition
run (4 parts of 86 to 162 images) exposed two bugs now fixed (concurrent mappings sharing one
SQLite file, and a mis-registered `merge_models`).

Scripted controller on the same session (partition path: parts of at most 150 images mapped
concurrently and merged, threshold sweep, verify, optimize; 80 min): **45.0** at 10 deg (30.4 at 3)
against 14.5 for the plain pipeline, 439 of 448 images in two models. The covisibility filter at
0.6 removed the wrong join; at 0.4 and 0.3 the parts shared too few registered images to be merged
(overlap raised from 15% to 30%, at least 10 images).

**The full 6,587-image CAB scene (22 sessions), 2026-11-03.** NetVLAD k=100 gives 454k pairs, of
which 296k verify; matching took 4.2 h on one GPU (7 kB per pair of matches), spectral partition
5 to 15 min, and mapping 68 parts of 60 to 190 images concurrently 3 h. Then three ways of getting
one model, all scored against the map session's GT trajectories (one global frame):

| model | registered | auc_all@10 / @3 |
|---|---|---|
| partition + merge, parts attached in size order (the code the run used) | 691 | not scored (50 of 68 parts shared 0 images with what had been merged) |
| partition + greedy merge through the overlaps (reconstruct_parts v3) | 4,990 | 0.8 / 0.5 |
| one global GLOMAP mapping, covisibility >= 0.6 | 6,583 | 2.7 / 0.4 |
| one global GLOMAP mapping, unfiltered | 6,583 | 1.7 / 0.5 |

The greedy merge fixed the attachment failure (90 of 116 units attached) but ninety sequential
similarity alignments through 5 to 10 shared cameras drift into a wrong model that the final
bundle adjustment cannot recover; on the 448-frame session it was better on two partitions and
worse on the one that mattered (threshold 0.6: 19.5 against 27.9). GLOMAP registers everything and
is right locally (0.7 deg median between consecutive frames) but wrong at range: 27 deg median
between frames 200 apart in the same session, 50 deg median across sessions, with per-session
self-consistency anywhere from 8% to 97%.

Why: a GT edge oracle (benchmark layer only: the median epipolar error of each pair's inlier
matches under the GT relative pose and GT intrinsics; true consecutive-frame pairs sit at 3 to
4 px) says the verified graph is mostly wrong. 59% of the verified pairs have a median error above
20 px, i.e. the two images do not see the same thing; among pairs with 300 or more inliers it is
still 15%, and among pairs FlexGlue scores at covisibility 0.9 or more it is 10%. No threshold
separates them: 0.6 keeps 96% of the true pairs and 45% of all pairs, of which roughly half are
still wrong. CAB is a building of identical corridors and offices, so geometric verification with
hundreds of inliers passes doppelganger pairs in bulk, within a session (at 300+ inliers, a pair
from the same walk more than 2 minutes apart is wrong 62% of the time) as much as across sessions.
Neither an incremental nor a global mapper survives a graph with a third of its strong edges
wrong, and no merge order fixes parts that are each folded inside.

Consequences. The scalability mechanism (partition, concurrent mapping, merge) works as plumbing
and is kept, with the greedy merge, but it is not what limits LaMAR: edge-level disambiguation at
scale is, which is a scoring problem the pair tools do not solve yet (the covisibility score ranks
true pairs above wrong ones only weakly here). The scripted controller's partition threshold was
raised from 400 to 1,000 images, because one COLMAP mapping of a 600-image WRIVA scene takes 45 min
and scores 74 to 80, which the merge path does not match yet. The original scripted run on the
full scene (old merge code) finishes with its own number in the results file; it is a run of the
plumbing, not a result to compare with.



### 2026-11-03: WRIVA held-out test split (partial) and the runs stopped for the change of direction

The 28 WRIVA test scenes with the fixed pipeline (COLMAP, ALIKED + FlexGlue, covisibility 0.6;
run `fixed_wriva_v12`): **65.1 / 37.9** `auc_all@10 / @3` over 28 scenes. The scripted controller
(run `scripted_v12`, LARGE_SCENE 1,000, three GPUs, largest scenes first) had finished 7 of the 28
when every reconstruction run was stopped on 2026-11-03 evening (below); the agent's test run
(`agent_v13`), queued behind it in `outputs/logs/wriva_test2.sh`, never started. On those 7 scenes
(3,128 images) the two controllers compare as on dev: the script adds under a point, mostly from
`optimize`, and the PTZ scenes are within noise.

| scene (images) | fixed `auc_all` @3 / 10 | scripted `auc_all` @3 / 10 |
|---|---|---|
| s04 A06 (500) | 46.8 / 79.6 | 47.1 / 79.7 |
| s00 A01 (300) | 52.4 / 83.3 | 55.9 / 84.4 |
| s01 A06_MultiZone (600) | 42.3 / 77.9 | 42.3 / 77.9 |
| s00 PTZ A06_FullTrailerPark (474) | 37.8 / 74.2 | 37.6 / 74.1 |
| s01 PTZ A06_FullTrailerPark (444) | 39.2 / 74.5 | 39.6 / 74.7 |
| s02 PTZ A06_FullTrailerPark (414) | 41.0 / 74.9 | 41.2 / 75.1 |
| s03 PTZ A06_FullTrailerPark (396) | 41.9 / 75.4 | 43.0 / 76.1 |
| **mean of 7** | **43.0 / 77.1** | **43.8 / 77.4** |

The models stay in `outputs/runs/{fixed_wriva_v12,scripted_v12}/<scene>/final`; the comparison is
`scripts/evaluate.py --dataset wriva --split test --run <name>=outputs/runs/<run>/{scene}/final`.

**Stopped on 2026-11-03 at about 22:50**, to start the work in [docs/DIRECTION.md](docs/DIRECTION.md):

- the scripted controller on the full LaMAR CAB scene (`outputs/logs/scripted_lamar_cab_ios.log`;
  18 h in; partition_graph 15 min, reconstruct_parts 2 h, examine, verify_poses done; killed in the
  second global bundle adjustment of the merged model, so it has no final number, which the
  greedy-merge analysis above makes moot anyway). Its matches (cache `26d6575684cd`, 454k pairs,
  4.2 GPU-hours) remain in the artifact store for reuse;
- the WRIVA test-split script above (scripted 7 of 28 scenes done, agent not started);
- a shared-intrinsics test (`optimize`) on CAB_ios.

The VLM server on GPUs 0 to 3 was left running.

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

## Plan

| Milestone | Content |
|---|---|
| M0 (done) | Evaluator matches GLUEMAP and COLMAP exactly; 8 WRIVA reference pipelines scored; per-scene oracle. |
| M1 (done 2026-11-02) | Harness core: workspace, artifact store with content-hash cache, backend runner across envs, trace with costs. Tools: `build_graph`, `match`, `score_pairs` (XDG, FlexGlue covis.), `filter_edges`, `reconstruct` (COLMAP, GLOMAP), `examine` v1 (stats, per-edge residuals, renders), `submit`. Fixed-pipeline controllers reproduce the table above from images. |
| M2 (first pass done 2026-11-02) | Agent loop with Qwen3.8 and the playbook; scripted controller; `inspect`, `verify_poses` (VGGT); first dev-set comparison. Answer so far: the scripted controller ties the best fixed pipeline, the agent is 3 points behind. |
| M3 (done 2026-11-02 / 03) | Repair: cycle consistency, `split_model` / `merge_models`, feedforward backends (VGGT, Pi3) for local reconstruction and bridging; GLUEMAP backends; `optimize`. |
| M4 (adapters and audits done 2026-11-03; ablations not run) | ETH3D and LaMAR adapters (no-harm check); ablations: text-only (no renders), thinking off, budget sweeps; cost accounting. |
| M5 (started 2026-11-03) | Open incremental solver ([docs/DIRECTION.md](docs/DIRECTION.md)): explicit scene state; `seed` / `frontier` / `localize` (PnP or feedforward with anchors) / `verify` / `refine` tools on pycolmap's BA primitives; a scripted grow controller first, then the agent as its scheduler; COLMAP becomes a baseline. Decision experiment: WRIVA dev, ETH3D dev, LaMAR CAB 448-frame session against COLMAP, GLOMAP, GLUEMAP. |
