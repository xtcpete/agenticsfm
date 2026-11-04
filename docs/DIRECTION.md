# Direction (2026-11-03): from choosing tools to owning the incremental solver

Written after pass 13 (see the README's results log) and a reading of
[Solv3R](https://arxiv.org/abs/2611.02857). This note records why the project changes direction,
what the new system is, what is kept from the current harness, and the experiment that decides
whether the change is right.

## Where pass 13 left us

| | fixed pipeline | scripted controller | agent (pass 13) |
|---|---|---|---|
| ETH3D 25 scenes, `colmap_auc` @1 / 3 / 5 | 61.9 / 84.6 / 90.3 | 68.5 / 88.2 / 92.7 | 68.7 / 88.3 / 92.8 |
| WRIVA 6 dev scenes, `auc_all` @10 / @3 | 58.9 / 19.5 | 59.7 / 22.5 | 59.7 / 22.5 |

The agent and the scripted controller are the same system. Both beat every fixed pipeline, both
beat the FlexGlue paper's own pipeline on ETH3D at 3 and 5 degrees with the paper's matcher, but
the agent only tracks the script. The reason is structural: the playbook encodes the script, and
every tool the agent has is a tool the script can call with the same arguments. The agent's
decisions are parameters around a black box (COLMAP's incremental mapper), so there is nothing for
it to decide that a rule cannot decide equally well.

LaMAR CAB (6,587 images) added a second limit: 59% of the geometrically verified pairs are wrong
under the GT edge oracle, and no mapper, incremental or global, survives a graph with a third of
its strong edges wrong. Correctness has to be enforced while the model grows, not by filtering the
graph beforehand or by choosing between finished models afterwards.

## What Solv3R shows

Solv3R is a learned looped transformer, fine-tuned from NVIDIA's DéjàView, that applies one update
operator to a multi-view state until the relative change of the state drops below 0.02. Its three
trained properties are the ones an incremental reconstruction needs: a solved state stays solved
under further iteration, a perturbed state is corrected, and new views can join an existing state.
Solved views are cached once (2.86 MB per view) and reused as context.

| Setting | Solv3R | Reference |
|---|---|---|
| Incremental 8 to 64 views, 8 solver calls per insertion, AUC@10 | 92.2 continuing from the state | 84.8 re-solving from scratch |
| Offline ETH3D pose AUC@3 | 70.4 | COLMAP + FlexGlue, about 87 at 3 degrees (different pair protocol) |
| Offline scale | about 100 images | VGGT runs out of memory at 320 |

Its stated failure case is our problem: two groups of views with little overlap, and the solver
places the second group at a wrong position and never recovers through call 64. It has no
correspondences and no way to say "I cannot place this image". Code was not released as of
2026-11-03; DéjàView's code and weights are public (github.com/nv-tlabs/dvlt, non-commercial).

## The idea: a coding agent as the iterative solver

The proposal (user, 2026-11-03): a feature matcher finds initial views, a feedforward model
initialises a scene, then the agent iteratively adds images (infers where an image roughly is,
places it with the feedforward model or with code written against the scene), re-optimises, and
finishes with bundle adjustment.

Assessment, piece by piece:

- **The agent must not place images itself.** VLMs are near chance at relative camera pose: in
  the "Lost in Space?" benchmark ([arXiv 2601.22228](https://arxiv.org/abs/2601.22228)) the best
  model reaches macro-F1 0.66 against a 0.50 random baseline and its answers flip when the two
  images are swapped. Our own agentic_dg result was 0.86 accuracy on pair verdicts against XDG's
  0.92. Location must come from retrieval, matches and PnP, or from the feedforward model.
- **A per-image loop is too slow and in the wrong place.** One VLM decision costs 30 to 60 s; a
  150-image scene becomes two hours and a 600-image scene a day, while a mapper registers an
  image in seconds. The loop has to advance by groups of 10 to 40 images.
- **Coding agents fail at exactly the three things an incremental loop must get right.**
  LEGO-Anything ([arXiv 2609.36380](https://arxiv.org/abs/2609.36380)), the closest coding-agent
  reconstruction paper, lists weak initialisation, regressive edits and unreliable
  self-evaluation. Pass 13 says the same from the other side.
- **"Code that defines a rough pose from the scene and the image"** is PnP of the new image
  against the current model's points through the matches already in the database. pycolmap 4.2.1
  exposes it (`IncrementalMapper.register_next_image`, `find_next_images`,
  `estimate_and_refine_absolute_pose`, local bundle adjustment); the harness has no such tool yet.
- **Feedforward initialisation, correspondence BA at the end**: agreed, and the evidence supports
  it. VGGT gains 14 AUC points on IMC from BA; GLUEMAP and MERG3R both finish with global BA;
  offline, feedforward solvers remain well behind correspondence SfM on ETH3D.

## The system we build: an open incremental solver

Treat the reconstruction as an explicit, inspectable state and grow it by groups. The agent is
the scheduler of the loop; every pose comes from a tool, as before.

**State.** Poses and intrinsics, tracks and points, per-image status (placed / deferred /
rejected, with the evidence), the view graph with edge scores, the scene frame. A plain Python
object in the workspace, exported to COLMAP format only for evaluation and baselines.

**Step operators (tools).**

1. `seed`: feedforward solve (VGGT, Pi3, DéjàView) of a seed cluster chosen by retrieval density.
2. `frontier`: the unplaced images with the strongest links to the current model, grouped, with
   anchor images already in the model.
3. `localize`: place a group. Two solvers: PnP on existing matches, then triangulation and local
   BA, when the matches to the model are strong; the feedforward model with anchors plus
   similarity alignment (today's `reconstruct(backend=vggt)` + `merge_models`) when they are weak.
   This is the "join" case where Solv3R fails and COLMAP cannot initialise.
4. `verify`: the corroborated rule (feedforward rotation disagreement > 15 deg AND match support
   < 0.5; precision 0.7 to 0.8 on both benchmarks) plus `examine`'s strong inconsistent fraction.
5. `accept` / `defer` / `reject`; a rejected group that conflicts with an accepted one means an
   edge to cut, not an image to drop (the BLH0001 lesson).
6. `refine`: local BA around the new group; global BA periodically and at the end. Owning the
   optimiser allows prior terms COLMAP's mapper cannot take: feedforward relative poses and
   depth as soft residuals (as GLUEMAP's augmented BA and MP-SfM do), which is what makes a
   low-overlap join hold under BA instead of drifting apart.
7. Stop when no group can be placed with evidence.

**Policy.** Which seed, which frontier next, which solver, when to cut an edge, when to run global
BA, when to stop; plus `analyze` for scene-specific hypotheses written as code against the state.
The scripted controller runs the same loop with fixed rules and remains the main ablation.

## What "moving away from COLMAP" means

Away from COLMAP's incremental mapper and the hidden loop inside it. Not away from correspondence
bundle adjustment: every system that drops correspondences pays for it on our data (GLUEMAP 44.6
against COLMAP 57 to 66 `auc_all@10` on WRIVA; Solv3R 70.4 against about 87 AUC@3 on ETH3D; GLOMAP
7 to 17 against 60 on the WRIVA PTZ scenes). Calling Ceres through pycolmap is using a library.

| Piece | Today | Proposal |
|---|---|---|
| Mapper loop: init pair, next image, register, filter, retriangulate | COLMAP binary, opaque | ours; each step a tool with inspectable input and output |
| Initialisation | two-view, COLMAP's heuristic | feedforward solve of a seed cluster |
| Registration | inside COLMAP | `localize`: PnP on matches, or feedforward with anchors |
| Outlier handling | reprojection-only filtering | verification by independent evidence; cut edges, defer, split |
| Scene state | COLMAP database and binary model | explicit Python state; COLMAP format as export |
| Triangulation and BA | pycolmap | pycolmap primitives (`IncrementalTriangulator`, `ObservationManager`, `BundleAdjuster`); pyceres (source in `~/gluemap/pyceres-src`) for custom residuals |
| Matching, retrieval, pair scoring, verify, merge, examine, renders | ours | unchanged |

Two things make this more than a rewrite: an explicit Python state is what a coding agent can
reason over and write code against, and an owned optimiser can take prior terms.

## Risks

- **Regression on easy scenes.** COLMAP's mapper is ten years of heuristics; v0 will start worse
  on ETH3D. ETH3D dev stays the do-no-harm gate and COLMAP becomes a baseline, not a backend.
- **Engineering.** Missing: the state object, `localize` with PnP, track building and
  triangulation scheduling, local BA around new images, retriangulation. Existing: the
  feedforward backend, `merge_models`, `frame`, `verify_poses`, `examine`, cycle consistency,
  partitioning.
- **Scale.** The state must hold 6,587 images for LaMAR; poses and tracks are small and the
  feedforward model only ever sees a group plus anchors.

## The experiment that decides it

A v0 mapper on the dev scenes: seed with VGGT on the 20 to 30 densest images, grow by groups with
PnP, triangulate, local BA, verify each group, global BA at the end. Compare with COLMAP, GLOMAP
and GLUEMAP on WRIVA dev (6 scenes), ETH3D dev (13 scenes) and the 448-frame LaMAR CAB session
(fixed 14.5, partition + merge 45.0 `auc_all@10`).

- Within a few points of COLMAP on ETH3D and ahead on LaMAR and BLH0001: the direction is
  validated; the agent goes on top as the scheduler, the script is the ablation.
- Ten points behind on ETH3D: keep COLMAP as one backend and own only the join and repair steps.

## Related work, for positioning

GLUEMAP ([arXiv 2605.26103](https://arxiv.org/abs/2605.26103)): feedforward star solves plus
global averaging and BA, no verification, not incremental. MERG3R
([arXiv 2603.02351](https://arxiv.org/abs/2603.02351)): divide-and-conquer clusters plus global
BA. MP-SfM ([arXiv 2504.20040](https://arxiv.org/abs/2504.20040)): monocular priors inside
COLMAP's loop. PoseAgent ([arXiv 2609.38755](https://arxiv.org/abs/2609.38755)): an agent that
orchestrates pair estimators with a learned verifier, up to +4.2 AUC@5. Solv3R: learns the loop,
cannot reject. None does verified incremental growth with a feedforward joiner and a
correspondence finisher.
