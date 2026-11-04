# Lessons (as of 2026-11-03)

Rules and findings that constrain the next changes. Each was paid for; the README's results log
has the evidence.

## How to run experiments

- **Never edit the source while a pass runs.** Backends and `analyze` snippets are subprocesses
  that read the files at call time; pass 6 lost three scenes to a temporary bug and one episode
  then invented model ids.
- **Ground truth only in the benchmark layer.** The harness `Scene` type has no GT field;
  `scripts/evaluate.py` and the `eval_*.py` audits are the only readers. Tool rules are set from
  GT audits over all dev models, never from one scene.
- **Judge every rule change on both dev splits** (WRIVA 6 scenes, ETH3D 13 scenes) before
  adopting it; a rule that helps one benchmark and hurts the other is not adopted (the
  inlier-weighted inconsistency ranks ETH3D candidates best and WRIVA candidates worst).
- **Only pass 13 and later agent numbers count.** Earlier passes could pick models built by other
  runs through `rank_models` (the cache is shared per scene); the pass 11 WRIVA lead was that leak.
- **One seed is noise.** COLMAP's randomness moves a scene by several points (BLH0001: 46 to 52
  on the same input). Differences under a point between controllers are not results.
- **GPU budget**: all 8 A100s may be used; 0 to 3 hold the vLLM server (`scripts/serve_vlm.sh`),
  4 to 7 are tool workers.

## What the data taught

- **Correctness over coverage.** Scenes contain distractors and doppelgangers; "every image
  registered" is not the goal. `auc_all` charges a dropped image exactly as much as a misplaced
  one, so drop only on a corroborated verdict.
- **The agent diagnoses well and decides badly**, and a script with the same tools decides as
  well as the agent. Beating the script needs a capability the script lacks, not a better prompt
  (see `DIRECTION.md`).
- **GLOMAP is a trap on WRIVA.** It registers everything, looks best to every match-based
  statistic (it fits exactly the pairwise constraints those statistics measure), and scores 1 to
  9 points lower than COLMAP at equal coverage; catastrophic on PTZ scenes (7 to 17 against 60).
  The quality rule penalises it by 0.10.
- **Reprojection error does not predict pose accuracy** (rank correlation -0.09 over models).
- **Intrinsics are the main precision lever** on WRIVA: `optimize` (shared intrinsics per size
  and focal cluster, OPENCV, BA) gains +1 to +12 on most scenes; GT intrinsics would lift M07 from
  77 to 87.
- **Misplaced images need two independent signals.** VGGT rotation disagreement > 15 deg AND
  match support < 0.5: precision 0.7 to 0.8 on both benchmarks. Either alone: 0.05 to 0.36.
  VGGT's translation-direction disagreement is noise as a signal (precision about 0.1).
- **Folds need edge removal, not image dropping.** On BLH0001 lowering the covisibility threshold
  until the graph connects (0.3) gave 57.6; dropping VGGT-flagged images gave 37.
- **Sub-model ownership.** COLMAP keeps abandoned initialisations as extra sub-models; an image
  belongs to the largest model that registers it, and `submit` drops redundant sub-models (COLMAP's
  set-based metric scores every sub-model).
- **Cycle consistency at 30 deg** is the best edge signal on large scenes and useless on BLH0001,
  whose doppelganger edges form self-consistent cycles.
- **LaMAR CAB is limited by doppelganger edges at scale, not by mapping scale.** 59% of verified
  pairs are wrong under the GT epipolar oracle, 15% of pairs with 300+ inliers, 10% of pairs at
  covisibility >= 0.9; no threshold separates them. Partition, concurrent mapping and merge work as
  plumbing; sequential similarity merges drift.
- **Model choice is not the bottleneck**: the GT-free quality rule picks within 1 to 1.5 points
  of the GT-best candidate; the candidates are what limit the result.
