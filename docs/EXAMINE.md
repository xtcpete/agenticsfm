# Examine: how a person judges a reconstruction, as questions an agent can answer

The feedback loop is the product. A mapper that cannot tell a right model from a wrong one grows
the wrong one with confidence; the open solver's acceptance test decides what the agent can and
cannot fix. This note states how a person evaluates a reconstruction without ground truth, and
turns each habit into a check that is either computed or asked of the VLM with evidence it can
actually read.

## What a person does

1. **Looks through a camera.** Pick an image, stand where the model says its camera stands, and
   compare what the model shows from there with the photo. If the walls, roofs and ground of the
   model land on the walls, roofs and ground of the photo, the camera is right. If the photo shows
   a facade and the model, from that pose, shows a different facade, a corner or empty space, the
   camera is wrong, however many matches supported it.
2. **Looks at the whole thing from above and from the side.** Two copies of one building, a facade
   mirrored onto its twin, cameras inside walls or under the ground, a flight line that breaks and
   restarts elsewhere: a fold or a misplacement shows in the layout before any statistic.
3. **Checks two photos the model says overlap.** If the model claims two cameras see the same
   wall, the two photos must show the same wall, and the points they share must sit on the same
   physical details in both.
4. **Asks why an image was left out.** A photo of the site that the model could not place is a
   defect; a photo of somewhere else is not. The person looks at the image and at the model's
   images and decides which it is.
5. **Weighs coverage against correctness.** A model with 32 correct cameras beats one with 39
   cameras and three wrong ones, because a wrong camera also drags its neighbours in bundle
   adjustment.

## What the harness has, and where it is blind

| Check | GT-free signal in the harness | Blind spot |
|---|---|---|
| matches agree with poses | `examine`: epipolar consistency of every verified pair; per-image match support | the matches themselves: lookalike (doppelganger) pairs are wrong *and* self-consistent, so a correct pose fails the test and a wrong pose can pass it. On BLH0001 every placed image is correct by GT while `strong_inconsistent_frac` reads 0.088, and the seven images that matter cannot be judged at all |
| a second model agrees | `verify_poses`: VGGT on the image and its neighbours, rotation disagreement > 15 deg, corroborated by match support | the corroboration is match support again; precision 0.7 to 0.8 and recall 0.5 |
| the model looks right | renders (top, side) in a gravity-aligned frame | nobody reads them yet except the agent, and a render cannot show whether *this* camera is right |
| coverage | registered count, components, unplaced list | says nothing about whether the unplaced images belong |

## Examine v2: render and compare

The first habit, formulated: **is photo i consistent with what the model says camera i sees?**

- Take the neighbours j of camera i in the model (nearest centres, similar viewing direction).
- Give each neighbour dense depth from the feedforward model (Pi3X on the image and its
  neighbours), scaled to the model with the sparse points j observes (median ratio).
- Warp j's pixels into i through the model's poses and intrinsics; compare j's colours with photo
  i at the pixels they land on (normalised cross-correlation on a coarse grid, plus the overlap).
- Score the placed pose **against its own perturbations** (yaw 5, 15 and 30 deg, shifts of 5% and
  15% of the scene span): a correct pose is a clear maximum; a wrong pose is flat or lower than a
  perturbation. The margin, not the absolute value, is the verdict, because texture and overlap
  set the absolute level per image.

Prototype on BLH0001 (model with 32 GT-correct cameras, 10 random images, 4 neighbours each):

| image | correct | yaw 5 | yaw 15 | yaw 30 | shift 5% | shift 15% | a neighbour's pose |
|---|---|---|---|---|---|---|---|
| 22 | 0.62 | 0.34 | -0.09 | -0.14 | 0.25 | 0.23 | 0.42 |
| 30 | 0.85 | 0.60 | 0.39 | 0.23 | 0.58 | 0.42 | 0.20 |
| 18 | 0.73 | 0.66 | 0.47 | 0.38 | 0.52 | 0.52 | 0.25 |
| 15 | 0.95 | 0.70 | 0.55 | 0.39 | 0.60 | 0.46 | 0.92 |
| 9 | 0.96 | 0.76 | 0.58 | 0.50 | 0.71 | 0.61 | 0.58 |
| 3 | 0.79 | 0.79 | 0.70 | 0.26 | 0.61 | 0.27 | 0.70 |
| 1 | 0.70 | 0.50 | 0.26 | -0.12 | 0.41 | 0.16 | 0.49 |
| 10 | 0.93 | 0.71 | 0.63 | 0.51 | 0.62 | 0.49 | 0.16 |
| 7 | 0.86 | 0.76 | 0.54 | 0.56 | 0.80 | 0.65 | 0.49 |
| 4 | 0.92 | 0.76 | 0.63 | 0.45 | 0.69 | 0.53 | 0.69 |

The correct pose is the maximum on every image; 5 deg of yaw costs 0.05 to 0.28; 15 deg costs
0.1 to 0.7. A neighbour's pose scores high only when the neighbour stands almost at the same spot
(image 15), which is the case where the two poses are nearly the same anyway. This signal needs no
matches, so it judges exactly the images match support cannot.

What the VLM reads, when the number is ambiguous: a three-panel figure per image, the photo, the
model rendered from the camera (neighbours' pixels warped in), and the overlay. The question is
one a person answers at a glance: *do the structures in the rendering lie on the same structures
in the photo, or is the model showing a different place or a shifted copy?* That is image
comparison, which the VLM does well, not pose estimation, which it does not.

## The other habits, as checks

- **Layout (habit 2).** The gravity-aligned top and side renders, with the question: *is there a
  duplicated structure, a camera inside or under the structure, a group of cameras detached from
  the rest?* Asked once per model, answers logged, acted on by re-running the hard-image stage
  for the flagged cameras.
- **Pairs (habit 3).** For the strongest pairs the model marks inconsistent, the pair drawn with
  its matches: *do the lines connect the same physical details?* Decides whether an inconsistent
  pair is a wrong match (cut the edge) or a wrong pose (re-place the image).
- **Left-out images (habit 4).** `same_scene` on a sample of the model's images and the unplaced
  images: *same site, or a different one?* Same site: the hard-image stage tries again with other
  anchors; different: a separate model, returned as such.
- **Coverage vs correctness (habit 5).** The quality rule already orders models this way; the
  per-image render-and-compare table makes the trade explicit per image.

## Where it goes in the loop

1. Every placement of a hard image is accepted on render-and-compare (margin over perturbations),
   with the two-anchor-set agreement and the advisor as the arbiter when the margin is thin.
2. After the final refinement the whole model is audited image by image; cameras below margin are
   re-placed by the hard-image stage or dropped; the table and the figures are the `examine`
   output the agent reads.
3. The VLM's verdicts are logged with their evidence so they can be scored against GT in the
   benchmark layer, as the match-based rules were.

## Calibration (2026-11-04)

302 GT-correct cameras of five scenes (BLH0001, M07, A08, terrace, delivery_area; the v1 models), each scored
at its model pose and at four wrong poses (yaw 10 and 20 degrees, a 10% shift, a far camera looking the same
way); 1,510 scored poses. Score is the NCC at the pose; margin is the score minus the best of the pose's own
small perturbations.

| rule | keeps of GT-correct | passes of 20 deg / far-wrong | passes of 10 deg / 10% wrong |
|---|---|---|---|
| score >= 0.3 | 0.99 | 0.26 | 0.53 |
| score >= 0.5 | 0.92 | 0.11 | 0.27 |
| margin > -0.05 | 0.98 | 0.26 | 0.21 |
| margin > 0 | 0.96 | 0.11 | 0.12 |
| margin > 0.05 | 0.88 | 0.02 | 0.04 |

So the verdict is: consistent above margin 0.05, inconsistent below -0.05 or below score 0.2, ambiguous in
between. It is a precision test: 10 degrees is caught three times out of four. What it is not is a test for a
coarse feedforward placement: on BLH0001 the correct Pi3X placements of the hard images (2 to 3 degrees and a few
percent of span off before any refinement) scored like wrong ones, and using the verdict to reject them (v1.1)
removed correct cameras. Until a tolerant variant exists (coarse grid, photometric pose refinement before
scoring), the verdict reports and the acceptance of hard images rests on the agreement of two anchor sets.

**The VLM on the figures.** Right on all five BLH0001 figures that had content, and "neither" on an empty one.
Wrong on a mixed-scene test (M07 plus terrace images in one folder): shown a terrace facade with a thin scatter of
M07 points rendered over it, it answered "A" twice. The lesson is the order of evidence: cheap gates first (a
hard candidate needs real links to the model, and no advisor pick may override a render verdict of inconsistent),
the VLM only on figures that pass them, and the prompt asks it to say what the rendering depicts before judging.
