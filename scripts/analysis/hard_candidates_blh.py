"""GT check of the hard-stage candidate placements (A/B) and final poses of p123_vlm_dev BLH0001 (benchmark layer, GT only for scoring)."""
import json, sys
import numpy as np
sys.path.insert(0, "/raid/home/gochen/cvpr_2027/agenticsfm/scripts/analysis")
from ff_offline import find_grow, largest, gt_errors, rot_angle
from ff_stability import sim_to_gt
scene = "t04_v09_s00_r04_VaryingAltitudes_BLH0001"
d, c, g = find_grow("p123_vlm_dev", scene)
W = g / "_work/scene0"
rec = largest(g); by = {im.name: im for im in rec.images.values() if im.has_pose}
gt = gt_errors(scene); (sg, Rg, tg), G = sim_to_gt(rec, gt)
gtc = np.array([im.projection_center() for im in G.values()]); gscale = float(np.median(np.linalg.norm(gtc - np.median(gtc, 0), axis=1)))
L = json.loads((g / "grow_log.json").read_text())
hard = {r["round"]: r for r in L["log"] if r.get("kind") == "hard"}

def gt_pose_err(Rcm, cm):
    """Pose in model frame (cam_from_model rotation, centre) -> error vs GT (deg, centre dist / GT scene scale)."""
    Rcg = Rcm @ Rg.T; cg = sg * Rg @ cm + tg
    return rot_angle(G_n.cam_from_world().rotation.matrix() @ Rcg.T), float(np.linalg.norm(cg - G_n.projection_center()) / gscale)

def align(E, names, anchors):
    use = [a for a in anchors if a in by and a in names]
    idx = {n: i for i, n in enumerate(names)}
    Ms = [by[a].cam_from_world().rotation.matrix().T @ E[idx[a]][:, :3] for a in use]
    U, _, Vt = np.linalg.svd(sum(Ms)); R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    src = np.array([-E[idx[a]][:, :3].T @ E[idx[a]][:, 3] for a in use]); dst = np.array([by[a].projection_center() for a in use])
    ms, md = src.mean(0), dst.mean(0); Rs = (R @ (src - ms).T).T
    s = float(np.sum((dst - md) * Rs) / np.sum(Rs * Rs)); t = md - s * R @ ms
    return s, R, t

for rnd, img in [(7, "image_000028.JPG"), (9, "image_000033.JPG")] + [(7, n) for n in ["image_000034.JPG","image_000013.JPG","image_000022.JPG","image_000012.JPG","image_000038.JPG"]] + [(9, n) for n in ["image_000032.JPG","image_000002.JPG","image_000036.JPG"]]:
    G_n = G[img]
    res = json.loads((W / f"ff_hard{rnd}/feedforward.result.json").read_text())["groups"]
    out = {}
    for tag, grp, anc in [("A", res[0], hard[rnd]["anchors_A"]), ("B", res[1], hard[rnd]["anchors_B"])]:
        E = np.array(grp["extrinsics"], dtype=float); names = grp["names"]
        s, R, t = align(E, names, anc)
        Ei = E[names.index(img)]; Rcm = Ei[:, :3] @ R.T; cm = s * R @ (-Ei[:, :3].T @ Ei[:, 3]) + t
        out[tag] = gt_pose_err(Rcm, cm)
    fin = gt_pose_err(by[img].cam_from_world().rotation.matrix(), by[img].projection_center()) if img in by else None
    info = hard[rnd]["images"][img]
    print(f"round {rnd} {img}: A {out['A'][0]:5.1f} deg / {out['A'][1]:.3f}   B {out['B'][0]:5.1f} deg / {out['B'][1]:.3f}   "
          f"final {'%.1f deg / %.3f' % fin if fin else 'not in model'}   [{info['why']}; status {L['summary'].get('images',{}).get(img) if isinstance(L['summary'].get('images'),dict) else ''}]")
errs = []
for n, im in by.items():
    G_n = G[n]; errs.append((n, *gt_pose_err(im.cam_from_world().rotation.matrix(), im.projection_center())))
errs.sort(key=lambda x: -x[2])
print("median rot %.1f, median centre %.3f" % (np.median([e[1] for e in errs]), np.median([e[2] for e in errs])))
print([(e[0][6:12], round(e[1],1), round(e[2],3)) for e in errs[:12]])
cen = np.median(gtc, 0)
for n in ["image_000002.JPG","image_000032.JPG","image_000038.JPG","image_000036.JPG","image_000034.JPG","image_000022.JPG","image_000012.JPG","image_000013.JPG","image_000028.JPG","image_000033.JPG","image_000001.JPG"]:
    im = by[n]; Gn = G[n]
    cg = sg * Rg @ im.projection_center() + tg; d = cg - Gn.projection_center()
    ax = Gn.cam_from_world().rotation.matrix()[2]  # optical axis in world
    print(n[6:12], "dist from centre %.2f" % (np.linalg.norm(Gn.projection_center()-cen)/gscale), "err %.3f" % (np.linalg.norm(d)/gscale),
          "along axis %.3f" % (d @ ax / gscale), "| status", L["summary"]["images"].get(n) if isinstance(L["summary"].get("images"), dict) else "")
