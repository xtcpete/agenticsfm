"""Offline Solv3R vs Pi3X test, step 3 (agenticsfm env): score against GT.

Relative-pose errors over image pairs (rotation angle of the relative rotation; angle between relative translation
directions), AUC of max(rot, trans) at 3/10/30 degrees, and the focal error. Compared on the same pairs:
  groups:  Pi3X (as run) vs Solv3R (same groups; ray head and camera head)
  scene:   Solv3R joint 16 / 32 calls, Solv3R any-order incremental, and the final p22 model (all pairs of GT images;
           the model's unregistered images count as failures at 180 deg)

usage: python scripts/analysis/solv3r_eval.py --prep <dir> --out <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def to_cfw(c2w):
    c2w = np.asarray(c2w)
    R = np.transpose(c2w[:, :3, :3], (0, 2, 1))
    t = -np.einsum("nij,nj->ni", R, c2w[:, :3, 3])
    return R, t


def pair_err(Ra, ta, Rb, tb, idx):
    """Pairs (i, j): relative pose j<-i in prediction a vs reference b."""
    i, j = idx[:, 0], idx[:, 1]
    rel = lambda R, t: (R[j] @ np.transpose(R[i], (0, 2, 1)), t[j] - np.einsum("nij,nj->ni", R[j] @ np.transpose(R[i], (0, 2, 1)), t[i]))
    Rpa, tpa = rel(Ra, ta)
    Rpb, tpb = rel(Rb, tb)
    er = ang(np.transpose(Rpb, (0, 2, 1)) @ Rpa)
    na, nb = np.linalg.norm(tpa, axis=1), np.linalg.norm(tpb, axis=1)
    et = np.degrees(np.arccos(np.clip(np.sum(tpa * tpb, 1) / np.maximum(na * nb, 1e-12), -1, 1)))
    return np.maximum(er, et), er


def auc(err, thr):
    err = np.sort(np.minimum(err, 180))
    rec = np.arange(1, len(err) + 1) / len(err)
    e = np.r_[0, err]; r = np.r_[0, rec]
    last = np.searchsorted(e, thr)
    x = np.r_[e[:last], thr]; y = np.r_[r[:last], r[last - 1]]
    return 100 * np.trapezoid(y, x) / thr


def gt_arrays(gt, names):
    R = np.array([gt[n]["R"] for n in names]); t = np.array([gt[n]["t"] for n in names])
    return R, t


def score(names, R, t, gt, ok=None):
    """All pairs among names; ok[k] False -> that image counts as failed (180)."""
    Rg, tg = gt_arrays(gt, names)
    n = len(names)
    idx = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
    e, er = pair_err(R, t, Rg, tg, idx)
    if ok is not None:
        bad = ~(ok[idx[:, 0]] & ok[idx[:, 1]])
        e[bad] = 180; er[bad] = 180
    return e, er


def fmt(e):
    return f"{auc(e, 3):5.1f} {auc(e, 10):5.1f} {auc(e, 30):5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tot = {}
    rows = []
    for f in sorted(Path(a.out).glob("*.json")):
        res = json.load(open(f)); prep = json.load(open(Path(a.prep) / f.name))
        gt = prep["gt"]; sc = res["scene"]
        acc = {k: [] for k in ("pi3x", "s3r_rays", "s3r_cam")}
        foc = {k: [] for k in acc}
        for gp, gs in zip(prep["groups"], res["groups"]):
            names = [n for n in gp["names"] if n in gt]
            if len(names) < 2:
                continue
            k = [gp["names"].index(n) for n in names]
            E = np.asarray(gp["E"])[k]
            Kp = np.asarray(gp["K"])[k]; tr = np.asarray(gp["transforms"])[k]
            e, _ = score(names, E[:, :, :3], E[:, :, 3], gt)
            acc["pi3x"].append(e)
            foc["pi3x"] += [abs(0.5 * (Kp[q][0][0] / tr[q][0] + Kp[q][1][1] / tr[q][1]) / gt[n]["f"] - 1) for q, n in enumerate(names)]
            for key in ("rays", "cam"):
                R, t = to_cfw(np.asarray(gs[key]["c2w"])[k])
                e, _ = score(names, R, t, gt)
                acc[f"s3r_{key}"].append(e)
                K = np.asarray(gs[key]["K"])[k]
                foc[f"s3r_{key}"] += [abs(0.5 * (K[q][0][0] + K[q][1][1]) / gt[n]["f"] - 1) for q, n in enumerate(names)]
        line = {"scene": sc[:38], "groups": len(res["groups"])}
        for key in acc:
            if acc[key]:
                e = np.concatenate(acc[key]); line[key] = (auc(e, 3), auc(e, 10), auc(e, 30), float(np.median(foc[key])))
                tot.setdefault(("grp", key), []).append(e)
        for key in ("joint16", "joint32", "anyorder"):
            if key in res:
                names = res[key]["names"]
                R, t = to_cfw(res[key]["rays"]["c2w"])
                e, _ = score(names, R, t, gt)
                K = np.asarray(res[key]["rays"]["K"])
                line[key] = (auc(e, 3), auc(e, 10), auc(e, 30), float(np.median([abs(0.5 * (K[q][0][0] + K[q][1][1]) / gt[n]["f"] - 1) for q, n in enumerate(names)])))
                tot.setdefault(("scn", key), []).append((sc, e))
        if "joint16" in res:
            names = res["joint16"]["names"]; m = prep["model"]
            ok = np.array([n in m for n in names])
            Rm = np.array([m[n]["R"] if n in m else np.eye(3) for n in names]); tm = np.array([m[n]["t"] if n in m else np.zeros(3) for n in names])
            e, _ = score(names, Rm, tm, gt, ok)
            line["p22"] = (auc(e, 3), auc(e, 10), auc(e, 30), float(np.median([abs(m[n]["f"] / gt[n]["f"] - 1) for n in names if n in m])))
            tot.setdefault(("scn", "p22"), []).append((sc, e))
        line["timing"] = res.get("timing", {})
        rows.append(line)
    hdr = "AUC@3 @10 @30 medFocErr"
    print("== same Pi3X groups (pairs within each group) ==", hdr)
    for r in rows:
        print(f"{r['scene']:40s} n={r['groups']:3d} | " + " | ".join(f"{k}: {r[k][0]:5.1f} {r[k][1]:5.1f} {r[k][2]:5.1f} f{100*r[k][3]:4.1f}%" for k in ("pi3x", "s3r_rays", "s3r_cam") if k in r))
    print("== whole scene (all pairs of GT images) ==")
    for r in rows:
        print(f"{r['scene']:40s} | " + " | ".join(f"{k}: {r[k][0]:5.1f} {r[k][1]:5.1f} {r[k][2]:5.1f} f{100*r[k][3]:4.1f}%" for k in ("p22", "joint16", "joint32", "anyorder") if k in r)
              + f" | t {r['timing'].get('joint16', 0):.0f}s {r['timing'].get('joint_peak_gb', 0):.1f}GB")
    print("== pooled ==")
    for (kind, key), v in sorted(tot.items()):
        if kind == "grp":
            e = np.concatenate(v)
            print(f"groups {key:10s} AUC {fmt(e)}  pairs {len(e)}")
        else:
            for ds, sel in (("wriva", lambda s: s.startswith("t0")), ("eth3d", lambda s: not s.startswith("t0"))):
                vals = [(auc(e, 3), auc(e, 10), auc(e, 30)) for s, e in v if sel(s)]
                if vals:
                    m = np.mean(vals, 0)
                    print(f"scene  {key:10s} {ds}: mean per-scene AUC {m[0]:5.1f} {m[1]:5.1f} {m[2]:5.1f} ({len(vals)} scenes)")


if __name__ == "__main__":
    main()
