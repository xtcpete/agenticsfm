"""GT check of every VLM placement (Phase 3): the Pi3X pose of each pending image (logged as ff_pose), its render verdict,
the gate's decision. GT only for offline scoring. Usage: vlm_place_gt.py --runs RUN [RUN ...] [--scenes ...]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from ff_offline import find_grow, gt_errors, largest, rot_angle, umeyama  # noqa: E402

SCENES = ["t04_v09_s00_r04_VaryingAltitudes_BLH0001", "t09_v04_s00_r10_PTZ_A08", "t09_v04_s01_r10_PTZ_A08", "t09_v04_s02_r10_PTZ_A08",
          "t04_v05_s03_r02_VaryingAltitudes_A08", "t04_v07_s01_r02_VaryingAltitudes_M07_building_1_door"]


def similarity(rec, gt):
    G = {im.name: im for im in gt.images.values()}
    names = [im.name for im in rec.images.values() if im.has_pose and im.name in G]
    by = {im.name: im for im in rec.images.values()}
    src = np.array([by[n].projection_center() for n in names]); dst = np.array([G[n].projection_center() for n in names])
    scale = np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1))
    rng = np.random.default_rng(0); best = None
    for _ in range(300):
        idx = rng.choice(len(names), 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except Exception:
            continue
        inl = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1) < 0.05 * scale
        if best is None or inl.sum() > best[0].sum():
            best = (inl, s, R, t)
    s, R, t = umeyama(src[best[0]], dst[best[0]])
    return G, s, R, t, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    a = ap.parse_args()
    tally = {}
    for run in a.runs:
        for sc in a.scenes:
            try:
                d, c, g = find_grow(run, sc)
            except Exception:
                continue
            log = json.loads((g / "grow_log.json").read_text())
            rows = log if isinstance(log, list) else log.get("log", log.get("rounds", []))
            vp = [r for r in rows if isinstance(r, dict) and r.get("kind") == "vlm_place"]
            if not vp:
                continue
            G, s, R, t, scale = similarity(largest(g), gt_errors(sc))
            print(f"== {run} {sc}")
            for r in vp:
                print(f"  group {len(r['group'])}  picks {len(r.get('picks', []))}  gate {r.get('render_gate')}  layout {(r.get('layout') or {}).get('answer')}  {r.get('result', '')}")
                for n, inf in r["images"].items():
                    why = inf.get("why", "")
                    p = inf.get("ff_pose")
                    e = ce = None
                    if p and n in G:
                        Rm, tm = np.array(p["R"]), np.array(p["t"])
                        e = rot_angle(G[n].cam_from_world().rotation.matrix() @ (Rm @ R.T).T)
                        cm = s * (R @ (-Rm.T @ tm)) + t
                        ce = np.linalg.norm(cm - G[n].projection_center()) / scale
                    rd = inf.get("render", {})
                    dec = "admit" if "registered unverified" in why else ("confirmed" if "supported PnP" in why else "out")
                    ok = None if e is None else (e < 5 and ce < 0.1)
                    print(f"    {n[-8:]:>8}  ff {'-' if e is None else f'{e:5.1f}°'} c {'-' if ce is None else f'{ce:.3f}'}  {str(rd.get('verdict')):>12} cov {rd.get('cover')}  {dec:>9}  {why}")
                    if ok is not None and dec != "confirmed":
                        k = (dec, "correct" if ok else "wrong")
                        tally[k] = tally.get(k, 0) + 1
    print("\ntally (unconfirmed images; correct = rotation < 5° and centre < 0.1 of camera spread):")
    for k in sorted(tally):
        print("  ", k, tally[k])


if __name__ == "__main__":
    main()
