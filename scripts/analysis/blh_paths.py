"""Per-run path of every image of one scene: the round and route that registered it, and its GT rotation error in the
submitted model (GT only for offline analysis). Usage: blh_paths.py --runs RUN ... [--scene SCENE]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ff_offline import ROOT, find_grow, gt_errors, largest, align_gt  # noqa: E402


def auc(run, scene):
    for l in (ROOT / "outputs/runs" / run / scene / "trace.jsonl").read_text().splitlines():
        d = json.loads(l)
        s = json.dumps(d.get("summary", {}))
        if "auc_all@10" in s:
            return d["summary"].get("auc_all@10") or d["summary"].get("metrics", {}).get("auc_all@10")
    return None


def paths(L):
    out = {}
    for r in L["log"]:
        k = r.get("kind")
        if k == "seed" and isinstance(r.get("attempts"), list):
            for n in r["attempts"][r.get("chosen", 0)]["group"]:
                out.setdefault(n, (0, "seed"))
        if k in ("ff", "hard", "pnp") and isinstance(r.get("images"), dict):
            for n in r.get("registered", []) or []:
                out.setdefault(n, (r.get("round"), f"{k}: {r['images'].get(n, {}).get('why', '?')}"[:60]))
        if k == "vlm_place":
            for n in r.get("unverified", []) + r.get("registered", []):
                out.setdefault(n, (r.get("round"), "vlm_place"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scene", default="t04_v09_s00_r04_VaryingAltitudes_BLH0001")
    ap.add_argument("--images", nargs="*", help="image name suffixes to show (default: those whose outcome varies)")
    a = ap.parse_args()
    gt = gt_errors(a.scene)
    res = {}
    for run in a.runs:
        try:
            d, c, g = find_grow(run, a.scene)
        except Exception:
            continue
        L = json.loads((g / "grow_log.json").read_text())
        rec = largest(g)
        err = align_gt(rec, gt)
        res[run] = (auc(run, a.scene), rec.num_reg_images(), paths(L), err)
    names = sorted({im.name for im in gt.images.values()})
    show = [n for n in names if any(s in n for s in a.images)] if a.images else names
    for run, (au, nreg, p, err) in res.items():
        bad = sorted(((n, e) for n, e in err.items() if e > 5), key=lambda x: -x[1])
        print(f"\n== {run}: reg {nreg}  >5 deg: " + ", ".join(f"{n[-8:-4]}={e:.0f}" for n, e in bad))
        for n in show:
            if n in p or n in err:
                rd, why = p.get(n, ("-", "?"))
                print(f"    {n[-8:-4]}  r{rd!s:>3}  {'-' if n not in err else f'{err[n]:5.1f}'}  {why}")
            else:
                print(f"    {n[-8:-4]}  not registered")


if __name__ == "__main__":
    main()
