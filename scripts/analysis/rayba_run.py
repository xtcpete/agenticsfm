"""Run the ray-anchored GPU bundle adjustment on problems exported by rayba_prep.py (any env with torch).

Variants: --variants plain prior10 ... (plain = no depth prior; priorX = log-depth prior with lambda X).
Saves per scene and variant the refined poses and intrinsics groups, the cost log and the wall time.

usage: python scripts/analysis/rayba_run.py --prep <dir> --out <dir> [--scenes ...] [--variants plain prior10]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agenticsfm.solver.rayba import Options, Problem, RayBA  # noqa: E402


def load(f, dev, model="simple_radial"):
    z = np.load(f)
    order = np.argsort(z["p_o"], kind="stable")
    T = lambda a, dt=torch.float64: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)  # noqa: E731
    L = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=dev)  # noqa: E731
    pb = Problem(R=T(z["R"]), t=T(z["t"]), c=T(z["c"]), sc=T(z["sc"]), group=L(z["group"]),
                 intr=Problem.intr_from_simple_radial(T(z["logf"]), T(z["k"]), model),
                 fixed=torch.as_tensor(z["fixed"], device=dev), img_p=L(z["img_p"]), x_p=T(z["x_p"]), logd=T(z["logd"]),
                 prior_logd=T(z["prior_logd"]), prior_conf=T(z["prior_conf"]), p_o=L(z["p_o"][order]), tgt=L(z["tgt"][order]),
                 xt=T(z["xt"][order]), s=T(z["s0"]), model=model)
    return pb, z


def merge_groups(pb, z, tol, size_tol=0.01):
    """Merge intrinsics groups whose refined focal lengths agree within tol (relative) and whose image sizes agree
    within size_tol (undistorted crops of one camera differ by a few pixels; principal points stay per image).
    Single linkage over groups sorted by focal length. GT-free."""
    G = pb.intr.shape[0]
    grp = pb.group.cpu().numpy()
    present = [g for g in range(G) if (grp == g).any()]
    wh = {g: np.asarray(z["WH"][grp == g][0], dtype=float) for g in present}
    f = pb.intr[:, 0].exp().cpu().numpy()
    order = sorted(present, key=lambda g: f[g])
    clusters = []
    for g in order:
        for c in clusters:
            if any(abs(f[g] - f[h]) / f[h] <= tol and np.all(np.abs(wh[g] - wh[h]) / wh[h] <= size_tol) for h in c):
                c.append(g); break
        else:
            clusters.append([g])
    new = {g: k for k, c in enumerate(clusters) for g in c}
    cnt = np.bincount(grp, minlength=G)
    intr = torch.zeros(len(clusters), pb.intr.shape[1], dtype=pb.intr.dtype, device=pb.intr.device)
    wsum = torch.zeros(len(clusters), dtype=pb.intr.dtype, device=pb.intr.device)
    for g, ng in new.items():
        intr[ng] += pb.intr[g] * float(cnt[g]); wsum[ng] += float(cnt[g])
    pb.intr = intr / wsum[:, None]
    pb.group = torch.as_tensor(np.array([new[g] for g in grp]), dtype=torch.long, device=pb.group.device)
    print(f"  merged intrinsics groups {len(present)} -> {len(clusters)}", flush=True)
    return len(present), len(clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--variants", nargs="*", default=["plain", "prior10", "prior100"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cauchy", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    for f in sorted(Path(a.prep).glob("*.npz")):
        sc = f.stem
        if a.scenes and sc not in a.scenes:
            continue
        for v in a.variants:
            dst = out / f"{sc}.{v}.npz"
            if dst.exists():
                continue
            # variant tokens: plain | prior<lambda>, opencv, c<cauchy>, it<iters>, r<rounds>, o<outlier px between rounds>
            tok = v.split("_")
            model = ("opencv" if "opencv" in tok else "simple_radial") + ("_pp" if "pp" in tok else "")
            solver = next((x for x in ("dense", "sparse", "implicit") if x in tok), "auto")
            tok = [x for x in tok if x not in ("opencv", "pp", "dense", "sparse", "implicit")]
            pb, z = load(f, dev, model)
            lam = 0.0 if tok[0] == "plain" else float(tok[0].replace("prior", ""))
            kw = {}
            ptol = next((float(x[3:]) for x in tok if x.startswith("tol")), 1e-6)
            tok = [x for x in tok if not x.startswith("tol")]
            for x in tok[1:]:
                k_ = "it" if x.startswith("it") else ("h" if x.startswith("hyb") else x[0])
                kw[k_] = float(x[3:] if x.startswith("hyb") else (x[2:] if x.startswith("it") else x[1:]))
            cauchy, iters, rounds = kw.get("c", a.cauchy), int(kw.get("it", a.iters)), int(kw.get("r", 1))
            othr = kw.get("o", 4.0 * cauchy)
            torch.cuda.synchronize(); t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            logs = []; pcg = []
            if "h" in kw:                                   # hybrid: tracks first, then the matches that fit
                full = (pb.p_o, pb.tgt, pb.xt)
                trk = torch.as_tensor(z["trk"][np.argsort(z["p_o"], kind="stable")], device=dev)
                pb.p_o, pb.tgt, pb.xt = full[0][trk], full[1][trk], full[2][trk]
                opt = Options(iters=iters, cauchy=cauchy, prior_weight=lam, solver=solver, pcg_tol=ptol)
                ba = RayBA(pb, opt); ba.run(); logs += opt.log; pcg = list(ba.pcg_iters)
                pb.p_o, pb.tgt, pb.xt = full
                sp = RayBA(pb, Options(cauchy=cauchy)).sampson_px()
                keep = trk | (sp < kw["h"])
                pb.p_o, pb.tgt, pb.xt = full[0][keep], full[1][keep], full[2][keep]
                print(f"  hybrid: {int(trk.sum())} track obs + {int((keep & ~trk).sum())} of {int((~trk).sum())} matches", flush=True)
            for rnd in range(rounds):
                cr = kw["f"] if ("f" in kw and rnd == rounds - 1) else cauchy      # f<x>: Cauchy of the last round
                opt = Options(iters=iters, cauchy=cr, prior_weight=lam, solver=solver, pcg_tol=ptol)
                ba = RayBA(pb, opt)
                ba.run()
                logs += opt.log; pcg += ba.pcg_iters
                if rnd + 1 < rounds:
                    keep = ba.residual_norms() < othr
                    pb.p_o, pb.tgt, pb.xt = pb.p_o[keep], pb.tgt[keep], pb.xt[keep]
                    if "g" in kw:
                        merge_groups(pb, z, kw["g"])
            opt.log = logs
            torch.cuda.synchronize(); wall = time.time() - t0
            np.savez(dst, R=pb.R.cpu().numpy(), t=pb.t.cpu().numpy(), intr=pb.intr.cpu().numpy(), model=model,
                     group=z["group"], s=pb.s.cpu().numpy(), wall=wall, log=json.dumps(opt.log),
                     peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, pcg=np.array(pcg))
            print(f"{sc} {v}: {len(opt.log) - 1} its, cost {opt.log[0]['cost']:.4g} -> {opt.log[-1]['cost']:.4g}, {wall:.1f}s, "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB, M={pb.p_o.shape[0]}" + (f", pcg its mean {np.mean(pcg):.0f} max {max(pcg)}" if pcg else ""), flush=True)
            del pb
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
