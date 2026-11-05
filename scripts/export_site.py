"""Export episodes for the showcase site (scripts/showcase/index.html): every VLM turn (thinking, message, tool call), every
tool step (parameters, summary, evidence images), the open solver's growth log, and the reconstructed models (points,
cameras, gravity-aligned frame) as JSON for the page's 3D viewer.

    python scripts/export_site.py --out outputs/site [--runs agent_v13:wriva agent_eth3d_v13:eth3d] [--solver-runs incremental_v0 incremental_v0_t06]
    second group, merged into an existing site: --solver-runs ff_vlm_dev ff_vlm_dev_s02 --solver-group solver-ffvlm --ep-suffix=-ffvlm
        --solver-title "..." --controller "Pi3X loop, VLM selection" --compare-run incremental_v13 --merge-into <site>/data/index.json

Then copy scripts/showcase/index.html next to the exported data/, img/ and models/ folders and serve or publish the folder.
GT scores shown on the page come from each run's results.jsonl (benchmark layer)."""
import argparse
import shutil
import io, json, re, struct, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/raid/home/gochen/cvpr_2027/agenticsfm/src")
import pycolmap
from PIL import Image
from agenticsfm.tools.frame import Frame, cameras_of, umeyama
sys.path.insert(0, "/raid/home/gochen/cvpr_2027/agenticsfm/scripts")
from analyze_failures import image_errors, rot_angle


def robust_similarity(src, dst):
    """s, R, t with dst ~ s R src + t, re-fitted on the inliers (residual < 3 x median) for three rounds."""
    inl = np.ones(len(src), bool)
    s_, R_, t_ = umeyama(src, dst)
    for _ in range(3):
        if inl.sum() < 3:
            break
        s_, R_, t_ = umeyama(src[inl], dst[inl])
        res = np.linalg.norm((s_ * (R_ @ src.T)).T + t_ - dst, axis=1)
        thr = max(3 * np.median(res[inl]), 1e-9)
        inl = res < thr
    return s_, R_, t_, inl


def gt_alignment(rec, gt):
    """Per-image errors of a model against GT (benchmark layer) and the GT cameras expressed in the model's frame."""
    names = [im.name for im in rec.images.values() if im.has_pose and gt.find_image_with_name(im.name) is not None]
    if len(names) < 3:
        return None
    Cm = np.array([rec.find_image_with_name(n).projection_center() for n in names])
    Cg = np.array([gt.find_image_with_name(n).projection_center() for n in names])
    s_, R_, t_, inl = robust_similarity(Cm, Cg)
    gt_all = [im for im in gt.images.values() if im.has_pose]
    gt_span = float(np.linalg.norm(np.ptp(np.array([im.projection_center() for im in gt_all]), axis=0))) or 1.0
    rel = image_errors(gt, rec)
    per = {}
    for n in names:
        Rm = rec.find_image_with_name(n).cam_from_world().rotation.matrix()
        Rg = gt.find_image_with_name(n).cam_from_world().rotation.matrix()
        c_in_gt = s_ * R_ @ rec.find_image_with_name(n).projection_center() + t_
        per[n] = {"rot_deg": round(rot_angle(Rm @ R_.T @ Rg.T), 2), "pos_frac": round(float(np.linalg.norm(c_in_gt - gt.find_image_with_name(n).projection_center())) / gt_span, 4),
                  "rel_deg": round(rel.get(n, float("nan")), 2)}
    gt_in_model = []
    for im in gt_all:
        c = (R_.T @ (im.projection_center() - t_)) / s_
        Rg = im.cam_from_world().rotation.matrix()
        gt_in_model.append((im.name, c, Rg @ R_, im))
    rot = np.array([v["rot_deg"] for v in per.values()])
    summary = {"aligned_on": int(inl.sum()), "of": len(names), "scale": round(float(s_), 4), "median_rot_deg": round(float(np.median(rot)), 2),
               "within_3": round(float((rot < 3).mean()), 3), "within_10": round(float((rot < 10).mean()), 3), "over_10": int((rot >= 10).sum()),
               "gt_images": len(gt_all), "unregistered": len(gt_all) - len(names), "median_pos_frac": round(float(np.median([v["pos_frac"] for v in per.values()])), 4)}
    return {"per": per, "gt_in_model": gt_in_model, "summary": summary}
pycolmap.logging.minloglevel = 2
ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=str(ROOT / "outputs/site"))
_ap.add_argument("--runs", nargs="*", default=["agent_v13:wriva", "agent_eth3d_v13:eth3d"], help="agent runs as <run>:<dataset>")
_ap.add_argument("--solver-runs", nargs="*", default=["incremental_v13", "incremental_v13_mixed"])
_ap.add_argument("--include-agent", action="store_true", help="also export the archived tool-selection agent episodes")
_ap.add_argument("--solver-group", default="solver", help="group id of the solver episodes")
_ap.add_argument("--solver-title", default="Open solver (Pi3X seed, verified growth, VLM-arbitrated hard images; no COLMAP mapper)")
_ap.add_argument("--controller", default=None, help="controller label of the solver episodes (default: open solver v1.3)")
_ap.add_argument("--ep-suffix", default="", help="suffix of the episode ids, so a second solver group does not overwrite the first")
_ap.add_argument("--compare-run", default=None, help="solver run whose score is shown next to each episode (e.g. incremental_v13)")
_ap.add_argument("--inline-images", action="store_true", help="embed the episode figures in the episode JSON as data URIs (keeps the published file count down)")
_ap.add_argument("--img-side", type=int, default=1100, help="longest side of the exported figures")
_ap.add_argument("--merge-into", default=None, help="existing site index.json: keep its groups and episodes, add (or replace) this group")
_args = _ap.parse_args()
SITE = Path(_args.out)
for _d in ("data/ep", "models", "img"):
    (SITE / _d).mkdir(parents=True, exist_ok=True)
MAX_POINTS, IMG_SIDE = 40000, _args.img_side

def inline_images(x, ep_id):
    """Replace img/<ep_id>/... paths with data URIs and delete the files."""
    import base64
    if isinstance(x, dict): return {k: inline_images(v, ep_id) for k, v in x.items()}
    if isinstance(x, list): return [inline_images(v, ep_id) for v in x]
    if isinstance(x, str) and x.startswith(f"img/{ep_id}/") and (SITE / x).is_file():
        return "data:image/jpeg;base64," + base64.b64encode((SITE / x).read_bytes()).decode()
    return x

def read_results(run):
    out = {}
    p = ROOT / "outputs/runs" / run / "results.jsonl"
    if p.is_file():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); out[r["scene"]] = r
    return out

def short_scene(s):
    m = re.match(r"^t\d+_v\d+_(s\d+)_r\d+_(.*)$", s)
    if not m: return s
    rest = m.group(2).replace("VaryingAltitudes_", "").replace("ReconstructedArea_", "").replace("_building_1_door", "")
    return f"{rest} {m.group(1)}" if "PTZ" in rest else rest

def trim(x, depth=0):
    if isinstance(x, dict):
        return {k: trim(v, depth + 1) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list):
        return [trim(v, depth + 1) for v in x[:12]] + ([f"... {len(x)} items"] if len(x) > 12 else [])
    if isinstance(x, str) and len(x) > 400:
        return x[:400] + " ..."
    if isinstance(x, float):
        return round(x, 4)
    return x

def save_image(src, dst):
    im = Image.open(src).convert("RGB")
    s = IMG_SIDE / max(im.size)
    if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=74, optimize=True)
    return im.size

def largest_submodel(model_dir):
    subs = [d for d in Path(model_dir).iterdir() if d.is_dir() and d.name.isdigit()]
    recs = [(pycolmap.Reconstruction(str(d)), d.name) for d in subs]
    recs.sort(key=lambda r: -r[0].num_reg_images())
    return recs

def export_model(model_dir, out_path, images_dir, order=None, gt=None):
    recs = largest_submodel(model_dir)
    rec, sub = recs[0]
    cams = cameras_of(rec)
    fr = Frame.build(cams)
    align = gt_alignment(rec, gt) if gt is not None else None
    pts = list(rec.points3D.values())
    cols = np.array([p.color for p in pts[:500]]) if pts else np.zeros((0, 3))
    if len(pts) and (cols.sum(1) > 0).mean() < 0.05:
        try:
            rec.extract_colors_for_all_images(str(images_dir)); pts = list(rec.points3D.values())
        except Exception as e:
            print("   color extraction failed:", e)
    rng = np.random.default_rng(0)
    if len(pts) > MAX_POINTS:
        tl = np.array([p.track.length() for p in pts]); w = (tl >= 3) * 1.0 + 0.2
        idx = rng.choice(len(pts), MAX_POINTS, replace=False, p=w / w.sum()); pts = [pts[i] for i in idx]
    X = np.array([p.xyz for p in pts]) if pts else np.zeros((0, 3))
    C = np.array([p.color for p in pts], dtype=np.uint8) if pts else np.zeros((0, 3), np.uint8)
    if len(X):
        X = fr.apply(X) / fr.span
        keep = np.all(np.abs(X) < 4.0, axis=1); X, C = X[keep], C[keep]   # far-flung outliers blow up the view
        if (C.sum(1) > 0).mean() < 0.05:   # still no colours: shade by height
            z = X[:, 2]; t = np.clip((z - np.percentile(z, 3)) / max(np.ptp(z) + 1e-9, 1e-6), 0, 1)
            C = (np.stack([60 + 160 * t, 90 + 110 * t, 200 - 120 * t], 1)).astype(np.uint8)
    camera_rows = []
    for im in rec.images.values():
        if not im.has_pose: continue
        cam = rec.cameras[im.camera_id]
        cfw = im.cam_from_world(); R = cfw.rotation.matrix()
        c = fr.apply(im.projection_center()[None])[0] / fr.span
        # camera axes in frame coordinates: rows of R are the camera x, y, z axes expressed in world
        axes = fr.apply_dir(R)  # (3,3): row i = camera axis i in frame coords
        f = cam.mean_focal_length()
        row = {"name": im.name, "c": [round(float(v), 5) for v in c], "ax": [round(float(v), 5) for v in axes.ravel()],
               "hw": round(cam.width / (2 * f), 4), "hh": round(cam.height / (2 * f), 4), "order": (order or {}).get(im.name)}
        if align and im.name in align["per"]:
            row.update(align["per"][im.name])
        camera_rows.append(row)
    gt_rows = []
    if align:
        reg_names = {r["name"] for r in camera_rows}
        for name, cg, Rg, gim in align["gt_in_model"]:
            gcam = gt.cameras[gim.camera_id]
            fg = gcam.mean_focal_length()
            gt_rows.append({"name": name, "c": [round(float(v), 5) for v in (fr.apply(cg[None])[0] / fr.span)],
                            "ax": [round(float(v), 5) for v in fr.apply_dir(Rg).ravel()], "hw": round(gcam.width / (2 * fg), 4), "hh": round(gcam.height / (2 * fg), 4),
                            "registered": name in reg_names})
    header = json.dumps({"num_points": int(len(X)), "num_cameras": len(camera_rows), "sub_models": [(name, r.num_reg_images()) for r, name in recs],
                         "reproj": round(rec.compute_mean_reprojection_error(), 3) if rec.num_points3D() else None,
                         "num_points_total": rec.num_points3D(), "cameras": camera_rows, "gt_cameras": gt_rows,
                         "gt_summary": align["summary"] if align else None,
                         "metres_per_unit": (fr.span * align["summary"]["scale"]) if align else None}).encode()
    import base64
    Path(out_path).write_text(json.dumps({"header": json.loads(header), "pos": base64.b64encode(X.astype("<f4").tobytes()).decode(),
                                          "col": base64.b64encode(C.astype(np.uint8).tobytes()).decode()}))
    return {"num_images": rec.num_reg_images(), "num_points": rec.num_points3D(), "reproj": round(rec.compute_mean_reprojection_error(), 3) if rec.num_points3D() else None,
            "num_models": len(recs), "gt": align["summary"] if align else None}

def load_trace(run_dir):
    return [json.loads(l) for l in (run_dir / "trace.jsonl").read_text().splitlines() if l.strip()]

def steps_from_trace(recs, ep_id, img_counter):
    steps = []
    for r in recs:
        if r.get("source") == "vlm":
            steps.append({"kind": "vlm", "reasoning": r.get("reasoning") or "", "content": (r.get("content") or "").strip(),
                          "tool_calls": [{"name": t.get("name"), "arguments": t.get("arguments")} for t in r.get("tool_calls", [])],
                          "usage": r.get("usage", {})})
        elif "tool" in r:
            imgs = []
            for p in r.get("images", []):
                if Path(p).is_file():
                    img_counter[0] += 1
                    rel = f"img/{ep_id}/{img_counter[0]:03d}.jpg"
                    try:
                        w, h = save_image(p, SITE / rel); imgs.append({"src": rel, "w": w, "h": h, "name": Path(p).name})
                    except Exception as e:
                        print("   image failed", p, e)
            steps.append({"kind": "tool", "tool": r["tool"], "params": r.get("params", {}), "artifact": r.get("artifact"), "cached": r.get("cached", False),
                          "wall": round(r.get("cost", {}).get("wall", 0), 1), "summary": trim(r.get("summary", {})), "error": (r.get("error") or "")[-1500:] or None,
                          "source": r.get("source"), "images": imgs})
        else:
            steps.append({"kind": "note", "data": trim({k: v for k, v in r.items() if k not in ("step", "time")})})
    return steps

def ref_scores(scene, dataset):
    refs = {}
    fx = read_results("fixed_wriva_v12" if dataset == "wriva" else "fixed_eth3d_v12").get(scene)
    sc = read_results("scripted_v12" if dataset == "wriva" else "scripted_eth3d_v12").get(scene)
    for k, r in (("fixed", fx), ("scripted", sc)):
        if r and r.get("status") == "ok":
            refs[k] = {m: round(100 * r[m], 1) for m in ("auc_all@3", "auc_all@10", "colmap_auc@1", "colmap_auc@3", "colmap_auc@5") if m in r}
            refs[k]["registered"] = r.get("num_images_reg")
    return refs

def score_of(r):
    if not r or r.get("status") != "ok": return None
    return {**{m: round(100 * r[m], 1) for m in ("auc_all@3", "auc_all@10", "colmap_auc@1", "colmap_auc@3", "colmap_auc@5") if m in r},
            "registered": r.get("num_images_reg"), "num_images": r.get("num_images_gt"), "num_models": r.get("num_models"), "wall": r.get("wall")}

def images_dir_of(run_dir):
    arts = json.loads((run_dir / "artifacts.json").read_text()) if (run_dir / "artifacts.json").is_file() else {}
    # the workspace store holds the images symlink
    for a in arts.values():
        store = ROOT / "outputs/cache" / Path(a["path"]).parts[0] if not Path(a["path"]).is_absolute() else None
    return None

def store_of(run_dir):
    arts = json.loads((run_dir / "artifacts.json").read_text())
    any_art = next(iter(arts.values()))
    # artifact paths are relative to the store: find the store by matching the id dir
    for cache in (ROOT / "outputs/cache").iterdir():
        if (cache / any_art["path"]).exists():
            return cache
    raise RuntimeError("store not found")

index = {"generated": time.strftime("%Y-%m-%d"), "groups": [], "episodes": {}}
plan = [(f"agent-{ds}", f"Agent, {ds.upper()} ({run})", run, ds, "agent") for run, ds in (r.split(":") for r in _args.runs)] if _args.include_agent else []
plan.append((_args.solver_group, _args.solver_title, None, None, "incremental"))
import pycolmap as _pc
from agenticsfm.config import load_config as _lc
from agenticsfm.datasets import load_benchmark as _lb
_cfg = _lc()
GT_DIRS = {b.name: b.gt_dir for ds in ("wriva", "eth3d") for b in _lb(_cfg, ds, "all")}
_gt_cache = {}
def gt_of(scene):
    if scene not in GT_DIRS: return None
    if scene not in _gt_cache: _gt_cache[scene] = _pc.Reconstruction(str(GT_DIRS[scene]))
    return _gt_cache[scene]
img_counter = [0]
for gid, gtitle, run, dataset, controller in plan:
    group = {"id": gid, "title": gtitle, "episodes": []}
    if controller == "agent":
        results = read_results(run)
        for sc_dir in sorted((ROOT / "outputs/runs" / run).iterdir()):
            if not (sc_dir / "trace.jsonl").is_file(): continue
            scene = sc_dir.name; ep_id = f"{gid}-{re.sub(r'[^a-z0-9]+', '-', short_scene(scene).lower()).strip('-')}"
            print("==", ep_id)
            store = store_of(sc_dir)
            steps = steps_from_trace(load_trace(sc_dir), ep_id, img_counter)
            models = []
            if (sc_dir / "final").is_dir():
                minfo = export_model(sc_dir / "final", SITE / "models" / f"{ep_id}-final.json", store / "images")
                models.append({"id": f"{ep_id}-final", "label": "submitted model", "file": f"models/{ep_id}-final.json", **minfo})
            r = results.get(scene, {})
            ep = {"id": ep_id, "group": gid, "title": short_scene(scene), "scene": scene, "dataset": dataset, "controller": "agent (Qwen3.8-27B)", "run": run,
                  "num_images": r.get("num_images_gt"), "score": score_of(r), "refs": ref_scores(scene, dataset), "wall": r.get("wall"),
                  "n_turns": sum(1 for s in steps if s["kind"] == "vlm"), "n_tools": sum(1 for s in steps if s["kind"] == "tool"),
                  "tokens": sum((s.get("usage") or {}).get("prompt_tokens", 0) for s in steps if s["kind"] == "vlm"),
                  "submitted": r.get("submitted"), "models": models}
            (SITE / "data/ep" / f"{ep_id}.json").write_text(json.dumps({**ep, "steps": steps}, default=str))
            index["episodes"][ep_id] = ep; group["episodes"].append(ep_id)
    else:
        eth = {s.name for s in (ROOT / "outputs/data/eth3d").iterdir()} if (ROOT / "outputs/data/eth3d").is_dir() else set()
        # a scene rerun in a later run (e.g. after a crash) replaces the earlier attempt unless that one failed less
        chosen = {}
        for run in _args.solver_runs:
            if not (ROOT / "outputs/runs" / run).is_dir(): continue
            res = read_results(run)
            for sc_dir in sorted((ROOT / "outputs/runs" / run).iterdir()):
                if not (sc_dir / "trace.jsonl").is_file(): continue
                ok = res.get(sc_dir.name, {}).get("status") == "ok"
                if ok or sc_dir.name not in chosen: chosen[sc_dir.name] = (run, sc_dir, res, ok)
        if True:
            for run, sc_dir, results, _ok in chosen.values():
                label, dataset_of = (" (filter 0.6)" if "t06" in run else ""), {s: "eth3d" for s in eth}
                scene = sc_dir.name; dataset = dataset_of.get(scene, "wriva" if not scene.startswith(("m07_", "mixed", "images")) else "mixed")
                ep_id = f"solver-{re.sub(r'[^a-z0-9]+', '-', short_scene(scene).lower()).strip('-')}" + ("-t06" if "t06" in run else "") + _args.ep_suffix
                print("==", ep_id)
                store = store_of(sc_dir)
                recs = load_trace(sc_dir)
                steps = steps_from_trace(recs, ep_id, img_counter)
                grow_art = next((x["artifact"] for x in recs if x.get("tool") == "grow_model" and x.get("artifact")), None)
                order, grow_log = {}, None
                if grow_art and (store / "model" / grow_art / "grow_log.json").is_file():
                    grow_log = json.loads((store / "model" / grow_art / "grow_log.json").read_text())
                    for row in grow_log["log"]:
                        if row.get("kind") == "seed":
                            for n in row["attempts"][row["chosen"]]["kept"]: order.setdefault(n, 0)
                        elif isinstance(row.get("registered"), list):
                            for n in row["registered"]: order.setdefault(n, row.get("round", 0))
                    # vlm rows: copy the figures / photos they were shown into the site
                    for row in grow_log["log"]:
                        if row.get("kind") == "vlm":
                            imgs = {}
                            for flabel, names_ in (row.get("images") or {}).items():
                                out_names = []
                                for nm in names_:
                                    if not Path(nm).is_absolute():          # scene photos: name only; the figures (absolute paths) are copied
                                        out_names.append({"src": None, "name": nm}); continue
                                    src = Path(nm)
                                    if src.is_file():
                                        img_counter[0] += 1
                                        rel = f"img/{ep_id}/{img_counter[0]:03d}.jpg"
                                        try:
                                            save_image(src, SITE / rel); out_names.append({"src": rel, "name": Path(nm).name})
                                        except Exception as e:
                                            print("   figure failed", src, e)
                                imgs[flabel] = out_names
                            row["images"] = imgs
                            row["reasoning"] = (row.get("reasoning") or "")[-1200:]
                    # splice the growth rounds after the grow_model tool step
                    for i, s in enumerate(steps):
                        if s["kind"] == "tool" and s["tool"] == "grow_model":
                            s["grow_log"] = grow_log["log"]; s["grow_summary"] = trim(grow_log["summary"]); break
                models = []
                if grow_art:
                    minfo = export_model(store / "model" / grow_art, SITE / "models" / f"{ep_id}-grown.json", store / "images", order, gt_of(scene))
                    models.append({"id": f"{ep_id}-grown", "label": "grown (before refinement)", "file": f"models/{ep_id}-grown.json", "has_order": True, **minfo})
                if (sc_dir / "final").is_dir():
                    minfo = export_model(sc_dir / "final", SITE / "models" / f"{ep_id}-final.json", store / "images", order, gt_of(scene))
                    models.append({"id": f"{ep_id}-final", "label": "submitted (after optimize)", "file": f"models/{ep_id}-final.json", "has_order": True, **minfo})
                r = results.get(scene, {})
                refs = ref_scores(scene, dataset)
                if _args.compare_run and score_of(read_results(_args.compare_run).get(scene)):
                    refs["solver"] = score_of(read_results(_args.compare_run).get(scene)); refs["solver_run"] = _args.compare_run
                title = short_scene(scene) + label + ("" if run == "incremental_v0" or _args.ep_suffix else f" [{run.replace('incremental_', '')}]")
                ep = {"id": ep_id, "group": gid, "title": title, "scene": scene, "dataset": dataset,
                      "controller": _args.controller or ("open solver " + ("v0" if run == "incremental_v0" else "v1.3")), "run": run,
                      "num_images": r.get("num_images_gt"), "score": score_of(r), "refs": refs, "wall": r.get("wall"),
                      "n_turns": 0, "n_tools": sum(1 for s in steps if s["kind"] == "tool"), "tokens": 0, "submitted": r.get("submitted"), "models": models,
                      "grow": trim(grow_log["summary"]) if grow_log else None,
                      "gt_align": next((m["gt"] for m in reversed(models) if m.get("gt")), None)}
                full = {**ep, "steps": steps}
                if _args.inline_images:
                    full = inline_images(full, ep_id); shutil.rmtree(SITE / "img" / ep_id, ignore_errors=True)
                (SITE / "data/ep" / f"{ep_id}.json").write_text(json.dumps(full, default=str))
                index["episodes"][ep_id] = ep; group["episodes"].append(ep_id)
    index["groups"].append(group)
if _args.merge_into:
    old = json.loads(Path(_args.merge_into).read_text())
    new_ids = {g["id"] for g in index["groups"]}
    index["groups"] = [g for g in old["groups"] if g["id"] not in new_ids] + index["groups"]
    index["episodes"] = {**{k: v for k, v in old["episodes"].items() if v.get("group") not in new_ids}, **index["episodes"]}
(SITE / "data/index.json").write_text(json.dumps(index, default=str))
import os
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(SITE) for f in fs)
print("DONE", len(index["episodes"]), "episodes;", img_counter[0], "images;", f"{total/1e6:.1f} MB;", sum(len(fs) for _, _, fs in os.walk(SITE)), "files")
