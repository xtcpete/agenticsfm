"""Render a run's trace as a self-contained web page: every VLM turn (thinking, decision, tool
call) and every tool step (parameters, summary, the images it returned as evidence).

    python scripts/trace_site.py outputs/runs/agent/<scene> --out outputs/sites/<scene>.html

Images are embedded (downscaled JPEGs), so the page is one file. The GT score, if the run was
evaluated, is shown in the header and labelled as benchmark-only: the agent never saw it.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

MAX_SIDE = 900


def embed(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    im = Image.open(p).convert("RGB")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_run(run_dir: Path) -> dict:
    trace = [json.loads(l) for l in (run_dir / "trace.jsonl").read_text().splitlines() if l]
    scene = run_dir.name
    run = run_dir.parent.name
    result = None
    results = run_dir.parent / "results.jsonl"
    if results.is_file():
        for l in results.read_text().splitlines():
            r = json.loads(l)
            if r.get("scene") == scene:
                result = r
    steps = []
    for r in trace:
        if r.get("source") == "vlm":
            steps.append({"kind": "vlm", "time": r["time"], "content": r.get("content") or "", "reasoning": r.get("reasoning") or "",
                          "tool_calls": r.get("tool_calls", []), "usage": r.get("usage", {}), "error": r.get("error")})
        elif "tool" in r:
            steps.append({"kind": "tool", "time": r["time"], "step": r["step"], "tool": r["tool"], "params": r["params"],
                          "artifact": r.get("artifact"), "cached": r.get("cached", False), "cost": r.get("cost", {}),
                          "summary": r.get("summary", {}), "error": r.get("error"), "source": r.get("source"),
                          "images": [embed(p) for p in r.get("images", [])]})
        else:
            steps.append({"kind": "note", "time": r["time"], "data": {k: v for k, v in r.items() if k not in ("step", "time")}})
    return {"run": run, "scene": scene, "steps": steps, "result": result}


PAGE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap">
<style>
/* Layout: one column of timeline cards with a sticky header; evidence images span the card. */
:root {
  --bg: #f6f5f1; --surface: #ffffff; --surface-2: #eeece6; --fg: #1d1c1a; --muted: #6b675f; --line: #d9d5cc;
  --vlm: #b8540f; --vlm-soft: #fbe9d9; --tool: #1f6f78; --tool-soft: #dcefef; --think: #6b5b95; --think-soft: #ebe6f5;
  --good: #2e7d4f; --bad: #b3261e; --warn: #9a6b00;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #16171a; --surface: #1f2126; --surface-2: #2a2d33; --fg: #ecebe7; --muted: #a09c93; --line: #3a3d44;
  --vlm: #f0a05a; --vlm-soft: #3a2a1c; --tool: #6fc6ce; --tool-soft: #16343a; --think: #b7a7e3; --think-soft: #2b2740;
  --good: #6dc48f; --bad: #ef7a73; --warn: #e3b44b; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #16171a; --surface: #1f2126; --surface-2: #2a2d33; --fg: #ecebe7; --muted: #a09c93; --line: #3a3d44;
  --vlm: #f0a05a; --vlm-soft: #3a2a1c; --tool: #6fc6ce; --tool-soft: #16343a; --think: #b7a7e3; --think-soft: #2b2740;
  --good: #6dc48f; --bad: #ef7a73; --warn: #e3b44b; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.5; margin: 0;
  padding-block: 0 48px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)) }
header { background: var(--bg); border-bottom: 1px solid var(--line); padding: 14px 0 10px }
.wrap { max-width: 1040px; margin: 0 auto }
h1 { font-family: var(--display); font-weight: 500; font-size: 26px; margin: 0 0 4px; text-wrap: balance }
h1 small { font-family: var(--mono); font-size: 13px; color: var(--muted); font-weight: 400; margin-left: 8px }
.facts { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted) }
.facts b { color: var(--fg); font-weight: 500; font-variant-numeric: tabular-nums }
.facts .gt { border-left: 3px solid var(--warn); padding-left: 8px }
.runs { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; font-size: 13px; color: var(--muted) }
.runs select { font: 500 13px var(--mono); padding: 5px 8px; border-radius: 6px; border: 1px solid var(--line); background: var(--surface); color: var(--fg); max-width: 100% }
.runs select:focus-visible { outline: 2px solid var(--tool); outline-offset: 2px }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 8px; font-size: 13px }
.controls label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer }
.legend { display: inline-flex; gap: 10px; margin-left: auto }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px }
.legend .l-vlm::before { background: var(--vlm) } .legend .l-tool::before { background: var(--tool) } .legend .l-think::before { background: var(--think) }
main { max-width: 1040px; margin: 0 auto; display: grid; gap: 14px; padding-top: 18px }
.turn { display: grid; grid-template-columns: 56px 1fr; gap: 0 14px }
.rail { display: flex; flex-direction: column; align-items: center; gap: 4px; padding-top: 4px }
.rail .n { font-family: var(--mono); font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums }
.rail .dot { width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--line); background: var(--surface) }
.turn.vlm .rail .dot { border-color: var(--vlm); background: var(--vlm) }
.turn.tool .rail .dot { border-color: var(--tool) }
.rail .line { flex: 1; width: 2px; background: var(--line); min-height: 18px }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; min-width: 0; overflow: hidden }
.card .head { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; padding: 10px 14px; border-bottom: 1px solid var(--line); background: var(--surface-2) }
.turn.vlm .card .head { background: var(--vlm-soft) }
.turn.tool .card .head { background: var(--tool-soft) }
.tag { font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; font-weight: 500 }
.turn.vlm .tag { color: var(--vlm) } .turn.tool .tag { color: var(--tool) }
.head .name { font-weight: 600; font-size: 15px }
.head .meta { color: var(--muted); font-size: 12.5px; font-family: var(--mono); margin-left: auto; font-variant-numeric: tabular-nums }
.body { padding: 12px 14px; display: grid; gap: 10px }
.decision { font-size: 15px; white-space: pre-wrap; max-width: 72ch }
.call { font-family: var(--mono); font-size: 13px; background: var(--surface-2); border-radius: 6px; padding: 8px 10px; overflow-x: auto }
.call .fn { color: var(--tool); font-weight: 500 }
details { border: 1px solid var(--line); border-radius: 6px; background: var(--surface) }
details summary { cursor: pointer; padding: 7px 10px; font-size: 13px; color: var(--muted); user-select: none; list-style: none; display: flex; gap: 8px; align-items: center }
details summary::before { content: "▸"; color: var(--muted); font-size: 11px } details[open] summary::before { content: "▾" }
details > div { padding: 4px 12px 12px }
details.think { border-color: var(--think); background: var(--think-soft) }
details.think summary { color: var(--think); font-weight: 500 }
details.think > div { font-size: 14px; white-space: pre-wrap; max-width: 80ch; line-height: 1.55 }
.hide-think details.think { display: none }
pre.json { font-family: var(--mono); font-size: 12px; line-height: 1.45; margin: 0; white-space: pre-wrap; word-break: break-word }
.kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 6px 14px; font-size: 13.5px }
.kv div { display: flex; flex-direction: column; min-width: 0 }
.kv .k { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em }
.kv .v { font-family: var(--mono); font-variant-numeric: tabular-nums; overflow-wrap: anywhere }
.kv .v.good { color: var(--good) } .kv .v.bad { color: var(--bad) } .kv .v.warn { color: var(--warn) }
.evidence { display: grid; gap: 8px }
.evidence figure { margin: 0 }
.evidence img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff }
.evidence figcaption { font-size: 12px; color: var(--muted); margin-top: 3px }
.pairs { font-family: var(--mono); font-size: 12px; border-collapse: collapse; width: 100% }
.pairs td, .pairs th { padding: 3px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap }
.pairs th { color: var(--muted); font-weight: 500 }
.scroll { overflow-x: auto }
.pill { display: inline-block; font-family: var(--mono); font-size: 11.5px; padding: 1px 7px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted) }
.pill.cached { color: var(--good); border-color: var(--good) }
.pill.err { color: var(--bad); border-color: var(--bad) }
.final { border: 2px solid var(--vlm) }
.summary-models { display: grid; gap: 8px }
.summary-models .m { display: grid; grid-template-columns: 110px 1fr; gap: 12px; align-items: baseline; font-size: 13.5px }
.summary-models .id { font-family: var(--mono); color: var(--tool) }
@media (max-width: 640px) { .turn { grid-template-columns: 34px 1fr } .rail .n { display: none } .head .meta { margin-left: 0 } }
@media (prefers-reduced-motion: no-preference) { details > div { animation: fade .15s ease-out } }
@keyframes fade { from { opacity: 0 } to { opacity: 1 } }
summary:focus-visible, input:focus-visible { outline: 2px solid var(--tool); outline-offset: 2px }
</style>
<header><div class="wrap">
  <nav class="runs"><label for="runSelect">episode</label><select id="runSelect"></select><span id="runCount"></span></nav>
  <h1 id="title"></h1>
  <div class="facts" id="facts"></div>
  <div class="controls">
    <label><input type="checkbox" id="showThink" checked> show the model's thinking</label>
    <label><input type="checkbox" id="openAll"> expand all raw results</label>
    <span class="legend"><span class="l-vlm">VLM decision</span><span class="l-tool">tool result</span><span class="l-think">thinking</span></span>
  </div>
</div></header>
<main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const all = JSON.parse(document.getElementById("data").textContent).runs;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d=1) => typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(d)) : v;
const pct = v => (100*v).toFixed(1);
function render(data) {
const steps = data.steps;
const t0 = steps.length ? steps[0].time : 0;
const tEnd = steps.length ? steps[steps.length-1].time : 0;
const toolSteps = steps.filter(s => s.kind === "tool");
const vlmSteps = steps.filter(s => s.kind === "vlm");
const tokens = vlmSteps.reduce((a, s) => ({p: a.p + (s.usage.prompt_tokens||0), c: a.c + (s.usage.completion_tokens||0)}), {p:0, c:0});
const submit = toolSteps.filter(s => s.tool === "submit").pop();
const r = data.result || {};
document.getElementById("title").innerHTML = `${esc(data.scene)} <small>run ${esc(data.run)}</small>`;
const facts = [
  `<span><b>${toolSteps.length}</b> tool calls</span>`,
  `<span><b>${vlmSteps.length}</b> VLM turns</span>`,
  `<span><b>${(tokens.p/1000).toFixed(0)}k</b> prompt + <b>${(tokens.c/1000).toFixed(1)}k</b> completion tokens</span>`,
  `<span><b>${((tEnd - t0)/60).toFixed(1)}</b> min wall</span>`,
  submit ? `<span>submitted <b>${esc(submit.params.model_id)}</b> (${esc(submit.summary.registered_images)} images, ${esc(submit.summary.num_models)} model${submit.summary.num_models===1?"":"s"})</span>` : `<span><b>no submission</b></span>`,
];
if (r["auc_all@10"] !== undefined) facts.push(`<span class="gt">benchmark score (GT, never shown to the agent): auc_all@3/10/30 = <b>${pct(r["auc_all@3"])} / ${pct(r["auc_all@10"])} / ${pct(r["auc_all@30"])}</b>, ${esc(r.num_images_reg)}/${esc(r.num_images_gt)} registered</span>`);
document.getElementById("facts").innerHTML = facts.join("");

function kv(items) { return `<div class="kv">${items.map(([k, v, cls]) => `<div><span class="k">${esc(k)}</span><span class="v ${cls||""}">${v}</span></div>`).join("")}</div>`; }
function keyFacts(s) {
  const S = s.summary || {}; const items = [];
  switch (s.tool) {
    case "build_graph": items.push(["method", S.method], ["images", S.num_images], ["pairs", S.num_pairs], ["degree min/med/max", S.degree ? `${S.degree.min} / ${fmt(S.degree.median)} / ${S.degree.max}` : ""]); break;
    case "match": items.push(["detector + matcher", `${S.detector} + ${S.matcher}`], ["verified pairs", S.num_verified_pairs], ["inliers / pair (p10 / med / p90)", S.inliers_per_pair ? `${fmt(S.inliers_per_pair.p10,0)} / ${fmt(S.inliers_per_pair.median,0)} / ${fmt(S.inliers_per_pair.p90,0)}` : ""], ["cameras", `${S.num_cameras} ${(S.camera_models||[]).join(",")}`]);
      if (S.matches && S.matches.covisibility) items.push(["covisibility < 0.5", pct(S.matches.covisibility["frac_below_0.5"]) + "% of pairs", "warn"]); break;
    case "score_pairs": items.push(["method", S.method], ["pairs", S.num_pairs], ["median score", fmt(S.median, 3)]); if (S.histogram) for (const [k, v] of Object.entries(S.histogram)) items.push([`pairs ${k}`, pct(v) + "%"]); break;
    case "filter_edges": items.push(["threshold", S.threshold], ["removed pairs", S.removed_pairs, "warn"], ["remaining verified pairs", S.num_verified_pairs], ["graph components", (S.graph_components||[]).join(" + "), (S.num_graph_components||1) > 1 ? "bad" : "good"]); break;
    case "reconstruct": items.push(["backend", S.backend], ["models", S.num_models, S.num_models === 1 ? "good" : "bad"], ["registered images", `${S.registered_images} / ${S.num_images_in_db}`, S.registered_images === S.num_images_in_db ? "good" : "warn"]);
      (S.models||[]).slice(0,3).forEach(m => items.push([`model ${m.id}`, `${m.num_images} img, ${m.num_points} pts, reproj ${fmt(m.mean_reproj_error_px,2)} px, track ${fmt(m.mean_track_length,1)}`])); break;
    case "examine": { const E = S.edges || {}; const C = E.consistency || {};
      items.push(["models", S.num_models, S.num_models === 1 ? "good" : "bad"], ["registered", `${S.registered_images} / ${S.num_images_total}`, S.num_unregistered === 0 ? "good" : "warn"]);
      if (C.strong_inconsistent_frac !== undefined) items.push(["strong inconsistent pairs", pct(C.strong_inconsistent_frac) + "% of " + C.strong_pairs, C.strong_inconsistent_frac > 0.05 ? "bad" : "good"]);
      if (C["frac_pairs_below_0.5"] !== undefined) items.push(["pairs < 50% consistent", pct(C["frac_pairs_below_0.5"]) + "%", C["frac_pairs_below_0.5"] > 0.2 ? "bad" : ""]);
      if (C.median !== undefined) items.push(["median consistency", fmt(C.median, 3)]);
      items.push(["pairs in one model / across / unregistered", `${E.pairs_in_one_model ?? "-"} / ${E.pairs_across_models ?? "-"} / ${E.pairs_with_unregistered_image ?? "-"}`]);
      (S.models||[]).slice(0,2).forEach(m => items.push([`model ${m.id}`, `${m.num_images} img, ${m.num_points} pts, reproj ${fmt(m.mean_reproj_error_px,2)} px`]));
      if (S.unregistered_images && S.unregistered_images.length) items.push(["unregistered", S.unregistered_images.slice(0,8).join(", ") + (S.unregistered_images.length > 8 ? " …" : ""), "warn"]);
      break; }
    case "inspect": items.push(["shown", (S.shown||[]).length ? S.shown.join(", ") : (S.pair||[]).join(" ↔ ")]); if (S.verified_inliers !== undefined) items.push(["verified inliers", S.verified_inliers]); break;
    case "submit": items.push(["model", S.submitted], ["backend", S.backend], ["registered images", `${S.registered_images} / ${S.num_images_in_db}`], ["models", S.num_models]); break;
  }
  return items.length ? kv(items) : "";
}
function pairsTable(s) {
  const E = (s.summary||{}).edges; if (!E || !E.inconsistent_pairs || !E.inconsistent_pairs.length) return "";
  const rows = E.inconsistent_pairs.slice(0, 10).map(p => `<tr><td>${esc(p.pair[0])}</td><td>${esc(p.pair[1])}</td><td>${p.inliers}</td><td class="${p.consistent_frac < 0.2 ? "bad" : ""}">${pct(p.consistent_frac)}%</td></tr>`).join("");
  const imgs = (E.images_in_most_inconsistent_pairs||[]).slice(0,6).map(([n, c]) => `${esc(n)} (${c})`).join(", ");
  return `<details><summary>inconsistent pairs flagged by examine (strong pairs whose matches disagree with the poses)</summary><div class="scroll"><table class="pairs"><tr><th>image</th><th>image</th><th>inliers</th><th>consistent</th></tr>${rows}</table>${imgs ? `<p style="font-size:12.5px;color:var(--muted);margin:8px 0 0">images in most flagged pairs: ${imgs}</p>` : ""}</div></details>`;
}
function stripSummary(S) { const o = {}; for (const [k, v] of Object.entries(S||{})) if (k !== "renders") o[k] = v; return o; }
let n = 0, html = "";
for (const s of steps) {
  const t = ((s.time - t0)/60).toFixed(1) + " min";
  if (s.kind === "vlm") {
    n++;
    const calls = s.tool_calls.map(c => { let a = c.arguments; try { a = JSON.stringify(JSON.parse(c.arguments)); } catch (e) {} return `<div class="call"><span class="fn">${esc(c.name)}</span>(${esc(a)})</div>`; }).join("");
    html += `<section class="turn vlm"><div class="rail"><span class="n">${n}</span><span class="dot"></span><span class="line"></span></div><div class="card">
      <div class="head"><span class="tag">VLM turn</span><span class="name">${s.tool_calls.length ? "decides to call " + esc(s.tool_calls[0].name) : (s.error ? "error" : "no tool call")}</span><span class="meta">${t} · ${s.usage.completion_tokens||0} tokens out</span></div>
      <div class="body">${s.reasoning ? `<details class="think"><summary>thinking (${s.reasoning.length} chars)</summary><div>${esc(s.reasoning)}</div></details>` : ""}
      ${s.content.trim() ? `<div class="decision">${esc(s.content.trim())}</div>` : ""}${calls}${s.error ? `<div class="call">${esc(s.error)}</div>` : ""}</div></div></section>`;
  } else if (s.kind === "tool") {
    n++;
    const imgs = s.images.filter(Boolean).map((src, i) => `<figure><img src="${src}" alt="${esc(s.tool)} output ${i+1}" loading="lazy"><figcaption>${esc(s.tool)} · ${s.tool === "examine" ? "cameras (red, with viewing direction) and points, two principal views" : s.tool === "inspect" && s.params.pair ? "verified matches between the two images (up to 150 drawn)" : "thumbnails"}</figcaption></figure>`).join("");
    const err = s.summary && s.summary.error;
    html += `<section class="turn tool ${s.tool === "submit" ? "final" : ""}"><div class="rail"><span class="n">${n}</span><span class="dot"></span><span class="line"></span></div><div class="card ${s.tool === "submit" ? "final" : ""}">
      <div class="head"><span class="tag">${s.source && s.source.startsWith("agent:opening") ? "harness" : "tool"}</span><span class="name">${esc(s.tool)}</span>
        ${s.artifact ? `<span class="pill">→ ${esc(s.artifact)}</span>` : ""}${s.cached ? `<span class="pill cached">cached</span>` : ""}${err ? `<span class="pill err">error</span>` : ""}
        <span class="meta">${t} · ${fmt(s.cost.wall||0,0)} s${s.cost.gpu_seconds ? " · gpu " + fmt(s.cost.gpu_seconds,0) + " s" : ""}</span></div>
      <div class="body"><div class="call">${esc(s.tool)}(${esc(JSON.stringify(s.params))})</div>
      ${err ? `<div class="call" style="color:var(--bad)">${esc(err)}</div>` : keyFacts(s)}${pairsTable(s)}
      ${imgs ? `<div class="evidence">${imgs}</div>` : ""}
      <details class="raw"><summary>raw result as the model saw it</summary><div><pre class="json">${esc(JSON.stringify(stripSummary(s.summary), null, 1))}</pre></div></details></div></div></section>`;
  } else {
    html += `<section class="turn"><div class="rail"><span class="n"></span><span class="dot"></span><span class="line"></span></div><div class="card"><div class="head"><span class="tag">note</span><span class="name">${esc(s.data.source||"")}</span></div><div class="body"><pre class="json">${esc(JSON.stringify(s.data, null, 1))}</pre></div></div></section>`;
  }
}
document.getElementById("main").innerHTML = html;
document.querySelectorAll("details.raw").forEach(d => d.open = openAll.checked);
}
const think = document.getElementById("showThink"), openAll = document.getElementById("openAll");
const sel = document.getElementById("runSelect");
let current = 0;
function select(i) {
  current = i;
  sel.value = String(i);
  render(all[i]);
  window.scrollTo(0, 0);
  try { if (all.length > 1) history.replaceState(null, "", "#" + all[i].scene); } catch (e) {}
}
all.forEach((r, i) => {
  const o = document.createElement("option"); o.value = String(i);
  const gt = r.result && r.result["auc_all@10"] !== undefined ? ` · GT ${(100 * r.result["auc_all@10"]).toFixed(1)}` : "";
  o.textContent = `${r.run} · ${r.scene.replace(/^t\d\d_v\d\d_s\d\d_r\d\d_/, "")}${gt}`;
  sel.appendChild(o);
});
sel.addEventListener("change", () => select(Number(sel.value)));
document.getElementById("runCount").textContent = `${all.length} episodes`;
if (all.length < 2) sel.parentElement.hidden = true;
try { const i = all.findIndex(r => r.scene === location.hash.replace("#", "")); if (i >= 0) current = i; } catch (e) {}
window.addEventListener("hashchange", () => { const i = all.findIndex(r => r.scene === location.hash.replace("#", "")); if (i >= 0 && i !== current) select(i); });
think.addEventListener("change", () => document.body.classList.toggle("hide-think", !think.checked));
openAll.addEventListener("change", () => document.querySelectorAll("details.raw").forEach(d => d.open = openAll.checked));
select(current);
try { const saved = localStorage.getItem("showThink"); if (saved === "0") { think.checked = false; document.body.classList.add("hide-think"); } think.addEventListener("change", () => localStorage.setItem("showThink", think.checked ? "1" : "0")); } catch (e) {}
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="one or several <run>/<scene> directories")
    parser.add_argument("--out")
    parser.add_argument("--title", default="Agent episodes")
    args = parser.parse_args()
    runs = [load_run(Path(d)) for d in args.run_dirs]
    page = PAGE.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", json.dumps({"runs": runs}).replace("</", "<\\/"))
    out = Path(args.out or (Path(args.run_dirs[0]) / "trace.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {sum(len(r['steps']) for r in runs)} steps in {len(runs)} runs)")


if __name__ == "__main__":
    main()
