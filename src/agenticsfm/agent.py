"""The agent: a VLM that reads tool summaries and renders, and chooses the next tool call.

One ``run(ws)`` per scene. The conversation starts with the system prompt (task, playbook,
protocol) and a thumbnail grid of the scene; each turn the VLM calls one tool, whose JSON summary
comes back as a tool message and whose images (renders, thumbnails) as a user message. The episode
ends at submit, or when the budget (tool calls, wall time) is used up, in which case the agent is
asked to submit and, failing that, the harness submits the model with the most registered images.
Every VLM turn is in the trace with its reasoning and token usage.
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

from PIL import Image

from agenticsfm import prompts
from agenticsfm.tools import TOOLS, ToolResult, call, schemas
from agenticsfm.workspace import Workspace


def encode_image(path: str, max_side: int) -> dict[str, Any]:
    image = Image.open(path).convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def for_vlm(summary: dict[str, Any]) -> dict[str, Any]:
    """The summary without file paths and other things the VLM has no use for."""
    return {k: v for k, v in summary.items() if k not in ("renders", "path", "features_id", "matches_id")}


class Agent:
    def __init__(self, cfg: dict[str, Any], max_steps: int = 25, max_minutes: float = 60, image_max_side: int = 1024,
                 tools: list[str] | None = None, prelude: bool = True, dataset: str | None = None):
        from openai import OpenAI

        vlm = cfg["vlm"]
        self.client = OpenAI(base_url=vlm["base_url"], api_key=vlm["api_key"], timeout=600)
        self.model = vlm["model"]
        self.max_tokens = vlm.get("max_tokens", 8192)
        self.request = dict(vlm.get("request", {}))
        self.max_steps = max_steps
        self.max_minutes = max_minutes
        self.image_max_side = image_max_side
        names = tools or list(TOOLS)
        self.schemas = [s for s in schemas() if s["function"]["name"] in names]
        self.prelude = prelude
        from agenticsfm.controllers.scripted import recipe_of

        self.recipe = recipe_of(cfg, dataset)

    def _chat(self, messages: list[dict], tool_choice: Any = "auto") -> Any:
        return self.client.chat.completions.create(model=self.model, messages=messages, tools=self.schemas,
                                                   tool_choice=tool_choice, max_tokens=self.max_tokens, **self.request)

    def _opening(self, ws: Workspace, cfg: dict[str, Any]) -> list[dict]:
        names = ws.image_names()
        sizes: dict[tuple, int] = {}
        for n in names[:: max(1, len(names) // 40)]:
            with Image.open(ws.images_dir / n) as im:
                sizes[im.size] = sizes.get(im.size, 0) + 1
        size_text = ", image sizes " + ", ".join(f"{w}x{h}" for (w, h) in sizes) if sizes else ""
        thumbs = call(ws, cfg, "inspect", {}, source="agent:opening")
        text = prompts.OPENING.format(name=ws.scene.name, num_images=len(names), sizes=size_text,
                                      num_shown=len(thumbs.summary.get("shown", [])))
        rc = self.recipe
        text += prompts.RECIPE.format(detector=rc["detector"], num_keypoints=rc["num_keypoints"], resize=rc["resize"],
                                      thresholds=", ".join(str(t) for t in rc["thresholds"]))
        content = [{"type": "text", "text": text}] + [encode_image(p, self.image_max_side) for p in thumbs.images]
        if self.prelude:
            from agenticsfm.controllers.scripted import candidates_and_best, make_step

            step = make_step(ws, cfg, "agent:prelude", verbose=True)
            try:
                cands, best, decisions = candidates_and_best(ws, cfg, step, None)
            except RuntimeError as e:
                content.append({"type": "text", "text": f"The default recipe failed: {e}. Start from build_graph."})
                return content
            ranking = call(ws, cfg, "rank_models", {}, source="agent:prelude")
            best_exam = next((e for m, e in cands if m == best), {})
            content.append({"type": "text", "text": prompts.PRELUDE.format(
                best=best, decisions=json.dumps(decisions, default=str), ranking=json.dumps(for_vlm(ranking.summary), default=str),
                exam=json.dumps(for_vlm(best_exam), default=str))})
            content += [encode_image(p, self.image_max_side) for p in best_exam.get("renders", [])]
        return content

    def run(self, ws: Workspace, cfg: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
        start = time.time()
        system = prompts.system_prompt(self.max_steps, int(self.max_minutes))
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": self._opening(ws, cfg)}]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "turns": 0}
        submitted: str | None = None
        tool_calls = 0
        idle_turns = 0
        forced = False
        while submitted is None:
            minutes_left = self.max_minutes - (time.time() - start) / 60
            calls_left = self.max_steps - tool_calls
            if (calls_left <= 0 or minutes_left <= 0) and not forced:
                messages.append({"role": "user", "content": prompts.FORCE_SUBMIT})
                forced = True
            elif forced:
                break
            elif calls_left <= 3 or minutes_left <= 5:
                messages.append({"role": "user", "content": prompts.BUDGET_WARNING.format(left=calls_left, minutes=minutes_left)})
            try:
                response = self._chat(messages, tool_choice={"type": "function", "function": {"name": "submit"}} if forced else "auto")
            except Exception as e:
                ws.log({"source": "vlm", "error": f"{type(e).__name__}: {e}"})
                break
            msg = response.choices[0].message
            u = response.usage
            usage["prompt_tokens"] += u.prompt_tokens
            usage["completion_tokens"] += u.completion_tokens
            usage["turns"] += 1
            record = msg.model_dump(exclude_none=True)
            reasoning = record.pop("reasoning_content", None) or record.pop("reasoning", None)
            messages.append(record)
            calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in (msg.tool_calls or [])]
            ws.log({"source": "vlm", "content": msg.content, "reasoning": reasoning, "tool_calls": calls,
                    "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}})
            if verbose:
                print(f"[{ws.scene.name}] vlm turn {usage['turns']}: {(msg.content or '')[:300]!r} -> "
                      f"{[(c['name'], c['arguments'][:120]) for c in calls]}", flush=True)
            if not calls:
                idle_turns += 1
                if idle_turns >= 3:
                    break
                messages.append({"role": "user", "content": "Call a tool (one per turn), or submit."})
                continue
            idle_turns = 0
            images_out: list[dict] = []
            for k, c in enumerate(calls):
                if k > 0:
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"skipped": prompts.ONE_TOOL})})
                    continue
                try:
                    params = json.loads(c["arguments"]) if c["arguments"] else {}
                except json.JSONDecodeError as e:
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"error": f"bad JSON arguments: {e}"})})
                    continue
                if c["name"] not in TOOLS:
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"error": f"unknown tool {c['name']}"})})
                    continue
                tool_calls += 1
                result: ToolResult = call(ws, cfg, c["name"], params, source="agent")
                content = {"artifact_id": result.artifact, **for_vlm(result.summary)} if result.artifact else for_vlm(result.summary)
                if result.cached:
                    content["cached"] = True
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(content, default=str)})
                if verbose:
                    print(f"[{ws.scene.name}]   {c['name']} -> {result.artifact} {json.dumps(content, default=str)[:300]}", flush=True)
                if result.images:
                    images_out.append({"type": "text", "text": f"Images from {c['name']}:"})
                    images_out += [encode_image(p, self.image_max_side) for p in result.images]
                if c["name"] == "submit" and not result.error:
                    submitted = params.get("model_id")
            if images_out:
                messages.append({"role": "user", "content": images_out})
        if submitted is None:
            submitted = self._fallback_submit(ws, cfg)
        return {"submitted": submitted, "tool_calls": tool_calls, "forced": forced, "fallback": forced and submitted is not None,
                "usage": usage, "seconds": time.time() - start}

    def _fallback_submit(self, ws: Workspace, cfg: dict[str, Any]) -> str | None:
        models = [a for a in ws.used.values() if a.kind == "model"]
        if not models:
            return None
        best = max(models, key=lambda a: (a.summary["models"][0]["num_images"] if a.summary.get("models") else 0))
        call(ws, cfg, "submit", {"model_id": best.id}, source="harness:fallback")
        return best.id
