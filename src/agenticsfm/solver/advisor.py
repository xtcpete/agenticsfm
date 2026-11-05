"""The VLM advisor of the open solver: judgements the geometry cannot make on its own.

Two questions, both answered in JSON and both logged (question, images, reasoning, answer) so the
decision can be audited against GT afterwards and shown in the episode page:

* ``same_place``: does a candidate image show the same physical place as the images that sit next
  to its proposed camera position? Used when the two feedforward placements of a hard image
  disagree (one anchor set says here, the other says there) and matches cannot arbitrate because
  they are the lookalike matches that made the image hard in the first place.
* ``same_scene``: are two groups of images (two models) photos of one site, or of different sites?
  Decides whether a second model is merged into the first or returned as a separate scene.

The VLM never outputs geometry; its answers gate which feedforward placement is kept.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from agenticsfm.agent import encode_image

SYSTEM = (
    "You are an expert photo analyst assisting a 3D reconstruction system. The photos come from one or more sites that may "
    "contain repeated structure: identical facades, symmetric buildings, repeated towers, windows or corridors. Two photos can "
    "look alike while showing different places, and two photos of the same place can look different because of viewpoint, zoom "
    "or altitude. Judge by physical evidence: the surroundings and what lies beyond the main structure, unique objects, signage, "
    "vegetation, vehicles, shadows and the geometry of the viewpoint. Answer with one JSON object only, no prose outside it."
)


class Advisor:
    def __init__(self, cfg: dict[str, Any], images_dir: Path, log: list[dict[str, Any]], max_side: int = 768, mode: str = "decide"):
        from openai import OpenAI

        vlm = cfg["vlm"]
        self.client = OpenAI(base_url=vlm["base_url"], api_key=vlm["api_key"], timeout=600)
        self.model = vlm["model"]
        self.request = dict(vlm.get("request", {}))
        extra = dict(self.request.get("extra_body", {}))
        ctk = dict(extra.get("chat_template_kwargs", {}))
        ctk["reasoning_effort"] = "low"
        extra["chat_template_kwargs"] = ctk
        self.request["extra_body"] = extra
        self.request["temperature"] = 0.3
        self.images_dir = images_dir
        self.log = log
        self.max_side = max_side
        self.mode = mode          # off | log | decide
        self.calls = 0
        self.seconds = 0.0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def ask(self, kind: str, parts: list[tuple[str, list[str]]], question: str, context: dict[str, Any] | None = None,
            max_side: int | None = None) -> dict[str, Any]:
        """parts: [(label, [image names]), ...]; the question ends with the JSON schema expected."""
        content: list[dict[str, Any]] = []
        shown = []
        for label, names in parts:
            content.append({"type": "text", "text": label})
            for n in names:
                path = Path(n) if Path(n).is_absolute() else self.images_dir / n
                content.append(encode_image(str(path), max_side or self.max_side))
                shown.append(n)
        content.append({"type": "text", "text": question})
        t0 = time.time()
        answer: dict[str, Any] = {"answer": "unsure"}
        reasoning, text, error = None, None, None
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                                                       max_tokens=4096, **self.request)
            msg = resp.choices[0].message
            text = msg.content or ""
            rec = msg.model_dump()
            reasoning = rec.get("reasoning_content") or rec.get("reasoning")
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                answer = json.loads(m.group(0))
        except Exception as e:  # the solver must not die on a VLM hiccup
            error = f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        self.calls += 1
        self.seconds += dt
        row = {"kind": "vlm", "question": kind, "context": context or {}, "images": {label: names for label, names in parts},
               "answer": answer, "reasoning": (reasoning or "")[-1500:], "text": (text or "")[:600], "error": error, "seconds": round(dt, 1), "mode": self.mode}
        self.log.append(row)
        return answer

    def same_place(self, candidate: str, near_a: list[str], near_b: list[str] | None = None, context: dict[str, Any] | None = None) -> str:
        """'A' / 'B' / 'neither' / 'unsure' when two groups are given; 'same' / 'different' / 'unsure' with one."""
        if near_b:
            parts = [("Candidate image:", [candidate]),
                     ("Group A: images taken next to the candidate's first proposed camera position. If that placement is right, they show the same part of the site as the candidate:", near_a),
                     ("Group B: images taken next to the candidate's second proposed camera position:", near_b)]
            q = ('Which group shows the same physical place as the candidate: A, B, or neither? Repeated structure can make both look similar; '
                 'decide from surroundings, unique details and viewpoint. Answer {"answer": "A" | "B" | "neither" | "unsure", "confidence": 0-1, "reason": "<one sentence>"}')
        else:
            parts = [("Candidate image:", [candidate]),
                     ("Images taken next to the candidate's proposed camera position. If the placement is right, they show the same part of the site as the candidate:", near_a)]
            q = ('Does the candidate show the same physical place as these images (the same structure seen from nearby, not a lookalike elsewhere on the site)? '
                 'Answer {"answer": "same" | "different" | "unsure", "confidence": 0-1, "reason": "<one sentence>"}')
        a = self.ask("same_place", parts, q, context)
        return str(a.get("answer", "unsure")).strip().lower() if isinstance(a, dict) else "unsure"

    FIGURE = ("Each figure has three panels. Left: the photo. Middle: what the 3D model shows from this camera's estimated position "
              "(pixels of neighbouring photos projected through the model's geometry; dark grey where the model knows nothing). Right: the "
              "overlay of the two. If the camera is placed correctly, the structures in the middle panel sit on the same structures in the "
              "photo: building edges, roofs, roads and ground coincide in the overlay (small blur is fine). If it is placed wrongly, the model "
              "shows a different place, or the same place shifted, rotated or at a different scale, and the overlay shows doubled edges. "
              "If the middle panel depicts other objects than the photo (a different building, terrain where the photo shows a wall), the placement is "
              "NOT aligned. A sparse rendering is normal: judge where its points fall, not how many there are.")

    def render_verdict(self, figure_path: str, context: dict[str, Any] | None = None) -> str:
        """'aligned' / 'misaligned' / 'unsure' for one render-and-compare figure."""
        parts = [(self.FIGURE + " Figure:", [figure_path])]
        q = 'Is the model aligned with the photo in this figure? Answer {"answer": "aligned" | "misaligned" | "unsure", "confidence": 0-1, "reason": "<one sentence>"}'
        v = self.ask("render_verdict", parts, q, context)
        return str(v.get("answer", "unsure")).strip().lower() if isinstance(v, dict) else "unsure"

    def render_verdicts(self, figures: list[str], context: dict[str, Any] | None = None) -> list[str]:
        """One call for several images of a round: 'aligned' / 'misaligned' / 'unsure' per figure, in order."""
        parts = [(self.FIGURE, [])] + [(f"Figure {k + 1}:", [f]) for k, f in enumerate(figures)]
        q = ('For each figure, is the model aligned with the photo? Answer {"verdicts": [{"figure": 1, "answer": "aligned" | "misaligned" | "unsure", '
             '"reason": "<a few words>"}, ...]} with one entry per figure, in order.')
        v = self.ask("render_verdicts", parts, q, context)
        out = ["unsure"] * len(figures)
        if isinstance(v, dict) and isinstance(v.get("verdicts"), list):
            for e in v["verdicts"]:
                try:
                    k = int(e.get("figure")) - 1
                    if 0 <= k < len(out):
                        out[k] = str(e.get("answer", "unsure")).strip().lower()
                except (TypeError, ValueError, AttributeError):
                    pass
        return out

    def render_choose(self, figure_a: str, figure_b: str, context: dict[str, Any] | None = None) -> str:
        """'a' / 'b' / 'neither' / 'unsure': which of two candidate placements renders consistently with the photo."""
        parts = [(self.FIGURE + " Figure A (first candidate placement):", [figure_a]), ("Figure B (second candidate placement):", [figure_b])]
        q = ('Which placement is aligned with the photo: A, B, both, or neither? Neither is the right answer whenever no rendering shows the '
             'photo\'s own structures in place. Answer {"answer": "A" | "B" | "both" | "neither" | "unsure", "confidence": 0-1, "middle_panels": "<what A and B depict>", "reason": "<one sentence>"}')
        v = self.ask("render_choose", parts, q, context)
        return str(v.get("answer", "unsure")).strip().lower() if isinstance(v, dict) else "unsure"

    def same_scene(self, images_a: list[str], images_b: list[str], context: dict[str, Any] | None = None) -> str:
        parts = [("Group A: a sample of the images in reconstruction A:", images_a), ("Group B: a sample of the images in reconstruction B:", images_b)]
        q = ('Are groups A and B photos of the same site (the same place or building complex, such that one reconstruction could contain both), '
             'or of different sites? Answer {"answer": "same" | "different" | "unsure", "confidence": 0-1, "reason": "<one sentence>"}')
        a = self.ask("same_scene", parts, q, context)
        return str(a.get("answer", "unsure")).strip().lower() if isinstance(a, dict) else "unsure"

    def sheet(self, names: list[str], out: Path, labels: list[str] | None = None, tile: int = 360, cols: int = 4) -> str:
        """Contact sheet of images (numbered when labels are given), so many images reach the VLM in one picture."""
        from PIL import Image, ImageDraw, ImageFont

        out.parent.mkdir(parents=True, exist_ok=True)
        rows = (len(names) + cols - 1) // cols
        th = int(tile * 0.75)
        canvas = Image.new("RGB", (cols * tile, rows * th), (20, 20, 20))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        except OSError:
            font = ImageFont.load_default()
        for a, n in enumerate(names):
            im = Image.open(self.images_dir / n)
            im.draft("RGB", (tile * 2, th * 2))
            im = im.convert("RGB")
            im.thumbnail((tile - 6, th - 6))
            x, y = (a % cols) * tile + (tile - im.width) // 2, (a // cols) * th + (th - im.height) // 2
            canvas.paste(im, (x, y))
            if labels:
                d = ImageDraw.Draw(canvas)
                lx, ly = (a % cols) * tile + 6, (a // cols) * th + 6
                d.rectangle([lx, ly, lx + 58, ly + 42], fill=(255, 210, 0))
                d.text((lx + 6, ly + 2), labels[a], fill=(0, 0, 0), font=font)
        canvas.save(out, quality=90)
        return str(out)

    def select_next(self, model_sheet: str, cand_sheet: str, num: int, pick: int, context: dict[str, Any] | None = None) -> tuple[list[int], dict[str, Any]]:
        """Which candidates should join the reconstruction next. Returns 0-based candidate indices (best first) and the answer."""
        parts = [("Images already reconstructed (the part of the 3D model the candidates connect to):", [model_sheet]),
                 (f"Candidate images, numbered 1 to {num} (yellow labels):", [cand_sheet])]
        q = (f"The reconstruction grows a few images at a time; each new image is placed next to the reconstructed images it overlaps. Pick up to {pick} "
             "candidates to add next: images that clearly show part of the same place as the reconstructed images, with plenty of overlap, so their "
             "placement will be reliable. Skip candidates that show a different place that only looks similar (a repeated facade, a twin building, "
             "a different street), and candidates with little usable content (mostly sky, water, blur, or a featureless close-up); they are tried "
             "later, when more of the model exists. Answer {\"pick\": [candidate numbers, best first], \"skip\": [{\"n\": number, \"why\": \"<a few words>\"}, ...]}")
        a = self.ask("select_next", parts, q, context, max_side=1440)
        out: list[int] = []
        if isinstance(a, dict) and isinstance(a.get("pick"), list):
            for v in a["pick"]:
                try:
                    k = int(v) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= k < num and k not in out:
                    out.append(k)
        return out[:pick], a
