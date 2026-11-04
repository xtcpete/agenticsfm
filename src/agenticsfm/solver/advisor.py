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

    def ask(self, kind: str, parts: list[tuple[str, list[str]]], question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """parts: [(label, [image names]), ...]; the question ends with the JSON schema expected."""
        content: list[dict[str, Any]] = []
        shown = []
        for label, names in parts:
            content.append({"type": "text", "text": label})
            for n in names:
                path = Path(n) if Path(n).is_absolute() else self.images_dir / n
                content.append(encode_image(str(path), self.max_side))
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
              "First say what the middle panel depicts. If it depicts other objects than the photo (a different building, terrain where the photo "
              "shows a wall), or is mostly dark grey or a thin scatter of dots with no recognisable structure, the placement is NOT aligned.")

    def render_verdict(self, figure_path: str, context: dict[str, Any] | None = None) -> str:
        """'aligned' / 'misaligned' / 'unsure' for one render-and-compare figure."""
        parts = [(self.FIGURE + " Figure:", [figure_path])]
        q = 'Is the model aligned with the photo in this figure? Answer {"answer": "aligned" | "misaligned" | "unsure", "confidence": 0-1, "reason": "<one sentence>"}'
        v = self.ask("render_verdict", parts, q, context)
        return str(v.get("answer", "unsure")).strip().lower() if isinstance(v, dict) else "unsure"

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
