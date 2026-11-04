"""Tools: typed operations on the workspace. Each has a JSON schema (what the agent sees) and a
``run(ws, cfg, **params) -> ToolResult``. Results are artifacts; repeated calls with the same
parameters and inputs return the cached artifact. ``call`` runs a tool and writes the trace.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from agenticsfm.workspace import Artifact, Workspace


@dataclass
class ToolResult:
    artifact: str | None                 # main output artifact id (None for submit / inspect)
    summary: dict[str, Any]              # what the agent reads
    cached: bool = False
    cost: dict[str, float] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)   # paths of renders for the VLM
    error: str | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON schema
    run: Callable[..., ToolResult]

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


TOOLS: dict[str, Tool] = {}


def register(name: str, description: str, parameters: dict[str, Any]) -> Callable:
    def deco(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        TOOLS[name] = Tool(name, description, parameters, fn)
        return fn
    return deco


def cached_result(ws: Workspace, id: str) -> ToolResult | None:
    if ws.has(id):
        art = ws.get(id)
        return ToolResult(id, dict(art.summary), cached=True, cost={})
    return None


def finish(ws: Workspace, art: Artifact, started: float, images: list[str] | None = None) -> ToolResult:
    art.cost = {"wall": time.time() - started, **art.cost}
    ws.register(art)
    return ToolResult(art.id, dict(art.summary), cost=dict(art.cost), images=images or [])


def call(ws: Workspace, cfg: dict[str, Any], name: str, params: dict[str, Any], source: str = "controller") -> ToolResult:
    """Run a tool and log the call."""
    ws.step += 1
    started = time.time()
    tool = TOOLS[name]
    try:
        result = tool.run(ws, cfg, **params)
    except Exception as e:  # the agent gets the error as a result, the trace keeps it
        result = ToolResult(None, {"error": f"{type(e).__name__}: {e}"}, error=traceback.format_exc())
    record = {"source": source, "tool": name, "params": params, "artifact": result.artifact,
              "cached": result.cached, "summary": result.summary, "cost": {"wall": time.time() - started, **result.cost},
              "images": result.images}
    if result.error:
        record["error"] = result.error
    ws.log(record)
    return result


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


from agenticsfm.tools import analyze, compare_models, examine, filter_edges, graph, inspect, match, merge_models, optimize, partition_graph, rank_models, reconstruct, reconstruct_parts, score_pairs, submit, verify_poses, view_graph  # noqa: E402,F401

for _name, _tool in TOOLS.items():  # a decorator placed before a helper would register the wrong function
    assert _tool.run.__name__ == _name, f"tool {_name} is bound to {_tool.run.__name__}"

__all__ = ["TOOLS", "Tool", "ToolResult", "cached_result", "call", "finish", "register", "schemas"]
