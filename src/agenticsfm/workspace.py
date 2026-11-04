"""Workspace: the artifact store of one scene and the trace of one run.

Artifacts are content-addressed: an artifact's ID is a hash of the tool, its parameters and the
IDs of its input artifacts, rooted in a hash of the scene's images. The store
(``<cache>/<scene_id>/<kind>/<id>``) is shared by every run on the scene, so a second controller,
or a second agent episode, re-uses matches and models computed before. Each run keeps its own
``trace.jsonl`` (every tool call: parameters, artifact, summary, cost) and ``artifacts.json`` (the
artifacts it produced or re-used, with provenance).

Nothing here reads ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenticsfm.config import REPO_ROOT
from agenticsfm.datasets import Scene

KINDS = ("graph", "features", "matches", "db", "scores", "model", "examine")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj: Any, n: int = 10) -> str:
    return hashlib.sha1(canonical(obj).encode()).hexdigest()[:n]


def scene_id(scene: Scene) -> str:
    """Hash of the image names and sizes (the content the tools see)."""
    entries = [(name, os.path.getsize(scene.images_dir / name)) for name in scene.image_names()]
    return digest(entries, 12)


@dataclass
class Artifact:
    id: str
    kind: str
    tool: str
    params: dict[str, Any]
    inputs: dict[str, str]           # role -> artifact id
    path: str                        # relative to the store root
    summary: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    created: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Workspace:
    def __init__(self, scene: Scene, run_dir: str | Path, cache_root: str | Path, gpus: list[int] | None = None):
        self.scene = scene
        self.scene_id = scene_id(scene)
        self.store = Path(cache_root) / self.scene_id
        self.run_dir = Path(run_dir)
        self.gpus = list(gpus or [])
        self.store.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        images = self.store / "images"
        if not images.is_symlink():
            images.symlink_to(scene.images_dir.resolve())
        (self.store / "scene.json").write_text(json.dumps(
            {"name": scene.name, "images_dir": str(scene.images_dir), "num_images": len(scene.image_names()),
             "hints": scene.hints}, indent=2))
        self.registry_path = self.store / "artifacts.json"
        self.registry: dict[str, Artifact] = {}
        if self.registry_path.is_file():
            self.registry = {k: Artifact(**v) for k, v in json.loads(self.registry_path.read_text()).items()}
        self.trace_path = self.run_dir / "trace.jsonl"
        self.used: dict[str, Artifact] = {}
        self.step = 0

    # -- artifacts ---------------------------------------------------------------------------
    @property
    def images_dir(self) -> Path:
        return self.store / "images"

    def image_names(self) -> list[str]:
        return self.scene.image_names()

    def artifact_id(self, tool: str, params: dict[str, Any], inputs: dict[str, str]) -> str:
        return digest({"scene": self.scene_id, "tool": tool, "params": params, "inputs": inputs})

    def path(self, kind: str, id: str, suffix: str = "") -> Path:
        d = self.store / kind
        d.mkdir(exist_ok=True)
        return d / f"{id}{suffix}"

    def has(self, id: str) -> bool:
        art = self.registry.get(id)
        return art is not None and (self.store / art.path).exists()

    def get(self, id: str) -> Artifact:
        if id not in self.registry:
            raise KeyError(f"unknown artifact {id}")
        return self.registry[id]

    def abspath(self, id: str) -> Path:
        return self.store / self.get(id).path

    def register(self, art: Artifact) -> Artifact:
        art.created = art.created or time.time()
        self.registry[art.id] = art
        self._save_registry()
        return art

    def reload_registry(self) -> None:
        if self.registry_path.is_file():
            try:
                for k, v in json.loads(self.registry_path.read_text()).items():
                    self.registry.setdefault(k, Artifact(**v))
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_registry(self) -> None:
        # Several runs may share the store: merge with what others registered since we loaded it.
        if self.registry_path.is_file():
            try:
                on_disk = {k: Artifact(**v) for k, v in json.loads(self.registry_path.read_text()).items()}
                for k, v in on_disk.items():
                    self.registry.setdefault(k, v)
            except (json.JSONDecodeError, TypeError):
                pass
        tmp = self.registry_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({k: v.to_json() for k, v in self.registry.items()}, indent=1))
        tmp.replace(self.registry_path)

    def lock(self, id: str):
        """Exclusive file lock on an artifact id: concurrent runs compute each artifact once."""
        import fcntl
        from contextlib import contextmanager

        @contextmanager
        def _lock():
            d = self.store / "locks"
            d.mkdir(exist_ok=True)
            with open(d / f"{id}.lock", "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        return _lock()

    def provenance(self, id: str) -> list[Artifact]:
        """The artifact and its ancestors, nearest first."""
        out, todo = [], [id]
        while todo:
            art = self.get(todo.pop(0))
            if art in out:
                continue
            out.append(art)
            todo.extend(art.inputs.values())
        return out

    def ancestor(self, id: str, kind: str) -> Artifact | None:
        for art in self.provenance(id):
            if art.kind == kind:
                return art
        return None

    def list(self, kind: str | None = None) -> list[Artifact]:
        return [a for a in self.registry.values() if kind is None or a.kind == kind]

    # -- trace -------------------------------------------------------------------------------
    def log(self, record: dict[str, Any]) -> None:
        record = {"step": self.step, "time": time.time(), **record}
        with open(self.trace_path, "a") as f:
            f.write(canonical(record) + "\n")
        if "artifact" in record and record["artifact"] in self.registry:
            self.used[record["artifact"]] = self.registry[record["artifact"]]
            (self.run_dir / "artifacts.json").write_text(
                json.dumps({k: v.to_json() for k, v in self.used.items()}, indent=1))

    def read_trace(self) -> list[dict[str, Any]]:
        if not self.trace_path.is_file():
            return []
        return [json.loads(line) for line in self.trace_path.read_text().splitlines() if line]


def run_dir_for(cfg: dict[str, Any], run: str, scene: Scene) -> Path:
    return REPO_ROOT / cfg["paths"]["runs"] / run / scene.name


def make_workspace(cfg: dict[str, Any], run: str, scene: Scene, gpus: list[int] | None = None) -> Workspace:
    return Workspace(scene, run_dir_for(cfg, run, scene), REPO_ROOT / cfg["paths"]["cache"],
                     gpus if gpus is not None else cfg["compute"]["tool_gpus"])
