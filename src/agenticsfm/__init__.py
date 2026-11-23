"""Agentic SfM: a VLM that evaluates reconstructions and chooses SfM tools."""
import sys as _sys
from pathlib import Path as _Path

# third_party/rayba (git submodule): the GPU bundle adjustment, importable as `rayba` without installing it
_RAYBA = _Path(__file__).resolve().parents[2] / "third_party" / "rayba"
if _RAYBA.is_dir() and str(_RAYBA) not in _sys.path:
    _sys.path.insert(0, str(_RAYBA))
