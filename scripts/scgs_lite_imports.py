"""Import SC-GS render path without scene/__init__.py (pulls broken pytorch3d)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def bootstrap_scgs_lite(scgs_root: Path) -> None:
    scgs_root = Path(scgs_root).resolve()
    if str(scgs_root) not in sys.path:
        sys.path.insert(0, str(scgs_root))
    if "scene.gaussian_model" in sys.modules:
        return
    scene = types.ModuleType("scene")
    scene.__path__ = [str(scgs_root / "scene")]
    sys.modules["scene"] = scene
    for name, rel in (
        ("gaussian_model", "scene/gaussian_model.py"),
        ("cameras", "scene/cameras.py"),
    ):
        key = f"scene.{name}"
        spec = importlib.util.spec_from_file_location(key, scgs_root / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
