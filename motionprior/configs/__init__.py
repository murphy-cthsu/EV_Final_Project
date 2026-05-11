"""YAML config loader for motionprior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_CONFIG_DIR = str(Path(__file__).parent)


def _resolve_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".yaml" and p.exists():
        return p
    candidate = Path(REPO_CONFIG_DIR) / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve config '{name_or_path}'. "
        f"Tried '{p}' and '{candidate}'."
    )


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name_or_path: str) -> dict[str, Any]:
    """Load a YAML config, deep-merging on top of default.yaml.

    `name_or_path` accepts either a bare name (e.g. "dnerf_jumpingjacks") which
    resolves to <REPO_CONFIG_DIR>/<name>.yaml, or an absolute/relative path to
    a YAML file.
    """
    path = _resolve_path(name_or_path)
    default_path = Path(REPO_CONFIG_DIR) / "default.yaml"

    with open(default_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if path.resolve() != default_path.resolve():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)

    return cfg


__all__ = ["load_config", "REPO_CONFIG_DIR"]
