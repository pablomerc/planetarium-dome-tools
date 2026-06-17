#!/usr/bin/env python3
"""Shared project path helpers for scripts in this repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = ROOT / "resources"
OUTPUTS_DIR = ROOT / "outputs"

_INPUT_DIRS = (
    ROOT,
    RESOURCES_DIR,
    RESOURCES_DIR / "tess",
    OUTPUTS_DIR,
    OUTPUTS_DIR / "masters",
    OUTPUTS_DIR / "videos",
    OUTPUTS_DIR / "previews",
    OUTPUTS_DIR / "frames",
    OUTPUTS_DIR / "debug",
)


def project_path(value):
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def existing_path(value):
    """Find an input path in the project root, resources, or output folders."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    direct = ROOT / path
    if direct.exists():
        return direct

    if len(path.parts) > 1:
        for base in (RESOURCES_DIR, OUTPUTS_DIR):
            candidate = base / path
            if candidate.exists():
                return candidate
        return direct

    for base in _INPUT_DIRS[1:]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return direct


def output_path(value, category=None):
    """Resolve an output path, placing bare filenames under outputs[/category]."""
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path
    elif path.parent == Path("."):
        base = OUTPUTS_DIR / category if category else OUTPUTS_DIR
        resolved = base / path
    else:
        resolved = ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def output_dir(value, category=None):
    """Resolve an output directory, placing bare names under outputs[/category]."""
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path
    elif path.parent == Path("."):
        base = OUTPUTS_DIR / category if category else OUTPUTS_DIR
        resolved = base / path
    else:
        resolved = ROOT / path
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resource_output_path(value):
    """Resolve a generated resource path, placing bare filenames under resources."""
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path
    elif path.parent == Path("."):
        resolved = RESOURCES_DIR / path
    else:
        resolved = ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def display_path(value):
    """Return a compact path for log messages."""
    path = Path(value)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
