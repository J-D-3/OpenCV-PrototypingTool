"""Locate and load the optional OPTICS-Clustering Python binding (``optics_py``).

`optics_py` is a compiled pybind11 module built from the sibling **OPTICS-Clustering**
repo (density-based clustering: OPTICS + HDBSCAN*). It is *not* on PyPI and is ABI-locked
to the Python version/platform it was built for, so we treat it as an **optional** backend:
the HDBSCAN node calls :func:`load` and, when the module is missing, receives a clear error
to surface in the UI (red node border / status bar) instead of crashing.

Search order: already-importable → ``$OPTICS_PY_DIR`` → the sibling repo's default build
output (``../OPTICS-Clustering/build-py/python/Release``). Kept Qt-free so it stays in
``core/``. See the project memory note "optics-clustering-integration" for build details.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

_module = None
_error: Optional[str] = None
_tried = False

# The build subpath the OPTICS repo's CMake produces (MSVC puts it in a Release/ dir).
_REL = os.path.join("OPTICS-Clustering", "build-py", "python", "Release")


def _candidate_dirs():
    """Directories that may hold ``optics_py.*.pyd`` / ``.so``, most specific first."""
    dirs = []
    env = os.environ.get("OPTICS_PY_DIR")
    if env:
        dirs.append(env)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(project_root)            # sibling repos live one level up
    dirs.append(os.path.join(parent, _REL))           # ../OPTICS-Clustering/build-py/...
    dirs.append(os.path.join(project_root, _REL))     # a checkout nested inside this repo
    return dirs


def available() -> bool:
    """True if the binding can be loaded (so the UI can probe without raising)."""
    try:
        load()
        return True
    except Exception:
        return False


def load():
    """Return the ``optics_py`` module, importing it on first use.

    Raises ``RuntimeError`` with an actionable message if it cannot be found or loaded.
    The result (or the error) is cached, so repeated calls are cheap."""
    global _module, _error, _tried
    if _module is not None:
        return _module
    if _tried and _error is not None:
        raise RuntimeError(_error)
    _tried = True

    try:
        _module = importlib.import_module("optics_py")
        return _module
    except Exception:
        pass  # not yet on the path — try the known build locations below

    for d in _candidate_dirs():
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        _module = importlib.import_module("optics_py")
        return _module
    except Exception as e:
        _error = (
            f"OPTICS clustering library (optics_py) not available: {e}. "
            "Build it in the OPTICS-Clustering repo (cmake -DOPTICS_BUILD_PYTHON=ON) or "
            "set OPTICS_PY_DIR to the folder holding optics_py.*.pyd."
        )
        raise RuntimeError(_error)
