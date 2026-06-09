"""Locate and load the optional **OPTICS-Clustering** Python package (``optics``).

`optics` is a pip-installable package (compiled `_optics` pybind11 core + a high-level
NumPy/OpenCV colour API) built from the sibling **OPTICS-Clustering** repo. It is *not* on
PyPI, so we treat it as an **optional** backend: the Density Cluster node calls :func:`load`
and, when the package is missing, receives a clear error to surface in the UI (red node
border / status bar) instead of crashing.

The high-level entry point is :func:`optics.cluster_image` — it dedups, voxel-quantizes, and
converts colour spaces (sRGB→CIELAB) internally, so the node just hands it the BGR image.

Install it with ``pip install <OPTICS-Clustering>/python``. We import ``optics`` directly; as
a fallback we add ``$OPTICS_PY_DIR`` (or the sibling repo's ``python/`` dir) to ``sys.path``,
which works when the package was built in place. See the project memory note
"optics-clustering-integration".
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

_module = None
_error: Optional[str] = None
_tried = False


def _candidate_dirs():
    """Dirs that may contain the ``optics`` package (for an in-place / non-installed build)."""
    dirs = []
    env = os.environ.get("OPTICS_PY_DIR")
    if env:
        dirs.append(env)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(project_root)            # sibling repos live one level up
    dirs.append(os.path.join(parent, "OPTICS-Clustering", "python"))
    return dirs


def _import_optics():
    """Import ``optics`` and verify it is the new high-level API (has ``cluster_image``)."""
    mod = importlib.import_module("optics")
    if not hasattr(mod, "cluster_image"):
        raise ImportError("the importable 'optics' package lacks cluster_image (stale/old build)")
    return mod


def available() -> bool:
    """True if the package can be loaded (so the UI can probe without raising)."""
    try:
        load()
        return True
    except Exception:
        return False


def load():
    """Return the ``optics`` module, importing it on first use.

    Raises ``RuntimeError`` with an actionable message if it cannot be found or loaded.
    The result (or the error) is cached, so repeated calls are cheap."""
    global _module, _error, _tried
    if _module is not None:
        return _module
    if _tried and _error is not None:
        raise RuntimeError(_error)
    _tried = True

    try:
        _module = _import_optics()
        return _module
    except Exception:
        pass  # not installed — try an in-place build location below

    for d in _candidate_dirs():
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        _module = _import_optics()
        return _module
    except Exception as e:
        _error = (
            f"OPTICS clustering library (the 'optics' package) not available: {e}. "
            "Install it with 'pip install <OPTICS-Clustering>/python', or set OPTICS_PY_DIR "
            "to a folder containing an in-place build of the 'optics' package."
        )
        raise RuntimeError(_error)
