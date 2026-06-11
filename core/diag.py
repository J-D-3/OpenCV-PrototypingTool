"""Diagnostics — opt-in crash/hang instrumentation (backend, Qt-free).

Why this exists
---------------
The app used to crash hard (window vanishes) or hang while loading a pipeline or
recomputing. Because :mod:`core.engine` wraps every ``compute`` in a
``try/except``, a *Python* exception can never do that — it becomes a red node
border. Such failures are **native** events (a segfault in OpenCV / the ``optics``
pybind11 core / OpenBLAS, or a deadlock) that leave no Python traceback and
discard buffered output. This module captures the evidence those failures need:

* **faulthandler** — on a fatal signal (SIGSEGV/SIGABRT/access violation, caught
  on Windows too) it writes *every thread's* Python stack, so we can see which op
  each thread was inside at the instant of the crash.
* **thread-tagged logging** of evaluation + background-worker + structural-edit
  events, flushed per line so the trailing lines survive a crash.
* a **concurrent-evaluation detector** (:func:`evaluation_guard`) that warns when
  two threads evaluate at once.

Everything here is stdlib-only (no Qt, no cv2), so ``core`` stays headless.

Turning it on (it's **off by default**)
---------------------------------------
Set the ``OCVPT_DIAG`` environment variable before launching:

* unset / ``0`` / ``off`` — **off**. No ``logs/`` dir, no handlers; logging calls
  are no-ops. faulthandler still arms itself to ``stderr`` (free crash insurance).
* ``1`` / ``on`` / ``info`` — **on**. INFO logging (worker start/end, structural
  edits, concurrency warnings) to ``logs/diag.log`` + stderr; faulthandler dumps
  to ``logs/faulthandler.log``.
* ``2`` / ``verbose`` / ``trace`` — adds **per-node / per-eval-pass timing**.

``init`` may also be called with an explicit ``level`` to override the env var
(e.g. from a future "Debug logging" menu toggle). It is idempotent.

    PowerShell:  $env:OCVPT_DIAG=1; .\.venv\Scripts\python.exe main.py
"""
from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# --- level resolution --------------------------------------------------------
OFF, ON, VERBOSE_LEVEL = 0, 1, 2
_LEVEL_WORDS = {
    "": OFF, "0": OFF, "off": OFF, "false": OFF, "no": OFF, "none": OFF,
    "1": ON, "on": ON, "true": ON, "yes": ON, "info": ON,
    "2": VERBOSE_LEVEL, "verbose": VERBOSE_LEVEL, "trace": VERBOSE_LEVEL, "debug": VERBOSE_LEVEL,
}


def _resolve_level(value: Optional[str]) -> int:
    """Map an OCVPT_DIAG value to a level; unknown non-empty values mean ``ON``."""
    if value is None:
        return OFF
    return _LEVEL_WORDS.get(value.strip().lower(), ON)


LEVEL = _resolve_level(os.environ.get("OCVPT_DIAG"))
ENABLED = LEVEL >= ON          # file logging + INFO events
VERBOSE = LEVEL >= VERBOSE_LEVEL  # adds per-node / per-pass timing

_log = logging.getLogger("ocvpt")
_initialized = False
_fault_fp = None  # kept open for the process lifetime so the dump can be written

# Concurrency tracking: which threads are currently inside engine evaluation.
_active: dict[int, str] = {}
_active_lock = threading.Lock()


def _log_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "logs"
    d.mkdir(exist_ok=True)
    return d


def init(level: Optional[int] = None) -> None:
    """Arm faulthandler and (when enabled) file logging. Idempotent.

    ``app.main`` calls this once. ``level`` overrides the ``OCVPT_DIAG`` env var
    when given (``diag.OFF`` / ``diag.ON`` / ``diag.VERBOSE_LEVEL``) — handy for a
    future in-app "enable debug logging" toggle. When the level is OFF we create no
    ``logs/`` dir and add no handlers (so logging calls stay no-ops), but still arm
    faulthandler to stderr so a surprise native crash is at least dumped there."""
    global _initialized, _fault_fp, LEVEL, ENABLED, VERBOSE
    if _initialized:
        return
    _initialized = True
    if level is not None:
        LEVEL = level
        ENABLED = LEVEL >= ON
        VERBOSE = LEVEL >= VERBOSE_LEVEL

    if not ENABLED:
        # Off: no files, no handlers. Free crash insurance to the console only.
        faulthandler.enable(file=sys.stderr, all_threads=True)
        return

    log_dir = _log_dir()

    # faulthandler: dump every thread's stack on a fatal signal. Keep the file
    # handle open for the whole process — faulthandler writes to this fd directly
    # from the signal handler, so it must outlive init().
    _fault_fp = open(log_dir / "faulthandler.log", "a", buffering=1)
    _fault_fp.write(f"\n--- session start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    _fault_fp.flush()
    faulthandler.enable(file=_fault_fp, all_threads=True)

    # Rotating file handler + stderr, so a trailing trace is both on disk and
    # visible in the console.
    _log.setLevel(logging.DEBUG if VERBOSE else logging.INFO)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S")
    fh = RotatingFileHandler(log_dir / "diag.log", maxBytes=2_000_000,
                             backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    _log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    _log.addHandler(sh)
    _log.propagate = False
    _log.info("diag initialized (level=%d, pid=%d) -> %s", LEVEL, os.getpid(), log_dir)


def is_enabled() -> bool:
    """True if INFO-level diagnostics are active (callers can skip building costly
    log strings when this is False)."""
    return ENABLED


def dump_now(label: str = "manual") -> None:
    """Dump every thread's stack right now — for diagnosing a *hang* (not a crash).
    Writes to the faulthandler log when enabled, else to stderr."""
    target = _fault_fp if _fault_fp is not None else sys.stderr
    target.write(f"\n--- manual dump '{label}' {time.strftime('%H:%M:%S')} ---\n")
    target.flush()
    faulthandler.dump_traceback(file=target, all_threads=True)


def thread_tag() -> str:
    t = threading.current_thread()
    return f"{t.name}#{t.ident}"


def log(msg: str, *, level: int = logging.INFO) -> None:
    """Thread-tagged log line. No-op when diagnostics are off (no handlers)."""
    if _log.handlers:
        _log.log(level, "[%s] %s", thread_tag(), msg)


def nodes_summary(nodes, limit: int = 8) -> str:
    """Compact ``id:op_id`` list for logging which nodes a worker (re)computed,
    e.g. ``"3:to_grayscale, 4:blur, 7:hdbscan_cluster ...+2"``."""
    parts = []
    for n in nodes[:limit]:
        op = getattr(n, "op", None)
        op_id = "source" if op is None else getattr(op, "id", "?")
        parts.append(f"{getattr(n, 'id', '?')}:{op_id}")
    extra = len(nodes) - limit
    if extra > 0:
        parts.append(f"...+{extra}")
    return ", ".join(parts) if parts else "none"


@contextmanager
def evaluation_guard(what: str):
    """Wrap one engine evaluation pass. Cheap and always active: registers this
    thread as 'evaluating' and **warns if another thread is already evaluating** —
    the overlap that can corrupt evaluation. Names both threads so the log (and any
    following faulthandler dump) pin down the race. When diagnostics are off, an
    overlap still prints once to stderr (it should never happen)."""
    tid = threading.get_ident()
    tag = thread_tag()
    with _active_lock:
        others = {k: v for k, v in _active.items() if k != tid}
        _active[tid] = what
    if others:
        msg = f"CONCURRENT EVALUATION: '{what}' on {tag} overlaps {list(others.values())}"
        if _log.handlers:
            _log.warning("[%s] %s", tag, msg)
        else:
            print(f"[diag] {msg}", file=sys.stderr)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        with _active_lock:
            _active.pop(tid, None)
        if VERBOSE:
            log(f"eval-pass '{what}' done in {dt:.1f} ms")


@contextmanager
def timed(label: str):
    """Verbose-only timing of a compute. Near-zero cost unless OCVPT_DIAG>=2."""
    if not VERBOSE:
        yield
        return
    t0 = time.perf_counter()
    log(f"{label} start")
    try:
        yield
    finally:
        log(f"{label} end ({(time.perf_counter() - t0) * 1000.0:.1f} ms)")
