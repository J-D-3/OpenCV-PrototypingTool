"""Diagnostics — crash-hunting instrumentation (backend, Qt-free).

The app occasionally crashes hard (the window vanishes) while loading a pipeline
or recomputing a node. Because :mod:`core.engine` wraps every ``compute`` in a
``try/except``, a *Python* exception can never do that — it becomes a red node
border. A hard crash is therefore a **native** event: a segfault / access
violation inside a C extension (OpenCV, the ``optics`` pybind11 core), or the
process being torn down. Those leave **no Python traceback** and discard any
buffered ``print`` output, so by the time the user notices, the evidence is gone.

This module gives us that evidence:

* **faulthandler** — on a fatal signal (SIGSEGV/SIGABRT/access violation, which
  faulthandler also catches on Windows) it writes *every thread's* Python stack
  to ``logs/faulthandler.log``. That single dump tells us which op each thread
  was inside at the instant of the crash — the whole game when the bug is a
  concurrent native call.
* **thread-tagged logging** of evaluation start/end and structural edits, flushed
  per line to ``logs/diag.log`` so the last lines survive the crash.
* a **concurrent-evaluation detector**: :func:`evaluation_guard` warns (always,
  cheaply) whenever two threads enter engine evaluation at once — the prime
  suspect for the crash. This fires *before* the segfault, naming both threads.

Everything here is stdlib-only (no Qt, no cv2), so ``core`` stays headless.

Usage
-----
Call :func:`init` once at startup (``app.main`` does). Verbose per-node timing is
gated behind the ``OCVPT_DIAG`` env var (set it to ``1`` during a crash hunt);
the faulthandler dump and the concurrency warning are always on once ``init`` has
run, because they are nearly free and only speak up when something is wrong.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Verbose per-node/per-element timing. Off by default (keeps normal runs and the
# headless test suites silent); set OCVPT_DIAG=1 to capture a full timing trace.
VERBOSE = os.environ.get("OCVPT_DIAG", "") not in ("", "0", "false", "False")

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


def init(verbose: Optional[bool] = None) -> None:
    """Set up faulthandler + file logging. Idempotent; safe to call repeatedly.

    Called from ``app.main``. The headless test suites do **not** call it, so they
    create no ``logs/`` dir and stay silent — but :func:`evaluation_guard` and
    :func:`log` still work (logging just has no handler, so they're no-ops)."""
    global _initialized, _fault_fp
    if _initialized:
        return
    _initialized = True
    if verbose is not None:
        global VERBOSE
        VERBOSE = verbose

    log_dir = _log_dir()

    # faulthandler: dump every thread's stack on a fatal signal. Keep the file
    # handle open for the whole process — faulthandler writes to this fd directly
    # from the signal handler, so it must outlive init().
    _fault_fp = open(log_dir / "faulthandler.log", "a", buffering=1)
    _fault_fp.write(f"\n--- session start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    _fault_fp.flush()
    faulthandler.enable(file=_fault_fp, all_threads=True)

    # Rotating file handler (per-line flush via RotatingFileHandler's default) +
    # stderr, so a trailing trace is both on disk and visible in the console.
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
    _log.info("diag initialized (verbose=%s, pid=%d) -> %s", VERBOSE, os.getpid(), log_dir)


def dump_now(label: str = "manual") -> None:
    """Dump every thread's stack right now (for diagnosing a hang, not a crash)."""
    if _fault_fp is not None:
        _fault_fp.write(f"\n--- manual dump '{label}' {time.strftime('%H:%M:%S')} ---\n")
        _fault_fp.flush()
        faulthandler.dump_traceback(file=_fault_fp, all_threads=True)


def thread_tag() -> str:
    t = threading.current_thread()
    return f"{t.name}#{t.ident}"


def log(msg: str, *, level: int = logging.INFO) -> None:
    """Thread-tagged log line (no-op if init() was never called and no handlers)."""
    if _log.handlers:
        _log.log(level, "[%s] %s", thread_tag(), msg)


@contextmanager
def evaluation_guard(what: str):
    """Wrap one engine evaluation pass. Always-on, cheap: registers this thread as
    'evaluating' and **warns if another thread is already evaluating** — the exact
    overlap we suspect is causing the native crash. Names both threads so the log
    (and any following faulthandler dump) pin down the race."""
    tid = threading.get_ident()
    tag = thread_tag()
    with _active_lock:
        others = {k: v for k, v in _active.items() if k != tid}
        _active[tid] = what
    if others:
        # This is the smoking gun: two evaluations on the engine at once.
        if _log.handlers:
            _log.warning("[%s] CONCURRENT EVALUATION: '%s' starting while %s already running",
                         tag, what, list(others.values()))
        else:  # logging not initialized (e.g. tests) — still surface it
            import sys
            print(f"[diag] CONCURRENT EVALUATION: {tag} '{what}' overlaps {list(others.values())}",
                  file=sys.stderr)
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
    """Verbose-only timing of a compute. Near-zero cost when OCVPT_DIAG is unset."""
    if not VERBOSE:
        yield
        return
    t0 = time.perf_counter()
    log(f"{label} start")
    try:
        yield
    finally:
        log(f"{label} end ({(time.perf_counter() - t0) * 1000.0:.1f} ms)")
