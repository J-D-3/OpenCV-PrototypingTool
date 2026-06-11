"""Pin the BLAS backend (OpenBLAS) to a single thread — import FIRST, before numpy.

**Why this exists.** numpy here is backed by OpenBLAS. The engine evaluates a
node's batch by mapping the op across a ``ThreadPoolExecutor`` (see
``core/engine._run_batched``) — many Python threads running OpenCV/numpy at once.
That is safe for OpenCV, but OpenBLAS's *own* internal thread pool is **not
re-entrant across concurrent caller threads**: when several fan-out threads each
call a BLAS routine (e.g. the ``@`` matmul in ``_srgb_to_lab``) at the same time,
OpenBLAS intermittently **deadlocks (hang) or segfaults (hard crash)**. This was
the cause of both the "occasionally crashes on load/recompute" and the
"recompute never finishes" reports — same root cause, two faces, dependent on
timing (hence non-reproducible). Verified: 8 threads × concurrent matmul segfault
without this; finish cleanly with it.

**Why pinning to 1 is the right fix, not a workaround.** Parallelism here comes
from the *batch fan-out* (one thread per image), not from BLAS splitting a single
matmul. So OpenBLAS's intra-op threads were always redundant — and on top of the
crash they oversubscribed the CPU (up to 8 fan-out threads × 24 BLAS threads).
One BLAS thread removes the crash, the hang, and the oversubscription.

**Why it must be imported first.** OpenBLAS reads these env vars **once, when the
shared library loads** (i.e. at the first ``import numpy``, including the one cv2
does internally). Setting them afterwards has no effect — so this module must run
before any numpy/cv2/core import. Each entry point (``app``, ``main``,
``engine_test``, ``smoke_test``) imports it on its very first line.

``setdefault`` so an explicit override in the environment is respected — note,
though, that raising it reintroduces the crash risk.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
