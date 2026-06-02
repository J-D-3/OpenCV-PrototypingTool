"""Engine — evaluates a GraphModel (backend, Qt-free).

Replaces the old recursive scene-walking + re-entrancy flags with a single
deterministic evaluator:

  * topological order (no node runs before its inputs),
  * dirty propagation (a changed node + everything downstream is recomputed),
  * per-node output caching (clean nodes with a result are not recomputed),
  * per-node error capture (surfaced via ``GraphNode.error``).

A node runs only when all of its input ports are connected and every input
produced a result; otherwise its output is ``None`` (no error).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np

from core.graph import GraphModel, GraphNode
from core.batch import Batch


def _infer_space(arr) -> str:
    """Best-effort color space from an array's channel count."""
    if isinstance(arr, Batch):
        for it in arr.items:
            if isinstance(it, np.ndarray):
                return _infer_space(it)
        return "unknown"
    if not isinstance(arr, np.ndarray):
        return "unknown"
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return "gray"
    return "bgr"


class Engine:
    def __init__(self, graph: GraphModel):
        self.graph = graph
        # Batch elements are mapped across a small thread pool (OpenCV/numpy
        # release the GIL during heavy work, so this gives real parallelism).
        self._max_workers = max(1, min(8, (os.cpu_count() or 2)))
        # Which batch element to compute first (the one the UI previews), so it
        # is ready as early as possible. Set by the controller.
        self.preview_index = 0

    @staticmethod
    def _derive_space(op, input_nodes, output) -> str:
        rule = getattr(op, "out_space", "auto")
        if rule == "passthrough":
            return input_nodes[0].color_space if input_nodes else "unknown"
        if rule in ("bgr", "gray", "hls", "binary"):
            return rule
        return _infer_space(output)

    def evaluate(self, node: GraphNode) -> None:
        """Evaluate a single node, assuming its inputs are already evaluated."""
        if node.is_source:
            node.output = node.source_image
            node.error = None
            node.color_space = _infer_space(node.source_image)
            node.dirty = False
            return

        op = node.op
        variadic = getattr(op, "variadic", False)
        raw = getattr(op, "raw", False)

        input_nodes = self.graph.inputs_of(node)
        ready = len(input_nodes) >= 1 if variadic else len(input_nodes) == node.arity
        if not ready:
            node.output = None
            node.error = None
            node.color_space = "unknown"
            node.dirty = False
            return

        inputs = [n.output for n in input_nodes]
        # Non-raw ops need every input present; raw ops (e.g. Create Batch) get
        # the inputs as-is and decide what to do with missing/batched ones.
        if not raw and any(img is None for img in inputs):
            node.output = None
            node.error = None
            node.color_space = "unknown"
            node.dirty = False
            return

        node.comp_time_ms = None
        try:
            if raw:
                t0 = time.perf_counter()
                result = self._call(op, inputs, node.params, input_nodes)
                node.comp_time_ms = (time.perf_counter() - t0) * 1000.0
                node.output = result
                node.error = None if result is not None else "operation returned no result (see console)"
            else:
                batches = [i for i in inputs if isinstance(i, Batch)]
                if batches:
                    node.output, node.error, node.comp_time_ms = self._run_batched(
                        op, inputs, batches, node.params, input_nodes)
                else:
                    t0 = time.perf_counter()
                    result = self._call(op, inputs, node.params, input_nodes)
                    node.comp_time_ms = (time.perf_counter() - t0) * 1000.0
                    node.output = result
                    node.error = None if result is not None else "operation returned no result (see console)"
        except Exception as e:  # noqa: BLE001 — surface, don't crash the UI
            node.output = None
            node.error = f"{type(e).__name__}: {e}"
            node.comp_time_ms = None
        node.color_space = self._derive_space(op, input_nodes, node.output)
        node.dirty = False

    def _call(self, op, inputs, params, input_nodes):
        if getattr(op, "space_aware", False):
            in_space = input_nodes[0].color_space if input_nodes else "unknown"
            return op.compute(inputs, params, in_space)
        return op.compute(inputs, params)

    def _run_batched(self, op, inputs, batches, params, input_nodes):
        """Map the op over a batch: zip equal-length batches; broadcast singles
        (and length-1 batches) against the batch length. Elements are computed
        in parallel across a thread pool, with the previewed element first."""
        n = max(len(b) for b in batches)
        for b in batches:
            if len(b) not in (1, n):
                raise ValueError(f"batch size mismatch: {len(b)} vs {n}")

        def compute_elem(k):
            elem = [(inp.items[k if len(inp) > 1 else 0] if isinstance(inp, Batch) else inp)
                    for inp in inputs]
            if any(e is None for e in elem):
                return None, None, None
            t0 = time.perf_counter()
            try:
                r = self._call(op, elem, params, input_nodes)
                return r, None, (time.perf_counter() - t0) * 1000.0
            except Exception as e:  # noqa: BLE001
                return None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000.0

        # Preview element first, then the rest — useful once results stream.
        order = list(range(n))
        pv = self.preview_index
        if 0 <= pv < n:
            order = [pv] + [k for k in range(n) if k != pv]

        results: List = [None] * n
        first_error = None
        times: List[float] = []
        workers = min(self._max_workers, n)
        if workers <= 1:
            for k in order:
                results[k], err, dt = compute_elem(k)
                first_error = first_error or err
                if dt is not None:
                    times.append(dt)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(compute_elem, k): k for k in order}
                for fut in as_completed(futures):
                    k = futures[fut]
                    results[k], err, dt = fut.result()
                    first_error = first_error or err
                    if dt is not None:
                        times.append(dt)
        mean_ms = (sum(times) / len(times)) if times else None
        return Batch(results), first_error, mean_ms

    def evaluate_all(self) -> List[GraphNode]:
        """Evaluate every dirty node in dependency order.

        Returns the nodes that were (re)computed this pass, in topo order, so
        callers can refresh exactly those views and fire side effects (e.g.
        save-to-file) after a node's inputs are ready.
        """
        recomputed: List[GraphNode] = []
        for node in self.graph.topo_order():
            if node.dirty:
                self.evaluate(node)
                recomputed.append(node)
        return recomputed
