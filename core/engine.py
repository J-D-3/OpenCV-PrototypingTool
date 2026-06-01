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

from typing import List

import numpy as np

from core.graph import GraphModel, GraphNode


def _infer_space(arr) -> str:
    """Best-effort color space from an array's channel count."""
    if not isinstance(arr, np.ndarray):
        return "unknown"
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return "gray"
    return "bgr"


class Engine:
    def __init__(self, graph: GraphModel):
        self.graph = graph

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

        input_nodes = self.graph.inputs_of(node)
        if len(input_nodes) != node.arity:
            node.output = None
            node.error = None
            node.color_space = "unknown"
            node.dirty = False
            return

        inputs = [n.output for n in input_nodes]
        if any(img is None for img in inputs):
            node.output = None
            node.error = None
            node.color_space = "unknown"
            node.dirty = False
            return

        op = node.op
        try:
            if getattr(op, "space_aware", False):
                in_space = input_nodes[0].color_space if input_nodes else "unknown"
                result = op.compute(inputs, node.params, in_space)
            else:
                result = op.compute(inputs, node.params)
            if result is None:
                node.output = None
                node.error = "operation returned no result (see console)"
            else:
                node.output = result
                node.error = None
        except Exception as e:  # noqa: BLE001 — surface, don't crash the UI
            node.output = None
            node.error = f"{type(e).__name__}: {e}"
        node.color_space = self._derive_space(op, input_nodes, node.output)
        node.dirty = False

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
