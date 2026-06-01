"""GraphModel — the topology source of truth (backend, Qt-free).

Nodes and directed edges live here, independent of any view. A node is either
a *source* (op is None, carries ``source_image``) or an *operation* (carries a
``core.operations.Operation`` and a parameter dict). Edges connect a source
node's output to a numbered input port of a destination node.

The model holds computed results (``output``/``error``) and a ``dirty`` flag;
the actual computation is performed by :mod:`core.engine`.
"""
from __future__ import annotations

from typing import Optional, List, Dict


class GraphNode:
    def __init__(self, nid: int, op=None, params: Optional[dict] = None,
                 source_image=None, pos=(0.0, 0.0)):
        self.id = nid
        self.op = op                       # None => source node
        self.params: dict = params or {}
        self.source_image = source_image   # set for source nodes
        self.pos = pos
        # filled in by the engine:
        self.output = source_image         # sources start with their image
        self.error: Optional[str] = None
        self.dirty: bool = True

    @property
    def is_source(self) -> bool:
        return self.op is None

    @property
    def arity(self) -> int:
        return 0 if self.op is None else len(self.op.inputs)


class Edge:
    def __init__(self, src: GraphNode, dst: GraphNode, dst_port: int):
        self.src = src
        self.dst = dst
        self.dst_port = dst_port


class GraphModel:
    def __init__(self):
        self._next_id = 0
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[Edge] = []

    # --- nodes -------------------------------------------------------------
    def add_node(self, op=None, params: Optional[dict] = None,
                 source_image=None, pos=(0.0, 0.0)) -> GraphNode:
        gn = GraphNode(self._next_id, op, params, source_image, pos)
        self.nodes[self._next_id] = gn
        self._next_id += 1
        return gn

    def remove_node(self, node: GraphNode) -> None:
        self.edges = [e for e in self.edges if e.src is not node and e.dst is not node]
        self.nodes.pop(node.id, None)

    # --- edges -------------------------------------------------------------
    def incoming(self, node: GraphNode) -> List[Edge]:
        return sorted((e for e in self.edges if e.dst is node), key=lambda e: e.dst_port)

    def add_edge(self, src: GraphNode, dst: GraphNode, dst_port: Optional[int] = None) -> Edge:
        if dst_port is None:
            dst_port = len(self.incoming(dst))  # next free input slot
        edge = Edge(src, dst, dst_port)
        self.edges.append(edge)
        self.mark_dirty(dst)
        return edge

    def remove_edge(self, src: GraphNode, dst: GraphNode) -> None:
        self.edges = [e for e in self.edges if not (e.src is src and e.dst is dst)]
        self.mark_dirty(dst)

    def inputs_of(self, node: GraphNode) -> List[GraphNode]:
        """Source nodes feeding this node, ordered by destination port."""
        return [e.src for e in self.incoming(node)]

    def dependents_of(self, node: GraphNode) -> List[GraphNode]:
        """Nodes that consume this node's output."""
        return [e.dst for e in self.edges if e.src is node]

    # --- dirtying & ordering ----------------------------------------------
    def mark_dirty(self, node: GraphNode) -> None:
        """Mark ``node`` and everything downstream as needing recomputation."""
        stack = [node]
        seen = set()
        while stack:
            n = stack.pop()
            if n.id in seen:
                continue
            seen.add(n.id)
            n.dirty = True
            stack.extend(self.dependents_of(n))

    def topo_order(self) -> List[GraphNode]:
        """Kahn topological sort. Nodes in cycles are omitted."""
        indeg = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            if e.dst.id in indeg:
                indeg[e.dst.id] += 1
        queue = [self.nodes[nid] for nid, d in indeg.items() if d == 0]
        order: List[GraphNode] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for dep in self.dependents_of(n):
                indeg[dep.id] -= 1
                if indeg[dep.id] == 0:
                    queue.append(dep)
        return order
