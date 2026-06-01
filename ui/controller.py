"""GraphController — bridges the Qt view items to the backend graph/engine.

Each canvas owns one controller. It holds the GraphModel + Engine and a
mapping from backend node id to the Qt node item. View items delegate their
data operations here (connect, set parameter); the controller re-evaluates the
graph and refreshes exactly the recomputed view items.

This is the only place where the frontend drives the backend, so it stays thin.
"""
from __future__ import annotations

from typing import Dict

from PyQt6 import QtCore

from core.graph import GraphModel
from core.engine import Engine
from core import datatypes


class ControllerSignals(QtCore.QObject):
    """One signal hub per canvas (cheaper than a QObject per node).
    Emits the Qt node item whose result just changed."""
    nodeChanged = QtCore.pyqtSignal(object)


class GraphController:
    def __init__(self):
        self.model = GraphModel()
        self.engine = Engine(self.model)
        self.signals = ControllerSignals()
        self._qt_by_gid: Dict[int, object] = {}
        self.preview_index = 0   # which batch element every node currently shows

    # --- registration ------------------------------------------------------
    def register_source(self, qt_node, image) -> None:
        gn = self.model.add_node(op=None, source_image=image)
        self._bind(qt_node, gn)

    def register_op(self, qt_node) -> None:
        gn = self.model.add_node(op=qt_node.op, params=qt_node.op.defaults())
        self._bind(qt_node, gn)

    def _bind(self, qt_node, gnode) -> None:
        qt_node.gnode = gnode
        qt_node.controller = self
        self._qt_by_gid[gnode.id] = qt_node

    def adopt(self, model) -> None:
        """Replace the backend with a freshly loaded model (clears bindings)."""
        self.model = model
        self.engine = Engine(model)
        self._qt_by_gid.clear()

    def bind(self, qt_node, gnode) -> None:
        """Bind a view item to an existing backend node (used when loading)."""
        self._bind(qt_node, gnode)

    def recompute_all(self) -> None:
        self._recompute(commit=True)

    def set_preview_index(self, index: int) -> None:
        """Change which batch element every node previews and re-render views."""
        self.preview_index = max(0, index)
        for qt in self._qt_by_gid.values():
            qt.refresh_from_model()

    def unregister(self, qt_node) -> None:
        gn = getattr(qt_node, "gnode", None)
        if gn is None:
            return
        self.model.remove_node(gn)
        self._qt_by_gid.pop(gn.id, None)
        self._recompute(commit=True)

    def delete_edge(self, src_qt, dst_qt) -> None:
        self.model.remove_edge(src_qt.gnode, dst_qt.gnode)
        self._recompute(commit=True)

    def swap_inputs(self, qt_node) -> bool:
        """Swap the two incoming edges of a 2-input node (e.g. Diff A<->B)."""
        edges = self.model.incoming(qt_node.gnode)
        if len(edges) != 2:
            return False
        edges[0].dst_port, edges[1].dst_port = edges[1].dst_port, edges[0].dst_port
        self.model.mark_dirty(qt_node.gnode)
        self._recompute(commit=True)
        return True

    # --- topology / parameters --------------------------------------------
    _VARIADIC_CAP = 64

    def can_connect(self, src_qt, dst_qt) -> bool:
        gn = dst_qt.gnode
        if self.model.creates_cycle(src_qt.gnode, gn):
            return False
        op = dst_qt.op
        n_in = len(self.model.incoming(gn))
        if getattr(op, "variadic", False):
            if n_in >= self._VARIADIC_CAP:
                return False
            in_type = op.inputs[0].type   # single template port for all inputs
        else:
            if n_in >= gn.arity:
                return False  # all input ports already filled
            in_type = op.inputs[n_in].type
        return datatypes.compatible(self._output_type(src_qt), in_type)

    def can_rewire(self, src_qt, dst_qt) -> bool:
        """A full single-input node can be re-pointed at a new (compatible) source."""
        gn = dst_qt.gnode
        if gn.is_source or gn.arity != 1 or getattr(dst_qt.op, "variadic", False):
            return False
        if len(self.model.incoming(gn)) != 1:
            return False
        if self.model.creates_cycle(src_qt.gnode, gn):
            return False
        if self.model.incoming(gn)[0].src is src_qt.gnode:
            return False  # already wired from this source
        return datatypes.compatible(self._output_type(src_qt), dst_qt.op.inputs[0].type)

    def replace_input(self, src_qt, dst_qt) -> None:
        """Replace a single-input node's connection with one from src_qt."""
        gn = dst_qt.gnode
        for edge in list(self.model.incoming(gn)):
            self.model.remove_edge(edge.src, gn)
        self.model.add_edge(src_qt.gnode, gn, 0)
        self._recompute(commit=True)

    def _output_type(self, qt) -> str:
        gn = qt.gnode
        if gn.is_source:
            return datatypes.IMAGE
        return qt.op.outputs[0].type

    def connect(self, src_qt, dst_qt) -> bool:
        if not self.can_connect(src_qt, dst_qt):
            return False
        self.model.add_edge(src_qt.gnode, dst_qt.gnode)
        self._recompute(commit=True)
        return True

    def set_param(self, qt_node, name, value, commit: bool) -> None:
        gn = qt_node.gnode
        gn.params[name] = value
        self.model.mark_dirty(gn)
        self._recompute(commit=commit)

    # --- evaluation --------------------------------------------------------
    def _recompute(self, commit: bool) -> None:
        for gnode in self.engine.evaluate_all():
            qt = self._qt_by_gid.get(gnode.id)
            if qt is None:
                continue
            qt.refresh_from_model()
            if commit:
                qt.on_commit()
            self.signals.nodeChanged.emit(qt)
