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

    def unregister(self, qt_node) -> None:
        gn = getattr(qt_node, "gnode", None)
        if gn is None:
            return
        self.model.remove_node(gn)
        self._qt_by_gid.pop(gn.id, None)
        self._recompute(commit=True)

    # --- topology / parameters --------------------------------------------
    def can_connect(self, src_qt, dst_qt) -> bool:
        gn = dst_qt.gnode
        port_index = len(self.model.incoming(gn))
        if port_index >= gn.arity:
            return False  # all input ports already filled
        in_type = dst_qt.op.inputs[port_index].type
        return datatypes.compatible(self._output_type(src_qt), in_type)

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
