"""GraphController — bridges the Qt view items to the backend graph/engine.

Each canvas owns one controller. It holds the GraphModel + Engine and a
mapping from backend node id to the Qt node item. View items delegate their
data operations here (connect, set parameter); the controller re-evaluates the
graph and refreshes exactly the recomputed view items.

This is the only place where the frontend drives the backend, so it stays thin.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict

from PyQt6 import QtCore, QtWidgets

from core.graph import GraphModel
from core.engine import Engine
from core import datatypes, diag


class ControllerSignals(QtCore.QObject):
    """One signal hub per canvas (cheaper than a QObject per node)."""
    nodeChanged = QtCore.pyqtSignal(object)        # the Qt node whose result changed
    previewIndexChanged = QtCore.pyqtSignal(int)   # the batch element being previewed
    evalDone = QtCore.pyqtSignal(object, bool)     # (recomputed gnodes, commit) — bg eval
    notify = QtCore.pyqtSignal(str, str)           # (level, message): view-layer feedback
                                                   # for the status bar; level in {error, info}


class GraphController:
    def __init__(self):
        self.model = GraphModel()
        self.engine = Engine(self.model)
        self.signals = ControllerSignals()
        self._qt_by_gid: Dict[int, object] = {}
        self.preview_index = 0   # which batch element every node currently shows
        # Background-eval state: param changes recompute off the UI thread so the
        # canvas stays responsive; rapid changes coalesce (latest wins).
        self._busy = False
        self._pending = None     # None or the commit flag of a queued re-run
        # Queued so the slot runs on the main (GUI) thread, not the worker.
        self.signals.evalDone.connect(
            self._on_eval_done, QtCore.Qt.ConnectionType.QueuedConnection)

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
        # Never swap the engine/model out from under a running background worker:
        # every other topology mutation waits for idle first; load must too, or a
        # stale worker keeps evaluating the old model while load builds the new one.
        diag.log(f"adopt: replacing model (busy={self._busy}, pending={self._pending})")
        self.wait_idle()
        self.model = model
        self.engine = Engine(model)
        self._qt_by_gid.clear()

    def bind(self, qt_node, gnode) -> None:
        """Bind a view item to an existing backend node (used when loading)."""
        self._bind(qt_node, gnode)

    def _pump(self) -> None:
        """Repaint without accepting user input (so a long synchronous load shows
        progress but can't be re-entered by clicks)."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def recompute_all(self) -> None:
        """Synchronous but *progressive* (used by pipeline load): paint the
        freshly-built graph, put a spinner on every pending node, then evaluate
        node-by-node in topological order, filling each node's preview as it
        finishes — so the user sees the load advance instead of a frozen window."""
        self.wait_idle()
        self.engine.preview_index = self.preview_index
        self._pump()                              # 1. draw the (empty) graph
        order = self.model.topo_order()
        pending = [n for n in order if n.dirty]
        diag.log(f"recompute_all: {len(pending)} pending nodes (busy={self._busy})")
        for gn in pending:                        # 2. show spinners on what's coming
            qt = self._qt_by_gid.get(gn.id)
            if qt is not None:
                qt.set_computing(True)
        self._pump()
        # Guard the whole progressive load: if a stale background worker is still
        # evaluating (e.g. from the graph we just replaced), this warns instead of
        # silently racing it on the engine.
        with diag.evaluation_guard("recompute_all"):
            for node in pending:                  # 3. evaluate + reveal one at a time
                self.engine.evaluate(node)
                qt = self._qt_by_gid.get(node.id)
                if qt is not None:
                    qt.set_computing(False)
                    qt.refresh_from_model()
                    qt.on_commit()
                    self.signals.nodeChanged.emit(qt)
                self._pump()

    def set_preview_index(self, index: int) -> None:
        """Change which batch element every node previews and re-render views."""
        index = max(0, index)
        if index == self.preview_index:
            return
        self.preview_index = index
        self.engine.preview_index = index   # compute this element first next eval
        for qt in self._qt_by_gid.values():
            qt.refresh_from_model()
        self.signals.previewIndexChanged.emit(index)

    def unregister(self, qt_node) -> None:
        gn = getattr(qt_node, "gnode", None)
        if gn is None:
            return
        self.wait_idle()             # never mutate topology under a running eval
        self.model.remove_node(gn)
        self._qt_by_gid.pop(gn.id, None)
        self._recompute_async(commit=True)

    def is_connected(self, src_qt, dst_qt) -> bool:
        gs, gd = src_qt.gnode, dst_qt.gnode
        return any(e.src is gs and e.dst is gd for e in self.model.edges)

    def delete_edge(self, src_qt, dst_qt) -> None:
        self.wait_idle()
        self.model.remove_edge(src_qt.gnode, dst_qt.gnode)
        self._recompute_async(commit=True)

    def swap_inputs(self, qt_node) -> bool:
        """Swap the two incoming edges of a 2-input node (e.g. Diff A<->B)."""
        edges = self.model.incoming(qt_node.gnode)
        if len(edges) != 2:
            return False
        self.wait_idle()
        edges[0].dst_port, edges[1].dst_port = edges[1].dst_port, edges[0].dst_port
        self.model.mark_dirty(qt_node.gnode)
        self._recompute_async(commit=True)
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
        self.wait_idle()
        for edge in list(self.model.incoming(gn)):
            self.model.remove_edge(edge.src, gn)
        self.model.add_edge(src_qt.gnode, gn, 0)
        self._recompute_async(commit=True)

    def _output_type(self, qt) -> str:
        gn = qt.gnode
        if gn.is_source:
            return datatypes.IMAGE
        return qt.op.outputs[0].type

    def connect(self, src_qt, dst_qt) -> bool:
        if not self.can_connect(src_qt, dst_qt):
            return False
        self.wait_idle()
        self.model.add_edge(src_qt.gnode, dst_qt.gnode)
        self._recompute_async(commit=True)
        return True

    def set_param(self, qt_node, name, value, commit: bool) -> None:
        gn = qt_node.gnode
        gn.params[name] = value
        self.model.mark_dirty(gn)
        self._recompute_async(commit=commit)   # off the UI thread; coalesced

    # --- evaluation --------------------------------------------------------
    def _recompute(self, commit: bool) -> None:
        """Synchronous recompute for structural edits (connect/delete/load).
        Waits for any in-flight background eval first so the two never run on the
        engine concurrently."""
        self.wait_idle()
        self._apply_results(self.engine.evaluate_all(), commit)

    def _apply_results(self, recomputed, commit: bool) -> None:
        self.engine.preview_index = self.preview_index
        for gnode in recomputed:
            qt = self._qt_by_gid.get(gnode.id)
            if qt is None:
                continue
            qt.set_computing(False)
            qt.refresh_from_model()
            if commit:
                qt.on_commit()
            self.signals.nodeChanged.emit(qt)

    def _recompute_async(self, commit: bool) -> None:
        """Recompute dirty nodes on a worker thread (param changes). Marks the
        affected nodes 'computing' (spinner) immediately; coalesces if busy."""
        self.engine.preview_index = self.preview_index
        for gnode in self.model.topo_order():
            if gnode.dirty:
                qt = self._qt_by_gid.get(gnode.id)
                if qt is not None:
                    qt.set_computing(True)
        if self._busy:
            self._pending = commit if self._pending is None else (commit or self._pending)
            diag.log(f"_recompute_async: busy, coalesced (pending={self._pending})")
            return
        self._busy = True
        diag.log("_recompute_async: starting worker")
        threading.Thread(target=self._eval_worker, args=(commit,),
                         name="eval-worker", daemon=True).start()

    def _eval_worker(self, commit: bool) -> None:
        diag.log(f"_eval_worker: begin (commit={commit})")
        t0 = time.perf_counter()
        try:
            recomputed = self.engine.evaluate_all()
            diag.log(f"_eval_worker: done in {(time.perf_counter() - t0) * 1000.0:.1f} ms, "
                     f"recomputed {len(recomputed)} node(s): {diag.nodes_summary(recomputed)}")
        except Exception as e:  # noqa: BLE001 — surface, never kill the worker silently
            # Cross-thread emit auto-queues to the GUI-thread status-bar slot.
            diag.log(f"_eval_worker: FAILED after {(time.perf_counter() - t0) * 1000.0:.1f} ms: {e}",
                     level=logging.ERROR)
            self.signals.notify.emit("error", f"Background evaluation failed: {e}")
            recomputed = []
        self.signals.evalDone.emit(recomputed, commit)   # -> _on_eval_done (GUI thread)

    def _on_eval_done(self, recomputed, commit: bool) -> None:
        self._apply_results(recomputed, commit)
        self._busy = False
        diag.log(f"_on_eval_done: applied {len(recomputed)} node(s), commit={commit}, "
                 f"pending={self._pending}")
        if self._pending is not None:
            commit2, self._pending = self._pending, None
            self._recompute_async(commit2)

    def wait_idle(self) -> None:
        """Block (pumping the event loop) until no background eval is in flight.
        Used by synchronous callers and tests so results are observable."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        guard = 0
        flag = QtCore.QEventLoop.ProcessEventsFlag.AllEvents
        while (self._busy or self._pending is not None) and guard < 1_000_000:
            guard += 1
            app.processEvents(flag, 20)   # block up to 20ms for the worker's signal
