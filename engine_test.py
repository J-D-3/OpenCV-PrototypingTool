"""Backend-only tests for core.graph + core.engine. No Qt, no QApplication.

Run: python engine_test.py
"""
import json
import numpy as np

from core.operations import REGISTRY
from core.graph import GraphModel
from core.engine import Engine
from core import persistence


def _src(model, img):
    return model.add_node(op=None, source_image=img)


def _op(model, op_id, **params):
    op = REGISTRY[op_id]
    p = op.defaults()
    p.update(params)
    return model.add_node(op=op, params=p)


def gradient(h=40, w=60):
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))[:, :, None].repeat(3, 2)


def test_linear_chain_and_caching():
    m = GraphModel(); e = Engine(m)
    s = _src(m, gradient())
    g = _op(m, "to_grayscale")
    b = _op(m, "blur", kernel_size=5)
    m.add_edge(s, g); m.add_edge(g, b)

    recomputed = e.evaluate_all()
    assert b.output is not None and b.output.ndim == 2
    assert {n.id for n in recomputed} == {s.id, g.id, b.id}

    # Nothing dirty -> nothing recomputed (cache holds).
    assert e.evaluate_all() == []
    print("OK  linear chain evaluates once; clean nodes are cached")


def test_dirty_propagation():
    m = GraphModel(); e = Engine(m)
    s = _src(m, gradient())
    t = _op(m, "threshold", threshold_value=50)
    b = _op(m, "blur", kernel_size=5)
    m.add_edge(s, t); m.add_edge(t, b)
    e.evaluate_all()
    before = b.output.copy()

    # Change upstream param -> upstream + downstream recompute, source does not.
    t.params["threshold_value"] = 200
    m.mark_dirty(t)
    recomputed = e.evaluate_all()
    assert {n.id for n in recomputed} == {t.id, b.id}, "dirty set should be threshold+blur only"
    assert not np.array_equal(before, b.output), "downstream did not change"
    print("OK  param change dirties exactly the node + its descendants")


def test_arity_gating():
    m = GraphModel(); e = Engine(m)
    a = _src(m, np.full((20, 20, 3), 10, np.uint8))
    s = _op(m, "sum")
    m.add_edge(a, s)            # only one of two inputs
    e.evaluate_all()
    assert s.output is None and s.error is None, "sum should idle with <2 inputs"

    b = _src(m, np.full((20, 20, 3), 20, np.uint8))
    m.add_edge(b, s)
    e.evaluate_all()
    assert s.output is not None, "sum should run once both inputs present"
    print("OK  multi-input node idles until all ports are connected")


def test_input_order():
    big = np.full((20, 20, 3), 200, np.uint8)
    small = np.full((20, 20, 3), 50, np.uint8)

    def run(first, second):
        m = GraphModel(); e = Engine(m)
        d = _op(m, "diff")
        m.add_edge(_src(m, first), d)
        m.add_edge(_src(m, second), d)
        e.evaluate_all()
        return d.output

    ab = run(big, small)   # 200-50 = 150
    ba = run(small, big)   # 50-200 -> 0
    assert int(ab.mean()) > 100 and int(ba.mean()) < 10
    print("OK  edge port order preserved: diff(A,B) != diff(B,A)")


def test_error_capture():
    m = GraphModel(); e = Engine(m)
    s = _src(m, gradient())
    b = _op(m, "blur", kernel_size=0)   # invalid kernel -> compute fails
    m.add_edge(s, b)
    e.evaluate_all()
    assert b.output is None and b.error, "blur with kernel 0 should record an error"
    print("OK  compute failure is captured on the node (error surfaced)")


def test_persistence_roundtrip():
    m = GraphModel()
    s = _src(m, gradient())
    t = _op(m, "threshold", threshold_value=80)
    b = _op(m, "blur", kernel_size=7)
    m.add_edge(s, t)
    m.add_edge(t, b)
    Engine(m).evaluate_all()
    expected = b.output.copy()

    d = json.loads(json.dumps(persistence.to_dict(m, {})))  # also asserts JSON-safe
    m2, _positions = persistence.from_dict(d)
    Engine(m2).evaluate_all()

    assert len(m2.nodes) == 3 and len(m2.edges) == 2, "structure not preserved"
    b2 = next(n for n in m2.nodes.values() if n.op and n.op.id == "blur")
    assert b2.output is not None and np.array_equal(expected, b2.output), "result not preserved"
    print("OK  persistence round-trips structure, params, image, and result")


def test_color_pipeline():
    m = GraphModel()
    s = _src(m, gradient())
    hls = _op(m, "to_hls")
    km = _op(m, "kmeans", k=4)
    red = _op(m, "reduce_colors")
    m.add_edge(s, hls)
    m.add_edge(hls, km)
    m.add_edge(km, red)
    Engine(m).evaluate_all()

    assert isinstance(km.output, dict) and km.output["k"] == 4, "kmeans should output a clusters payload"
    assert km.output["centers"].shape[0] == 4
    assert red.output is not None and red.output.shape == s.source_image.shape
    uniq = np.unique(red.output.reshape(-1, red.output.shape[2]), axis=0)
    assert uniq.shape[0] <= 4, f"reduced image should have <= 4 colors, got {uniq.shape[0]}"
    print("OK  color pipeline: to_hls -> kmeans -> reduce_colors (<= k colors)")


def main():
    test_linear_chain_and_caching()
    test_dirty_propagation()
    test_arity_gating()
    test_input_order()
    test_error_capture()
    test_persistence_roundtrip()
    test_color_pipeline()
    print("\nENGINE OK: 7 backend tests passed")


if __name__ == "__main__":
    main()
