"""Backend-only tests for core.graph + core.engine. No Qt, no QApplication.

Run: python engine_test.py
"""
import json
import numpy as np
import cv2

from core.operations import REGISTRY
from core.graph import GraphModel
from core.engine import Engine
from core.batch import Batch
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


def test_contours():
    img = np.zeros((80, 80, 3), np.uint8)
    cv2.rectangle(img, (5, 5), (24, 24), (255, 255, 255), -1)    # small (~361 px area)
    cv2.rectangle(img, (40, 40), (70, 70), (255, 255, 255), -1)  # large (~841 px area)

    m = GraphModel()
    s = _src(m, img)
    fc = _op(m, "find_contours")
    flt = _op(m, "contour_filter", min_area=500, max_area=10_000_000)
    m.add_edge(s, fc)
    m.add_edge(fc, flt)
    Engine(m).evaluate_all()

    assert isinstance(fc.output, dict) and len(fc.output["contours"]) == 2, "should find both squares"
    assert len(flt.output["contours"]) == 1, "area filter should drop the small square"

    preview = REGISTRY["find_contours"].render_preview(None, fc.output, {})
    assert isinstance(preview, np.ndarray) and preview.ndim == 3, "contours preview should be a BGR image"
    assert REGISTRY["find_contours"].summary(fc.output, {})["contours"] == 2
    print("OK  contours: find + drawContours preview + area filter")


def test_fourier_roundtrip():
    rng = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)  # deterministic gray image
    m = GraphModel()
    s = _src(m, rng)
    d = _op(m, "dft")
    i = _op(m, "idft")
    m.add_edge(s, d)
    m.add_edge(d, i)
    Engine(m).evaluate_all()

    assert isinstance(d.output, dict) and "dft" in d.output, "DFT should output a spectrum payload"
    back = i.output
    assert back is not None, "inverse DFT produced no result"
    # idft(dft(img)) == img  (DFT_SCALE makes the round-trip exact up to float error)
    max_err = float(np.abs(back - rng.astype(np.float32)).max())
    assert max_err < 1e-2, f"round-trip error too large: {max_err}"
    print(f"OK  fourier: idft(dft(img)) == img (max error {max_err:.2e})")


def test_more_ops():
    img = gradient()
    for op_id in ("gaussian_blur", "morphology", "canny", "sobel", "laplacian"):
        m = GraphModel()
        s = _src(m, img)
        o = _op(m, op_id)
        m.add_edge(s, o)
        Engine(m).evaluate_all()
        assert isinstance(o.output, np.ndarray), f"{op_id} produced no image (err={o.error})"

    m = GraphModel()
    s = _src(m, img)
    h = _op(m, "histogram")
    m.add_edge(s, h)
    Engine(m).evaluate_all()
    assert isinstance(h.output, dict) and "hist" in h.output, "histogram should output a payload"
    assert isinstance(REGISTRY["histogram"].render_preview(None, h.output, {}), np.ndarray)
    print("OK  more ops: gaussian/morphology/canny/sobel/laplacian + histogram")


def test_conversions():
    bgr = gradient()  # source inferred as BGR

    # BGR -> To HSL -> To BGR round-trips, and the engine tracks the spaces.
    m = GraphModel()
    s = _src(m, bgr)
    h = _op(m, "to_hls")
    b = _op(m, "to_bgr")
    m.add_edge(s, h)
    m.add_edge(h, b)
    Engine(m).evaluate_all()
    assert h.color_space == "hls" and b.color_space == "bgr", "spaces not tracked"
    assert int(np.abs(b.output.astype(int) - bgr.astype(int)).max()) <= 3, "hls<->bgr not reversible"

    # To Grayscale delegates correctly from HLS (reconstruct BGR, then gray) and
    # matches converting straight from BGR.
    g_from_hls = REGISTRY["to_grayscale"].compute([h.output], {}, "hls")
    g_from_bgr = REGISTRY["to_grayscale"].compute([bgr], {}, "bgr")
    assert int(np.abs(g_from_hls.astype(int) - g_from_bgr.astype(int)).max()) <= 3
    assert g_from_hls.ndim == 2, "grayscale output should be single channel"

    # A single-channel input is promoted to BGR by To BGR.
    gray = REGISTRY["to_grayscale"].compute([bgr], {}, "bgr")
    promoted = REGISTRY["to_bgr"].compute([gray], {}, "gray")
    assert promoted.ndim == 3 and promoted.shape[2] == 3
    print("OK  conversions: space-aware, arbitrary input -> target space")


def test_batched():
    imgs = [np.full((10, 10, 3), v, np.uint8) for v in (10, 100, 200)]

    # One chain, three images: blur maps over the batch (constant -> constant).
    m = GraphModel()
    s = m.add_node(op=None, source_image=Batch(imgs))
    b = _op(m, "blur", kernel_size=3)
    m.add_edge(s, b)
    Engine(m).evaluate_all()
    assert isinstance(b.output, Batch) and len(b.output) == 3
    assert [int(x.mean()) for x in b.output.items] == [10, 100, 200]

    # Broadcast: a single reference image diffs against every batch element.
    ref = m.add_node(op=None, source_image=np.zeros((10, 10, 3), np.uint8))
    d = _op(m, "diff")
    m.add_edge(s, d)
    m.add_edge(ref, d)
    Engine(m).evaluate_all()
    assert isinstance(d.output, Batch) and len(d.output) == 3
    assert [int(x.mean()) for x in d.output.items] == [10, 100, 200]

    # Mismatched batch lengths (>1) are an error, not a crash.
    s2 = m.add_node(op=None, source_image=Batch(imgs[:2]))
    d2 = _op(m, "diff")
    m.add_edge(s, d2)
    m.add_edge(s2, d2)
    Engine(m).evaluate_all()
    assert d2.error is not None

    # Batch sources survive save/load.
    doc = json.loads(json.dumps(persistence.to_dict(m, {})))
    m2, _pos = persistence.from_dict(doc)
    batch_sources = [n for n in m2.nodes.values() if n.is_source and isinstance(n.source_image, Batch)]
    assert any(len(b.source_image) == 3 for b in batch_sources), "batch source not persisted"
    print("OK  batched: op maps over a batch; single inputs broadcast; persists")


def test_create_batch():
    a = np.full((10, 10, 3), 10, np.uint8)
    b = np.full((10, 12), 50, np.uint8)        # grayscale, different size
    c = np.full((10, 10, 3), 200, np.uint8)
    m = GraphModel()
    sa, sb, sc = (m.add_node(op=None, source_image=x) for x in (a, b, c))
    cb = _op(m, "create_batch")
    for s in (sa, sb, sc):
        m.add_edge(s, cb)
    blur = _op(m, "blur", kernel_size=3)
    m.add_edge(cb, blur)
    Engine(m).evaluate_all()

    assert isinstance(cb.output, Batch) and len(cb.output) == 3, "should assemble a batch of 3"
    assert all(x.ndim == 3 and x.shape[2] == 3 for x in cb.output.items), "all elements normalized to BGR"
    assert int(cb.output.items[1].mean()) == 50, "grayscale input promoted to 3-channel"
    assert isinstance(blur.output, Batch) and len(blur.output) == 3, "downstream fans out over the batch"
    print("OK  create_batch: variadic inputs -> one homogeneous BGR batch")


def test_resize():
    m = GraphModel()
    s = _src(m, np.zeros((20, 30, 3), np.uint8))
    r = _op(m, "resize", scale=2.0)
    m.add_edge(s, r)
    Engine(m).evaluate_all()
    assert r.output is not None and r.output.shape[:2] == (40, 60), "resize should scale up"
    assert "interpolation" in REGISTRY["resize"].defaults(), "resize exposes interpolation mode"
    print("OK  resize: scale (up/down) + interpolation mode")


def test_rotate():
    img = np.zeros((20, 40, 3), np.uint8)   # 20 tall x 40 wide

    m = GraphModel()
    s = _src(m, img)
    r = _op(m, "rotate", angle=90, expand=True)
    m.add_edge(s, r)
    Engine(m).evaluate_all()
    assert r.output is not None and r.output.shape[:2] == (40, 20), \
        f"expand-rotate 90 should swap dims, got {None if r.output is None else r.output.shape[:2]}"

    m2 = GraphModel()
    s2 = _src(m2, img)
    r2 = _op(m2, "rotate", angle=90, expand=False)
    m2.add_edge(s2, r2)
    Engine(m2).evaluate_all()
    assert r2.output.shape[:2] == (20, 40), "no-expand rotate keeps the original size"
    print("OK  rotate: arbitrary angle + expand-to-fit")


def test_normalize():
    # Low-contrast gray (values ~100..150) -> stretch should span ~0..255.
    row = np.linspace(100, 150, 40, dtype=np.uint8)
    img = np.tile(row, (20, 1))

    m = GraphModel()
    s = _src(m, img)
    n = _op(m, "normalize", mode="stretch")
    m.add_edge(s, n)
    Engine(m).evaluate_all()
    assert n.output is not None
    assert int(n.output.min()) <= 2 and int(n.output.max()) >= 253, \
        f"stretch should expand the range, got {n.output.min()}..{n.output.max()}"

    # equalize / clahe run on gray and color without error.
    for mode in ("equalize", "clahe"):
        for src_img in (np.tile(row, (20, 1)), cv2.cvtColor(np.tile(row, (20, 1)), cv2.COLOR_GRAY2BGR)):
            mm = GraphModel()
            ss = _src(mm, src_img)
            e = _op(mm, "normalize", mode=mode)
            mm.add_edge(ss, e)
            Engine(mm).evaluate_all()
            assert e.output is not None, f"normalize mode={mode} produced no output"
    print("OK  normalize: stretch expands range; equalize/clahe run (gray + color)")


def test_cycle_prevention():
    m = GraphModel()
    a = _op(m, "blur")
    b = _op(m, "blur")
    m.add_edge(a, b)
    assert m.creates_cycle(b, a) is True, "b->a would close a cycle"
    assert m.creates_cycle(a, _op(m, "blur")) is False, "fresh edge is acyclic"
    print("OK  cycle detection prevents back-edges")


def main():
    test_linear_chain_and_caching()
    test_dirty_propagation()
    test_arity_gating()
    test_input_order()
    test_error_capture()
    test_persistence_roundtrip()
    test_color_pipeline()
    test_contours()
    test_fourier_roundtrip()
    test_more_ops()
    test_conversions()
    test_batched()
    test_create_batch()
    test_resize()
    test_rotate()
    test_normalize()
    test_cycle_prevention()
    print("\nENGINE OK: 17 backend tests passed")


if __name__ == "__main__":
    main()
