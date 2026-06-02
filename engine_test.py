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

    # Filled mode draws each contour in its stable, id-based palette colour.
    from core.operations import _CONTOUR_COLORS
    half = len(_CONTOUR_COLORS) // 2
    col = lambda cid, depth=0: _CONTOUR_COLORS[(cid % half) + (depth % 2) * half]
    filled = REGISTRY["find_contours"].render_preview(None, fc.output, {"filled": True})
    assert np.any(np.all(filled == col(0), axis=2)), "id 0 filled in its palette colour"
    assert np.any(np.all(filled == col(1), axis=2)), "id 1 filled in its palette colour"
    assert col(0) != col(1), "adjacent ids get different colours"

    # Colour binds to the stable contour id: filtering one out must not recolour
    # the survivor.
    flt_filled = REGISTRY["contour_filter"].render_preview(None, flt.output, {"filled": True})
    kept_ids = flt.output["ids"]
    assert len(kept_ids) == 1
    assert np.any(np.all(flt_filled == col(kept_ids[0]), axis=2)), "kept contour keeps its colour"
    dropped_id = next(i for i in fc.output["ids"] if i not in kept_ids)
    if dropped_id % half != kept_ids[0] % half:
        assert not np.any(np.all(flt_filled == col(dropped_id), axis=2)), \
            "the filtered-out contour's colour is gone (no recolouring)"
    print("OK  contours: stable id colours + depth/size draw order; filled preview")


def test_contour_nesting_colors():
    # Filled white square with a black hole -> outer (depth 0) + hole (depth 1).
    img = np.zeros((100, 100, 3), np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.rectangle(img, (40, 40), (60, 60), (0, 0, 0), -1)
    m = GraphModel(); s = _src(m, img)
    fc = _op(m, "find_contours", mode=cv2.RETR_CCOMP)
    m.add_edge(s, fc); Engine(m).evaluate_all()
    out = fc.output
    depths = out["depths"]
    assert sorted(depths) == [0, 1], f"expected an outer + one hole, got depths {depths}"

    from core.operations import _CONTOUR_COLORS
    half = len(_CONTOUR_COLORS) // 2
    col = lambda i: _CONTOUR_COLORS[(out["ids"][i] % half) + (depths[i] % 2) * half]
    parent, child = depths.index(0), depths.index(1)
    assert col(parent) != col(child), "immediate parent and child must get distinct colours"

    prev = REGISTRY["find_contours"].render_preview(None, out, {"filled": True})
    assert tuple(int(v) for v in prev[50, 50]) == col(child), \
        "filled hole must stay visible (child colour on top, not the parent's)"
    print("OK  contour nesting: immediate parent/child get distinct colours")


def test_label_regions():
    # Three solid color blocks -> exactly three regions (exact-equality, delta=0).
    img = np.zeros((60, 90, 3), np.uint8)
    img[:, :30] = (200, 0, 0); img[:, 30:60] = (0, 200, 0); img[:, 60:] = (0, 0, 200)
    m = GraphModel(); s = _src(m, img)
    lr = _op(m, "label_regions", channel=-1, delta=0, connectivity=4)
    m.add_edge(s, lr)
    Engine(m).evaluate_all()
    assert len(lr.output["contours"]) == 3, f"expected 3 regions, got {len(lr.output['contours'])}"
    assert lr.output["shape"] == img.shape and lr.output["background"].ndim == 3

    def regions(image, **params):
        mm = GraphModel(); ss = _src(mm, image)
        nn = _op(mm, "label_regions", **params); mm.add_edge(ss, nn)
        Engine(mm).evaluate_all()
        return len(nn.output["contours"])

    # Connectivity: a color checkerboard splits on 4-conn, merges on 8-conn.
    chk = np.zeros((40, 40, 3), np.uint8)
    yy, xx = np.mgrid[0:40, 0:40]
    chk[(yy // 10 + xx // 10) % 2 == 0] = (255, 255, 255)
    n4, n8 = regions(chk, delta=0, connectivity=4), regions(chk, delta=0, connectivity=8)
    assert n4 > n8, f"4-conn should split diagonal touches more than 8-conn ({n4} vs {n8})"

    # Channel: same hue at two brightnesses is one region on H, two on full color.
    red = np.zeros((30, 60, 3), np.uint8); red[:, :30] = (40, 40, 220); red[:, 30:] = (20, 20, 110)
    assert regions(red, channel=0, delta=10) < regions(red, channel=-1, delta=10), \
        "Hue channel should merge same-hue/different-brightness areas"

    # Delta tolerance: near-equal grays merge as delta grows.
    g = np.zeros((30, 60, 3), np.uint8); g[:, :30] = 100; g[:, 30:] = 112
    assert regions(g, channel=-1, delta=4) == 2 and regions(g, channel=-1, delta=20) == 1

    # Emits a CONTOURS payload that the Filter Contours node consumes.
    flt = _op(m, "contour_filter", min_area=100, max_area=1_000_000)
    m.add_edge(lr, flt)
    Engine(m).evaluate_all()
    assert len(flt.output["contours"]) == 3, "regions should flow into Filter Contours"
    print("OK  label_regions: flood-fill regions (channel/delta/connectivity) -> contours")


def test_floodfill_nesting_colors():
    # Blue background (touches the image border) with a red square inside it.
    img = np.zeros((100, 100, 3), np.uint8)
    img[:] = (200, 0, 0)
    img[40:60, 40:60] = (0, 0, 200)
    m = GraphModel(); s = _src(m, img)
    ff = _op(m, "label_regions", channel=-1, delta=0, connectivity=4)
    m.add_edge(s, ff); Engine(m).evaluate_all()
    out = ff.output
    # No contour hierarchy in a label map -> recovered: outer depth 0, hole depth 1.
    assert set(out["depths"]) == {0, 1}, f"expected recovered nesting, got {out['depths']}"
    from core.operations import _CONTOUR_COLORS
    half = len(_CONTOUR_COLORS) // 2
    cols = {(out["ids"][i] % half) + (out["depths"][i] % 2) * half for i in range(len(out["depths"]))}
    assert len(cols) == 2, "parent and hole must land in different palette halves"
    print("OK  flood-fill nesting: hole recovered as depth 1 (distinct colour)")


def test_connected_components():
    # Two separated white squares on black -> two components.
    img = np.zeros((40, 80), np.uint8)
    img[8:24, 8:24] = 255
    img[8:24, 48:64] = 255
    m = GraphModel(); s = _src(m, img)
    cc = _op(m, "connected_components", connectivity=8)
    m.add_edge(s, cc)
    Engine(m).evaluate_all()
    assert len(cc.output["contours"]) == 2, f"expected 2 blobs, got {len(cc.output['contours'])}"

    # delta==0 Label Regions should agree with connectedComponents on a binary
    # image (both count the same connected blobs).
    lr = _op(m, "label_regions", channel=1, delta=0, connectivity=8)
    m.add_edge(s, lr)
    Engine(m).evaluate_all()
    bg = [c for c in lr.output["contours"] if cv2.contourArea(c) > 4]
    assert len(bg) >= 2, "delta=0 label_regions should also find the two blobs"
    print("OK  connected_components: binary blobs; delta=0 label_regions agrees")


def test_segmentation_nodes():
    # White background, one rotated dark rectangle = the "object".
    img = np.full((120, 160, 3), 255, np.uint8)
    box = cv2.boxPoints(((80, 60), (90, 40), 20))
    cv2.fillPoly(img, [box.astype(np.int32)], (30, 30, 30))
    m = GraphModel(); s = _src(m, img)
    mask = _op(m, "color_mask", blue=255, green=255, red=255, delta=30, select="outside")
    fc = _op(m, "find_contours", mode=cv2.RETR_EXTERNAL)
    lc = _op(m, "largest_contour", count=1)
    crop = _op(m, "crop_to_contour", border=4)
    m.add_edge(s, mask); m.add_edge(mask, fc); m.add_edge(fc, lc)
    m.add_edge(s, crop, 0)            # image  -> crop port 0
    m.add_edge(lc, crop, 1)          # contour -> crop port 1
    Engine(m).evaluate_all()

    assert set(np.unique(mask.output).tolist()) <= {0, 255}, "color mask must be binary"
    assert mask.output[0, 0] == 0, "white background -> 0 with select=outside"
    assert len(lc.output["contours"]) == 1, "largest keeps exactly one contour"
    c = crop.output
    assert isinstance(c, np.ndarray) and c.ndim == 3 and c.size > 0, "crop produced an image"
    assert c.shape[0] < img.shape[0] and c.shape[1] < img.shape[1], "crop is just the object + border"
    print("OK  segmentation: color mask -> contours -> largest -> deskew & crop")


def test_codegen():
    from core import codegen
    m = GraphModel()
    s = _src(m, np.zeros((20, 20, 3), np.uint8))
    km = _op(m, "kmeans", k=4, cluster_space="lab", lum_weight=0.3)
    rc = _op(m, "reduce_colors")
    ex = _op(m, "export_code")
    m.add_edge(s, km); m.add_edge(km, rc); m.add_edge(rc, ex)
    code = codegen.generate_pseudocode(m, ex)
    assert 'imread(' in code, "source should render as imread"
    assert 'cv::kmeans' in code and 'KMEANS_PP_CENTERS' in code, "kmeans block missing"
    assert "space='lab'" in code, "params should be reflected"
    assert 'reduce_colors' not in code or 'reshape(shape)' in code, "reduce_colors custom block"
    assert 'pipeline result ->' in code, "should name the result variable"
    # Single-op tooltip pseudocode + description fallback to docstring.
    assert 'cv::blur' in codegen.op_pseudocode(REGISTRY['blur'], {'kernel_size': 7})
    assert codegen.op_description(REGISTRY['mean_shift']).lower().startswith('mean-shift')
    print("OK  codegen: upstream walk -> pseudocode (cv:: + custom blocks + params)")


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


def test_invert():
    img = np.full((5, 5, 3), (10, 20, 30), np.uint8)
    m = GraphModel()
    s = _src(m, img)
    n = _op(m, "invert")
    m.add_edge(s, n)
    Engine(m).evaluate_all()
    assert n.output is not None and tuple(int(v) for v in n.output[0, 0]) == (245, 235, 225), \
        "invert should be 255 - value"
    print("OK  invert: 255 - value")


def test_local_hdr():
    rng = np.random.RandomState(0)
    low = (100 + rng.rand(64, 64) * 8).astype(np.uint8)   # low local contrast

    m = GraphModel()
    s = _src(m, low)
    n = _op(m, "local_hdr", radius=12, amplitude=40, strength=1.0)
    m.add_edge(s, n)
    Engine(m).evaluate_all()
    out = n.output
    assert out is not None and out.shape == low.shape and out.dtype == np.uint8
    assert float(out.std()) > float(low.std()) * 2, "local HDR should boost local contrast"

    # color images run (processed on luminance) without error.
    mm = GraphModel()
    ss = _src(mm, cv2.cvtColor(low, cv2.COLOR_GRAY2BGR))
    nn = _op(mm, "local_hdr")
    mm.add_edge(ss, nn)
    Engine(mm).evaluate_all()
    assert nn.output is not None and nn.output.shape[2] == 3
    print("OK  local_hdr: smooth local contrast normalization (gray + color)")


def test_auto_cluster():
    # Three flat intensity bands -> ~3 histogram peaks -> auto k ~= 3.
    img = np.zeros((30, 30, 3), np.uint8)
    img[:10] = 20
    img[10:20] = 128
    img[20:] = 240
    m = GraphModel()
    s = _src(m, img)
    a = _op(m, "auto_cluster", max_k=12, smoothing=2.0, min_prominence=0.05)
    m.add_edge(s, a)
    Engine(m).evaluate_all()
    assert isinstance(a.output, dict), "auto_cluster should output a clusters payload"
    assert 2 <= a.output["k"] <= 4, f"expected ~3 auto-detected clusters, got {a.output['k']}"

    # Channel choice: three bands of distinct hue but matched lightness. The
    # Luminance (L) channel is ~flat (few peaks) while Hue (H) sees three modes.
    hls = np.zeros((30, 30, 3), np.uint8)
    hls[..., 1] = 128                    # L flat across all bands
    hls[..., 2] = 200                    # S high so hue is meaningful
    hls[:10, :, 0], hls[10:20, :, 0], hls[20:, :, 0] = 10, 70, 130  # H bands
    bands = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    m2 = GraphModel()
    s2 = _src(m2, bands)
    by_l = _op(m2, "auto_cluster", max_k=12, smoothing=2.0, min_prominence=0.05, channel=1)
    by_h = _op(m2, "auto_cluster", max_k=12, smoothing=2.0, min_prominence=0.05, channel=0)
    m2.add_edge(s2, by_l)
    m2.add_edge(s2, by_h)
    Engine(m2).evaluate_all()
    assert by_h.output["k"] > by_l.output["k"], (
        f"Hue channel should see more modes than flat Luminance "
        f"(H={by_h.output['k']}, L={by_l.output['k']})")
    print("OK  auto_cluster: detects cluster count from histogram peaks (per channel)")


def _color_scene(light=False):
    """4 distinct colored blobs on gray; optionally re-lit (gain + L->R ramp)."""
    img = np.full((80, 80, 3), 110, np.uint8)
    img[8:32, 8:32] = (40, 40, 200)
    img[8:32, 48:72] = (40, 200, 40)
    img[48:72, 8:32] = (200, 120, 40)
    img[48:72, 48:72] = (40, 200, 200)
    if not light:
        return img
    ramp = np.linspace(0.6, 1.5, img.shape[1])[None, :, None]
    return np.clip(img.astype(np.float32) * 1.3 * ramp, 0, 255).astype(np.uint8)


def test_cluster_space():
    base = _color_scene()
    m = GraphModel()
    s = _src(m, base)
    a = _op(m, "kmeans", k=5, cluster_space="lab", lum_weight=0.3)
    b = _op(m, "kmeans", k=5, cluster_space="lab", lum_weight=0.3)
    m.add_edge(s, a); m.add_edge(s, b)
    Engine(m).evaluate_all()
    # #2: same image -> identical labels (pinned RNG) and canonical dark->light order.
    assert np.array_equal(a.output["labels"], b.output["labels"]), "labels must be reproducible"
    lum = cv2.cvtColor(np.clip(a.output["centers"], 0, 255).astype(np.uint8)
                       .reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).ravel()
    assert np.all(np.diff(lum) >= -1.0), f"centers should be ordered dark->light, got {lum}"

    # #1: under realistic (non-uniform, multiplicative) lighting, chroma-weighted
    # Lab clustering keeps blob labels far more stable than raw BGR.
    from core.operations import _kmeans_clusters
    def agree(space, w):
        l1 = _kmeans_clusters(_color_scene(False), 5, space=space, lum_weight=w)["labels"]
        l2 = _kmeans_clusters(_color_scene(True), 5, space=space, lum_weight=w)["labels"]
        return (l1 == l2).mean()
    bgr_agree, lab_agree = agree("bgr", 1.0), agree("lab", 0.2)
    assert lab_agree > bgr_agree + 0.3, (
        f"Lab+low-lum should beat BGR under lighting change "
        f"(lab={lab_agree:.2f}, bgr={bgr_agree:.2f})")
    print("OK  cluster space: Lab+lum-weight is lighting-stable; labels reproducible")


def test_mean_shift():
    rng = np.random.RandomState(1)
    img = (rng.rand(40, 40, 3) * 255).astype(np.uint8)
    m = GraphModel()
    s = _src(m, img)
    ms = _op(m, "mean_shift", spatial=10, color=40)
    m.add_edge(s, ms)
    Engine(m).evaluate_all()
    out = ms.output
    assert out is not None and out.shape == img.shape and out.dtype == np.uint8
    before = len(np.unique(img.reshape(-1, 3), axis=0))
    after = len(np.unique(out.reshape(-1, 3), axis=0))
    assert after < before, "mean shift should merge colors (fewer uniques)"
    print("OK  mean_shift: mode-seeking segmentation reduces unique colors")


def test_comp_timing_and_traversal():
    img = np.zeros((40, 40, 3), np.uint8)
    m = GraphModel(); s = _src(m, img)
    b = _op(m, "gaussian_blur")
    m.add_edge(s, b)
    Engine(m).evaluate_all()
    assert s.comp_time_ms is None, "source nodes are not timed"
    assert isinstance(b.comp_time_ms, float) and b.comp_time_ms >= 0, "op records compute time"

    # Batch: a mean per-element time is recorded.
    mb = GraphModel(); sb = _src(mb, Batch([img.copy() for _ in range(4)]))
    bb = _op(mb, "gaussian_blur"); mb.add_edge(sb, bb)
    Engine(mb).evaluate_all()
    assert isinstance(bb.comp_time_ms, float), "batch op records a mean compute time"

    # ancestors / descendants drive the flow highlight.
    g = GraphModel()
    a = _op(g, "blur"); b2 = _op(g, "blur"); c = _op(g, "blur")
    g.add_edge(a, b2); g.add_edge(b2, c)
    assert g.ancestors(c) == {a.id, b2.id}, "ancestors are transitive predecessors"
    assert g.descendants(a) == {b2.id, c.id}, "descendants are transitive successors"
    print("OK  per-node compute timing (single + batch mean) + ancestors/descendants")


def test_codegen_covers_cv_calls():
    """Every op's codegen emitter must name every cv2 function its compute (and
    helpers) actually calls — so the sidebar search finds a node by any OpenCV
    function it uses, on any code path."""
    import ast
    import inspect
    from core import operations as ops_mod, codegen

    tree = ast.parse(inspect.getsource(ops_mod))
    direct, calls = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            cv, loc = set(), set()
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    f = n.func
                    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "cv2":
                        cv.add(f.attr)             # cv2.X(...) — a real call, not a constant
                    elif isinstance(f, ast.Name):
                        loc.add(f.id)              # local helper call -> recurse
            direct[node.name] = cv
            calls[node.name] = loc

    def closure(fname, seen):
        if fname in seen or fname not in direct:
            return set()
        seen.add(fname)
        out = set(direct[fname])
        for c in calls[fname]:
            out |= closure(c, seen)
        return out

    missing = {}
    for op in ops_mod.OPS:
        needed = closure(op.compute.__name__, set())
        have = {c.split("::", 1)[1] for c in codegen.op_cv_calls(op)}
        if needed - have:
            missing[op.id] = sorted(needed - have)
    assert not missing, f"codegen emitters omit cv2 calls the op makes: {missing}"
    print("OK  codegen emitters name every cv2 function each op calls (searchable)")


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
    test_contour_nesting_colors()
    test_label_regions()
    test_floodfill_nesting_colors()
    test_connected_components()
    test_segmentation_nodes()
    test_codegen()
    test_fourier_roundtrip()
    test_more_ops()
    test_conversions()
    test_batched()
    test_create_batch()
    test_resize()
    test_rotate()
    test_normalize()
    test_invert()
    test_local_hdr()
    test_auto_cluster()
    test_cluster_space()
    test_mean_shift()
    test_comp_timing_and_traversal()
    test_codegen_covers_cv_calls()
    test_cycle_prevention()
    print("\nENGINE OK: 30 backend tests passed")


if __name__ == "__main__":
    main()
