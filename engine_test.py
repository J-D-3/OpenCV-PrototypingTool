"""Backend-only tests for core.graph + core.engine. No Qt, no QApplication.

Run: python engine_test.py
"""
import core._threadlimit  # noqa: F401 — first import: pin OpenBLAS before numpy loads
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


def test_blas_thread_pinned():
    """Regression: the batch fan-out runs numpy/BLAS across many threads, and
    concurrent OpenBLAS calls deadlock/segfault the process. core._threadlimit
    (imported first, before numpy) pins BLAS to one thread to prevent it. Guard
    the wiring so it can't silently regress (e.g. a reordered import)."""
    import os
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", \
        "BLAS not pinned — core._threadlimit must be imported before numpy"
    print("OK  BLAS pinned to 1 thread (concurrent-fan-out crash/hang guard)")


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


def test_hdbscan_cluster():
    """Density colour clustering via the optional `optics` package (optics.cluster_image,
    which dedups, voxel-quantizes and converts sRGB->CIELAB internally) -> a CLUSTERS payload
    Reduce Colors consumes like K-Means. Skips cleanly if the package isn't installed. Three
    well-separated colour modes + sparse 'bridge' pixels (which become noise)."""
    from core import optics_backend
    if not optics_backend.available():
        print("OK  density cluster: SKIPPED (optics package unavailable)")
        return
    rng = np.random.default_rng(0)
    blocks = [np.clip(rng.normal(mean, 6, (1200, 3)), 0, 255)
              for mean in [(40, 40, 200), (60, 180, 60), (200, 120, 40)]]
    pix = np.vstack(blocks).astype(np.uint8)
    # Sparse, uniformly-random "bridge" pixels — density clustering should call these noise.
    pix = np.vstack([pix, rng.integers(0, 255, (120, 3)).astype(np.uint8)])
    rng.shuffle(pix)
    img = pix[:3600].reshape(60, 60, 3)

    def run(**params):
        m = GraphModel()
        s = _src(m, img)
        hd = _op(m, "hdbscan_cluster", **params)
        red = _op(m, "reduce_colors")
        m.add_edge(s, hd); m.add_edge(hd, red)
        Engine(m).evaluate_all()
        return hd, red

    # HDBSCAN in Lab, FLAG noise mode: 3 modes -> clusters; sparse pixels -> magenta noise.
    hd, red = run(algorithm="hdbscan", min_cluster_size=80, color_space="lab", voxel_bin=2,
                  noise_handling="flag", min_cluster_frac=0.01)
    assert isinstance(hd.output, dict), f"hdbscan should output a clusters payload (error={hd.error})"
    k = hd.output["k"]
    assert k >= 2, f"the 3 colour modes should form clusters, got k={k}"
    assert red.output is not None and red.output.shape == img.shape
    if hd.output["n_noise"] > 0:    # sparse bridge pixels exist -> a trailing magenta centre
        assert hd.output["noise_index"] == k
        assert hd.output["centers"].shape == (k + 1, 3)
        assert list(hd.output["centers"][k]) == [200.0, 0.0, 200.0], "noise centre is the magenta flag"

    # NEAREST noise mode (default): noise folded into clusters -> no noise centre, no magenta.
    hdn, redn = run(algorithm="hdbscan", min_cluster_size=80, color_space="lab", voxel_bin=2,
                    noise_handling="nearest", min_cluster_frac=0.01)
    kn = hdn.output["k"]
    assert hdn.output["noise_index"] == -1, "nearest mode has no noise centre"
    assert hdn.output["centers"].shape == (kn, 3), "nearest mode: only real centres"
    assert hdn.output["labels"].max() < kn, "every pixel assigned to a real cluster"
    uniqn = np.unique(redn.output.reshape(-1, 3), axis=0)
    assert not (uniqn == [200, 0, 200]).all(axis=1).any(), "nearest mode has no magenta flag"

    # All algorithm modes (exact + approximate) produce a valid payload + round-trip.
    for algo in ("optics-xi", "optics-threshold", "shdbscan", "soptics"):
        hd2, red2 = run(algorithm=algo, min_cluster_size=80, color_space="rgb",
                        metric="l2", seed=42, voxel_bin=2, min_cluster_frac=0.01)
        assert isinstance(hd2.output, dict), f"{algo} payload (error={hd2.error})"
        assert red2.output is not None and red2.output.shape == img.shape, f"{algo}: reduce_colors round-trips"

    # Thread-safety: a Batch fans out across the engine's thread pool, and the native
    # optics extension is NOT thread-safe — without the serialization lock this crashes
    # the process. Getting payloads back for every element proves the lock holds.
    bm = GraphModel()
    bs = _src(bm, Batch([img, img[::-1].copy(), img[:, ::-1].copy()]))
    bhd = _op(bm, "hdbscan_cluster", algorithm="hdbscan", color_space="lab",
              voxel_bin=2, min_cluster_size=80, min_cluster_frac=0.01)
    bm.add_edge(bs, bhd)
    Engine(bm).evaluate_all()
    assert isinstance(bhd.output, Batch) and len(bhd.output.items) == 3, "batch -> 3 payloads"
    assert all(isinstance(o, dict) for o in bhd.output.items), "batch fan-out survived (thread-safe)"

    # Reachability-plot diagnostic: stashed in diag, and render_preview stacks it.
    hd3, _ = run(algorithm="hdbscan", min_cluster_size=80, color_space="lab",
                 voxel_bin=2, show_reachability=True)
    diag = hd3.output["diag"]
    assert "reach" in diag and "reach_labels" in diag, "reachability diag should be precomputed"
    assert len(diag["reach"]) == len(diag["reach_labels"]) > 0
    prev = REGISTRY["hdbscan_cluster"].render_preview([img], hd3.output, {})
    assert prev is not None and prev.shape[0] > img.shape[0], "preview should stack image + reachability"

    print("OK  density cluster: cluster_image (HDBSCAN/OPTICS/sHDBSCAN/sOPTICS); noise nearest+flag; reachability")


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


def test_crop_no_clip_when_deskewed():
    # A long thin rect at 45 deg: deskewing rotates it ~45 deg. The old code warped
    # into the *source* canvas (100x100), clipping the 110-long object to ~100.
    img = np.zeros((100, 100, 3), np.uint8)
    box = cv2.boxPoints(((50, 50), (110, 20), 45)).astype(np.int32)
    cv2.fillPoly(img, [box], (255, 255, 255))
    m = GraphModel(); s = _src(m, img)
    fc = _op(m, "find_contours", mode=cv2.RETR_EXTERNAL)
    crop = _op(m, "crop_to_contour", border=4, scale=1.0)
    m.add_edge(s, fc); m.add_edge(s, crop, 0); m.add_edge(fc, crop, 1)
    Engine(m).evaluate_all()
    c = crop.output
    assert isinstance(c, np.ndarray)
    assert max(c.shape[:2]) >= 110, \
        f"deskewed crop must keep the full object (no clip), got {c.shape[:2]}"
    print("OK  crop: deskew canvas sized to the rect (no clipping when rotated)")


def test_crop_negative_border():
    # A negative border trims inward — the crop is tighter than the contour's box.
    img = np.full((120, 160, 3), 255, np.uint8)
    cv2.fillPoly(img, [cv2.boxPoints(((80, 60), (90, 40), 20)).astype(np.int32)], (30, 30, 30))

    def crop_at(border):
        m = GraphModel(); s = _src(m, img)
        mask = _op(m, "color_mask", blue=255, green=255, red=255, delta=30, select="outside")
        fc = _op(m, "find_contours"); lc = _op(m, "largest_contour")
        crop = _op(m, "crop_to_contour", border=border)
        m.add_edge(s, mask); m.add_edge(mask, fc); m.add_edge(fc, lc)
        m.add_edge(s, crop, 0); m.add_edge(lc, crop, 1)
        Engine(m).evaluate_all()
        return crop.output

    pos, neg = crop_at(10), crop_at(-10)
    assert isinstance(pos, np.ndarray) and isinstance(neg, np.ndarray)
    # border adds 2*border to each dimension, so +10 vs -10 differ by exactly 40 px.
    assert pos.shape[0] - neg.shape[0] == 40 and pos.shape[1] - neg.shape[1] == 40, \
        f"negative border should trim inward (+10 vs -10 differ by 40px): {pos.shape[:2]} vs {neg.shape[:2]}"
    print("OK  crop: negative border trims inward (tighter than the contour box)")


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


def test_codegen_clustering_detail():
    from core import codegen
    # K-Means spells out the feature space, params, and the k-means/centre/order steps.
    km = codegen.op_pseudocode(REGISTRY["kmeans"],
                               {"k": 6, "cluster_space": "lab", "lum_weight": 0.3})
    for tok in ("k=6", "space='lab'", "lum_weight=0.3", "BGR2Lab", "cv::kmeans",
                "mean INPUT-space", "dark->light"):
        assert tok in km, f"K-Means pseudocode missing {tok!r}:\n{km}"
    # Detect Color Centers shows the LCh detection (CIELAB, hue + neutral L*).
    dc = codegen.op_pseudocode(REGISTRY["detect_centers"],
                               {"max_k": 5, "smoothing": 3.0, "min_prominence": 0.15,
                                "chroma_threshold": 8.0, "sat_weight": 1.0})
    for tok in ("CIELAB", "LCh", "hue histogram", "lightness L*", "sigma=3.0", "CENTERS"):
        assert tok in dc, f"Detect Color Centers pseudocode missing {tok!r}:\n{dc}"
    # Assign to Centers spells out the nearest/refine assignment + CLUSTERS output.
    asg = codegen.op_pseudocode(REGISTRY["assign_centers"],
                                {"algorithm": "kmeans", "lum_weight": 0.3})
    for tok in ("nearest centre", "lum_weight=0.3", "KMEANS_USE_INITIAL_LABELS", "CLUSTERS"):
        assert tok in asg, f"Assign to Centers pseudocode missing {tok!r}:\n{asg}"
    print("OK  codegen: clustering pseudocode spells out every step + parameter")


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


def test_auto_threshold():
    # Bimodal gray: dark background + a bright square. Every method should find the
    # split automatically and output a binary mask separating the two.
    img = np.full((60, 60), 40, np.uint8)
    img[20:40, 20:40] = 200
    for method in ("otsu", "triangle", "valley"):
        m = GraphModel(); s = _src(m, img)
        n = _op(m, "auto_threshold", method=method)
        m.add_edge(s, n); Engine(m).evaluate_all()
        out = n.output
        assert out is not None and set(np.unique(out)).issubset({0, 255}), f"{method} must be binary"
        assert out[30, 30] == 255 and out[5, 5] == 0, f"{method} should separate square from background"
    m = GraphModel(); s = _src(m, img)
    n = _op(m, "auto_threshold", method="otsu", invert=True)
    m.add_edge(s, n); Engine(m).evaluate_all()
    assert n.output[30, 30] == 0 and n.output[5, 5] == 255, "invert swaps foreground/background"
    print("OK  auto_threshold: Otsu/Triangle/Valley separate a bimodal image; invert swaps")


def test_backproject():
    # Model = a red patch; target = a red square on a blue background. The hue
    # histogram of the model, backprojected onto the target, must light up the red
    # square and leave the blue background dark.
    model_img = np.full((20, 20, 3), (40, 40, 200), np.uint8)   # red (BGR)
    target = np.full((60, 60, 3), (200, 40, 40), np.uint8)       # blue background
    target[20:40, 20:40] = (40, 40, 200)                         # red square
    m = GraphModel()
    s_model, s_target = _src(m, model_img), _src(m, target)
    hist = _op(m, "histogram", color_space="hls")
    bp = _op(m, "backproject", chroma_only=True)
    m.add_edge(s_model, hist)
    m.add_edge(s_target, bp, 0)         # target image -> port 0
    m.add_edge(hist, bp, 1)             # histogram model -> port 1
    Engine(m).evaluate_all()
    out = bp.output
    assert out is not None and out.ndim == 2, "backproject -> a single-channel likelihood map"
    assert int(out[30, 30]) > 150, "the matching red square should light up"
    assert int(out[30, 30]) > int(out[5, 5]) + 100, "non-matching blue background stays dark"
    print("OK  backproject: histogram model lights up the matching colour in the target")


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


def test_resize_fixed_length():
    # 'fixed' mode scales so the LONGER edge hits the target length (aspect kept).
    m = GraphModel(); s = _src(m, np.zeros((40, 80, 3), np.uint8))   # longer edge = width 80
    r = _op(m, "resize", mode="fixed", length=160)
    m.add_edge(s, r); Engine(m).evaluate_all()
    assert r.output.shape[:2] == (80, 160), f"longer edge -> 160 (got {r.output.shape[:2]})"

    m2 = GraphModel(); s2 = _src(m2, np.zeros((120, 30, 3), np.uint8))  # longer edge = height 120
    r2 = _op(m2, "resize", mode="fixed", length=60)
    m2.add_edge(s2, r2); Engine(m2).evaluate_all()
    assert r2.output.shape[:2] == (60, 15), f"tall image longer edge -> 60 (got {r2.output.shape[:2]})"

    # Contours too: scaled by length / (longer edge of the reference shape).
    img = np.zeros((50, 100), np.uint8); cv2.rectangle(img, (10, 10), (40, 30), 255, -1)
    m3 = GraphModel(); s3 = _src(m3, img)
    fc = _op(m3, "find_contours"); rz = _op(m3, "resize", mode="fixed", length=200)  # 100 -> 200, x2
    m3.add_edge(s3, fc); m3.add_edge(fc, rz); Engine(m3).evaluate_all()
    a0 = cv2.contourArea(fc.output["contours"][0])
    a1 = cv2.contourArea(rz.output["contours"][0])
    assert abs(a1 - 4 * a0) < 0.2 * a0, f"longer edge 100->200 (2x) -> ~4x contour area ({a1} vs {a0})"
    print("OK  resize: 'fixed length' scales the longer edge to the target (images + contours)")


def test_resize_contours():
    # Resize also scales a CONTOURS payload: segment on a downscaled image, then
    # scale the contours back up to the original resolution.
    small = np.zeros((60, 80), np.uint8)
    cv2.rectangle(small, (10, 10), (30, 40), 255, -1)
    m = GraphModel(); s = _src(m, small)
    fc = _op(m, "find_contours"); up = _op(m, "resize", scale=2.0)
    m.add_edge(s, fc); m.add_edge(fc, up); Engine(m).evaluate_all()
    assert isinstance(up.output, dict) and "contours" in up.output, "resize passes a contours payload"
    a_small = cv2.contourArea(fc.output["contours"][0])
    a_up = cv2.contourArea(up.output["contours"][0])
    assert abs(a_up - 4 * a_small) < 0.2 * a_small, f"2x scale -> ~4x area ({a_up} vs {a_small})"
    x, y, _, _ = cv2.boundingRect(up.output["contours"][0])
    assert 16 <= x <= 24 and 16 <= y <= 24, f"top-left ~ (20,20) after 2x, got {(x, y)}"
    assert up.output["background"].shape[:2] == (120, 160), "preview background scaled with the contours"

    # End-to-end: segment small, scale contours up, crop the ORIGINAL region.
    orig = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(orig, (40, 30), (110, 90), (200, 180, 160), -1)   # 70 x 60 object
    small_c = cv2.resize(orig, None, fx=0.5, fy=0.5)
    m2 = GraphModel()
    a, b = _src(m2, small_c), _src(m2, orig)
    mask = _op(m2, "color_mask", blue=0, green=0, red=0, delta=30, select="outside")
    fc2 = _op(m2, "find_contours"); lc = _op(m2, "largest_contour")
    up2 = _op(m2, "resize", scale=2.0); crop = _op(m2, "crop_to_contour", border=0)
    m2.add_edge(a, mask); m2.add_edge(mask, fc2); m2.add_edge(fc2, lc); m2.add_edge(lc, up2)
    m2.add_edge(b, crop, 0); m2.add_edge(up2, crop, 1)
    Engine(m2).evaluate_all()
    out = crop.output
    assert out is not None and out.ndim == 3, "crop produced an image"
    # Crop area ~ 70*60 = 4200 (full-res). Without scaling the contours it would be
    # ~1/4 of that — so this proves the contours were mapped back up.
    area = out.shape[0] * out.shape[1]
    assert 3000 < area < 5800, f"cropped original region ~4200 px, got {area} ({out.shape[:2]})"
    print("OK  resize: scales contours (points+shape+bg); downscaled segmentation -> crop the original")


def test_largest_contour_outline():
    img = np.zeros((120, 160), np.uint8)
    cv2.circle(img, (45, 60), 30, 255, -1); cv2.circle(img, (120, 40), 14, 255, -1)
    m = GraphModel(); s = _src(m, img)
    fc = _op(m, "find_contours"); lc = _op(m, "largest_contour", count=1)
    m.add_edge(s, fc); m.add_edge(fc, lc); Engine(m).evaluate_all()
    prev = REGISTRY["largest_contour"].render_preview([fc.output], lc.output, dict(lc.params))
    assert isinstance(prev, np.ndarray) and prev.ndim == 3, "largest_contour draws a preview"
    # The big circle's interior (a flat 255 blob) is dimmed so the bold outline pops.
    assert int(prev[60, 45].max()) < 160, "kept-contour preview dims the binary backdrop"
    print("OK  largest_contour: kept outlines drawn boldly on a dimmed backdrop")


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


def test_detect_hue_robust():
    from core.operations import _detect_centers as dcz
    def chrom(out):
        return out["kinds"].count("chromatic")

    # One saturated hue -> exactly one chromatic centre.
    red = np.zeros((100, 100, 3), np.uint8); red[:] = (0, 0, 200)          # saturated red
    assert chrom(dcz(red, "bgr", 8, 3.0, 0.3, 8.0)) == 1, "one saturated hue -> one chromatic centre"

    # Neutral (gray) pixels carry no hue: a gray-noise region (R=G=B, so C*≈0) must
    # NOT create a phantom hue centre — the C* gate routes it to neutral L* instead.
    mixed = red.copy()
    g = np.random.RandomState(0).randint(80, 180, (100, 40, 1)).astype(np.uint8)
    mixed[:, 60:] = np.repeat(g, 3, axis=2)                                # chroma 0
    assert chrom(dcz(mixed, "bgr", 8, 3.0, 0.3, 8.0)) == 1, \
        "neutral (gray) pixels must not add a phantom hue centre"

    # Circular hue: two reds either side of the 0°/360° wrap are ONE centre, not two.
    import colorsys
    wrap = np.zeros((100, 100, 3), np.uint8)
    for col, deg in ((slice(None, 50), 4), (slice(50, None), 356)):
        r, gg, b = colorsys.hsv_to_rgb(deg / 360.0, 1.0, 0.8)
        wrap[:, col] = (int(b * 255), int(gg * 255), int(r * 255))
    assert chrom(dcz(wrap, "bgr", 8, 2.0, 0.1, 8.0)) == 1, \
        "hue wraps: ~4° and ~356° are adjacent, one centre"

    # sat_weight reshapes the chroma-weighted hue histogram: a vivid hue and a paler
    # hue of a different colour weight differently as the exponent changes.
    tinted = np.zeros((100, 100, 3), np.uint8)
    tinted[:, :50] = (0, 0, 200)              # vivid red (high C*)
    tinted[:, 50:] = (200, 150, 150)          # pale blue tint (lower C*)
    d0 = dcz(tinted, "bgr", 8, 3.0, 0.3, 8.0, sat_weight=0.0, return_diag=True)
    d3 = dcz(tinted, "bgr", 8, 3.0, 0.3, 8.0, sat_weight=3.0, return_diag=True)
    assert not np.allclose(d0["detdiag"]["hue"]["smooth"], d3["detdiag"]["hue"]["smooth"]), \
        "sat_weight reshapes the chroma-weighted hue histogram"
    print("OK  detect_centers hue: C*-gated + chroma-weighted + circular (no phantom/duplicate centres)")


def test_peak_prominence():
    from core.operations import _detect_centers as dcz, _topographic_prominence as tp
    # A tiny saturated-red feature (~0.6%) on a large saturated-blue background: the
    # small but isolated hue peak is kept as its own chromatic centre (topographic
    # prominence judges it against its own valley, not the dominant background peak).
    img = np.full((100, 100, 3), (200, 60, 60), np.uint8)   # blue-ish background
    img[2:10, 2:10] = (60, 60, 200)                          # small red feature
    assert dcz(img, "bgr", 8, 2.0, 0.2, 8.0)["kinds"].count("chromatic") >= 2, \
        "a small colored feature on a uniform background must be its own centre"

    # Topographic prominence in detail (the shared peak helper): an isolated peak
    # rising from an empty valley has ~full prominence; a bump on the shoulder of a
    # taller peak has low prominence and is rejected as noise.
    hist = np.array([0, 2, 0, 0, 10, 6, 6.5, 6, 0], np.float32)  # peaks at 1, 4, shoulder at 6
    assert tp(hist, 1, False) == 2.0, "isolated small peak: prominence == its height"
    assert tp(hist, 4, False) == 10.0, "the tall peak rises the full way from the valley"
    assert tp(hist, 6, False) == 0.5, "the shoulder bump beside the tall peak has tiny prominence"
    assert tp(np.array([0, 0, 5, 5, 5, 0, 0], np.float32), 2, False) == 5.0, "plateau keeps full prominence"
    assert tp(np.array([10, 8, 2, 0], np.float32), 0, False) == 10.0, "boundary peak keeps full prominence"
    print("OK  detect_centers: topographic prominence keeps small features + boundaries, rejects shoulders")


def test_peak_subpeak_vs_step():
    from core.operations import _find_prominent_peaks as fp
    # A sub-peak nested in a 'mountain range': the 5 (just before a taller 8) is a
    # real mode and must be detected, even though it only dips 1 on the side facing
    # the 8 — the mean-valley test sees it rise well above mean(0, 4).
    peaks = fp([0, 0, 3, 5, 4, 7, 8, 3, 0], 0.3, False)
    assert 3 in peaks and 6 in peaks, f"sub-peak (idx 3) and main peak (idx 6) detected, got {peaks}"
    # A quasi-flat step: the plateau shoulder does NOT dip on its right side, so the
    # both-sides-dip guard rejects it — only the genuine peak (the 9) is kept.
    step = fp([0, 0, 5, 5, 5, 5, 9, 0], 0.3, False)
    assert 6 in step and 2 not in step, f"flat-step shoulder rejected, only idx 6 kept, got {step}"
    print("OK  peak detection: mean-valley prominence keeps sub-peaks; both-sides-dip rejects flat steps")


def test_detect_centers():
    from core.operations import _detect_centers, _pixel_chroma
    # Red + green blobs (two hues) + white + black (two neutral levels).
    # LCh detection: 2 chromatic hue seeds + 2 neutral L* seeds = 4.
    seg = np.zeros((40, 40, 3), np.uint8)
    seg[:20, :20] = (40, 40, 200); seg[:20, 20:] = (40, 200, 40)
    seg[20:, :20] = (255, 255, 255); seg[20:, 20:] = (10, 10, 10)
    out = _detect_centers(seg, "bgr", max_k=12, smoothing=2.0,
                          min_prominence=0.3, chroma_threshold=10.0)
    assert out["k"] == 4, f"2 hues + 2 neutral levels = 4 seeds (got {out['k']})"
    assert out["lab"].shape == (4, 3) and out["bgr"].shape == (4, 3)
    assert out["kinds"].count("chromatic") == 2 and out["kinds"].count("neutral") == 2
    chrom = _pixel_chroma(np.clip(out["bgr"], 0, 255).astype(np.uint8).reshape(1, -1, 3))
    assert int((chrom < 25).sum()) == 2, "the two neutral seeds are achromatic (white/black)"

    # The neutral count is ADAPTIVE, not a fixed gray_levels: add a mid-gray band
    # and a third neutral L* seed appears on its own (the old node always gave 2).
    g3 = np.zeros((30, 30, 3), np.uint8)
    g3[:10] = 10; g3[10:20] = 128; g3[20:] = 245      # black / mid / white, no hue
    o3 = _detect_centers(g3, "bgr", max_k=12, smoothing=2.0,
                         min_prominence=0.3, chroma_threshold=10.0)
    assert o3["kinds"].count("neutral") == 3, \
        f"neutral levels adapt to the image (got {o3['kinds'].count('neutral')}, want 3)"

    # max_k caps the seed count (best-supported kept).
    capped = _detect_centers(seg, "bgr", max_k=2, smoothing=2.0,
                             min_prominence=0.3, chroma_threshold=10.0)
    assert capped["k"] == 2, f"max_k caps the seeds (got {capped['k']})"

    # min_area drops a tiny-but-prominent speck: a big red field + a 9px blue speck
    # (~0.36%). It's its own centre with no floor, but filtered at a 1% floor — while
    # the single biggest centre is always kept (never an empty result).
    speck = np.zeros((50, 50, 3), np.uint8); speck[:] = (0, 0, 200)      # red field
    speck[0:3, 0:3] = (200, 0, 0)                                        # 9px blue speck
    no_floor = _detect_centers(speck, "bgr", 8, 2.0, 0.1, 8.0, min_area=0.0)
    assert no_floor["kinds"].count("chromatic") == 2, "no floor: the speck is its own centre"
    floored = _detect_centers(speck, "bgr", 8, 2.0, 0.1, 8.0, min_area=0.01)  # 1% = 25px
    assert floored["kinds"].count("chromatic") == 1, "min_area drops the 9px (<1%) speck"
    assert floored["k"] >= 1, "the biggest centre is always kept (never empty)"
    # The preview's hue-histogram markers track min_area too: the dropped speck loses
    # its indicator (like min_prominence drops an undetected peak), not just the seed.
    nf_d = _detect_centers(speck, "bgr", 8, 2.0, 0.1, 8.0, min_area=0.0, return_diag=True)
    fl_d = _detect_centers(speck, "bgr", 8, 2.0, 0.1, 8.0, min_area=0.01, return_diag=True)
    assert len(nf_d["detdiag"]["hue"]["peaks"]) == 2, "no floor: both hue peaks marked"
    assert len(fl_d["detdiag"]["hue"]["peaks"]) == 1, "min_area removes the speck's histogram marker"
    assert len(fl_d["detdiag"]["hue"]["peak_colors"]) == 1, "marker colours stay aligned with kept peaks"

    # End-to-end through the registered op: it emits a CENTERS payload with a
    # detection diagnostic for the preview.
    m = GraphModel()
    s = _src(m, seg)
    d = _op(m, "detect_centers", max_k=12, smoothing=2.0, min_prominence=0.3,
            chroma_threshold=10.0)
    m.add_edge(s, d)
    Engine(m).evaluate_all()
    assert isinstance(d.output, dict) and d.output["k"] == 4, "op should emit 4 seeds"
    assert "detdiag" in d.output, "op stashes the detection diagnostic for the preview"
    from core import codegen
    from core.operations import REGISTRY
    code = codegen.op_pseudocode(REGISTRY["detect_centers"],
                                 {"max_k": 12, "smoothing": 4.0, "min_prominence": 0.3,
                                  "chroma_threshold": 8.0, "sat_weight": 1.0})
    assert "CIELAB" in code and "CENTERS" in code, code
    print("OK  detect_centers: LCh hue + adaptive neutral L* seeds, capped by max_k")


def test_merge_close_seeds():
    from core.operations import _merge_close_seeds

    def seed(lab, n, kind, source, b):           # comps carry (source, bin, colour, support)
        col = np.array([n % 256, n % 256, n % 256], np.float32)
        return {"lab": np.array(lab, np.float32), "bgr": col,
                "support": n, "kind": kind, "comps": [(source, b, col, n)]}

    # The chroma-cut artefact: one bright cluster split into a neutral-L* seed and a
    # hue seed at nearly the same Lab point (ΔE ≈ 8.5). They merge into one centre,
    # and the LOWER-support part (the hue seed) folds into the higher (neutral) one.
    a = seed([80, 6, 6], 300, "neutral", "lum", 200)
    b = seed([80, 12, 12], 100, "chromatic", "hue", 20)
    merged = _merge_close_seeds([a, b], max_de=15.0)
    assert len(merged) == 1, "two near-identical Lab centres merge into one"
    m = merged[0]
    assert m["support"] == 400, "merged support = sum of parts"
    assert m["kind"] == "chromatic", "a merged centre with any hue is chromatic"
    expect = (np.array([80, 6, 6]) * 300 + np.array([80, 12, 12]) * 100) / 400
    assert np.allclose(m["lab"], expect), "merged centre = support-weighted mean (exact union centroid)"
    assert {(c[0], c[1]) for c in m["comps"]} == {("lum", 200), ("hue", 20)}, "owns BOTH basins"
    dom = max(m["comps"], key=lambda c: c[3])    # highest-support component = drawn solid
    assert dom[0] == "lum" and dom[3] == 300, "the higher-support part is dominant (solid); lower is dashed"

    # Distinct hues (far apart in Lab) never merge; max_de=0 disables merging entirely.
    far = _merge_close_seeds([seed([50, 70, 55], 100, "chromatic", "hue", 0),
                              seed([60, -50, 40], 100, "chromatic", "hue", 60)], max_de=15.0)
    assert len(far) == 2, "distinct hues stay separate"
    assert len(_merge_close_seeds([a, b], max_de=0.0)) == 2, "max_de=0 disables merging"
    print("OK  merge_close_seeds: re-fuses near-identical Lab centres; lower-support folds into higher")


def test_detect_merge_straddle():
    from core.operations import _detect_centers, _lab_lch
    # A single bright warm-white cluster whose two halves straddle the chroma cut: a
    # near-neutral half (low C*) and a faintly-tinted half (higher C*, ~same L/hue).
    # With the cut between them, detection makes a neutral-L* seed AND a hue seed at
    # almost the same Lab point — merge_distance re-fuses them into one centre.
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :30] = (238, 240, 243)     # near-neutral bright (low chroma), 1200px — the LARGER half
    img[:, 30:] = (228, 238, 250)     # faint warm tint (higher chroma), 400px — the smaller half
    _, _, C, _ = _lab_lch(img)
    C2 = C.reshape(40, 40)
    cA, cB = float(C2[:, :30].mean()), float(C2[:, 30:].mean())
    thr = 0.5 * (cA + cB)
    assert cA < thr < cB, f"fixture must straddle the cut (cA={cA:.1f}, cB={cB:.1f})"
    off = _detect_centers(img, "bgr", 8, 2.0, 0.2, thr, merge_distance=0.0)
    on = _detect_centers(img, "bgr", 8, 2.0, 0.2, thr, merge_distance=40.0)
    assert off["k"] == 2, f"without merge: the cut splits one cluster into two seeds (got {off['k']})"
    assert on["k"] == 1, f"merge re-fuses them into one centre (got {on['k']})"
    assert int(on["support"][0]) == 1600, "the merged centre keeps all the pixels"
    # The surviving merged centre still owns both histograms' pixels in the scatter,
    # AND the smaller (hue) component is preserved as a DASHED marker on the hue band
    # while the larger (neutral) one is the solid centre.
    diag = _detect_centers(img, "bgr", 8, 2.0, 0.2, thr, merge_distance=40.0, return_diag=True)
    assert int((diag["member"] >= 0).sum()) == 1600, "merged centre claims both basins (no noise)"
    dd = diag["detdiag"]
    assert len(dd["lum"]["peaks"]) == 1, "the larger (neutral) component stays solid"
    assert len(dd["hue"]["peaks"]) == 0, "the smaller (hue) component is not solid"
    assert len(dd["hue"]["dashed_peaks"]) == 1, "the merged-away (smaller) component is drawn dashed"
    print("OK  detect_centers: merge re-fuses a split cluster; lower-support peak kept as a dashed marker")


def test_peak_basin():
    from core.operations import _peak_basin, _basin_bins
    # Isolated circular peak on a near-zero plain: the basin must CONTAIN the peak.
    # (The old code never saw a "rise", so it wrapped around and returned the
    # complement arc that EXCLUDED the peak — half a cluster shown as scatter noise.)
    hist = np.zeros(180, np.float32); hist[34] = 10.0; hist[33] = hist[35] = 4.0
    lo, hi = _peak_basin(hist, 34, circular=True)
    bins = set(_basin_bins(lo, hi, 180, True).tolist())
    assert 34 in bins, "an isolated circular peak's basin must contain the peak"
    assert len(bins) < 60, "basin is the local bump, not the whole-circle complement"
    # A real valley between two peaks still bounds the basin (non-circular).
    h2 = np.array([0, 6, 10, 6, 1, 0, 1, 7, 12, 7, 0], np.float32)
    lo2, hi2 = _peak_basin(h2, 2, circular=False)
    assert lo2 <= 2 <= hi2 and hi2 <= 5, "basin stops at the valley between the two peaks"
    print("OK  peak_basin: isolated peak keeps its bump; a valley bounds adjacent peaks")


def test_assign_centers():
    from core import datatypes, codegen
    from core.operations import REGISTRY
    # Detect -> Assign -> Reduce: four flat regions round-trip to ~themselves.
    seg = np.zeros((40, 40, 3), np.uint8)
    seg[:20, :20] = (40, 40, 200); seg[:20, 20:] = (40, 200, 40)
    seg[20:, :20] = (255, 255, 255); seg[20:, 20:] = (10, 10, 10)

    for algo in ("nearest", "kmeans"):
        m = GraphModel()
        s = _src(m, seg)
        det = _op(m, "detect_centers", max_k=12, smoothing=2.0, min_prominence=0.3,
                  chroma_threshold=10.0)
        asg = _op(m, "assign_centers", algorithm=algo, lum_weight=1.0)
        red = _op(m, "reduce_colors")
        m.add_edge(s, det)
        m.add_edge(s, asg, 0)            # image   -> port 0
        m.add_edge(det, asg, 1)          # centers -> port 1
        m.add_edge(asg, red)
        Engine(m).evaluate_all()
        out = asg.output
        assert isinstance(out, dict) and "labels" in out, f"{algo}: CLUSTERS payload"
        assert out["k"] == 4, f"{algo}: 4 centres -> 4 clusters (got {out['k']})"
        lab = out["labels"].reshape(40, 40)
        quads = {int(lab[10, 10]), int(lab[10, 30]), int(lab[30, 10]), int(lab[30, 30])}
        assert len(quads) == 4, f"{algo}: the 4 regions get 4 distinct labels (got {quads})"
        quant = red.output
        assert quant is not None and quant.shape == seg.shape, f"{algo}: reduce rebuilds the image"
        # each region recoloured close to its own mean (round-trip is near-lossless here)
        assert np.allclose(quant[5, 5], seg[5, 5], atol=12), f"{algo}: red region preserved"

    # CENTERS feeds an assign port but NOT a clusters port (Reduce Colors).
    assert datatypes.compatible(datatypes.CENTERS, datatypes.CENTERS)
    assert not datatypes.compatible(datatypes.CENTERS, datatypes.CLUSTERS), \
        "CENTERS must not wire straight into a CLUSTERS input"
    acode = codegen.op_pseudocode(REGISTRY["assign_centers"],
                                  {"algorithm": "kmeans", "lum_weight": 1.0})
    assert "KMEANS_USE_INITIAL_LABELS" in acode and "CLUSTERS" in acode, acode
    print("OK  assign_centers: nearest + k-means-refine label every pixel; CENTERS type-gated")


def test_cluster_preview_diag():
    # The clustering preview is data-driven: compute() must stash the diagnostics
    # (per-cluster counts/spread + a subsampled feature-space scatter for K-Means /
    # Assign; the detection histograms for Detect) so render_preview is a cheap draw.
    img = np.full((90, 90, 3), 110, np.uint8)
    img[5:40, 5:40] = (40, 40, 200); img[5:40, 50:85] = (40, 200, 40)
    img[50:85, 5:40] = (200, 120, 40); img[50:85, 50:85] = (90, 90, 90)

    def run(op_id, **params):
        m = GraphModel(); s = _src(m, img)
        n = _op(m, op_id, **params); m.add_edge(s, n); Engine(m).evaluate_all()
        return n.output, REGISTRY[op_id].render_preview([img], n.output, params)

    km, prev = run("kmeans", k=5, cluster_space="lab", lum_weight=0.4)
    d = km["diag"]
    assert d["counts"].sum() == 90 * 90, "palette counts must cover every pixel"
    assert d["spread"].shape == (5,) and d["scatter"].shape[1] == 2, "scatter is 2D, spread per-k"
    assert len(d["scatter"]) == len(d["scatter_labels"]) and d["scatter_labels"].max() < 5
    assert prev is not None and prev.ndim == 3 and prev.shape[2] == 3, "kmeans preview is a BGR image"

    # Detect Color Centers stashes the two detection histograms for its preview.
    det, prev = run("detect_centers", max_k=8, smoothing=2.0, min_prominence=0.3,
                    chroma_threshold=8.0)
    dd = det["detdiag"]
    assert "hue" in dd and "lum" in dd, "detection diagnostic carries both histograms"
    assert len(dd["hue"]["raw"]) == len(dd["hue"]["smooth"]), "hue curves same length"
    assert prev is not None and prev.ndim == 3, "detect_centers preview renders"

    # Assign to Centers (detect -> assign) produces the standard K-Means-style diag.
    m = GraphModel(); s = _src(m, img)
    de = _op(m, "detect_centers", max_k=8, smoothing=2.0, min_prominence=0.3,
             chroma_threshold=8.0)
    asg = _op(m, "assign_centers", algorithm="nearest", lum_weight=0.4)
    m.add_edge(s, de); m.add_edge(s, asg, 0); m.add_edge(de, asg, 1)
    Engine(m).evaluate_all()
    ad = asg.output["diag"]
    assert ad["counts"].sum() == 90 * 90, "assign palette counts cover every pixel"
    aprev = REGISTRY["assign_centers"].render_preview([img], asg.output, {})
    assert aprev is not None and aprev.ndim == 3, "assign_centers preview renders"
    print("OK  cluster preview: compute() stashes palette/scatter/spread + detection histograms")


def test_normalize_lighting():
    base = np.full((60, 60, 3), 110, np.uint8); base[10:50, 10:50] = (40, 40, 200)
    lit = np.clip(base.astype(np.float32) * np.array([0.8, 1.0, 1.4]) * 1.2,
                  0, 255).astype(np.uint8)   # warm cast + brightness gain

    def norm(img, mode):
        m = GraphModel(); s = _src(m, img)
        n = _op(m, "normalize_lighting", mode=mode); m.add_edge(s, n)
        Engine(m).evaluate_all()
        return n.output

    before = float(np.abs(base.astype(int) - lit.astype(int)).mean())
    for mode in ("grayworld", "global", "flatfield"):
        after = float(np.abs(norm(base, mode).astype(int) - norm(lit, mode).astype(int)).mean())
        assert after < before, f"{mode} should shrink the same-object lighting gap"
    print("OK  normalize_lighting: shrinks same-object cross-lighting differences")


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


def test_param_help_present():
    # Every parameter the panel shows must carry a `help` blurb (how it affects the
    # result) so its tooltip explains it — not just repeats the name.
    missing = [f"{op.id}.{p.name}" for op in REGISTRY.values() for p in op.params
               if getattr(p, "show", True) and not (getattr(p, "help", "") or "").strip()]
    assert not missing, f"visible params without a help blurb: {missing}"
    print(f"OK  every visible parameter documents its effect ({sum(1 for op in REGISTRY.values() for p in op.params if getattr(p, 'show', True))} params)")


def main():
    test_linear_chain_and_caching()
    test_dirty_propagation()
    test_arity_gating()
    test_input_order()
    test_error_capture()
    test_persistence_roundtrip()
    test_color_pipeline()
    test_hdbscan_cluster()
    test_contours()
    test_contour_nesting_colors()
    test_label_regions()
    test_floodfill_nesting_colors()
    test_connected_components()
    test_segmentation_nodes()
    test_crop_no_clip_when_deskewed()
    test_crop_negative_border()
    test_codegen()
    test_codegen_clustering_detail()
    test_fourier_roundtrip()
    test_more_ops()
    test_auto_threshold()
    test_backproject()
    test_conversions()
    test_batched()
    test_create_batch()
    test_resize()
    test_resize_fixed_length()
    test_resize_contours()
    test_largest_contour_outline()
    test_rotate()
    test_normalize()
    test_invert()
    test_local_hdr()
    test_detect_hue_robust()
    test_peak_prominence()
    test_peak_subpeak_vs_step()
    test_detect_centers()
    test_merge_close_seeds()
    test_detect_merge_straddle()
    test_peak_basin()
    test_assign_centers()
    test_cluster_preview_diag()
    test_normalize_lighting()
    test_cluster_space()
    test_mean_shift()
    test_comp_timing_and_traversal()
    test_codegen_covers_cv_calls()
    test_cycle_prevention()
    test_param_help_present()
    test_blas_thread_pinned()
    print("\nENGINE OK: 50 backend tests passed")


if __name__ == "__main__":
    main()
