"""Pseudocode generation from a pipeline (backend, Qt-free).

Walks the graph *upstream* from a target node and emits a readable,
language-neutral program (cv:: -style calls + comments) describing how to
reproduce the prototyped pipeline: source image -> each operation with its
parameters -> the target. Simple nodes that are a single OpenCV call render as
one line; custom nodes (clustering, region labelling, …) render as a short
commented block explaining what they actually do.

Used by the Export Code node (full pipeline) and the Function-info tooltip
(single op, via :func:`op_pseudocode`).
"""
from __future__ import annotations

import re
from typing import List, Optional

from core.graph import GraphModel, GraphNode

_CV_CALL_RE = re.compile(r"cv::\w+")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _first_para(text: Optional[str]) -> str:
    """First paragraph of a docstring, whitespace-collapsed."""
    if not text:
        return ""
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            break
        out.append(line)
    return " ".join(out)


def op_description(op) -> str:
    """The op's human description: its explicit ``description`` or, failing that,
    the first paragraph of its ``compute`` docstring."""
    desc = getattr(op, "description", "") or ""
    if desc:
        return desc
    return _first_para(getattr(getattr(op, "compute", None), "__doc__", "") or "")


def _enum_label(op, name, value):
    """Map an enum/choice param value back to its menu label (e.g. 0 -> 'Hue (H)')."""
    for spec in op.params:
        if spec.name == name and spec.kind in ("enum", "choice") and spec.choices:
            for label, val in spec.choices:
                if val == value:
                    return label
    return None


def _param_bits(op, params) -> List[str]:
    """`key=value` fragments for an op's params, enums shown by their label."""
    bits = []
    for spec in op.params:
        if not getattr(spec, "show", True):
            continue
        val = params.get(spec.name, spec.default)
        lab = _enum_label(op, spec.name, val)
        bits.append(f"{spec.name}={lab!r}" if lab is not None else f"{spec.name}={val}")
    return bits


def _var_base(op) -> str:
    """A short variable-name stem from an op id (e.g. 'adaptive_threshold' -> 'thr')."""
    return _VAR_BASE.get(op.id, op.id.split("_")[0])


_VAR_BASE = {
    "to_grayscale": "gray", "to_bgr": "bgr", "to_hls": "hls",
    "gaussian_blur": "blur", "adaptive_threshold": "thr", "threshold": "thr",
    "find_contours": "contours", "contour_filter": "contours",
    "label_regions": "regions", "connected_components": "regions",
    "kmeans": "clusters", "reduce_colors": "quant",
    "detect_centers": "centres", "assign_centers": "clusters",
    "mean_shift": "seg", "morphology": "morph", "local_hdr": "hdr",
    "create_batch": "batch", "normalize": "norm",
}


# ---------------------------------------------------------------------------
# per-op emitters: (out_var, in_vars, params) -> str | list[str]
# Simple cv:: calls render inline; custom ops add explanatory comments.
# ---------------------------------------------------------------------------
def _cvt(code):
    return lambda o, i, p: f"{o} = cv::cvtColor({i[0]}, {code})"


_CODE = {
    # --- conversions ---
    "to_grayscale": _cvt("COLOR_BGR2GRAY"),
    "to_bgr": lambda o, i, p: f"{o} = cv::cvtColor({i[0]}, COLOR_*2BGR)   # from tracked space",
    "to_hls": _cvt("COLOR_BGR2HLS"),
    # --- blur / edges ---
    "blur": lambda o, i, p: f"{o} = cv::blur({i[0]}, ksize=({p.get('kernel_size')},{p.get('kernel_size')}))",
    "gaussian_blur": lambda o, i, p: f"{o} = cv::GaussianBlur({i[0]}, ksize=({p.get('kernel_size')},{p.get('kernel_size')}), sigma=0)",
    "canny": lambda o, i, p: f"{o} = cv::Canny({i[0]}, {p.get('threshold1')}, {p.get('threshold2')}, apertureSize={p.get('aperture')})   # gray via cv::cvtColor",
    "sobel": lambda o, i, p: f"{o} = cv::convertScaleAbs(cv::Sobel({i[0]}, CV_64F, dx={p.get('dx')}, dy={p.get('dy')}, ksize={p.get('ksize')}))   # gray via cv::cvtColor",
    "laplacian": lambda o, i, p: f"{o} = cv::convertScaleAbs(cv::Laplacian({i[0]}, CV_64F, ksize={p.get('ksize')}))   # gray via cv::cvtColor",
    "morphology": lambda o, i, p: f"{o} = cv::morphologyEx({i[0]}, op={p.get('operation')}, kernel=cv::getStructuringElement(MORPH_RECT, ({p.get('kernel_size')},{p.get('kernel_size')})), iterations={p.get('iterations')})",
    # --- threshold ---
    "threshold": lambda o, i, p: f"{o} = cv::threshold({i[0]}, thresh={p.get('threshold_value')}, maxval={p.get('max_value')}, type={p.get('threshold_type')})   # gray via cv::cvtColor",
    "adaptive_threshold": lambda o, i, p: f"{o} = cv::adaptiveThreshold({i[0]}, maxValue={p.get('max_value')}, method={p.get('adaptive_method')}, type={p.get('threshold_type')}, blockSize={p.get('block_size')}, C={p.get('c')})   # gray via cv::cvtColor",
    # --- geometry ---
    "resize": lambda o, i, p: (
        f"s = {p.get('length')} / max(h, w); {o} = cv::resize({i[0]}, (round(w*s), round(h*s)), "
        f"interpolation={p.get('interpolation')})   # longer edge -> {p.get('length')} px (or scale contour points by s)"
        if p.get("mode") == "fixed" else
        f"{o} = cv::resize({i[0]}, None, fx={p.get('scale')}, fy={p.get('scale')}, "
        f"interpolation={p.get('interpolation')})   # or scale contour points by {p.get('scale')} if {i[0]} is contours"),
    "rotate": lambda o, i, p: [f"M = cv::getRotationMatrix2D(center, angle={p.get('angle')}, scale=1.0)",
                               f"{o} = cv::warpAffine({i[0]}, M, dsize)   # expand={p.get('expand')}"],
    # --- arithmetic (the second input is aligned to the first: cv::resize/cv::cvtColor) ---
    "sum": lambda o, i, p: f"{o} = cv::addWeighted({i[0]}, {p.get('alpha')}, {i[1]}, {1 - float(p.get('alpha', 0.5))}, 0)   # + cv::resize to align",
    "and": lambda o, i, p: f"{o} = cv::bitwise_and({i[0]}, {i[1]})   # + cv::resize / cv::cvtColor to align",
    "diff": lambda o, i, p: f"{o} = cv::subtract({i[0]}, {i[1]})   # + cv::resize / cv::cvtColor to align",
    "invert": lambda o, i, p: f"{o} = cv::bitwise_not({i[0]})",
    # --- fourier ---
    "dft": lambda o, i, p: f"{o} = cv::dft(float({i[0]}), flags=DFT_COMPLEX_OUTPUT)   # gray via cv::cvtColor; preview: cv::magnitude",
    "idft": lambda o, i, p: f"{o} = cv::idft({i[0]}, flags=DFT_REAL_OUTPUT | DFT_SCALE)",
    # --- binary connected components ---
    "connected_components": lambda o, i, p: [
        f"# label connected blobs in a binary image ({p.get('connectivity')}-connectivity)",
        f"n, labels = cv::connectedComponents(({i[0]} > 0), connectivity={p.get('connectivity')})",
        f"# background for drawing: cv::cvtColor(gray -> BGR)",
        f"{o} = [cv::findContours outer contour of (labels == k) for k in 1..n-1]"],
}


def _emit_feature_block(src, space, lum_weight):
    """The exact feature-space construction shared by K-Means and Auto Cluster."""
    lines = [f"bgr = as_bgr({src})                                  # cv::cvtColor from the tracked colour space"]
    if space == "lab":
        lines += [
            "feat = cv::cvtColor(bgr, COLOR_BGR2Lab).reshape(-1, 3)   # cluster in Lab",
            f"feat[:, 0] *= {lum_weight}                           # scale L (lightness) by lum_weight",
        ]
    elif space == "hls":
        lines += [
            "feat = cv::cvtColor(bgr, COLOR_BGR2HLS).reshape(-1, 3)   # cluster in HLS",
            f"feat[:, 1] *= {lum_weight}                           # scale L (lightness) by lum_weight",
        ]
    else:
        lines.append("feat = bgr.reshape(-1, 3).astype(float32)            # cluster in BGR (lum_weight n/a)")
    return lines


def _emit_cluster_tail(o, p):
    """The k-means call + center/order steps, shared by both clustering ops."""
    return [
        "cv::setRNGSeed(0)                                    # deterministic init (no per-call seed)",
        "_, labels, _ = cv::kmeans(feat, K, criteria=(EPS|MAX_ITER, 10, 1.0), attempts=5, KMEANS_PP_CENTERS)",
        "centers[c] = mean INPUT-space colour of pixels labelled c   # report true colours, not feature space",
        "reorder clusters dark->light by perceptual luminance; remap labels   # stable ids/colours",
        f"{o} = {{centers, labels, shape, k=K}}               # CLUSTERS payload",
    ]


def _emit_kmeans(o, i, p):
    space, lw = p.get("cluster_space"), p.get("lum_weight")
    out = [f"# K-Means colour clustering: k={p.get('k')}, space={space!r}, lum_weight={lw}, attempts={p.get('attempts')}"]
    out += _emit_feature_block(i[0], space, lw)
    out += [f"K = {p.get('k')}"]
    out += _emit_cluster_tail(o, p)
    return out


def _emit_hdbscan(o, i, p):
    space = p.get("color_space", "lab")
    algo = {"optics": "optics-xi"}.get(p.get("algorithm", "hdbscan"), p.get("algorithm", "hdbscan"))
    noise = ("assign each noise pixel to its nearest cluster"
             if p.get("noise_handling", "nearest") == "nearest" else "paint noise pixels the flag colour")
    return [
        f"# Density colour clustering (no k) via optics.cluster_image: algo={algo!r}, space={space!r}",
        f"bgr = as_bgr({i[0]})                                 # cv::cvtColor from the tracked colour space",
        f"res = optics.cluster_image(bgr, algo={algo!r}, space={space!r}, voxel={p.get('voxel_bin')}, "
        f"bgr=True, min_cluster_size={p.get('min_cluster_size')}, min_cluster_frac={p.get('min_cluster_frac')})",
        "#   the library converts sRGB->CIELAB, voxel-quantizes and dedups internally",
        "labels = res.labels                                  # per-pixel cluster id (HxW), -1 = noise",
        f"centers[c] = mean BGR of pixels labelled c; reorder dark->light; {noise}",
        f"{o} = {{centers, labels, shape, k}}   # CLUSTERS payload",
    ]


def _emit_detect_centers(o, i, p):
    return [
        f"# Detect colour-cluster seeds in CIELAB/LCh: max_k={p.get('max_k')}, "
        f"chroma_threshold(C*)={p.get('chroma_threshold')}",
        f"bgr = as_bgr({i[0]})                                 # cv::cvtColor from the tracked colour space",
        "lab = srgb_to_lab(bgr)                               # true CIELAB (L 0..100), pure NumPy",
        "L, C, h = lab[:,0], hypot(lab[:,1], lab[:,2]), atan2(lab[:,2], lab[:,1])  # LCh",
        f"# chromatic (C* >= {p.get('chroma_threshold')}): circular hue histogram weighted by "
        f"C*^{p.get('sat_weight')}, cv::GaussianBlur(sigma={p.get('smoothing')})",
        f"# neutral   (C* <  {p.get('chroma_threshold')}): lightness L* histogram (adaptive count)",
        f"peaks = local maxima with prominence >= {p.get('min_prominence')} * height (both-sides dip)",
        f"{o} = {{lab seeds, bgr seeds}} = mean Lab/BGR of each peak's pixels   # CENTERS payload",
    ]


def _emit_assign_centers(o, i, p):
    algo = p.get("algorithm", "nearest")
    out = [
        f"# Assign each pixel to a detected colour centre ({i[1]}), in CIELAB "
        f"(L* scaled by lum_weight={p.get('lum_weight')})",
        f"bgr = as_bgr({i[0]})                                 # cv::cvtColor from the tracked space",
        f"lab = srgb_to_lab(bgr)                               # true CIELAB",
        f"labels = argmin_c || lab_pixel - {i[1]}.seeds[c] ||²    # nearest centre (ΔE 1-NN)",
    ]
    if algo == "kmeans":
        out += ["cv::setRNGSeed(0)                                   # deterministic refine",
                f"labels = cv::kmeans(lab, K={i[1]}.k, bestLabels=labels, "
                "KMEANS_USE_INITIAL_LABELS)   # refine the centres from the seeds"]
    else:
        out.append("# (algorithm='kmeans' would cv::setRNGSeed(0) + cv::kmeans to refine the centres from these labels)")
    out += [
        "centers[c] = mean INPUT-space colour of pixels labelled c; reorder dark->light",
        f"{o} = {{centers, labels, shape, k}}   # CLUSTERS payload",
    ]
    return out


def _emit_reduce_colors(o, i, p):
    return [f"{o} = centers[labels].reshape(shape)   # rebuild image from {i[0]} (CLUSTERS)"]


def _emit_label_regions(o, i, p):
    chan = p.get("channel")
    exact = p.get("delta", 0) == 0
    # Mention both code paths' cv:: calls regardless of the current delta, so the
    # node is searchable by either OpenCV function it can call.
    return [
        f"# group connected near-uniform regions (channel={chan!r}, "
        f"delta={p.get('delta')}, {p.get('connectivity')}-connectivity):",
        "#   the channel is selected via cv::cvtColor (BGR<->HLS)",
        "#   delta==0 -> cv::connectedComponents per unique value (exact)",
        "#   delta>0  -> cv::floodFill region-grow (FIXED_RANGE) per seed",
        (f"labels = exact_label({i[0]}, connectivity={p.get('connectivity')})" if exact
         else f"labels = floodfill_label({i[0]}, delta={p.get('delta')}, connectivity={p.get('connectivity')})"),
        f"{o} = [cv::findContours outer contour of (labels == k) for each region k]",
    ]


def _emit_contour_filter(o, i, p):
    return [f"{o} = [c for c in {i[0]} if {p.get('min_area')} <= cv::contourArea(c) <= {p.get('max_area')}]   # preview: cv::drawContours"]


def _emit_find_contours(o, i, p):
    return [
        f"gray = cv::cvtColor({i[0]}, COLOR_BGR2GRAY)",
        f"{o}, hierarchy = cv::findContours(gray, mode={p.get('mode')}, CHAIN_APPROX_SIMPLE)   # preview: cv::drawContours",
    ]


def _emit_mser(o, i, p):
    return [
        f"gray = cv::cvtColor({i[0]}, COLOR_BGR2GRAY)",
        f"mser = cv::MSER_create(delta={p.get('delta')}, min_area={p.get('min_area')}, max_area={p.get('max_area')})",
        f"{o} = mser.detectRegions(gray)"]


def _emit_mean_shift(o, i, p):
    return [
        f"bgr = cv::cvtColor({i[0]} -> BGR)",
        f"{o} = cv::pyrMeanShiftFiltering(bgr, sp={p.get('spatial')}, sr={p.get('color')})"]


def _emit_normalize(o, i, p):
    return [
        f"# mode={p.get('mode')!r} (colour handled per-channel via cv::cvtColor YCrCb):",
        "#   stretch -> cv::normalize(NORM_MINMAX); equalize -> cv::equalizeHist; clahe -> cv::createCLAHE",
        f"{o} = normalize({i[0]}, mode={p.get('mode')!r})"]


def _emit_local_hdr(o, i, p):
    return [
        "# split luma (cv::cvtColor YCrCb), divide by a cv::GaussianBlur low-pass, recombine",
        f"{o} = local_contrast({i[0]})"]


def _emit_histogram(o, i, p):
    space = p.get("color_space", "bgr")
    sig = float(p.get("smoothing", 0.0))
    src = i[0]
    conv = f"cv::cvtColor({i[0]}, COLOR_BGR2HLS)"
    out = []
    if space == "hls":
        out.append(f"img = {conv}   # histogram in HLS (Hue 0..179)")
        src = "img"
    else:
        out.append(f"# color_space='hls' would first {conv}")
    out.append(f"{o} = cv::calcHist({src}) per channel")
    blur = f"cv::GaussianBlur({o}, sigma={p.get('smoothing')})"
    out.append(f"{o} = {blur}   # smooth the curve" if sig > 0
               else f"# smoothing > 0 would {blur} the curve")
    return out


def _emit_create_batch(o, i, p):
    return [f"{o} = Batch([cv::cvtColor(x -> BGR) for x in inputs])   # homogeneous 3-channel stack"]


def _emit_save_to_file(o, i, p):
    return [f"cv::imwrite(path, {i[0]})"]


def _emit_normalize_lighting(o, i, p):
    return [
        f"# reduce lighting differences (mode={p.get('mode')!r}); input via cv::cvtColor:",
        "#   grayworld -> per-channel gain to a common mean; global -> single gain;",
        "#   flatfield -> divide by a large cv::GaussianBlur then rescale",
        f"{o} = normalize_lighting({i[0]}, mode={p.get('mode')!r})"]


def _emit_color_mask(o, i, p):
    return [
        f"# binary mask of pixels within +/-{p.get('delta')} of "
        f"BGR=({p.get('blue')},{p.get('green')},{p.get('red')}):",
        f"{o} = cv::inRange(cv::cvtColor({i[0]} -> BGR), colour-delta, colour+delta)",
        f"# 'outside' (foreground) inverts: {o} = cv::bitwise_not({o})",
    ]


def _emit_auto_threshold(o, i, p):
    m = p.get("method", "otsu")
    inv = "_INV" if p.get("invert") else ""
    out = [f"gray = cv::cvtColor({i[0]}, COLOR_BGR2GRAY)"]
    if m == "valley":
        out += [f"hist = cv::calcHist(gray); hist = cv::GaussianBlur(hist, sigma=2)",
                f"t = deepest valley between the two largest modes",
                f"{o} = cv::threshold(gray, t, 255, BINARY{inv})"]
    else:
        flag = "OTSU" if m == "otsu" else "TRIANGLE"
        out += [f"# Valley method would cv::calcHist + cv::GaussianBlur to find the dip",
                f"{o} = cv::threshold(gray, 0, 255, BINARY{inv} | {flag})"]
    return out


def _emit_backproject(o, i, p):
    return [f"# backproject a histogram model ({i[1]}) onto the target {i[0]}",
            f"px = cv::cvtColor({i[0]}, into the model's colour space)",
            f"{o} = product over channels of model_hist[px], rescaled 0..255   # likelihood map"]


def _emit_largest_contour(o, i, p):
    return [f"{o} = top {p.get('count')} of {i[0]} by cv::contourArea (descending)"]


def _emit_crop_to_contour(o, i, p):
    return [
        f"cnt = max({i[1]} by cv::contourArea)",
        f"rect = cv::minAreaRect(cnt); box = cv::boxPoints(rect)",
        f"M = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0)",
        f"rot = cv::warpAffine(cv::cvtColor({i[0]} -> BGR), M, size)",
        f"{o} = crop(rot, bbox(M*box) + border={p.get('border')})   # + cv::resize if scale != 1",
    ]


_CODE.update({
    "kmeans": _emit_kmeans,
    "detect_centers": _emit_detect_centers,
    "assign_centers": _emit_assign_centers,
    "hdbscan_cluster": _emit_hdbscan,
    "reduce_colors": _emit_reduce_colors,
    "label_regions": _emit_label_regions,
    "mser": _emit_mser,
    "mean_shift": _emit_mean_shift,
    "normalize": _emit_normalize,
    "local_hdr": _emit_local_hdr,
    "histogram": _emit_histogram,
    "auto_threshold": _emit_auto_threshold,
    "backproject": _emit_backproject,
    "create_batch": _emit_create_batch,
    "save_to_file": _emit_save_to_file,
    "color_mask": _emit_color_mask,
    "largest_contour": _emit_largest_contour,
    "crop_to_contour": _emit_crop_to_contour,
    "normalize_lighting": _emit_normalize_lighting,
    "contour_filter": _emit_contour_filter,
    "find_contours": _emit_find_contours,
})


def _emit_op(op, out, ins, params) -> List[str]:
    """Pseudocode line(s) for one op. Uses a per-op emitter when available, else a
    generic call with a description comment from the op's docstring."""
    fn = _CODE.get(op.id)
    if fn is not None:
        res = fn(out, ins, params)
        return res if isinstance(res, list) else [res]
    # Generic fallback: still useful — names the op, its inputs, and params, with
    # a one-line description comment (especially for ops without a cv:: mapping).
    lines = []
    desc = op_description(op)
    if desc:
        lines.append(f"# {desc}")
    arg = ", ".join(ins)
    bits = _param_bits(op, params)
    extra = (", " + ", ".join(bits)) if bits else ""
    lines.append(f"{out} = {op.id}({arg}{extra})")
    return lines


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _ordered_ancestors(model: GraphModel, target: GraphNode) -> List[GraphNode]:
    """All nodes upstream of (and including) ``target``, topologically ordered
    so every node appears after the inputs it depends on."""
    order: List[GraphNode] = []
    seen = set()

    def visit(n: GraphNode):
        if n.id in seen:
            return
        seen.add(n.id)
        for src in model.inputs_of(n):
            visit(src)
        order.append(n)

    visit(target)
    return order


def generate_pseudocode(model: GraphModel, target: GraphNode) -> str:
    """Language-neutral pseudocode for the sub-pipeline ending at ``target``."""
    order = _ordered_ancestors(model, target)
    names = {}
    counts = {}
    lines = [
        "# Auto-generated from the prototyped pipeline (language-neutral pseudocode).",
        "# cv:: calls map 1:1 to OpenCV; custom blocks describe what the node does.",
        "",
    ]
    for gn in order:
        if gn.op is not None and gn.op.id == "export_code" and gn is target:
            continue  # the export node itself is not part of the program
        ins = [names[s.id] for s in model.inputs_of(gn) if s.id in names]
        if gn.is_source:
            base = "img"
        else:
            base = _var_base(gn.op)
        counts[base] = counts.get(base, 0) + 1
        var = f"{base}{counts[base]}"
        names[gn.id] = var
        if gn.is_source:
            lines.append(f'{var} = imread("input.png")   # source image (H x W x C)')
        elif gn.op.id == "export_code":
            lines.append(f"# {var}: Export Code node (introspection only)")
        elif gn.op.id == "save_to_file":
            arg = ins[0] if ins else "?"
            lines.append(f'imwrite("output.png", {arg})   # Save to File')
        else:
            lines.extend(_emit_op(gn.op, var, ins, gn.params))
    result = names.get(target.id)
    upstream = model.inputs_of(target)
    if target.op is not None and target.op.id == "export_code" and upstream:
        result = names.get(upstream[0].id)
    if result:
        lines += ["", f"# pipeline result -> {result}"]
    return "\n".join(lines)


def op_pseudocode(op, params=None) -> str:
    """Single-op pseudocode (for tooltips), using default params unless given."""
    if op is None:
        return ""
    p = dict(op.defaults()) if hasattr(op, "defaults") else {}
    if params:
        p.update(params)
    ins = [f"in{n + 1}" for n in range(len(op.inputs))] or ["in1"]
    return "\n".join(_emit_op(op, "out", ins, p))


def op_cv_calls(op) -> List[str]:
    """The cv:: function names this op's pseudocode uses (e.g. ['cv::GaussianBlur']).
    Used by the sidebar search so you can find a node by the OpenCV call it makes."""
    if op is None:
        return []
    try:
        return _CV_CALL_RE.findall(op_pseudocode(op))
    except Exception:  # noqa: BLE001 — search must never raise
        return []
