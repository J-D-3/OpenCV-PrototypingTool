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
    "kmeans": "clusters", "auto_cluster": "clusters", "reduce_colors": "quant",
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
    "canny": lambda o, i, p: f"{o} = cv::Canny({i[0]}, {p.get('threshold1')}, {p.get('threshold2')}, apertureSize={p.get('aperture')})",
    "sobel": lambda o, i, p: f"{o} = cv::Sobel({i[0]}, CV_64F, dx={p.get('dx')}, dy={p.get('dy')}, ksize={p.get('ksize')})",
    "laplacian": lambda o, i, p: f"{o} = cv::Laplacian({i[0]}, CV_64F, ksize={p.get('ksize')})",
    "morphology": lambda o, i, p: f"{o} = cv::morphologyEx({i[0]}, op={p.get('operation')}, kernel=ones({p.get('kernel_size')}), iterations={p.get('iterations')})",
    # --- threshold ---
    "threshold": lambda o, i, p: f"{o} = cv::threshold({i[0]}, thresh={p.get('threshold_value')}, maxval={p.get('max_value')}, type={p.get('threshold_type')})",
    "adaptive_threshold": lambda o, i, p: f"{o} = cv::adaptiveThreshold({i[0]}, maxValue={p.get('max_value')}, method={p.get('adaptive_method')}, type={p.get('threshold_type')}, blockSize={p.get('block_size')}, C={p.get('c')})",
    # --- geometry ---
    "resize": lambda o, i, p: f"{o} = cv::resize({i[0]}, None, fx={p.get('scale')}, fy={p.get('scale')}, interpolation={p.get('interpolation')})",
    "rotate": lambda o, i, p: [f"M = cv::getRotationMatrix2D(center, angle={p.get('angle')}, scale=1.0)",
                               f"{o} = cv::warpAffine({i[0]}, M, dsize)   # expand={p.get('expand')}"],
    # --- arithmetic ---
    "sum": lambda o, i, p: f"{o} = cv::addWeighted({i[0]}, {p.get('alpha')}, {i[1]}, {1 - float(p.get('alpha', 0.5))}, 0)",
    "and": lambda o, i, p: f"{o} = cv::bitwise_and({i[0]}, {i[1]})",
    "diff": lambda o, i, p: f"{o} = cv::absdiff({i[0]}, {i[1]})",
    "invert": lambda o, i, p: f"{o} = 255 - {i[0]}",
    # --- fourier ---
    "dft": lambda o, i, p: f"{o} = cv::dft(float({i[0]}), flags=DFT_COMPLEX_OUTPUT)",
    "idft": lambda o, i, p: f"{o} = cv::idft({i[0]}, flags=DFT_REAL_OUTPUT | DFT_SCALE)",
    # --- binary connected components ---
    "connected_components": lambda o, i, p: [
        f"# label connected blobs in a binary image ({p.get('connectivity')}-connectivity)",
        f"n, labels = cv::connectedComponents(({i[0]} > 0), connectivity={p.get('connectivity')})",
        f"{o} = [outer contour of (labels == k) for k in 1..n-1]"],
}


def _emit_kmeans(o, i, p):
    return [
        "# cluster pixels with k-means in a chosen feature space; report each",
        "# center as the mean *input-space* color, ordered dark->light (stable).",
        f"feat = features({i[0]}, space={p.get('cluster_space')!r}, lum_weight={p.get('lum_weight')})",
        f"_, labels, _ = cv::kmeans(feat, K={p.get('k')}, criteria, attempts={p.get('attempts')}, KMEANS_PP_CENTERS)",
        f"{o} = {{centers, labels, shape}}   # CLUSTERS payload"]


def _emit_auto_cluster(o, i, p):
    return [
        "# pick K by counting smoothed histogram peaks on one HLS channel, then k-means.",
        f"K = count_peaks({i[0]}, channel={p.get('channel')!r}, smoothing={p.get('smoothing')}, min_prominence={p.get('min_prominence')}, max_k={p.get('max_k')})",
        f"feat = features({i[0]}, space={p.get('cluster_space')!r}, lum_weight={p.get('lum_weight')})",
        f"_, labels, _ = cv::kmeans(feat, K, criteria, attempts, KMEANS_PP_CENTERS)",
        f"{o} = {{centers, labels, shape}}   # CLUSTERS payload"]


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
        "#   delta==0 -> cv::connectedComponents per unique value (exact)",
        "#   delta>0  -> cv::floodFill region-grow (FIXED_RANGE) per seed",
        (f"labels = exact_label({i[0]}, connectivity={p.get('connectivity')})" if exact
         else f"labels = floodfill_label({i[0]}, delta={p.get('delta')}, connectivity={p.get('connectivity')})"),
        f"{o} = [outer contour of (labels == k) for each region k]",
    ]


def _emit_contour_filter(o, i, p):
    return [f"{o} = [c for c in {i[0]} if {p.get('min_area')} <= cv::contourArea(c) <= {p.get('max_area')}]"]


def _emit_find_contours(o, i, p):
    return [f"{o}, hierarchy = cv::findContours({i[0]}, mode={p.get('mode')}, CHAIN_APPROX_SIMPLE)"]


_CODE.update({
    "kmeans": _emit_kmeans,
    "auto_cluster": _emit_auto_cluster,
    "reduce_colors": _emit_reduce_colors,
    "label_regions": _emit_label_regions,
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
