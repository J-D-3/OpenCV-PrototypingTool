"""Pure-Python (Qt-free) OpenCV operation registry.

Each operation is registered exactly once and fully describes a node:
its input/output ports, its parameter schema, the compute function, and
optional hooks for visual inspection. The GUI (sidebar tree, node factory,
and — from Phase 2 — the parameter panel) is generated from this registry,
so adding a new OpenCV function means adding one ``Operation`` here.

This module intentionally has NO Qt dependency: it can be imported and
unit-tested headlessly. Rendering/inspection (Qt) lives in node.py / main.py.

Optional inspection hooks (used from Phase 4 onward):
  * ``render_preview(inputs, output, params) -> np.ndarray`` — produce an image
    representation for ops whose native output is not itself an image
    (e.g. FindContours, drawn back onto the input via cv2.drawContours).
  * ``summary(output, params) -> dict`` — key facts to show in the GUI
    (e.g. {"contours": 42}).
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --- Ports ------------------------------------------------------------------
# A richer type system (ImageBGR/Gray/Binary/Contours/...) arrives in Phase 4.
# For now a port is just a named slot; "image" is the only flowing type.
@dataclass(frozen=True)
class Port:
    name: str
    type: str = "image"


# --- Parameter schema -------------------------------------------------------
# Phase 2 builds the generic parameter panel from these specs. Phase 1 only
# needs the defaults, but the full hints are encoded now to avoid a second pass.
@dataclass
class ParamSpec:
    name: str
    default: Any
    kind: str = "float"          # int | float | bool | choice | enum | str | path
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    odd: bool = False            # value constrained to odd numbers (kernels)
    choices: Optional[list] = None  # list of (label, value) for choice/enum
    label: Optional[str] = None  # display label; defaults to name.title()


@dataclass
class Operation:
    id: str
    label: str
    category: str
    inputs: list
    outputs: list
    params: list
    compute: Callable[[list, dict], Optional[np.ndarray]]
    color: tuple = (120, 120, 120)
    in_label: str = ""
    out_label: str = ""
    render_preview: Optional[Callable[[list, Any, dict], np.ndarray]] = None
    summary: Optional[Callable[[Any, dict], dict]] = None

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}


# ---------------------------------------------------------------------------
# compute helpers
# ---------------------------------------------------------------------------
def _to_gray_if_color(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _align_like(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize/convert/cast b so it matches a (used by AND/Diff)."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if a.ndim != b.ndim:
        if a.ndim == 2 and b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        elif a.ndim == 3 and b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    return b


# ---------------------------------------------------------------------------
# compute functions (mirror the previous per-class _execute_function bodies)
# ---------------------------------------------------------------------------
def _compute_blur(inputs, p):
    try:
        k = int(p["kernel_size"])
        return cv2.blur(inputs[0], (k, k))
    except Exception as e:  # noqa: BLE001  (behaviour-preserving; Phase 3 surfaces errors)
        print(f"Error executing blur: {e}")
        return None


def _compute_threshold(inputs, p):
    try:
        gray = _to_gray_if_color(inputs[0])
        _, result = cv2.threshold(gray, p["threshold_value"], p["max_value"], p["threshold_type"])
        return result
    except Exception as e:
        print(f"Error executing threshold: {e}")
        return None


def _compute_adaptive_threshold(inputs, p):
    try:
        gray = _to_gray_if_color(inputs[0])
        return cv2.adaptiveThreshold(
            gray, p["max_value"], p["adaptive_method"],
            p["threshold_type"], p["block_size"], p["c"],
        )
    except Exception as e:
        print(f"Error executing adaptive_threshold: {e}")
        return None


def _compute_to_grayscale(inputs, p):
    try:
        img = inputs[0]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    except Exception as e:
        print(f"Error executing to_grayscale: {e}")
        return None


def _compute_to_bgr(inputs, p):
    try:
        img = inputs[0]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    except Exception as e:
        print(f"Error executing to_bgr: {e}")
        return None


def _compute_sum(inputs, p):
    try:
        a, b = inputs
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        alpha = float(p.get("alpha", 0.5))
        return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0)
    except Exception as e:
        print(f"Error executing sum: {e}")
        return None


def _compute_and(inputs, p):
    try:
        a, b = inputs
        b = _align_like(a, b)
        return cv2.bitwise_and(a, b)
    except Exception as e:
        print(f"Error executing AND: {e}")
        return None


def _compute_diff(inputs, p):
    try:
        a, b = inputs
        b = _align_like(a, b)
        return cv2.subtract(a, b)
    except Exception as e:
        print(f"Error executing diff: {e}")
        return None


def _compute_mser(inputs, p):
    try:
        input_image = inputs[0]
        gray = _to_gray_if_color(input_image) if input_image.ndim == 3 else input_image.copy()

        mser = cv2.MSER_create(
            delta=p["delta"],
            min_area=p["min_area"],
            max_area=p["max_area"],
            max_variation=p["max_variation"],
            min_diversity=p["min_diversity"],
            max_evolution=p["max_evolution"],
            area_threshold=p["area_threshold"],
            min_margin=p["min_margin"],
            edge_blur_size=p["edge_blur_size"],
        )
        regions, _ = mser.detectRegions(gray)

        if input_image.ndim == 3:
            output = input_image.copy()
        else:
            output = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

        for region in regions:
            if len(region) < 3:
                continue
            region_points = np.array(region, dtype=np.int32)

            sample_size = min(100, len(region_points))
            if sample_size < len(region_points):
                sample_indices = np.random.choice(len(region_points), sample_size, replace=False)
                sample_pixels = region_points[sample_indices]
            else:
                sample_pixels = region_points

            sample_colors = []
            for pixel in sample_pixels:
                x, y = pixel[0], pixel[1]
                if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
                    sample_colors.append(input_image[y, x])
            if not sample_colors:
                continue

            if input_image.ndim == 3:
                mean_color = np.mean(np.array(sample_colors), axis=0)
            else:
                m = np.mean(np.array(sample_colors))
                mean_color = np.array([m, m, m])
            mean_color = tuple(int(c) for c in mean_color)

            for pixel in region_points:
                x, y = pixel[0], pixel[1]
                if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
                    output[y, x] = mean_color

        return output
    except Exception as e:
        print(f"Error executing MSER: {e}")
        return None


# save_to_file is genuinely special: it has a side effect (writing a file),
# carries per-node state (timestamp/index), and must be suppressed during
# preview/propagation. Its behaviour lives in node.SaveToFileNode, so its
# registry entry has no compute function (the factory routes it specially).
def _compute_noop(inputs, p):
    return inputs[0] if inputs else None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_THRESH_TYPES = [
    ("Binary", cv2.THRESH_BINARY),
    ("Binary Inv", cv2.THRESH_BINARY_INV),
    ("Trunc", cv2.THRESH_TRUNC),
    ("To Zero", cv2.THRESH_TOZERO),
    ("To Zero Inv", cv2.THRESH_TOZERO_INV),
]
_ADAPTIVE_THRESH_TYPES = [
    ("Binary", cv2.THRESH_BINARY),
    ("Binary Inv", cv2.THRESH_BINARY_INV),
]
_ADAPTIVE_METHODS = [
    ("Mean C", cv2.ADAPTIVE_THRESH_MEAN_C),
    ("Gaussian C", cv2.ADAPTIVE_THRESH_GAUSSIAN_C),
]

# Registration order also determines the sidebar order within each category.
OPS: list = [
    Operation(
        id="save_to_file", label="Save to File", category="Input/Output",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("filename", "", kind="path"),
            ParamSpec("use_custom", False, kind="bool", label="Use custom filename"),
        ],
        compute=_compute_noop, color=(76, 175, 80),
        in_label="Mat (Any)", out_label="File",
    ),
    Operation(
        id="to_grayscale", label="To Grayscale", category="Conversions",
        inputs=[Port("in")], outputs=[Port("out")], params=[],
        compute=_compute_to_grayscale, color=(96, 96, 96),
        in_label="Mat (BGR)", out_label="Mat (Gray)",
    ),
    Operation(
        id="to_bgr", label="To BGR", category="Conversions",
        inputs=[Port("in")], outputs=[Port("out")], params=[],
        compute=_compute_to_bgr, color=(33, 150, 243),
        in_label="Mat (Gray)", out_label="Mat (BGR)",
    ),
    Operation(
        id="blur", label="Blur", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("kernel_size", 15, kind="int", min=1, max=101, step=2, odd=True,
                          label="Kernel Size")],
        compute=_compute_blur, color=(156, 39, 176),
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="threshold", label="Threshold", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("threshold_value", 127, kind="int", min=0, max=255, label="Threshold Value"),
            ParamSpec("max_value", 255, kind="int", min=0, max=255, label="Max Value"),
            ParamSpec("threshold_type", cv2.THRESH_BINARY, kind="enum",
                      choices=_THRESH_TYPES, label="Threshold Type"),
        ],
        compute=_compute_threshold, color=(255, 152, 0),
        in_label="Mat (Gray)", out_label="Mat (Binary/Gray)",
    ),
    Operation(
        id="adaptive_threshold", label="Adaptive Threshold", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("max_value", 255, kind="int", min=1, max=255, label="Max Value"),
            ParamSpec("adaptive_method", cv2.ADAPTIVE_THRESH_MEAN_C, kind="enum",
                      choices=_ADAPTIVE_METHODS, label="Adaptive Method"),
            ParamSpec("threshold_type", cv2.THRESH_BINARY, kind="enum",
                      choices=_ADAPTIVE_THRESH_TYPES, label="Threshold Type"),
            ParamSpec("block_size", 11, kind="int", min=3, max=51, step=2, odd=True,
                      label="Block Size"),
            ParamSpec("c", 2, kind="int", min=-10, max=10, label="C Value"),
        ],
        compute=_compute_adaptive_threshold, color=(255, 152, 0),
        in_label="Mat (Gray)", out_label="Mat (Binary)",
    ),
    Operation(
        id="mser", label="MSER", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("delta", 5, kind="int", min=1, max=20, label="Delta"),
            ParamSpec("min_area", 60, kind="int", min=10, max=1000, label="Min Area"),
            ParamSpec("max_area", 14400, kind="int", min=1000, max=50000, label="Max Area"),
            ParamSpec("max_variation", 0.25, kind="float", min=0.0, max=1.0, step=0.01,
                      label="Max Variation"),
            ParamSpec("min_diversity", 0.2, kind="float", min=0.0, max=1.0, step=0.01,
                      label="Min Diversity"),
            ParamSpec("max_evolution", 200, kind="int"),
            ParamSpec("area_threshold", 1.01, kind="float"),
            ParamSpec("min_margin", 0.003, kind="float"),
            ParamSpec("edge_blur_size", 5, kind="int"),
        ],
        compute=_compute_mser, color=(34, 139, 34),
        in_label="Mat (Gray)", out_label="Mat (BGR)",
    ),
    Operation(
        id="sum", label="Sum", category="Arithmetic Operations",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")],
        params=[ParamSpec("alpha", 0.5, kind="float", min=0.0, max=1.0, step=0.01,
                          label="Alpha (Weight)")],
        compute=_compute_sum, color=(128, 0, 128),
        in_label="Mat (BGR/Gray) + Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="and", label="AND", category="Arithmetic Operations",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")], params=[],
        compute=_compute_and, color=(0, 0, 139),
        in_label="Mat (BGR/Gray) & Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="diff", label="Diff", category="Arithmetic Operations",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")], params=[],
        compute=_compute_diff, color=(220, 20, 60),
        in_label="Mat (BGR/Gray) - Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
]

# Categories shown in the sidebar, in order. Geometry/Fourier are intentionally
# present-but-empty placeholders (populated in Phase 6).
CATEGORY_ORDER = [
    "Input/Output",
    "Conversions",
    "Geometry",
    "Arithmetic Operations",
    "Local Operations",
    "Fourier",
]

REGISTRY = {op.id: op for op in OPS}
by_label = {op.label: op for op in OPS}

ops_by_category: dict = {cat: [] for cat in CATEGORY_ORDER}
for _op in OPS:
    ops_by_category.setdefault(_op.category, []).append(_op)
