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

from core import datatypes
from core.batch import Batch


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
    show: bool = True            # render a control in the parameter panel?


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
    # Color-space tracking (see core.engine): out_space is "bgr"|"gray"|"hls"|
    # "binary" (fixed), "passthrough" (= first input's space), or "auto" (infer
    # from the output array). space_aware ops receive the input space as a 3rd
    # compute() argument.
    out_space: str = "auto"
    space_aware: bool = False
    # variadic: accepts arbitrarily many inputs (the single declared input port
    # is a template). raw: the engine passes inputs as-is (no per-element batch
    # fan-out) so the op can assemble/consume batches itself.
    variadic: bool = False
    raw: bool = False

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


# --- color-space conversions (space-aware: they receive the input space) -----
# A single op per target space delegates to the right cv2 conversion based on
# the input's tracked color space (the engine passes it in). One source array
# of 3 channels is ambiguous (BGR vs HLS) on its own, which is why the space
# tag matters.
def _is_single_channel(img):
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


def _as_bgr(img, space):
    """Return a 3-channel BGR image given its current color space."""
    if _is_single_channel(img):
        gray = img if img.ndim == 2 else img[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if space == "hls":
        return cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    return img  # already BGR (or unknown 3-channel, assumed BGR)


def _compute_to_grayscale(inputs, p, in_space):
    try:
        img = inputs[0]
        if _is_single_channel(img):
            return img if img.ndim == 2 else img[:, :, 0]
        return cv2.cvtColor(_as_bgr(img, in_space), cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error executing to_grayscale: {e}")
        return None


def _compute_to_bgr(inputs, p, in_space):
    try:
        return _as_bgr(inputs[0], in_space)
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


def _compute_to_hls(inputs, p, in_space):
    try:
        img = inputs[0]
        if not _is_single_channel(img) and in_space == "hls":
            return img  # already HLS
        return cv2.cvtColor(_as_bgr(img, in_space), cv2.COLOR_BGR2HLS)
    except Exception as e:
        print(f"Error executing to_hls: {e}")
        return None


def _compute_kmeans(inputs, p):
    """Cluster an image's pixels with k-means. Output is a clusters payload
    (centers + per-pixel labels + shape) — not an image — consumed downstream
    by Reduce Colors."""
    try:
        img = inputs[0]
        channels = img.shape[2] if img.ndim == 3 else 1
        data = img.reshape(-1, channels).astype(np.float32)
        k = max(1, int(p["k"]))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = int(p.get("attempts", 3))
        _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts,
                                        cv2.KMEANS_RANDOM_CENTERS)
        return {"centers": centers, "labels": labels.flatten(),
                "shape": img.shape, "k": k}
    except Exception as e:
        print(f"Error executing kmeans: {e}")
        return None


def _render_kmeans(inputs, output, p):
    """Inspector preview: a horizontal swatch of the cluster colors."""
    if not isinstance(output, dict):
        return None
    centers = np.clip(output["centers"], 0, 255).astype(np.uint8)
    k = len(centers)
    if k == 0:
        return None
    cell_w, cell_h = 60, 80
    swatch = np.zeros((cell_h, cell_w * k, 3), np.uint8)
    for i, c in enumerate(centers):
        color = tuple(int(v) for v in c) if len(c) == 3 else (int(c[0]),) * 3
        swatch[:, i * cell_w:(i + 1) * cell_w] = color
    return swatch


def _summary_kmeans(output, p):
    if not isinstance(output, dict):
        return {}
    labels = output.get("labels")
    info = {"clusters": int(output.get("k", 0))}
    if labels is not None and len(labels):
        counts = np.bincount(labels, minlength=output.get("k", 0))
        biggest = int(np.argmax(counts))
        info["largest cluster"] = f"#{biggest} ({100 * counts[biggest] // len(labels)}%)"
    return info


def _compute_reduce_colors(inputs, p):
    """Rebuild a quantized image from a clusters payload (centers[labels])."""
    try:
        clusters = inputs[0]
        if not isinstance(clusters, dict):
            return None
        centers = np.clip(clusters["centers"], 0, 255).astype(np.uint8)
        labels = clusters["labels"]
        return centers[labels].reshape(clusters["shape"])
    except Exception as e:
        print(f"Error executing reduce_colors: {e}")
        return None


def _compute_resize(inputs, p):
    try:
        img = inputs[0]
        scale = float(p["scale"])
        if scale <= 0 or scale == 1.0:
            return img
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=int(p["interpolation"]))
    except Exception as e:
        print(f"Error executing resize: {e}")
        return None


def _compute_rotate(inputs, p):
    try:
        img = inputs[0]
        angle = float(p["angle"])
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        if p.get("expand", False):
            cos, sin = abs(m[0, 0]), abs(m[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            m[0, 2] += nw / 2.0 - cx
            m[1, 2] += nh / 2.0 - cy
            return cv2.warpAffine(img, m, (nw, nh))
        return cv2.warpAffine(img, m, (w, h))
    except Exception as e:
        print(f"Error executing rotate: {e}")
        return None


def _to_gray_u8(img):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray if gray.dtype == np.uint8 else gray.astype(np.uint8)


def _contour_depths(hierarchy, n):
    """Nesting depth of each contour from the OpenCV hierarchy (col 3 = parent).
    All zeros when there is no nesting (e.g. RETR_EXTERNAL)."""
    if hierarchy is None or n == 0:
        return [0] * n
    h = hierarchy[0]
    depths = []
    for i in range(n):
        d, parent, seen = 0, int(h[i][3]), set()
        while parent != -1 and parent not in seen:
            seen.add(parent)
            d += 1
            parent = int(h[parent][3])
        depths.append(d)
    return depths


def _compute_find_contours(inputs, p):
    """Find contours in a (binary) image. Output is a CONTOURS payload carrying
    the contours, a stable per-contour id, each contour's nesting depth, and a
    BGR background to draw them on for inspection."""
    try:
        img = inputs[0]
        gray = _to_gray_u8(img)
        contours, hierarchy = cv2.findContours(gray, int(p["mode"]), cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        return {"contours": contours, "hierarchy": hierarchy,
                "ids": list(range(len(contours))),
                "depths": _contour_depths(hierarchy, len(contours)),
                "shape": img.shape,
                "background": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)}
    except Exception as e:
        print(f"Error executing find_contours: {e}")
        return None


# Cycle R, G, B, C, M, Y so adjacent contours are easy to tell apart (BGR order).
_CONTOUR_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]


def _draw_contours_preview(payload, filled=False):
    bg = payload.get("background")
    if bg is None:
        return None
    contours = payload.get("contours", [])
    ids = payload.get("ids", list(range(len(contours))))
    depths = payload.get("depths", [0] * len(contours))
    out = bg.copy()
    # Draw order: outermost (smallest nesting depth) first, larger first within a
    # depth — so filled inner contours stay visible. With no hierarchy (e.g.
    # RETR_EXTERNAL, all depths 0) this is just largest-first.
    order = sorted(range(len(contours)),
                   key=lambda i: (depths[i], -cv2.contourArea(contours[i])))
    thickness = cv2.FILLED if filled else 1
    for i in order:
        # Colour is bound to the *stable id*, not the draw order, so a contour
        # keeps its colour as others are filtered in/out.
        cv2.drawContours(out, contours, i, _CONTOUR_COLORS[ids[i] % len(_CONTOUR_COLORS)], thickness)
    return out


def _render_find_contours(inputs, output, p):
    if not isinstance(output, dict):
        return None
    return _draw_contours_preview(output, filled=bool(p.get("filled", False)))


def _summary_find_contours(output, p):
    if not isinstance(output, dict):
        return {}
    return {"contours": len(output.get("contours", []))}


def _compute_filter_contours(inputs, p):
    """Keep only contours whose area is within [min_area, max_area], preserving
    each survivor's stable id and depth so colours stay put across filtering."""
    try:
        payload = inputs[0]
        if not isinstance(payload, dict):
            return None
        contours = payload.get("contours", [])
        ids = payload.get("ids", list(range(len(contours))))
        depths = payload.get("depths", [0] * len(contours))
        lo, hi = float(p["min_area"]), float(p["max_area"])
        keep = [i for i, c in enumerate(contours) if lo <= cv2.contourArea(c) <= hi]
        out = dict(payload)
        out["contours"] = [contours[i] for i in keep]
        out["ids"] = [ids[i] for i in keep]
        out["depths"] = [depths[i] for i in keep]
        out["_total"] = len(contours)
        return out
    except Exception as e:
        print(f"Error executing contour_filter: {e}")
        return None


def _render_filter_contours(inputs, output, p):
    if not isinstance(output, dict):
        return None
    return _draw_contours_preview(output, filled=bool(p.get("filled", False)))


def _summary_filter_contours(output, p):
    if not isinstance(output, dict):
        return {}
    return {"kept": len(output.get("contours", [])), "of": int(output.get("_total", 0))}


def _compute_dft(inputs, p):
    """Forward DFT of a grayscale image. Output is a SPECTRUM payload holding the
    full complex transform so the inverse DFT can reconstruct the image."""
    try:
        gray = _to_gray_u8(inputs[0]).astype(np.float32)
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        return {"dft": dft, "shape": gray.shape}
    except Exception as e:
        print(f"Error executing dft: {e}")
        return None


def _render_dft(inputs, output, p):
    """Inspector preview: the log-magnitude spectrum, low frequencies centered."""
    if not isinstance(output, dict):
        return None
    dft = output["dft"]
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    return np.fft.fftshift(np.log1p(mag))   # float; cv_to_qimage normalizes for display


def _summary_dft(output, p):
    if not isinstance(output, dict):
        return {}
    h, w = output.get("shape", (0, 0))[:2]
    return {"size": f"{w}x{h}"}


def _compute_idft(inputs, p):
    """Inverse DFT back to a real image. DFT_SCALE makes idft(dft(x)) == x."""
    try:
        spec = inputs[0]
        if not isinstance(spec, dict):
            return None
        return cv2.idft(spec["dft"], flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    except Exception as e:
        print(f"Error executing idft: {e}")
        return None


def _compute_gaussian_blur(inputs, p):
    try:
        k = int(p["kernel_size"])
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(inputs[0], (k, k), float(p.get("sigma", 0.0)))
    except Exception as e:
        print(f"Error executing gaussian_blur: {e}")
        return None


def _compute_morphology(inputs, p):
    try:
        ksize = max(1, int(p["kernel_size"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.morphologyEx(inputs[0], int(p["operation"]), kernel,
                                iterations=max(1, int(p.get("iterations", 1))))
    except Exception as e:
        print(f"Error executing morphology: {e}")
        return None


def _compute_canny(inputs, p):
    try:
        gray = _to_gray_u8(inputs[0])
        ap = int(p.get("aperture", 3))
        return cv2.Canny(gray, float(p["threshold1"]), float(p["threshold2"]), apertureSize=ap)
    except Exception as e:
        print(f"Error executing canny: {e}")
        return None


def _compute_sobel(inputs, p):
    try:
        gray = _to_gray_u8(inputs[0])
        dx, dy = int(p["dx"]), int(p["dy"])
        if dx == 0 and dy == 0:
            dx = 1
        k = int(p["ksize"])
        if k % 2 == 0:
            k += 1
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=k))
    except Exception as e:
        print(f"Error executing sobel: {e}")
        return None


def _compute_laplacian(inputs, p):
    try:
        gray = _to_gray_u8(inputs[0])
        k = int(p["ksize"])
        if k % 2 == 0:
            k += 1
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=k))
    except Exception as e:
        print(f"Error executing laplacian: {e}")
        return None


def _compute_normalize(inputs, p):
    """Histogram normalization. 'stretch' = min-max contrast stretch to 0..255;
    'equalize'/'clahe' redistribute intensities (on luminance for color images,
    so hues are preserved)."""
    try:
        img = inputs[0]
        mode = p.get("mode", "stretch")
        if mode == "stretch":
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        work = img if img.dtype == np.uint8 else \
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        apply = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply
                 if mode == "clahe" else cv2.equalizeHist)
        if work.ndim == 2:
            return apply(work)
        ycc = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb)
        ycc[:, :, 0] = apply(ycc[:, :, 0])
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    except Exception as e:
        print(f"Error executing normalize: {e}")
        return None


def _compute_invert(inputs, p):
    try:
        return cv2.bitwise_not(inputs[0])
    except Exception as e:
        print(f"Error executing invert: {e}")
        return None


def _compute_local_hdr(inputs, p):
    """Local (adaptive) histogram normalization with a smooth Gaussian window.

    Subtract the Gaussian local mean and divide by the Gaussian local std, then
    re-amplify — this equalizes local contrast (flat regions are boosted, busy
    ones damped) for an HDR-like look, without CLAHE's tile artifacts. Color
    images are processed on luminance so hues are preserved.
    """
    try:
        img = inputs[0]
        sigma = max(1.0, float(p["radius"]))
        amplitude = float(p["amplitude"])
        strength = float(p["strength"])
        color = img.ndim == 3 and img.shape[2] == 3
        if color:
            ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y = ycc[:, :, 0].astype(np.float32)
        else:
            y = (img if img.ndim == 2 else img[:, :, 0]).astype(np.float32)

        mean = cv2.GaussianBlur(y, (0, 0), sigma)
        detail = y - mean
        local_std = np.sqrt(cv2.GaussianBlur(detail * detail, (0, 0), sigma))
        enhanced = mean + (detail / (local_std + 1.0)) * amplitude
        out = np.clip((1.0 - strength) * y + strength * enhanced, 0, 255).astype(np.uint8)

        if color:
            ycc[:, :, 0] = out
            return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        return out
    except Exception as e:
        print(f"Error executing local_hdr: {e}")
        return None


def _compute_histogram(inputs, p):
    """Per-channel intensity histogram. Output is a HISTOGRAM payload."""
    try:
        img = inputs[0]
        if img.ndim == 2:
            hists = [cv2.calcHist([img], [0], None, [256], [0, 256])]
        else:
            hists = [cv2.calcHist([img], [c], None, [256], [0, 256])
                     for c in range(img.shape[2])]
        return {"hist": hists, "channels": len(hists)}
    except Exception as e:
        print(f"Error executing histogram: {e}")
        return None


def _render_histogram(inputs, output, p):
    """Inspector preview: draw the histogram curve(s)."""
    if not isinstance(output, dict):
        return None
    hists = output["hist"]
    h, w = 220, 256
    canvas = np.full((h, w, 3), 255, np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] if len(hists) == 3 else [(0, 0, 0)]
    for hist, color in zip(hists, colors):
        norm = cv2.normalize(hist, None, 0, h - 1, cv2.NORM_MINMAX).flatten()
        for x in range(1, w):
            cv2.line(canvas, (x - 1, h - 1 - int(norm[x - 1])),
                     (x, h - 1 - int(norm[x])), color, 1)
    return canvas


def _summary_histogram(output, p):
    if not isinstance(output, dict):
        return {}
    return {"channels": int(output.get("channels", 0))}


# save_to_file is genuinely special: it has a side effect (writing a file),
# carries per-node state (timestamp/index), and must be suppressed during
# preview/propagation. Its behaviour lives in node.SaveToFileNode, so its
# registry entry has no compute function (the factory routes it specially).
def _compute_noop(inputs, p):
    return inputs[0] if inputs else None


def _to_bgr3(im):
    """Normalize an image to 3-channel BGR (so a batch is homogeneous)."""
    if im is None:
        return None
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.ndim == 3 and im.shape[2] == 1:
        return cv2.cvtColor(im[:, :, 0], cv2.COLOR_GRAY2BGR)
    if im.ndim == 3 and im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im


def _compute_create_batch(inputs, p):
    """Assemble a Batch from arbitrarily many image inputs (raw + variadic).

    Inputs that are themselves batches are flattened in. Every element is
    normalized to 3-channel BGR so the resulting batch is homogeneous.
    """
    items = []
    for inp in inputs:
        if isinstance(inp, Batch):
            items.extend(inp.items)
        elif inp is not None:
            items.append(inp)
    out = [_to_bgr3(im) for im in items if im is not None]
    return Batch(out) if out else None


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
_RETR_MODES = [
    ("External", cv2.RETR_EXTERNAL),
    ("List", cv2.RETR_LIST),
    ("Tree", cv2.RETR_TREE),
    ("Connected (CComp)", cv2.RETR_CCOMP),
]
_INTERP_MODES = [
    ("Area (shrink)", cv2.INTER_AREA),
    ("Linear", cv2.INTER_LINEAR),
    ("Cubic", cv2.INTER_CUBIC),
    ("Nearest", cv2.INTER_NEAREST),
    ("Lanczos4", cv2.INTER_LANCZOS4),
]
_NORMALIZE_MODES = [
    ("Stretch (min-max)", "stretch"),
    ("Equalize", "equalize"),
    ("CLAHE", "clahe"),
]
_MORPH_OPS = [
    ("Erode", cv2.MORPH_ERODE),
    ("Dilate", cv2.MORPH_DILATE),
    ("Open", cv2.MORPH_OPEN),
    ("Close", cv2.MORPH_CLOSE),
    ("Gradient", cv2.MORPH_GRADIENT),
    ("Top Hat", cv2.MORPH_TOPHAT),
    ("Black Hat", cv2.MORPH_BLACKHAT),
]

# Registration order also determines the sidebar order within each category.
OPS: list = [
    Operation(
        id="save_to_file", label="Save to File", category="Input/Output",
        inputs=[Port("in", datatypes.ANY)], outputs=[Port("out")],
        params=[
            ParamSpec("filename", "", kind="path"),
            ParamSpec("use_custom", False, kind="bool", label="Use custom filename"),
        ],
        compute=_compute_noop, color=(76, 175, 80), out_space="passthrough",
        in_label="Mat (Any)", out_label="File",
    ),
    Operation(
        id="create_batch", label="Create Batch", category="Input/Output",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[], compute=_compute_create_batch, color=(121, 134, 203),
        in_label="Mat (any) ×N", out_label="Batch (BGR)",
        out_space="bgr", variadic=True, raw=True,
    ),
    Operation(
        id="to_grayscale", label="To Grayscale", category="Conversions",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE_GRAY)],
        params=[], compute=_compute_to_grayscale, color=(96, 96, 96),
        in_label="Mat (any)", out_label="Mat (Gray)",
        out_space="gray", space_aware=True,
    ),
    Operation(
        id="to_bgr", label="To BGR", category="Conversions",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE_BGR)],
        params=[], compute=_compute_to_bgr, color=(33, 150, 243),
        in_label="Mat (any)", out_label="Mat (BGR)",
        out_space="bgr", space_aware=True,
    ),
    Operation(
        id="blur", label="Blur", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("kernel_size", 15, kind="int", min=1, max=101, step=2, odd=True,
                          label="Kernel Size")],
        compute=_compute_blur, color=(156, 39, 176), out_space="passthrough",
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
        id="gaussian_blur", label="Gaussian Blur", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("kernel_size", 5, kind="int", min=1, max=51, step=2, odd=True,
                      label="Kernel Size"),
            ParamSpec("sigma", 0.0, kind="float", min=0.0, max=10.0, step=0.5, label="Sigma"),
        ],
        compute=_compute_gaussian_blur, color=(103, 58, 183), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="morphology", label="Morphology", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("operation", cv2.MORPH_ERODE, kind="enum",
                      choices=_MORPH_OPS, label="Operation"),
            ParamSpec("kernel_size", 3, kind="int", min=1, max=31, label="Kernel Size"),
            ParamSpec("iterations", 1, kind="int", min=1, max=10, label="Iterations"),
        ],
        compute=_compute_morphology, color=(96, 125, 139), out_space="passthrough",
        in_label="Mat (Binary/Gray)", out_label="Mat (Binary/Gray)",
    ),
    Operation(
        id="canny", label="Canny Edges", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out", datatypes.IMAGE_BINARY)],
        params=[
            ParamSpec("threshold1", 100, kind="int", min=0, max=500, label="Threshold 1"),
            ParamSpec("threshold2", 200, kind="int", min=0, max=500, label="Threshold 2"),
            ParamSpec("aperture", 3, kind="int", show=False),
        ],
        compute=_compute_canny, color=(255, 193, 7),
        in_label="Mat (Gray)", out_label="Mat (Binary edges)",
    ),
    Operation(
        id="sobel", label="Sobel", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("dx", 1, kind="int", min=0, max=2, label="dx (x order)"),
            ParamSpec("dy", 0, kind="int", min=0, max=2, label="dy (y order)"),
            ParamSpec("ksize", 3, kind="int", min=1, max=7, step=2, odd=True, label="Kernel Size"),
        ],
        compute=_compute_sobel, color=(255, 193, 7),
        in_label="Mat (Gray)", out_label="Mat (Gray gradient)",
    ),
    Operation(
        id="laplacian", label="Laplacian", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("ksize", 3, kind="int", min=1, max=31, step=2, odd=True,
                          label="Kernel Size")],
        compute=_compute_laplacian, color=(255, 193, 7),
        in_label="Mat (Gray)", out_label="Mat (Gray gradient)",
    ),
    Operation(
        id="normalize", label="Normalize", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("mode", "stretch", kind="choice",
                          choices=_NORMALIZE_MODES, label="Mode")],
        compute=_compute_normalize, color=(0, 121, 107), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="invert", label="Invert", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")], params=[],
        compute=_compute_invert, color=(69, 90, 100), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="local_hdr", label="Local HDR", category="Local Operations",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("radius", 25, kind="int", min=2, max=120, label="Radius"),
            ParamSpec("amplitude", 35, kind="int", min=5, max=100, label="Detail strength"),
            ParamSpec("strength", 1.0, kind="float", min=0.0, max=1.0, step=0.05, label="Strength"),
        ],
        compute=_compute_local_hdr, color=(255, 138, 0), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
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
            ParamSpec("max_evolution", 200, kind="int", show=False),
            ParamSpec("area_threshold", 1.01, kind="float", show=False),
            ParamSpec("min_margin", 0.003, kind="float", show=False),
            ParamSpec("edge_blur_size", 5, kind="int", show=False),
        ],
        compute=_compute_mser, color=(34, 139, 34),
        in_label="Mat (Gray)", out_label="Mat (BGR)",
    ),
    Operation(
        id="sum", label="Sum", category="Arithmetic Operations",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")],
        params=[ParamSpec("alpha", 0.5, kind="float", min=0.0, max=1.0, step=0.01,
                          label="Alpha (Weight)")],
        compute=_compute_sum, color=(128, 0, 128), out_space="passthrough",
        in_label="Mat (BGR/Gray) + Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="and", label="AND", category="Arithmetic Operations",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")], params=[],
        compute=_compute_and, color=(0, 0, 139), out_space="passthrough",
        in_label="Mat (BGR/Gray) & Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="diff", label="Diff", category="Arithmetic Operations",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")], params=[],
        compute=_compute_diff, color=(220, 20, 60), out_space="passthrough",
        in_label="Mat (BGR/Gray) - Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    # --- Color & Clustering ------------------------------------------------
    Operation(
        id="to_hls", label="To HSL", category="Conversions",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.IMAGE)], params=[],
        compute=_compute_to_hls, color=(255, 87, 34),
        in_label="Mat (any)", out_label="Mat (HLS)",
        out_space="hls", space_aware=True,
    ),
    Operation(
        id="kmeans", label="K-Means Cluster", category="Color & Clustering",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CLUSTERS)],
        params=[
            ParamSpec("k", 6, kind="int", min=2, max=16, label="Clusters (k)"),
            ParamSpec("attempts", 3, kind="int", min=1, max=10, show=False),
        ],
        compute=_compute_kmeans, color=(0, 150, 136), out_space="passthrough",
        in_label="Mat (any)", out_label="Clusters",
        render_preview=_render_kmeans, summary=_summary_kmeans,
    ),
    Operation(
        id="reduce_colors", label="Reduce Colors", category="Color & Clustering",
        inputs=[Port("in", datatypes.CLUSTERS)],
        outputs=[Port("out", datatypes.IMAGE)], params=[],
        compute=_compute_reduce_colors, color=(0, 150, 136), out_space="passthrough",
        in_label="Clusters", out_label="Mat (quantized)",
    ),
    # --- Geometry ----------------------------------------------------------
    Operation(
        id="resize", label="Resize", category="Geometry",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[
            ParamSpec("scale", 0.5, kind="float", min=0.1, max=4.0, step=0.05, label="Scale"),
            ParamSpec("interpolation", cv2.INTER_AREA, kind="enum",
                      choices=_INTERP_MODES, label="Interpolation"),
        ],
        compute=_compute_resize, color=(63, 81, 181), out_space="passthrough",
        in_label="Mat (any)", out_label="Mat (any)",
    ),
    Operation(
        id="rotate", label="Rotate", category="Geometry",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[
            ParamSpec("angle", 0, kind="int", min=-180, max=180, label="Angle (deg)"),
            ParamSpec("expand", False, kind="bool", label="Expand to fit"),
        ],
        compute=_compute_rotate, color=(63, 81, 181), out_space="passthrough",
        in_label="Mat (any)", out_label="Mat (any)",
    ),
    # --- Contours ----------------------------------------------------------
    Operation(
        id="find_contours", label="Find Contours", category="Contours",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[
            ParamSpec("mode", cv2.RETR_EXTERNAL, kind="enum",
                      choices=_RETR_MODES, label="Retrieval Mode"),
            ParamSpec("filled", False, kind="bool", label="Draw filled"),
        ],
        compute=_compute_find_contours, color=(233, 30, 99),
        in_label="Mat (Binary)", out_label="Contours",
        render_preview=_render_find_contours, summary=_summary_find_contours,
    ),
    Operation(
        id="contour_filter", label="Filter Contours", category="Contours",
        inputs=[Port("in", datatypes.CONTOURS)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[
            ParamSpec("min_area", 50, kind="int", min=0, max=20000, label="Min Area"),
            ParamSpec("max_area", 100000, kind="int", min=0, max=1000000, label="Max Area"),
            ParamSpec("filled", False, kind="bool", label="Draw filled"),
        ],
        compute=_compute_filter_contours, color=(233, 30, 99),
        in_label="Contours", out_label="Contours",
        render_preview=_render_filter_contours, summary=_summary_filter_contours,
    ),
    # --- Fourier -----------------------------------------------------------
    Operation(
        id="dft", label="DFT", category="Fourier",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.SPECTRUM)], params=[],
        compute=_compute_dft, color=(121, 85, 72),
        in_label="Mat (Gray)", out_label="Spectrum",
        render_preview=_render_dft, summary=_summary_dft,
    ),
    Operation(
        id="idft", label="Inverse DFT", category="Fourier",
        inputs=[Port("in", datatypes.SPECTRUM)],
        outputs=[Port("out", datatypes.IMAGE_FLOAT)], params=[],
        compute=_compute_idft, color=(121, 85, 72),
        in_label="Spectrum", out_label="Mat (Float)",
    ),
    # --- Analysis ----------------------------------------------------------
    Operation(
        id="histogram", label="Histogram", category="Analysis",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.HISTOGRAM)], params=[],
        compute=_compute_histogram, color=(0, 188, 212),
        in_label="Mat (any)", out_label="Histogram",
        render_preview=_render_histogram, summary=_summary_histogram,
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
    "Analysis",
    "Color & Clustering",
    "Contours",
    "Fourier",
]

REGISTRY = {op.id: op for op in OPS}
by_label = {op.label: op for op in OPS}

ops_by_category: dict = {cat: [] for cat in CATEGORY_ORDER}
for _op in OPS:
    ops_by_category.setdefault(_op.category, []).append(_op)
