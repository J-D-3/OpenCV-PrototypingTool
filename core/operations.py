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

import threading
from collections import defaultdict

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core import datatypes
from core import optics_backend
from core.batch import Batch

# cv2.setRNGSeed sets *global* state, so k-means must be atomic with respect to
# it. The engine maps batch elements across threads; this lock keeps parallel
# k-means calls from corrupting each other's seeded run (determinism).
_KMEANS_LOCK = threading.Lock()

# The native `_optics` extension (Density Cluster) is NOT thread-safe: running
# cluster_image concurrently across batch elements crashes the process (access
# violation). The engine fans batches across threads, so serialize every optics
# call through this lock. (Verified: 4 concurrent cluster_image calls segfault.)
_OPTICS_LOCK = threading.Lock()


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
    log: bool = False            # int slider: logarithmic response (fine at the
                                 # low end) — for wide-range values like areas
    live: bool = False           # slider recomputes on every drag step (not just
                                 # on release) — for cheap, see-it-live params
    help: str = ""               # one-line "how it affects the result" blurb,
                                 # shown under the name in the control's tooltip
    enabled_if: Optional[tuple] = None
    # Gray this control out unless another param has a given value. Form:
    # ``("other_param", value)`` (equality) or ``("other_param", (v1, v2, ...))``
    # (membership); or a **list** of such tuples, all of which must hold (AND) —
    # e.g. ``[("k_method", "peaks"), ("channel", 0)]``. The control still exists
    # and persists; it's only disabled in the panel when the condition is unmet —
    # so mode-specific params (e.g. Auto Cluster's peak-detection knobs in 'elbow'
    # mode) read as inactive.


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
    # Human description for the Function-info tooltip / code export. If empty,
    # callers fall back to the first paragraph of ``compute.__doc__``.
    description: str = ""
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
    # The render_preview is a synthetic chart (histogram / cluster diagnostics),
    # not a sampleable image — so the inspector hides its per-channel histogram
    # (running a histogram over a plotted graph is meaningless).
    preview_is_chart: bool = False

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


_AUTO_THRESH_METHODS = [("Otsu", "otsu"), ("Triangle", "triangle"), ("Valley", "valley")]


def _valley_threshold(gray):
    """Threshold at the deepest histogram valley between the two largest modes;
    falls back to Otsu when fewer than two modes are found."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float32)
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (0, 0), 2.0).flatten()
    peaks = [i for i in range(1, 255) if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1]]
    if len(peaks) < 2:
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return int(t)
    a, b = sorted(sorted(peaks, key=lambda i: float(hist[i]), reverse=True)[:2])
    return a + int(np.argmin(hist[a:b + 1]))


def _compute_auto_threshold(inputs, p):
    """Binary-threshold a grayscale image at an automatically chosen level
    (Otsu / Triangle / Valley) — no manual threshold value needed."""
    try:
        gray = _to_gray_u8(inputs[0])
        ttype = cv2.THRESH_BINARY_INV if p.get("invert", False) else cv2.THRESH_BINARY
        method = p.get("method", "otsu")
        if method == "triangle":
            _, out = cv2.threshold(gray, 0, 255, ttype | cv2.THRESH_TRIANGLE)
        elif method == "valley":
            _, out = cv2.threshold(gray, _valley_threshold(gray), 255, ttype)
        else:
            _, out = cv2.threshold(gray, 0, 255, ttype | cv2.THRESH_OTSU)
        return out
    except Exception as e:
        print(f"Error executing auto_threshold: {e}")
        return None


def _compute_backproject(inputs, p, in_space="bgr"):
    """Histogram backprojection: turn a histogram 'model' (a Histogram node) into a
    likelihood map on a target image. The target (input 0) is converted into the
    model's colour space, then each model-channel histogram is looked up per pixel
    and the per-channel likelihoods are multiplied — bright where the target
    matches the modelled distribution. For an HLS model, ``chroma_only`` ignores
    luminance (uses H + S) for lighting-robust colour matching."""
    try:
        image, model = inputs[0], inputs[1]
        if not isinstance(model, dict) or "hist" not in model:
            return None
        hists, names = model["hist"], model.get("names", [])
        space = model.get("space", "bgr")
        bgr = _as_bgr(image, in_space)
        if space == "hls":
            conv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        elif space == "gray":
            conv = _to_gray_u8(bgr)[:, :, None]
        else:
            conv = bgr
        chans = conv.shape[2] if conv.ndim == 3 else 1
        use = list(range(min(chans, len(hists))))
        if space == "hls" and p.get("chroma_only", True):   # drop luminance for robustness
            use = [c for c in use if c < len(names) and names[c] in ("H", "S")] or use
        like = np.ones(conv.shape[:2], np.float32)
        for c in use:
            h = np.asarray(hists[c], np.float32)
            h = h / (float(h.max()) or 1.0)
            chan = conv[:, :, c] if conv.ndim == 3 else conv
            like *= h[np.clip(chan.astype(np.int64), 0, len(h) - 1)]
        mx = float(like.max()) or 1.0
        return np.clip(like / mx * 255.0, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error executing backproject: {e}")
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


def _cluster_features(img, space, lum_weight, in_space="bgr"):
    """Build the per-pixel features k-means clusters on. Clustering in Lab/HLS
    and down-weighting luminance (``lum_weight`` < 1) makes the same physical
    color land in the same cluster across lighting changes. Features affect only
    the distance metric — cluster *colors* are still measured in the input space.
    ``space`` == "bgr" has no separable luminance channel, so the weight is a
    no-op there (the features are the raw input pixels)."""
    bgr = _as_bgr(img, in_space)
    if space == "lab":
        conv, lum_idx = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB), 0
    elif space == "hls":
        conv, lum_idx = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS), 1
    else:  # bgr
        conv, lum_idx = bgr, None
    feat = conv.reshape(-1, conv.shape[2]).astype(np.float32)
    if lum_idx is not None:
        feat = feat.copy()
        feat[:, lum_idx] *= float(lum_weight)
    return feat


def _center_luminance(centers, in_space):
    """Perceptual luminance of each (input-space) center — the canonical sort key,
    so 'index 0 = darkest' holds regardless of clustering or input color space."""
    c = np.clip(centers, 0, 255).astype(np.uint8).reshape(1, -1, centers.shape[1])
    gray = cv2.cvtColor(_as_bgr(c, in_space), cv2.COLOR_BGR2GRAY)
    return gray.reshape(-1).astype(np.float32)


def _pixel_chroma(bgr):
    """Per-pixel chroma = max(B,G,R) − min(B,G,R), 0..255. A colorfulness measure
    that is ~0 for white, gray AND black alike — unlike HLS saturation, which blows
    up near white/black as the double-cone narrows (a faintly-tinted near-white pixel
    reads as fully 'saturated' in HLS). So chroma is the right gate for "does this
    pixel have a real hue?"."""
    flat = bgr.reshape(-1, 3).astype(np.float32)
    return flat.max(axis=1) - flat.min(axis=1)


def _run_kmeans(feat, k, attempts=5):
    """Seeded, locked k-means partition (labels only), clamped to k<=len(feat). The
    seed+kmeans pair is global state, so the lock keeps it deterministic under batch
    fan-out; k=1 short-circuits (one cluster)."""
    k = max(1, min(int(k), len(feat)))
    if k == 1:
        return np.zeros(len(feat), np.int32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    with _KMEANS_LOCK:
        cv2.setRNGSeed(0)
        _, labels, _ = cv2.kmeans(np.ascontiguousarray(feat, np.float32), k, None,
                                  criteria, int(attempts), cv2.KMEANS_PP_CENTERS)
    return labels.flatten().astype(np.int32)


def _finalize_clusters(img, labels, feat, space, in_space):
    """Build the CLUSTERS payload from per-pixel labels: each center is the mean
    *input-space* color of its cluster; clusters are ordered dark->light (stable
    across runs, empty last); the preview diag is computed from ``feat``."""
    labels = labels.astype(np.int32)
    k = int(labels.max()) + 1 if labels.size else 1
    channels = img.shape[2] if img.ndim == 3 else 1
    orig = img.reshape(-1, channels).astype(np.float32)
    centers = np.zeros((k, channels), np.float32)
    nonempty = np.zeros(k, bool)
    for c in range(k):
        m = labels == c
        if m.any():
            centers[c] = orig[m].mean(axis=0)
            nonempty[c] = True
    key = _center_luminance(centers, in_space)
    key[~nonempty] = np.inf
    order = np.argsort(key)
    remap = np.empty(k, np.int32)
    remap[order] = np.arange(k)
    labels = remap[labels]
    centers = centers[order]
    return {"centers": centers, "labels": labels, "shape": img.shape, "k": k,
            "diag": _cluster_diag(feat, labels, k, space)}


def _kmeans_clusters(img, k, attempts=5, space="bgr", lum_weight=1.0, in_space="bgr"):
    """Run k-means on an image's pixels -> CLUSTERS payload. Clusters in the chosen
    feature space (with optional luminance down-weighting), reporting each center as
    the mean *input-space* color, with clusters ordered dark->light for stable
    labels/swatches/summary across runs."""
    feat = _cluster_features(img, space, lum_weight, in_space)
    labels = _run_kmeans(feat, max(1, int(k)), attempts)
    return _finalize_clusters(img, labels, feat, space, in_space)


def _kmeans_clusters_chroma_split(img, k_chromatic, gray_levels, chroma_min,
                                  space, lum_weight, in_space, attempts=5):
    """Cluster achromatic and colorful pixels separately. Pixels whose chroma is
    below ``chroma_min`` (white/gray/black) are clustered by LIGHTNESS into up to
    ``gray_levels`` clusters; the rest cluster by the chosen feature space (hue/Lab).
    This keeps desaturated pixels out of the coloured clusters — and, because chroma
    is ~0 for near-white/near-black too, it handles the HLS double-cone problem."""
    bgr = _as_bgr(img, in_space)
    achromatic = _pixel_chroma(bgr) < float(chroma_min)
    feat = _cluster_features(img, space, lum_weight, in_space)
    labels = np.zeros(feat.shape[0], np.int32)
    next_label = 0
    if achromatic.any() and int(gray_levels) >= 1:
        gray = _to_gray_u8(bgr).reshape(-1, 1).astype(np.float32)[achromatic]
        al = _run_kmeans(gray, int(gray_levels), attempts)
        labels[achromatic] = al + next_label
        next_label += int(al.max()) + 1
    chromatic = ~achromatic
    if chromatic.any() and int(k_chromatic) >= 1:
        cl = _run_kmeans(feat[chromatic], int(k_chromatic), attempts)
        labels[chromatic] = cl + next_label
    return _finalize_clusters(img, labels, feat, space, in_space)


def _cluster_diag(feat, labels, k, space, max_points=4000):
    """Precompute the data the cluster preview draws — done once here (in compute)
    so ``render_preview`` stays a cheap pure-drawing pass (it runs for every node
    on every batch-preview switch). Captures: per-cluster pixel ``counts``; a
    subsampled feature-space ``scatter`` (the two highest-variance feature
    channels) with each point's ``labels`` and the per-cluster centroids
    (``centers2d``) for marking; and per-cluster ``spread`` (RMS distance to the
    cluster's feature centroid = how tight / how mixed each cluster is)."""
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    dims = feat.shape[1]
    names = {"lab": ("L", "a", "b"), "hls": ("H", "L", "S")}.get(space, ("B", "G", "R"))
    # Two most-spread feature channels make the most informative 2D projection.
    if dims >= 2:
        var = feat.var(axis=0)
        ax0, ax1 = (int(a) for a in np.argsort(var)[::-1][:2])
    else:
        ax0 = ax1 = 0
    n = feat.shape[0]
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, size=max_points, replace=False)
    else:
        sel = np.arange(n)
    scatter = feat[np.ix_(sel, [ax0, ax1])].astype(np.float32)
    scatter_labels = labels[sel].astype(np.int32)
    centers2d = np.zeros((k, 2), np.float32)
    spread = np.zeros(k, np.float32)
    for c in range(k):
        m = labels == c
        if m.any():
            pts = feat[m]
            centers2d[c] = pts[:, [ax0, ax1]].mean(axis=0)
            d = pts - pts.mean(axis=0)
            spread[c] = float(np.sqrt((d * d).sum(axis=1).mean()))
    return {"counts": counts, "scatter": scatter, "scatter_labels": scatter_labels,
            "centers2d": centers2d, "spread": spread,
            "axes": (ax0, ax1), "axnames": (names[ax0], names[ax1])}


def _compute_kmeans(inputs, p, in_space="bgr"):
    """Cluster an image's pixels with k-means. Output is a clusters payload
    (centers + per-pixel labels + shape) — not an image — consumed downstream
    by Reduce Colors."""
    try:
        return _kmeans_clusters(inputs[0], p["k"], p.get("attempts", 5),
                                p.get("cluster_space", "bgr"),
                                p.get("lum_weight", 1.0), in_space)
    except Exception as e:
        print(f"Error executing kmeans: {e}")
        return None


# A genuine mode must dip on BOTH sides; the shallower-side drop must be at least
# this fraction of the peak's height. It rejects the shoulder of a quasi-flat step
# (one side ~flat -> shallower drop ~0) without rejecting a real sub-peak.
_PEAK_DIP_FLOOR = 0.05


def _peak_bases(hist, i, circular):
    """The two *key cols* of peak ``i``: descend left and right until the terrain
    rises above the peak (or the array bounds), returning the lowest point reached
    on each side as ``(left_base, right_base)``. For a non-circular channel (L/S)
    the region beyond the histogram is treated as count 0 — there are no pixels
    outside the range — so a peak sitting *on* the boundary (a pure-black L=0 or
    white L=255 background) descends to a real 0 valley on the open side. Hue is
    circular, so it has no boundary."""
    n = len(hist)
    h = float(hist[i])

    def base(step):
        m = h
        j = i
        for _ in range(n):
            j = (j + step) % n if circular else j + step
            if not circular and (j < 0 or j >= n):
                m = min(m, 0.0)  # outside the histogram the count is 0 (a valley floor)
                break
            v = float(hist[j])
            if v > h:            # reached higher terrain -> stop
                break
            if v < m:
                m = v
        return m

    return base(-1), base(1)


def _topographic_prominence(hist, i, circular):
    """Standard topographic prominence: peak height above its *key col* (the higher
    of the two surrounding valleys). Compares the peak to its own valley, not to the
    global maximum, so a small feature on a large uniform background still scores
    as a genuine peak while a bump on a bigger peak's shoulder scores low."""
    lb, rb = _peak_bases(hist, i, circular)
    return float(hist[i]) - max(lb, rb)


def _find_prominent_peaks(hist, min_prominence, circular):
    """Indices of the histogram's modes. A bin qualifies when it is a local maximum
    that BOTH:

    * rises above the **mean** of its two surrounding valleys by at least
      ``min_prominence`` of its own height. Using the mean (not the higher valley,
      as standard topographic prominence does) is lenient toward a *sub-peak* nested
      in a 'mountain range' — e.g. the 5 in ``0,0,3,5,4,7,8,3,0`` rises well above
      the average of its valleys (0 and 4) even though it only dips 1 on the side
      facing the taller 8; standard prominence judges it solely by that shallow side
      and drops it.
    * dips on **both** sides by at least ``_PEAK_DIP_FLOOR`` of its height. This
      rejects the shoulder of a quasi-flat step (one side ~flat), which the mean
      test alone would accept."""
    n = len(hist)
    thr = max(0.0, float(min_prominence))
    peaks = []
    for i in range(n):
        if circular:
            left, right = hist[(i - 1) % n], hist[(i + 1) % n]
        else:
            left = hist[i - 1] if i > 0 else -1.0
            right = hist[i + 1] if i < n - 1 else -1.0
        h = float(hist[i])
        if h <= 0 or not (h > left and h >= right):     # not a local maximum
            continue
        lb, rb = _peak_bases(hist, i, circular)
        mean_prominence = h - 0.5 * (lb + rb)            # rise above the average valley
        bilateral_dip = h - max(lb, rb)                  # the shallower side's drop
        if mean_prominence >= thr * h and bilateral_dip >= _PEAK_DIP_FLOOR * h:
            peaks.append(i)
    return peaks


def _detect_cluster_count(img, sigma, min_prominence, max_k, channel=1,
                          in_space="bgr", return_diag=False, sat_weight=1.0,
                          chroma_gate=0.0):
    """Pick k by smoothing one channel's histogram (Gaussian) and counting its
    modes (``_find_prominent_peaks``): local maxima that rise above the **mean** of
    their two surrounding valleys by ``min_prominence`` of their height *and* dip on
    both sides. Judging by the mean valley (not the global maximum, nor only the
    higher valley) keeps both a small colored feature on a big background and a
    sub-peak nested in a 'mountain range', while the both-sides-dip test rejects the
    shoulder of a quasi-flat step. The channel is an HLS index (1=Luminance/L,
    0=Hue, 2=Saturation); the input is first mapped to BGR via its tracked color
    space.

    Hue (channel 0) gets two extra treatments because raw hue is unreliable: the
    histogram is **weighted by chroma** (max−min of BGR), so washed-out pixels —
    whose hue is essentially noise — don't create phantom peaks, and smoothing + peak
    detection are **circular** (hue 0 and 179 are adjacent). Chroma (not HLS
    saturation) is used because HLS S blows up near white/black as the double-cone
    narrows — a faintly-tinted near-white pixel reads as fully saturated — whereas
    chroma is ~0 for white, gray AND black. ``sat_weight`` is the exponent on the
    chroma weight: each hue-bin pixel contributes ``(chroma/255) ** sat_weight``. 1.0
    (default) = linear; 0 = ignore chroma (every hue counts equally, so achromatic
    pixels form phantom peaks); > 1 = favour vivid pixels more aggressively. Hue only.

    With ``return_diag`` the function also returns a dict describing the curves
    and peaks it found, so the preview can draw exactly what k-detection saw: the
    original (undamped) histogram, the smoothed/saturation-damped curve the peaks
    were detected on, and the peak bin positions."""
    bgr = _as_bgr(img, in_space)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)   # H, L, S
    ch = int(channel)
    sig = max(0.5, float(sigma))
    if ch == 0:                                   # Hue: chroma-weighted + circular
        # OpenCV's 8-bit BGR2HLS can emit hue 180 (documented 0..179) for some
        # pixels; hue is circular so 180 (=360°) wraps to 0. Without this the
        # histogram would gain a spurious 181st bin.
        h = (hls[:, :, 0].reshape(-1).astype(np.int64)) % 180  # 0..179
        chroma = _pixel_chroma(bgr)                            # ~0 for white/gray/black
        sw = np.power(chroma / 255.0, max(0.0, float(sat_weight)))  # exponent (1.0 = linear)
        if chroma_gate > 0:                                   # hard-exclude achromatic pixels
            sw = np.where(chroma >= float(chroma_gate), sw, 0.0)  # (when the split is on)
        # "original" = plain hue count; the damped curve weights each pixel by its
        # chroma so achromatic pixels barely contribute — showing both makes the
        # chroma damping visible in the preview.
        raw = np.bincount(h, minlength=180).astype(np.float32)
        base = np.bincount(h, weights=sw, minlength=180).astype(np.float32)
        nbins, circular = 180, True
    else:
        chan = hls[:, :, ch].astype(np.uint8)
        raw = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten().astype(np.float32)
        base = raw.copy()
        nbins, circular = 256, False

    if circular:                                  # wrap-pad so smoothing is seamless
        pad = int(np.ceil(sig * 3)) + 1
        ext = np.concatenate([base[-pad:], base, base[:pad]])
        ext = cv2.GaussianBlur(ext.reshape(-1, 1), (0, 0), sig).flatten()
        hist = ext[pad:pad + nbins]
    else:
        hist = cv2.GaussianBlur(base.reshape(-1, 1), (0, 0), sig).flatten()

    peak_idx = _find_prominent_peaks(hist, min_prominence, circular)
    k = max(1, min(len(peak_idx), int(max_k)))
    if not return_diag:
        return k
    diag = {"mode": "peaks", "raw": raw, "smooth": hist, "peaks": peak_idx,
            "channel": ch, "nbins": nbins, "circular": circular,
            "chname": _CLUSTER_CHANNEL_NAMES.get(ch, "channel"),
            "peak_colors": _peak_mean_colors(bgr, hls, ch, nbins, circular,
                                             peak_idx, sig, sat_weight)}
    return k, diag


def _peak_mean_colors(bgr, hls, ch, nbins, circular, peak_idx, sig, sat_weight=1.0):
    """The real color each detected peak stands for: the **chroma-weighted mean BGR**
    of the pixels that fall in (a small ±window around) the peak's bin of the
    detection channel — same weighting the peak detection uses, so vivid pixels
    dominate the colour. Falls back to a synthetic swatch where there's no chromatic
    support (e.g. an all-gray bin)."""
    vals = hls[:, :, ch].reshape(-1).astype(np.int64) % nbins  # bin per pixel (hue wraps 180->0)
    chroma = _pixel_chroma(bgr) / 255.0
    w = np.maximum(np.power(chroma, max(0.0, float(sat_weight))), 1e-3)  # chroma weight
    flat = bgr.reshape(-1, 3).astype(np.float32)
    wsum = np.bincount(vals, weights=w, minlength=nbins)
    chan_sum = [np.bincount(vals, weights=w * flat[:, c], minlength=nbins) for c in range(3)]
    win = max(1, int(round(sig)))
    colors = []
    for pk in peak_idx:
        if circular:
            idx = [(pk + d) % nbins for d in range(-win, win + 1)]
        else:
            idx = list(range(max(0, pk - win), min(nbins, pk + win + 1)))
        wtot = float(wsum[idx].sum())
        if wtot > 1e-6:
            col = tuple(int(np.clip(chan_sum[c][idx].sum() / wtot, 0, 255)) for c in range(3))
        else:
            col = _peak_color(pk, ch)
        colors.append(col)
    return colors


def _elbow_k(ks, inertias, k_bias=0):
    """The k at the inertia knee, nudged by ``k_bias`` clusters. The knee is the
    point of greatest perpendicular distance below the first->last chord of the
    *normalized* inertia curve (``dist = |x + y - 1|``); ``k_bias`` then offsets the
    chosen index by that many steps (clamped to the available range) — a direct,
    predictable nudge toward more (+) or fewer (−) clusters. A blind index offset is
    used (rather than tilting the score) because past the knee the inertia curve is
    flat, so a score tilt would jump straight to max_k instead of nudging."""
    if not ks:
        return 2
    if len(ks) == 1:
        return int(ks[0])
    x = np.array(ks, np.float64)
    y = np.array(inertias, np.float64)
    x = (x - x[0]) / ((x[-1] - x[0]) or 1.0)
    y = (y - y.min()) / ((y.max() - y.min()) or 1.0)   # decreasing -> y[0]~1, y[-1]~0
    knee = int(np.argmax(np.abs(x + y - 1.0)))         # chord (0,1)->(1,0): x+y-1=0
    idx = min(len(ks) - 1, max(0, knee + int(round(k_bias))))
    return int(ks[idx])


def _detect_k_elbow(feat, max_k, attempts=3, return_diag=False, k_bias=0):
    """Data-driven k: run k-means for k=2..max_k on the *full feature space* and
    pick the elbow (knee) of the inertia/compactness curve (``_elbow_k``), nudged
    by ``k_bias`` toward more/fewer clusters. No colour-channel assumption, so it
    treats colour and gray uniformly and is reproducible across lighting (the
    channel/smoothing params don't apply in this mode).

    With ``return_diag`` it also returns the (ks, inertias) curve and the chosen
    knee, so the preview can draw the inertia curve the elbow was picked from."""
    mk = max(2, int(max_k))
    ks = list(range(2, mk + 1))
    if len(ks) <= 1:
        k = ks[0] if ks else 2
        return (k, {"mode": "elbow", "ks": ks, "inertias": [], "chosen": k}) \
            if return_diag else k
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    inertias = []
    for k in ks:
        with _KMEANS_LOCK:
            cv2.setRNGSeed(0)
            compactness, _, _ = cv2.kmeans(feat, k, None, criteria, int(attempts),
                                           cv2.KMEANS_PP_CENTERS)
        inertias.append(float(compactness))
    chosen = _elbow_k(ks, inertias, k_bias)
    if not return_diag:
        return chosen
    return chosen, {"mode": "elbow", "ks": ks, "inertias": inertias, "chosen": chosen}


def _compute_auto_cluster(inputs, p, in_space="bgr"):
    """K-means with an auto-detected cluster count. 'k_method' picks how k is
    found: 'peaks' counts modes in a chosen HLS channel histogram; 'elbow' runs
    k-means over a range and takes the inertia knee in the full feature space
    (colour + gray, more stable). With 'separate_achromatic', white/gray/black
    pixels (chroma below the threshold) are pulled out and clustered by lightness so
    they don't pollute the coloured clusters; the detected k then counts only the
    colourful clusters."""
    try:
        img = inputs[0]
        space, lw = p.get("cluster_space", "bgr"), p.get("lum_weight", 1.0)
        if p.get("k_method", "peaks") == "elbow":
            feat = _cluster_features(img, space, lw, in_space)
            k, kdiag = _detect_k_elbow(feat, p["max_k"], return_diag=True,
                                       k_bias=p.get("k_bias", 0))
        else:
            gate = p.get("chroma_min", 20) if p.get("separate_achromatic", False) else 0
            k, kdiag = _detect_cluster_count(img, p["smoothing"], p["min_prominence"],
                                             p["max_k"], p.get("channel", 1), in_space,
                                             return_diag=True,
                                             sat_weight=p.get("sat_weight", 1.0),
                                             chroma_gate=gate)
        if p.get("separate_achromatic", False):
            out = _kmeans_clusters_chroma_split(img, k, p.get("gray_levels", 2),
                                                p.get("chroma_min", 20), space, lw, in_space)
        else:
            out = _kmeans_clusters(img, k, 5, space, lw, in_space)
        if isinstance(out, dict):
            out["kdiag"] = kdiag       # how k was chosen (peaks histogram / elbow curve)
        return out
    except Exception as e:
        print(f"Error executing auto_cluster: {e}")
        return None


def _compute_mean_shift(inputs, p):
    """Mean-shift segmentation (mode-seeking, no k). Returns a segmented image."""
    try:
        img = inputs[0]
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(_to_gray_u8(img), cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return cv2.pyrMeanShiftFiltering(img, float(p["spatial"]), float(p["color"]))
    except Exception as e:
        print(f"Error executing mean_shift: {e}")
        return None


# The diagnostic bands were authored against a 320px-wide design; _S scales every
# length, font, and stroke up to the real preview width so the image is drawn at
# full resolution (crisp when the inspector zooms in) rather than upscaled.
_PREVIEW_W = 1024
_S = _PREVIEW_W / 320.0
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fs(base):
    return base * _S                          # font scale at the current width


def _tk(base=1):
    return max(1, int(round(base * _S)))      # stroke thickness


def _px(base):
    return int(round(base * _S))              # a length in pixels


def _center_bgr(c):
    """A cluster center's display BGR (centers are stored in the input space)."""
    return tuple(int(v) for v in c) if len(c) == 3 else (int(c[0]),) * 3


def _titled(band, title):
    """Prepend a small dark caption strip to a plotted band."""
    h = _px(16)
    strip = np.full((h, band.shape[1], 3), 45, np.uint8)
    cv2.putText(strip, title, (_px(4), int(h * 0.72)), _FONT, _fs(0.36),
                (210, 210, 210), _tk(1), cv2.LINE_AA)
    return np.vstack([strip, band])


def _simple_swatch(centers, width=_PREVIEW_W, height=None):
    """Fallback preview (no diagnostics available): equal-width color cells."""
    k = len(centers)
    if k == 0:
        return None
    height = height or _px(60)
    band = np.zeros((height, width, 3), np.uint8)
    for i, c in enumerate(centers):
        band[:, i * width // k:(i + 1) * width // k] = _center_bgr(c)
    return band


def _band_palette(centers, counts, width=_PREVIEW_W):
    """Proportional palette: each cluster's width ∝ its pixel population, % labelled.
    Kept short — about twice the label height — since it only shows the colors."""
    fs = _fs(0.34)
    (_, gh), _ = cv2.getTextSize("100%", _FONT, fs, _tk(1))
    height = gh * 2 + _px(4)
    band = np.full((height, width, 3), 30, np.uint8)
    total = int(counts.sum()) or 1
    x = 0
    for i, c in enumerate(centers):
        w = width - x if i == len(centers) - 1 else int(round(width * counts[i] / total))
        if w <= 0:
            continue
        color = _center_bgr(c)
        band[:, x:x + w] = color
        if w >= _px(26):
            tcol = (0, 0, 0) if sum(color) > 384 else (235, 235, 235)
            cv2.putText(band, f"{100 * counts[i] / total:.0f}%",
                        (x + _px(3), (height + gh) // 2), _FONT, fs, tcol, _tk(1), cv2.LINE_AA)
        x += w
    return band


def _band_scatter(diag, centers, width=_PREVIEW_W, height=None):
    """Feature-space scatter: subsampled pixels in the 2 most-spread feature
    channels, colored by cluster, with each cluster center ringed."""
    height = height or _px(150)
    band = np.full((height, width, 3), 30, np.uint8)
    pts = diag.get("scatter")
    if pts is None or len(pts) == 0:
        return band
    labs = diag["scatter_labels"]
    mn = pts.min(axis=0)
    rng = np.where((pts.max(axis=0) - mn) == 0, 1.0, pts.max(axis=0) - mn)
    m = _px(12)
    def project(p):
        nx = (p[0] - mn[0]) / rng[0]
        ny = (p[1] - mn[1]) / rng[1]
        return int(m + nx * (width - 2 * m)), int((height - m) - ny * (height - 2 * m))
    r = _tk(1)
    for p, l in zip(pts, labs):
        cv2.circle(band, project(p), r, _center_bgr(centers[l]), -1, cv2.LINE_AA)
    for c2, c in zip(diag.get("centers2d", []), centers):
        px, py = project(c2)
        cv2.circle(band, (px, py), _px(6), (255, 255, 255), _tk(1), cv2.LINE_AA)
        cv2.circle(band, (px, py), _px(5), _center_bgr(c), -1, cv2.LINE_AA)
    ax = diag.get("axnames", ("", ""))
    cv2.putText(band, ax[0], (width - _px(16), height - _px(4)), _FONT, _fs(0.34),
                (170, 170, 170), _tk(1), cv2.LINE_AA)
    cv2.putText(band, ax[1], (_px(4), _px(12)), _FONT, _fs(0.34),
                (170, 170, 170), _tk(1), cv2.LINE_AA)
    return band


def _band_spread(centers, spread, width=_PREVIEW_W, height=None):
    """A bar per cluster = intra-cluster RMS spread (taller = looser / more mixed)."""
    height = height or _px(70)
    band = np.full((height, width, 3), 30, np.uint8)
    k = len(spread)
    if k == 0:
        return band
    mx = float(spread.max()) or 1.0
    bw = width / k
    for i in range(k):
        x0, x1 = int(i * bw) + _px(1), int((i + 1) * bw) - _px(1)
        bh = int((spread[i] / mx) * (height - _px(14)))
        cv2.rectangle(band, (x0, height - 1), (x1, height - 1 - bh),
                      _center_bgr(centers[i]), -1)
    return band


def _peak_color(v, ch):
    """The display color a histogram peak at bin ``v`` in HLS channel ``ch`` stands
    for: a hue peak → that fully-saturated hue; an L/S peak → a neutral gray."""
    if ch == 0:
        hls = np.uint8([[[int(v), 128, 200]]])
    elif ch == 2:
        hls = np.uint8([[[0, 128, int(v)]]])
    else:
        return (int(v), int(v), int(v))
    return tuple(int(x) for x in cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)[0, 0])


def _band_hist(kdiag, width=_PREVIEW_W, height=None):
    """The k-detection histogram: original curve (gray) vs the smoothed,
    saturation-damped curve peaks were detected on (amber), with each detected
    peak marked by a vertical line in the color that peak represents."""
    height = height or _px(150)
    band = np.full((height, width, 3), 30, np.uint8)
    raw, sm = kdiag["raw"], kdiag["smooth"]
    n = len(sm)
    if n < 2:
        return band
    pad = _px(14)
    def plot(arr, color):
        mx = float(arr.max()) or 1.0
        prev = None
        for b in range(n):
            x = int(b / (n - 1) * (width - 1))
            y = int((height - pad) - (arr[b] / mx) * (height - 2 * pad))
            if prev is not None:
                cv2.line(band, prev, (x, y), color, _tk(1), cv2.LINE_AA)
            prev = (x, y)
    plot(raw, (120, 120, 120))                 # original histogram
    plot(sm, (0, 200, 255))                     # smoothed / saturation-damped
    smx = float(sm.max()) or 1.0
    peak_colors = kdiag.get("peak_colors")      # real saturation-weighted mean colors
    for i, pk in enumerate(kdiag["peaks"]):
        x = int(pk / (n - 1) * (width - 1))
        col = peak_colors[i] if peak_colors and i < len(peak_colors) \
            else _peak_color(pk, kdiag["channel"])
        y = int((height - pad) - (sm[pk] / smx) * (height - 2 * pad))
        cv2.line(band, (x, pad), (x, height - pad), col, _tk(1), cv2.LINE_AA)
        cv2.circle(band, (x, y), _tk(3), col, -1, cv2.LINE_AA)
        # light ring so a near-black real colour is still locatable on the dark band
        cv2.circle(band, (x, y), _tk(3), (210, 210, 210), _tk(1), cv2.LINE_AA)
    cv2.putText(band, f"{kdiag.get('chname', '')}  |  original",
                (_px(4), _px(12)), _FONT, _fs(0.34), (150, 150, 150), _tk(1), cv2.LINE_AA)
    cv2.putText(band, "damped+smoothed", (width - _px(118), _px(12)), _FONT, _fs(0.34),
                (0, 200, 255), _tk(1), cv2.LINE_AA)
    return band


def _band_elbow(kdiag, width=_PREVIEW_W, height=None):
    """The inertia-vs-k curve the elbow method chose from; chosen k marked green."""
    height = height or _px(150)
    band = np.full((height, width, 3), 30, np.uint8)
    ks, ys = kdiag["ks"], np.array(kdiag["inertias"], np.float64)
    if len(ks) < 1 or len(ys) != len(ks):
        return band
    mn, mx = float(ys.min()), float(ys.max())
    rng = (mx - mn) or 1.0
    m = _px(16)
    def xof(i):
        return int(m + (i / max(1, len(ks) - 1)) * (width - 2 * m))
    def yof(v):
        return int((height - m) - ((v - mn) / rng) * (height - 2 * m))
    for i in range(len(ks) - 1):
        cv2.line(band, (xof(i), yof(ys[i])), (xof(i + 1), yof(ys[i + 1])),
                 (0, 200, 255), _tk(1), cv2.LINE_AA)
    for i in range(len(ks)):
        cv2.circle(band, (xof(i), yof(ys[i])), _tk(2), (200, 200, 200), -1, cv2.LINE_AA)
    if kdiag.get("chosen") in ks:
        ci = ks.index(kdiag["chosen"])
        cv2.line(band, (xof(ci), m), (xof(ci), height - m), (0, 230, 0), _tk(1), cv2.LINE_AA)
        cv2.circle(band, (xof(ci), yof(ys[ci])), _px(5), (0, 230, 0), _tk(1), cv2.LINE_AA)
        cv2.putText(band, f"k={kdiag['chosen']}", (xof(ci) + _px(4), m + _px(12)),
                    _FONT, _fs(0.36), (0, 230, 0), _tk(1), cv2.LINE_AA)
    cv2.putText(band, "inertia vs k", (_px(4), _px(12)), _FONT, _fs(0.34),
                (150, 150, 150), _tk(1), cv2.LINE_AA)
    return band


# --- Density clustering via the optional OPTICS-Clustering `optics` package -------
# Driven by optics.cluster_image, which dedups, voxel-quantizes, and converts the colour
# space (sRGB -> CIELAB) internally — the node just hands it the BGR image.
# Algorithm modes map 1:1 to cluster_image's `algo`. HDBSCAN / OPTICS are exact; sHDBSCAN
# / sOPTICS are scalable approximate variants (CEOs random projections).
_HDBSCAN_ALGOS = [
    ("HDBSCAN", "hdbscan"),
    ("OPTICS — Xi", "optics-xi"),
    ("OPTICS — threshold", "optics-threshold"),
    ("sHDBSCAN (approx)", "shdbscan"),
    ("sOPTICS (approx)", "soptics"),
]
# Colour space the library clusters in: perceptual CIELAB (the study's pick) or raw RGB.
_HDBSCAN_SPACES = [("Lab (perceptual)", "lab"), ("RGB", "rgb")]
# What to do with points the algorithm calls noise (-1). "nearest" assigns each to its
# closest cluster (a usable quantization, like K-Means); "flag" paints them magenta so
# you can SEE what was sparse — handy for tuning, but turns a high-noise result all-pink.
_NOISE_MODES = [
    ("Assign to nearest cluster", "nearest"),
    ("Flag colour (magenta)", "flag"),
]
# Distance metric — only the approximate variants (sHDBSCAN / sOPTICS) use it. L2 is correct
# for colour; cosine clusters by hue and merges black/white/gray (rarely wanted for colour).
_HDBSCAN_METRICS = [
    ("Euclidean (L2)", "l2"),
    ("Manhattan (L1)", "l1"),
    ("Cosine (hue)", "cosine"),
]
_NOISE_BGR = (200, 0, 200)   # magenta: the flag colour painted onto noise (-1) pixels


def _attach_reachability(out, opt, bgr, space, voxel, min_pts):
    """Precompute the OPTICS reachability plot (the signature density diagnostic) into the
    payload's diag, so render_preview stays a cheap pure-draw. cluster_image doesn't return
    the ordering, so we recompute it the same way it preprocesses: sRGB→Lab via the library,
    voxel-quantize, dedup, then order — colouring each ordered point by its FINAL cluster.
    A diagnostic only; never let it fail the clustering."""
    try:
        rgb = np.ascontiguousarray(bgr[..., ::-1].reshape(-1, 3).astype(np.float64))
        coords = opt.srgb_to_lab(rgb) if space == "lab" else rgb
        if voxel and voxel > 0:
            coords = opt.quantize(np.ascontiguousarray(coords), float(voxel))
        coords = np.ascontiguousarray(coords)
        uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
        # A SMALL min_pts gives a crisp plot (visible peaks). min_cluster_size (often
        # hundreds) over-smooths the reachability into a featureless ramp.
        rmp = int(min(25, max(5, len(uniq) // 80)))
        reach = opt.compute_reachability(np.ascontiguousarray(uniq), min_pts=rmp)
        per_unique = np.empty(len(uniq), np.int32)
        per_unique[inverse.reshape(-1)] = out["labels"]
        order = np.asarray(reach["point_index"]).astype(np.int64)
        out["diag"]["reach"] = np.asarray(reach["reachability"], np.float32)
        out["diag"]["reach_labels"] = per_unique[order]
    except Exception:
        pass


def _finalize_hdbscan(bgr, labels, in_space, noise_mode="nearest"):
    """Build a CLUSTERS payload from density labels (-1 = noise): real clusters ordered
    dark→light with their mean BGR centre. Noise handling: ``nearest`` reassigns each
    noise pixel to its closest cluster (a usable quantization), ``flag`` appends a magenta
    NOISE centre the -1 pixels map onto (so you can see what was sparse). Either way the
    payload is the standard centers/labels, so Reduce Colors + the diagnostics consume it
    like K-Means. ``n_noise`` always reports how many pixels were originally noise."""
    labels = labels.astype(np.int32)
    real = labels[labels >= 0]
    k = int(real.max()) + 1 if real.size else 0
    flat = bgr.reshape(-1, 3).astype(np.float32)
    centers = np.zeros((k, 3), np.float32)
    counts = np.zeros(k, np.int64)
    for c in range(k):
        m = labels == c
        if m.any():
            centers[c] = flat[m].mean(axis=0)
            counts[c] = int(m.sum())
    if k:
        order = np.argsort(_center_luminance(centers, "bgr"))   # centres are BGR
        remap = np.empty(k, np.int32)
        remap[order] = np.arange(k)
        nonneg = labels >= 0
        labels[nonneg] = remap[labels[nonneg]]
        centers, counts = centers[order], counts[order]
    noise_n = int((labels < 0).sum())
    if noise_n == 0:
        noise_index = -1                    # nothing to handle either way
    elif noise_mode == "nearest" and k > 0:
        # Assign each noise pixel to the nearest cluster centre (BGR distance). One pass
        # per centre keeps memory at O(n_noise) instead of O(n_noise × k).
        nm = labels < 0
        npx = flat[nm]
        best = np.full(len(npx), np.inf, np.float32)
        near = np.zeros(len(npx), np.int32)
        for c in range(k):
            dc = ((npx - centers[c]) ** 2).sum(1)
            upd = dc < best
            best[upd] = dc[upd]
            near[upd] = c
        labels[nm] = near
        counts = np.bincount(labels, minlength=k).astype(np.int64)
        noise_index = -1                    # no noise centre in this mode
    else:
        # Flag mode (or nothing clustered: k == 0): append a magenta noise centre.
        centers = np.vstack([centers, np.array([list(_NOISE_BGR)], np.float32)])
        labels[labels < 0] = k              # -1 -> the noise centre, always in range
        counts = np.append(counts, noise_n)
        noise_index = k
    # Subsample (colour, final-cluster) pairs for the 3-D colour-space scatter — done
    # here so render_preview stays a cheap pure-draw.
    nlab = labels.shape[0]
    sel = (np.random.default_rng(0).choice(nlab, 4000, replace=False)
           if nlab > 4000 else np.arange(nlab))
    diag = {"counts": counts,
            "scatter3d": flat[sel].astype(np.float32),       # original BGR colours
            "scatter3d_labels": labels[sel].astype(np.int32)}
    return {"centers": centers, "labels": labels, "shape": bgr.shape, "k": k,
            "diag": diag, "n_noise": noise_n, "noise_index": noise_index}


def _compute_hdbscan(inputs, p, in_space="bgr"):
    """Density-based colour clustering via OPTICS-Clustering's high-level ``cluster_image``.
    No k — just a minimum size; sparse 'bridge' colours fall out as NOISE. The library
    handles dedup, voxel quantization, and the sRGB→CIELAB conversion internally, so the node
    just hands it the BGR image. Algorithm modes map 1:1 to its ``algo``: exact HDBSCAN /
    OPTICS, or approximate sHDBSCAN / sOPTICS. Returns a CLUSTERS payload. A missing library
    propagates a clear error, which the engine shows as the node's red error border."""
    img = inputs[0]
    if img is None:
        return None
    opt = optics_backend.load()             # RuntimeError (clear msg) if unavailable
    bgr = _as_bgr(img, in_space)
    # Map the node's algorithm to a cluster_image `algo` (migrate a legacy "optics" value).
    algo = {"optics": "optics-xi"}.get(p.get("algorithm", "hdbscan"), p.get("algorithm", "hdbscan"))
    if algo not in ("hdbscan", "optics-xi", "optics-threshold", "shdbscan", "soptics"):
        algo = "hdbscan"
    space = "lab" if p.get("color_space", "lab") == "lab" else "rgb"
    vb = int(p.get("voxel_bin", 2))
    voxel = float(vb) if vb > 0 else 0.0    # 0 disables; else the grid size in colour units
    mcs = max(2, int(p.get("min_cluster_size", 50)))
    # The native extension is not thread-safe; serialize against the engine's batch
    # fan-out (and any other Density Cluster node running concurrently).
    with _OPTICS_LOCK:
        res = opt.cluster_image(
            bgr, algo=algo, space=space, voxel=voxel, bgr=True,
            min_cluster_size=mcs, min_pts=mcs,
            min_cluster_frac=float(p.get("min_cluster_frac", 0.003)),
            metric=p.get("metric", "l2"), seed=int(p.get("seed", 42)), max_dim=None)
        labels = np.asarray(res.labels).reshape(-1).astype(np.int32)   # per-pixel, -1 = noise
    out = _finalize_hdbscan(bgr, labels, in_space, noise_mode=p.get("noise_handling", "nearest"))
    if p.get("show_reachability"):
        with _OPTICS_LOCK:
            _attach_reachability(out, opt, bgr, space, voxel, mcs)
    return out


def _band_reachability(reach, labels, centers, height=None):
    """The OPTICS reachability plot: ordered points as bars (height = reachability),
    each coloured by the cluster it belongs to (noise = the flag colour). Valleys (runs
    of low bars) are clusters; peaks are the gaps between them. UNDEFINED (-1) reach is
    drawn full-height. Columns subsample the ordering, so it's O(width) to draw."""
    width = _PREVIEW_W
    height = height or _px(70)
    band = np.full((height, width, 3), 30, np.uint8)
    n = len(reach)
    if n == 0:
        return band
    r = np.asarray(reach, np.float32).copy()
    finite = r[r >= 0]
    rmax = float(finite.max()) if finite.size else 1.0
    rmax = rmax if rmax > 0 else 1.0
    r[r < 0] = rmax                                      # UNDEFINED -> full-height peak
    cen = np.clip(centers, 0, 255).astype(np.uint8)
    lab = np.asarray(labels, np.int64)
    col_idx = np.clip(np.arange(width) * n // width, 0, n - 1)
    heights = (r[col_idx] / rmax * (height - 1)).astype(int)
    col_lab = lab[col_idx]
    for x in range(width):
        l = int(col_lab[x])
        c = cen[l] if 0 <= l < len(cen) else (160, 160, 160)
        band[height - 1 - heights[x]:height - 1, x] = c
    return band


def _band_cluster_ribbon(labels, centers, height=None):
    """A thin strip aligned with the reachability plot's x-axis: each ordered point painted
    its CLUSTER's mean colour. Read it as 'where does each cluster start/end in the ordering'
    — the colours match the bars directly above it."""
    width = _PREVIEW_W
    height = height or _px(13)
    band = np.full((height, width, 3), 30, np.uint8)
    n = len(labels)
    if n == 0:
        return band
    cen = np.clip(centers, 0, 255).astype(np.uint8)
    lab = np.asarray(labels, np.int64)[np.clip(np.arange(width) * n // width, 0, n - 1)]
    for x in range(width):
        l = int(lab[x])
        band[:, x] = cen[l] if 0 <= l < len(cen) else (160, 160, 160)
    return band


def _render_hdbscan(inputs, output, p):
    """Inspector preview: the image recoloured by cluster mean (noise pixels flagged) —
    what a downstream Reduce Colors would output. When the reachability diagnostic was
    computed (the 'Show reachability plot' toggle), stack the OPTICS reachability plot
    below it so the density landscape behind the clusters is visible."""
    if not isinstance(output, dict):
        return None
    centers = np.clip(output["centers"], 0, 255).astype(np.uint8)
    labels = output.get("labels")
    if labels is None or len(centers) == 0:
        return None
    k = int(output.get("k", 0))
    n_noise = int(output.get("n_noise", 0))
    flagged = int(output.get("noise_index", -1)) >= 0
    diag = output.get("diag") or {}
    counts = diag.get("counts")
    if counts is None:
        counts = np.bincount(labels, minlength=len(centers)).astype(np.int64)

    recolored = centers[labels].reshape(output["shape"])
    w = _PREVIEW_W
    h = max(1, int(round(recolored.shape[0] * w / recolored.shape[1])))
    img = cv2.resize(recolored, (w, h), interpolation=cv2.INTER_NEAREST)

    # cv2's Hershey font is ASCII-only — keep titles plain (no em-dash etc.).
    title = f"clustered image:  {k} cluster{'' if k == 1 else 's'}"
    if n_noise:
        pct = 100 * n_noise // max(1, len(labels))
        title += "  (+ noise)" if flagged else f"  ({pct}% noise -> nearest)"
    # (The 3-D colour-space scatter lives in the inspector pane now — interactive there,
    #  driven by diag['scatter3d'] — so it's not baked into this static preview.)
    bands = [
        _titled(img, title),
        _titled(_band_palette(centers, counts), "extracted colours  (bar width = pixel share)"),
    ]
    if "reach" in diag:
        reach = np.vstack([
            _band_reachability(diag["reach"], diag["reach_labels"], centers),
            _band_cluster_ribbon(diag["reach_labels"], centers),
        ])
        bands.append(_titled(reach, "OPTICS reachability  (height = density; strip = cluster)"))
    return np.vstack(bands)


def _summary_hdbscan(output, p):
    if not isinstance(output, dict):
        return {}
    info = {"clusters": int(output.get("k", 0))}
    labels = output.get("labels")
    if labels is not None and len(labels):
        info["noise"] = f"{100 * int(output.get('n_noise', 0)) // len(labels)}%"
    return info


def _render_kmeans(inputs, output, p):
    """Inspector preview for K-Means: a proportional palette (width ∝ population),
    a feature-space scatter colored by cluster, and per-cluster spread bars."""
    if not isinstance(output, dict):
        return None
    centers = np.clip(output["centers"], 0, 255).astype(np.uint8)
    if len(centers) == 0:
        return None
    diag = output.get("diag")
    if diag is None:
        return _simple_swatch(centers)
    return np.vstack([
        _titled(_band_palette(centers, diag["counts"]), "palette after clustering"),
        _titled(_band_scatter(diag, centers), "feature space (clusters)"),
        _titled(_band_spread(centers, diag["spread"]), "cluster spread (tightness)"),
    ])


def _render_auto_cluster(inputs, output, p):
    """Inspector preview for Auto Cluster: how k was chosen (the peak-detection
    histogram or the elbow curve), then the resulting proportional palette."""
    if not isinstance(output, dict):
        return None
    centers = np.clip(output["centers"], 0, 255).astype(np.uint8)
    if len(centers) == 0:
        return None
    diag = output.get("diag")
    if diag is None:
        return _simple_swatch(centers)
    kdiag = output.get("kdiag") or {}
    if kdiag.get("mode") == "elbow":
        top = _titled(_band_elbow(kdiag), f"elbow: chose k={output.get('k', '?')}")
    elif kdiag.get("mode") == "peaks":
        top = _titled(_band_hist(kdiag),
                      f"k from {kdiag.get('chname', '')} peaks  (k={output.get('k', '?')})")
    else:
        top = None
    palette = _titled(_band_palette(centers, diag["counts"]), "palette after clustering")
    return np.vstack([b for b in (top, palette) if b is not None])


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


def _scale_contours(payload, scale):
    """Scale a CONTOURS payload's geometry by ``scale`` — the contour points, the
    reference shape, and the preview background — so contours found on a downscaled
    image can be mapped back onto the full-resolution original (and vice versa)."""
    out = dict(payload)
    out["contours"] = [np.round(np.asarray(c, np.float32) * scale).astype(np.int32)
                       for c in payload.get("contours", [])]
    shape = payload.get("shape")
    if shape is not None and len(shape) >= 2:
        out["shape"] = ((max(1, int(round(shape[0] * scale))),
                         max(1, int(round(shape[1] * scale)))) + tuple(shape[2:]))
    bg = payload.get("background")
    if bg is not None:
        nw = max(1, int(round(bg.shape[1] * scale)))
        nh = max(1, int(round(bg.shape[0] * scale)))
        out["background"] = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return out


_RESIZE_MODES = [("Scale factor", "scale"), ("Longer edge → length", "fixed")]


def _resize_factor(p, h, w):
    """The scale factor Resize applies, given the source height/width. 'scale' uses
    the factor directly; 'fixed' scales so the longer edge becomes ``length`` px."""
    if p.get("mode", "scale") == "fixed":
        longer = max(int(h), int(w))
        return (float(p["length"]) / longer) if longer > 0 else 1.0
    return float(p.get("scale", 1.0))


def _compute_resize(inputs, p):
    """Scale an image — or a CONTOURS payload (scaling the contour coordinates), so a
    segmentation done on a downscaled image can be mapped back to the original
    resolution. ``mode`` is either a direct scale factor or 'fixed' (scale so the
    image's longer edge becomes ``length`` px). Interpolation applies to images only."""
    try:
        data = inputs[0]
        if isinstance(data, dict) and "contours" in data:
            shape = data.get("shape") or (0, 0)
            h, w = (shape[0], shape[1]) if len(shape) >= 2 else (0, 0)
            s = _resize_factor(p, h, w)
            return data if s == 1.0 else _scale_contours(data, s)
        img = data
        h, w = img.shape[:2]
        s = _resize_factor(p, h, w)
        if s <= 0 or s == 1.0:
            return img
        if p.get("mode", "scale") == "fixed":
            # Size explicitly so the longer edge lands exactly on the target length.
            return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                              interpolation=int(p["interpolation"]))
        return cv2.resize(img, None, fx=s, fy=s, interpolation=int(p["interpolation"]))
    except Exception as e:
        print(f"Error executing resize: {e}")
        return None


def _render_resize(inputs, output, p):
    """When Resize is fed contours, preview the (scaled) contours; an image output
    displays itself."""
    if isinstance(output, dict) and "contours" in output:
        return _draw_contours_preview(output)
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


def _build_contour_palette(per_half=6):
    """2*per_half evenly-spaced, full-saturation colours for contour previews.
    The first ``per_half`` are the even-depth palette, the rest the odd-depth one;
    their hues interleave around the wheel, so the two halves are disjoint and each
    is evenly spread for maximum contrast (and an immediate parent/child, being in
    different halves, always get different hues). Returns a list of BGR tuples."""
    n = 2 * per_half
    hsv = np.zeros((1, n, 3), np.uint8)
    hsv[0, :, 1:] = 255                                   # full saturation + value
    hsv[0, :, 0] = [int(round(k * 180.0 / n)) % 180 for k in range(n)]  # OpenCV hue
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    even = [tuple(int(v) for v in bgr[k]) for k in range(0, n, 2)]      # hues 0,2,4..
    odd = [tuple(int(v) for v in bgr[k]) for k in range(1, n, 2)]       # hues 1,3,5..
    return even + odd


# Evenly-spaced, high-contrast palette; split in half by nesting-depth parity.
_CONTOUR_COLORS = _build_contour_palette(6)


def _draw_contours_preview(payload, filled=False, dim=1.0, thickness=1):
    bg = payload.get("background")
    if bg is None:
        return None
    contours = payload.get("contours", [])
    ids = payload.get("ids", list(range(len(contours))))
    depths = payload.get("depths", [0] * len(contours))
    out = bg.copy()
    if dim < 1.0:                       # dim the backdrop so bright outlines pop
        out = (out.astype(np.float32) * dim).astype(np.uint8)
    thickness = cv2.FILLED if filled else thickness
    nc = len(_CONTOUR_COLORS)
    half = nc // 2
    # Batch by (nesting depth, colour): ONE cv2.drawContours call per group instead
    # of one per contour (so thousands of contours become a handful of calls — the
    # per-contour loop was the batch-switch lag). Depth ascending keeps filled
    # children on top of parents.
    #
    # Colour = id within a depth-parity palette: even depths use colours 0..half-1,
    # odd depths use half..nc-1. An immediate parent (depth d) and child (depth d+1)
    # are always in different halves, so a filled hole never vanishes into its
    # parent's colour. Depth d+2 reuses depth d's palette (allowed — the depth d+1
    # ring in between is a different colour). Still stable per (id, depth).
    groups = defaultdict(list)
    for i, c in enumerate(contours):
        ci = (ids[i] % half) + (depths[i] % 2) * half
        groups[(depths[i], ci)].append(c)
    for d, ci in sorted(groups):
        cv2.drawContours(out, groups[(d, ci)], -1, _CONTOUR_COLORS[ci], thickness)
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


def _render_largest_contour(inputs, output, p):
    """Draw the kept contours' outlines boldly on a dimmed backdrop, so which
    contours survived is obvious (a 1px outline on a bright binary blob is not)."""
    if not isinstance(output, dict):
        return None
    return _draw_contours_preview(output, dim=0.4, thickness=2)


def _summary_filter_contours(output, p):
    if not isinstance(output, dict):
        return {}
    return {"kept": len(output.get("contours", [])), "of": int(output.get("_total", 0))}


_REGION_CHANNELS = [
    ("Full color", -1),       # compare similarity on all BGR channels
    ("Luminance (L)", 1),     # otherwise an HLS channel index
    ("Hue (H)", 0),
    ("Saturation (S)", 2),
]
_CONNECTIVITY = [("4-connected", 4), ("8-connected", 8)]


def _region_compare_image(img, channel, in_space):
    """The 1- or 3-channel uint8 image floodFill measures similarity on.
    ``channel`` == -1 -> full BGR; else an HLS channel (1=L, 0=H, 2=S)."""
    bgr = _as_bgr(img, in_space)
    if int(channel) < 0:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)[:, :, int(channel)].copy()


def _region_depths(labels, n_regions):
    """Nesting depth per region id (1..n_regions): 0 if the region touches the image
    border, else 1 + the depth of the region that *surrounds* it (its majority
    4-neighbour). A flood-fill / connected-components label map carries no contour
    hierarchy, so we recover one cheaply here — one boundary-adjacency pass + a
    parent-chain walk — so holes get a different colour from their parents."""
    if n_regions <= 0:
        return []
    lab = labels.astype(np.int64)
    base = n_regions + 1
    border = set(int(v) for v in np.unique(np.concatenate(
        [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])).tolist())
    # shared-boundary length for each ordered neighbour pair (a, b), a != b.
    counts = defaultdict(dict)
    def accumulate(a, b):
        m = a != b
        if not m.any():
            return
        uk, uc = np.unique(a[m].ravel() * base + b[m].ravel(), return_counts=True)
        for k, c in zip(uk.tolist(), uc.tolist()):
            av, bv = divmod(k, base)
            counts[av][bv] = counts[av].get(bv, 0) + int(c)
    accumulate(lab[:, :-1], lab[:, 1:]); accumulate(lab[:, 1:], lab[:, :-1])
    accumulate(lab[:-1, :], lab[1:, :]); accumulate(lab[1:, :], lab[:-1, :])
    # parent = the non-background neighbour sharing the most boundary
    parent = {}
    for r in range(1, n_regions + 1):
        if r in border:
            continue
        nb = [(c, b) for b, c in counts.get(r, {}).items() if b not in (0, r)]
        if nb:
            parent[r] = max(nb)[1]
    depth = {}
    def depth_of(r, seen):
        if r in border or r not in parent:
            return 0
        if r in depth:
            return depth[r]
        if r in seen:               # cycle guard -> treat as root
            return 0
        seen.add(r)
        depth[r] = 1 + depth_of(parent[r], seen)
        return depth[r]
    return [depth_of(r, set()) for r in range(1, n_regions + 1)]


def _labels_to_payload(labels, n_regions, img, background):
    """Turn an int label image (1..n_regions, 0 = none) into a CONTOURS payload:
    one outer contour per connected region, drawn on `background` for inspection.
    Region nesting depth is recovered so the preview colours holes vs parents."""
    region_depth = _region_depths(labels, n_regions)
    contours, depths = [], []
    for rid in range(1, n_regions + 1):
        cs, _ = cv2.findContours((labels == rid).astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)
        depths.extend([region_depth[rid - 1]] * len(cs))
    n = len(contours)
    return {"contours": contours, "hierarchy": None,
            "ids": list(range(n)), "depths": depths,
            "shape": img.shape, "background": background}


_EXACT_MAX_COLORS = 256   # above this, the per-colour exact path loses to floodFill


def _color_key(comp):
    """Per-pixel integer key: the single channel, or packed BGR (24-bit)."""
    if comp.ndim == 2:
        return comp.astype(np.int64)
    c = comp.astype(np.int64)
    return (c[:, :, 0] << 16) | (c[:, :, 1] << 8) | c[:, :, 2]


def _label_regions_exact(key, connectivity, uniq=None):
    """delta == 0 path for *few* distinct colours: connected components per unique
    value. Fast only when there are few colours (its cost is O(#colours * H*W));
    the caller falls back to floodFill otherwise. Returns (label image, count)."""
    if uniq is None:
        uniq = np.unique(key)
    labels = np.zeros(key.shape, np.int32)
    next_id = 0
    for val in uniq:
        m = key == val
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=int(connectivity))
        # Offset this colour's components (1..n-1) into the global id space in one
        # masked write — not a full-image pass per component.
        labels[m] = lab[m].astype(np.int32) + next_id
        next_id += n - 1
    return labels, next_id


def _label_regions_floodfill(comp, delta, connectivity):
    """delta > 0 path: tolerance region-growing with cv2.floodFill (FIXED_RANGE,
    compared to each seed). Returns (label image, region count)."""
    h, w = comp.shape[:2]
    ch = 1 if comp.ndim == 2 else comp.shape[2]
    lo = up = (delta,) * ch
    flags = int(connectivity) | (2 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    labels = np.zeros((h, w), np.int32)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    lab_flat = labels.reshape(-1)             # a view: stays in sync with `labels`
    seeds = 0
    for start in range(h * w):
        if lab_flat[start]:                   # already assigned to a region
            continue
        y, x = divmod(start, w)
        _, _, _, (rx, ry, rw, rh) = cv2.floodFill(comp, ffmask, (x, y), 0, lo, up, flags)
        seeds += 1
        # Confine per-region work to the fill's bounding box. The mask is offset
        # by +1 on both axes relative to the image.
        sub = ffmask[ry + 1:ry + rh + 1, rx + 1:rx + rw + 1]
        new = sub == 2
        labels[ry:ry + rh, rx:rx + rw][new] = seeds
        sub[new] = 1                          # mark blocked so it can't refill
    return labels, seeds


def _compute_label_regions(inputs, p, in_space="bgr"):
    """Group connected pixels of (near-)uniform color into regions and emit them
    as a CONTOURS payload — so the existing Filter Contours node and contour
    previews work directly. Similarity is measured on the chosen channel within
    +/- delta of the *seed* pixel (cv2.floodFill, FIXED_RANGE) at 4- or
    8-connectivity. delta == 0 takes a fast exact-equality path (connected
    components per unique value) — ideal straight after Reduce Colors; delta > 0
    tolerates anti-aliasing / noise at region edges. Note: Hue is treated
    linearly, so reds near the 0/179 wrap won't merge."""
    try:
        img = inputs[0]
        comp = _region_compare_image(img, p.get("channel", -1), in_space)
        if comp.dtype != np.uint8:
            comp = np.clip(comp, 0, 255).astype(np.uint8)
        delta = int(p.get("delta", 8))
        conn = int(p.get("connectivity", 4))
        if delta == 0:
            # Exact connected-components per colour is only fast when colours are
            # few (e.g. after Reduce Colors); otherwise a single floodFill(0)
            # sweep — equivalent result — is much faster.
            key = _color_key(comp)
            uniq = np.unique(key)
            if len(uniq) <= _EXACT_MAX_COLORS:
                labels, n = _label_regions_exact(key, conn, uniq)
            else:
                labels, n = _label_regions_floodfill(comp, 0, conn)
        else:
            labels, n = _label_regions_floodfill(comp, delta, conn)
        return _labels_to_payload(labels, n, img, _as_bgr(img, in_space))
    except Exception as e:
        print(f"Error executing label_regions: {e}")
        return None


def _compute_connected_components(inputs, p):
    """Label connected foreground blobs in a BINARY image with
    cv2.connectedComponents (4/8-connectivity), then emit them as a CONTOURS
    payload. Foreground = non-zero pixels; the zero background is one region."""
    try:
        img = inputs[0]
        gray = _to_gray_u8(img)
        binary = (gray > 0).astype(np.uint8)
        n, labels = cv2.connectedComponents(binary, connectivity=int(p.get("connectivity", 8)))
        bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return _labels_to_payload(labels, n - 1, img, bg)   # label 0 = background
    except Exception as e:
        print(f"Error executing connected_components: {e}")
        return None


# --- segmentation pipeline: colour mask -> largest contour -> deskew & crop ---
_COLOR_SELECT = [("Outside (foreground)", "outside"), ("Inside (matches colour)", "inside")]


def _compute_color_mask(inputs, p, in_space="bgr"):
    """Binary mask from a target colour +/- delta (cv2.inRange). 'outside' keeps
    pixels NOT near the colour (the foreground); 'inside' keeps the match."""
    try:
        img = _as_bgr(inputs[0], in_space)
        b, g, r, d = int(p["blue"]), int(p["green"]), int(p["red"]), int(p["delta"])
        lo = np.array([max(0, b - d), max(0, g - d), max(0, r - d)], np.uint8)
        hi = np.array([min(255, b + d), min(255, g + d), min(255, r + d)], np.uint8)
        mask = cv2.inRange(img, lo, hi)            # 255 where within delta of colour
        if p.get("select", "outside") == "outside":
            mask = cv2.bitwise_not(mask)
        return mask
    except Exception as e:
        print(f"Error executing color_mask: {e}")
        return None


def _compute_largest_contour(inputs, p):
    """Keep only the N largest contours (by area) from a CONTOURS payload."""
    try:
        payload = inputs[0]
        if not isinstance(payload, dict):
            return None
        contours = payload.get("contours", [])
        ids = payload.get("ids", list(range(len(contours))))
        depths = payload.get("depths", [0] * len(contours))
        k = max(1, int(p.get("count", 1)))
        top = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]),
                     reverse=True)[:k]
        keep = sorted(top)                          # stable display order + colours
        out = dict(payload)
        out["contours"] = [contours[i] for i in keep]
        out["ids"] = [ids[i] for i in keep]
        out["depths"] = [depths[i] for i in keep]
        out["_total"] = len(contours)
        return out
    except Exception as e:
        print(f"Error executing largest_contour: {e}")
        return None


def _compute_crop_to_contour(inputs, p, in_space="bgr"):
    """Deskew + crop the image to the largest contour's min-area rectangle, with a
    border and optional output scale. Inputs: (image, CONTOURS)."""
    try:
        img = _as_bgr(inputs[0], in_space)
        payload = inputs[1]
        if not isinstance(payload, dict) or not payload.get("contours"):
            return None
        cnt = max(payload["contours"], key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)                 # ((cx,cy),(w,h),angle)
        (cx, cy), _, angle = rect
        border = int(p.get("border", 8))
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        # Rotated rect corners -> their bbox (robust to OpenCV's angle/size
        # convention) sets the OUTPUT canvas size. Sizing the warp to the rect
        # (not the source image) means a near-90deg deskew can't clip the object.
        box = cv2.boxPoints(rect)
        rb = (M @ np.hstack([box, np.ones((4, 1), np.float32)]).T).T
        xmin, ymin = float(rb[:, 0].min()), float(rb[:, 1].min())
        ow = int(np.ceil(rb[:, 0].max() - xmin)) + 2 * border
        oh = int(np.ceil(rb[:, 1].max() - ymin)) + 2 * border
        if ow <= 0 or oh <= 0:
            return None
        # Translate so the rect's bbox lands at (border, border) in that canvas.
        M[0, 2] += border - xmin
        M[1, 2] += border - ymin
        crop = cv2.warpAffine(img, M, (ow, oh))
        if crop.size == 0:
            return None
        scale = float(p.get("scale", 1.0))
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
        return crop
    except Exception as e:
        print(f"Error executing crop_to_contour: {e}")
        return None


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


_LIGHTING_MODES = [
    ("Gray-world WB", "grayworld"),
    ("Global brightness", "global"),
    ("Flat-field (/ blur)", "flatfield"),
]


def _compute_normalize_lighting(inputs, p, in_space="bgr"):
    """Reduce lighting differences so the SAME object under different light maps to
    similar pixel values (more reproducible clustering across photos):
      grayworld  — scale each channel so its mean equals the overall mean
                   (removes a colour cast + brightness gain);
      global     — one gain so the mean brightness hits a target (gain only);
      flatfield  — divide by a large Gaussian blur (removes a smooth lighting
                   gradient / vignette), then rescale."""
    try:
        img = _as_bgr(inputs[0], in_space).astype(np.float32)
        mode = p.get("mode", "grayworld")
        if mode == "grayworld":
            means = img.reshape(-1, 3).mean(axis=0) + 1e-6
            img *= float(means.mean()) / means
        elif mode == "global":
            img *= 128.0 / (float(img.mean()) + 1e-6)
        else:  # flatfield
            blur = cv2.GaussianBlur(img, (0, 0), max(1.0, float(p.get("radius", 25))))
            img = img / (blur + 1e-6) * float(blur.mean())
        return np.clip(img, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error executing normalize_lighting: {e}")
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


# Color space + per-channel curve colours for the Histogram node (mirrors the
# inspector pane's BGR/HSL toggle).
_HIST_SPACES = [("BGR", "bgr"), ("HLS", "hls")]
_HIST_CH_BGR = {"B": (255, 0, 0), "G": (0, 170, 0), "R": (0, 0, 255),
                "H": (200, 80, 255), "L": (130, 130, 130), "S": (0, 200, 255),
                "Gray": (40, 40, 40)}


def _compute_histogram(inputs, p, in_space="bgr"):
    """Per-channel histogram of the image in the chosen colour space (BGR or HLS),
    optionally Gaussian-smoothed — the same controls as the inspector pane's
    histogram. Output is a HISTOGRAM payload carrying the per-channel curves, their
    names, the space, and per-channel bin counts (Hue is 0..179, so it gets 180
    bins) — so the preview and any downstream consumer know the binning."""
    try:
        img = inputs[0]
        sig = max(0.0, float(p.get("smoothing", 0.0)))
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            data, names, bins, space = _to_gray_u8(img), ["Gray"], [256], "gray"
        elif p.get("color_space", "bgr") == "hls":
            data = cv2.cvtColor(_as_bgr(img, in_space), cv2.COLOR_BGR2HLS)
            names, bins, space = ["H", "L", "S"], [180, 256, 256], "hls"
        else:
            data, names, bins, space = _as_bgr(img, in_space), ["B", "G", "R"], [256, 256, 256], "bgr"
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        hists = []
        for c, nb in enumerate(bins):
            hist = cv2.calcHist([data], [c], None, [nb], [0, nb]).flatten().astype(np.float32)
            if sig > 0:
                if space == "hls" and c == 0:        # Hue is circular: wrap-pad so 0/179 join
                    pad = int(np.ceil(sig * 3)) + 1
                    ext = np.concatenate([hist[-pad:], hist, hist[:pad]])
                    ext = cv2.GaussianBlur(ext.reshape(-1, 1), (0, 0), sig).flatten()
                    hist = ext[pad:pad + nb]
                else:
                    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (0, 0), sig).flatten()
            hists.append(hist)
        return {"hist": hists, "channels": len(hists), "names": names,
                "space": space, "bins": bins}
    except Exception as e:
        print(f"Error executing histogram: {e}")
        return None


def _render_histogram(inputs, output, p):
    """Preview: draw the (already computed and smoothed) per-channel curves, each
    coloured by its channel and scaled to its own maximum."""
    if not isinstance(output, dict):
        return None
    hists = output.get("hist", [])
    names = output.get("names") or (["B", "G", "R"] if len(hists) == 3 else ["Gray"])
    h, w = 240, 360
    canvas = np.full((h, w, 3), 255, np.uint8)
    for hist, name in zip(hists, names):
        arr = np.asarray(hist, np.float32).flatten()
        n = len(arr)
        if n == 0:
            continue
        mx = float(arr.max()) or 1.0
        color = _HIST_CH_BGR.get(name, (0, 0, 0))
        prev = None
        for b in range(n):
            x = int(b / (n - 1) * (w - 1)) if n > 1 else 0
            y = int((h - 1) - (arr[b] / mx) * (h - 5))
            if prev is not None:
                cv2.line(canvas, prev, (x, y), color, 1, cv2.LINE_AA)
            prev = (x, y)
    return canvas


def _summary_histogram(output, p):
    if not isinstance(output, dict):
        return {}
    return {"channels": int(output.get("channels", 0)), "space": output.get("space", "")}


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
# Channel for Auto Cluster's peak detection. Values are HLS channel indices.
_CLUSTER_CHANNELS = [
    ("Luminance (L)", 1),
    ("Hue (H)", 0),
    ("Saturation (S)", 2),
]
# Short HLS channel names, keyed by index — used to caption the peak histogram.
_CLUSTER_CHANNEL_NAMES = {0: "Hue", 1: "Luminance", 2: "Saturation"}
# Feature space the clustering distance is measured in. Lab/HLS separate a
# luminance channel that the "Luminance weight" param can down-weight for
# lighting-stable, chroma-driven clusters. "bgr" = cluster on raw input pixels.
_CLUSTER_SPACES = [
    ("BGR (as-is)", "bgr"),
    ("Lab", "lab"),
    ("HLS", "hls"),
]
# How Auto Cluster picks k.
_KMETHODS = [
    ("Histogram peaks", "peaks"),
    ("Elbow (3D, data-driven)", "elbow"),
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
            ParamSpec("filename", "", kind="path",
                      help="Path to write to; used only when 'Use custom filename' is on."),
            ParamSpec("use_custom", False, kind="bool", label="Use custom filename",
                      help="On: write to the path above. Off: auto-name each output in ./output."),
        ],
        compute=_compute_noop, color=(76, 175, 80), out_space="passthrough",
        in_label="Mat (Any)", out_label="File",
        description="Write the incoming image (or each batch element) to ./output "
                    "on a committed evaluation. A view-layer side effect.",
    ),
    Operation(
        id="export_code", label="Export Code", category="Input/Output",
        inputs=[Port("in", datatypes.ANY)], outputs=[Port("out")],
        params=[],
        compute=_compute_noop, color=(96, 125, 139), out_space="passthrough",
        in_label="Mat (Any)", out_label="Pseudocode",
        description="Introspection node: walks the pipeline upstream from here and "
                    "generates language-neutral pseudocode (source -> ops -> params) "
                    "you can port to Python/C++. Shown in the Inspector and written "
                    "to ./output on commit.",
    ),
    Operation(
        id="create_batch", label="Create Batch", category="Input/Output",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[], compute=_compute_create_batch, color=(121, 134, 203),
        in_label="Mat (any) ×N", out_label="Batch (BGR)",
        out_space="bgr", variadic=True, raw=True,
    ),
    Operation(
        id="to_grayscale", label="To Grayscale", category="Color Spaces",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE_GRAY)],
        params=[], compute=_compute_to_grayscale, color=(96, 96, 96),
        in_label="Mat (any)", out_label="Mat (Gray)",
        out_space="gray", space_aware=True,
    ),
    Operation(
        id="to_bgr", label="To BGR", category="Color Spaces",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE_BGR)],
        params=[], compute=_compute_to_bgr, color=(33, 150, 243),
        in_label="Mat (any)", out_label="Mat (BGR)",
        out_space="bgr", space_aware=True,
    ),
    Operation(
        id="blur", label="Blur", category="Filtering & Morphology",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("kernel_size", 15, kind="int", min=1, max=101, step=2, odd=True,
                          label="Kernel Size",
                          help="Averaging window; larger = stronger, blockier blur (odd values).")],
        compute=_compute_blur, color=(156, 39, 176), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="threshold", label="Threshold", category="Thresholding",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("threshold_value", 127, kind="int", min=0, max=255, label="Threshold Value",
                      help="Cut level: pixels above it pass (become Max Value), below become 0."),
            ParamSpec("max_value", 255, kind="int", min=0, max=255, label="Max Value",
                      help="Value assigned to pixels that pass the threshold."),
            ParamSpec("threshold_type", cv2.THRESH_BINARY, kind="enum",
                      choices=_THRESH_TYPES, label="Threshold Type",
                      help="How the test maps to output: binary, inverted, truncate, or to-zero."),
        ],
        compute=_compute_threshold, color=(255, 152, 0),
        in_label="Mat (Gray)", out_label="Mat (Binary/Gray)",
    ),
    Operation(
        id="adaptive_threshold", label="Adaptive Threshold", category="Thresholding",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("max_value", 255, kind="int", min=1, max=255, label="Max Value",
                      help="Value assigned to pixels that pass the local threshold."),
            ParamSpec("adaptive_method", cv2.ADAPTIVE_THRESH_MEAN_C, kind="enum",
                      choices=_ADAPTIVE_METHODS, label="Adaptive Method",
                      help="Local threshold = plain mean or Gaussian-weighted mean of the block."),
            ParamSpec("threshold_type", cv2.THRESH_BINARY, kind="enum",
                      choices=_ADAPTIVE_THRESH_TYPES, label="Threshold Type",
                      help="Binary or inverted-binary output."),
            ParamSpec("block_size", 11, kind="int", min=3, max=51, step=2, odd=True,
                      label="Block Size",
                      help="Neighbourhood size for the local threshold; larger = coarser (odd)."),
            ParamSpec("c", 2, kind="int", min=-10, max=10, label="C Value",
                      help="Constant subtracted from the local mean; higher = fewer foreground pixels."),
        ],
        compute=_compute_adaptive_threshold, color=(255, 152, 0),
        in_label="Mat (Gray)", out_label="Mat (Binary)",
    ),
    Operation(
        id="auto_threshold", label="Auto Threshold", category="Thresholding",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE_BINARY)],
        params=[
            ParamSpec("method", "otsu", kind="enum", choices=_AUTO_THRESH_METHODS, label="Method",
                      help="How the cut level is chosen automatically: Otsu (maximises "
                           "between-class variance), Triangle (geometric), or Valley "
                           "(deepest dip between the two largest histogram modes)."),
            ParamSpec("invert", False, kind="bool", label="Invert",
                      help="Output the inverse mask (foreground and background swapped)."),
        ],
        compute=_compute_auto_threshold, color=(255, 152, 0), out_space="binary",
        in_label="Mat (Gray)", out_label="Mat (Binary)",
        description="Threshold a grayscale image at an automatically chosen level "
                    "(Otsu / Triangle / Valley) — no manual threshold value needed.",
    ),
    Operation(
        id="color_mask", label="Color Mask", category="Thresholding",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[
            ParamSpec("blue", 255, kind="int", min=0, max=255, label="Blue",
                      help="Blue value of the reference colour to match."),
            ParamSpec("green", 255, kind="int", min=0, max=255, label="Green",
                      help="Green value of the reference colour to match."),
            ParamSpec("red", 255, kind="int", min=0, max=255, label="Red",
                      help="Red value of the reference colour to match."),
            ParamSpec("delta", 30, kind="int", min=0, max=255, label="Delta",
                      help="Tolerance around the reference colour; larger matches more pixels."),
            ParamSpec("select", "outside", kind="enum", choices=_COLOR_SELECT, label="Keep",
                      help="Keep pixels Outside (foreground) or Inside (near the colour) the band."),
        ],
        compute=_compute_color_mask, color=(120, 144, 156), out_space="binary",
        space_aware=True, in_label="Mat (any)", out_label="Mat (Binary)",
        description="Binary mask from a background colour +/- delta (cv2.inRange). "
                    "'Outside' keeps the foreground (pixels not near the colour) — "
                    "drop a background to 0 ahead of Find Contours.",
    ),
    Operation(
        id="gaussian_blur", label="Gaussian Blur", category="Filtering & Morphology",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("kernel_size", 5, kind="int", min=1, max=51, step=2, odd=True,
                      label="Kernel Size",
                      help="Gaussian window size; larger = smoother (odd values)."),
            ParamSpec("sigma", 0.0, kind="float", min=0.0, max=10.0, step=0.5, label="Sigma",
                      help="Gaussian spread; 0 derives it from the kernel size. Higher = stronger blur."),
        ],
        compute=_compute_gaussian_blur, color=(103, 58, 183), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="morphology", label="Morphology", category="Filtering & Morphology",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("operation", cv2.MORPH_ERODE, kind="enum",
                      choices=_MORPH_OPS, label="Operation",
                      help="How the structuring element reshapes blobs: erode/dilate/open/close/etc."),
            ParamSpec("kernel_size", 3, kind="int", min=1, max=31, label="Kernel Size",
                      help="Structuring-element size; larger = stronger effect."),
            ParamSpec("iterations", 1, kind="int", min=1, max=10, label="Iterations",
                      help="Times to repeat the operation; more = stronger."),
        ],
        compute=_compute_morphology, color=(96, 125, 139), out_space="passthrough",
        in_label="Mat (Binary/Gray)", out_label="Mat (Binary/Gray)",
    ),
    Operation(
        id="canny", label="Canny Edges", category="Edges & Gradients",
        inputs=[Port("in")], outputs=[Port("out", datatypes.IMAGE_BINARY)],
        params=[
            ParamSpec("threshold1", 100, kind="int", min=0, max=500, label="Threshold 1",
                      help="Lower hysteresis threshold; weak edges touching strong ones are kept."),
            ParamSpec("threshold2", 200, kind="int", min=0, max=500, label="Threshold 2",
                      help="Upper hysteresis threshold; gradients above it start a strong edge."),
            ParamSpec("aperture", 3, kind="int", show=False),
        ],
        compute=_compute_canny, color=(255, 193, 7),
        in_label="Mat (Gray)", out_label="Mat (Binary edges)",
    ),
    Operation(
        id="sobel", label="Sobel", category="Edges & Gradients",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("dx", 1, kind="int", min=0, max=2, label="dx (x order)",
                      help="Order of the x-derivative; 1 detects vertical edges, 0 = none."),
            ParamSpec("dy", 0, kind="int", min=0, max=2, label="dy (y order)",
                      help="Order of the y-derivative; 1 detects horizontal edges, 0 = none."),
            ParamSpec("ksize", 3, kind="int", min=1, max=7, step=2, odd=True, label="Kernel Size",
                      help="Sobel kernel size; larger = smoother, thicker gradients (odd)."),
        ],
        compute=_compute_sobel, color=(255, 193, 7),
        in_label="Mat (Gray)", out_label="Mat (Gray gradient)",
    ),
    Operation(
        id="laplacian", label="Laplacian", category="Edges & Gradients",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("ksize", 3, kind="int", min=1, max=31, step=2, odd=True,
                          label="Kernel Size",
                          help="2nd-derivative kernel size; larger = smoother, thicker response (odd).")],
        compute=_compute_laplacian, color=(255, 193, 7),
        in_label="Mat (Gray)", out_label="Mat (Gray gradient)",
    ),
    Operation(
        id="normalize", label="Normalize", category="Intensity & Enhancement",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[ParamSpec("mode", "stretch", kind="choice",
                          choices=_NORMALIZE_MODES, label="Mode",
                          help="Stretch (min-max), histogram Equalize, or CLAHE (local contrast).")],
        compute=_compute_normalize, color=(0, 121, 107), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="invert", label="Invert", category="Intensity & Enhancement",
        inputs=[Port("in")], outputs=[Port("out")], params=[],
        compute=_compute_invert, color=(69, 90, 100), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="normalize_lighting", label="Normalize Lighting", category="Intensity & Enhancement",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[
            ParamSpec("mode", "grayworld", kind="enum", choices=_LIGHTING_MODES, label="Mode",
                      help="Gray-world white balance, global brightness, or flat-field correction."),
            ParamSpec("radius", 25, kind="int", min=3, max=200, label="Flat-field radius",
                      help="Illumination blur scale (flat-field mode only); larger = gentler correction."),
        ],
        compute=_compute_normalize_lighting, color=(0, 121, 107), out_space="bgr",
        space_aware=True, in_label="Mat (any)", out_label="Mat (BGR)",
        description="Reduce lighting differences (gray-world white balance, global "
                    "brightness, or flat-field) so the same object under different "
                    "light clusters consistently. Put it before K-Means / Auto Cluster.",
    ),
    Operation(
        id="local_hdr", label="Local HDR", category="Intensity & Enhancement",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("radius", 25, kind="int", min=2, max=120, label="Radius",
                      help="Neighbourhood radius for the local mean/std; larger = broader tone equalisation."),
            ParamSpec("amplitude", 35, kind="int", min=5, max=100, label="Detail strength",
                      help="Local-contrast boost; higher amplifies fine detail more."),
            ParamSpec("strength", 1.0, kind="float", min=0.0, max=1.0, step=0.05, label="Strength",
                      help="Blend with the original; 0 = unchanged, 1 = full local normalisation."),
        ],
        compute=_compute_local_hdr, color=(255, 138, 0), out_space="passthrough",
        in_label="Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="mser", label="MSER", category="Regions & Contours",
        inputs=[Port("in")], outputs=[Port("out")],
        params=[
            ParamSpec("delta", 5, kind="int", min=1, max=20, label="Delta",
                      help="Intensity step for the stability test; higher = fewer, more stable regions."),
            ParamSpec("min_area", 60, kind="int", min=10, max=1000, label="Min Area",
                      help="Smallest region (px) to keep."),
            ParamSpec("max_area", 14400, kind="int", min=1000, max=50000, label="Max Area",
                      help="Largest region (px) to keep."),
            ParamSpec("max_variation", 0.25, kind="float", min=0.0, max=1.0, step=0.01,
                      label="Max Variation",
                      help="Reject regions whose area varies more than this (less stable)."),
            ParamSpec("min_diversity", 0.2, kind="float", min=0.0, max=1.0, step=0.01,
                      label="Min Diversity",
                      help="Prune nested regions too similar to their parent; higher = fewer duplicates."),
            ParamSpec("max_evolution", 200, kind="int", show=False),
            ParamSpec("area_threshold", 1.01, kind="float", show=False),
            ParamSpec("min_margin", 0.003, kind="float", show=False),
            ParamSpec("edge_blur_size", 5, kind="int", show=False),
        ],
        compute=_compute_mser, color=(34, 139, 34),
        in_label="Mat (Gray)", out_label="Mat (BGR)",
    ),
    Operation(
        id="sum", label="Sum", category="Arithmetic",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")],
        params=[ParamSpec("alpha", 0.5, kind="float", min=0.0, max=1.0, step=0.01,
                          label="Alpha (Weight)",
                          help="Blend weight of input A; B gets (1 - alpha). 0.5 = equal mix.")],
        compute=_compute_sum, color=(128, 0, 128), out_space="passthrough",
        in_label="Mat (BGR/Gray) + Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="and", label="AND", category="Arithmetic",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")], params=[],
        compute=_compute_and, color=(0, 0, 139), out_space="passthrough",
        in_label="Mat (BGR/Gray) & Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    Operation(
        id="diff", label="Diff", category="Arithmetic",
        inputs=[Port("a"), Port("b")], outputs=[Port("out")], params=[],
        compute=_compute_diff, color=(220, 20, 60), out_space="passthrough",
        in_label="Mat (BGR/Gray) - Mat (BGR/Gray)", out_label="Mat (BGR/Gray)",
    ),
    # --- Color & Clustering ------------------------------------------------
    Operation(
        id="to_hls", label="To HSL", category="Color Spaces",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.IMAGE)], params=[],
        compute=_compute_to_hls, color=(255, 87, 34),
        in_label="Mat (any)", out_label="Mat (HLS)",
        out_space="hls", space_aware=True,
    ),
    Operation(
        id="kmeans", label="K-Means Cluster", category="Color Quantization",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CLUSTERS)],
        params=[
            ParamSpec("k", 6, kind="int", min=2, max=16, label="Clusters (k)",
                      help="Number of colour clusters to quantize the image into."),
            ParamSpec("cluster_space", "bgr", kind="enum", choices=_CLUSTER_SPACES,
                      label="Cluster space",
                      help="Distance space for clustering: BGR, Lab, or HLS (Lab/HLS split chroma from luminance)."),
            ParamSpec("lum_weight", 1.0, kind="float", min=0.0, max=2.0, step=0.1,
                      label="Luminance weight",
                      help="Scales luminance in Lab/HLS; <1 = chroma-driven, more lighting-stable."),
            ParamSpec("attempts", 5, kind="int", min=1, max=10, show=False),
        ],
        compute=_compute_kmeans, color=(0, 150, 136), out_space="passthrough",
        space_aware=True, in_label="Mat (any)", out_label="Clusters",
        render_preview=_render_kmeans, summary=_summary_kmeans, preview_is_chart=True,
    ),
    Operation(
        id="auto_cluster", label="Auto Cluster", category="Color Quantization",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CLUSTERS)],
        params=[
            ParamSpec("max_k", 12, kind="int", min=2, max=24, label="Max clusters",
                      help="Upper bound on the auto-detected cluster count k."),
            ParamSpec("k_method", "peaks", kind="enum", choices=_KMETHODS,
                      label="k detection",
                      help="How k is chosen: count histogram peaks, or the k-means inertia elbow."),
            ParamSpec("channel", 1, kind="enum", choices=_CLUSTER_CHANNELS,
                      label="Peak channel", enabled_if=("k_method", "peaks"),
                      help="HLS channel whose histogram peaks are counted (Hue / Luminance / Saturation)."),
            ParamSpec("smoothing", 4.0, kind="float", min=0.5, max=15.0, step=0.5,
                      label="Histogram smoothing", enabled_if=("k_method", "peaks"),
                      help="Gaussian blur of the histogram before peak finding; higher = fewer peaks."),
            ParamSpec("min_prominence", 0.3, kind="float", min=0.0, max=1.0, step=0.01,
                      label="Min peak prominence", enabled_if=("k_method", "peaks"),
                      help="How far a peak must rise above the MEAN of its two "
                           "surrounding valleys (fraction of its own height) to count "
                           "as a colour — lenient enough to catch a small feature or a "
                           "sub-peak of a 'mountain range'. A peak must also dip on "
                           "both sides, so a quasi-flat step isn't counted. Higher = "
                           "stricter; raise 'smoothing' to drop noise."),
            ParamSpec("sat_weight", 1.0, kind="float", min=0.0, max=4.0, step=0.1,
                      label="Chroma weight",
                      enabled_if=[("k_method", "peaks"), ("channel", 0)],
                      help="Exponent on chroma (max−min of BGR) in the hue histogram; "
                           ">1 favours vivid pixels, 0 ignores chroma. Chroma is used "
                           "instead of HLS saturation because HLS S wrongly reads "
                           "near-white/near-black pixels as fully saturated."),
            ParamSpec("k_bias", 0, kind="int", min=-5, max=5, label="k nudge (elbow)",
                      enabled_if=("k_method", "elbow"),
                      help="Shift the auto-detected k by this many clusters relative to "
                           "the inertia knee: + for more clusters, − for fewer; 0 = the "
                           "plain knee. Clamped to [2, Max clusters]."),
            ParamSpec("separate_achromatic", False, kind="bool", label="Separate gray/white/black",
                      help="Cluster achromatic pixels (low chroma — white, gray AND black) "
                           "separately by lightness, so they don't fall into the coloured "
                           "clusters. The detected k then counts only the colourful clusters."),
            ParamSpec("chroma_min", 20, kind="int", min=0, max=128, label="Chroma threshold",
                      enabled_if=("separate_achromatic", True),
                      help="Pixels with chroma (max−min of BGR) below this are treated as "
                           "achromatic. ~20 ≈ 8%; raise it to pull more washed-out pixels out."),
            ParamSpec("gray_levels", 2, kind="int", min=1, max=6, label="Gray levels",
                      enabled_if=("separate_achromatic", True),
                      help="How many achromatic clusters to split by lightness (e.g. 2 = "
                           "dark/light, 3 = black/gray/white)."),
            ParamSpec("cluster_space", "bgr", kind="enum", choices=_CLUSTER_SPACES,
                      label="Cluster space",
                      help="Distance space for k-means: BGR, Lab, or HLS (Lab/HLS split chroma from luminance)."),
            ParamSpec("lum_weight", 1.0, kind="float", min=0.0, max=2.0, step=0.1,
                      label="Luminance weight",
                      help="Scales luminance in Lab/HLS; <1 = chroma-driven, more lighting-stable."),
        ],
        compute=_compute_auto_cluster, color=(0, 150, 136), out_space="passthrough",
        space_aware=True, in_label="Mat (any)", out_label="Clusters (auto k)",
        render_preview=_render_auto_cluster, summary=_summary_kmeans, preview_is_chart=True,
    ),
    Operation(
        id="hdbscan_cluster", label="Density Cluster", category="Color Quantization",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CLUSTERS)],
        params=[
            ParamSpec("algorithm", "hdbscan", kind="enum", choices=_HDBSCAN_ALGOS,
                      label="Algorithm",
                      help="HDBSCAN/OPTICS are exact (best colour recall; HDBSCAN leaves edge "
                           "pixels as noise, OPTICS-Xi labels every pixel). sHDBSCAN/sOPTICS are "
                           "approximate (only faster above ~100k unique colours; otherwise exact "
                           "HDBSCAN is both faster and better)."),
            ParamSpec("min_cluster_size", 50, kind="int", min=2, max=20000, log=True,
                      label="Min cluster size",
                      help="Smallest group of pixels called a colour cluster — the main knob (no k "
                           "needed). Larger = fewer, broader colours; smaller = more, finer. Roughly "
                           "tracks image size (the library dedups, so it stays in pixel units)."),
            ParamSpec("min_cluster_frac", 0.003, kind="float", min=0.0, max=0.2, step=0.001,
                      label="Min region size",
                      help="Clusters smaller than this fraction of the image become noise — the most "
                           "intuitive 'how many colours' control. Bigger = fewer, broader colours."),
            ParamSpec("color_space", "lab", kind="enum", choices=_HDBSCAN_SPACES,
                      label="Colour space",
                      help="Distance space: Lab (perceptual CIELAB — the study's pick; separates "
                           "orange/red, brown/red that RGB muddles) or raw RGB. The library does the "
                           "sRGB→Lab conversion internally."),
            ParamSpec("voxel_bin", 2, kind="int", min=0, max=8, label="Voxel bin",
                      help="Snap colours to a grid before clustering (speed/quality knob; the library "
                           "also dedups). 0 = off; ~2 for Lab / ~4 for RGB is the sweet spot. ≥8 "
                           "over-merges and can collapse everything to one cluster."),
            ParamSpec("metric", "l2", kind="enum", choices=_HDBSCAN_METRICS,
                      label="Metric (approx only)", enabled_if=("algorithm", ("shdbscan", "soptics")),
                      help="Distance for the approximate variants. L2 is correct for colour; cosine "
                           "clusters by hue and merges black/white/gray (rarely wanted)."),
            ParamSpec("seed", 42, kind="int", min=0, max=9999, label="Seed (approx only)",
                      enabled_if=("algorithm", ("shdbscan", "soptics")),
                      help="The approximate variants are randomized but reproducible: the same seed "
                           "always gives the same clustering."),
            ParamSpec("noise_handling", "nearest", kind="enum", choices=_NOISE_MODES,
                      label="Noise pixels",
                      help="Density clustering labels sparse/edge pixels as noise. 'Assign to nearest "
                           "cluster' folds them into the closest colour (a usable quantization, like "
                           "K-Means); 'Flag colour' paints them magenta so you can SEE what was sparse "
                           "— useful for tuning, but a high-noise result then looks all-pink."),
            ParamSpec("show_reachability", False, kind="bool", label="Show reachability plot",
                      help="Compute the OPTICS reachability plot (the density landscape — valleys "
                           "are clusters) and show it under the preview. Costs an extra OPTICS pass "
                           "on the unique colours, so it's off by default."),
        ],
        compute=_compute_hdbscan, color=(0, 150, 136), out_space="passthrough",
        space_aware=True, in_label="Mat (any)", out_label="Clusters (density)",
        render_preview=_render_hdbscan, summary=_summary_hdbscan, preview_is_chart=True,
        description="Density-based colour clustering (no k) via the OPTICS-Clustering library — "
                    "algorithms OPTICS, HDBSCAN, sOPTICS and sHDBSCAN. Finds colour modes at "
                    "differing densities and labels sparse 'bridge' colours as noise. Best when "
                    "there are well-separated colour modes; for smooth photos prefer K-Means / "
                    "Auto Cluster. Emits a CLUSTERS payload (use Reduce Colors to render).",
    ),
    Operation(
        id="reduce_colors", label="Reduce Colors", category="Color Quantization",
        inputs=[Port("in", datatypes.CLUSTERS)],
        outputs=[Port("out", datatypes.IMAGE)], params=[],
        compute=_compute_reduce_colors, color=(0, 150, 136), out_space="passthrough",
        in_label="Clusters", out_label="Mat (quantized)",
    ),
    Operation(
        id="mean_shift", label="Mean Shift", category="Color Quantization",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE_BGR)],
        params=[
            ParamSpec("spatial", 20, kind="int", min=2, max=50, label="Spatial radius",
                      help="Spatial window radius; larger merges over wider areas."),
            ParamSpec("color", 40, kind="int", min=2, max=100, label="Color radius",
                      help="Colour window radius; larger merges more distinct colours together."),
        ],
        compute=_compute_mean_shift, color=(0, 150, 136), out_space="bgr",
        in_label="Mat (BGR)", out_label="Mat (segmented)",
    ),
    # --- Geometry ----------------------------------------------------------
    Operation(
        id="resize", label="Resize", category="Geometry",
        inputs=[Port("in", datatypes.ANY)], outputs=[Port("out", datatypes.ANY)],
        params=[
            ParamSpec("mode", "scale", kind="enum", choices=_RESIZE_MODES, label="Mode",
                      help="'Scale factor' multiplies the size; 'Longer edge → length' "
                           "scales so the image's longer edge becomes a fixed length "
                           "(to normalize varying input sizes)."),
            ParamSpec("scale", 0.5, kind="float", min=0.1, max=4.0, step=0.05, label="Scale",
                      enabled_if=("mode", "scale"),
                      help="Output size factor; <1 shrinks, >1 enlarges. Scales contour "
                           "coordinates too, to map a downscaled segmentation back to the original."),
            ParamSpec("length", 1024, kind="int", min=16, max=8192, label="Longer-edge length",
                      enabled_if=("mode", "fixed"),
                      help="Target length (px) for the longer edge; the shorter edge scales "
                           "proportionally. Aspect ratio is preserved."),
            ParamSpec("interpolation", cv2.INTER_AREA, kind="enum",
                      choices=_INTERP_MODES, label="Interpolation",
                      help="Resampling: AREA (shrink), LINEAR/CUBIC (enlarge), NEAREST "
                           "(hard edges). Images only — contours scale exactly."),
        ],
        compute=_compute_resize, color=(63, 81, 181), out_space="passthrough",
        in_label="Mat / Contours", out_label="Mat / Contours",
        render_preview=_render_resize,
        description="Scale an image, or a Contours payload (scaling the contour "
                    "coordinates) — so a segmentation done on a downscaled image can "
                    "be mapped back to the full-resolution original.",
    ),
    Operation(
        id="rotate", label="Rotate", category="Geometry",
        inputs=[Port("in", datatypes.IMAGE)], outputs=[Port("out", datatypes.IMAGE)],
        params=[
            ParamSpec("angle", 0, kind="int", min=-180, max=180, label="Angle (deg)",
                      help="Rotation in degrees (counter-clockwise)."),
            ParamSpec("expand", False, kind="bool", label="Expand to fit",
                      help="On: grow the canvas so corners aren't clipped. Off: keep original size."),
        ],
        compute=_compute_rotate, color=(63, 81, 181), out_space="passthrough",
        in_label="Mat (any)", out_label="Mat (any)",
    ),
    # --- Contours ----------------------------------------------------------
    Operation(
        id="find_contours", label="Find Contours", category="Regions & Contours",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[
            ParamSpec("mode", cv2.RETR_EXTERNAL, kind="enum",
                      choices=_RETR_MODES, label="Retrieval Mode",
                      help="External (outer contours only) or all/nested contours."),
            ParamSpec("filled", False, kind="bool", label="Draw filled",
                      help="Preview only: fill each contour instead of outlining it."),
        ],
        compute=_compute_find_contours, color=(233, 30, 99),
        in_label="Mat (Binary)", out_label="Contours",
        render_preview=_render_find_contours, summary=_summary_find_contours,
    ),
    Operation(
        id="label_regions", label="Flood Fill", category="Regions & Contours",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[
            ParamSpec("channel", -1, kind="enum", choices=_REGION_CHANNELS,
                      label="Similarity channel",
                      help="Channel(s) compared for region similarity (-1 = all channels)."),
            ParamSpec("delta", 8, kind="int", min=0, max=64, label="Color delta",
                      help="Max value difference for pixels to join a region; higher = larger regions."),
            ParamSpec("connectivity", 4, kind="enum", choices=_CONNECTIVITY,
                      label="Connectivity",
                      help="4- or 8-neighbour adjacency when growing regions."),
            ParamSpec("filled", False, kind="bool", label="Draw filled",
                      help="Preview only: fill each region instead of outlining it."),
        ],
        compute=_compute_label_regions, color=(233, 30, 99), space_aware=True,
        in_label="Mat (any)", out_label="Contours (regions)",
        render_preview=_render_find_contours, summary=_summary_find_contours,
    ),
    Operation(
        id="connected_components", label="Connected Components", category="Regions & Contours",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[
            ParamSpec("connectivity", 8, kind="enum", choices=_CONNECTIVITY,
                      label="Connectivity",
                      help="4- or 8-neighbour adjacency for grouping foreground pixels."),
            ParamSpec("filled", False, kind="bool", label="Draw filled",
                      help="Preview only: fill each blob instead of outlining it."),
        ],
        compute=_compute_connected_components, color=(233, 30, 99),
        in_label="Mat (Binary)", out_label="Contours (regions)",
        render_preview=_render_find_contours, summary=_summary_find_contours,
        description="Label connected foreground blobs in a binary image "
                    "(cv2.connectedComponents, 4/8-neighbourhood). Foreground is "
                    "any non-zero pixel; each blob becomes one region/contour.",
    ),
    Operation(
        id="contour_filter", label="Filter Contours", category="Regions & Contours",
        inputs=[Port("in", datatypes.CONTOURS)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[
            ParamSpec("min_area", 50, kind="int", min=0, max=20000, label="Min Area", log=True, live=True,
                      help="Drop contours smaller than this area (px)."),
            ParamSpec("max_area", 100000, kind="int", min=0, max=1000000, label="Max Area", log=True, live=True,
                      help="Drop contours larger than this area (px)."),
            ParamSpec("filled", False, kind="bool", label="Draw filled",
                      help="Preview only: fill the kept contours instead of outlining them."),
        ],
        compute=_compute_filter_contours, color=(233, 30, 99),
        in_label="Contours", out_label="Contours",
        render_preview=_render_filter_contours, summary=_summary_filter_contours,
    ),
    Operation(
        id="largest_contour", label="Largest Contour", category="Regions & Contours",
        inputs=[Port("in", datatypes.CONTOURS)],
        outputs=[Port("out", datatypes.CONTOURS)],
        params=[ParamSpec("count", 1, kind="int", min=1, max=20, label="Keep N largest",
                          help="How many of the biggest contours (by area) to keep.")],
        compute=_compute_largest_contour, color=(233, 30, 99),
        in_label="Contours", out_label="Contours",
        render_preview=_render_largest_contour, summary=_summary_filter_contours,
        description="Keep only the N largest contours by area (default 1).",
    ),
    Operation(
        id="crop_to_contour", label="Deskew & Crop", category="Regions & Contours",
        inputs=[Port("image", datatypes.IMAGE), Port("contours", datatypes.CONTOURS)],
        outputs=[Port("out", datatypes.IMAGE)],
        params=[
            ParamSpec("border", 0, kind="int", min=-100, max=100, label="Border (px)",
                      help="Padding (px) around the cropped box; negative trims inward "
                           "(crops tighter than the contour's box)."),
            ParamSpec("scale", 1.0, kind="float", min=0.1, max=4.0, step=0.05, label="Scale",
                      help="Output size factor for the crop."),
        ],
        compute=_compute_crop_to_contour, color=(63, 81, 181), out_space="bgr",
        space_aware=True, in_label="Mat + Contours", out_label="Mat (crop)",
        description="Rotate the image so the largest contour's min-area box is "
                    "upright, then crop to it with a border and optional scale.",
    ),
    # --- Fourier -----------------------------------------------------------
    Operation(
        id="dft", label="DFT", category="Frequency (Fourier)",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.SPECTRUM)], params=[],
        compute=_compute_dft, color=(121, 85, 72),
        in_label="Mat (Gray)", out_label="Spectrum",
        render_preview=_render_dft, summary=_summary_dft,
    ),
    Operation(
        id="idft", label="Inverse DFT", category="Frequency (Fourier)",
        inputs=[Port("in", datatypes.SPECTRUM)],
        outputs=[Port("out", datatypes.IMAGE_FLOAT)], params=[],
        compute=_compute_idft, color=(121, 85, 72),
        in_label="Spectrum", out_label="Mat (Float)",
    ),
    # --- Analysis ----------------------------------------------------------
    Operation(
        id="histogram", label="Histogram", category="Analysis",
        inputs=[Port("in", datatypes.IMAGE)],
        outputs=[Port("out", datatypes.HISTOGRAM)],
        params=[
            ParamSpec("color_space", "bgr", kind="enum", choices=_HIST_SPACES,
                      label="Color space",
                      help="Histogram the BGR channels or HLS (Hue / Lum / Sat). "
                           "Ignored for a single-channel (gray) input."),
            ParamSpec("smoothing", 0.0, kind="float", min=0.0, max=15.0, step=0.5,
                      label="Smoothing",
                      help="Gaussian smoothing of the histogram curve; 0 = off."),
        ],
        compute=_compute_histogram, color=(0, 188, 212), space_aware=True,
        in_label="Mat (any)", out_label="Histogram",
        render_preview=_render_histogram, summary=_summary_histogram, preview_is_chart=True,
    ),
    Operation(
        id="backproject", label="Backproject", category="Analysis",
        inputs=[Port("image", datatypes.IMAGE), Port("hist", datatypes.HISTOGRAM)],
        outputs=[Port("out", datatypes.IMAGE_GRAY)],
        params=[
            ParamSpec("chroma_only", True, kind="bool", label="Chroma only (HLS)",
                      help="For an HLS histogram model, match on Hue + Saturation only "
                           "(ignore Luminance) for lighting-robust colour matching."),
        ],
        compute=_compute_backproject, color=(0, 188, 212), out_space="gray",
        space_aware=True, in_label="Mat + Histogram", out_label="Mat (likelihood)",
        description="Histogram backprojection: project a Histogram-node model onto a "
                    "target image to get a likelihood map (bright where the image "
                    "matches the modelled colour/intensity distribution) — for "
                    "colour-based segmentation or tracking. Feed it into Threshold.",
    ),
]

# Categories shown in the sidebar, grouped by intent (roughly the order a
# pipeline flows: load -> colour/filter/threshold -> arithmetic/geometry ->
# segmentation/regions -> analysis/output).
CATEGORY_ORDER = [
    "Input/Output",
    "Color Spaces",
    "Filtering & Morphology",
    "Edges & Gradients",
    "Thresholding",
    "Intensity & Enhancement",
    "Arithmetic",
    "Geometry",
    "Color Quantization",
    "Regions & Contours",
    "Frequency (Fourier)",
    "Analysis",
]

REGISTRY = {op.id: op for op in OPS}
by_label = {op.label: op for op in OPS}

ops_by_category: dict = {cat: [] for cat in CATEGORY_ORDER}
for _op in OPS:
    ops_by_category.setdefault(_op.category, []).append(_op)
