"""Port data types and connection compatibility (backend, Qt-free).

Ports carry a data *type* so connections can be validated. Image types
interconvert freely (ops tolerate/convert BGR<->Gray internally), while the
non-image payloads below (contours, histograms, labels, clusters, spectra)
flow only between the ops that understand them — e.g. a contour list must not
be fed into an image input. ``ANY`` is a wildcard for polymorphic nodes (Save
to File, Resize); see :func:`compatible`.
"""

# Image-ish types (all interconvertible — ops convert between them as needed).
IMAGE = "image"            # generic / unknown image
IMAGE_BGR = "image_bgr"
IMAGE_GRAY = "image_gray"
IMAGE_BINARY = "image_binary"
IMAGE_FLOAT = "image_float"

# Non-image payloads (introduced with the ops that produce them).
CONTOURS = "contours"
HISTOGRAM = "histogram"
LABELS = "labels"
SCALAR = "scalar"
CLUSTERS = "clusters"   # k-means result: centers + per-pixel labels + shape
CENTERS = "centers"     # detected colour-cluster seeds (Lab + display BGR), no labels
SPECTRUM = "spectrum"   # complex DFT result (carried between DFT and inverse DFT)

IMAGE_TYPES = frozenset({IMAGE, IMAGE_BGR, IMAGE_GRAY, IMAGE_BINARY, IMAGE_FLOAT})

ANY = "any"   # accepts any output (e.g. Save to File, which can save a preview)


def compatible(out_type: str, in_type: str) -> bool:
    """Can an output of ``out_type`` feed an input expecting ``in_type``?"""
    if in_type == ANY or out_type == ANY:
        # ANY is a wildcard both ways: an ANY input accepts any output, and an ANY
        # output (a polymorphic node like Resize, whose result type mirrors its
        # input — image or contours) can feed any input. The receiving op validates
        # the actual payload at compute time.
        return True
    if out_type == in_type:
        return True
    # Images are permissive: operations convert between BGR/Gray/binary/float.
    if out_type in IMAGE_TYPES and in_type in IMAGE_TYPES:
        return True
    return False
