"""Port data types and connection compatibility (backend, Qt-free).

Ports carry a data *type* so connections can be validated. Today every
operation produces/consumes images, so image types interconvert freely (the
ops tolerate/convert BGR<->Gray internally). The non-image types exist for the
operations introduced later (FindContours, Histogram, KMeans, …), where, e.g.,
a contour list must not be fed into an image input.
"""

# Image-ish types (all mutually connectable for now).
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
