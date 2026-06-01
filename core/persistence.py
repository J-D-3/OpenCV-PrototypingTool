"""Save/load a GraphModel to a plain dict / JSON (backend, Qt-free).

The serialized form is a dict (JSON-serializable):

    {
      "version": 1,
      "nodes": [
        {"id": 0, "op": null, "params": {}, "pos": [x, y], "image": "<base64 png>"},
        {"id": 1, "op": "blur", "params": {"kernel_size": 15}, "pos": [x, y]},
        ...
      ],
      "edges": [{"src": 0, "dst": 1, "port": 0}, ...]
    }

Source nodes (``op == null``) embed their image as base64 PNG so a pipeline is
fully self-contained. Operation nodes store the op id (looked up in the
registry on load) and their parameter values. Node positions are passed in by
the caller (the UI owns them) and round-tripped here.
"""
from __future__ import annotations

import base64
from typing import Dict, Tuple

import cv2
import numpy as np

from core.operations import REGISTRY
from core.graph import GraphModel
from core.batch import Batch


def encode_image(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image(s: str):
    if not s:
        return None
    data = base64.b64decode(s.encode("ascii"))
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def to_dict(model: GraphModel, positions: Dict[int, Tuple[float, float]]) -> dict:
    nodes = []
    for gid, gn in model.nodes.items():
        entry = {
            "id": gid,
            "op": None if gn.is_source else gn.op.id,
            "params": {k: v for k, v in gn.params.items()},
            "pos": list(positions.get(gid, (0.0, 0.0))),
        }
        if gn.is_source and gn.source_image is not None:
            if isinstance(gn.source_image, Batch):
                entry["images"] = [encode_image(im) for im in gn.source_image.items]
            else:
                entry["image"] = encode_image(gn.source_image)
        nodes.append(entry)
    edges = [{"src": e.src.id, "dst": e.dst.id, "port": e.dst_port} for e in model.edges]
    return {"version": 1, "nodes": nodes, "edges": edges}


def from_dict(d: dict):
    """Rebuild a GraphModel from a dict. Returns (model, positions_by_new_id).

    Node ids are reassigned by the new model, so positions are keyed by the new
    id and edges are remapped through the old->new id mapping.
    """
    model = GraphModel()
    id_map = {}
    positions: Dict[int, Tuple[float, float]] = {}

    for n in d.get("nodes", []):
        if n.get("op") is None:
            if "images" in n:
                source = Batch([decode_image(s) for s in n["images"]])
            else:
                source = decode_image(n.get("image", ""))
            gn = model.add_node(op=None, source_image=source)
        else:
            op = REGISTRY.get(n["op"])
            if op is None:
                continue  # unknown operation id — skip rather than crash
            params = op.defaults()
            params.update(n.get("params", {}))
            gn = model.add_node(op=op, params=params)
        id_map[n["id"]] = gn
        pos = n.get("pos", [0.0, 0.0])
        positions[gn.id] = (float(pos[0]), float(pos[1]))

    # Add edges in (dst, port) order so destination ports are assigned correctly.
    for e in sorted(d.get("edges", []), key=lambda e: (e["dst"], e["port"])):
        src = id_map.get(e["src"])
        dst = id_map.get(e["dst"])
        if src is not None and dst is not None:
            model.add_edge(src, dst, e["port"])

    return model, positions
