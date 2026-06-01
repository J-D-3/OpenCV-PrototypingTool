"""Batch — a stack of items flowing along one edge (backend, Qt-free).

When a source provides several images, its output is a ``Batch`` of them. The
engine then maps each operation over the batch element-by-element (single-image
compute code is unchanged), so one chain processes many images. A dedicated
type (rather than a plain list) keeps a batch distinguishable from the list of
inputs the engine passes to ``compute``.
"""
from __future__ import annotations

from typing import List


class Batch:
    def __init__(self, items):
        self.items: List = list(items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return f"Batch({len(self.items)} items)"
