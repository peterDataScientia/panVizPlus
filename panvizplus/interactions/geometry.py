"""Geometry helpers for interaction detection."""

from __future__ import annotations

from math import acos, degrees, sqrt

from panvizplus.chemistry.models import Atom


def atom_distance(a: Atom, b: Atom) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def angle_degrees(a: Atom, vertex: Atom, c: Atom) -> float | None:
    v1 = (a.x - vertex.x, a.y - vertex.y, a.z - vertex.z)
    v2 = (c.x - vertex.x, c.y - vertex.y, c.z - vertex.z)
    n1 = sqrt(sum(x * x for x in v1))
    n2 = sqrt(sum(x * x for x in v2))
    if n1 == 0.0 or n2 == 0.0:
        return None
    cosine = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))
