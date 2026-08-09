"""Geometric helpers for zone tests and motion analysis.

Kept dependency-light on purpose: everything here works on plain (x, y) pixel
tuples so these functions can be unit-tested without OpenCV or a model loaded.
Polygons are lists of (x, y) vertices.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def centroid_of_bbox(bbox: Sequence[float]) -> Point:
    """Center point of an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test. True if the point is inside."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Signed area sign of triangle abc: >0 CCW, <0 CW, 0 colinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1->p2 crosses segment p3->p4 (proper intersection)."""
    d1 = _orientation(p3, p4, p1)
    d2 = _orientation(p3, p4, p2)
    d3 = _orientation(p1, p2, p3)
    d4 = _orientation(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def side_of_line(point: Point, line_a: Point, line_b: Point) -> float:
    """Signed side of `point` relative to the directed line a->b."""
    return _orientation(line_a, line_b, point)


def unit_vector(v: Point) -> Point:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]
