"""Tests for the geometry primitives every rule is built on. No dependencies.

    python tests/test_geometry.py     # or: pytest tests/
"""
from __future__ import annotations

import sys

from _harness import run_tests  # noqa: E402  (also puts src on sys.path)

from src.geometry import (centroid_of_bbox, dot, point_in_polygon,  # noqa: E402
                          segments_intersect, side_of_line, unit_vector)

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
# Chevron shape: the notch means a naive bounding-box test would be wrong.
CONCAVE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 3.0), (0.0, 10.0)]


def test_point_inside_and_outside_a_square():
    assert point_in_polygon((5.0, 5.0), SQUARE) is True
    assert point_in_polygon((15.0, 5.0), SQUARE) is False
    assert point_in_polygon((-1.0, 5.0), SQUARE) is False


def test_concave_polygon_notch_is_outside():
    assert point_in_polygon((5.0, 8.0), CONCAVE) is False, "notch read as inside"
    assert point_in_polygon((5.0, 1.0), CONCAVE) is True


def test_degenerate_polygon_is_never_inside():
    assert point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)]) is False
    assert point_in_polygon((0.0, 0.0), []) is False


def test_crossing_segments_intersect():
    assert segments_intersect((0.0, 0.0), (10.0, 10.0),
                              (0.0, 10.0), (10.0, 0.0)) is True


def test_parallel_and_short_segments_do_not_intersect():
    assert segments_intersect((0.0, 0.0), (10.0, 0.0),
                              (0.0, 5.0), (10.0, 5.0)) is False
    # Stops short of the line it would otherwise cross.
    assert segments_intersect((0.0, 0.0), (1.0, 1.0),
                              (5.0, 0.0), (5.0, 10.0)) is False


def test_side_of_line_is_signed_and_opposite_across_the_line():
    a, b = (0.0, 5.0), (10.0, 5.0)
    above = side_of_line((5.0, 0.0), a, b)
    below = side_of_line((5.0, 10.0), a, b)
    assert above * below < 0, "points either side must have opposite signs"
    assert side_of_line((5.0, 5.0), a, b) == 0, "a point on the line is side 0"


def test_unit_vector_normalises_and_survives_zero_length():
    ux, uy = unit_vector((3.0, 4.0))
    assert abs((ux ** 2 + uy ** 2) ** 0.5 - 1.0) < 1e-9
    # A stationary track yields a zero vector; this must not divide by zero.
    assert unit_vector((0.0, 0.0)) == (0.0, 0.0)


def test_dot_encodes_direction_agreement():
    assert dot(unit_vector((0.0, 1.0)), unit_vector((0.0, 1.0))) == 1.0
    assert dot(unit_vector((0.0, 1.0)), unit_vector((0.0, -1.0))) == -1.0
    assert abs(dot(unit_vector((1.0, 0.0)), unit_vector((0.0, 1.0)))) < 1e-9


def test_centroid_of_bbox():
    assert centroid_of_bbox((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals()), "geometry"))
