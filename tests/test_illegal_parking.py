"""Tests for IllegalParkingRule. No dependencies required.

    python tests/test_illegal_parking.py     # or: pytest tests/
"""
from __future__ import annotations

import sys

from _harness import FakeZone, run_tests, track_at        # noqa: E402

from src.violations.base import FrameContext              # noqa: E402
from src.violations.illegal_parking import IllegalParkingRule  # noqa: E402

FPS = 25.0
ZONE = FakeZone(polygon=[(400.0, 300.0), (800.0, 300.0),
                         (800.0, 520.0), (400.0, 520.0)],
                name="no_parking_1")
INSIDE = (600.0, 400.0)
OUTSIDE = (100.0, 100.0)


def feed(rule, points, tid=1):
    """Push (x, y) positions through the rule at FPS; return violations."""
    fired = []
    for i, (x, y) in enumerate(points):
        ctx = FrameContext(frame_idx=i, timestamp=i / FPS, fps=FPS)
        fired.extend(rule.update([track_at(x, y, tid=tid)], ctx))
    return fired


def hold(point, seconds):
    return [point] * int(seconds * FPS)


def test_flags_a_vehicle_parked_past_the_threshold():
    fired = feed(IllegalParkingRule([ZONE]), hold(INSIDE, 10.0))
    assert len(fired) == 1, f"expected one violation, got {len(fired)}"
    assert fired[0].type == "illegal_parking"
    # Default min_seconds is 8.0; it must not fire early.
    assert fired[0].timestamp >= 8.0, f"fired early at {fired[0].timestamp:.1f}s"
    assert fired[0].extra["zone"] == "no_parking_1"


def test_does_not_flag_before_the_threshold():
    assert feed(IllegalParkingRule([ZONE]), hold(INSIDE, 6.0)) == []


def test_ignores_a_vehicle_parked_outside_the_zone():
    assert feed(IllegalParkingRule([ZONE]), hold(OUTSIDE, 20.0)) == []


def test_ignores_traffic_driving_through_the_zone():
    """Moving through a no-parking zone is not parking in it."""
    crossing = [(410.0 + 4.0 * i, 400.0) for i in range(int(15.0 * FPS))]
    assert feed(IllegalParkingRule([ZONE]), crossing) == []


def test_dwell_timer_resets_when_the_vehicle_moves_off():
    """Park 6s (under the bar), drive away, come back and park 6s again."""
    points = hold(INSIDE, 6.0)
    points += [(600.0 + 30.0 * i, 400.0) for i in range(1, 8)]   # leaves the zone
    points += hold(INSIDE, 6.0)
    assert feed(IllegalParkingRule([ZONE]), points) == []


def test_one_violation_per_track():
    fired = feed(IllegalParkingRule([ZONE]), hold(INSIDE, 60.0))
    assert len(fired) == 1, f"debounce failed, emitted {len(fired)}"


def test_departed_tracks_are_pruned():
    rule = IllegalParkingRule([ZONE])
    for i in range(50):
        ctx = FrameContext(frame_idx=i, timestamp=i / FPS, fps=FPS)
        rule.update([track_at(*INSIDE, tid=i)], ctx)
    assert len(rule._state) <= 1, (
        f"per-track state leaked: {len(rule._state)} entries after 50 vehicles")


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals()), "illegal_parking"))
