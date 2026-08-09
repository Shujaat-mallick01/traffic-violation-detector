"""Tests for WrongWayRule, focused on stride-independence.

These run without ultralytics or OpenCV: the rule only touches src.geometry,
which is plain-Python by design.

    python -m tests.test_wrong_way        (from the project root)
    pytest tests/                          (if pytest is installed)
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from _harness import FakeLane, run_tests, track_at     # noqa: E402

from src.violations.base import FrameContext           # noqa: E402
from src.violations.wrong_way import WrongWayRule      # noqa: E402

FPS = 25.0
LANE_X = 450.0                                  # vehicles travel up/down this column
# Lane runs down the middle of the frame; travelling DOWN (+y) is correct.
LANE = FakeLane(polygon=[(250.0, 100.0), (650.0, 100.0), (650.0, 900.0), (250.0, 900.0)],
                direction=(0.0, 1.0), name="lane_1")


def run_positions(positions: List[Tuple[float, float]], stride: int = 1,
                  rule: Optional[WrongWayRule] = None) -> List[float]:
    """Feed (timestamp, y) samples through the rule; return violation timestamps.

    Only every `stride`-th sample is shown to the rule, mimicking the pipeline's
    --stride. Timestamps stay true to the source frame, exactly as pipeline.run
    keeps them.
    """
    rule = rule or WrongWayRule([LANE])
    fired: List[float] = []
    for i, (ts, y) in enumerate(positions):
        if i % stride != 0:
            continue
        ctx = FrameContext(frame_idx=i, timestamp=ts, fps=FPS)
        for v in rule.update([track_at(LANE_X, y)], ctx):
            fired.append(v.timestamp)
    return fired


def constant_against(duration_s: float, speed_px_s: float = 100.0):
    """Vehicle driving UP the frame (against the lane) at a constant speed."""
    n = int(duration_s * FPS)
    return [(i / FPS, 800.0 - speed_px_s * (i / FPS)) for i in range(n)]


def test_flags_a_wrong_way_vehicle():
    fired = run_positions(constant_against(2.0))
    assert len(fired) == 1, f"expected exactly one violation, got {len(fired)}"
    assert fired[0] < 0.5, f"should flag within ~0.3s, flagged at {fired[0]:.2f}s"
    print(f"  baseline (stride=1): flagged at {fired[0]:.2f}s")


def test_stride_does_not_change_when_it_fires():
    """The whole point of the time-based rewrite.

    With the old frame-counting rule, --stride 3 meant 8 *processed* frames =
    0.96s of real time instead of 0.32s. Detection time must now depend only on
    the vehicle's behaviour, not on how often we sample it.
    """
    samples = constant_against(3.0)
    baseline = run_positions(samples, stride=1)[0]
    for stride in (2, 3, 5):
        fired = run_positions(samples, stride=stride)
        assert fired, f"stride={stride} never flagged"
        # Can't resolve faster than the sampling interval, so allow one step.
        tolerance = stride / FPS + 1e-6
        drift = abs(fired[0] - baseline)
        assert drift <= tolerance, (
            f"stride={stride} flagged at {fired[0]:.2f}s vs baseline "
            f"{baseline:.2f}s (drift {drift:.2f}s > {tolerance:.2f}s)")
        print(f"  stride={stride}: flagged at {fired[0]:.2f}s "
              f"(baseline {baseline:.2f}s, drift {drift:.3f}s)")


def test_going_the_right_way_is_never_flagged():
    n = int(2.0 * FPS)
    downhill = [(i / FPS, 200.0 + 100.0 * (i / FPS)) for i in range(n)]
    assert run_positions(downhill) == []


def test_stationary_time_is_not_evidence():
    """A brief wrong-way nudge, a long stop, then another nudge.

    Neither nudge alone is long enough to flag. The stop must not be credited as
    'against' time, or the vehicle is flagged the instant it moves again.
    """
    samples = []
    t = 0.0
    y = 800.0
    for _ in range(4):                       # ~0.16s against, below the 0.3s bar
        samples.append((t, y)); t += 1 / FPS; y -= 4.0
    for _ in range(int(5.0 * FPS)):          # 5s parked
        samples.append((t, y)); t += 1 / FPS
    for _ in range(4):                       # another short nudge
        samples.append((t, y)); t += 1 / FPS; y -= 4.0

    fired = run_positions(samples)
    assert fired == [], f"stationary time was credited as against-time: {fired}"


def test_one_violation_per_track():
    fired = run_positions(constant_against(10.0))
    assert len(fired) == 1, f"debounce failed, emitted {len(fired)} violations"


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals()), "wrong_way"))
