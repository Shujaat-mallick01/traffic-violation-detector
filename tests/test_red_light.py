"""Tests for RedLightRule and the HSV light-state estimator.

The rule tests need nothing installed. The estimator tests need cv2 + numpy and
skip themselves cleanly if those are missing.

    python tests/test_red_light.py     # or: pytest tests/
"""
from __future__ import annotations

import sys

from _harness import (FakeLightRegion, FakeStopLine, run_tests,  # noqa: E402
                      track_at)

from src.violations.base import FrameContext          # noqa: E402
from src.violations.red_light import (RedLightRule,   # noqa: E402
                                      estimate_light_state_hsv)

# Horizontal stop line across the frame. side_of_line is positive below it
# (y > 460) and negative above, so with before_sign=+1 a vehicle approaches from
# below and commits the violation by moving upward across the line.
LINE = FakeStopLine(a=(200.0, 460.0), b=(900.0, 460.0), before_sign=1.0,
                    name="stopline_1")


def feed(rule, positions, light_state, tid=1):
    """Push a sequence of y-positions through the rule; return violations."""
    fired = []
    for i, y in enumerate(positions):
        ctx = FrameContext(frame_idx=i, timestamp=i / 25.0, fps=25.0,
                           light_state=light_state)
        fired.extend(rule.update([track_at(450.0, y, tid=tid)], ctx))
    return fired


# --------------------------------------------------------------------------
# Rule logic
# --------------------------------------------------------------------------

def test_flags_crossing_on_red():
    fired = feed(RedLightRule([LINE]), [480.0, 440.0], "red")
    assert len(fired) == 1, f"expected one violation, got {len(fired)}"
    assert fired[0].type == "red_light"
    assert "stopline_1" in fired[0].detail


def test_no_violation_when_light_is_green():
    assert feed(RedLightRule([LINE]), [480.0, 440.0], "green") == []


def test_unknown_light_state_never_produces_a_violation():
    """The rule's core safety promise: we never guess a red light.

    A camera with no signal in view, or an estimator that cannot read the lamp,
    yields light_state=None. Enforcing on that would invent violations against
    drivers who did nothing wrong.
    """
    assert feed(RedLightRule([LINE]), [480.0, 440.0], None) == []


def test_no_violation_crossing_from_the_far_side():
    """Traffic leaving the junction crosses the same line the other way."""
    assert feed(RedLightRule([LINE]), [440.0, 480.0], "red") == []


def test_one_violation_per_track():
    # Cross, then wander back and forth over the line.
    fired = feed(RedLightRule([LINE]), [480.0, 440.0, 480.0, 440.0], "red")
    assert len(fired) == 1, f"debounce failed, emitted {len(fired)}"


def test_departed_tracks_are_pruned():
    """Guards the unbounded-growth fix: a 24/7 stream must not accumulate state."""
    rule = RedLightRule([LINE])
    for i in range(50):                      # 50 vehicles pass through, one at a time
        ctx = FrameContext(frame_idx=i, timestamp=i / 25.0, fps=25.0,
                           light_state="green")
        rule.update([track_at(450.0, 500.0, tid=i)], ctx)
    assert len(rule._last_centroid) <= 1, (
        f"per-track state leaked: {len(rule._last_centroid)} entries retained "
        f"after 50 vehicles passed")


# --------------------------------------------------------------------------
# HSV estimator
# --------------------------------------------------------------------------

def _lamp(colour_bgr, lit_pixels, size=(100, 40)):
    """A dark crop with `lit_pixels` of one colour burned into the top-left."""
    import numpy as np
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    side = int(lit_pixels ** 0.5)
    img[0:side, 0:side] = colour_bgr
    return img


RED, GREEN = (0, 0, 255), (0, 255, 0)
FULL_REGION = FakeLightRegion(bbox=(0.0, 0.0, 40.0, 100.0))


def test_hsv_reads_a_clearly_lit_red_lamp():
    frame = _lamp(RED, lit_pixels=900)        # 900/4000 = 22% of the crop
    assert estimate_light_state_hsv(frame, FULL_REGION) == "red"


def test_hsv_ignores_a_stray_red_pixel():
    """The bug this guard exists for.

    Counting any non-zero pixel as a reading meant one speck — glare, a brake
    light behind the signal, a red jacket — set the state to 'red' and armed
    enforcement. Below min_lit_fraction the answer must be 'I don't know'.
    """
    frame = _lamp(RED, lit_pixels=4)          # 4/4000 = 0.1% of the crop
    assert estimate_light_state_hsv(frame, FULL_REGION) is None


def test_hsv_refuses_an_ambiguous_crop():
    """Red and green neck-and-neck is noise, not a reading."""
    import numpy as np
    frame = np.zeros((100, 40, 3), dtype=np.uint8)
    frame[0:20, 0:20] = RED                   # 400 px
    frame[40:58, 0:18] = GREEN                # 324 px -> ratio 1.23 < min_margin
    assert estimate_light_state_hsv(frame, FULL_REGION) is None


def test_hsv_handles_an_empty_crop():
    frame = _lamp(RED, lit_pixels=900)
    off_frame = FakeLightRegion(bbox=(0.0, 0.0, 0.0, 0.0))
    assert estimate_light_state_hsv(frame, off_frame) is None


if __name__ == "__main__":
    ns = dict(globals())
    try:
        import cv2, numpy  # noqa: F401
    except ImportError:
        print("(cv2/numpy missing — skipping HSV estimator tests)\n")
        ns = {k: v for k, v in ns.items() if "hsv" not in k}
    sys.exit(run_tests(ns, "red_light"))
