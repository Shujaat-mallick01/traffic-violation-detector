"""Red-light running (Phase 2).

The vehicle side is fully implemented: while the signal is red, a vehicle track
whose centroid crosses the stop line from the 'before' side to the far side is
flagged. The piece that needs real-footage tuning is a *reliable* light state.
`estimate_light_state_hsv` is a deliberately simple starting point — replace it
with a small classifier trained on crops of the actual signal heads for
production (glare, LED bloom, sun behind the lamp, and partial occlusion all
break naive color thresholds).

Safety choice: if the light state is unknown, NO red-light violation is emitted.
We never guess a red light.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..geometry import segments_intersect, side_of_line
from .base import FrameContext, Violation, ViolationRule


class RedLightRule(ViolationRule):
    name = "red_light"

    def __init__(self, stop_lines):
        self.stop_lines = stop_lines
        self._last_centroid: Dict[int, tuple] = {}
        self._flagged: set = set()

    def update(self, tracks, ctx: FrameContext) -> List[Violation]:
        out: List[Violation] = []
        active = {t.track_id for t in tracks}

        # Only enforce on a confirmed red. Still update positions so we have a
        # valid 'previous' point on the frame the light turns red.
        if ctx.light_state != "red":
            for t in tracks:
                self._last_centroid[t.track_id] = t.centroid
            self._prune(active)
            return out

        for t in tracks:
            if not t.is_vehicle:
                continue
            prev = self._last_centroid.get(t.track_id)
            cur = t.centroid
            self._last_centroid[t.track_id] = cur
            if prev is None or t.track_id in self._flagged:
                continue

            for line in self.stop_lines:
                if not segments_intersect(prev, cur, line.a, line.b):
                    continue
                prev_side = side_of_line(prev, line.a, line.b)
                cur_side = side_of_line(cur, line.a, line.b)
                crossed_forward = (prev_side * line.before_sign > 0) and \
                                  (cur_side * line.before_sign < 0)
                if crossed_forward:
                    self._flagged.add(t.track_id)
                    out.append(Violation(
                        type=self.name, track_id=t.track_id, class_name=t.class_name,
                        frame_idx=ctx.frame_idx, timestamp=ctx.timestamp, bbox=t.bbox,
                        detail=f"crossed '{line.name}' on red",
                        extra={"line": line.name},
                    ))
                    break

        self._prune(active)
        return out

    def _prune(self, active) -> None:
        """Forget tracks that have left the frame.

        Without this the per-track dicts grow for the lifetime of the process —
        harmless on a short clip, an unbounded leak on a 24/7 RTSP feed. The
        flagged set is deliberately NOT pruned: it holds one id per real
        violation, and clearing it would let a track that flickers out and
        re-acquires the same id be reported twice.
        """
        for tid in list(self._last_centroid.keys()):
            if tid not in active:
                self._last_centroid.pop(tid, None)


def estimate_light_state_hsv(frame, light_region, min_lit_fraction: float = 0.02,
                             min_margin: float = 1.5) -> Optional[str]:
    """Rough traffic-light state from color dominance inside the light bbox.

    Starting point only — see module docstring. Returns "red"/"yellow"/"green"
    or None when no colored lamp is confidently visible.

    Two guards keep this from inventing a state, because a wrong "red" here
    manufactures a violation against a driver who did nothing:

    `min_lit_fraction` — the winning colour must cover at least this share of the
    crop. Without it a single qualifying pixel (glare, a brake light behind the
    signal, a red jacket) decides the state.
    `min_margin` — the winner must beat the runner-up by this factor. An ambiguous
    crop where red and green are neck-and-neck is not a reading, it is noise.

    Returning None is always safe: RedLightRule enforces only on a confirmed
    "red" and emits nothing when the state is unknown.
    """
    import cv2

    x1, y1, x2, y2 = [int(v) for v in light_region.bbox]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # red hue wraps around 0/180, so two bands
    red = (cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)) |
           cv2.inRange(hsv, (170, 90, 90), (180, 255, 255)))
    yellow = cv2.inRange(hsv, (18, 90, 90), (35, 255, 255))
    green = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    # inRange returns 0/255, so count set pixels rather than summing intensities.
    total = float(crop.shape[0] * crop.shape[1])
    counts = {
        "red": int(cv2.countNonZero(red)),
        "yellow": int(cv2.countNonZero(yellow)),
        "green": int(cv2.countNonZero(green)),
    }
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    (state, top), (_, second) = ranked[0], ranked[1]

    if top / total < min_lit_fraction:
        return None                      # nothing bright enough to be a lamp
    if second > 0 and top < second * min_margin:
        return None                      # two colours too close to call
    return state
