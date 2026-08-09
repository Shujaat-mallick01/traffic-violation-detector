"""Wrong-way driving.

For each vehicle we keep a short history of centroids, estimate a smoothed motion
vector, and compare it to the allowed direction of the lane it is in. If the
vehicle moves clearly against the allowed direction for long enough, it is
flagged. Works today with COCO-pretrained detection.

Every threshold here is expressed in **seconds and pixels-per-second**, never in
frames. That is deliberate: the pipeline can run with `--stride N` to detect on
every Nth frame only, so a frame count means a different amount of real time
depending on stride and source fps. Time-based thresholds behave identically at
any stride; frame-based ones would silently change meaning.

Direction and speed are measured over *different* windows, on purpose. Direction
wants a long window to average out bbox jitter; the "is it still moving" gate
wants a short one, because a long window keeps reporting the motion of a vehicle
that has already stopped — long enough for a brief wrong-way nudge followed by a
park to bank enough evidence to fire.

Tuning knobs:
  history_seconds  window used to smooth the motion *direction*.
  speed_window_s   shorter window used to decide the vehicle is still moving.
  min_speed_px_s   ignore near-stationary vehicles (parked cars aren't wrong-way).
  against_seconds  how long motion must stay 'against' before flagging (debounce).
  reset_seconds    a gap this long in against-motion restarts the evidence, so a
                   vehicle that nudges, parks, then nudges again is not treated
                   as having driven the wrong way continuously. Must stay
                   comfortably above the sampling interval (stride / fps).
  dot_thresh       cosine similarity below which motion counts as "against"
                   (-0.5 ~= more than 120 degrees off the allowed direction).

The defaults reproduce the previous frame-based behaviour on 25 fps footage
(history 10 frames = 0.4 s, 8 consecutive frames = ~0.3 s, 3 px per window
= 7.5 px/s).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

from ..geometry import dot, point_in_polygon, unit_vector
from .base import FrameContext, Violation, ViolationRule

# Safety bound on the per-track history so a bogus fps can't grow it without
# limit; the real trimming is by age, below.
_MAX_SAMPLES = 120


def _oldest_within(buf, now: float, window: float):
    """Oldest (timestamp, point) in `buf` no older than `window` seconds.

    `buf` is ordered oldest-first and must hold at least two samples. Always
    returns a sample strictly older than the newest one, so the caller gets a
    non-zero interval: at a large --stride the sampling interval can exceed
    `window` entirely, and measuring the newest sample against itself would read
    as zero speed and gate every vehicle out.
    """
    last = len(buf) - 1
    for i, sample in enumerate(buf):
        if now - sample[0] <= window:
            return buf[last - 1] if i == last else sample
    return buf[last - 1]


class WrongWayRule(ViolationRule):
    name = "wrong_way"

    def __init__(self, lanes, history_seconds: float = 0.4,
                 min_speed_px_s: float = 7.5, against_seconds: float = 0.3,
                 dot_thresh: float = -0.5, speed_window_s: float = 0.15,
                 reset_seconds: float = 1.0):
        self.lanes = lanes                    # list[Lane] with .polygon and .direction
        self.history_seconds = history_seconds
        self.speed_window_s = speed_window_s
        self.min_speed = min_speed_px_s
        self.against_seconds = against_seconds
        self.reset_seconds = reset_seconds
        self.dot_thresh = dot_thresh
        # track_id -> deque[(timestamp, centroid)]
        self._pos: Dict[int, deque] = {}
        # track_id -> seconds of *moving* travel accumulated against the lane
        self._against: Dict[int, float] = {}
        # track_id -> timestamp this track was last evaluated, to measure the step
        self._last_eval: Dict[int, float] = {}
        # track_id -> timestamp we last saw it actually moving against the lane
        self._last_against: Dict[int, float] = {}
        # Never cleared: one entry per real violation, and clearing it would let a
        # track that flickers out and re-acquires the same id be logged twice.
        self._flagged: set = set()

    def _lane_for(self, point):
        for lane in self.lanes:
            if point_in_polygon(point, lane.polygon):
                return lane
        return None

    def update(self, tracks, ctx: FrameContext) -> List[Violation]:
        out: List[Violation] = []
        active = set()

        for t in tracks:
            if not t.is_vehicle:
                continue
            active.add(t.track_id)
            c = t.centroid

            buf = self._pos.setdefault(t.track_id, deque(maxlen=_MAX_SAMPLES))
            buf.append((ctx.timestamp, c))
            # Drop samples that have aged out of the smoothing window, but always
            # keep at least two so a motion vector stays computable.
            while len(buf) > 2 and ctx.timestamp - buf[0][0] > self.history_seconds:
                buf.popleft()

            # How much real time this observation stands for. Consumed even when
            # we bail out below, so time spent stationary or out-of-lane is never
            # later credited as evidence of going the wrong way. Clamped so a
            # tracking dropout can't dump one huge step into the accumulator.
            prev_eval = self._last_eval.get(t.track_id)
            self._last_eval[t.track_id] = ctx.timestamp
            step = 0.0 if prev_eval is None else min(ctx.timestamp - prev_eval,
                                                     self.history_seconds)

            if len(buf) < 2:
                continue

            lane = self._lane_for(c)
            if lane is None:
                self._against.pop(t.track_id, None)
                continue

            # Speed gate over a SHORT recent window, so a vehicle that has just
            # stopped stops counting promptly instead of coasting on stale motion
            # still sitting in the smoothing window.
            (ts0, sp0) = _oldest_within(buf, ctx.timestamp, self.speed_window_s)
            (ts1, sp1) = buf[-1]
            recent_dt = ts1 - ts0
            if recent_dt <= 0:
                continue
            recent = (sp1[0] - sp0[0], sp1[1] - sp0[1])
            speed = ((recent[0] ** 2 + recent[1] ** 2) ** 0.5) / recent_dt  # px/s
            if speed < self.min_speed:
                # Stationary is not wrong-way, but it is not evidence of going the
                # right way either: hold the accumulator rather than resetting it.
                continue

            # Direction over the FULL window, which averages out bbox jitter.
            (t0, p0), (t1, p1) = buf[0], buf[-1]
            if t1 - t0 <= 0:
                continue
            move = (p1[0] - p0[0], p1[1] - p0[1])

            alignment = dot(unit_vector(move), unit_vector(lane.direction))
            if alignment >= self.dot_thresh:
                self._against[t.track_id] = 0.0
                self._last_against.pop(t.track_id, None)
                continue

            # Evidence must be roughly contiguous. A long gap since the last
            # against-observation means the vehicle stopped or behaved normally in
            # between, so the earlier seconds no longer describe one manoeuvre.
            last = self._last_against.get(t.track_id)
            banked = self._against.get(t.track_id, 0.0)
            if last is not None and ctx.timestamp - last > self.reset_seconds:
                banked = 0.0
            self._last_against[t.track_id] = ctx.timestamp

            against = banked + step
            self._against[t.track_id] = against
            if against >= self.against_seconds and t.track_id not in self._flagged:
                self._flagged.add(t.track_id)
                out.append(Violation(
                    type=self.name, track_id=t.track_id, class_name=t.class_name,
                    frame_idx=ctx.frame_idx, timestamp=ctx.timestamp, bbox=t.bbox,
                    detail=f"moving against '{lane.name}' direction",
                    extra={"lane": lane.name, "alignment": round(alignment, 2),
                           "against_s": round(against, 2)},
                ))

        for tid in list(self._pos.keys()):
            if tid not in active:
                self._pos.pop(tid, None)
                self._against.pop(tid, None)
                self._last_eval.pop(tid, None)
                self._last_against.pop(tid, None)
        return out
