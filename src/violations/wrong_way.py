"""Wrong-way driving.

For each vehicle we keep a short history of centroids, estimate a smoothed motion
vector, and compare it to the allowed direction of the lane it is in. If the
vehicle moves clearly against the allowed direction for several consecutive
frames, it is flagged. Works today with COCO-pretrained detection.

Tuning knobs:
  min_speed_px   ignore near-stationary vehicles (parked cars aren't wrong-way).
  against_frames how many consecutive "against" frames before flagging (debounce).
  dot_thresh     cosine similarity below which motion counts as "against"
                 (-0.5 ~= more than 120 degrees off the allowed direction).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from ..geometry import dot, point_in_polygon, unit_vector
from .base import FrameContext, Violation, ViolationRule


class WrongWayRule(ViolationRule):
    name = "wrong_way"

    def __init__(self, lanes, history: int = 10, min_speed_px: float = 3.0,
                 against_frames: int = 8, dot_thresh: float = -0.5):
        self.lanes = lanes                    # list[Lane] with .polygon and .direction
        self.history = history
        self.min_speed = min_speed_px
        self.against_frames = against_frames
        self.dot_thresh = dot_thresh
        self._pos: Dict[int, deque] = {}
        self._against: Dict[int, int] = {}
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
            buf = self._pos.setdefault(t.track_id, deque(maxlen=self.history))
            buf.append(c)
            if len(buf) < 2:
                continue

            lane = self._lane_for(c)
            if lane is None:
                self._against[t.track_id] = 0
                continue

            move = (buf[-1][0] - buf[0][0], buf[-1][1] - buf[0][1])
            speed = (move[0] ** 2 + move[1] ** 2) ** 0.5
            if speed < self.min_speed:
                continue

            alignment = dot(unit_vector(move), unit_vector(lane.direction))
            if alignment < self.dot_thresh:
                self._against[t.track_id] = self._against.get(t.track_id, 0) + 1
            else:
                self._against[t.track_id] = 0

            if (self._against.get(t.track_id, 0) >= self.against_frames
                    and t.track_id not in self._flagged):
                self._flagged.add(t.track_id)
                out.append(Violation(
                    type=self.name, track_id=t.track_id, class_name=t.class_name,
                    frame_idx=ctx.frame_idx, timestamp=ctx.timestamp, bbox=t.bbox,
                    detail=f"moving against '{lane.name}' direction",
                    extra={"lane": lane.name, "alignment": round(alignment, 2)},
                ))

        for tid in list(self._pos.keys()):
            if tid not in active:
                self._pos.pop(tid, None)
                self._against.pop(tid, None)
        return out
