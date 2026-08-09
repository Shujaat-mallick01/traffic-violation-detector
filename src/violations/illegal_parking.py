"""Illegal parking / stopping in a no-parking zone.

A vehicle whose centroid stays inside a no-parking polygon and barely moves for
longer than `min_seconds` is flagged once (debounced per track). Works today with
COCO-pretrained detection — no custom data required.
"""
from __future__ import annotations

from typing import Dict, List

from ..geometry import point_in_polygon
from .base import FrameContext, Violation, ViolationRule


class IllegalParkingRule(ViolationRule):
    name = "illegal_parking"

    def __init__(self, zones, min_seconds: float = 8.0, move_thresh_px: float = 25.0):
        self.zones = zones                    # list[NoParkingZone]
        self.min_seconds = min_seconds
        self.move_thresh = move_thresh_px
        # track_id -> {since, anchor, zone, flagged}
        self._state: Dict[int, dict] = {}

    def update(self, tracks, ctx: FrameContext) -> List[Violation]:
        out: List[Violation] = []
        active_ids = set()

        for t in tracks:
            if not t.is_vehicle:
                continue
            c = t.centroid

            zone_name = None
            for z in self.zones:
                if point_in_polygon(c, z.polygon):
                    zone_name = z.name
                    break
            if zone_name is None:
                self._state.pop(t.track_id, None)
                continue

            active_ids.add(t.track_id)
            st = self._state.get(t.track_id)
            if st is None or st["zone"] != zone_name:
                self._state[t.track_id] = {
                    "since": ctx.timestamp, "anchor": c,
                    "zone": zone_name, "flagged": False,
                }
                continue

            moved = ((c[0] - st["anchor"][0]) ** 2 + (c[1] - st["anchor"][1]) ** 2) ** 0.5
            if moved > self.move_thresh:
                # still moving through the zone — reset the dwell timer
                st["since"] = ctx.timestamp
                st["anchor"] = c
                continue

            dwell = ctx.timestamp - st["since"]
            if dwell >= self.min_seconds and not st["flagged"]:
                st["flagged"] = True
                out.append(Violation(
                    type=self.name, track_id=t.track_id, class_name=t.class_name,
                    frame_idx=ctx.frame_idx, timestamp=ctx.timestamp, bbox=t.bbox,
                    detail=f"stationary {dwell:.1f}s in {zone_name}",
                    extra={"zone": zone_name, "dwell_s": round(dwell, 1)},
                ))

        # forget tracks that left the frame
        for tid in list(self._state.keys()):
            if tid not in active_ids:
                self._state.pop(tid, None)
        return out
