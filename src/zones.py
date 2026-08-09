"""Per-camera zone definitions loaded from a YAML config.

A camera config describes the *fixed geometry* of one view: no-parking polygons,
stop lines, allowed lane directions, and optionally the traffic-light region to
monitor. Define these once per camera with calibrate.py. The same violation
rules are meaningless without the geometry of that specific view, which is why
this lives in config, not code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml

Point = Tuple[float, float]


@dataclass
class StopLine:
    a: Point
    b: Point
    # A vehicle moving from the 'before' side of the line to the far side while
    # the light is red is running the light. `before_sign` is the sign of
    # side_of_line() for the approaching side; flip it if crossings read backwards.
    before_sign: float = 1.0
    name: str = "stopline"


@dataclass
class Lane:
    polygon: List[Point]
    direction: Point            # allowed travel direction (dx, dy) in image space
    name: str = "lane"


@dataclass
class NoParkingZone:
    polygon: List[Point]
    name: str = "no_parking"


@dataclass
class LightRegion:
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    name: str = "light"


@dataclass
class CameraConfig:
    name: str = "camera"
    fps: Optional[float] = None
    no_parking: List[NoParkingZone] = field(default_factory=list)
    stop_lines: List[StopLine] = field(default_factory=list)
    lanes: List[Lane] = field(default_factory=list)
    light_regions: List[LightRegion] = field(default_factory=list)

    # ---- IO -------------------------------------------------------------
    @staticmethod
    def load(path: str) -> "CameraConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = CameraConfig(name=raw.get("name", "camera"), fps=raw.get("fps"))
        for z in raw.get("no_parking", []) or []:
            cfg.no_parking.append(NoParkingZone(
                polygon=[tuple(p) for p in z["polygon"]],
                name=z.get("name", "no_parking"),
            ))
        for s in raw.get("stop_lines", []) or []:
            cfg.stop_lines.append(StopLine(
                a=tuple(s["a"]), b=tuple(s["b"]),
                before_sign=float(s.get("before_sign", 1.0)),
                name=s.get("name", "stopline"),
            ))
        for l in raw.get("lanes", []) or []:
            cfg.lanes.append(Lane(
                polygon=[tuple(p) for p in l["polygon"]],
                direction=tuple(l["direction"]),
                name=l.get("name", "lane"),
            ))
        for lr in raw.get("light_regions", []) or []:
            cfg.light_regions.append(LightRegion(
                bbox=tuple(lr["bbox"]),
                name=lr.get("name", "light"),
            ))
        return cfg

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fps": self.fps,
            "no_parking": [
                {"name": z.name, "polygon": [list(p) for p in z.polygon]}
                for z in self.no_parking
            ],
            "stop_lines": [
                {"name": s.name, "a": list(s.a), "b": list(s.b),
                 "before_sign": s.before_sign}
                for s in self.stop_lines
            ],
            "lanes": [
                {"name": l.name, "polygon": [list(p) for p in l.polygon],
                 "direction": list(l.direction)}
                for l in self.lanes
            ],
            "light_regions": [
                {"name": lr.name, "bbox": list(lr.bbox)}
                for lr in self.light_regions
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
