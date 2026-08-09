"""Shared fixtures + a tiny test runner.

Everything here is duck-typed rather than imported from src.zones, because that
module pulls in PyYAML and these tests are meant to run on a bare interpreter
with nothing installed. The rules only ever read attributes, never isinstance.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Make `src` importable when a test file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Point = Tuple[float, float]


@dataclass
class FakeTrack:
    """Stands in for detector.Track without needing ultralytics imported."""
    track_id: int
    class_name: str = "car"
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    vehicle: bool = True

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def is_vehicle(self) -> bool:
        return self.vehicle


def track_at(x: float, y: float, tid: int = 1, **kw) -> FakeTrack:
    """A vehicle track whose centroid sits exactly at (x, y)."""
    return FakeTrack(track_id=tid, bbox=(x - 10, y - 10, x + 10, y + 10), **kw)


@dataclass
class FakeLane:
    polygon: List[Point]
    direction: Point
    name: str = "lane"


@dataclass
class FakeZone:
    polygon: List[Point]
    name: str = "no_parking"


@dataclass
class FakeStopLine:
    a: Point
    b: Point
    before_sign: float = 1.0
    name: str = "stopline"


@dataclass
class FakeLightRegion:
    bbox: Tuple[float, float, float, float]
    name: str = "light"


def run_tests(namespace: dict, title: str = "") -> int:
    """Run every test_* callable in `namespace`. Returns a process exit code."""
    tests = [v for k, v in sorted(namespace.items())
             if k.startswith("test_") and callable(v)]
    if title:
        print(f"=== {title} ===")
    failures = 0
    for fn in tests:
        print(f"{fn.__name__} ...")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL: {e}")
        except Exception as e:                       # noqa: BLE001
            failures += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0
