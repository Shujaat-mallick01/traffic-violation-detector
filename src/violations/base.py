"""Base types shared by all violation rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Violation:
    type: str
    track_id: int
    class_name: str
    frame_idx: int
    timestamp: float                      # seconds into the stream
    bbox: Tuple[float, float, float, float]
    detail: str = ""
    snapshot_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameContext:
    """Everything a rule might need for the current frame besides the tracks."""
    frame_idx: int
    timestamp: float
    fps: float
    light_state: Optional[str] = None     # "red" | "yellow" | "green" | None
    frame: Any = None                     # raw BGR image, for rules that need pixels


class ViolationRule:
    """Stateful rule. `update()` is called once per frame with current tracks.

    Rules keep their own per-track history and are responsible for debouncing so
    a single violation is emitted once, not every frame it persists.
    """
    name: str = "rule"

    def update(self, tracks, ctx: FrameContext) -> List[Violation]:
        raise NotImplementedError
