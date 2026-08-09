"""Rider-safety violations: no-helmet, triple-riding, no-seatbelt (Phase 3).

These require a *custom-trained* detector — COCO has no 'helmet' / 'no-helmet'
class and no way to read a seatbelt. This file defines the intended interfaces
and the plan; the logic is intentionally left unimplemented (rules raise on
construction) until we have annotated data, so nothing here silently returns
wrong results.

PLAN
====
no_helmet / triple_riding
  1. Fine-tune a YOLO26 detector with classes:
       motorcycle, rider, helmet, no-helmet   (+ keep person for association)
     Bootstrap from public helmet datasets, then add local Karachi footage —
     local lighting, plate styles, and rider density differ enough to matter.
  2. Per frame, associate riders/helmets to each motorcycle by bbox overlap.
  3. no_helmet    -> a rider on a motorcycle whose head region is 'no-helmet'.
     triple_riding -> more than 2 riders associated with one motorcycle.

no_seatbelt (hardest)
  Needs a high-resolution, near-frontal view through the windshield. Usually a
  dedicated camera plus a classifier on the driver crop. Treat as a separate
  track; a standard overhead traffic cam cannot see this reliably.

When data is ready, replace the __init__ bodies with real logic and register the
rules in pipeline.build_rules().
"""
from __future__ import annotations

from typing import List

from .base import FrameContext, Violation, ViolationRule

_NOT_READY = (
    "{rule} needs a custom-trained detector. See this module's docstring and "
    "README 'Phase 3 — data & training' before enabling it."
)


class NoHelmetRule(ViolationRule):
    name = "no_helmet"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_READY.format(rule="no_helmet"))

    def update(self, tracks, ctx: FrameContext) -> List[Violation]:
        return []


class TripleRidingRule(ViolationRule):
    name = "triple_riding"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_READY.format(rule="triple_riding"))

    def update(self, tracks, ctx: FrameContext) -> List[Violation]:
        return []
