"""Main per-frame pipeline: detect+track -> rules -> evidence -> optional view."""
from __future__ import annotations

from typing import List, Optional

from .detector import Detector
from .evidence import EvidenceWriter
from .violations.base import FrameContext, ViolationRule
from .violations.illegal_parking import IllegalParkingRule
from .violations.red_light import RedLightRule, estimate_light_state_hsv
from .violations.wrong_way import WrongWayRule
from .zones import CameraConfig


def build_rules(cfg: CameraConfig) -> List[ViolationRule]:
    """Enable only the rules the camera has geometry for.

    Rider-safety rules are intentionally not registered here yet — they need a
    custom-trained model (see violations/rider_safety.py and README Phase 3).
    """
    rules: List[ViolationRule] = []
    if cfg.no_parking:
        rules.append(IllegalParkingRule(cfg.no_parking))
    if cfg.lanes:
        rules.append(WrongWayRule(cfg.lanes))
    if cfg.stop_lines:
        rules.append(RedLightRule(cfg.stop_lines))
    return rules


def run(source, cfg: CameraConfig, model_path: str = "yolo26n.pt",
        show: bool = False, out_dir: str = "evidence",
        conf: float = 0.3, device: Optional[str] = None,
        stride: int = 1, imgsz: int = 640) -> int:
    """Process `source` against `cfg`, writing evidence for every violation.

    stride > 1 runs detection on every Nth frame only — the main CPU speedup.
    Skipped frames are still *grabbed* (decoded-but-not-processed) so `frame_idx`
    keeps counting true source frames and every timestamp stays wall-accurate.
    Rules must therefore express their thresholds in seconds, never in frames,
    or their behaviour would silently change with stride.
    """
    import cv2

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    detector = Detector(model_path=model_path, conf=conf, device=device, imgsz=imgsz)
    rules = build_rules(cfg)
    evidence = EvidenceWriter(out_dir)
    print(f"Enabled rules: {[r.name for r in rules] or '(none — check camera config)'}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    fps = cfg.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    if stride > 1:
        print(f"Stride {stride}: detecting at ~{fps / stride:.1f} of {fps:.0f} fps "
              f"(timestamps remain true).")

    frame_idx = 0
    processed = 0
    total = 0
    try:
        while True:
            # Cheaply discard the frames this stride skips: grab() decodes without
            # the colour-convert + array copy that retrieve()/read() would do.
            if frame_idx % stride != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            ts = frame_idx / fps
            tracks = detector.track(frame)

            light_state = None
            if cfg.light_regions:
                light_state = estimate_light_state_hsv(frame, cfg.light_regions[0])

            ctx = FrameContext(frame_idx=frame_idx, timestamp=ts, fps=fps,
                               light_state=light_state, frame=frame)

            frame_violations = []
            for rule in rules:
                frame_violations.extend(rule.update(tracks, ctx))

            if frame_violations:
                evidence.save(frame, frame_violations)
                total += len(frame_violations)
                for v in frame_violations:
                    print(f"[{ts:7.2f}s] {v.type} #{v.track_id} "
                          f"({v.class_name}) — {v.detail}")

            if show:
                _draw_debug(cv2, frame, tracks, light_state)
                cv2.imshow("traffic-violation-detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\nDone. {frame_idx} frames read, {processed} detected on, "
          f"{total} violations logged to {out_dir}/")
    return total


def _draw_debug(cv2, frame, tracks, light_state):
    for t in tracks:
        x1, y1, x2, y2 = [int(c) for c in t.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(frame, f"{t.class_name}#{t.track_id}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    if light_state:
        cv2.putText(frame, f"light: {light_state}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
