"""Evidence sink: annotated snapshots + a CSV log of violations.

Kept simple on purpose. Natural extensions: save a short clip around each event
(buffer N seconds of frames), and attach a license-plate crop + OCR once the
plate reader lands.
"""
from __future__ import annotations

import csv
import os
from typing import List

from .violations.base import Violation


class EvidenceWriter:
    def __init__(self, out_dir: str = "evidence"):
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "snapshots")
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "violations.csv")
        self._csv_started = os.path.exists(self.csv_path)

    FONT = 0                      # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    def _draw(self, frame, v: Violation):
        """Box the offending vehicle and label it, keeping the label on-screen.

        A violation near the right or top edge is exactly the case where the
        label used to run off the frame, so the evidence image lost the very
        text that explained it.
        """
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in v.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"{v.type} #{v.track_id} {v.detail}".strip()
        (tw, th), baseline = cv2.getTextSize(label, self.FONT, self.FONT_SCALE,
                                             self.FONT_THICKNESS)
        # Keep the text box inside the frame on both axes.
        tx = max(0, min(x1, w - tw - 2))
        ty = y1 - 8
        if ty - th < 0:                       # no room above the box: go below it
            ty = min(h - baseline - 1, y2 + th + 8)

        # Solid backing so red text stays readable over light road surfaces.
        cv2.rectangle(frame, (tx, ty - th - 3), (tx + tw + 2, ty + baseline),
                      (255, 255, 255), -1)
        cv2.putText(frame, label, (tx + 1, ty), self.FONT, self.FONT_SCALE,
                    (0, 0, 255), self.FONT_THICKNESS)
        return frame

    def save(self, frame, violations: List[Violation]):
        if not violations:
            return
        import cv2

        for v in violations:
            annotated = self._draw(frame.copy(), v)
            fname = f"{v.type}_{v.track_id}_{v.frame_idx}.jpg"
            path = os.path.join(self.img_dir, fname)
            cv2.imwrite(path, annotated)
            v.snapshot_path = path
            self._append_csv(v)

    def _append_csv(self, v: Violation):
        write_header = not self._csv_started
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp_s", "type", "track_id", "class",
                            "frame_idx", "detail", "snapshot"])
                self._csv_started = True
            w.writerow([f"{v.timestamp:.2f}", v.type, v.track_id, v.class_name,
                        v.frame_idx, v.detail, v.snapshot_path or ""])
