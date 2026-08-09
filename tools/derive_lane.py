"""Derive a lane polygon + travel direction from observed traffic. Headless.

`src/calibrate.py` is the tool for real deployments — a human drawing the true
geometry beats anything inferred. This exists for the two cases it cannot cover:

  * a headless box (server, container, SSH with no display), where the OpenCV
    GUI calibrate.py depends on is unavailable;
  * bootstrapping a rough config for a new clip in seconds, to sanity-check the
    pipeline before investing in careful calibration.

It runs the detector over the clip, keeps the vehicles travelling in the
dominant direction, and fits a polygon around where they actually drove. On a
divided road that isolates ONE carriageway, which is what you want: a lane
covering both directions flags all oncoming traffic as wrong-way.

    python tools/derive_lane.py --source samples/cars2.mp4 --out config/cam1.yaml

Inspect the result before trusting it. It only sees where traffic went during
this clip, so a lane that was empty is a lane it does not know about.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zones import CameraConfig, Lane, NoParkingZone  # noqa: E402


def collect_tracks(source, imgsz=640, stride=2, conf=0.3, device=None):
    """Return {track_id: [centroid, ...]} plus the frame size."""
    import cv2

    from src.detector import Detector

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det = Detector(imgsz=imgsz, conf=conf, device=device)
    paths = defaultdict(list)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            for t in det.track(frame):
                if t.is_vehicle:
                    paths[t.track_id].append(t.centroid)
        idx += 1
    cap.release()
    return paths, w, h


def dominant_flow(paths, min_points=4, min_travel=20.0):
    """Split tracks by travel direction; return (unit_vector, kept_paths).

    Tracks that barely moved carry no directional information and are dropped —
    including them would drag the mean toward zero.
    """
    moving = {}
    for tid, pts in paths.items():
        if len(pts) < min_points:
            continue
        dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        if (dx * dx + dy * dy) ** 0.5 >= min_travel:
            moving[tid] = (dx, dy)
    if not moving:
        raise SystemExit("No vehicles travelled far enough to infer a direction.")

    sx = sum(v[0] for v in moving.values())
    sy = sum(v[1] for v in moving.values())
    n = (sx * sx + sy * sy) ** 0.5 or 1.0
    ux, uy = sx / n, sy / n

    # Keep only tracks flowing WITH the dominant direction: on a divided road the
    # rest are the opposite carriageway and belong to a different lane.
    kept = {tid: paths[tid] for tid, (dx, dy) in moving.items()
            if dx * ux + dy * uy > 0}
    return (ux, uy), kept, len(moving) - len(kept)


def aligned_points(paths, direction, min_step=1.5):
    """Points whose LOCAL motion follows `direction`, not merely whose track does.

    Filtering whole tracks by net displacement is not enough. ByteTrack
    occasionally switches an id between two vehicles, and on a divided road the
    swap can be across the median — the track's net motion still looks correct
    while half its points sit on the oncoming carriageway. Fitting a lane to
    those points drags it across the median.

    Judging each step individually discards the mismatched half instead of the
    whole track, which also keeps honest tracks on a curving road.
    """
    ux, uy = direction
    kept = []
    for path in paths.values():
        for a, b in zip(path, path[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            mag = (dx * dx + dy * dy) ** 0.5
            if mag < min_step:
                continue                       # too small to have a direction
            if (dx * ux + dy * uy) / mag > 0.5:  # within ~60 deg of the flow
                kept.append(b)
    return kept


def fit_polygon(points, w, h, bands=6, margin=18.0, lo=5.0, hi=95.0):
    """Fit a banded polygon around the points traffic actually passed through."""
    import numpy as np

    pts = np.array(points, dtype=float)
    if len(pts) < bands * 2:
        raise SystemExit(f"Only {len(pts)} usable track points — too few to fit "
                         f"a lane. Try a longer clip or --bands fewer.")

    y_lo, y_hi = np.percentile(pts[:, 1], [2, 98])
    edges = np.linspace(y_lo, y_hi, bands + 1)

    left, right = [], []
    for i in range(bands):
        y0, y1 = edges[i], edges[i + 1]
        in_band = pts[(pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
        if len(in_band) < 3:
            continue
        xl, xr = np.percentile(in_band[:, 0], [lo, hi])
        yc = (y0 + y1) / 2.0
        left.append((max(0.0, xl - margin), yc))
        right.append((min(float(w), xr + margin), yc))

    if len(left) < 2:
        raise SystemExit("Traffic too sparse to fit a lane polygon.")
    # Down the left edge, back up the right, so the ring is not self-crossing.
    # Cast to plain floats: numpy scalars leak in from percentile() and PyYAML's
    # safe_dump refuses to represent them.
    return [(float(round(x, 1)), float(round(y, 1)))
            for x, y in left + right[::-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file or RTSP URL")
    ap.add_argument("--out", required=True, help="YAML config to write")
    ap.add_argument("--name", default="camera")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--bands", type=int, default=6,
                    help="how many horizontal slices to fit the lane width over")
    ap.add_argument("--margin", type=float, default=18.0,
                    help="pixels of padding either side of observed traffic")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reverse", action="store_true",
                    help="write the direction REVERSED — a positive control that "
                         "makes ordinary traffic trip the wrong-way rule, to prove "
                         "the rule fires at all")
    ap.add_argument("--no-parking", action="store_true",
                    help="also add a no-parking zone covering the same polygon")
    args = ap.parse_args()

    paths, w, h = collect_tracks(args.source, imgsz=args.imgsz,
                                 stride=args.stride, device=args.device)
    (ux, uy), kept, dropped = dominant_flow(paths)
    points = aligned_points(kept, (ux, uy))
    polygon = fit_polygon(points, w, h, bands=args.bands, margin=args.margin)

    direction = (-ux, -uy) if args.reverse else (ux, uy)
    cfg = CameraConfig(name=args.name)
    cfg.lanes.append(Lane(polygon=polygon, direction=(round(direction[0], 3),
                                                      round(direction[1], 3)),
                          name="lane_1"))
    if args.no_parking:
        cfg.no_parking.append(NoParkingZone(polygon=polygon, name="no_parking_1"))
    cfg.save(args.out)

    print(f"frame            : {w}x{h}")
    print(f"tracks used      : {len(kept)}")
    print(f"tracks dropped   : {dropped} (opposite carriageway / other flow)")
    print(f"points fitted    : {len(points)} (steps aligned with the flow)")
    print(f"travel direction : ({ux:+.2f}, {uy:+.2f})"
          + ("  [WRITTEN REVERSED]" if args.reverse else ""))
    print(f"lane polygon     : {len(polygon)} points")
    print(f"wrote            : {args.out}")
    if dropped:
        print(f"\nNote: {dropped} tracks ran counter to this lane. On a divided "
              f"road that is\nthe other carriageway — add it as a second lane, "
              f"or they will read as wrong-way.")


if __name__ == "__main__":
    main()
