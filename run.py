"""CLI entry point for the traffic violation detector.

Examples
--------
Calibrate zones for a camera (draw once on a snapshot):
    python -m src.calibrate --source sample.mp4 --out config/cam1.yaml

Run on a recorded file with a live annotated window:
    python run.py --source sample.mp4 --config config/cam1.yaml --show

Run on an RTSP / CCTV stream:
    python run.py --source "rtsp://user:pass@host:554/stream" --config config/cam1.yaml

Use YOLO11 instead of YOLO26, on GPU:
    python run.py --source sample.mp4 --config config/cam1.yaml --model yolo11n.pt --device 0
"""
from __future__ import annotations

import argparse

from src.pipeline import run
from src.zones import CameraConfig


def parse_source(s: str):
    return int(s) if s.isdigit() else s   # allow webcam indices like "0"


def main():
    ap = argparse.ArgumentParser(description="Traffic violation detector")
    ap.add_argument("--source", required=True,
                    help="video file, RTSP URL, or webcam index")
    ap.add_argument("--config", required=True,
                    help="camera zone YAML (make one with src/calibrate.py)")
    ap.add_argument("--model", default="yolo26n.pt",
                    help="Ultralytics weights (e.g. yolo26n.pt, yolo11n.pt)")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="detection confidence threshold")
    ap.add_argument("--device", default=None,
                    help="e.g. '0' for first CUDA GPU, or 'cpu'")
    ap.add_argument("--out", default="evidence",
                    help="output directory for evidence")
    ap.add_argument("--show", action="store_true",
                    help="show a live annotated window (press q to quit)")
    args = ap.parse_args()

    cfg = CameraConfig.load(args.config)
    run(parse_source(args.source), cfg, model_path=args.model,
        show=args.show, out_dir=args.out, conf=args.conf, device=args.device)


if __name__ == "__main__":
    main()
