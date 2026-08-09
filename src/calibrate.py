"""Interactive zone calibration.

Grabs a frame from the source and lets you draw the fixed geometry for one camera
view, then saves it to a YAML the pipeline can load.

Controls
--------
  Mode keys:  p = no-parking polygon   s = stop line (2 clicks)
              l = lane (polygon, then 2 clicks for allowed direction)
              g = light region (2 clicks: opposite corners)
  Left click: add a point
  n         : finish the current shape (start another of the same kind)
  u         : undo last point
  w         : write YAML (and keep editing)
  q         : quit

Lane workflow: press 'l', click the lane polygon, press 'n', then click 2 points
for the allowed travel direction (arrow start -> end), press 'n' again.

Run:
  python -m src.calibrate --source sample.mp4 --out config/cam1.yaml
"""
from __future__ import annotations

import argparse

from .zones import CameraConfig, Lane, LightRegion, NoParkingZone, StopLine


def grab_frame(source):
    import cv2

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {source}")
    return frame


def main():
    import cv2

    ap = argparse.ArgumentParser(description="Draw camera zones -> YAML")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="camera")
    args = ap.parse_args()

    frame = grab_frame(args.source)
    cfg = CameraConfig(name=args.name)
    state = {"mode": "p", "points": [], "pending_lane_polygon": None}
    colors = {"p": (0, 165, 255), "s": (0, 0, 255),
              "l": (255, 128, 0), "g": (0, 255, 255)}

    def redraw():
        img = frame.copy()
        for z in cfg.no_parking:
            pts = [(int(x), int(y)) for x, y in z.polygon]
            for i in range(len(pts)):
                cv2.line(img, pts[i], pts[(i + 1) % len(pts)], colors["p"], 2)
        for s in cfg.stop_lines:
            cv2.line(img, (int(s.a[0]), int(s.a[1])),
                     (int(s.b[0]), int(s.b[1])), colors["s"], 2)
        for lane in cfg.lanes:
            pts = [(int(x), int(y)) for x, y in lane.polygon]
            for i in range(len(pts)):
                cv2.line(img, pts[i], pts[(i + 1) % len(pts)], colors["l"], 1)
        for lr in cfg.light_regions:
            x1, y1, x2, y2 = [int(v) for v in lr.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), colors["g"], 2)
        for p in state["points"]:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, colors[state["mode"]], -1)
        cv2.putText(img, f"mode={state['mode']}  p/s/l/g  n=next u=undo w=write q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("calibrate", img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append((float(x), float(y)))
            redraw()

    def finish_shape():
        pts = state["points"]
        m = state["mode"]
        if m == "p" and len(pts) >= 3:
            cfg.no_parking.append(NoParkingZone(
                polygon=list(pts), name=f"no_parking_{len(cfg.no_parking) + 1}"))
        elif m == "s" and len(pts) >= 2:
            cfg.stop_lines.append(StopLine(
                a=pts[0], b=pts[1], name=f"stopline_{len(cfg.stop_lines) + 1}"))
        elif m == "g" and len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cfg.light_regions.append(LightRegion(
                bbox=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                name=f"light_{len(cfg.light_regions) + 1}"))
        elif m == "l":
            if state["pending_lane_polygon"] is None and len(pts) >= 3:
                state["pending_lane_polygon"] = list(pts)
                print("Lane polygon captured — click 2 points for the allowed "
                      "direction, then press 'n'.")
            elif state["pending_lane_polygon"] is not None and len(pts) >= 2:
                d = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
                cfg.lanes.append(Lane(
                    polygon=state["pending_lane_polygon"], direction=d,
                    name=f"lane_{len(cfg.lanes) + 1}"))
                state["pending_lane_polygon"] = None
        state["points"] = []
        redraw()

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("p"), ord("s"), ord("l"), ord("g")):
            state["mode"] = chr(key)
            state["points"] = []
            state["pending_lane_polygon"] = None
            redraw()
        elif key == ord("u") and state["points"]:
            state["points"].pop()
            redraw()
        elif key == ord("n"):
            finish_shape()
        elif key == ord("w"):
            finish_shape()
            cfg.save(args.out)
            print(f"Saved {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
