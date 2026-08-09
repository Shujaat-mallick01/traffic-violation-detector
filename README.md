# Traffic Violation Detector

Detect traffic violations from recorded video or live CCTV/RTSP feeds. Built as a
four-stage pipeline (detect → track → apply rules → save evidence) so violation
logic never touches raw model output and the detector stays swappable.

This is **Phase 1**: the foundation, plus the two violations that work with no
custom data. The roadmap below shows what comes next.

---

## What works right now (zero custom data)

Using YOLO26's COCO-pretrained classes (car, motorcycle, bus, truck, person,
traffic light) plus per-camera geometry:

- **Illegal parking / stopping** — a vehicle stationary inside a no-parking zone
  beyond a time threshold.
- **Wrong-way driving** — a vehicle moving against a lane's allowed direction.

## What's scaffolded

- **Red-light running** (Phase 2) — the vehicle-crossing logic is implemented and
  runs; the missing piece is a *reliable* traffic-light state. A simple HSV
  color estimator is included as a starting point but needs tuning / a small
  classifier on real footage.
- **No-helmet, triple-riding** (Phase 3) — needs a custom-trained detector
  (COCO has no "helmet" class). Interfaces + training plan are in
  `src/violations/rider_safety.py`.
- **No-seatbelt** — hardest; needs a high-res frontal windshield camera, not a
  standard overhead view. Treat as a separate track.

---

## Architecture

```
video / RTSP ──▶ Detector (YOLO26 + ByteTrack) ──▶ tracks (persistent IDs)
                                                        │
                     CameraConfig (per-view zones) ─────┤
                                                        ▼
                                              Rule engine (stateful)
                                     illegal_parking · wrong_way · red_light
                                                        │
                                                        ▼
                                       EvidenceWriter (snapshot + CSV log)
```

- `src/detector.py` — the only file that knows about Ultralytics. Swap it to move
  to another detector (e.g. RF-DETR) without touching anything downstream.
- `src/zones.py` + `src/calibrate.py` — per-camera geometry. **Every camera view
  needs its zones drawn once**; the rules mean nothing without them.
- `src/violations/` — one file per rule, each stateful and self-debouncing.
- `src/pipeline.py` — the per-frame loop wiring it together.
- `tools/derive_lane.py` — headless lane bootstrap, for when the GUI is not an
  option. `eval/` — precision/recall scoring. `tests/` — 42 dependency-free tests.

Every rule threshold is expressed in **seconds**, never frames, so `--stride`
changes throughput without changing behaviour. Keep it that way when adding rules.

---

## Install

```bash
pip install -r requirements.txt
```

On a GPU machine, install a CUDA build of PyTorch first (see pytorch.org) so
Ultralytics uses the GPU. On first run, model weights download automatically.

## 1. Calibrate a camera

Draw the no-parking zones, stop lines, and lane directions once per view:

```bash
python -m src.calibrate --source sample.mp4 --out config/cam1.yaml
```

Keys: `p` no-parking polygon · `s` stop line · `l` lane (+direction) ·
`g` light region · `n` finish shape · `u` undo · `w` write · `q` quit.
See `config/camera_example.yaml` for the resulting format.

**This needs a desktop session** — it opens an OpenCV window, so it will not run
over a plain SSH connection or in a container. Calibrate locally, ship the YAML.

### Bootstrapping a config without the GUI

For a headless box, or just to get a rough config for a new clip in seconds:

```bash
python tools/derive_lane.py --source samples/cars2.mp4 --out config/cam1.yaml
```

It tracks the traffic, keeps the vehicles flowing in the dominant direction, and
fits a lane around where they actually drove. On a divided road that isolates one
carriageway — which is what you want, since a lane spanning both directions flags
all oncoming traffic as wrong-way.

Inspect the result before trusting it: it only knows about lanes that had traffic
during the clip. Add `--reverse` to write the direction backwards on purpose —
ordinary traffic then trips the wrong-way rule, which is a useful positive
control proving the rule fires at all rather than being silent for some other
reason.

### Sample clips

`samples/` is gitignored (fetched binaries, not authored). To reproduce the demo,
fetch a public clip and derive its config:

```bash
mkdir -p samples
curl -L -o samples/cars2.mp4 \
  https://raw.githubusercontent.com/Abdelrahman5ames/Vehicle-Detection-Counting/HEAD/project/cars2.mp4
python run.py --source samples/cars2.mp4 --config config/cars2_demo.yaml --stride 3 --show
```

`config/cars2_demo.yaml` is committed, so the demo runs without re-deriving
geometry. These clips are fine for exercising the pipeline; they are **not**
licensed for training data or redistribution, and at 640×360 they are well below
real CCTV resolution.

## 2. Run

```bash
# recorded file, with a live annotated window
python run.py --source sample.mp4 --config config/cam1.yaml --show

# live CCTV / RTSP stream
python run.py --source "rtsp://user:pass@host:554/stream" --config config/cam1.yaml

# GPU + YOLO11 instead of YOLO26
python run.py --source sample.mp4 --config config/cam1.yaml --model yolo11n.pt --device 0
```

### Running on a CPU-only machine

Workable for prototyping on **recorded clips**. Live RTSP will not keep up — the
capture buffer grows faster than you can drain it and latency runs away.

```bash
python run.py --source sample.mp4 --config config/cam1.yaml --show \
              --stride 3 --imgsz 416
```

- `--stride N` detects on every Nth frame. Near-linear speedup.
- `--imgsz` sets inference resolution (default 640). Cost is roughly quadratic,
  so 416 is about 2× faster — at the cost of small and distant objects, which is
  exactly what a traffic camera is full of. Check recall before keeping it.

Measured with `yolo26n.pt` on a CPU-only laptop (Python 3.13, torch 2.13+cpu):

| `--imgsz` | throughput |
|---|---|
| 640 (default) | ~16 FPS |
| 416 | ~29 FPS |

Treat those as an **upper bound**: they were measured on an empty frame, so no
boxes were detected and no tracks associated. Real traffic footage — dozens of
vehicles, continuous tracker association — runs materially slower. Benchmark on
your own clip before assuming a stride setting keeps up.

Skipped frames are still *grabbed*, so `frame_idx` keeps counting true source
frames and **every timestamp stays wall-accurate at any stride**. This is why
rule thresholds are written in seconds rather than frames: `--stride 3` used to
mean wrong-way needed 3× as much real evidence before firing, silently. If you
add a rule, express its thresholds in seconds too — `tests/test_wrong_way.py`
asserts that detection time is stride-invariant, and will catch you if you don't.

## Measuring accuracy

"It found 80 violations" is not an accuracy claim — a rule that fires on every
vehicle also finds every violation. Score against hand-labelled ground truth:

```bash
python eval/evaluate.py --truth eval/ground_truth/cars2.csv \
                        --pred evidence/violations.csv --verbose
```

Reports per-type precision, recall and F1. **For enforcement, precision matters
more than recall**: a miss costs a ticket, a false positive accuses someone who
did nothing. See `eval/README.md` for how to label honestly and how to split
clips so you do not leak your test set into your tuning.

## Tests

```bash
python tests/run_all.py            # or: pytest tests/
```

42 tests over the geometry primitives, all three active rules, and the eval
harness. They need no model, no weights and no network — that logic is
deliberately dependency-free, so the suite runs on a bare interpreter. Only the
four HSV light-state tests touch OpenCV, and they skip themselves if it is absent.

The suite is written to pin down the behaviour that is easy to break silently:

- detection time is **stride-invariant** (`test_wrong_way`)
- stationary time is never counted as evidence of moving
- an unknown light state **never** produces a red-light violation
- a lone red pixel never sets the light state to red
- each rule emits **one** violation per track, not one per frame
- per-track state is pruned, so a 24/7 stream does not leak memory

## Output

- `evidence/snapshots/*.jpg` — one annotated frame per violation.
- `evidence/violations.csv` — `timestamp_s, type, track_id, class, frame_idx, detail, snapshot`.

---

## Roadmap

| Phase | Adds | Needs |
|------|------|-------|
| 1 ✅ | pipeline, calibration, illegal-parking, wrong-way | nothing (COCO) |
| 2 | reliable red-light | traffic-light state classifier on real footage |
| 3 | no-helmet, triple-riding | collected + annotated local data, fine-tuned YOLO26 |
| 4 | plate crop + OCR on each violation (enforcement-grade evidence) | ALPR / OCR model |
| 5 | scale to cloud GPU | multi-stream batching, ONNX/TensorRT export |

---

## Phase 3 — data & training (the real bottleneck)

For helmet / triple-riding you need a custom detector. Plan:

1. **Classes:** `motorcycle, rider, helmet, no-helmet` (keep `person` too).
2. **Bootstrap** from public helmet-detection datasets to get moving fast.
3. **Add local data.** Karachi traffic — rider density, lighting, plate styles —
   differs enough from Western datasets that some local footage materially
   improves accuracy. Collect from the same camera angles you'll deploy on.
4. **Annotate** in a tool that exports YOLO format (one `.txt` per image).
5. **Train / fine-tune** YOLO26; for small distant objects consider the P2
   small-object head variant. Track mAP@0.5 and mAP@0.5:0.95, per-class, on a
   held-out set from footage the model never trained on.

Good practices baked into the plan: honest train/val/test split with **no leakage**
(don't let frames from the same clip land in both train and test), transfer
learning from pretrained weights, sensible augmentation, and per-class metrics
rather than a single aggregate number.

---

## Note on model licensing

Ultralytics YOLO (incl. YOLO26 / YOLO11) is **AGPL-3.0**. Fine for prototyping and
internal use. If this becomes a commercial or government enforcement product, the
copyleft may not fit — the main permissive alternative is **RF-DETR** (Apache 2.0).
Because the detector is isolated in `src/detector.py`, switching later is a
contained change.

## Honest limitations

- **No accuracy has been measured yet.** The pipeline has been verified to run
  end-to-end and to fire correctly under a reversed-lane positive control, but
  nobody has labelled ground truth, so there are no precision/recall numbers.
  `eval/` exists to produce them; until it has been run on labelled clips, treat
  every claim about how well this works as unverified.
- **Small, distant and blurred vehicles track poorly**, and a rule can only fire
  on a track that survives long enough to accumulate evidence. Observed directly:
  on a 640×360 clip, vehicles on the far carriageway fragmented badly enough that
  wrong-way missed them entirely. Expect recall to fall off with distance.
- Accuracy depends heavily on camera angle, resolution, and correct calibration.
- The HSV light-state estimator is a placeholder, not production-ready. It now
  refuses to guess — it returns "unknown" unless one colour clearly dominates,
  and the rule emits nothing on unknown — but refusing to guess is not the same
  as being right. Replace it with a classifier trained on real signal crops
  before enabling red-light enforcement.
- Occlusion and dense traffic degrade tracking; expect tuning per site.
- None of this is a substitute for legally admissible evidence standards — treat
  outputs as *candidates* for human review, especially for enforcement.
