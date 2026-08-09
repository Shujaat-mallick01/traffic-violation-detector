# Evaluation

"It found 80 violations" is not an accuracy claim. A rule that fires on every
vehicle also finds every violation. The two numbers that mean something are:

- **Precision** — of the things it flagged, how many were real?
- **Recall** — of the real violations, how many did it catch?

For enforcement, **precision matters more**. A missed violation is a missed
ticket. A false positive is an accusation against someone who did nothing, and
enough of those destroy trust in the whole system. Tune accordingly, and keep
human review in the loop regardless.

## Workflow

```bash
# 1. run the pipeline over a clip
python run.py --source samples/cars2.mp4 --config config/cars2_demo.yaml \
              --out evidence/cars2

# 2. score it against what you labelled by hand
python eval/evaluate.py --truth eval/ground_truth/cars2.csv \
                        --pred evidence/cars2/violations.csv --verbose
```

`--verbose` lists every missed and every spurious event, which is what you
actually act on — the summary table only tells you whether things got better.

## Labelling ground truth

Watch the clip and write down every violation you see, with the timestamp it
becomes unambiguous to a human. One row per event:

```csv
timestamp_s,type,note
14.2,wrong_way,white hatchback against traffic in lane_1
31.0,illegal_parking,van stops on the hard shoulder
```

`type` must match the rule names exactly: `wrong_way`, `illegal_parking`,
`red_light`.

Rules to keep the numbers honest:

- **Label before you look at the output.** Labelling after is how you end up
  agreeing with whatever the pipeline produced.
- **Label everything, including what the model has no chance of catching** —
  a vehicle too distant or too occluded still happened. Excluding it inflates
  recall and hides the system's real limits.
- **Timestamps are approximate**, which is why matching uses a tolerance window.
  Detection necessarily lags: wrong-way needs ~0.3 s of evidence, illegal
  parking needs its full dwell, and the tracker takes a few frames to lock on.
  The default `--tolerance 2.0` covers that. Do not widen it to make results
  look better — a wide window starts crediting coincidences.

## Splitting data without fooling yourself

The most common way to produce a great-looking number that means nothing is
leakage: near-duplicate data in both the set you tuned on and the set you
report on.

**Split by clip and by camera, never by frame.** Consecutive frames are almost
identical; putting some in "tune" and others in "test" means you are testing on
data you effectively already saw. Two clips from the same camera at the same
hour are barely better.

Practical discipline for this project:

1. **Tune set** — clips you may replay endlessly while adjusting thresholds,
   zones, `--imgsz`, `--conf`.
2. **Held-out set** — clips from a *different* time of day and ideally a
   different camera, that you score against **once**, when you think you are
   done. If you tune after looking, it is no longer held out; retire it and cut
   a new one.

Report per-type metrics, not one aggregate. A system with excellent
illegal-parking numbers and useless wrong-way numbers averages out to
"acceptable", which tells you nothing about where to work next.

## What is not measured here

This harness scores **violations**, the end of the pipeline. It cannot tell you
*why* something was missed. When recall is poor, the cause is usually upstream:

- the detector never found the vehicle (small, distant, occluded, motion-blurred),
- the tracker fragmented its identity so no rule accumulated enough evidence,
- or the zone geometry does not cover where the event happened.

Check those in that order before touching a rule threshold. Running with
`--show` on the failing clip usually makes it obvious within seconds.

`ground_truth/` is where labelled CSVs live. `template.csv` is a starting point;
copy it per clip and name it after the clip.
