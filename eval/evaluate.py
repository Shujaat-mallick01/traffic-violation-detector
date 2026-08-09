"""Score the pipeline's violations against human-labelled ground truth.

    python eval/evaluate.py --truth eval/ground_truth/cam1.csv \
                            --pred evidence/violations.csv

Without this you can only say "it found 80 things", which is not a claim about
accuracy: a rule that fires on every vehicle also finds every violation. The
number that matters is how many of those were real (precision) and how many real
ones were missed (recall).

Matching
--------
A prediction matches a ground-truth event when the type is identical and the
timestamps are within `--tolerance` seconds. The window exists because detection
necessarily lags the event: wrong-way needs ~0.3 s of evidence, illegal parking
needs its full dwell, and the tracker takes a few frames to lock on.

Matching is greedy nearest-first, not globally optimal. With events seconds
apart that is the same answer as optimal assignment; if you ever label events
denser than the tolerance window, tighten the tolerance rather than trusting it.

Ground-truth CSV format (one row per event you saw with your own eyes):

    timestamp_s,type,note
    14.2,wrong_way,white hatchback against traffic
    31.0,illegal_parking,van stops on the hard shoulder
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class Event(NamedTuple):
    timestamp: float
    type: str
    note: str = ""


class TypeScore(NamedTuple):
    type: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)


def read_events(path: str, ts_field="timestamp_s", type_field="type") -> List[Event]:
    """Read an events CSV. Blank lines and '#' comments are ignored.

    Hand-labelled files accumulate notes-to-self, so tolerating comments is
    cheaper than watching someone's first scoring run die on one.
    """
    rows: List[Event] = []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
        for i, row in enumerate(csv.DictReader(lines), start=2):
            if ts_field not in row or type_field not in row:
                raise SystemExit(
                    f"{path}: line {i} is missing '{ts_field}' or '{type_field}'. "
                    f"Found columns: {list(row)}")
            try:
                ts = float(row[ts_field])
            except (TypeError, ValueError):
                raise SystemExit(f"{path}: line {i} has a non-numeric timestamp "
                                 f"{row[ts_field]!r}")
            rows.append(Event(ts, (row[type_field] or "").strip(),
                              (row.get("note") or "").strip()))
    return rows


def match(truth: List[Event], pred: List[Event], tolerance: float):
    """Greedy nearest match within `tolerance`. Returns (pairs, missed, spurious)."""
    remaining = sorted(pred, key=lambda e: e.timestamp)
    used = [False] * len(remaining)
    pairs, missed = [], []

    for gt in sorted(truth, key=lambda e: e.timestamp):
        best, best_gap = None, None
        for i, p in enumerate(remaining):
            if used[i] or p.type != gt.type:
                continue
            gap = abs(p.timestamp - gt.timestamp)
            if gap <= tolerance and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best is None:
            missed.append(gt)
        else:
            used[best] = True
            pairs.append((gt, remaining[best], best_gap))

    spurious = [p for i, p in enumerate(remaining) if not used[i]]
    return pairs, missed, spurious


def score(truth, pred, tolerance) -> Dict[str, TypeScore]:
    pairs, missed, spurious = match(truth, pred, tolerance)
    types = {e.type for e in truth} | {e.type for e in pred}
    tp = defaultdict(int); fn = defaultdict(int); fp = defaultdict(int)
    for gt, _, _ in pairs:
        tp[gt.type] += 1
    for e in missed:
        fn[e.type] += 1
    for e in spurious:
        fp[e.type] += 1
    return {t: TypeScore(t, tp[t], fp[t], fn[t]) for t in sorted(types)}


def _pct(v: Optional[float]) -> str:
    return "   n/a" if v is None else f"{v * 100:5.1f}%"


def report(scores: Dict[str, TypeScore], tolerance: float) -> None:
    print(f"\nMatching window: +/-{tolerance:.1f}s\n")
    print(f"{'violation type':<22}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'precision':>11}{'recall':>9}{'F1':>9}")
    print("-" * 66)
    tt = tf = tn = 0
    for s in scores.values():
        tt += s.tp; tf += s.fp; tn += s.fn
        print(f"{s.type:<22}{s.tp:>5}{s.fp:>5}{s.fn:>5}"
              f"{_pct(s.precision):>11}{_pct(s.recall):>9}{_pct(s.f1):>9}")
    overall = TypeScore("OVERALL", tt, tf, tn)
    print("-" * 66)
    print(f"{'OVERALL':<22}{tt:>5}{tf:>5}{tn:>5}"
          f"{_pct(overall.precision):>11}{_pct(overall.recall):>9}"
          f"{_pct(overall.f1):>9}")

    if tt + tf + tn == 0:
        print("\nNothing to score — both files are empty.")
    elif tn and not tt:
        print("\nEvery labelled event was missed. Check that the config's zones "
              "cover\nwhere the events happen, and that the rule is enabled at all.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score violations against ground truth")
    ap.add_argument("--truth", required=True, help="hand-labelled CSV")
    ap.add_argument("--pred", required=True, help="pipeline's violations.csv")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="seconds a prediction may lag the labelled event")
    ap.add_argument("--verbose", action="store_true",
                    help="list every missed and spurious event")
    args = ap.parse_args()

    for p in (args.truth, args.pred):
        if not Path(p).exists():
            raise SystemExit(f"No such file: {p}")

    truth = read_events(args.truth)
    pred = read_events(args.pred)
    print(f"ground truth : {len(truth)} events from {args.truth}")
    print(f"predictions  : {len(pred)} events from {args.pred}")

    report(score(truth, pred, args.tolerance), args.tolerance)

    if args.verbose:
        _, missed, spurious = match(truth, pred, args.tolerance)
        if missed:
            print("\nMISSED (in ground truth, never detected):")
            for e in missed:
                print(f"  {e.timestamp:8.2f}s  {e.type:<20} {e.note}")
        if spurious:
            print("\nSPURIOUS (detected, not in ground truth):")
            for e in spurious:
                print(f"  {e.timestamp:8.2f}s  {e.type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
