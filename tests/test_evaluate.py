"""Tests for the eval harness. No dependencies.

A scoring tool that is quietly wrong is worse than having none — it produces
confident numbers you will believe. These pin down the matching rules.

    python tests/test_evaluate.py     # or: pytest tests/
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from _harness import run_tests  # noqa: E402  (also puts the project root on sys.path)

# Loaded by path rather than `import eval.evaluate`: the directory is named
# `eval`, and a top-level module by that name is needless confusion.
_spec = importlib.util.spec_from_file_location(
    "violation_eval", Path(__file__).resolve().parents[1] / "eval" / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

E = ev.Event
WW, IP = "wrong_way", "illegal_parking"


def s(truth, pred, tol=2.0):
    return ev.score(truth, pred, tol)


def test_perfect_match_scores_100_percent():
    truth = [E(10.0, WW), E(20.0, IP)]
    pred = [E(10.3, WW), E(20.4, IP)]
    r = s(truth, pred)
    assert r[WW].tp == 1 and r[WW].fp == 0 and r[WW].fn == 0
    assert r[IP].tp == 1
    assert r[WW].precision == 1.0 and r[WW].recall == 1.0


def test_missed_event_counts_as_a_false_negative():
    r = s([E(10.0, WW), E(50.0, WW)], [E(10.1, WW)])
    assert (r[WW].tp, r[WW].fn, r[WW].fp) == (1, 1, 0)
    assert r[WW].recall == 0.5
    assert r[WW].precision == 1.0


def test_spurious_detection_counts_as_a_false_positive():
    r = s([E(10.0, WW)], [E(10.1, WW), E(80.0, WW)])
    assert (r[WW].tp, r[WW].fn, r[WW].fp) == (1, 0, 1)
    assert r[WW].precision == 0.5
    assert r[WW].recall == 1.0


def test_type_must_agree():
    """A wrong-way detection must never be credited to a parking event."""
    r = s([E(10.0, IP)], [E(10.0, WW)])
    assert r[IP].tp == 0 and r[IP].fn == 1
    assert r[WW].fp == 1


def test_tolerance_boundary():
    assert s([E(10.0, WW)], [E(11.9, WW)], tol=2.0)[WW].tp == 1
    assert s([E(10.0, WW)], [E(12.1, WW)], tol=2.0)[WW].tp == 0


def test_one_prediction_cannot_satisfy_two_events():
    """Two real violations and one detection is 50% recall, not 100%."""
    r = s([E(10.0, WW), E(10.5, WW)], [E(10.2, WW)], tol=2.0)
    assert r[WW].tp == 1, "a single prediction was double-counted"
    assert r[WW].fn == 1


def test_nearest_prediction_wins():
    truth = [E(10.0, WW)]
    pred = [E(11.8, WW), E(10.1, WW)]
    pairs, _, spurious = ev.match(truth, pred, 2.0)
    assert len(pairs) == 1
    assert pairs[0][1].timestamp == 10.1, "did not pick the closest prediction"
    assert spurious[0].timestamp == 11.8


def test_empty_inputs_do_not_crash_or_invent_scores():
    assert s([], []) == {}
    r = s([E(10.0, WW)], [])
    assert r[WW].fn == 1 and r[WW].precision is None   # no predictions to be right
    r = s([], [E(10.0, WW)])
    assert r[WW].fp == 1 and r[WW].recall is None      # nothing real to find


def test_f1_is_none_when_either_side_is_zero():
    assert s([E(10.0, WW)], [E(99.0, WW)])[WW].f1 is None


def test_reader_skips_comments_and_blank_lines():
    """The shipped ground-truth template carries '#' notes; reading must survive."""
    import tempfile
    csv_text = (
        "timestamp_s,type,note\n"
        "# a note to self\n"
        "\n"
        "14.2,wrong_way,white hatchback\n"
        "   # indented comment\n"
        "31.0,illegal_parking,van on the shoulder\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(csv_text)
        path = fh.name
    events = ev.read_events(path)
    Path(path).unlink()
    assert len(events) == 2, f"expected 2 events, got {len(events)}"
    assert events[0].timestamp == 14.2 and events[0].type == WW
    assert events[1].type == IP


def test_shipped_template_is_readable():
    tpl = Path(__file__).resolve().parents[1] / "eval" / "ground_truth" / "template.csv"
    events = ev.read_events(str(tpl))
    assert events, "template parsed to zero events"
    assert {e.type for e in events} <= {WW, IP, "red_light"}, (
        f"template uses a type no rule emits: {[e.type for e in events]}")


if __name__ == "__main__":
    sys.exit(run_tests(dict(globals()), "evaluate"))
