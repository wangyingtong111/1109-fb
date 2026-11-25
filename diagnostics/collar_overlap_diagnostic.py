#!/usr/bin/env python
"""Rerunnable collar / skip-overlap domain diagnostic for DetectionErrorRate.

Run from the repository root:

    .venv\\Scripts\\python.exe diagnostics\\collar_overlap_diagnostic.py
    # or
    uv run python diagnostics/collar_overlap_diagnostic.py

The script does not change pyannote.metrics scoring rules.  It:

1. runs the real ``DetectionErrorRate`` implementation;
2. exposes the UEM/domain intervals actually used after ``uemify``;
3. independently recomputes support, FA and miss durations with small
   half-open interval operations from the raw annotation segments;
4. asserts that both paths and the hand-derived expected slices agree.
"""

from __future__ import annotations

import sys
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from pyannote.core import Annotation, Segment, Timeline  # noqa: E402
from pyannote.metrics.detection import (  # noqa: E402
    DER_FALSE_ALARM,
    DER_MISS,
    DER_NAME,
    DER_TOTAL,
    DetectionErrorRate,
)

Number = int | float | Fraction
Interval = tuple[Fraction, Fraction]
SegmentSpec = tuple[str, Number, Number]


# ---------------------------------------------------------------------------
# Independent interval arithmetic (deliberately not UEMSupportMixin.uemify)
# ---------------------------------------------------------------------------


def frac(value: Number) -> Fraction:
    if isinstance(value, Fraction):
        return value
    # Fraction(str(0.5)) avoids binary-float representation noise.
    return Fraction(str(value))


def interval(start: Number, end: Number) -> Interval:
    return frac(start), frac(end)


def normalize(intervals: list[tuple[Number, Number] | Interval]) -> list[Interval]:
    """Merge ordered/overlapping/touching intervals into a support timeline."""
    ordered = sorted((frac(a), frac(b)) for a, b in intervals if frac(b) > frac(a))
    merged: list[Interval] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
    return merged


def intersect(left: list[Interval], right: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in normalize(left):
        for c, d in normalize(right):
            start, end = max(a, c), min(b, d)
            if start < end:
                out.append((start, end))
    return normalize(out)


def subtract(source: list[Interval], cutters: list[Interval]) -> list[Interval]:
    """Return source \\ cutters, preserving all resulting pieces."""
    out: list[Interval] = []
    for source_start, source_end in normalize(source):
        pieces: list[Interval] = [(source_start, source_end)]
        for cut_start, cut_end in normalize(cutters):
            next_pieces: list[Interval] = []
            for start, end in pieces:
                overlap_start, overlap_end = max(start, cut_start), min(end, cut_end)
                if overlap_start >= overlap_end:
                    next_pieces.append((start, end))
                    continue
                if start < overlap_start:
                    next_pieces.append((start, overlap_start))
                if overlap_end < end:
                    next_pieces.append((overlap_end, end))
            pieces = next_pieces
        out.extend(pieces)
    return normalize(out)


def duration(intervals: list[Interval]) -> Fraction:
    return sum((end - start for start, end in intervals), Fraction(0))


def fmt_time(value: Number) -> str:
    value = frac(value)
    return f"{float(value):g}"


def fmt_interval(item: Interval) -> str:
    start, end = item
    return f"[{fmt_time(start)},{fmt_time(end)}]"


def fmt_intervals(intervals: list[Interval], with_duration: bool = False) -> str:
    intervals = normalize(intervals)
    if not intervals:
        return "∅"
    text = ", ".join(fmt_interval(item) for item in intervals)
    if with_duration:
        text += f"  (duration={fmt_time(duration(intervals))})"
    return text


def fmt_rate(numerator: Fraction, denominator: Fraction) -> str:
    if denominator == 0:
        return "n/a (total=0)"
    rate = numerator / denominator
    return f"{rate.numerator}/{rate.denominator}={float(rate):.6g}"


# ---------------------------------------------------------------------------
# Scenario construction and independent recomputation
# ---------------------------------------------------------------------------

REFERENCE_A: list[SegmentSpec] = [("A", 2, 4)]
REFERENCE_AB: list[SegmentSpec] = [
    ("A", 2, 4),
    ("B", 3.25, 5.5),
]
HYPOTHESIS: list[SegmentSpec] = [("H", 0, 3)]
UEM_EXPLICIT: list[tuple[Number, Number]] = [(1, 6)]


def make_annotation(specs: list[SegmentSpec], uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    for track, start, end in specs:
        annotation[Segment(float(start), float(end)), track] = track
    return annotation


def implicit_domain(
    reference: list[SegmentSpec], hypothesis: list[SegmentSpec]
) -> list[Interval]:
    """Independent equivalent of the metric's r_extent | h_extent UEM fallback."""
    starts = [frac(s) for _, s, _ in reference + hypothesis]
    ends = [frac(e) for _, _, e in reference + hypothesis]
    return [(min(starts), max(ends))]


def collar_regions(reference: list[SegmentSpec], collar: Number) -> list[Interval]:
    half = frac(collar) / 2
    regions: list[Interval] = []
    for _, start, end in reference:
        start, end = frac(start), frac(end)
        regions.extend(
            [
                (start - half, start + half),
                (end - half, end + half),
            ]
        )
    return normalize(regions)


def overlap_regions(reference: list[SegmentSpec]) -> list[Interval]:
    """Intersections of distinct reference tracks/annotations."""
    regions: list[Interval] = []
    for i, (track1, s1, e1) in enumerate(reference):
        for track2, s2, e2 in reference[i + 1 :]:
            if track1 == track2:
                continue
            start, end = max(frac(s1), frac(s2)), min(frac(e1), frac(e2))
            if start < end:
                regions.append((start, end))
    return normalize(regions)


def evaluate_independently(
    reference: list[SegmentSpec],
    hypothesis: list[SegmentSpec],
    uem: list[tuple[Number, Number]] | None,
    collar: Number,
    skip_overlap: bool,
) -> dict[str, Any]:
    raw_domain = (
        implicit_domain(reference, hypothesis)
        if uem is None
        else normalize([interval(s, e) for s, e in uem])
    )

    collars = collar_regions(reference, collar) if frac(collar) > 0 else []
    overlaps = overlap_regions(reference) if skip_overlap else []
    excluded = normalize(collars + overlaps)

    domain = subtract(raw_domain, excluded)
    reference_support = normalize([(s, e) for _, s, e in reference])
    hypothesis_support = normalize([(s, e) for _, s, e in hypothesis])

    reference_included = intersect(reference_support, domain)
    hypothesis_included = intersect(hypothesis_support, domain)

    # Binary detection classes on the scored domain:
    #   speech = reference support; non-speech = D \\ R
    #   FA = H \\ R; miss = R \\ H.
    false_alarm = subtract(hypothesis_included, reference_included)
    miss = subtract(reference_included, hypothesis_included)

    total = duration(reference_included)
    fa_duration = duration(false_alarm)
    miss_duration = duration(miss)
    der = (
        Fraction(0)
        if total == 0 and fa_duration + miss_duration == 0
        else (fa_duration + miss_duration) / total
        if total != 0
        else Fraction(1)
    )

    return {
        "raw_domain": raw_domain,
        "collars": collars,
        "overlaps": overlaps,
        "excluded": excluded,
        "domain": domain,
        "reference": reference_included,
        "hypothesis": hypothesis_included,
        "false_alarm": false_alarm,
        "miss": miss,
        "total": total,
        "fa_duration": fa_duration,
        "miss_duration": miss_duration,
        "der": der,
    }


# Hand-derived oracle.  The script checks both the metric and the independent
# interval implementation against these exact slices/durations.
EXPECTED: dict[str, dict[str, Any]] = {
    "A-implicit": {
        "domain": [(0, 4)],
        "reference": [(2, 4)],
        "hypothesis": [(0, 3)],
        "false_alarm": [(0, 2)],
        "miss": [(3, 4)],
        "total": 2,
        "fa_duration": 2,
        "miss_duration": 1,
        "der": "1.5",
    },
    "A-explicit": {
        "domain": [(1, 6)],
        "reference": [(2, 4)],
        "hypothesis": [(1, 3)],
        "false_alarm": [(1, 2)],
        "miss": [(3, 4)],
        "total": 2,
        "fa_duration": 1,
        "miss_duration": 1,
        "der": "1",
    },
    "AB-baseline": {
        "domain": [(1, 6)],
        "reference": [(2, 5.5)],
        "hypothesis": [(1, 3)],
        "false_alarm": [(1, 2)],
        "miss": [(3, 5.5)],
        "total": 3.5,
        "fa_duration": 1,
        "miss_duration": 2.5,
        "der": "1",
    },
    "AB-collar-only": {
        "domain": [
            (1, 1.75),
            (2.25, 3),
            (3.5, 3.75),
            (4.25, 5.25),
            (5.75, 6),
        ],
        "reference": [(2.25, 3), (3.5, 3.75), (4.25, 5.25)],
        "hypothesis": [(1, 1.75), (2.25, 3)],
        "false_alarm": [(1, 1.75)],
        "miss": [(3.5, 3.75), (4.25, 5.25)],
        "total": 2,
        "fa_duration": 0.75,
        "miss_duration": 1.25,
        "der": "1",
    },
    "AB-skip-overlap-only": {
        "domain": [(1, 3.25), (4, 6)],
        "reference": [(2, 3.25), (4, 5.5)],
        "hypothesis": [(1, 3)],
        "false_alarm": [(1, 2)],
        "miss": [(3, 3.25), (4, 5.5)],
        "total": 2.75,
        "fa_duration": 1,
        "miss_duration": 1.75,
        "der": "1",
    },
    "AB-both": {
        "domain": [(1, 1.75), (2.25, 3), (4.25, 5.25), (5.75, 6)],
        "reference": [(2.25, 3), (4.25, 5.25)],
        "hypothesis": [(1, 1.75), (2.25, 3)],
        "false_alarm": [(1, 1.75)],
        "miss": [(4.25, 5.25)],
        "total": 1.75,
        "fa_duration": 0.75,
        "miss_duration": 1,
        "der": "1",
    },
}

CASES: list[dict[str, Any]] = [
    {
        "id": "A-implicit",
        "heading": "Reference A=[2,4], H=[0,3], implicit UEM",
        "reference": REFERENCE_A,
        "uem": None,
        "collar": 0.0,
        "skip_overlap": False,
    },
    {
        "id": "A-explicit",
        "heading": "Reference A=[2,4], H=[0,3], explicit UEM=[1,6]",
        "reference": REFERENCE_A,
        "uem": UEM_EXPLICIT,
        "collar": 0.0,
        "skip_overlap": False,
    },
    {
        "id": "AB-baseline",
        "heading": "A=[2,4] + B=[3.25,5.5], UEM=[1,6], no exclusion (bridge)",
        "reference": REFERENCE_AB,
        "uem": UEM_EXPLICIT,
        "collar": 0.0,
        "skip_overlap": False,
    },
    {
        "id": "AB-collar-only",
        "heading": "A+B, UEM=[1,6], collar=0.5 only",
        "reference": REFERENCE_AB,
        "uem": UEM_EXPLICIT,
        "collar": 0.5,
        "skip_overlap": False,
    },
    {
        "id": "AB-skip-overlap-only",
        "heading": "A+B, UEM=[1,6], skip_overlap=True only",
        "reference": REFERENCE_AB,
        "uem": UEM_EXPLICIT,
        "collar": 0.0,
        "skip_overlap": True,
    },
    {
        "id": "AB-both",
        "heading": "A+B, UEM=[1,6], collar=0.5 and skip_overlap=True",
        "reference": REFERENCE_AB,
        "uem": UEM_EXPLICIT,
        "collar": 0.5,
        "skip_overlap": True,
    },
]


def assert_intervals_equal(
    actual: list[Interval], expected: list[tuple[Number, Number]], label: str, case_id: str
) -> None:
    actual_n = normalize(actual)
    expected_n = normalize([interval(s, e) for s, e in expected])
    if len(actual_n) != len(expected_n):
        raise AssertionError(
            f"{case_id}: {label} mismatch\n  actual={fmt_intervals(actual_n)}\n"
            f"  expected={fmt_intervals(expected_n)}"
        )
    for (a, b), (c, d) in zip(actual_n, expected_n):
        if abs(float(a - c)) > 1e-10 or abs(float(b - d)) > 1e-10:
            raise AssertionError(
                f"{case_id}: {label} endpoint mismatch: "
                f"{fmt_interval((a,b))} != {fmt_interval((c,d))}"
            )


def run_real_metric(case: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    reference = make_annotation(case["reference"], uri=case["id"])
    hypothesis = make_annotation(HYPOTHESIS, uri=case["id"])
    uem = (
        None
        if case["uem"] is None
        else Timeline(segments=[Segment(float(s), float(e)) for s, e in case["uem"]])
    )

    metric = DetectionErrorRate(
        collar=float(case["collar"]), skip_overlap=bool(case["skip_overlap"])
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        details = dict(
            metric(reference, hypothesis, uem=uem, detailed=True)
        )
        cropped_r, cropped_h, actual_uem = metric.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=float(case["collar"]),
            skip_overlap=bool(case["skip_overlap"]),
            returns_uem=True,
        )

    actual_domain = [(frac(s.start), frac(s.end)) for s in actual_uem]

    reference_support = cropped_r.get_timeline(copy=False).support()
    hypothesis_support = cropped_h.get_timeline(copy=False).support()
    reference_gaps = reference_support.gaps(support=actual_uem)
    hypothesis_gaps = hypothesis_support.gaps(support=actual_uem)

    actual_fa: list[Interval] = []
    for gap, hyp_segment in reference_gaps.co_iter(hypothesis_support):
        overlap = gap & hyp_segment
        if overlap:
            actual_fa.append((frac(overlap.start), frac(overlap.end)))

    actual_miss: list[Interval] = []
    for ref_segment, gap in reference_support.co_iter(hypothesis_gaps):
        overlap = ref_segment & gap
        if overlap:
            actual_miss.append((frac(overlap.start), frac(overlap.end)))

    warning_text = "; ".join(str(w.message) for w in caught)
    interval_evidence = {
        "domain": normalize(actual_domain),
        "reference": normalize(
            [(s.start, s.end) for s in reference_support]
        ),
        "hypothesis": normalize(
            [(s.start, s.end) for s in hypothesis_support]
        ),
        "false_alarm": normalize(actual_fa),
        "miss": normalize(actual_miss),
        "warning": warning_text,
    }
    return details, interval_evidence


def check_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["id"]
    independent = evaluate_independently(
        case["reference"],
        HYPOTHESIS,
        case["uem"],
        case["collar"],
        case["skip_overlap"],
    )
    details, actual = run_real_metric(case)
    expected = EXPECTED[case_id]

    # Hand-derived oracle vs independent interval arithmetic.
    for key in [
        "domain",
        "reference",
        "hypothesis",
        "false_alarm",
        "miss",
    ]:
        assert_intervals_equal(independent[key], expected[key], key, case_id)

    for key in ["total", "fa_duration", "miss_duration"]:
        if abs(float(independent[key] - frac(expected[key]))) > 1e-10:
            raise AssertionError(f"{case_id}: independent {key} is wrong")
    if abs(float(independent["der"] - frac(expected["der"]))) > 1e-10:
        raise AssertionError(f"{case_id}: independent DER is wrong")

    # Real metric intervals vs oracle.
    for key in [
        "domain",
        "reference",
        "hypothesis",
        "false_alarm",
        "miss",
    ]:
        assert_intervals_equal(actual[key], expected[key], f"metric {key}", case_id)

    # Real metric scalar components vs independent recomputation.
    scalar_pairs = {
        DER_TOTAL: independent["total"],
        DER_FALSE_ALARM: independent["fa_duration"],
        DER_MISS: independent["miss_duration"],
    }
    for metric_key, independent_value in scalar_pairs.items():
        if abs(float(details[metric_key] - independent_value)) > 1e-10:
            raise AssertionError(
                f"{case_id}: metric {metric_key}={details[metric_key]} != "
                f"independent {float(independent_value)}"
            )

    if abs(float(details[DER_NAME] - independent["der"])) > 1e-10:
        raise AssertionError(f"{case_id}: metric DER does not match")

    return {"independent": independent, "metric": details, "actual": actual}


def print_case(case: dict[str, Any], result: dict[str, Any]) -> None:
    ind = result["independent"]
    metric = result["metric"]
    print("=" * 88)
    print(f"[{case['id']}] {case['heading']}")
    if result["actual"]["warning"]:
        print(f"  metric warning : {result['actual']['warning']}")
    print(f"  scored domain  : {fmt_intervals(ind['domain'])}")
    print(f"  ref support R  : {fmt_intervals(ind['reference'], with_duration=True)}")
    print(f"  hyp support H  : {fmt_intervals(ind['hypothesis'])}")
    print(f"  false alarm H\\R: {fmt_intervals(ind['false_alarm'], with_duration=True)}")
    print(f"  miss R\\H       : {fmt_intervals(ind['miss'], with_duration=True)}")
    print(
        "  metric output  : "
        f"total={metric[DER_TOTAL]:g}, FA={metric[DER_FALSE_ALARM]:g}, "
        f"miss={metric[DER_MISS]:g}, miss_rate="
        f"{fmt_rate(ind['miss_duration'], ind['total'])}, "
        f"DER={metric[DER_NAME]:g}"
    )
    print(
        "  check          : real DetectionErrorRate == independent recomputation "
        "== hand-derived slices [OK]"
    )


def print_exclusion_partition() -> None:
    uem = normalize([interval(1, 6)])
    collars = intersect(collar_regions(REFERENCE_AB, 0.5), uem)
    overlaps = intersect(overlap_regions(REFERENCE_AB), uem)

    collar_only = subtract(collars, overlaps)
    overlap_only = subtract(overlaps, collars)
    joint = intersect(collars, overlaps)
    union = normalize(collar_only + overlap_only + joint)

    expected = {
        "collars": [(1.75, 2.25), (3, 3.5), (3.75, 4.25), (5.25, 5.75)],
        "overlaps": [(3.25, 4)],
        "collar_only": [
            (1.75, 2.25),
            (3, 3.25),
            (4, 4.25),
            (5.25, 5.75),
        ],
        "overlap_only": [(3.5, 3.75)],
        "joint": [(3.25, 3.5), (3.75, 4)],
        "union": [(1.75, 2.25), (3, 4.25), (5.25, 5.75)],
    }
    actual = {
        "collars": collars,
        "overlaps": overlaps,
        "collar_only": collar_only,
        "overlap_only": overlap_only,
        "joint": joint,
        "union": union,
    }
    for key, value in expected.items():
        assert_intervals_equal(actual[key], value, key, "AB-exclusion-partition")

    raw_annotation_duration = sum(
        (frac(e) - frac(s) for _, s, e in REFERENCE_AB), Fraction(0)
    )
    support_duration = duration(normalize([(s, e) for _, s, e in REFERENCE_AB]))

    print("=" * 88)
    print("[AB-exclusion-partition] Explicit UEM=[1,6], collar=0.5, reference overlap")
    print(f"  collar C             : {fmt_intervals(collars, with_duration=True)}")
    print(f"  overlap O            : {fmt_intervals(overlaps, with_duration=True)}")
    print(f"  collar-only C\\O      : {fmt_intervals(collar_only, with_duration=True)}")
    print(f"  overlap-only O\\C     : {fmt_intervals(overlap_only, with_duration=True)}")
    print(f"  joint C∩O            : {fmt_intervals(joint, with_duration=True)}")
    print(f"  union C∪O (dedup)    : {fmt_intervals(union, with_duration=True)}")
    print(
        "  reference annotation durations sum to "
        f"{fmt_time(raw_annotation_duration)}s; detection support is "
        f"{fmt_time(support_duration)}s because O is counted once."
    )
    print(
        "  skip_overlap=True extrudes O before DER_TOTAL; that existing metric "
        "behavior is left unchanged."
    )
    print("  partition oracle check [OK]")


def main() -> int:
    metric_file = sys.modules[DetectionErrorRate.__module__].__file__
    print("Collar / skip-overlap diagnostic for pyannote.metrics DetectionErrorRate")
    print(f"Repository root: {ROOT}")
    print(f"Metric code    : {metric_file}")
    print_exclusion_partition()
    for case in CASES:
        result = check_case(case)
        print_case(case, result)
    print("=" * 88)
    print("ALL DIAGNOSTIC CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
