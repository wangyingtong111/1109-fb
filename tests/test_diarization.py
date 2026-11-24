import numpy.testing as npt
import pytest
from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.diarization import (
    DiarizationCoverage,
    DiarizationErrorRate,
    DiarizationPurity,
    DiarizationSpeakerCountAccuracy,
    DiarizationSpeakerCountError,
    GreedyDiarizationErrorRate,
    JaccardErrorRate,
)
from pyannote.metrics.matcher import HungarianMapper


def speaker_annotation(count, duration=1.0):
    annotation = Annotation()
    for speaker in range(count):
        annotation[Segment(0, duration), speaker] = f"speaker_{speaker}"
        annotation[Segment(duration, 2 * duration), speaker] = f"speaker_{speaker}"
    return annotation


@pytest.mark.parametrize(
    "expected,predicted", [(3, 3), (3, 1), (1, 4), (0, 0), (0, 3), (3, 0)]
)
def test_speaker_count(expected, predicted):
    reference = speaker_annotation(expected)
    hypothesis = speaker_annotation(predicted).rename_labels(generator="int")
    uem = Timeline([Segment(0, 2)])
    assert DiarizationSpeakerCountAccuracy()(reference, hypothesis, uem=uem) == float(
        expected == predicted
    )
    assert DiarizationSpeakerCountError()(reference, hypothesis, uem=uem) == abs(
        predicted - expected
    )


@pytest.mark.parametrize(
    "metric_class,component,scores",
    [
        (DiarizationSpeakerCountAccuracy, "correct", [1.0, 0.0, 0.0]),
        (DiarizationSpeakerCountError, "error", [0.0, 2.0, 3.0]),
    ],
)
def test_speaker_count_accumulation(metric_class, component, scores):
    metric = metric_class()
    assert abs(metric) == 0.0
    for index, (expected, predicted, duration) in enumerate(
        [(3, 3, 1), (3, 1, 10), (1, 4, 100)]
    ):
        details = metric(
            speaker_annotation(expected, duration),
            speaker_annotation(predicted, duration),
            uem=Timeline([Segment(0, 2 * duration)]),
            uri=f"file_{index}",
            detailed=True,
        )
        assert details == {
            component: scores[index],
            "files": 1.0,
            metric.name: scores[index],
        }
    assert metric["files"] == 3
    assert metric[component] == sum(scores)
    assert abs(metric) == pytest.approx(sum(scores) / 3)
    report = metric.report()
    unit = "" if metric_class is DiarizationSpeakerCountError else "%"
    scale = 1 if unit == "" else 100
    assert report.loc["TOTAL", (metric.name, unit)] == pytest.approx(
        scale * sum(scores) / 3
    )
    assert report.loc["file_1", (metric.name, unit)] == scale * scores[1]
    metric.reset()
    assert metric[:] == {component: 0.0, "files": 0.0}
    assert abs(metric) == 0.0


@pytest.mark.parametrize(
    "metric_class", [DiarizationSpeakerCountAccuracy, DiarizationSpeakerCountError]
)
def test_speaker_count_uem(metric_class):
    reference = speaker_annotation(2)
    hypothesis = speaker_annotation(1)
    reference[Segment(10, 20)] = "outside_reference"
    hypothesis[Segment(10, 20)] = "outside_hypothesis_1"
    hypothesis[Segment(30, 40)] = "outside_hypothesis_2"
    metric = metric_class()
    accuracy = metric_class is DiarizationSpeakerCountAccuracy
    with pytest.warns(UserWarning, match="uem"):
        assert metric(reference, hypothesis) == (1.0 if accuracy else 0.0)
    assert metric(reference, hypothesis, uem=Timeline([Segment(0, 2)])) == (
        0.0 if accuracy else 1.0
    )
    assert metric(reference, hypothesis, uem=Timeline()) == (1.0 if accuracy else 0.0)
    assert len(reference.labels()) == 3
    assert len(hypothesis.labels()) == 3


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def reference_with_overlap():
    reference = Annotation()
    reference[Segment(0, 13)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(2, 13)] = "a"
    hypothesis[Segment(13, 14)] = "d"
    hypothesis[Segment(14, 20)] = "b"
    hypothesis[Segment(22, 38)] = "c"
    hypothesis[Segment(38, 40)] = "d"
    return hypothesis


def test_error_rate(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    error_rate = diarizationErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.5161290322580645, decimal=7)


def test_optimal_mapping(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    mapping = diarizationErrorRate.optimal_mapping(reference, hypothesis)
    assert mapping == {"a": "A", "b": "B", "c": "C"}


def test_detailed(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    details = diarizationErrorRate(reference, hypothesis, detailed=True)

    confusion = details["confusion"]
    npt.assert_almost_equal(confusion, 7.0, decimal=7)

    correct = details["correct"]
    npt.assert_almost_equal(correct, 22.0, decimal=7)

    rate = details["diarization error rate"]
    npt.assert_almost_equal(rate, 0.5161290322580645, decimal=7)

    false_alarm = details["false alarm"]
    npt.assert_almost_equal(false_alarm, 7.0, decimal=7)

    missed_detection = details["missed detection"]
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details["total"]
    npt.assert_almost_equal(total, 31.0, decimal=7)


def test_purity(reference, hypothesis):
    diarizationPurity = DiarizationPurity()
    purity = diarizationPurity(reference, hypothesis)
    npt.assert_almost_equal(purity, 0.6666, decimal=3)


def test_coverage(reference, hypothesis):
    diarizationCoverage = DiarizationCoverage()
    coverage = diarizationCoverage(reference, hypothesis)
    npt.assert_almost_equal(coverage, 0.7096, decimal=3)


def test_skip_overlap(reference_with_overlap, hypothesis):
    metric = DiarizationErrorRate(skip_overlap=True)
    total = metric(reference_with_overlap, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 32, decimal=3)


def test_leep_overlap(reference_with_overlap, hypothesis):
    metric = DiarizationErrorRate(skip_overlap=False)
    total = metric(reference_with_overlap, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 34, decimal=3)


def test_bug_16():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    hypothesis = Annotation()

    metric = DiarizationErrorRate(collar=1)
    total = metric(reference, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 9, decimal=3)

    metric = DiarizationErrorRate(collar=0)
    total = metric(reference, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 10, decimal=3)


def test_jaccard_error_rate_empty_reference():
    from pyannote.metrics.diarization import JaccardErrorRate

    hypothesis = Annotation()
    hypothesis[Segment(0, 10)] = "spk"
    # empty reference -> zero speaker count -> must not ZeroDivisionError
    assert JaccardErrorRate()(Annotation(), hypothesis) == 1.0


def reversed_lock_annotations():
    """Reference A/B, hypothesis Y/X with the opposite intersection order."""
    reference = Annotation()
    reference[Segment(0, 2)] = "A"
    reference[Segment(2, 4)] = "B"
    hypothesis = Annotation()
    hypothesis[Segment(0, 2)] = "Y"
    hypothesis[Segment(2, 4)] = "X"
    return reference, hypothesis


def test_fixed_mapping_unset_matches_free_optimum(reference, hypothesis):
    free = DiarizationErrorRate()
    unset = DiarizationErrorRate(fixed_mapping=None)
    assert unset.optimal_mapping(reference, hypothesis) == free.optimal_mapping(
        reference, hypothesis
    )
    assert unset(reference, hypothesis) == pytest.approx(
        free(reference, hypothesis)
    )
    # an explicit empty mapping is the same as not fixing anything
    assert free.optimal_mapping(
        reference, hypothesis, fixed_mapping={}
    ) == free.optimal_mapping(reference, hypothesis)


def test_fixed_mapping_agreeing_lock():
    reference, hypothesis = reversed_lock_annotations()
    # with this intersection order, the free optimum is X->B and Y->A
    assert DiarizationErrorRate().optimal_mapping(reference, hypothesis) == {
        "Y": "A",
        "X": "B",
    }

    metric = DiarizationErrorRate(fixed_mapping={"X": "B"})
    assert metric.optimal_mapping(reference, hypothesis) == {"Y": "A", "X": "B"}
    details = metric(reference, hypothesis, detailed=True)
    assert details["confusion"] == pytest.approx(0.0)
    assert details["correct"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "metric_class,mapping_method",
    [
        (DiarizationErrorRate, "optimal_mapping"),
        (GreedyDiarizationErrorRate, "greedy_mapping"),
    ],
)
def test_fixed_mapping_disagreeing_lock(metric_class, mapping_method):
    reference, hypothesis = reversed_lock_annotations()

    metric = metric_class(fixed_mapping={"X": "A"})
    mapping = getattr(metric, mapping_method)(reference, hypothesis)
    assert mapping["X"] == "A"
    # the pinned edge consumes A, so Y cannot be assigned to A as well
    assert mapping.get("Y") != "A"
    assert len(set(mapping.values())) == len(mapping)

    details = metric(reference, hypothesis, detailed=True)
    # X speaks over B (0-2s) but is locked to A, and Y over A (2-4s) cannot
    # take A: the two 2-second tracks are both counted as identity errors
    assert details["confusion"] == pytest.approx(4.0)
    assert details["correct"] == pytest.approx(0.0)
    assert details["total"] == pytest.approx(4.0)
    assert details["diarization error rate"] == pytest.approx(1.0)


def test_fixed_mapping_reexamined_within_same_region():
    reference = Annotation()
    reference[Segment(0, 4)] = "A"
    reference[Segment(4, 8)] = "B"
    reference[Segment(8, 12)] = "C"
    hypothesis = Annotation()
    hypothesis[Segment(0, 4)] = "P"
    hypothesis[Segment(4, 8)] = "Q"
    hypothesis[Segment(8, 12)] = "R"

    # the free optimum is the obvious P->A, Q->B, R->C
    assert DiarizationErrorRate().optimal_mapping(reference, hypothesis) == {
        "P": "A",
        "Q": "B",
        "R": "C",
    }

    # locking Q->A forces the remaining assignment to be recomputed
    mapping = DiarizationErrorRate(fixed_mapping={"Q": "A"}).optimal_mapping(
        reference, hypothesis
    )
    assert mapping == {"Q": "A", "R": "C"}

    details = DiarizationErrorRate(fixed_mapping={"Q": "A"})(
        reference, hypothesis, detailed=True
    )
    assert details["confusion"] == pytest.approx(8.0)


def test_fixed_mapping_kept_when_target_not_evaluated():
    reference = Annotation()
    reference[Segment(0, 2)] = "A"
    reference[Segment(2, 4)] = "B"
    hypothesis = Annotation()
    hypothesis[Segment(0, 2)] = "X"
    hypothesis[Segment(2, 4)] = "Y"

    # B is outside of the evaluation map, but X stays locked to it: the lock
    # is not silently released, so X cannot fall back onto the evaluated A.
    metric = DiarizationErrorRate(fixed_mapping={"X": "B"})
    details = metric(
        reference, hypothesis, uem=Timeline([Segment(0, 2)]), detailed=True
    )
    assert details["confusion"] == pytest.approx(2.0)
    assert details["correct"] == pytest.approx(0.0)

    # without the lock, X is matched with the evaluated A
    free = DiarizationErrorRate()(
        reference, hypothesis, uem=Timeline([Segment(0, 2)]), detailed=True
    )
    assert free["correct"] == pytest.approx(2.0)
    assert free["confusion"] == pytest.approx(0.0)


def test_fixed_mapping_ignores_unknown_hypothesis_label():
    reference, hypothesis = reversed_lock_annotations()
    free = DiarizationErrorRate()
    pinned = DiarizationErrorRate(fixed_mapping={"unknown": "A"})
    assert pinned.optimal_mapping(reference, hypothesis) == free.optimal_mapping(
        reference, hypothesis
    )


def test_fixed_mapping_survives_accumulation_and_report():
    reference, hypothesis = reversed_lock_annotations()
    # the constraint survives repeated scoring (accumulation) and reporting
    metric = DiarizationErrorRate(fixed_mapping={"X": "A"})
    for uri in ("file_1", "file_2"):
        value = metric(reference, hypothesis, detailed=True, uri=uri)
        assert value["confusion"] == pytest.approx(4.0)
    assert metric["confusion"] == pytest.approx(8.0)
    assert abs(metric) == pytest.approx(1.0)

    report = metric.report()
    assert report.loc["file_1", ("confusion", "")] == pytest.approx(4.0)
    assert report.loc["TOTAL", (metric.name, "%")] == pytest.approx(100.0)


def test_fixed_mapping_jaccard():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    hypothesis = Annotation()
    hypothesis[Segment(0, 10)] = "a"
    hypothesis[Segment(12, 20)] = "b"

    assert JaccardErrorRate()(reference, hypothesis) == pytest.approx(0.0)
    # locking `a` onto the wrong speaker is honored by the Jaccard pairing
    details = JaccardErrorRate(fixed_mapping={"a": "B"})(
        reference, hypothesis, detailed=True
    )
    assert details["jaccard error rate"] == pytest.approx(1.0)

    # the constraint also applies to the public mapping helper, whose label
    # order is the opposite of DER's
    mapping = JaccardErrorRate(fixed_mapping={"a": "B"}).optimal_mapping(
        hypothesis, reference
    )
    assert mapping["B"] == "a"


def test_fixed_mapping_per_call_override():
    reference, hypothesis = reversed_lock_annotations()
    metric = DiarizationErrorRate(fixed_mapping={"X": "A"})

    assert metric(
        reference, hypothesis, fixed_mapping={}, detailed=True
    )["confusion"] == pytest.approx(0.0)
    assert metric(
        reference, hypothesis, fixed_mapping={"X": "A"}, detailed=True
    )["confusion"] == pytest.approx(4.0)
    # `None` keeps the instantiation-time constraint
    assert metric(reference, hypothesis, detailed=True)[
        "confusion"
    ] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "fixed_mapping,message",
    [
        ([("X", "A"), ("X", "A")], "duplicate"),
        ([("X", "A"), ("X", "B")], "contradictory"),
        ([("X", "A"), ("Y", "A")], "mutually exclusive"),
        ([("X", "A", "B")], "pairs"),
    ],
)
@pytest.mark.parametrize(
    "metric_class", [DiarizationErrorRate, GreedyDiarizationErrorRate, JaccardErrorRate]
)
def test_fixed_mapping_diagnostics(metric_class, fixed_mapping, message):
    reference, hypothesis = reversed_lock_annotations()
    # the constraint is validated before any error is measured
    with pytest.raises(ValueError, match=message):
        metric_class(fixed_mapping=fixed_mapping)
    with pytest.raises(ValueError, match=message):
        metric_class()(reference, hypothesis, fixed_mapping=fixed_mapping)


def test_fixed_mapping_hungarian_mapper(reference, hypothesis):
    free = HungarianMapper()
    assert free(hypothesis, reference) == {"a": "A", "b": "B", "c": "C"}

    locked = HungarianMapper(fixed_mapping={"c": "A"})
    mapping = locked(hypothesis, reference)
    assert mapping["c"] == "A"
    assert mapping.get("a") != "A"
    assert len(set(mapping.values())) == len(mapping)
