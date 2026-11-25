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
)
from pyannote.metrics.identification import IdentificationErrorRate


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


@pytest.fixture
def collar_reference():
    """Reference speaker 'A' occupying the whole [0, 2] evaluation region."""
    reference = Annotation()
    reference[Segment(0, 2)] = "A"
    return reference


@pytest.fixture
def collar_hypothesis():
    """Hypothesis where only 'Y' (in [0.8, 1.2]) survives a 1.6s collar."""
    hypothesis = Annotation()
    hypothesis[Segment(0.0, 0.8)] = "X"
    hypothesis[Segment(0.8, 1.2)] = "Y"
    hypothesis[Segment(1.2, 2.0)] = "X"
    return hypothesis


@pytest.mark.parametrize(
    "metric_class,mapping_method",
    [
        (DiarizationErrorRate, "optimal_mapping"),
        (GreedyDiarizationErrorRate, "greedy_mapping"),
    ],
)
def test_mapping_matches_collar_evaluation(
    metric_class, mapping_method, collar_reference, collar_hypothesis
):
    """Public mapping must be computed on the regions actually evaluated.

    With collar=1.6, only the central [0.8, 1.2] region survives: hypothesis
    'X' (which only speaks outside of it) must not be mapped, 'Y' must be
    mapped to 'A', and DER must be 0.
    """
    uem = Timeline([Segment(0, 2)])
    metric = metric_class(collar=1.6)

    details = metric(collar_reference, collar_hypothesis, uem=uem, detailed=True)
    assert details["total"] == pytest.approx(0.4)
    assert details[metric.name] == pytest.approx(0.0)

    mapping = getattr(metric, mapping_method)(
        collar_reference, collar_hypothesis, uem=uem
    )
    assert mapping == {"Y": "A"}
    # same result when uem defaults to the union of both extents
    with pytest.warns(UserWarning, match="uem"):
        mapping = getattr(metric, mapping_method)(collar_reference, collar_hypothesis)
    assert mapping == {"Y": "A"}


@pytest.mark.parametrize(
    "metric_class,mapping_method",
    [
        (DiarizationErrorRate, "optimal_mapping"),
        (GreedyDiarizationErrorRate, "greedy_mapping"),
    ],
)
def test_mapping_matches_skip_overlap_evaluation(
    metric_class, mapping_method, reference_with_overlap, hypothesis
):
    """skip_overlap must be applied before looking for the mapping."""
    uem = Timeline([Segment(0, 40)])

    metric = metric_class(skip_overlap=True)
    details = metric(reference_with_overlap, hypothesis, uem=uem, detailed=True)
    assert details["total"] == pytest.approx(32)

    mapping = getattr(metric, mapping_method)(
        reference_with_overlap, hypothesis, uem=uem
    )
    # apply the public mapping and recompute IER on the same regions
    mapped = hypothesis.rename_labels(mapping=mapping)
    ier = IdentificationErrorRate(skip_overlap=True)(
        reference_with_overlap, mapped, uem=uem
    )
    assert details[metric.name] == pytest.approx(ier)


@pytest.mark.parametrize(
    "metric_class,mapping_method",
    [
        (DiarizationErrorRate, "optimal_mapping"),
        (GreedyDiarizationErrorRate, "greedy_mapping"),
    ],
)
def test_mapping_matches_der(metric_class, mapping_method, reference, hypothesis):
    """DER recomputed from the public mapping must equal the reported DER."""
    for collar in (0.0, 1.0):
        metric = metric_class(collar=collar)
        with pytest.warns(UserWarning, match="uem"):
            der = metric(reference, hypothesis)
            mapping = getattr(metric, mapping_method)(reference, hypothesis)
        mapped = hypothesis.rename_labels(mapping=mapping)
        with pytest.warns(UserWarning, match="uem"):
            ier = IdentificationErrorRate(collar=collar)(reference, mapped)
        assert der == pytest.approx(ier)


@pytest.mark.parametrize(
    "metric_class,mapping_method",
    [
        (DiarizationErrorRate, "optimal_mapping"),
        (GreedyDiarizationErrorRate, "greedy_mapping"),
    ],
)
def test_mapping_empty_uem(metric_class, mapping_method, reference, hypothesis):
    """An explicit empty evaluation map must yield an empty mapping.

    A falsy-but-provided empty ``uem`` must not be mistaken for ``None``
    (which would approximate it with the union of both extents).
    """
    metric = metric_class()
    assert getattr(metric, mapping_method)(reference, hypothesis, uem=Timeline()) == {}
    assert metric(reference, hypothesis, uem=Timeline()) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "metric_class", [DiarizationErrorRate, GreedyDiarizationErrorRate]
)
def test_evaluation_is_cropped_only_once(
    metric_class, collar_reference, collar_hypothesis
):
    """Collars must be removed once, not once per preprocessing step.

    Applying the collar a second time to the already extruded [0.8, 1.2]
    region would leave nothing to evaluate (total would drop to 0).
    """
    uem = Timeline([Segment(0, 2)])
    metric = metric_class(collar=1.6)
    details = metric(collar_reference, collar_hypothesis, uem=uem, detailed=True)
    assert details["total"] == pytest.approx(0.4)
    assert details["correct"] == pytest.approx(0.4)


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
