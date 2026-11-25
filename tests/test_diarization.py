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


def _half_second_reference():
    """A speaks during the first half second, B during the second half."""
    reference = Annotation()
    reference[Segment(0, 0.5)] = "A"
    reference[Segment(0.5, 1)] = "B"
    return reference


def _aligned_hypothesis():
    """X and Y follow the same order as A and B."""
    hypothesis = Annotation()
    hypothesis[Segment(0, 0.5)] = "X"
    hypothesis[Segment(0.5, 1)] = "Y"
    return hypothesis


def _swapped_hypothesis():
    """X and Y are swapped compared with A and B."""
    hypothesis = Annotation()
    hypothesis[Segment(0, 0.5)] = "Y"
    hypothesis[Segment(0.5, 1)] = "X"
    return hypothesis


@pytest.mark.parametrize(
    "metric_class", [DiarizationErrorRate, GreedyDiarizationErrorRate]
)
def test_group_global_mapping_detects_swap(metric_class):
    files = [
        (_half_second_reference(), _aligned_hypothesis()),
        (_half_second_reference(), _swapped_hypothesis()),
    ]

    metric = metric_class()
    with pytest.warns(UserWarning, match="uem"):
        mapping = metric.optimal_mapping_group(files)

    # the single global mapping is decided once for both files
    assert mapping == {"X": "A", "Y": "B"}

    with pytest.warns(UserWarning, match="uem"):
        _, results = metric.evaluate_group(files, detailed=True)

    # each file is still scored independently
    assert results[0]["confusion"] == 0.0
    assert results[0]["diarization error rate"] == 0.0
    assert results[1]["confusion"] == 1.0
    assert results[1]["diarization error rate"] == 1.0

    # the swap is no longer masked: one second of confusion in total
    assert metric["confusion"] == 1.0
    assert metric["total"] == 2.0
    assert abs(metric) == 0.5


@pytest.mark.parametrize(
    "metric_class", [DiarizationErrorRate, GreedyDiarizationErrorRate]
)
def test_per_file_mapping_masks_swap_without_global_mapping(metric_class):
    # without a shared mapping, every file remaps its own labels and both
    # files end up with a zero error rate, hiding the label swap
    metric = metric_class()
    with pytest.warns(UserWarning, match="uem"):
        first = metric(_half_second_reference(), _aligned_hypothesis(), detailed=True)
    with pytest.warns(UserWarning, match="uem"):
        second = metric(_half_second_reference(), _swapped_hypothesis(), detailed=True)
    assert first["diarization error rate"] == 0.0
    assert second["diarization error rate"] == 0.0
    assert metric["confusion"] == 0.0


def test_group_global_mapping_can_be_passed_file_by_file():
    files = [
        (_half_second_reference(), _aligned_hypothesis()),
        (_half_second_reference(), _swapped_hypothesis()),
    ]

    metric = DiarizationErrorRate()
    with pytest.warns(UserWarning, match="uem"):
        mapping = metric.optimal_mapping_group(files)

    for reference, hypothesis in files:
        with pytest.warns(UserWarning, match="uem"):
            metric(reference, hypothesis, mapping=mapping)

    assert metric["confusion"] == 1.0
    assert abs(metric) == 0.5


def test_group_evaluation_uses_independent_uem_and_collar():
    reference = _half_second_reference()
    files = [
        (reference, _aligned_hypothesis(), Timeline([Segment(0, 1)])),
        # only the first half second of the second file is evaluated
        (reference, _swapped_hypothesis(), Timeline([Segment(0, 0.5)])),
    ]

    metric = DiarizationErrorRate()
    _, results = metric.evaluate_group(files, detailed=True)

    # the second file only contributes half a second of reference speech
    assert [result["total"] for result in results] == [1.0, 0.5]
    assert [result["confusion"] for result in results] == [0.0, 0.5]
    assert metric["total"] == 1.5
    assert metric["confusion"] == 0.5

    # collars are applied on a per-file basis: with a 0.2s collar, three
    # reference boundaries (t = 0, 0.5, 1) are excluded from every file
    collar_metric = DiarizationErrorRate(collar=0.2)
    with pytest.warns(UserWarning, match="uem"):
        _, collar_results = collar_metric.evaluate_group(
            [
                (reference, _aligned_hypothesis()),
                (reference, _swapped_hypothesis()),
            ],
            detailed=True,
        )
    assert [result["total"] for result in collar_results] == pytest.approx([0.6, 0.6])


def test_group_mapping_accumulates_cooccurrence_across_files():
    # file 1 alone would favor X -> B because X is split across A and B,
    # but pooling both files recovers the globally optimal assignment.
    reference = _half_second_reference()

    first_hypothesis = Annotation()
    first_hypothesis[Segment(0, 0.5)] = "X"
    first_hypothesis[Segment(0.5, 0.75)] = "Y"
    first_hypothesis[Segment(0.75, 1)] = "Z"

    files = [
        (reference, first_hypothesis),
        (reference, _swapped_hypothesis()),
    ]

    metric = DiarizationErrorRate()
    with pytest.warns(UserWarning, match="uem"):
        mapping = metric.optimal_mapping_group(files)
    assert mapping == {"X": "B", "Y": "A"}


def test_group_mapping_empty_group_and_empty_hypothesis():
    metric = DiarizationErrorRate()
    assert metric.optimal_mapping_group([]) == {}

    reference = _half_second_reference()
    _, results = metric.evaluate_group(
        [(reference, Annotation()), (reference, Annotation())],
        detailed=True,
    )
    assert [result["missed detection"] for result in results] == [1.0, 1.0]
    assert metric["total"] == 2.0
    assert abs(metric) == 1.0


def test_group_mapping_keeps_unmapped_labels_from_matching_reference():
    # 'Z' is an extra hypothesis speaker absent from the global one-to-one
    # assignment. It must be isolated and never counted as correct, even
    # though it overlaps reference speech for a quarter of a second.
    reference = _half_second_reference()

    hypothesis = Annotation()
    hypothesis[Segment(0, 0.5)] = "X"
    hypothesis[Segment(0.5, 0.75)] = "Y"
    hypothesis[Segment(0.75, 1)] = "Z"

    metric = DiarizationErrorRate()
    with pytest.warns(UserWarning, match="uem"):
        mapping, results = metric.evaluate_group(
            [(reference, hypothesis)], detailed=True
        )

    assert mapping == {"X": "A", "Y": "B"}
    assert "Z" not in mapping
    details = results[0]
    assert details["correct"] == 0.75
    # the unmapped 'Z' label overlaps B's speech and is counted as
    # confusion rather than silently matched to any reference label
    assert details["confusion"] == pytest.approx(0.25)
    assert details["false alarm"] == 0.0
    assert details["missed detection"] == 0.0
    assert details["diarization error rate"] == pytest.approx(0.25)


def test_single_file_calls_remain_unchanged(reference, hypothesis):
    # explicit mapping=None must preserve the historical per-file behavior,
    # including integer hypothesis labels and string reference labels
    metric = DiarizationErrorRate()
    value = metric(reference, hypothesis, mapping=None)
    npt.assert_almost_equal(value, 0.5161290322580645, decimal=7)

    details = metric(reference, hypothesis, mapping=None, detailed=True)
    npt.assert_almost_equal(details["confusion"], 7.0, decimal=7)
    npt.assert_almost_equal(details["correct"], 22.0, decimal=7)


def test_jaccard_error_rate_also_supports_global_mapping():
    from pyannote.metrics.diarization import JaccardErrorRate

    files = [
        (_half_second_reference(), _aligned_hypothesis()),
        (_half_second_reference(), _swapped_hypothesis()),
    ]

    metric = JaccardErrorRate()
    with pytest.warns(UserWarning, match="uem"):
        mapping, results = metric.evaluate_group(files, detailed=True)

    assert mapping == {"X": "A", "Y": "B"}
    assert results[0]["jaccard error rate"] == 0.0
    assert results[1]["jaccard error rate"] == 1.0
    assert abs(metric) == 0.5


def test_group_evaluation_labels_report_rows_with_file_uris():
    metric = DiarizationErrorRate()
    files = [
        (_half_second_reference(), _aligned_hypothesis(), Timeline([Segment(0, 1)]), "part-001"),
        (_half_second_reference(), _swapped_hypothesis(), Timeline([Segment(0, 1)]), "part-002"),
    ]
    metric.evaluate_group(files)

    report = metric.report()
    assert list(report.index) == ["part-001", "part-002", "TOTAL"]
    assert report.loc["part-001", ("confusion", "")] == 0.0
    assert report.loc["part-002", ("confusion", "")] == 1.0
