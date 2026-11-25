from pyannote.core import Segment, Timeline

from pyannote.metrics.segmentation import (
    PR_MATCHED,
    SegmentationPrecision,
    SegmentationRecall,
)


def test_segmentation_precision_empty_hypothesis():
    ref = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    # empty hypothesis with >=2 reference segments used to crash np.zeros((N, -1))
    assert SegmentationPrecision()(ref, Timeline()) == 1.0

    # accumulating an empty hypothesis used to push a boundary count of -1,
    # giving a corpus precision > 1
    m = SegmentationPrecision()
    m(ref, ref)
    m(Timeline([Segment(0, 3)]), Timeline())
    assert abs(m) <= 1.0


def make_reference():
    # reference boundaries at 1.0 and 2.0
    return Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])


def make_hypothesis():
    # hypothesis boundaries at 0.4 and 1.2
    return Timeline([Segment(0, 0.4), Segment(0.4, 1.2), Segment(1.2, 3)])


def test_optimal_matching_matches_both_pairs():
    # with tolerance 0.9, greedy matching pairs 1.0 with 1.2 first and can no
    # longer match 2.0 with 0.4, while the optimal alignment pairs boundaries
    # in order and matches both
    reference, hypothesis = make_reference(), make_hypothesis()

    greedy = SegmentationPrecision(tolerance=0.9)
    assert greedy(reference, hypothesis) == 0.5

    precision = SegmentationPrecision(tolerance=0.9, matching="optimal")
    assert precision(reference, hypothesis) == 1.0

    recall = SegmentationRecall(tolerance=0.9, matching="optimal")
    assert recall(reference, hypothesis) == 1.0


def test_optimal_matching_detailed_results():
    # actual matched boundaries can be checked in detailed results, as
    # (reference, hypothesis) pairs for both precision and recall
    reference, hypothesis = make_reference(), make_hypothesis()

    detail = SegmentationPrecision(tolerance=0.9, matching="optimal")(
        reference, hypothesis, detailed=True
    )
    assert detail[PR_MATCHED] == [(1.0, 0.4), (2.0, 1.2)]

    detail = SegmentationRecall(tolerance=0.9, matching="optimal")(
        reference, hypothesis, detailed=True
    )
    assert detail[PR_MATCHED] == [(1.0, 0.4), (2.0, 1.2)]


def test_optimal_matching_prefers_fewer_deviation():
    # both alignments match two pairs: the one with smaller total deviation
    # must be selected
    reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    hypothesis = Timeline([Segment(0, 1.1), Segment(1.1, 1.9), Segment(1.9, 3)])

    detail = SegmentationPrecision(tolerance=0.5, matching="optimal")(
        reference, hypothesis, detailed=True
    )
    assert detail[PR_MATCHED] == [(1.0, 1.1), (2.0, 1.9)]


def test_optimal_matching_tie_prefers_earlier_pairing():
    # both candidate pairings have one match with the same deviation: the
    # earlier one must be selected
    reference = Timeline([Segment(0, 1), Segment(1, 2)])
    hypothesis = Timeline([Segment(0, 0.5), Segment(0.5, 1.5), Segment(1.5, 2)])

    detail = SegmentationPrecision(tolerance=0.6, matching="optimal")(
        reference, hypothesis, detailed=True
    )
    assert detail[PR_MATCHED] == [(1.0, 0.5)]


def test_optimal_matching_uses_each_boundary_once():
    reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    hypothesis = Timeline([Segment(0, 1.5), Segment(1.5, 3)])

    detail = SegmentationPrecision(tolerance=0.6, matching="optimal")(
        reference, hypothesis, detailed=True
    )
    matched = detail[PR_MATCHED]
    assert len(matched) == 1
    assert len({r for r, _ in matched}) == len(matched)
    assert len({h for _, h in matched}) == len(matched)


def test_invalid_matching_raises():
    import pytest

    with pytest.raises(ValueError):
        SegmentationPrecision(matching="unknown")

