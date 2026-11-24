import pytest
from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.segmentation import (
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


def test_continuous_evaluation_unchanged():
    """Without a UEM, the historical continuous entry point is preserved."""
    reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 4)])
    hypothesis = Timeline(
        [Segment(0, 1), Segment(1, 2), Segment(2, 3), Segment(3, 4)]
    )

    assert SegmentationPrecision()(reference, hypothesis) == 2 / 3
    assert SegmentationRecall()(reference, hypothesis) == 1.0

    single = Timeline([Segment(0, 4)])
    assert SegmentationPrecision()(reference, single) == 1.0
    assert SegmentationRecall()(reference, single) == 0.0


@pytest.fixture
def scoring_domains():
    # two disjoint scoring domains with an unscored gap in between
    return Timeline([Segment(0, 2), Segment(4, 6)])


def test_domain_gap_is_not_a_boundary(scoring_domains):
    """One segment per domain in both reference and hypothesis: there is no
    candidate boundary at all, even though the two segments are adjacent in
    the stitched timeline (the end of the first one, t=2, used to be counted
    as a boundary and matched as a hit)."""

    reference = Timeline([Segment(0, 2), Segment(4, 6)])
    hypothesis = Timeline([Segment(0, 2), Segment(4, 6)])

    for metric_class in (SegmentationPrecision, SegmentationRecall):
        metric = metric_class(tolerance=0.5)
        details = metric(
            reference, hypothesis, uem=scoring_domains, detailed=True
        )
        assert details["number of boundaries"] == 0.0
        assert details["number of matches"] == 0.0
        assert details[metric.name] == 1.0


def test_real_in_domain_boundaries_are_scored(scoring_domains):
    """Boundaries strictly inside a scoring domain are still matched."""

    # one real boundary at t=1 inside the first domain, none in the second
    reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(4, 6)])
    hypothesis = Timeline([Segment(0, 1), Segment(1, 2), Segment(4, 6)])

    for metric_class in (SegmentationPrecision, SegmentationRecall):
        details = metric_class(tolerance=0.5)(
            reference, hypothesis, uem=scoring_domains, detailed=True
        )
        assert details["number of boundaries"] == 1.0
        assert details["number of matches"] == 1.0


def test_tolerance_does_not_bridge_domains(scoring_domains):
    """A match can never bridge two scoring domains, even with a tolerance
    larger than the distance between the candidate boundaries."""

    # reference boundary at t=1.9 (inside first domain)
    reference = Timeline([Segment(0, 1.9), Segment(1.9, 2), Segment(4, 6)])
    # hypothesis boundary at t=4.1 (inside second domain)
    hypothesis = Timeline(
        [Segment(0, 2), Segment(4, 4.1), Segment(4.1, 6)]
    )

    for metric_class in (SegmentationPrecision, SegmentationRecall):
        details = metric_class(tolerance=3.0)(
            reference, hypothesis, uem=scoring_domains, detailed=True
        )
        assert details["number of matches"] == 0.0


def test_in_domain_tolerance_still_applies(scoring_domains):
    """Inside a single domain, the tolerance keeps matching near boundaries."""

    reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(4, 6)])
    hypothesis = Timeline([Segment(0, 1.1), Segment(1.1, 2), Segment(4, 6)])

    strict = SegmentationPrecision(tolerance=0.0)(
        reference, hypothesis, uem=scoring_domains, detailed=True
    )
    assert strict["number of matches"] == 0.0

    loose = SegmentationPrecision(tolerance=0.2)(
        reference, hypothesis, uem=scoring_domains, detailed=True
    )
    assert loose["number of matches"] == 1.0


def test_segment_spanning_the_gap_is_split(scoring_domains):
    """A hypothesis segment covering the unscored gap is cropped per domain;
    the domain edges must not be counted as internal boundaries."""

    reference = Timeline([Segment(0, 2), Segment(4, 6)])
    hypothesis = Timeline([Segment(0, 6)])

    details = SegmentationPrecision()(
        reference, hypothesis, uem=scoring_domains, detailed=True
    )
    assert details["number of boundaries"] == 0.0
    assert details["number of matches"] == 0.0


def test_annotation_inputs_with_uem(scoring_domains):
    reference = Annotation()
    reference[Segment(0, 1)] = "A"
    reference[Segment(1, 2)] = "A"
    reference[Segment(4, 6)] = "B"

    hypothesis = Annotation()
    hypothesis[Segment(0, 1)] = "x"
    hypothesis[Segment(1, 2)] = "y"
    hypothesis[Segment(4, 6)] = "z"

    details = SegmentationPrecision(tolerance=0.5)(
        reference, hypothesis, uem=scoring_domains, detailed=True
    )
    assert details["number of boundaries"] == 1.0
    assert details["number of matches"] == 1.0


def test_empty_uem():
    reference = Timeline([Segment(0, 1), Segment(1, 2)])
    hypothesis = Timeline([Segment(0, 1), Segment(1, 2)])
    assert SegmentationPrecision()(reference, hypothesis, uem=Timeline()) == 1.0
    assert SegmentationRecall()(reference, hypothesis, uem=Timeline()) == 1.0


def test_boundaries_accumulate_across_domains():
    """Matches and boundary counts accumulate over every scoring domain."""

    uem = Timeline([Segment(0, 2), Segment(4, 6), Segment(8, 10)])
    # one in-domain boundary per domain, perfectly aligned on both sides
    reference = Timeline(
        [
            Segment(0, 1),
            Segment(1, 2),
            Segment(4, 5),
            Segment(5, 6),
            Segment(8, 9),
            Segment(9, 10),
        ]
    )

    precision = SegmentationPrecision(tolerance=0.5)
    details = precision(reference, reference, uem=uem, detailed=True)
    assert details["number of boundaries"] == 3.0
    assert details["number of matches"] == 3.0
    assert abs(precision) == 1.0
