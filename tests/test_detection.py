# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import numpy.testing as npt
import pytest
from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.detection import (
    DetectionAccuracy,
    DetectionCostFunction,
    DetectionErrorRate,
    DetectionPrecision,
    DetectionPrecisionRecallFMeasure,
    DetectionRecall,
    ProbabilityConflictError,
)

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# Reference   |--------------|  |-----------|     |-----|  |--------------|

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# Hypothesis     |-----------------|-----|     |-----------------|  |-----|
#                                  |--------|

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# UEM            |--------------------------------------|


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 5)] = "A"
    reference[Segment(6, 10)] = "B"
    reference[Segment(12, 14)] = "A"
    reference[Segment(15, 20)] = "C"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(1, 7)] = "A"
    hypothesis[Segment(7, 9)] = "D"
    hypothesis[Segment(7, 10)] = "B"
    hypothesis[Segment(11, 17)] = "C"
    hypothesis[Segment(18, 20)] = "D"
    return hypothesis


@pytest.fixture
def uem():
    return Timeline([Segment(1, 14)])


def test_error_rate(reference, hypothesis):
    detectionErrorRate = DetectionErrorRate()
    error_rate = detectionErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.3125, decimal=7)


def test_detailed(reference, hypothesis):
    detectionErrorRate = DetectionErrorRate()
    details = detectionErrorRate(reference, hypothesis, detailed=True)

    rate = details["detection error rate"]
    npt.assert_almost_equal(rate, 0.3125, decimal=7)

    false_alarm = details["false alarm"]
    npt.assert_almost_equal(false_alarm, 3.0, decimal=7)

    missed_detection = details["miss"]
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details["total"]
    npt.assert_almost_equal(total, 16.0, decimal=7)


def test_accuracy(reference, hypothesis):
    # 15 correct / 20 total
    detectionAccuracy = DetectionAccuracy()
    accuracy = detectionAccuracy(reference, hypothesis)
    npt.assert_almost_equal(accuracy, 0.75, decimal=3)


def test_precision(reference, hypothesis):
    # 14 true positive / 17 detected
    detectionPrecision = DetectionPrecision()
    precision = detectionPrecision(reference, hypothesis)
    npt.assert_almost_equal(precision, 0.8235, decimal=3)


def test_recall(reference, hypothesis):
    # 14 true positive / 16 expected
    detectionRecall = DetectionRecall()
    recall = detectionRecall(reference, hypothesis)
    npt.assert_almost_equal(recall, 0.875, decimal=3)


def test_fscore(reference, hypothesis):
    # expected 28/33 since it
    # is computed as :
    # 2*precision*recall / (precision+recall)
    detectionFMeasure = DetectionPrecisionRecallFMeasure()

    fscore = detectionFMeasure(reference, hypothesis)
    npt.assert_almost_equal(fscore, 0.848, decimal=3)


def test_decision_cost_function(reference, hypothesis, uem):
    # No UEM.
    expected = 0.28125
    dcf = DetectionCostFunction(fa_weight=0.25, miss_weight=0.75)
    actual = dcf(reference, hypothesis)
    npt.assert_almost_equal(actual, expected, decimal=7)

    # UEM.
    expected = 1 / 6.0
    dcf = DetectionCostFunction(fa_weight=0.25, miss_weight=0.75)
    actual = dcf(reference, hypothesis, uem=uem)
    npt.assert_almost_equal(actual, expected, decimal=7)


def test_probabilistic_detection_error_components():
    reference = Annotation()
    reference[Segment(0, 2)] = "speech"

    probabilities = [
        (Segment(0, 1), 0.75),
        (Segment(1, 3), 0.25),
    ]

    details = DetectionErrorRate()(
        reference,
        probabilities,
        uem=Timeline([Segment(0, 3)]),
        detailed=True,
    )

    npt.assert_almost_equal(details["miss"], 1.0)
    npt.assert_almost_equal(details["false alarm"], 0.25)
    npt.assert_almost_equal(details["total"], 2.0)
    npt.assert_almost_equal(details["detection error rate"], 1.25 / 2.0)


def test_probabilistic_detection_accepts_mapping_and_annotation_labels():
    reference = Annotation()
    reference[Segment(0, 2)] = "speech"
    uem = Timeline([Segment(0, 3)])

    probability_mapping = {
        Segment(0, 1): 0.75,
        Segment(1, 3): 0.25,
    }
    from_mapping = DetectionErrorRate()(
        reference, probability_mapping, uem=uem, detailed=True
    )
    npt.assert_almost_equal(from_mapping["miss"], 1.0)
    npt.assert_almost_equal(from_mapping["false alarm"], 0.25)

    probability_annotation = Annotation()
    probability_annotation[Segment(0, 1), 0] = 0.75
    probability_annotation[Segment(1, 3), 0] = 0.25

    from_annotation = DetectionErrorRate(probabilistic=True)(
        reference, probability_annotation, uem=uem, detailed=True
    )
    npt.assert_almost_equal(from_annotation["miss"], 1.0)
    npt.assert_almost_equal(from_annotation["false alarm"], 0.25)

    # Numeric probability labels are also detected automatically in an
    # otherwise ordinary Annotation call.
    automatic_annotation = DetectionErrorRate()(
        reference, probability_annotation, uem=uem, detailed=True
    )
    npt.assert_almost_equal(automatic_annotation["miss"], 1.0)
    npt.assert_almost_equal(automatic_annotation["false alarm"], 0.25)

    from_keyword = DetectionErrorRate()(
        reference,
        probabilities=probability_annotation,
        uem=uem,
        detailed=True,
    )
    npt.assert_almost_equal(from_keyword["miss"], 1.0)
    npt.assert_almost_equal(from_keyword["false alarm"], 0.25)


def test_binary_probabilities_match_hard_detection():
    reference = Annotation()
    reference[Segment(0, 2)] = "speech"
    uem = Timeline([Segment(0, 3)])

    binary_probabilities = [
        (0, 1, 1.0),
        (1, 3, 0.0),
    ]
    soft = DetectionErrorRate()(
        reference, binary_probabilities, uem=uem, detailed=True
    )

    hard_hypothesis = Annotation()
    hard_hypothesis[Segment(0, 1)] = "speech"
    hard = DetectionErrorRate()(
        reference, hard_hypothesis, uem=uem, detailed=True
    )

    npt.assert_almost_equal(soft["miss"], hard["miss"])
    npt.assert_almost_equal(soft["false alarm"], hard["false alarm"])
    npt.assert_almost_equal(soft["total"], hard["total"])
    npt.assert_almost_equal(
        soft["detection error rate"], hard["detection error rate"]
    )

    binary_annotation = Annotation()
    binary_annotation[Segment(0, 1), 0] = 1
    binary_annotation[Segment(1, 3), 0] = 0
    automatic = DetectionErrorRate()(
        reference, binary_annotation, uem=uem, detailed=True
    )
    npt.assert_almost_equal(automatic["miss"], hard["miss"])
    npt.assert_almost_equal(automatic["false alarm"], hard["false alarm"])
    npt.assert_almost_equal(automatic["total"], hard["total"])

    # Numeric labels can still be evaluated as historical hard support when
    # the caller explicitly requests that interpretation.
    forced_hard = DetectionErrorRate()(
        reference,
        binary_annotation,
        uem=uem,
        probabilistic=False,
        detailed=True,
    )
    npt.assert_almost_equal(forced_hard["miss"], 0.0)
    npt.assert_almost_equal(forced_hard["false alarm"], 1.0)


def test_probabilistic_uem_gap_and_duplicate_same_value():
    reference = Annotation()
    reference[Segment(0, 2)] = "speech"

    uem = Timeline([Segment(0, 1.5), Segment(2.25, 3)])

    # The nested [1.25, 2.5] row repeats the same 0.25 value as [1, 3].
    # It must not double-count that probability.  The UEM gap [1.5, 2.25]
    # is not integrated.
    probabilities = [
        (0, 1, 0.75),
        (1, 3, 0.25),
        (1.25, 2.5, 0.25),
    ]

    details = DetectionErrorRate()(
        reference, probabilities, uem=uem, detailed=True
    )

    npt.assert_almost_equal(details["miss"], 0.625)
    npt.assert_almost_equal(details["false alarm"], 0.1875)
    npt.assert_almost_equal(details["total"], 1.5)
    npt.assert_almost_equal(
        details["detection error rate"], (0.625 + 0.1875) / 1.5
    )


def test_exact_duplicate_probability_segments_count_once():
    reference = Annotation()
    reference[Segment(0, 1)] = "speech"

    probabilities = {
        Segment(0, 1): 0.25,
    }
    # Add an identical second row through an iterable instead of a mapping.
    probability_rows = [
        (Segment(0, 1), 0.25),
        (Segment(0, 1), 0.25),
    ]

    mapping_details = DetectionErrorRate()(
        reference, probabilities, uem=Timeline([Segment(0, 1)]), detailed=True
    )
    rows_details = DetectionErrorRate()(
        reference, probability_rows, uem=Timeline([Segment(0, 1)]), detailed=True
    )

    npt.assert_almost_equal(mapping_details["miss"], 0.75)
    npt.assert_almost_equal(rows_details["miss"], 0.75)


def test_conflicting_probabilities_in_scored_domain_raise():
    reference = Annotation()
    reference[Segment(0, 2)] = "speech"

    probabilities = [
        (0, 1, 0.5),
        (0.5, 1.5, 0.25),
    ]

    with pytest.raises(
        ProbabilityConflictError, match="Conflicting speech probabilities"
    ):
        DetectionErrorRate()(
            reference,
            probabilities,
            uem=Timeline([Segment(0, 1)]),
            detailed=True,
        )


def test_conflict_at_touching_boundary_or_outside_uem_is_ignored():
    reference = Annotation()
    reference[Segment(0, 2)] = "speech"

    probabilities = [
        (0, 1, 0.5),
        (0.5, 1.5, 0.25),
    ]

    details = DetectionErrorRate()(
        reference,
        probabilities,
        uem=Timeline([Segment(0, 0.5)]),
        detailed=True,
    )
    npt.assert_almost_equal(details["miss"], 0.25)

    touching_boundary = DetectionErrorRate()(
        reference,
        probabilities,
        uem=Timeline([Segment(1, 1.5)]),
        detailed=True,
    )
    npt.assert_almost_equal(touching_boundary["miss"], 0.375)


def test_probabilistic_components_accumulate():
    reference = Annotation()
    reference[Segment(0, 1)] = "speech"
    probabilities = [(0, 1, 0.5)]
    uem = Timeline([Segment(0, 1)])

    metric = DetectionErrorRate()
    first = metric(reference, probabilities, uem=uem, detailed=True)
    second = metric(reference, probabilities, uem=uem, detailed=True)

    npt.assert_almost_equal(first["miss"], 0.5)
    npt.assert_almost_equal(second["miss"], 0.5)
    npt.assert_almost_equal(metric["miss"], 1.0)
    npt.assert_almost_equal(metric["total"], 2.0)
    npt.assert_almost_equal(abs(metric), 0.5)
