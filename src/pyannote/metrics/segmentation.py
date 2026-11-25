# The MIT License (MIT)

# Copyright (c) 2012-2019 CNRS

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
# Camille Guinaudeau - https://sites.google.com/site/cguinaudeau/
# Mamadou Doumbia
# Diego Fustes diego.fustes at toptal.com

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.core.utils.generators import pairwise

from .base import BaseMetric, f_measure
from .types import Details, MetricComponents
from .utils import UEMSupportMixin

#  TODO: can't we put these as class attributes?
PURITY_NAME = "segmentation purity"
COVERAGE_NAME = "segmentation coverage"
PURITY_COVERAGE_NAME = "segmentation F[purity|coverage]"
PTY_CVG_TOTAL = "total duration"
PTY_CVG_INTER = "intersection duration"

PTY_TOTAL = "pty total duration"
PTY_INTER = "pty intersection duration"
CVG_TOTAL = "cvg total duration"
CVG_INTER = "cvg intersection duration"

PRECISION_NAME = "segmentation precision"
RECALL_NAME = "segmentation recall"

PR_BOUNDARIES = "number of boundaries"
PR_MATCHES = "number of matches"
PR_MATCHED = "matched boundaries"


class SegmentationCoverage(BaseMetric):
    """Segmentation coverage

    Parameters
    ----------
    tolerance : float, optional
        When provided, preprocess reference by filling intra-label gaps shorter
        than `tolerance` (in seconds).

    """

    def __init__(self, tolerance: float = 0.500, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance

    def _partition(self, timeline: Timeline, coverage: Timeline) -> Annotation:

        # boundaries (as set of timestamps)
        boundaries = set()
        for segment in timeline:
            boundaries.add(segment.start)
            boundaries.add(segment.end)

        # partition (as timeline)
        partition = Annotation()
        for start, end in pairwise(sorted(boundaries)):
            segment = Segment(start, end)
            partition[segment] = "_"

        return partition.crop(coverage, mode="intersection").relabel_tracks()

    def _preprocess(
        self, reference: Annotation, hypothesis: Annotation | Timeline
    ) -> tuple[Annotation, Annotation]:

        if not isinstance(reference, Annotation):
            raise TypeError("reference must be an instance of `Annotation`")

        if isinstance(hypothesis, Annotation):
            hypothesis: Timeline = hypothesis.get_timeline()

        # reference where short intra-label gaps are removed
        filled = Timeline()
        for label in reference.labels():
            label_timeline = reference.label_timeline(label)
            for gap in label_timeline.gaps():
                if gap.duration < self.tolerance:
                    label_timeline.add(gap)

            for segment in label_timeline.support():
                filled.add(segment)

        # reference coverage after filling gaps
        coverage = filled.support()

        reference_partition = self._partition(filled, coverage)
        hypothesis_partition = self._partition(hypothesis, coverage)

        return reference_partition, hypothesis_partition

    def _process(self, reference: Annotation, hypothesis: Annotation) -> Details:

        detail = self.init_components()

        # cooccurrence matrix
        K = reference * hypothesis
        detail[PTY_CVG_TOTAL] = np.sum(K).item()
        # np.max has no identity on an empty matrix, which happens when either
        # annotation is empty. Nothing is covered in that case.
        detail[PTY_CVG_INTER] = np.sum(np.max(K, axis=1)).item() if K.size else 0.0

        return detail

    @classmethod
    def metric_name(cls):
        return COVERAGE_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [PTY_CVG_TOTAL, PTY_CVG_INTER]

    def compute_components(
        self, reference: Annotation, hypothesis: Annotation | Timeline, **kwargs
    ):
        reference, hypothesis = self._preprocess(reference, hypothesis)
        return self._process(reference, hypothesis)

    def compute_metric(self, detail: Details) -> float:
        if detail[PTY_CVG_TOTAL] == 0.0:
            return 1.0
        return detail[PTY_CVG_INTER] / detail[PTY_CVG_TOTAL]


class SegmentationPurity(SegmentationCoverage):
    """Segmentation purity

    Parameters
    ----------
    tolerance : float, optional
        When provided, preprocess reference by filling intra-label gaps shorter
        than `tolerance` (in seconds).

    """

    @classmethod
    def metric_name(cls) -> str:
        return PURITY_NAME

    # TODO : Use type from parent class
    def compute_components(
        self, reference: Annotation, hypothesis: Annotation | Timeline, **kwargs
    ) -> Details:
        reference, hypothesis = self._preprocess(reference, hypothesis)
        return self._process(hypothesis, reference)


class SegmentationPurityCoverageFMeasure(SegmentationCoverage):
    """
    Compute segmentation purity and coverage, and return their F-score.


    Parameters
    ----------
    tolerance : float, optional
        When provided, preprocess reference by filling intra-label gaps shorter
        than `tolerance` (in seconds).

    beta : float, optional
            When beta > 1, greater importance is given to coverage.
            When beta < 1, greater importance is given to purity.
            Defaults to 1.

    See also
    --------
    pyannote.metrics.segmentation.SegmentationPurity
    pyannote.metrics.segmentation.SegmentationCoverage
    pyannote.metrics.base.f_measure
    """

    def __init__(self, tolerance=0.500, beta=1, **kwargs):
        super().__init__(tolerance=tolerance, **kwargs)
        self.beta = beta

    def _process(
        self, reference: Annotation, hypothesis: Annotation | Timeline
    ) -> Details:
        reference, hypothesis = self._preprocess(reference, hypothesis)

        detail = self.init_components()

        # cooccurrence matrix coverage
        K = reference * hypothesis
        detail[CVG_TOTAL] = np.sum(K).item()
        detail[CVG_INTER] = np.sum(np.max(K, axis=1)).item() if K.size else 0.0

        # cooccurrence matrix purity
        detail[PTY_TOTAL] = detail[CVG_TOTAL]
        detail[PTY_INTER] = np.sum(np.max(K, axis=0)).item() if K.size else 0.0

        return detail

    def compute_components(
        self, reference: Annotation, hypothesis: Annotation | Timeline, **kwargs
    ) -> Details:
        return self._process(reference, hypothesis)

    def compute_metric(self, detail: Details) -> float:
        _, _, value = self.compute_metrics(detail=detail)
        return value

    def compute_metrics(
        self, detail: Details | None = None
    ) -> tuple[float, float, float]:
        detail = self.accumulated_ if detail is None else detail

        purity = (
            1.0 if detail[PTY_TOTAL] == 0.0 else detail[PTY_INTER] / detail[PTY_TOTAL]
        )

        coverage = (
            1.0 if detail[CVG_TOTAL] == 0.0 else detail[CVG_INTER] / detail[CVG_TOTAL]
        )

        return purity, coverage, f_measure(purity, coverage, beta=self.beta)

    @classmethod
    def metric_name(cls) -> str:
        return PURITY_COVERAGE_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [PTY_TOTAL, PTY_INTER, CVG_TOTAL, CVG_INTER]


class SegmentationPrecision(UEMSupportMixin, BaseMetric):
    """Segmentation precision

    Parameters
    ----------
    tolerance : float, optional
        Maximum distance (in seconds) between two boundaries for them to be
        considered a match. Defaults to 0.
    matching : {"greedy", "optimal"}, optional
        Boundary matching algorithm. "greedy" (default) repeatedly matches the
        closest pair of remaining boundaries. "optimal" computes the best
        alignment between reference and hypothesis boundaries: it maximizes
        the number of matched pairs first, then minimizes the total deviation
        between matched boundaries, each boundary being used at most once.
        Remaining ties are broken in favor of the pairing that happens earlier
        in time. The same pairing is used for both SegmentationPrecision and
        SegmentationRecall.

    The detailed result contains the list of actually matched boundaries
    under the "matched boundaries" key, as (reference, hypothesis) timestamp
    pairs.

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationPrecision
    >>> precision = SegmentationPrecision()

    >>> reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 4)])

    >>> hypothesis = Timeline(
    ...     [Segment(0, 1), Segment(1, 2), Segment(2, 3), Segment(3, 4)]
    ... )
    >>> precision(reference, hypothesis)
    0.6666666666666666

    >>> hypothesis = Timeline([Segment(0, 4)])
    >>> precision(reference, hypothesis)
    1.0

    With "optimal" matching, boundaries are aligned so that as many pairs
    as possible fall within `tolerance`:

    >>> reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    >>> hypothesis = Timeline([Segment(0, 0.4), Segment(0.4, 1.2), Segment(1.2, 3)])

    >>> precision = SegmentationPrecision(tolerance=0.9, matching="optimal")
    >>> precision(reference, hypothesis)
    1.0

    """

    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls):
        return [PR_MATCHES, PR_BOUNDARIES]

    def __init__(self, tolerance=0.0, matching="greedy", **kwargs):

        super().__init__(**kwargs)
        self.tolerance = tolerance
        if matching not in ("greedy", "optimal"):
            raise ValueError(
                f"'matching' must be 'greedy' or 'optimal' (got {matching!r})."
            )
        self.matching = matching

    def _greedy_matching(
        self, ref_boundaries: list[float], hyp_boundaries: list[float]
    ) -> list[tuple[float, float]]:
        """Match closest pairs of boundaries first"""

        N, M = len(ref_boundaries), len(hyp_boundaries)

        # temporal delta between all pairs of boundaries
        delta = np.zeros((N, M))
        for r, refBoundary in enumerate(ref_boundaries):
            for h, hypBoundary in enumerate(hyp_boundaries):
                delta[r, h] = abs(refBoundary - hypBoundary)

        # make sure boundaries too far apart from each other cannot be matched
        # (this is what np.inf is used for)
        delta[np.where(delta > self.tolerance)] = np.inf

        matches = []

        # h always contains the minimum value in delta matrix
        # h == np.inf means that no boundary can be matched
        h = np.amin(delta)

        # while there are still boundaries to match
        while h < np.inf:
            # find boundaries to match
            k = np.argmin(delta)
            i = k // M
            j = k % M

            matches.append((ref_boundaries[i], hyp_boundaries[j]))

            # make sure they cannot be matched again
            delta[i, :] = np.inf
            delta[:, j] = np.inf

            # update minimum value in delta
            h = np.amin(delta)

        return sorted(matches)

    def _optimal_matching(
        self, ref_boundaries: list[float], hyp_boundaries: list[float]
    ) -> list[tuple[float, float]]:
        """Find best alignment between reference and hypothesis boundaries

        Maximize the number of matched pairs first, then minimize the total
        deviation between matched boundaries. Remaining ties are broken in
        favor of the pairing that happens earlier in time. Each boundary is
        used at most once.
        """

        N, M = len(ref_boundaries), len(hyp_boundaries)

        # because boundaries are sorted and deviation is a distance on the
        # line, there always exists an optimal non-crossing matching: dynamic
        # programming over prefixes of both boundary sequences is therefore
        # exact. dp[i][j] is the best (number of matches, total deviation,
        # matched pairs) alignment between ref_boundaries[:i] and
        # hyp_boundaries[:j].
        dp = [[(0, 0.0, ())] * (M + 1) for _ in range(N + 1)]

        def better(a, b):
            # is alignment `a` strictly better than alignment `b`?
            if a[0] != b[0]:
                # more matches is better
                return a[0] > b[0]
            if a[1] != b[1]:
                # smaller total deviation is better
                return a[1] < b[1]
            # prefer the pairing that happens earlier in time
            return a[2] < b[2]

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # leave ref_boundaries[i-1] or hyp_boundaries[j-1] unmatched
                best = dp[i - 1][j]
                if better(dp[i][j - 1], best):
                    best = dp[i][j - 1]

                # match ref_boundaries[i-1] with hyp_boundaries[j-1]
                deviation = abs(ref_boundaries[i - 1] - hyp_boundaries[j - 1])
                if deviation <= self.tolerance:
                    prev = dp[i - 1][j - 1]
                    candidate = (
                        prev[0] + 1,
                        prev[1] + deviation,
                        prev[2]
                        + ((ref_boundaries[i - 1], hyp_boundaries[j - 1]),),
                    )
                    if better(candidate, best):
                        best = candidate

                dp[i][j] = best

        return list(dp[N][M][2])

    def compute_components(
        self,
        reference: Annotation | Timeline,
        hypothesis: Annotation | Timeline,
        **kwargs,
    ) -> Details:

        # extract timeline if needed
        if isinstance(reference, Annotation):
            reference = reference.get_timeline()
        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        detail = self.init_components()

        # number of boundaries in reference and hypothesis
        N = max(0, len(reference) - 1)
        M = max(0, len(hypothesis) - 1)

        # number of boundaries in hypothesis
        detail[PR_BOUNDARIES] = M

        # corner case (no boundary in hypothesis or in reference)
        if M == 0 or N == 0:
            detail[PR_MATCHES] = 0.0
            detail[PR_MATCHED] = []
            return detail

        # reference and hypothesis boundaries
        ref_boundaries = [segment.end for segment in reference][:-1]
        hyp_boundaries = [segment.end for segment in hypothesis][:-1]

        if self.matching == "optimal":
            matches = self._optimal_matching(ref_boundaries, hyp_boundaries)
        else:
            matches = self._greedy_matching(ref_boundaries, hyp_boundaries)

        detail[PR_MATCHES] = float(len(matches))
        detail[PR_MATCHED] = matches
        return detail

    def compute_metric(self, detail: Details) -> float:

        numerator = detail[PR_MATCHES]
        denominator = detail[PR_BOUNDARIES]

        if denominator == 0.0:
            if numerator == 0:
                return 1.0
            else:
                raise ValueError("")
        else:
            return numerator / denominator


class SegmentationRecall(SegmentationPrecision):
    """Segmentation recall

    Parameters
    ----------
    tolerance : float, optional
        Maximum distance (in seconds) between two boundaries for them to be
        considered a match. Defaults to 0.
    matching : {"greedy", "optimal"}, optional
        Boundary matching algorithm. See SegmentationPrecision. Both metrics
        rely on the same pairing when instantiated with the same options.

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationRecall
    >>> recall = SegmentationRecall()

    >>> reference = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 4)])

    >>> hypothesis = Timeline(
    ...     [Segment(0, 1), Segment(1, 2), Segment(2, 3), Segment(3, 4)]
    ... )
    >>> recall(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline([Segment(0, 4)])
    >>> recall(reference, hypothesis)
    0.0

    """

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    def compute_components(
        self,
        reference: Annotation | Timeline,
        hypothesis: Annotation | Timeline,
        **kwargs,
    ) -> Details:
        detail = super().compute_components(hypothesis, reference)
        # report matched boundaries as (reference, hypothesis) pairs
        detail[PR_MATCHED] = [(h, r) for r, h in detail[PR_MATCHED]]
        return detail
