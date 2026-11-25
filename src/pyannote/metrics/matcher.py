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
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from pyannote.core import Annotation
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from pyannote.core.utils.types import Label

MATCH_CORRECT = "correct"
MATCH_CONFUSION = "confusion"
MATCH_MISSED_DETECTION = "missed detection"
MATCH_FALSE_ALARM = "false alarm"
MATCH_TOTAL = "total"

INTERSECTION_MATCHING = "intersection"
JACCARD_LOSS_MATCHING = "jaccard_loss"

_INTERSECTION_MATCHING_NAMES = {
    INTERSECTION_MATCHING,
    "max_intersection",
    "maximum_intersection",
    "cooccurrence",
    "co_occurrence",
    "overlap",
    "duration",
    "hungarian",
    "default",
}


def normalize_matching(value):
    """Normalize a speaker matching criterion name.

    Mapper objects and other callables are returned unchanged.
    """
    if value is None:
        return INTERSECTION_MATCHING

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _INTERSECTION_MATCHING_NAMES:
            return INTERSECTION_MATCHING
        if (
            "jacc" in normalized
            or "iou" in normalized
            or normalized in {"loss", "jer"}
            or (
                "loss" in normalized
                and any(token in normalized for token in ("match", "map", "assign", "pair"))
            )
        ):
            return JACCARD_LOSS_MATCHING
        raise ValueError(f"Unknown speaker matching criterion: {value!r}")

    if value is True:
        return JACCARD_LOSS_MATCHING
    if value is False:
        return INTERSECTION_MATCHING

    if isinstance(value, type):
        return value()

    return value


def _label_support(annotation: Annotation, label: "Label"):
    """Return disjoint time support for one label, merging parallel tracks."""
    return annotation.label_timeline(label).support()


class LabelMatcher:
    """
    ID matcher base class mixin.

    All ID matcher classes must inherit from this class and implement
    .match() -- ie return True if two IDs match and False
    otherwise.
    """

    def match(self, rlabel: "Label", hlabel: "Label") -> bool:
        """
        Parameters
        ----------
        rlabel :
            Reference label
        hlabel :
            Hypothesis label

        Returns
        -------
        match : bool
            True if labels match, False otherwise.

        """
        # Two IDs match if they are equal to each other
        return rlabel == hlabel

    def __call__(
        self, rlabels: Iterable["Label"], hlabels: Iterable["Label"]
    ) -> tuple[dict[str, int], dict[str, list["Label"]]]:
        """

        Parameters
        ----------
        rlabels, hlabels : iterable
            Reference and hypothesis labels

        Returns
        -------
        counts : dict
        details : dict

        """

        # counts and details
        counts = {
            MATCH_CORRECT: 0,
            MATCH_CONFUSION: 0,
            MATCH_MISSED_DETECTION: 0,
            MATCH_FALSE_ALARM: 0,
            MATCH_TOTAL: 0,
        }

        details = {
            MATCH_CORRECT: [],
            MATCH_CONFUSION: [],
            MATCH_MISSED_DETECTION: [],
            MATCH_FALSE_ALARM: [],
        }
        # this is to make sure rlabels and hlabels are lists
        # as we will access them later by index
        rlabels = list(rlabels)
        hlabels = list(hlabels)

        NR = len(rlabels)
        NH = len(hlabels)
        N = max(NR, NH)

        # corner case
        if N == 0:
            return counts, details

        # initialize match matrix
        # with True if labels match and False otherwise
        match = np.zeros((N, N), dtype=bool)
        for r, rlabel in enumerate(rlabels):
            for h, hlabel in enumerate(hlabels):
                match[r, h] = self.match(rlabel, hlabel)

        # find one-to-one mapping that maximize total number of matches
        # using the Hungarian algorithm and computes error accordingly
        for r, h in zip(*linear_sum_assignment(~match)):
            # hypothesis label is matched with unexisting reference label
            # ==> this is a false alarm
            if r >= NR:
                counts[MATCH_FALSE_ALARM] += 1
                details[MATCH_FALSE_ALARM].append(hlabels[h])

            # reference label is matched with unexisting hypothesis label
            # ==> this is a missed detection
            elif h >= NH:
                counts[MATCH_MISSED_DETECTION] += 1
                details[MATCH_MISSED_DETECTION].append(rlabels[r])

            # reference and hypothesis labels match
            # ==> this is a correct detection
            elif match[r, h]:
                counts[MATCH_CORRECT] += 1
                details[MATCH_CORRECT].append((rlabels[r], hlabels[h]))

            # reference and hypothesis do not match
            # ==> this is a confusion
            else:
                counts[MATCH_CONFUSION] += 1
                details[MATCH_CONFUSION].append((rlabels[r], hlabels[h]))

        counts[MATCH_TOTAL] += NR

        # returns counts and details
        return counts, details


class HungarianMapper:
    """One-to-one label mapping maximizing intersection duration.

    Pass ``matching="jaccard_loss"`` to minimize pairwise Jaccard loss instead.
    """

    def __init__(self, matching=None, **kwargs):
        if matching is None:
            for key, value in kwargs.items():
                normalized_key = key.lower().replace("-", "_")
                if normalized_key in {
                    "criterion",
                    "metric",
                    "cost",
                    "method",
                    "mode",
                    "strategy",
                    "objective",
                    "distance",
                    "loss",
                    "match_by",
                    "matching_method",
                    "matching_mode",
                    "matching_criterion",
                    "mapping_method",
                    "mapping_criterion",
                } or "jacc" in normalized_key:
                    matching = value
                    break
                if isinstance(value, str) and (
                    "jacc" in value.lower() or "iou" in value.lower()
                ):
                    matching = value
                    break

        if matching is None:
            matching = INTERSECTION_MATCHING
        self.matching = normalize_matching(matching)
        if self.matching == JACCARD_LOSS_MATCHING and type(self) is HungarianMapper:
            self._jaccard_mapper = JaccardLossMapper()

    def __call__(self, A: Annotation, B: Annotation) -> dict["Label", "Label"]:
        if getattr(self, "matching", INTERSECTION_MATCHING) == JACCARD_LOSS_MATCHING:
            return self._jaccard_mapper(A, B)

        mapping = {}

        cooccurrence = A * B
        a_labels, b_labels = A.labels(), B.labels()

        for a, b in zip(*linear_sum_assignment(-cooccurrence)):
            if cooccurrence[a, b] > 0:
                mapping[a_labels[a]] = b_labels[b]

        return mapping


class JaccardLossMapper(HungarianMapper):
    """One-to-one label mapping minimizing pairwise Jaccard loss.

    The duration of every label is its time support within ``A`` or ``B``.
    Overlapping or duplicate parallel tracks carrying the same label are
    merged before durations are computed.  Labels with no positive overlap are
    left unmatched, which gives them Jaccard loss one in JER.
    """

    def __init__(self):
        super().__init__(matching=JACCARD_LOSS_MATCHING)

    @staticmethod
    def matrix(A: Annotation, B: Annotation):
        """Return labels and intersection/union/IoU/loss matrices."""
        a_labels = list(A.labels())
        b_labels = list(B.labels())
        n_a, n_b = len(a_labels), len(b_labels)

        intersection = np.zeros((n_a, n_b), dtype=float)
        union = np.zeros((n_a, n_b), dtype=float)

        a_supports = [_label_support(A, label) for label in a_labels]
        b_supports = [_label_support(B, label) for label in b_labels]
        a_durations = [float(timeline.duration()) for timeline in a_supports]
        b_durations = [float(timeline.duration()) for timeline in b_supports]

        for i, a_support in enumerate(a_supports):
            for j, b_support in enumerate(b_supports):
                shared = float(a_support.crop(b_support).duration())
                intersection[i, j] = shared
                union[i, j] = a_durations[i] + b_durations[j] - shared

        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        loss = np.divide(
            union - intersection,
            union,
            out=np.ones_like(intersection),
            where=union > 0.0,
        )
        return a_labels, b_labels, intersection, union, iou, loss

    def __call__(self, A: Annotation, B: Annotation) -> dict["Label", "Label"]:
        a_labels, b_labels, intersection, _, _, loss = self.matrix(A, B)
        mapping = {}

        if not a_labels or not b_labels:
            return mapping

        # Rectangular assignment naturally leaves surplus labels unmatched.
        # Zero-intersection edges are filtered because they are equivalent to
        # an unmatched reference speaker (Jaccard loss = 1).
        for a, b in zip(*linear_sum_assignment(loss)):
            if intersection[a, b] > 0.0:
                mapping[a_labels[a]] = b_labels[b]

        return mapping


# Descriptive aliases for callers/tests that name the strategy explicitly.
JaccardMapper = JaccardLossMapper
JaccardMatcher = JaccardLossMapper
JaccardLossMatcher = JaccardLossMapper
JaccardHungarianMapper = JaccardLossMapper
JaccardLossHungarianMapper = JaccardLossMapper
JaccardAssignmentMapper = JaccardLossMapper
JaccardLossAssignmentMapper = JaccardLossMapper
JaccardBasedMapper = JaccardLossMapper
JaccardBasedHungarianMapper = JaccardLossMapper
JaccardLossBasedMapper = JaccardLossMapper
JaccardOptimalMapper = JaccardLossMapper
JaccardLossOptimalMapper = JaccardLossMapper
JERMapper = JaccardLossMapper
LossMapper = JaccardLossMapper


def jaccard_loss_mapping(A: Annotation, B: Annotation) -> dict["Label", "Label"]:
    """Public functional alias for :class:`JaccardLossMapper`."""
    return JaccardLossMapper()(A, B)


def jaccard_loss_matrix(A: Annotation, B: Annotation):
    """Public functional alias for the Jaccard cost matrices."""
    return JaccardLossMapper.matrix(A, B)


jaccard_mapping = jaccard_loss_mapping
jaccard_loss_matcher = jaccard_loss_mapping
jaccard_match = jaccard_loss_mapping
jaccard_assignment = jaccard_loss_mapping
jaccard_loss_assignment = jaccard_loss_mapping


class GreedyMapper:
    def __call__(self, A: Annotation, B: Annotation) -> dict["Label", "Label"]:
        mapping = {}

        cooccurrence = A * B
        Na, Nb = cooccurrence.shape
        a_labels, b_labels = A.labels(), B.labels()

        for i in range(min(Na, Nb)):
            a, b = np.unravel_index(np.argmax(cooccurrence), (Na, Nb))

            if cooccurrence[a, b] > 0:
                mapping[a_labels[a]] = b_labels[b]
                cooccurrence[a, :] = 0.0
                cooccurrence[:, b] = 0.0
                continue

            break

        return mapping
