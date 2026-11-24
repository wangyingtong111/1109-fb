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
from collections.abc import Iterable, Mapping
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


def check_fixed_mapping(
    fixed_mapping: (
        "Mapping[Label, Label] | Iterable[tuple[Label, Label]] | None"
    ) = None,
) -> dict["Label", "Label"]:
    """Normalize and validate a fixed one-to-one label mapping.

    A fixed mapping assigns some hypothesis labels to reference labels before
    any automatic matching takes place. It is given either as a
    ``{hypothesis_label: reference_label}`` dictionary or as an iterable of
    ``(hypothesis_label, reference_label)`` pairs.

    Duplicate or mutually exclusive fixed edges are diagnosed here, i.e.
    before any error measurement is computed:

    * the same hypothesis label fixed to two different reference labels is
      rejected as a conflict (fixing it twice to the same reference label is
      rejected as a duplicate);
    * two different hypothesis labels fixed to the same reference label are
      rejected as mutually exclusive, because the mapping must remain
      one-to-one.

    Parameters
    ----------
    fixed_mapping : mapping or iterable of pairs, optional
        Fixed hypothesis-to-reference label assignment.

    Returns
    -------
    fixed_mapping : dict
        Validated ``{hypothesis_label: reference_label}`` dictionary.

    Raises
    ------
    ValueError
        When the fixed mapping is not a collection of label pairs, contains a
        duplicate hypothesis label, or maps two hypothesis labels onto the
        same reference label.
    """

    if fixed_mapping is None:
        return {}

    items = (
        fixed_mapping.items()
        if isinstance(fixed_mapping, Mapping)
        else fixed_mapping
    )

    normalized: dict["Label", "Label"] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "Fixed mapping entries must be "
                "(hypothesis_label, reference_label) pairs, "
                f"got {item!r}."
            )
        hlabel, rlabel = item
        if hlabel in normalized:
            if normalized[hlabel] == rlabel:
                raise ValueError(
                    "Fixed mapping contains duplicate edge "
                    f"{hlabel!r} -> {rlabel!r}. "
                    "Declare each hypothesis label at most once."
                )
            raise ValueError(
                "Fixed mapping contains contradictory edges for hypothesis "
                f"label {hlabel!r}: {normalized[hlabel]!r} and {rlabel!r}. "
                "A hypothesis label can be fixed to a single reference label."
            )
        normalized[hlabel] = rlabel

    owners: dict["Label", "Label"] = {}
    for hlabel, rlabel in normalized.items():
        if rlabel in owners:
            raise ValueError(
                "Fixed mapping is mutually exclusive: reference label "
                f"{rlabel!r} is fixed to both hypothesis labels "
                f"{owners[rlabel]!r} and {hlabel!r}. "
                "The mapping must remain one-to-one."
            )
        owners[rlabel] = hlabel

    return normalized


def _reserved_indexes(
    a_labels: list["Label"],
    b_labels: list["Label"],
    fixed_mapping: dict["Label", "Label"],
) -> tuple[list[int], list[int]]:
    """Indexes of rows/columns taken by the fixed edges.

    A fixed ``A`` label reserves its row as soon as it occurs in ``A``: this
    is what prevents it from being re-assigned to another ``B`` label. Its
    column is reserved only when the pinned ``B`` label occurs in ``B``; when
    it does not, there is no column to withdraw, but the fixed edge is still
    reported by the mapper so that the caller can score the pinned speech
    against the (absent) reference label.

    A fixed edge whose ``A`` label does not occur in ``A`` cannot be part of
    the mapping at all and is ignored entirely: it must not withdraw a ``B``
    column from labels which are actually present.
    """

    a_positions = {label: index for index, label in enumerate(a_labels)}
    b_positions = {label: index for index, label in enumerate(b_labels)}

    rows = set()
    columns = set()
    for a_label, b_label in fixed_mapping.items():
        if a_label not in a_positions:
            continue
        rows.add(a_positions[a_label])
        if b_label in b_positions:
            columns.add(b_positions[b_label])

    return [i for i in range(len(a_labels)) if i not in rows], [
        j for j in range(len(b_labels)) if j not in columns
    ]


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
    """Optimal one-to-one mapping between two sets of labels.

    The mapping is computed from the cooccurrence matrix of annotations ``A``
    and ``B``: label pairs are selected with the Hungarian algorithm so that
    the total cooccurrence of matched pairs is maximal.

    Parameters
    ----------
    fixed_mapping : mapping, optional
        ``{A_label: B_label}`` edges that must be part of the resulting
        mapping, whatever their cooccurrence. Labels are identified by their
        value in ``A`` and ``B`` (not by their position). Fixed edges take
        their rows and columns first and the Hungarian algorithm re-optimizes
        the remaining assignment on the untouched cooccurrence submatrix, so
        that the outcome stays one-to-one and the residual confusion is
        measured on the same evaluated region.

        A fixed edge is kept even when its cooccurrence is zero: lacking a
        correspondence is not a reason to silently release the fixed label.
        When the pinned ``B`` label does not occur in ``B``, the edge is still
        reported (without withdrawing any column). When the fixed ``A`` label
        does not occur in ``A``, the edge cannot be part of the mapping and is
        ignored, so that it does not withhold a ``B`` label from the other
        ``A`` labels.

    Examples
    --------
    >>> mapper = HungarianMapper()
    >>> mapper(A, B)                          # doctest: +SKIP
    {'X': 'B', 'Y': 'A'}

    >>> mapper = HungarianMapper(fixed_mapping={"X": "A"})
    >>> mapper(A, B)                          # doctest: +SKIP
    {'X': 'A'}
    """

    def __init__(self, fixed_mapping=None):
        self.fixed_mapping_ = check_fixed_mapping(fixed_mapping)

    def __call__(
        self,
        A: Annotation,
        B: Annotation,
        fixed_mapping=None,
    ) -> dict["Label", "Label"]:
        """Optimal ``A_label -> B_label`` mapping.

        Parameters
        ----------
        fixed_mapping : mapping, optional
            When not None, overrides the mapping given at instantiation time.
            An empty mapping releases every fixed edge for this call.

        Returns
        -------
        mapping : dict
        """

        if fixed_mapping is not None:
            fixed_mapping = check_fixed_mapping(fixed_mapping)
        else:
            fixed_mapping = self.fixed_mapping_

        a_labels, b_labels = A.labels(), B.labels()

        mapping = {
            a_label: fixed_mapping[a_label]
            for a_label in a_labels
            if a_label in fixed_mapping
        }

        free_a, free_b = _reserved_indexes(a_labels, b_labels, fixed_mapping)
        if not free_a or not free_b:
            return mapping

        cooccurrence = A * B
        submatrix = cooccurrence[np.ix_(free_a, free_b)]

        for a, b in zip(*linear_sum_assignment(-submatrix)):
            if submatrix[a, b] > 0:
                mapping[a_labels[free_a[a]]] = b_labels[free_b[b]]

        return mapping


class GreedyMapper:
    """Greedy one-to-one mapping between two sets of labels.

    Label pairs are matched iteratively by decreasing cooccurrence duration,
    stopping as soon as the largest remaining cooccurrence is zero.

    Parameters
    ----------
    fixed_mapping : mapping, optional
        ``{A_label: B_label}`` edges that must be part of the resulting
        mapping. Their rows and columns are removed before the greedy search
        starts, so that the search keeps re-optimizing the remaining
        assignment. See :class:`HungarianMapper` for details.
    """

    def __init__(self, fixed_mapping=None):
        self.fixed_mapping_ = check_fixed_mapping(fixed_mapping)

    def __call__(
        self,
        A: Annotation,
        B: Annotation,
        fixed_mapping=None,
    ) -> dict["Label", "Label"]:
        """Greedy ``A_label -> B_label`` mapping.

        Parameters
        ----------
        fixed_mapping : mapping, optional
            When not None, overrides the mapping given at instantiation time.
            An empty mapping releases every fixed edge for this call.

        Returns
        -------
        mapping : dict
        """

        if fixed_mapping is not None:
            fixed_mapping = check_fixed_mapping(fixed_mapping)
        else:
            fixed_mapping = self.fixed_mapping_

        a_labels, b_labels = A.labels(), B.labels()

        mapping = {
            a_label: fixed_mapping[a_label]
            for a_label in a_labels
            if a_label in fixed_mapping
        }

        free_a, free_b = _reserved_indexes(a_labels, b_labels, fixed_mapping)
        if not free_a or not free_b:
            return mapping

        cooccurrence = (A * B)[np.ix_(free_a, free_b)]
        Na, Nb = cooccurrence.shape

        for i in range(min(Na, Nb)):
            a, b = np.unravel_index(np.argmax(cooccurrence), (Na, Nb))

            if cooccurrence[a, b] > 0:
                mapping[a_labels[free_a[a]]] = b_labels[free_b[b]]
                cooccurrence[a, :] = 0.0
                cooccurrence[:, b] = 0.0
                continue

            break

        return mapping
