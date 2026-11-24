# The MIT License (MIT)

# Copyright (c) 2012- CNRS

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

"""Metrics for diarization"""

from collections.abc import Iterable

import numpy as np
from pyannote.core import Annotation, Timeline
from pyannote.core.utils.generators import int_generator, string_generator
from pyannote.core.utils.types import Label

from .base import BaseMetric
from .identification import IdentificationErrorRate
from .matcher import GreedyMapper, HungarianMapper, check_fixed_mapping
from .types import Details, MetricComponents
from .utils import UEMSupportMixin

# TODO: can't we put these as class attributes?
DER_NAME = "diarization error rate"


def _label_renaming(
    annotation: Annotation,
    generator_factory,
    extra_labels: Iterable[Label] = (),
) -> dict[Label, Label]:
    """Label renaming that `Annotation.rename_labels` would apply.

    Labels are renamed in the same order as
    `annotation.rename_labels(generator=...)` (i.e. sorted by `str`), so that
    normalized labels are identical to the historical behavior whenever
    `extra_labels` is empty.

    `extra_labels` are renamed as well even when they do not occur in
    `annotation`. This lets a fixed identity survive when its pinned label is
    not evaluated in the current region (e.g. cropped away by `uem`): the
    pinned label still receives its own normalized name, which no reference
    track uses, so the corresponding hypothesis speech is scored accordingly
    instead of silently stealing another speaker's identity.

    The mapping is returned (instead of the renamed annotation) so that
    callers can also translate externally-provided labels such as fixed
    identities into the normalized label space.
    """

    labels = sorted(set(annotation.labels()) | set(extra_labels), key=str)
    generator = generator_factory()
    return {label: next(generator) for label in labels}


class DiarizationSpeakerCountAccuracy(UEMSupportMixin, BaseMetric):
    """Fraction of files with the correct number of speakers.

    Each file scores 1 when reference and hypothesis contain the same number
    of distinct speaker labels, and 0 otherwise. Speaker identities and speech
    durations do not affect the score. Two empty annotations score 1.

    Pass an optional evaluation map as ``uem`` when calling the metric to
    count only speakers present in that region. ``abs(metric)`` returns the
    accuracy over all evaluated files, with each file weighted equally.
    Before any files have been evaluated, the accuracy is 0.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "diarization speaker count accuracy"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["correct", "files"]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)
        return {
            "correct": float(len(reference.labels()) == len(hypothesis.labels())),
            "files": 1.0,
        }

    def compute_metric(self, detail: Details) -> float:
        if detail["files"] == 0:
            return 0.0
        return detail["correct"] / detail["files"]


class DiarizationSpeakerCountError(UEMSupportMixin, BaseMetric):
    """Mean absolute speaker count error per file.

    Each file scores ``abs(predicted_num_speakers - expected_num_speakers)``,
    counting distinct labels in hypothesis and reference. Empty annotations
    contain zero speakers. The error is expressed in speakers, without
    normalization by the reference speaker count.

    Pass an optional evaluation map as ``uem`` when calling the metric to
    count only speakers present in that region. ``abs(metric)`` returns the
    mean error over all evaluated files, with each file weighted equally.
    Before any files have been evaluated, the error is 0.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "diarization speaker count error"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["error", "files"]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)
        return {
            "error": float(abs(len(hypothesis.labels()) - len(reference.labels()))),
            "files": 1.0,
        }

    def compute_metric(self, detail: Details) -> float:
        if detail["files"] == 0:
            return 0.0
        return detail["error"] / detail["files"]

    def report(self, display: bool = False):
        """Return a report with the mean error in speakers, not percent."""
        report = super().report(display=False)
        report[(self.name, "%")] /= 100.0
        report.columns = report.columns.map(
            lambda column: (self.name, "") if column == (self.name, "%") else column
        )
        if display:
            print(report.to_string(float_format=lambda value: f"{value:.2f}"))
        return report


class DiarizationErrorRate(IdentificationErrorRate):
    """Diarization error rate

    First, the optimal mapping between reference and hypothesis labels
    is obtained using the Hungarian algorithm. Then, the actual diarization
    error rate is computed as the identification error rate with each hypothesis
    label translated into the corresponding reference label.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    fixed_mapping : dict or iterable of pairs, optional
        Optional one-to-one constraint on the label mapping, given as
        ``{hypothesis_label: reference_label}`` (or as an iterable of such
        pairs). Fixed edges are applied first and the remaining hypothesis
        labels are still assigned by the optimal mapping, restricted to the
        reference labels that are not already taken. As a consequence, the
        error breakdown and the score keep honoring the fixed identities for
        the whole evaluation.

        The constraint is expressed on the hypothesis labels as they appear in
        the hypothesis annotation passed to the metric, before the internal
        normalization to integer labels. When unset, the metric is the usual
        unconstrained optimal diarization error rate.

        Duplicate edges (same hypothesis label twice) and mutually exclusive
        edges (two hypothesis labels fixed to the same reference label) raise
        a `ValueError` before any error measurement is computed. A fixed
        hypothesis label is kept even when it does not overlap the fixed
        reference label in the evaluated region, and even when that reference
        label is not evaluated at all: it still consumes that reference label,
        so the remaining hypothesis labels are matched against a smaller set
        of reference labels and the corresponding duration keeps counting as
        an identity error.

    Examples
    --------
    * Diarization error rate between `reference` and `hypothesis` annotations

        >>> metric = DiarizationErrorRate()
        >>> reference = Annotation(...)           # doctest: +SKIP
        >>> hypothesis = Annotation(...)          # doctest: +SKIP
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Same, with a manually confirmed identity `X` being speaker `A`

        >>> metric = DiarizationErrorRate(fixed_mapping={"X": "A"})
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Compute global diarization error rate and confidence interval
      over multiple documents

        >>> for reference, hypothesis in ...      # doctest: +SKIP
        ...    metric(reference, hypothesis)      # doctest: +SKIP
        >>> global_value = abs(metric)            # doctest: +SKIP
        >>> mean, (lower, upper) = metric.confidence_interval() # doctest: +SKIP

    * Get diarization error rate detailed components

        >>> components = metric(reference, hypothesis, detailed=True) #doctest +SKIP

    * Get accumulated components

        >>> components = metric[:]                # doctest: +SKIP
        >>> metric['confusion']                   # doctest: +SKIP

    See Also
    --------
    :class:`pyannote.metric.base.BaseMetric`: details on accumulation
    :class:`pyannote.metric.identification.IdentificationErrorRate`: identification error rate

    """

    @classmethod
    def metric_name(cls) -> str:
        return DER_NAME

    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        fixed_mapping=None,
        **kwargs,
    ):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.fixed_mapping_ = check_fixed_mapping(fixed_mapping)
        self.mapper_ = HungarianMapper(self.fixed_mapping_)

    def optimal_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        fixed_mapping=None,
    ) -> dict[Label, Label]:
        """Optimal label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map
        fixed_mapping : dict or iterable of pairs, optional
            Override the fixed mapping set at instantiation time for this call.
            Label pairs use the hypothesis-to-reference direction, on the
            labels of the given annotations. Pass an empty mapping to compute
            the unconstrained optimal mapping. Defaults to the mapping given at
            instantiation time.

        Returns
        -------
        mapping : dict
            Mapping between hypothesis (key) and reference (value) labels
        """

        # NOTE that this 'uemification' will not be called when
        # 'optimal_mapping' is called from 'compute_components' as it
        # has already been done in 'compute_components'
        if uem:
            reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)

        # call hungarian mapper ('None' keeps the mapping set at instantiation
        # time, while an empty mapping disables every fixed edge)
        if fixed_mapping is None:
            return self.mapper_(hypothesis, reference)
        return self.mapper_(hypothesis, reference, fixed_mapping=fixed_mapping)

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        fixed_mapping=None,
        **kwargs,
    ) -> Details:
        # normalize the fixed mapping before anything else, so that duplicate
        # or mutually exclusive fixed edges are diagnosed before any error is
        # measured
        if fixed_mapping is None:
            fixed_mapping = self.fixed_mapping_
        else:
            fixed_mapping = check_fixed_mapping(fixed_mapping)

        # crop reference and hypothesis to evaluated regions (uem)
        # remove collars around reference segment boundaries
        # remove overlap regions (if requested)
        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )
        # NOTE that this 'uemification' must be done here because it
        # might have an impact on the search for the optimal mapping.

        # make sure reference only contains string labels ('A', 'B', ...)
        # and hypothesis only contains integer labels (1, 2, ...)
        # NOTE that fixed identities pinned to a reference label which is not
        # evaluated in this region are renamed too: the lock then keeps the
        # pinned name out of every evaluated reference label, so that the
        # hypothesis speech is still scored as an identity error instead of
        # silently taking over an evaluated speaker.
        reference_renaming = _label_renaming(
            reference, string_generator, extra_labels=fixed_mapping.values()
        )
        hypothesis_renaming = _label_renaming(
            hypothesis, int_generator, extra_labels=fixed_mapping.keys()
        )
        reference = reference.rename_labels(mapping=reference_renaming)
        hypothesis = hypothesis.rename_labels(mapping=hypothesis_renaming)

        # fixed identities are declared on the caller's labels: translate them
        # so that they stay in force for both the mapping and the error
        # breakdown.
        fixed_mapping = {
            hypothesis_renaming[hlabel]: reference_renaming[rlabel]
            for hlabel, rlabel in fixed_mapping.items()
        }

        # optimal (int --> str) mapping, constrained by the fixed edges
        mapping = self.optimal_mapping(
            reference, hypothesis, fixed_mapping=fixed_mapping
        )

        # compute identification error rate based on mapped hypothesis
        # NOTE that collar is set to 0.0 because 'uemify' has already
        # been applied (same reason for setting skip_overlap to False)
        mapped = hypothesis.rename_labels(mapping=mapping)
        return super().compute_components(
            reference, mapped, uem=uem, collar=0.0, skip_overlap=False, **kwargs
        )


class GreedyDiarizationErrorRate(IdentificationErrorRate):
    """Greedy diarization error rate

    First, the greedy mapping between reference and hypothesis labels is
    obtained. Then, the actual diarization error rate is computed as the
    identification error rate with each hypothesis label translated into the
    corresponding reference label.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    fixed_mapping : dict or iterable of pairs, optional
        Optional one-to-one constraint on the label mapping. See
        :class:`DiarizationErrorRate` for details. Fixed edges are applied
        before the greedy search, which keeps re-optimizing the remaining
        assignment on the reference labels that are not already taken.

    Examples
    --------
    * Greedy diarization error rate between `reference` and `hypothesis` annotations

        >>> metric = GreedyDiarizationErrorRate()
        >>> reference = Annotation(...)           # doctest: +SKIP
        >>> hypothesis = Annotation(...)          # doctest: +SKIP
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Same, with a manually confirmed identity `X` being speaker `A`

        >>> metric = GreedyDiarizationErrorRate(fixed_mapping={"X": "A"})
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Compute global greedy diarization error rate and confidence interval
      over multiple documents

        >>> for reference, hypothesis in ...      # doctest: +SKIP
        ...    metric(reference, hypothesis)      # doctest: +SKIP
        >>> global_value = abs(metric)            # doctest: +SKIP
        >>> mean, (lower, upper) = metric.confidence_interval() # doctest: +SKIP

    * Get greedy diarization error rate detailed components

        >>> components = metric(reference, hypothesis, detailed=True) #doctest +SKIP

    * Get accumulated components

        >>> components = metric[:]                # doctest: +SKIP
        >>> metric['confusion']                   # doctest: +SKIP

    See Also
    --------
    :class:`pyannote.metric.base.BaseMetric`: details on accumulation

    """

    @classmethod
    def metric_name(cls):
        return DER_NAME

    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        fixed_mapping=None,
        **kwargs,
    ):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.fixed_mapping_ = check_fixed_mapping(fixed_mapping)
        self.mapper_ = GreedyMapper(self.fixed_mapping_)

    def greedy_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        fixed_mapping=None,
    ) -> dict[Label, Label]:
        """Greedy label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map
        fixed_mapping : dict or iterable of pairs, optional
            Override the fixed mapping set at instantiation time for this call.
            Pass an empty mapping to compute the unconstrained greedy mapping.
            Defaults to the mapping given at instantiation time.

        Returns
        -------
        mapping : dict
            Mapping between hypothesis (key) and reference (value) labels
        """
        if uem:
            reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)

        # ('None' keeps the mapping set at instantiation time, while an empty
        # mapping disables every fixed edge)
        if fixed_mapping is None:
            return self.mapper_(hypothesis, reference)
        return self.mapper_(hypothesis, reference, fixed_mapping=fixed_mapping)

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        fixed_mapping=None,
        **kwargs,
    ) -> Details:
        # normalize the fixed mapping before anything else, so that duplicate
        # or mutually exclusive fixed edges are diagnosed before any error is
        # measured
        if fixed_mapping is None:
            fixed_mapping = self.fixed_mapping_
        else:
            fixed_mapping = check_fixed_mapping(fixed_mapping)

        # crop reference and hypothesis to evaluated regions (uem)
        # remove collars around reference segment boundaries
        # remove overlap regions (if requested)
        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )
        # NOTE that this 'uemification' must be done here because it
        # might have an impact on the search for the greedy mapping.

        # make sure reference only contains string labels ('A', 'B', ...)
        # and hypothesis only contains integer labels (1, 2, ...)
        # NOTE that fixed identities pinned to a reference label which is not
        # evaluated in this region are renamed too: the lock then keeps the
        # pinned name out of every evaluated reference label, so that the
        # hypothesis speech is still scored as an identity error instead of
        # silently taking over an evaluated speaker.
        reference_renaming = _label_renaming(
            reference, string_generator, extra_labels=fixed_mapping.values()
        )
        hypothesis_renaming = _label_renaming(
            hypothesis, int_generator, extra_labels=fixed_mapping.keys()
        )
        reference = reference.rename_labels(mapping=reference_renaming)
        hypothesis = hypothesis.rename_labels(mapping=hypothesis_renaming)

        # translate the declared fixed identities into the normalized label
        # space, so that the lock keeps applying to the greedy assignment
        fixed_mapping = {
            hypothesis_renaming[hlabel]: reference_renaming[rlabel]
            for hlabel, rlabel in fixed_mapping.items()
        }

        # greedy (int --> str) mapping, constrained by the fixed edges
        mapping = self.greedy_mapping(
            reference, hypothesis, fixed_mapping=fixed_mapping
        )

        # compute identification error rate based on mapped hypothesis
        # NOTE that collar is set to 0.0 because 'uemify' has already
        # been applied (same reason for setting skip_overlap to False)
        mapped = hypothesis.rename_labels(mapping=mapping)
        return super().compute_components(
            reference, mapped, uem=uem, collar=0.0, skip_overlap=False, **kwargs
        )


JER_NAME = "jaccard error rate"
JER_SPEAKER_ERROR = "speaker error"
JER_SPEAKER_COUNT = "speaker count"


class JaccardErrorRate(DiarizationErrorRate):
    """Jaccard error rate

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    Examples
    --------
    >>> metric = JaccardErrorRate()
    >>> reference = Annotation(...)           # doctest: +SKIP
    >>> hypothesis = Annotation(...)          # doctest: +SKIP
    >>> jer = metric(reference, hypothesis)   # doctest: +SKIP

    Reference
    ---------

    Second DIHARD Challenge Evaluation Plan. Version 1.1
    N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy, M. Liberman
    https://coml.lscp.ens.fr/dihard/2019/second_dihard_eval_plan_v1.1.pdf

    "The Jaccard error rate is based on the Jaccard index, a similarity measure
    used to evaluate the output of image segmentation systems. An optimal
    mapping between reference and system speakers is determined and for each
    pair the Jaccard index is computed. The Jaccard error rate is then defined
    as 1 minus the average of these scores. While similar to DER, it weights
    every speaker’s contribution equally, regardless of how much speech they
    actually produced.

    More concretely, assume we have N reference speakers and M system speakers.
    An optimal mapping between speakers is determined using the Hungarian
    algorithm so that each reference speaker is paired with at most one system
    speaker and each system speaker with at most one reference speaker. Then,
    for each reference speaker ref the speaker-specific Jaccard error rate
    JERref is computed as JERref = (FA + MISS) / TOTAL where

    - TOTAL is the duration of the union of reference and system speaker
      segments; if the reference speaker was not paired with a system
      speaker, it is the duration of all reference speaker segments
    - FA is the total system speaker time not attributed to the reference
      speaker; if the reference speaker was not paired with a system speaker,
      it is 0
    - MISS is the total reference speaker time not attributed to the system
      speaker; if the reference speaker was not paired with a system speaker,
      it is equal to TOTAL

    The Jaccard error rate then is the average of the speaker specific Jaccard
    error rates.

    JER and DER are highly correlated with JER typically being higher,
    especially in recordings where one or more speakers is particularly
    dominant. Where it tends to track DER is in outliers where the diarization
    is especially bad, resulting in one or more unmapped system speakers whose
    speech is not then penalized. In these cases, where DER can easily exceed
    500%, JER will never exceed 100% and may be far lower if the reference
    speakers are handled correctly."

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    fixed_mapping : dict or iterable of pairs, optional
        Optional one-to-one constraint on the label mapping, expressed as
        ``{hypothesis_label: reference_label}`` like in
        :class:`DiarizationErrorRate`. Fixed identities are honored when
        selecting the pairing that minimizes the per-speaker Jaccard error.

    """

    @classmethod
    def metric_name(cls):
        return JER_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            JER_SPEAKER_COUNT,
            JER_SPEAKER_ERROR,
        ]

    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        fixed_mapping=None,
        **kwargs,
    ):
        super().__init__(
            collar=collar,
            skip_overlap=skip_overlap,
            fixed_mapping=fixed_mapping,
            **kwargs,
        )
        # JER optimizes its own objective on the (reference, hypothesis) label
        # order: the mapper set by 'DiarizationErrorRate' would apply the fixed
        # edges in the opposite direction, so it is replaced here. Fixed edges
        # are handed to each mapping call in the right direction instead.
        self.mapper_ = HungarianMapper()

    def optimal_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        fixed_mapping=None,
    ) -> dict[Label, Label]:
        """Optimal label mapping, minimizing the per-speaker Jaccard error.

        JER is computed on the opposite label order than DER, so the fixed
        identities are inverted here. Parameters are those of
        :meth:`DiarizationErrorRate.optimal_mapping`, with ``fixed_mapping``
        still expressed as ``{hypothesis_label: reference_label}``.
        """
        if fixed_mapping is None:
            # the constraint stored at instantiation time is expressed
            # hypothesis-to-reference; this method is called with the opposite
            # label order, so hand it over inverted
            fixed_mapping = {
                reference_label: hypothesis_label
                for hypothesis_label, reference_label in self.fixed_mapping_.items()
            }
        return super().optimal_mapping(
            reference, hypothesis, uem=uem, fixed_mapping=fixed_mapping
        )

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        fixed_mapping=None,
        **kwargs,
    ) -> Details:
        # normalize the fixed mapping before anything else, so that duplicate
        # or mutually exclusive fixed edges are diagnosed before any error is
        # measured
        if fixed_mapping is None:
            fixed_mapping = self.fixed_mapping_
        else:
            fixed_mapping = check_fixed_mapping(fixed_mapping)

        # crop reference and hypothesis to evaluated regions (uem)
        # remove collars around reference segment boundaries
        # remove overlap regions (if requested)
        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )
        # NOTE that this 'uemification' must be done here because it
        # might have an impact on the search for the optimal mapping.

        # make sure reference only contains string labels ('A', 'B', ...)
        # and hypothesis only contains integer labels (1, 2, ...)
        # NOTE that fixed identities are renamed even when their pinned label
        # is not evaluated in this region, so that the lock is preserved.
        reference_renaming = _label_renaming(
            reference, string_generator, extra_labels=fixed_mapping.values()
        )
        hypothesis_renaming = _label_renaming(
            hypothesis, int_generator, extra_labels=fixed_mapping.keys()
        )
        reference = reference.rename_labels(mapping=reference_renaming)
        hypothesis = hypothesis.rename_labels(mapping=hypothesis_renaming)

        # JER calls 'optimal_mapping(hypothesis, reference)', so the mapper
        # receives the reference labels first: invert the fixed edges, which
        # are expressed hypothesis-to-reference, into that direction. This
        # explicit mapping overrides the (already inverted) one that
        # 'JaccardErrorRate.optimal_mapping' would otherwise derive itself.
        fixed_mapping = {
            reference_renaming[rlabel]: hypothesis_renaming[hlabel]
            for hlabel, rlabel in fixed_mapping.items()
        }

        # optimal (str --> int) mapping, constrained by the fixed edges
        mapping = self.optimal_mapping(
            hypothesis, reference, fixed_mapping=fixed_mapping
        )

        detail = self.init_components()

        for ref_speaker in reference.labels():
            hyp_speaker = mapping.get(ref_speaker, None)

            if hyp_speaker is None:
                # if the reference speaker was not paired with a system speaker
                # [total] is the duration of all reference speaker segments

                # if the reference speaker was not paired with a system speaker
                # [fa] is 0

                # if the reference speaker was not paired with a system speaker
                # [miss] is equal to total

                # overall: jer = (fa + miss) / total = (0 + total) / total = 1
                jer = 1.0

            else:
                # total is the duration of the union of reference and system
                # speaker segments
                r = reference.label_timeline(ref_speaker)
                h = hypothesis.label_timeline(hyp_speaker)
                total = r.union(h).support().duration()

                # fa is the total system speaker time not attributed to the
                # reference speaker
                fa = h.duration() - h.crop(r).duration()

                # miss is the total reference speaker time not attributed to
                # the system speaker
                miss = r.duration() - r.crop(h).duration()

                jer = (fa + miss) / total

            detail[JER_SPEAKER_COUNT] += 1
            detail[JER_SPEAKER_ERROR] += jer

        return detail

    def compute_metric(self, detail: Details) -> float:
        if detail[JER_SPEAKER_COUNT] > 0.0:
            return detail[JER_SPEAKER_ERROR] / detail[JER_SPEAKER_COUNT]
        return 1.0


PURITY_NAME = "purity"
PURITY_TOTAL = "total"
PURITY_CORRECT = "correct"


class DiarizationPurity(UEMSupportMixin, BaseMetric):
    """Cluster purity

    A hypothesized annotation has perfect purity if all of its labels overlap
    only segments which are members of a single reference label.
    """

    @classmethod
    def metric_name(cls):
        return PURITY_NAME

    @classmethod
    def metric_components(cls):
        return [PURITY_TOTAL, PURITY_CORRECT]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        detail = self.init_components()

        duration = 0
        largest = 0

        for label in hypothesis.labels():
            support = hypothesis.label_timeline(label)
            duration += support.duration()
            chart = reference.crop(support).chart()
            largest += 0 if not chart else chart[0][1]

        detail[PURITY_TOTAL] = duration
        detail[PURITY_CORRECT] = largest

        return detail

    def compute_metric(self, detail: Details) -> float:
        if detail[PURITY_TOTAL] > 0.0:
            return detail[PURITY_CORRECT] / detail[PURITY_TOTAL]
        return 1.0


COVERAGE_NAME = "coverage"


class DiarizationCoverage(DiarizationPurity):
    """Cluster coverage

    A hypothesized annotation has perfect coverage if all segments from a
    given reference label are assigned to the same cluster.
    """

    @classmethod
    def metric_name(cls):
        return COVERAGE_NAME

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        return super().compute_components(hypothesis, reference, uem=uem, **kwargs)


HOMOGENEITY_NAME = "homogeneity"
HOMOGENEITY_ENTROPY = "entropy"
HOMOGENEITY_CROSS_ENTROPY = "cross-entropy"


class DiarizationHomogeneity(UEMSupportMixin, BaseMetric):
    """Cluster homogeneity

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    """

    @classmethod
    def metric_name(cls):
        return HOMOGENEITY_NAME

    @classmethod
    def metric_components(cls):
        return [HOMOGENEITY_ENTROPY, HOMOGENEITY_CROSS_ENTROPY]

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        detail = self.init_components()

        # crop reference and hypothesis to evaluated regions (uem)
        reference, hypothesis = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
        )

        # cooccurrence matrix
        matrix = reference * hypothesis

        duration = np.sum(matrix)
        rduration = np.sum(matrix, axis=1)
        hduration = np.sum(matrix, axis=0)

        # reference entropy and reference/hypothesis cross-entropy
        ratio = np.ma.divide(rduration, duration).filled(0.0)
        detail[HOMOGENEITY_ENTROPY] = -np.sum(ratio * np.ma.log(ratio).filled(0.0))

        ratio = np.ma.divide(matrix, duration).filled(0.0)
        hratio = np.ma.divide(matrix, hduration).filled(0.0)
        detail[HOMOGENEITY_CROSS_ENTROPY] = -np.sum(
            ratio * np.ma.log(hratio).filled(0.0)
        )

        return detail

    def compute_metric(self, detail):
        numerator = 1.0 * detail[HOMOGENEITY_CROSS_ENTROPY]
        denominator = 1.0 * detail[HOMOGENEITY_ENTROPY]
        if denominator == 0.0:
            if numerator == 0:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0 - numerator / denominator


COMPLETENESS_NAME = "completeness"


class DiarizationCompleteness(DiarizationHomogeneity):
    """Cluster completeness

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    """

    @classmethod
    def metric_name(cls):
        return COMPLETENESS_NAME

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
        return super().compute_components(hypothesis, reference, uem=uem, **kwargs)
