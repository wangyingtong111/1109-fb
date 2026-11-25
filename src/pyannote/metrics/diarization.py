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
from pyannote.core.utils.types import Label

from .base import BaseMetric
from .identification import IdentificationErrorRate
from .matcher import GreedyMapper, HungarianMapper, greedy_mapping, hungarian_mapping
from .types import Details, MetricComponents
from .utils import UEMSupportMixin

# TODO: can't we put these as class attributes?
DER_NAME = "diarization error rate"

# one file of a conversation: (reference, hypothesis[, uem[, uri]])
GroupFile = (
    tuple[Annotation, Annotation]
    | tuple[Annotation, Annotation, Timeline]
    | tuple[Annotation, Annotation, Timeline, str]
)


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


class MappedDiarizationErrorRate(IdentificationErrorRate):
    """Base class for diarization error rates based on a label mapping

    Concrete subclasses provide a one-to-one mapping between hypothesis and
    reference labels (e.g. the optimal mapping from the Hungarian algorithm
    or a greedy mapping). The actual diarization error rate is then computed
    as the identification error rate with each hypothesis label translated
    into the corresponding reference label.

    This class also supports evaluating several files that belong to the same
    conversation (e.g. a continuous recording split into multiple files) with
    a single global label mapping: see :meth:`optimal_mapping_group` and
    :meth:`evaluate_group`. Each file is still scored independently, with its
    own evaluation map (``uem``) and collar.

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
    * Diarization error rate between `reference` and `hypothesis` annotations

        >>> metric = DiarizationErrorRate()
        >>> reference = Annotation(...)           # doctest: +SKIP
        >>> hypothesis = Annotation(...)          # doctest: +SKIP
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Compute global diarization error rate and confidence interval
      over multiple documents

        >>> for reference, hypothesis in ...      # doctest: +SKIP
        ...    metric(reference, hypothesis)      # doctest: +SKIP
        >>> global_value = abs(metric)            # doctest: +SKIP
        >>> mean, (lower, upper) = metric.confidence_interval() # doctest: +SKIP

    * Evaluate several files known to belong to the same conversation
      (e.g. a continuous recording split into multiple files) with a single
      global label mapping shared by every file. Each file is still scored
      independently -- with its own evaluation map and collar -- and the
      components keep accumulating in the metric instance. This prevents a
      label swap between files from being hidden by a per-file remapping:

        >>> files = [                             # doctest: +SKIP
        ...     (reference_1, hypothesis_1, uem_1),
        ...     (reference_2, hypothesis_2, uem_2),
        ... ]
        >>> mapping = metric.optimal_mapping_group(files)  # doctest: +SKIP
        >>> for reference, hypothesis, uem in files:       # doctest: +SKIP
        ...     metric(reference, hypothesis, uem=uem, mapping=mapping)
        >>> global_value = abs(metric)            # doctest: +SKIP

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

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    @staticmethod
    def _unpack_file(
        file: tuple[Annotation, Annotation]
        | tuple[Annotation, Annotation, Timeline]
        | tuple[Annotation, Annotation, Timeline, str],
    ) -> tuple[Annotation, Annotation, Timeline | None, str | None]:
        """Split a ``(reference, hypothesis[, uem[, uri]])`` file tuple"""
        if len(file) == 2:
            reference, hypothesis = file
            return reference, hypothesis, None, None
        if len(file) == 3:
            reference, hypothesis, uem = file
            return reference, hypothesis, uem, None
        reference, hypothesis, uem, uri = file
        return reference, hypothesis, uem, uri

    @staticmethod
    def _apply_mapping(
        hypothesis: Annotation, mapping: dict[Label, Label]
    ) -> Annotation:
        """Rename hypothesis labels with `mapping`, isolating unmapped labels.

        Unmapped labels are renamed to unique tuple labels so that they can
        never be mistaken for a reference label when the identification error
        rate is computed. This mirrors the per-file behavior where hypothesis
        labels are systematically renamed to integers and therefore cannot
        match any (string) reference label.
        """
        full_mapping = dict(mapping)
        for index, label in enumerate(hypothesis.labels()):
            if label not in full_mapping:
                full_mapping[label] = ("__unmapped_hypothesis_label__", index)
        return hypothesis.rename_labels(mapping=full_mapping)

    def _solve_mapping(
        self,
        cooccurrence: np.ndarray,
        hypothesis_labels: list[Label],
        reference_labels: list[Label],
    ) -> dict[Label, Label]:
        """Solve the one-to-one label assignment from a cooccurrence matrix"""
        return hungarian_mapping(cooccurrence, hypothesis_labels, reference_labels)

    def _per_file_mapping(
        self, reference: Annotation, hypothesis: Annotation
    ) -> dict[Label, Label]:
        """Compute the optimal mapping for a single file"""
        return self.optimal_mapping(reference, hypothesis)

    def optimal_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
    ) -> dict[Label, Label]:
        """Optimal label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map

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

        # call hungarian mapper
        return self.mapper_(hypothesis, reference)

    def optimal_mapping_group(
        self,
        files: Iterable[GroupFile],
    ) -> dict[Label, Label]:
        """Optimal label mapping shared by every file of one conversation

        When a continuous recording is split into several files, business
        speaker identities are expected to stay consistent across files. This
        method accumulates the reference/hypothesis cooccurrence durations of
        every file and solves the label assignment once, globally, instead of
        allowing each file to pick its own permutation of the labels.

        Each file is preprocessed independently: its own evaluation map
        (``uem``), collar and overlap settings are applied before its
        cooccurrence matrix is accumulated. Labels themselves are left
        untouched, so the same hypothesis label found in several files is
        treated as the same speaker.

        Parameters
        ----------
        files : iterable of tuple
            One ``(reference, hypothesis)``, ``(reference, hypothesis, uem)``
            or ``(reference, hypothesis, uem, uri)`` tuple per file of the
            conversation. All files are expected to use the same reference and
            hypothesis label namespaces.

        Returns
        -------
        mapping : dict
            Mapping between hypothesis (key) and reference (value) labels,
            valid for every file of the group. Pass it to subsequent calls of
            the metric through the ``mapping`` keyword argument.
        """

        # accumulate cooccurrence durations in nested dictionaries keyed by
        # raw hypothesis and reference labels
        cooccurrence: dict[Label, dict[Label, float]] = {}

        for file in files:
            reference, hypothesis, uem, _ = self._unpack_file(file)

            # crop each file independently to its own evaluated regions,
            # removing its own collars / overlap regions
            reference, hypothesis = self.uemify(
                reference,
                hypothesis,
                uem=uem,
                collar=self.collar,
                skip_overlap=self.skip_overlap,
            )

            matrix = hypothesis * reference
            h_labels = hypothesis.labels()
            r_labels = reference.labels()
            for i, h_label in enumerate(h_labels):
                row = cooccurrence.setdefault(h_label, {})
                for j, r_label in enumerate(r_labels):
                    row[r_label] = row.get(r_label, 0.0) + matrix[i, j]

        hypothesis_labels = sorted(cooccurrence, key=str)
        reference_labels = sorted(
            {r_label for row in cooccurrence.values() for r_label in row},
            key=str,
        )
        matrix = np.zeros((len(hypothesis_labels), len(reference_labels)))
        for i, h_label in enumerate(hypothesis_labels):
            for j, r_label in enumerate(reference_labels):
                matrix[i, j] = cooccurrence[h_label].get(r_label, 0.0)

        return self._solve_mapping(matrix, hypothesis_labels, reference_labels)

    def evaluate_group(
        self,
        files: Iterable[GroupFile],
        detailed: bool = False,
    ) -> tuple[dict[Label, Label], list]:
        """Evaluate several files of one conversation with one global mapping

        This first computes a single global label mapping with
        :meth:`optimal_mapping_group`, then scores every file independently
        -- each file keeps its own evaluation map and collar -- and
        accumulates their components into this metric instance, as if the
        files had been evaluated one by one.

        Parameters
        ----------
        files : iterable of tuple
            One ``(reference, hypothesis)``, ``(reference, hypothesis, uem)``
            or ``(reference, hypothesis, uem, uri)`` tuple per file of the
            conversation. The optional ``uri`` is used to label per-file rows
            of the evaluation report.
        detailed : bool, optional
            By default (False), each per-file result is the metric value only.
            Set to True to return the per-file component dictionaries.

        Returns
        -------
        mapping : dict
            The global hypothesis-to-reference mapping used for every file.
        results : list
            Per-file metric values (or component dictionaries when
            ``detailed`` is True), in the same order as ``files``.
        """
        files = list(files)
        mapping = self.optimal_mapping_group(files)
        results = []
        for file in files:
            reference, hypothesis, uem, uri = self._unpack_file(file)
            results.append(
                self(
                    reference,
                    hypothesis,
                    detailed=detailed,
                    uem=uem,
                    uri=uri,
                    mapping=mapping,
                )
            )
        return mapping, results

    def _prepare_and_map(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None,
        mapping: dict[Label, Label] | None,
    ) -> tuple[Annotation, Annotation, Timeline, dict[Label, Label]]:
        """Apply per-file preprocessing and label mapping"""
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

        if mapping is None:
            # make sure reference only contains string labels ('A', 'B', ...)
            reference = reference.rename_labels(generator="string")

            # make sure hypothesis only contains integer labels (1, 2, ...)
            hypothesis = hypothesis.rename_labels(generator="int")

            # per-file optimal (int --> str) mapping
            mapping = self._per_file_mapping(reference, hypothesis)

        # when a group-level mapping is provided, labels are not renamed: the
        # very same mapping must apply consistently to every file of the group.

        # compute identification error rate based on mapped hypothesis
        # NOTE that collar is set to 0.0 because 'uemify' has already
        # been applied (same reason for setting skip_overlap to False)
        mapped = self._apply_mapping(hypothesis, mapping)
        return reference, mapped, uem, mapping

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        mapping: dict[Label, Label] | None = None,
        **kwargs,
    ) -> Details:
        reference, mapped, uem, _ = self._prepare_and_map(
            reference, hypothesis, uem, mapping
        )
        return super().compute_components(
            reference, mapped, uem=uem, collar=0.0, skip_overlap=False, **kwargs
        )


class DiarizationErrorRate(MappedDiarizationErrorRate):
    """Diarization error rate

    First, the optimal mapping between reference and hypothesis labels
    is obtained using the Hungarian algorithm. Then, the actual diarization
    error rate is computed as the identification error rate with each hypothesis
    label translated into the corresponding reference label.

    When several files belong to the same conversation, a single global
    mapping can be shared across files through
    :meth:`MappedDiarizationErrorRate.optimal_mapping_group` and
    :meth:`MappedDiarizationErrorRate.evaluate_group`.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    def _solve_mapping(
        self,
        cooccurrence: np.ndarray,
        hypothesis_labels: list[Label],
        reference_labels: list[Label],
    ) -> dict[Label, Label]:
        return hungarian_mapping(cooccurrence, hypothesis_labels, reference_labels)

    def _per_file_mapping(
        self, reference: Annotation, hypothesis: Annotation
    ) -> dict[Label, Label]:
        """Compute the optimal mapping for a single file"""
        return self.optimal_mapping(reference, hypothesis)

    def optimal_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
    ) -> dict[Label, Label]:
        """Optimal label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map

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

        # call hungarian mapper
        return self.mapper_(hypothesis, reference)


class GreedyDiarizationErrorRate(MappedDiarizationErrorRate):
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

    Examples
    --------
    * Greedy diarization error rate between `reference` and `hypothesis` annotations

        >>> metric = GreedyDiarizationErrorRate()
        >>> reference = Annotation(...)           # doctest: +SKIP
        >>> hypothesis = Annotation(...)          # doctest: +SKIP
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

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = GreedyMapper()

    def _solve_mapping(
        self,
        cooccurrence: np.ndarray,
        hypothesis_labels: list[Label],
        reference_labels: list[Label],
    ) -> dict[Label, Label]:
        # the greedy mapper modifies the cooccurrence matrix in place
        return greedy_mapping(cooccurrence, hypothesis_labels, reference_labels)

    def _per_file_mapping(
        self, reference: Annotation, hypothesis: Annotation
    ) -> dict[Label, Label]:
        """Compute the greedy mapping for a single file"""
        return self.greedy_mapping(reference, hypothesis)

    def greedy_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
    ) -> dict[Label, Label]:
        """Greedy label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map

        Returns
        -------
        mapping : dict
            Mapping between hypothesis (key) and reference (value) labels
        """
        if uem:
            reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)
        return self.mapper_(hypothesis, reference)


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

    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        mapping: dict[Label, Label] | None = None,
        **kwargs,
    ) -> Details:
        # crop reference and hypothesis to evaluated regions (uem), remove
        # collars / overlap regions, and translate hypothesis labels into
        # reference labels. When `mapping` is None, a per-file optimal mapping
        # is used; otherwise the shared group-level mapping is enforced.
        reference, hypothesis, _, _ = self._prepare_and_map(
            reference, hypothesis, uem, mapping
        )

        # after renaming, every paired hypothesis speaker carries the label of
        # its reference speaker; unmapped hypothesis speakers carry unique
        # tuple labels and therefore cannot match any reference speaker.
        hypothesis_labels = set(hypothesis.labels())

        detail = self.init_components()

        for ref_speaker in reference.labels():
            hyp_speaker = ref_speaker if ref_speaker in hypothesis_labels else None

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
