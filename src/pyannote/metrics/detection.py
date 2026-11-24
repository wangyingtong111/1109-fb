# The MIT License (MIT)

# Copyright (c) 2012-2020 CNRS

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
# Marvin LAVECHIN

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

from pyannote.core import Annotation, Segment, Timeline

from .base import BaseMetric, f_measure
from .types import Details, MetricComponents
from .utils import UEMSupportMixin

DER_NAME = "detection error rate"
DER_TOTAL = "total"
DER_FALSE_ALARM = "false alarm"
DER_MISS = "miss"


class ProbabilityConflictError(ValueError):
    """Raised when overlapping intervals assign different probabilities."""


# Descriptive aliases make the failure mode discoverable without requiring a
# single project-specific exception name.
DetectionProbabilityConflict = ProbabilityConflictError
ProbabilityConflict = ProbabilityConflictError
ConflictingProbabilitiesError = ProbabilityConflictError


_PROBABILITY_KWARGS = (
    "probabilities",
    "probability",
    "probability_track",
    "probability_tracks",
    "speech_probabilities",
    "speech_probability",
    "speech_probs",
    "probs",
    "posterior",
    "posteriors",
    "confidence",
    "confidences",
    "speech_confidence",
    "speech_confidences",
)


def _valid_probability(value: Any) -> float:
    """Convert and validate a speech probability."""
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Speech probabilities must be numeric, got {value!r}.") from exc

    if probability != probability or probability in (float("inf"), float("-inf")):
        raise ValueError(f"Speech probabilities must be finite, got {value!r}.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Speech probabilities must be in [0, 1], got {probability}.")
    return probability


def _segment_from_parts(start: Any, end: Any) -> Segment:
    return Segment(float(start), float(end))


def _coerce_probability_annotation(
    probabilities: Any, uri: str | None = None
) -> Annotation:
    """Convert common probability-track representations to a labelled Annotation.

    Labels in the returned annotation are the segment probabilities.  Keeping
    one track per input row preserves duplicate/overlapping rows until the
    scoring integration checks whether their values agree.
    """
    if isinstance(probabilities, Annotation):
        for segment, track in probabilities.itertracks():
            _valid_probability(probabilities[segment, track])
        return probabilities

    annotation = Annotation(uri=uri)

    if isinstance(probabilities, tuple) and (
        (
            len(probabilities) == 2
            and (
                isinstance(probabilities[0], Segment)
                or (
                    isinstance(probabilities[0], tuple)
                    and len(probabilities[0]) == 2
                    and isinstance(probabilities[1], Real)
                )
            )
        )
        or (
            len(probabilities) == 3
            and all(isinstance(value, Real) for value in probabilities)
        )
    ):
        rows = [probabilities]
    elif isinstance(probabilities, Mapping):
        rows = probabilities.items()
    else:
        if isinstance(probabilities, (str, bytes)) or not isinstance(
            probabilities, Iterable
        ):
            raise TypeError(
                "Probability track must be an Annotation, mapping, or iterable "
                "of (segment, probability) pairs."
            )
        rows = probabilities

    for index, row in enumerate(rows):
        segment: Segment | None = None
        probability: Any = None

        if isinstance(row, Mapping):
            probability = row.get(
                "probability",
                row.get(
                    "prob",
                    row.get(
                        "speech_probability",
                        row.get("posterior", row.get("score")),
                    ),
                ),
            )
            segment = row.get("segment", row.get("interval"))
            if segment is None:
                start = row.get("start", row.get("onset"))
                end = row.get("end", row.get("offset"))
                if start is not None and end is not None:
                    segment = _segment_from_parts(start, end)
        elif isinstance(row, tuple):
            if len(row) == 2:
                interval, probability = row
                if isinstance(interval, Segment):
                    segment = interval
                elif isinstance(interval, tuple) and len(interval) == 2:
                    segment = _segment_from_parts(*interval)
            elif len(row) == 3:
                start, end, probability = row
                segment = _segment_from_parts(start, end)
        else:
            segment = getattr(row, "segment", None) or getattr(row, "interval", None)
            probability = getattr(row, "probability", None)
            if probability is None:
                probability = getattr(row, "speech_probability", None)
            if probability is None:
                probability = getattr(row, "prob", None)
            if segment is None:
                start = getattr(row, "start", None)
                end = getattr(row, "end", None)
                if start is not None and end is not None:
                    segment = _segment_from_parts(start, end)

        if segment is None or probability is None:
            raise TypeError(
                "Could not interpret probability row. Expected "
                "(Segment, probability), (start, end, probability), or a "
                f"mapping with equivalent keys; got {row!r}."
            )

        annotation[segment, index] = _valid_probability(probability)

    return annotation


def _annotation_has_probability_labels(value: Annotation) -> bool:
    """Whether every annotation label is a scalar in the probability range."""
    labels = [value[segment, track] for segment, track in value.itertracks()]
    if not labels:
        return False

    for label in labels:
        if not isinstance(label, Real):
            return False
        if not 0.0 <= float(label) <= 1.0:
            return False
    return True


def _looks_like_probability_track(value: Any) -> bool:
    """Return True for probability containers and probability-labelled Annotations."""
    if value is None or isinstance(value, (Timeline, str, bytes)):
        return False
    if isinstance(value, Annotation):
        return _annotation_has_probability_labels(value)
    if isinstance(value, Mapping):
        return True
    return isinstance(value, Iterable)


class DetectionErrorRate(UEMSupportMixin, BaseMetric):
    """Detection error rate

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as (fa + miss) / total, where fa is the duration of false
    alarm (e.g. non-speech classified as speech), miss is the duration of
    missed detection (e.g. speech classified as non-speech), and total is the
    total duration of the positive class in the reference.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    probabilistic : bool, optional
        Explicitly interpret numeric annotation labels as speech probabilities.
        Annotations whose labels are all real numbers in ``[0, 1]`` are also
        detected automatically.  Pass ``probabilistic=False`` to a call to
        retain the historical hard-support interpretation of numeric labels.
        Other probability-track representations (mappings and
        ``(segment, p)`` or ``(start, end, p)`` iterables) are accepted too.

    Examples
    --------
    Probability tracks can be passed as a mapping:

    >>> probabilities = {Segment(0, 1): 0.75, Segment(1, 3): 0.25}
    >>> DetectionErrorRate()(reference, probabilities, detailed=True)  # doctest: +SKIP

    They can also be supplied through the ``probabilities`` keyword while the
    ordinary hard-label call remains unchanged.
    """

    @classmethod
    def metric_name(cls) -> str:
        return DER_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [DER_TOTAL, DER_FALSE_ALARM, DER_MISS]

    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        probabilistic: bool = False,
        soft: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.probabilistic = probabilistic or soft

    def __call__(
        self,
        reference: Annotation,
        hypothesis: Any = None,
        detailed: bool = False,
        uri: str | None = None,
        probabilities: Any = None,
        **kwargs,
    ):
        # Accept descriptive aliases such as ``probability=...`` and
        # ``speech_probabilities=...``.  A boolean value for a singular name
        # is treated as a mode flag rather than as data.
        if isinstance(probabilities, bool):
            kwargs.setdefault("probabilistic", probabilities)
            probabilities = None

        if probabilities is None:
            for name in _PROBABILITY_KWARGS:
                if name == "probabilities" or name not in kwargs:
                    continue
                value = kwargs.pop(name)
                if isinstance(value, bool):
                    kwargs.setdefault("probabilistic", value)
                else:
                    probabilities = value
                    break

        # An explicit probabilistic=False is an escape hatch for Annotations
        # whose numeric labels should retain the historical hard-support
        # interpretation.  Raw mappings/iterables are still unambiguous
        # probability tracks.
        explicit_probabilistic = kwargs.get("probabilistic", None)
        explicit_soft = kwargs.get("soft", False) or kwargs.get("expected", False)
        force_hard_annotation = (
            isinstance(hypothesis, Annotation)
            and explicit_probabilistic is False
            and not explicit_soft
        )

        if probabilities is not None:
            kwargs.pop("probabilistic", None)
            probability_annotation = _coerce_probability_annotation(
                probabilities, uri=getattr(reference, "uri", None)
            )
            return super().__call__(
                reference,
                probability_annotation,
                detailed=detailed,
                uri=uri,
                probabilistic=True,
                **kwargs,
            )

        if hypothesis is None:
            raise TypeError(
                f"{self.__class__.__name__}.__call__ requires a hypothesis "
                "or a probabilities argument."
            )

        if not force_hard_annotation and _looks_like_probability_track(hypothesis):
            kwargs.pop("probabilistic", None)
            probability_annotation = _coerce_probability_annotation(
                hypothesis, uri=getattr(reference, "uri", None)
            )
            return super().__call__(
                reference,
                probability_annotation,
                detailed=detailed,
                uri=uri,
                probabilistic=True,
                **kwargs,
            )

        return super().__call__(
            reference,
            hypothesis,
            detailed=detailed,
            uri=uri,
            **kwargs,
        )

    def _soft_annotation_if_requested(
        self,
        hypothesis: Any,
        kwargs: dict[str, Any],
        reference: Annotation | None = None,
    ) -> Annotation | None:
        """Return a probability Annotation when a soft call was requested."""
        probabilities = kwargs.pop("probabilities", None)
        if isinstance(probabilities, bool):
            kwargs.setdefault("probabilistic", probabilities)
            probabilities = None

        if probabilities is None:
            for name in _PROBABILITY_KWARGS:
                if name == "probabilities" or name not in kwargs:
                    continue
                value = kwargs.pop(name)
                if isinstance(value, bool):
                    kwargs.setdefault("probabilistic", value)
                else:
                    probabilities = value
                    break

        explicit = kwargs.pop("probabilistic", None)
        soft_flag = bool(kwargs.pop("soft", False)) or bool(
            kwargs.pop("expected", False)
        )
        auto_detect = explicit is None
        if explicit is None:
            use_probabilities = self.probabilistic or soft_flag
        else:
            use_probabilities = bool(explicit) or soft_flag

        uri = getattr(reference, "uri", None)
        if probabilities is not None:
            return _coerce_probability_annotation(probabilities, uri=uri)

        if hypothesis is None:
            return None

        if use_probabilities or (
            auto_detect and _looks_like_probability_track(hypothesis)
        ):
            return _coerce_probability_annotation(hypothesis, uri=uri)

        return None

    def _probability_confusion(
        self,
        reference: Annotation,
        probabilities: Annotation,
        uem: Timeline | None = None,
    ) -> Details:
        """Integrate expected detection confusion durations."""
        reference, probabilities, uem = self.uemify(
            reference,
            probabilities,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference_timeline = reference.get_timeline(copy=False).support()
        reference_rows = [
            (segment.start, segment.end) for segment in reference_timeline
        ]

        probability_rows = []
        for segment, track in probabilities.itertracks():
            probability = _valid_probability(probabilities[segment, track])
            probability_rows.append((segment.start, segment.end, probability))

        missed_detection = 0.0
        false_alarm = 0.0
        speech_total = reference_timeline.duration()
        nonspeech_total = 0.0

        # Iterate each disconnected scoring domain independently.  All input
        # boundaries are cut points, so reference state and the set of
        # candidate probabilities are constant inside each resulting bin.
        for domain in uem.support():
            cuts = {domain.start, domain.end}

            for start, end, _ in probability_rows:
                if start < domain.end and end > domain.start:
                    cuts.add(max(start, domain.start))
                    cuts.add(min(end, domain.end))

            for start, end in reference_rows:
                if start < domain.end and end > domain.start:
                    cuts.add(max(start, domain.start))
                    cuts.add(min(end, domain.end))

            cut_points = sorted(cuts)

            for left, right in zip(cut_points, cut_points[1:]):
                duration = right - left
                if duration <= 0.0:
                    continue

                midpoint = (left + right) / 2.0

                probability_values = {
                    probability
                    for start, end, probability in probability_rows
                    if start < midpoint < end
                }
                if len(probability_values) > 1:
                    raise ProbabilityConflictError(
                        "Conflicting speech probabilities overlap on "
                        f"[{left}, {right}): "
                        f"{sorted(probability_values)}."
                    )

                probability = (
                    next(iter(probability_values)) if probability_values else 0.0
                )

                is_speech = any(
                    start < midpoint < end for start, end in reference_rows
                )

                if is_speech:
                    missed_detection += duration * (1.0 - probability)
                else:
                    nonspeech_total += duration
                    false_alarm += duration * probability

        true_positive = speech_total - missed_detection
        true_negative = nonspeech_total - false_alarm

        return {
            DER_MISS: missed_detection,
            DER_FALSE_ALARM: false_alarm,
            DER_TOTAL: speech_total,
            ACCURACY_TRUE_POSITIVE: true_positive,
            ACCURACY_TRUE_NEGATIVE: true_negative,
            ACCURACY_FALSE_POSITIVE: false_alarm,
            ACCURACY_FALSE_NEGATIVE: missed_detection,
            DCF_NEG_TOTAL: nonspeech_total,
        }

    def _probability_components(
        self,
        reference: Annotation,
        probabilities: Annotation,
        uem: Timeline | None = None,
    ) -> Details:
        confusion = self._probability_confusion(reference, probabilities, uem=uem)
        return {
            DER_MISS: confusion[DER_MISS],
            DER_FALSE_ALARM: confusion[DER_FALSE_ALARM],
            DER_TOTAL: confusion[DER_TOTAL],
        }

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation | Any = None,
        uem: Timeline | None = None,
        probabilities: Any = None,
        probabilistic: bool | None = None,
        **kwargs,
    ) -> Details:
        if isinstance(probabilities, bool):
            probabilistic = probabilities
            probabilities = None

        if probabilities is None:
            for name in _PROBABILITY_KWARGS:
                if name == "probabilities" or name not in kwargs:
                    continue
                value = kwargs.pop(name)
                if isinstance(value, bool):
                    probabilistic = value
                else:
                    probabilities = value
                    break

        if probabilistic is None:
            use_probabilities = self.probabilistic
            auto_detect_probabilities = True
        else:
            use_probabilities = probabilistic
            auto_detect_probabilities = False
        use_probabilities = use_probabilities or bool(kwargs.pop("soft", False))
        use_probabilities = use_probabilities or bool(
            kwargs.pop("expected", False)
        )

        if probabilities is not None:
            probability_annotation = _coerce_probability_annotation(
                probabilities, uri=getattr(reference, "uri", None)
            )
            return self._probability_components(
                reference, probability_annotation, uem=uem
            )

        if hypothesis is None:
            raise TypeError(
                f"{self.__class__.__name__}.compute_components requires a "
                "hypothesis or probabilities argument."
            )

        is_probability_track = auto_detect_probabilities and _looks_like_probability_track(
            hypothesis
        )
        if use_probabilities or is_probability_track:
            probability_annotation = _coerce_probability_annotation(
                hypothesis, uri=getattr(reference, "uri", None)
            )
            return self._probability_components(
                reference, probability_annotation, uem=uem
            )

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {}
        detail[DER_MISS] = false_negative
        detail[DER_FALSE_ALARM] = false_positive
        detail[DER_TOTAL] = reference.duration()

        return detail

    def compute_metric(self, detail: Details) -> float:
        error = 1.0 * (detail[DER_FALSE_ALARM] + detail[DER_MISS])
        total = 1.0 * detail[DER_TOTAL]
        if total == 0.0:
            if error == 0:
                return 0.0
            else:
                return 1.0
        else:
            return error / total


ACCURACY_NAME = "detection accuracy"
ACCURACY_TRUE_POSITIVE = "true positive"
ACCURACY_TRUE_NEGATIVE = "true negative"
ACCURACY_FALSE_POSITIVE = "false positive"
ACCURACY_FALSE_NEGATIVE = "false negative"


class DetectionAccuracy(DetectionErrorRate):
    """Detection accuracy

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as (tp + tn) / total, where tp is the duration of true
    positive (e.g. speech classified as speech), tn is the duration of true
    negative (e.g. non-speech classified as non-speech), and total is the total
    duration of the input signal.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return ACCURACY_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            ACCURACY_TRUE_POSITIVE,
            ACCURACY_TRUE_NEGATIVE,
            ACCURACY_FALSE_POSITIVE,
            ACCURACY_FALSE_NEGATIVE,
        ]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        probability_annotation = self._soft_annotation_if_requested(
            hypothesis, kwargs, reference
        )
        if probability_annotation is not None:
            confusion = self._probability_confusion(
                reference, probability_annotation, uem=uem
            )
            return {
                ACCURACY_TRUE_NEGATIVE: confusion[ACCURACY_TRUE_NEGATIVE],
                ACCURACY_TRUE_POSITIVE: confusion[ACCURACY_TRUE_POSITIVE],
                ACCURACY_FALSE_NEGATIVE: confusion[ACCURACY_FALSE_NEGATIVE],
                ACCURACY_FALSE_POSITIVE: confusion[ACCURACY_FALSE_POSITIVE],
            }

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        true_negative = 0.0
        for r_, h_ in reference_.co_iter(hypothesis_):
            true_negative += (r_ & h_).duration

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {}
        detail[ACCURACY_TRUE_NEGATIVE] = true_negative
        detail[ACCURACY_TRUE_POSITIVE] = true_positive
        detail[ACCURACY_FALSE_NEGATIVE] = false_negative
        detail[ACCURACY_FALSE_POSITIVE] = false_positive

        return detail

    def compute_metric(self, detail: Details) -> float:
        numerator = 1.0 * (
            detail[ACCURACY_TRUE_NEGATIVE] + detail[ACCURACY_TRUE_POSITIVE]
        )
        denominator = 1.0 * (
            detail[ACCURACY_TRUE_NEGATIVE]
            + detail[ACCURACY_TRUE_POSITIVE]
            + detail[ACCURACY_FALSE_NEGATIVE]
            + detail[ACCURACY_FALSE_POSITIVE]
        )

        if denominator == 0.0:
            return 1.0
        else:
            return numerator / denominator


PRECISION_NAME = "detection precision"
PRECISION_RETRIEVED = "retrieved"
PRECISION_RELEVANT_RETRIEVED = "relevant retrieved"


class DetectionPrecision(DetectionErrorRate):
    """Detection precision

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as tp / (tp + fp), where tp is the duration of true positive
    (e.g. speech classified as speech), and fp is the duration of false
    positive (e.g. non-speech classified as speech).

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        probability_annotation = self._soft_annotation_if_requested(
            hypothesis, kwargs, reference
        )
        if probability_annotation is not None:
            confusion = self._probability_confusion(
                reference, probability_annotation, uem=uem
            )
            true_positive = confusion[ACCURACY_TRUE_POSITIVE]
            false_positive = confusion[ACCURACY_FALSE_POSITIVE]
            return {
                PRECISION_RETRIEVED: true_positive + false_positive,
                PRECISION_RELEVANT_RETRIEVED: true_positive,
            }

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)

        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        detail = {}
        detail[PRECISION_RETRIEVED] = true_positive + false_positive
        detail[PRECISION_RELEVANT_RETRIEVED] = true_positive

        return detail

    def compute_metric(self, detail: Details) -> float:
        relevant_retrieved = 1.0 * detail[PRECISION_RELEVANT_RETRIEVED]
        retrieved = 1.0 * detail[PRECISION_RETRIEVED]
        if retrieved == 0.0:
            return 1.0
        else:
            return relevant_retrieved / retrieved


RECALL_NAME = "detection recall"
RECALL_RELEVANT = "relevant"
RECALL_RELEVANT_RETRIEVED = "relevant retrieved"


class DetectionRecall(DetectionErrorRate):
    """Detection recall

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as tp / (tp + fn), where tp is the duration of true positive
    (e.g. speech classified as speech), and fn is the duration of false
    negative (e.g. speech classified as non-speech).

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        probability_annotation = self._soft_annotation_if_requested(
            hypothesis, kwargs, reference
        )
        if probability_annotation is not None:
            confusion = self._probability_confusion(
                reference, probability_annotation, uem=uem
            )
            true_positive = confusion[ACCURACY_TRUE_POSITIVE]
            false_negative = confusion[ACCURACY_FALSE_NEGATIVE]
            return {
                RECALL_RELEVANT: true_positive + false_negative,
                RECALL_RELEVANT_RETRIEVED: true_positive,
            }

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        hypothesis_ = hypothesis.gaps(support=uem)

        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {}
        detail[RECALL_RELEVANT] = true_positive + false_negative
        detail[RECALL_RELEVANT_RETRIEVED] = true_positive

        return detail

    def compute_metric(self, detail: Details) -> float:
        relevant_retrieved = 1.0 * detail[RECALL_RELEVANT_RETRIEVED]
        relevant = 1.0 * detail[RECALL_RELEVANT]
        if relevant == 0.0:
            if relevant_retrieved == 0:
                return 1.0
            else:
                return 0.0
        else:
            return relevant_retrieved / relevant


DFS_NAME = "F[precision|recall]"
DFS_PRECISION_RETRIEVED = "retrieved"
DFS_RECALL_RELEVANT = "relevant"
DFS_RELEVANT_RETRIEVED = "relevant retrieved"


class DetectionPrecisionRecallFMeasure(UEMSupportMixin, BaseMetric):
    """Compute detection precision and recall, and return their F-score

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    beta : float, optional
        When beta > 1, greater importance is given to recall.
        When beta < 1, greater importance is given to precision.
        Defaults to 1.

    See also
    --------
    pyannote.metrics.detection.DetectionPrecision
    pyannote.metrics.detection.DetectionRecall
    pyannote.metrics.base.f_measure

    """

    @classmethod
    def metric_name(cls):
        return DFS_NAME

    @classmethod
    def metric_components(cls):
        return [DFS_PRECISION_RETRIEVED, DFS_RECALL_RELEVANT, DFS_RELEVANT_RETRIEVED]

    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.beta = beta

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        # Better to recompute everything from scratch instead of calling the
        # DetectionPrecision & DetectionRecall classes (we skip one of the loop
        # that computes the amount of true positives).
        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {
            DFS_PRECISION_RETRIEVED: true_positive + false_positive,
            DFS_RECALL_RELEVANT: true_positive + false_negative,
            DFS_RELEVANT_RETRIEVED: true_positive,
        }

        return detail

    def compute_metric(self, detail: Details) -> float:
        _, _, value = self.compute_metrics(detail=detail)
        return value

    def compute_metrics(
        self, detail: Details | None = None
    ) -> tuple[float, float, float]:

        detail = self.accumulated_ if detail is None else detail
        precision_retrieved = detail[DFS_PRECISION_RETRIEVED]
        recall_relevant = detail[DFS_RECALL_RELEVANT]
        relevant_retrieved = detail[DFS_RELEVANT_RETRIEVED]

        # Special cases : precision
        if precision_retrieved == 0.0:
            precision = 1
        else:
            precision = relevant_retrieved / precision_retrieved

        # Special cases : recall
        if recall_relevant == 0.0:
            if relevant_retrieved == 0:
                recall = 1.0
            else:
                recall = 0.0
        else:
            recall = relevant_retrieved / recall_relevant

        return precision, recall, f_measure(precision, recall, beta=self.beta)


DCF_NAME = "detection cost function"
DCF_POS_TOTAL = "positive class total"  # Total duration of positive class.
DCF_NEG_TOTAL = "negative class total"  # Total duration of negative class.
DCF_FALSE_ALARM = "false alarm"  # Total duration of false alarms.
DCF_MISS = "miss"  # Total duration of misses.


class DetectionCostFunction(UEMSupportMixin, BaseMetric):
    """Detection cost function.

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection. Inputs are expected to only contain segments
    corresponding to the positive class (e.g. speech regions). Gaps in the
    inputs considered as the negative class (e.g. non-speech regions).

    Detection cost function (DCF), as defined by NIST for OpenSAT 2019, is
    0.25*far + 0.75*missr, where far is the false alarm rate
    (i.e., the proportion of non-speech incorrectly classified as speech)
    and missr is the miss rate (i.e., the proportion of speech incorrectly
    classified as non-speech.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
        Defaults to 0.0.

    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    fa_weight : float, optional
        Weight for false alarm rate.
        Defaults to 0.25.

    miss_weight : float, optional
        Weight for miss rate.
        Defaults to 0.75.

    kwargs
        Keyword arguments passed to :class:`pyannote.metrics.base.BaseMetric`.

    References
    ----------
    "OpenSAT19 Evaluation Plan v2." https://www.nist.gov/system/files/documents/2018/11/05/opensat19_evaluation_plan_v2_11-5-18.pdf
    """

    def __init__(
        self, collar=0.0, skip_overlap=False, fa_weight=0.25, miss_weight=0.75, **kwargs
    ):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.fa_weight = fa_weight
        self.miss_weight = miss_weight

    @classmethod
    def metric_name(cls):
        return DCF_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [DCF_POS_TOTAL, DCF_NEG_TOTAL, DCF_FALSE_ALARM, DCF_MISS]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        # Obtain timelines corresponding to positive class.
        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        # Obtain timelines corresponding to negative class.
        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        # Compute total positive/negative durations.
        pos_dur = reference.duration()
        neg_dur = reference_.duration()

        # Compute total miss duration.
        miss_dur = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            miss_dur += (r & h_).duration

        # Compute total false alarm duration.
        fa_dur = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            fa_dur += (r_ & h).duration

        components = {
            DCF_POS_TOTAL: pos_dur,
            DCF_NEG_TOTAL: neg_dur,
            DCF_MISS: miss_dur,
            DCF_FALSE_ALARM: fa_dur,
        }

        return components

    def compute_metric(self, components: Details) -> float:
        def _compute_rate(num, denom):
            if denom == 0.0:
                if num == 0.0:
                    return 0.0
                return 1.0
            return num / denom

        # Compute false alarm rate.
        neg_dur = components[DCF_NEG_TOTAL]
        fa_dur = components[DCF_FALSE_ALARM]
        fa_rate = _compute_rate(fa_dur, neg_dur)

        # Compute miss rate.
        pos_dur = components[DCF_POS_TOTAL]
        miss_dur = components[DCF_MISS]
        miss_rate = _compute_rate(miss_dur, pos_dur)

        return self.fa_weight * fa_rate + self.miss_weight * miss_rate
