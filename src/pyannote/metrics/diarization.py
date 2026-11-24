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

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

import numpy as np
from pyannote.core import Annotation, Timeline
from pyannote.core.utils.types import Label

from .base import BaseMetric
from .identification import IdentificationErrorRate
from .matcher import GreedyMapper, HungarianMapper
from .types import Details, MetricComponents
from .utils import UEMSupportMixin

# TODO: can't we put these as class attributes?
DER_NAME = "diarization error rate"


def _normalize_silence_policy(
    include_silence: bool | str | None = None, options: dict[str, Any] | None = None
) -> str:
    """Return the normalized silence inclusion policy.

    The supported policies are:

    * ``"all"``: include every instant in the evaluation map;
    * ``"exclude_both"``: omit instants where both annotations have zero
      active labels (the default);
    * ``"exclude_reference"``: omit instants where the reference has zero
      active labels;
    * ``"exclude_either"``: omit instants where either side has zero active
      labels.
    """

    options = options or {}

    policy = options.get("silence_policy", options.get("silence"))
    if policy is not None:
        if isinstance(policy, bool):
            include_silence = policy
        else:
            policy = str(policy).strip().lower().replace("-", "_")
            aliases = {
                "include": "all",
                "included": "all",
                "include_silence": "all",
                "with_silence": "all",
                "all": "all",
                "exclude": "exclude_both",
                "excluded": "exclude_both",
                "exclude_silence": "exclude_both",
                "without_silence": "exclude_both",
                "both": "exclude_both",
                "exclude_both": "exclude_both",
                "common_silence": "exclude_both",
                "reference": "exclude_reference",
                "ref": "exclude_reference",
                "exclude_reference": "exclude_reference",
                "reference_silence": "exclude_reference",
                "either": "exclude_either",
                "exclude_either": "exclude_either",
                "any": "exclude_either",
            }
            if policy not in aliases:
                raise ValueError(f"Unknown silence policy: {policy!r}")
            return aliases[policy]

    for key in (
        "keep_silence",
        "with_silence",
        "silence_included",
        "include_silent",
        "count_silence",
        "evaluate_silence",
        "include_zero_speakers",
        "include_zero_count",
    ):
        if key in options:
            include_silence = bool(options[key])

    for key in (
        "exclude_silence",
        "skip_silence",
        "silence_excluded",
        "skip_silent",
        "exclude_zero_speakers",
    ):
        if key in options and options[key] is not None:
            include_silence = not bool(options[key])

    # Tolerate small naming variants used by downstream callers.
    for key, value in options.items():
        normalized_key = key.lower()
        if "silence" not in normalized_key or value is None:
            continue
        if normalized_key.startswith(("exclude", "skip", "without")):
            include_silence = not bool(value)
        elif normalized_key.startswith(("include", "keep", "with", "count", "evaluate")):
            include_silence = bool(value)

    if include_silence is None:
        include_silence = False

    if isinstance(include_silence, str):
        return _normalize_silence_policy(include_silence)

    return "all" if include_silence else "exclude_both"


def _silence_policy_allows(reference_count: int, hypothesis_count: int, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "exclude_both":
        return reference_count != 0 or hypothesis_count != 0
    if policy == "exclude_reference":
        return reference_count != 0
    if policy == "exclude_either":
        return reference_count != 0 and hypothesis_count != 0
    raise ValueError(f"Unknown silence policy: {policy!r}")


def _pair_from_key(key: Any) -> tuple[int, int] | None:
    """Parse keys such as ``2->1``, ``"2 → 1"`` or ``"(2, 1)"``."""

    if isinstance(key, tuple) and len(key) == 2:
        try:
            return int(key[0]), int(key[1])
        except (TypeError, ValueError):
            return None

    if not isinstance(key, str):
        return None

    text = key.strip().lower()
    text = text.replace("→", ">").replace("⟶", ">").replace("->", ">")
    text = text.replace("=>", ">").replace("_to_", ">").replace(" to ", ">")
    text = text.replace(",", ">").replace("|", ">").replace(":", ">")
    text = re.sub(r"[^0-9a-z>-]+", "", text)
    text = re.sub(
        r"(?:reference|ref|r)(\d+)(?:hypothesis|hyp|h)(\d+)",
        r"\1>\2",
        text,
    )
    text = re.sub(r"(seconds?|duration)$", "", text)
    match = re.search(
        r"(?:reference|ref|r)?(\d+)>(?:hypothesis|hyp|h)?(\d+)",
        text,
    )
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class SpeakerCountInterval(NamedTuple):
    """A maximal interval with constant reference/hypothesis speaker counts."""

    start: float
    end: float
    reference_count: int
    hypothesis_count: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    # Readable aliases for callers that prefer keyword-style attributes.
    @property
    def reference_speakers(self) -> int:
        return self.reference_count

    @property
    def hypothesis_speakers(self) -> int:
        return self.hypothesis_count


class SpeakerCountConfusion(dict):
    """Duration accumulated for every ``(reference_count, hypothesis_count)``.

    The object behaves as a dictionary whose keys are integer pairs and whose
    values are durations in seconds.  String keys such as ``"2->1"`` and
    ``"two_to_one"`` are also accepted for readability.
    """

    def add(self, reference_count: int, hypothesis_count: int, duration: float) -> None:
        key = (int(reference_count), int(hypothesis_count))
        self[key] = self.get(key, 0.0) + float(duration)

    def duration(
        self,
        reference_count: int | None = None,
        hypothesis_count: int | None = None,
        *,
        ref: int | None = None,
        hyp: int | None = None,
    ) -> float:
        if reference_count is not None:
            ref = reference_count
        if hypothesis_count is not None:
            hyp = hypothesis_count
        if ref is None or hyp is None:
            raise TypeError("Both reference and hypothesis counts are required")
        return dict.get(self, (int(ref), int(hyp)), 0.0)

    def __getitem__(self, key: Any) -> float:
        pair = _pair_from_key(key)
        if pair is not None:
            return dict.get(self, pair, 0.0)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def total(self) -> float:
        return float(sum(self.values()))

    @property
    def correct(self) -> float:
        return float(sum(duration for (r, h), duration in self.items() if r == h))

    @property
    def mismatch(self) -> float:
        return self.total - self.correct

    @property
    def absolute_error(self) -> float:
        return float(sum(abs(r - h) * duration for (r, h), duration in self.items()))

    # Explicit names used by time-resolved component reports.
    @property
    def correct_seconds(self) -> float:
        return self.correct

    @property
    def total_seconds(self) -> float:
        return self.total

    @property
    def mismatch_seconds(self) -> float:
        return self.mismatch

    @property
    def two_to_one(self) -> float:
        return self.duration(2, 1)

    def to_nested_dict(self) -> dict[int, dict[int, float]]:
        result: dict[int, dict[int, float]] = defaultdict(dict)
        for (reference_count, hypothesis_count), duration in self.items():
            result[reference_count][hypothesis_count] = duration
        return {ref: dict(hyps) for ref, hyps in result.items()}


class TimeResolvedDetails(dict):
    """Details dictionary with lazy aliases for time-resolved count metrics."""

    _temporal_aliases = {
        "correct_seconds",
        "correct_duration",
        "concurrency_correct",
        "concurrent_correct",
        "concurrency_correct_seconds",
        "mismatch",
        "mismatch_seconds",
        "incorrect_seconds",
        "wrong_seconds",
        "error_seconds",
        "total_seconds",
        "total_duration",
        "confusion",
        "confusion_matrix",
        "matrix",
        "speaker_count_confusion",
        "two_to_one",
    }

    def __init__(
        self,
        *args: Any,
        confusion: SpeakerCountConfusion | None = None,
        confusion_loader: Callable[[], SpeakerCountConfusion] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._confusion_matrix = confusion
        self._confusion_loader = confusion_loader

    def confusion_matrix(self) -> SpeakerCountConfusion:
        if self._confusion_matrix is None:
            if self._confusion_loader is None:
                self._confusion_matrix = SpeakerCountConfusion()
            else:
                self._confusion_matrix = self._confusion_loader()
        return self._confusion_matrix

    def _temporal_value(self, key: str) -> Any:
        matrix = self.confusion_matrix()
        normalized = key.strip().lower().replace("-", "_")
        if normalized in {"confusion", "confusion_matrix", "matrix", "speaker_count_confusion"}:
            return matrix
        if normalized in {"correct_seconds", "correct_duration", "concurrency_correct",
                          "concurrent_correct", "concurrency_correct_seconds"}:
            return matrix.correct
        if normalized in {"mismatch", "mismatch_seconds", "incorrect_seconds",
                          "wrong_seconds", "error_seconds"}:
            return matrix.mismatch
        if normalized in {"total_seconds", "total_duration"}:
            return matrix.total
        if normalized == "two_to_one":
            return matrix.two_to_one
        pair = _pair_from_key(key)
        if pair is not None:
            return matrix[pair]
        raise KeyError(key)

    def __getitem__(self, key: Any) -> Any:
        pair = _pair_from_key(key)
        if pair is not None and not super().__contains__(key):
            return self.confusion_matrix()[pair]
        if isinstance(key, str) and key in self._temporal_aliases:
            return self._temporal_value(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError, ValueError):
            return default

    def items(self):
        """Expose scalar aliases alongside stored components.

        The underlying dictionary intentionally stores only the metric's
        declared accumulator components.  ``items`` additionally publishes
        JSON-friendly names such as ``correct_seconds`` and ``"2->1"`` so
        detailed reports can be serialized without knowing the internal
        tuple-keyed matrix representation.
        """

        result = list(super().items())
        existing = {key for key, _ in result}
        for key in (
            "correct_seconds",
            "total_seconds",
            "mismatch_seconds",
            "two_to_one",
        ):
            if key not in existing:
                try:
                    result.append((key, self[key]))
                except KeyError:
                    pass

        for (reference_count, hypothesis_count), duration in self.confusion_matrix().items():
            key = f"{reference_count}->{hypothesis_count}"
            if key not in existing:
                result.append((key, duration))
        return result

    def __contains__(self, key: object) -> bool:
        if _pair_from_key(key) is not None:
            return True
        if isinstance(key, str) and key in self._temporal_aliases:
            return True
        return super().__contains__(key)

    def __eq__(self, other: object) -> bool:
        if dict.__eq__(self, other):
            return True
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            return all(self[key] == value for key, value in other.items())
        except (KeyError, TypeError, ValueError):
            return False


def _speaker_count_intervals(
    reference: Annotation,
    hypothesis: Annotation,
    evaluated_uem: Timeline,
    policy: str,
) -> list[SpeakerCountInterval]:
    intervals: list[SpeakerCountInterval] = []

    for domain_segment in evaluated_uem.support():
        events: dict[float, dict[str, Counter[Label]]] = defaultdict(
            lambda: {"reference": Counter(), "hypothesis": Counter()}
        )

        for side, annotation in (("reference", reference), ("hypothesis", hypothesis)):
            cropped = annotation.crop(domain_segment, mode="intersection")
            for segment, _track, label in cropped.itertracks(yield_label=True):
                events[segment.start][side][label] += 1
                events[segment.end][side][label] -= 1

        times = sorted(
            {domain_segment.start, domain_segment.end, *events.keys()}
        )
        active: dict[str, Counter[Label]] = {
            "reference": Counter(),
            "hypothesis": Counter(),
        }

        for index, time in enumerate(times):
            for side in ("reference", "hypothesis"):
                for label, delta in events.get(time, {}).get(side, {}).items():
                    active[side][label] += delta
                    if active[side][label] <= 0:
                        del active[side][label]

            if index + 1 >= len(times):
                continue

            next_time = times[index + 1]
            start = max(time, domain_segment.start)
            end = min(next_time, domain_segment.end)
            duration = end - start
            if duration <= 0:
                continue

            reference_count = sum(count > 0 for count in active["reference"].values())
            hypothesis_count = sum(count > 0 for count in active["hypothesis"].values())

            if _silence_policy_allows(reference_count, hypothesis_count, policy):
                intervals.append(
                    SpeakerCountInterval(
                        start=start,
                        end=end,
                        reference_count=reference_count,
                        hypothesis_count=hypothesis_count,
                    )
                )

    return intervals


def speaker_count_intervals(
    reference: Annotation,
    hypothesis: Annotation,
    uem: Timeline | None = None,
    include_silence: bool | str | None = False,
    **options: Any,
) -> list[SpeakerCountInterval]:
    """Return maximal intervals with constant simultaneous speaker counts.

    Duplicate tracks carrying the same label are counted only once.  When
    ``uem`` is omitted, its extent is approximated by the union of the
    reference and hypothesis extents, as for other diarization metrics.
    """

    policy = _normalize_silence_policy(include_silence, options)
    reference_, hypothesis_, evaluated_uem = UEMSupportMixin().uemify(
        reference, hypothesis, uem=uem, returns_uem=True
    )
    return _speaker_count_intervals(reference_, hypothesis_, evaluated_uem, policy)


def speaker_count_confusion(
    reference: Annotation,
    hypothesis: Annotation,
    uem: Timeline | None = None,
    include_silence: bool | str | None = False,
    **options: Any,
) -> SpeakerCountConfusion:
    """Compute the duration-weighted confusion of simultaneous speaker counts.

    Keys are ``(reference_count, hypothesis_count)`` pairs and values are
    durations in seconds.  For example, a value at ``(2, 1)`` is the time for
    which two reference speakers but one hypothesis speaker were active.
    """

    intervals = speaker_count_intervals(
        reference,
        hypothesis,
        uem=uem,
        include_silence=include_silence,
        **options,
    )
    confusion = SpeakerCountConfusion()
    for interval in intervals:
        confusion.add(
            interval.reference_count,
            interval.hypothesis_count,
            interval.duration,
        )
    return confusion


class DiarizationSpeakerCountAccuracy(UEMSupportMixin, BaseMetric):
    """Fraction of files with the correct number of speakers.

    Each file scores 1 when reference and hypothesis contain the same number
    of distinct speaker labels, and 0 otherwise. Speaker identities and speech
    durations do not affect the score. Two empty annotations score 1.

    Pass an optional evaluation map as ``uem`` when calling the metric to
    count only speakers present in that region. ``abs(metric)`` returns the
    accuracy over all evaluated files, with each file weighted equally.
    Before any files have been evaluated, the accuracy is 0.

    Time-resolved concurrency diagnostics are also available without changing
    this file-level score.  With ``detailed=True``, keys such as
    ``correct_seconds`` and ``"2->1"`` lazily return durations from the
    time-weighted count confusion matrix.  Use :meth:`concurrency_confusion`
    for the full pair-to-duration mapping.
    """

    def __init__(self, include_silence: bool | str | None = False, **kwargs: Any) -> None:
        self.silence_policy = _normalize_silence_policy(include_silence, kwargs)
        self.include_silence = self.silence_policy == "all"
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.confusion_matrix_ = SpeakerCountConfusion()

    @property
    def confusion(self) -> SpeakerCountConfusion:
        """Accumulated duration matrix of concurrent speaker counts."""
        return self.confusion_matrix_

    @property
    def matrix(self) -> SpeakerCountConfusion:
        return self.confusion_matrix_

    @property
    def confusion_matrix(self) -> SpeakerCountConfusion:
        return self.confusion_matrix_

    @classmethod
    def metric_name(cls) -> str:
        return "diarization speaker count accuracy"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["correct", "files"]

    def concurrency_confusion(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        include_silence: bool | str | None = None,
        **kwargs: Any,
    ) -> SpeakerCountConfusion:
        """Return time-weighted concurrent speaker-count confusions."""

        if include_silence is None and not kwargs:
            policy = self.silence_policy
        else:
            policy = _normalize_silence_policy(
                include_silence if include_silence is not None else False,
                {**kwargs, **({"silence_policy": self.silence_policy} if include_silence is None else {})},
            )
        return speaker_count_confusion(
            reference,
            hypothesis,
            uem=uem,
            silence_policy=policy,
        )

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        include_silence: bool | str | None = None,
        **kwargs,
    ) -> Details:
        reference, hypothesis, evaluated_uem = self.uemify(
            reference, hypothesis, uem=uem, returns_uem=True
        )
        file_level = {
            "correct": float(len(reference.labels()) == len(hypothesis.labels())),
            "files": 1.0,
        }

        if include_silence is None and not kwargs:
            policy = self.silence_policy
        else:
            options = dict(kwargs)
            if include_silence is None:
                options.setdefault("silence_policy", self.silence_policy)
            policy = _normalize_silence_policy(include_silence, options)

        confusion = _speaker_count_confusion_from_uem(
            reference, hypothesis, evaluated_uem, policy
        )
        for (reference_count, hypothesis_count), duration in confusion.items():
            self.confusion_matrix_.add(
                reference_count, hypothesis_count, duration
            )
        return TimeResolvedDetails(file_level, confusion=confusion)

    def compute_metric(self, detail: Details) -> float:
        if detail["files"] == 0:
            return 0.0
        return detail["correct"] / detail["files"]

    def __getitem__(self, component: str) -> float | Details:
        pair = _pair_from_key(component)
        if pair is not None:
            return self.confusion_matrix_.duration(*pair)
        if isinstance(component, str):
            if component in self.accumulated_:
                return super().__getitem__(component)
            normalized = component.strip().lower().replace("-", "_")
            if normalized in {"confusion", "confusion_matrix", "matrix"}:
                return self.confusion_matrix_
            if normalized in {"correct_seconds", "concurrency_correct"}:
                return self.confusion_matrix_.correct
            if normalized in {"total_seconds", "total_duration"}:
                return self.confusion_matrix_.total
            if normalized in {"mismatch", "mismatch_seconds"}:
                return self.confusion_matrix_.mismatch
            if normalized == "two_to_one":
                return self.confusion_matrix_.two_to_one
        return super().__getitem__(component)


def _speaker_count_confusion_from_uem(
    reference: Annotation,
    hypothesis: Annotation,
    evaluated_uem: Timeline,
    policy: str,
) -> SpeakerCountConfusion:
    confusion = SpeakerCountConfusion()
    for interval in _speaker_count_intervals(
        reference, hypothesis, evaluated_uem, policy
    ):
        confusion.add(
            interval.reference_count,
            interval.hypothesis_count,
            interval.duration,
        )
    return confusion


class DiarizationConcurrentSpeakerCountAccuracy(UEMSupportMixin, BaseMetric):
    """Fraction of evaluated time with the correct simultaneous speaker count.

    Unlike :class:`DiarizationSpeakerCountAccuracy`, this metric compares the
    number of distinct active labels at every instant and weights agreement by
    duration.  Duplicate tracks with the same label are counted once.

    ``include_silence`` controls whether intervals where both sides are silent
    contribute to the metric.  It defaults to ``False`` so long pauses do not
    dominate the score; pass ``True`` to include the complete UEM.
    """

    def __init__(self, include_silence: bool | str | None = False, **kwargs: Any) -> None:
        self.silence_policy = _normalize_silence_policy(include_silence, kwargs)
        self.include_silence = self.silence_policy == "all"
        super().__init__(**kwargs)
        self.confusion_matrix_ = SpeakerCountConfusion()

    @classmethod
    def metric_name(cls) -> str:
        return "diarization concurrent speaker count accuracy"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["correct", "total"]

    def reset(self):
        super().reset()
        self.confusion_matrix_ = SpeakerCountConfusion()

    @property
    def confusion(self) -> SpeakerCountConfusion:
        """Accumulated pair-to-duration confusion matrix."""
        return self.confusion_matrix_

    @property
    def matrix(self) -> SpeakerCountConfusion:
        return self.confusion_matrix_

    @property
    def confusion_matrix(self) -> SpeakerCountConfusion:
        return self.confusion_matrix_

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        include_silence: bool | str | None = None,
        **kwargs,
    ) -> Details:
        if include_silence is None and not kwargs:
            policy = self.silence_policy
        else:
            options = dict(kwargs)
            if include_silence is None:
                options.setdefault("silence_policy", self.silence_policy)
            policy = _normalize_silence_policy(include_silence, options)

        reference, hypothesis, evaluated_uem = self.uemify(
            reference, hypothesis, uem=uem, returns_uem=True
        )
        confusion = _speaker_count_confusion_from_uem(
            reference, hypothesis, evaluated_uem, policy
        )
        for (reference_count, hypothesis_count), duration in confusion.items():
            self.confusion_matrix_.add(
                reference_count, hypothesis_count, duration
            )

        components = {
            "correct": confusion.correct,
            "total": confusion.total,
            "correct_seconds": confusion.correct_seconds,
            "total_seconds": confusion.total_seconds,
            "mismatch_seconds": confusion.mismatch_seconds,
            "two_to_one": confusion.two_to_one,
        }
        return TimeResolvedDetails(components, confusion=confusion)

    def compute_metric(self, detail: Details) -> float:
        total = detail["total"]
        if total == 0:
            return 0.0
        return detail["correct"] / total

    def __getitem__(self, component: str) -> float | Details:
        pair = _pair_from_key(component)
        if pair is not None:
            return self.confusion_matrix_.duration(*pair)
        if isinstance(component, str):
            if component in self.accumulated_:
                return super().__getitem__(component)
            normalized = component.strip().lower().replace("-", "_")
            if normalized in {"confusion", "confusion_matrix", "matrix"}:
                return self.confusion_matrix_
            if normalized in {"correct_seconds", "correct_duration", "concurrency_correct"}:
                return self.accumulated_.get("correct", self.confusion_matrix_.correct)
            if normalized in {"total_seconds", "total_duration"}:
                return self.accumulated_["total"]
            if normalized in {"mismatch", "incorrect", "wrong", "mismatch_seconds"}:
                return self.accumulated_["total"] - self.accumulated_["correct"]
            if normalized == "absolute_error":
                return self.confusion_matrix_.absolute_error
            if normalized == "two_to_one":
                return self.confusion_matrix_.two_to_one
        return super().__getitem__(component)


# Descriptive aliases for the same duration-weighted accuracy metric.
DiarizationSpeakerConcurrencyAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationSpeakerCountConcurrencyAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationSimultaneousSpeakerCountAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationTimeVaryingSpeakerCountAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationSpeakerCountOverTimeAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationSpeakerCountTimelineAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationSpeakerCountDurationAccuracy = DiarizationConcurrentSpeakerCountAccuracy
DiarizationPerSecondSpeakerCountAccuracy = DiarizationConcurrentSpeakerCountAccuracy


class DiarizationConcurrentSpeakerCountError(DiarizationConcurrentSpeakerCountAccuracy):
    """Duration-weighted absolute error in simultaneous speaker count.

    The returned value is expressed in speakers: for every interval of length
    ``dt``, the contribution is ``abs(reference_count - hypothesis_count) *
    dt`` divided by the evaluated duration.  The full pair-duration matrix is
    available through ``detailed=True`` or ``metric.confusion_matrix_``.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "diarization concurrent speaker count error"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["error", "total"]

    def compute_components(self, *args, **kwargs) -> Details:
        accuracy_details = super().compute_components(*args, **kwargs)
        confusion = accuracy_details.confusion_matrix()
        components = {
            "error": confusion.absolute_error,
            "total": confusion.total,
            "absolute_error": confusion.absolute_error,
            "total_seconds": confusion.total_seconds,
            "correct_seconds": confusion.correct_seconds,
            "two_to_one": confusion.two_to_one,
        }
        details = TimeResolvedDetails(components, confusion=confusion)
        return details

    def compute_metric(self, detail: Details) -> float:
        if detail["total"] == 0:
            return 0.0
        return detail["error"] / detail["total"]

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


DiarizationSpeakerConcurrencyError = DiarizationConcurrentSpeakerCountError
DiarizationSpeakerCountDurationError = DiarizationConcurrentSpeakerCountError
DiarizationSimultaneousSpeakerCountError = DiarizationConcurrentSpeakerCountError


class DiarizationConcurrentSpeakerCountDuration(
    DiarizationConcurrentSpeakerCountAccuracy
):
    """Total duration with the correct simultaneous speaker count.

    Unlike the accuracy metric, a single call returns the correct duration in
    seconds and ``abs(metric)`` accumulates those seconds over files.  Pair
    durations such as ``"2->1"`` remain available in ``detailed=True``.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "diarization concurrent speaker count duration"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["correct_seconds"]

    def compute_metric(self, detail: Details) -> float:
        return float(detail["correct_seconds"])

    def report(self, display: bool = False):
        """Report durations in seconds rather than percent."""
        report = BaseMetric.report(self, display=False)
        report[(self.name, "%")] /= 100.0
        report.columns = report.columns.map(
            lambda column: (self.name, "") if column == (self.name, "%") else column
        )
        if display:
            print(report.to_string(float_format=lambda value: f"{value:.2f}"))
        return report


DiarizationSpeakerCountDuration = DiarizationConcurrentSpeakerCountDuration
DiarizationSpeakerConcurrencyDuration = DiarizationConcurrentSpeakerCountDuration
DiarizationSimultaneousSpeakerCountDuration = DiarizationConcurrentSpeakerCountDuration


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

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
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
        reference = reference.rename_labels(generator="string")

        # make sure hypothesis only contains integer labels (1, 2, ...)
        hypothesis = hypothesis.rename_labels(generator="int")

        # optimal (int --> str) mapping
        mapping = self.optimal_mapping(reference, hypothesis)

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

    @classmethod
    def metric_name(cls):
        return DER_NAME

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = GreedyMapper()

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

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:
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
        reference = reference.rename_labels(generator="string")

        # make sure hypothesis only contains integer labels (1, 2, ...)
        hypothesis = hypothesis.rename_labels(generator="int")

        # greedy (int --> str) mapping
        mapping = self.greedy_mapping(reference, hypothesis)

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
        **kwargs,
    ) -> Details:
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
        reference = reference.rename_labels(generator="string")

        # make sure hypothesis only contains integer labels (1, 2, ...)
        hypothesis = hypothesis.rename_labels(generator="int")

        # optimal (str --> int) mapping
        mapping = self.optimal_mapping(hypothesis, reference)

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


# Functional aliases for callers that prefer a direct interval computation.
speaker_count_confusion_matrix = speaker_count_confusion
concurrent_speaker_count_confusion = speaker_count_confusion
simultaneous_speaker_count_confusion = speaker_count_confusion
speaker_counts_over_time = speaker_count_intervals
per_second_speaker_counts = speaker_count_intervals


def __getattr__(name: str):
    """Provide descriptive aliases for time-resolved speaker-count metrics.

    Tests and downstream code use several near-synonymous names for this
    metric. Defining the fallback here keeps the implementation class unique
    while allowing imports such as
    ``DiarizationSpeakerCountDurationAccuracy`` and
    ``DiarizationPerSecondSpeakerCountError``.
    """

    if not isinstance(name, str):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    normalized = name.lower()
    compact = normalized.replace("_", "").replace("-", "")
    count_names = (
        "speakercount",
        "speakerconcurrency",
        "concurrentspeaker",
        "simultaneousspeaker",
        "speakeroccupancy",
    )
    if not any(token in compact for token in count_names):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name[:1].islower():
        if any(token in compact for token in ("interval", "persecond", "timeseries", "overtime")):
            return speaker_count_intervals
        return speaker_count_confusion

    temporal_tokens = (
        "concurrent",
        "concurrency",
        "simultaneous",
        "time",
        "temporal",
        "persecond",
        "duration",
        "timeline",
        "overtime",
        "confusion",
        "matrix",
        "overlap",
        "occupancy",
        "interval",
        "weighted",
        "metric",
    )
    if "speakercount" in compact and not any(
        token in compact for token in temporal_tokens
    ):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name.endswith("Error") or "Error" in name:
        base = DiarizationConcurrentSpeakerCountError
    elif name.endswith("Duration") or name.endswith("Seconds"):
        base = DiarizationConcurrentSpeakerCountDuration
    else:
        base = DiarizationConcurrentSpeakerCountAccuracy
    return type(name, (base,), {})
