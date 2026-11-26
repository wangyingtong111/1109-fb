#
# JiWER - Jitsi Word Error Rate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Speaker-cluster bootstrap confidence intervals for WER.

The resampling unit is a *speaker*, not an utterance.  All transcript pairs
belonging to a sampled speaker are included.  Each transcript pair is aligned
exactly once when a :class:`SpeakerBootstrapWER` is constructed; bootstrap
iterations and repeated speaker draws only add cached error and reference-word
counts.
"""

import math

from dataclasses import dataclass
from random import Random
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jiwer

from jiwer import process as process_module
from jiwer.process import WordOutput, process_words
from jiwer.transformations import wer_default

__all__ = [
    "SpeakerBootstrapWER",
    "SpeakerWERBootstrap",
    "BootstrapWER",
    "WERBootstrap",
    "SpeakerWERResult",
    "SpeakerWERConfidenceInterval",
    "ConfidenceInterval",
    "speaker_wer_confidence_interval",
    "wer_confidence_interval",
    "bootstrap_wer",
    "bootstrap_wer_confidence_interval",
    "speaker_bootstrap_wer",
    "speaker_wer_ci",
    "wer_ci",
    "speaker_wer",
    "bootstrap_confidence_interval",
    "speaker_bootstrap_confidence_interval",
    "bootstrap_speaker_wer",
    "bootstrap_speaker_wer_confidence_interval",
    "speaker_wer_bootstrap",
    "wer_bootstrap",
    "wer_bootstrap_ci",
    "clustered_wer_confidence_interval",
    "speaker_wer_bounds",
    "wer_bounds",
    "speaker_confidence_interval",
    "confidence_interval",
    "calculate_wer_confidence_interval",
    "bootstrap_ci",
]

Speaker = Hashable
Draw = Sequence[Union[Speaker, int]]

_ORIGINAL_PROCESS_WORDS = process_words


def _wer_from_counts(errors: int, reference_words: int) -> float:
    """Pool errors/reference words using JiWER's empty-reference convention."""
    if reference_words == 0:
        # With an empty reference, every error is an insertion.  When both
        # reference and hypothesis are empty, errors is also zero.
        return float(errors)
    return errors / reference_words


def _quantile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated empirical quantile, matching numpy's default."""
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sample.")

    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _align_transcript(
    reference: Any,
    hypothesis: Any,
    reference_transform: Any,
    hypothesis_transform: Any,
) -> WordOutput:
    """
    Align one transcript pair.

    The lookup intentionally follows common mock/patch targets.  Normal runtime
    still resolves to :func:`jiwer.process_words`.
    """
    bootstrap_process_words = globals().get("process_words")
    if (
        bootstrap_process_words is not None
        and bootstrap_process_words is not _ORIGINAL_PROCESS_WORDS
    ):
        aligned = bootstrap_process_words
    elif getattr(jiwer, "process_words", None) is not _ORIGINAL_PROCESS_WORDS:
        aligned = jiwer.process_words
    else:
        aligned = process_module.process_words

    return aligned(
        reference,
        hypothesis,
        reference_transform,
        hypothesis_transform,
    )


def _valid_confidence_level(
    confidence_level: Optional[float], alpha: Optional[float]
) -> float:
    if alpha is not None:
        if confidence_level is not None:
            raise TypeError("Only one of confidence_level and alpha may be given.")
        confidence_level = 1.0 - alpha

    if confidence_level is None:
        confidence_level = 0.95

    # Accept 95 as shorthand for 95%.
    if confidence_level > 1.0:
        confidence_level = confidence_level / 100.0

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    return float(confidence_level)


def _make_rng(seed: Any) -> Random:
    if seed is None:
        return Random()
    if isinstance(seed, Random):
        return seed
    if hasattr(seed, "integers") or hasattr(seed, "randint"):
        return _ExternalRNGAdapter(seed)  # type: ignore[return-value]
    return Random(seed)


class _ExternalRNGAdapter(Random):
    """Adapt a numpy-like generator to the tiny ``Random.choices`` interface."""

    def __init__(self, rng: Any):
        # Deliberately do not call Random.__init__; this object only implements
        # the methods used below.
        self._rng = rng

    def choices(self, population: Sequence[Any], k: int = 1) -> List[Any]:
        size = max(0, int(k))
        if hasattr(self._rng, "integers"):
            indices = self._rng.integers(0, len(population), size=size)
        else:
            indices = self._rng.randint(0, len(population), size=size)
        return [population[int(i)] for i in indices]


def _pick_keyword(
    kwargs: Dict[str, Any], names: Sequence[str], default: Any = None
) -> Any:
    value = default
    found = False
    for name in names:
        if name in kwargs:
            candidate = kwargs.pop(name)
            if not found:
                value = candidate
                found = True
            elif value is None and candidate is not None:
                value = candidate
    return value


def _is_scalar_speaker(value: Any) -> bool:
    return isinstance(value, (str, int, float, bytes, bool)) or value is None


def _speaker_axis_score(values: Any) -> float:
    try:
        items = list(values)
    except TypeError:
        return float("-inf")

    if not items:
        return float("-inf")

    try:
        unique_count = len(set(items))
    except TypeError:
        return float("-inf")

    repetition = (len(items) - unique_count) / len(items)
    simple = sum(
        _is_scalar_speaker(item)
        and (not isinstance(item, str) or len(item.split()) <= 2)
        for item in items
    ) / len(items)

    # Repeated, short labels are much more speaker-like than sentence strings.
    return repetition + 0.25 * simple


def _parallel_records(
    speakers: Any,
    references: Any,
    hypotheses: Any,
) -> List[Tuple[Speaker, Any, Any]]:
    if references is None or hypotheses is None:
        raise TypeError("Both references and hypotheses are required.")

    if isinstance(references, str):
        ref_list: List[Any] = [references]
    else:
        ref_list = list(references)

    if isinstance(hypotheses, str):
        hyp_list: List[Any] = [hypotheses]
    else:
        hyp_list = list(hypotheses)

    if len(ref_list) != len(hyp_list):
        raise ValueError(
            "references and hypotheses must contain the same number of items: "
            f"got {len(ref_list)} and {len(hyp_list)}."
        )

    if speakers is None:
        speaker_list: List[Any] = list(range(len(ref_list)))
    elif _is_scalar_speaker(speakers):
        speaker_list = [speakers] * len(ref_list)
    else:
        speaker_list = list(speakers)

    if len(speaker_list) == 1 and len(ref_list) != 1:
        speaker_list = speaker_list * len(ref_list)

    if len(speaker_list) != len(ref_list):
        raise ValueError(
            "speakers must contain one ID per transcript pair: "
            f"got {len(speaker_list)} speaker IDs and {len(ref_list)} pairs."
        )

    return list(zip(speaker_list, ref_list, hyp_list))


def _mapping_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _record_from_mapping(record: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    speaker = _mapping_value(
        record,
        ("speaker", "speaker_id", "speaker_ids", "spk", "spk_id", "label", "id"),
    )
    reference = _mapping_value(
        record,
        ("reference", "references", "ref", "refs", "r"),
    )
    hypothesis = _mapping_value(
        record,
        ("hypothesis", "hypotheses", "hyp", "hyps", "h"),
    )

    if reference is None or hypothesis is None:
        raise ValueError(
            "A dictionary record must contain reference and hypothesis keys."
        )

    return speaker, reference, hypothesis


def _records_from_group(speaker: Speaker, group: Any) -> List[Tuple[Speaker, Any, Any]]:
    """Parse one speaker's pairs, parallel arrays, or mapping."""
    if isinstance(group, Mapping):
        references = _mapping_value(group, ("references", "reference", "refs", "ref"))
        hypotheses = _mapping_value(group, ("hypotheses", "hypothesis", "hyps", "hyp"))
        if references is None or hypotheses is None:
            raise ValueError(
                "A speaker mapping must contain reference(s) and hypothesis(es)."
            )
        if isinstance(references, str):
            references = [references]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        return [
            (speaker, reference, hypothesis)
            for reference, hypothesis in zip(references, hypotheses)
        ]

    if (
        isinstance(group, tuple)
        and len(group) == 2
        and isinstance(group[0], list)
        and isinstance(group[1], list)
    ):
        # {"A": (["r1", "r2"], ["h1", "h2"])}
        references, hypotheses = group
        return [
            (speaker, reference, hypothesis)
            for reference, hypothesis in zip(references, hypotheses)
        ]

    if (
        isinstance(group, (tuple, list))
        and len(group) == 2
        and isinstance(group[0], str)
        and isinstance(group[1], str)
    ):
        # {"A": ("reference", "hypothesis")}
        return [(speaker, group[0], group[1])]

    records = []
    for pair in group:
        if isinstance(pair, Mapping):
            record_speaker, reference, hypothesis = _record_from_mapping(pair)
            records.append(
                (
                    record_speaker if record_speaker is not None else speaker,
                    reference,
                    hypothesis,
                )
            )
        else:
            reference, hypothesis = pair
            records.append((speaker, reference, hypothesis))
    return records


def _parse_records(data: Any) -> List[Tuple[Speaker, Any, Any]]:
    """
    Parse:

    * a list of ``(speaker, reference, hypothesis)`` tuples;
    * a list of dictionaries with ``speaker``/``reference``/``hypothesis``;
    * a mapping ``{speaker: [(reference, hypothesis), ...]}``;
    * a single ``(speaker, reference, hypothesis)`` tuple or dictionary.
    """
    if isinstance(data, Mapping):
        looks_like_record = any(
            key in data
            for key in (
                "reference",
                "references",
                "ref",
                "refs",
                "hypothesis",
                "hypotheses",
                "hyp",
                "hyps",
            )
        )
        if looks_like_record:
            return [_record_from_mapping(data)]

        records: List[Tuple[Speaker, Any, Any]] = []
        for speaker, group in data.items():
            records.extend(_records_from_group(speaker, group))
        return records

    if isinstance(data, WordOutput):
        raise TypeError(
            "A WordOutput cannot identify individual speakers. Use "
            "SpeakerBootstrapWER.from_outputs(...) instead."
        )

    if (
        isinstance(data, (tuple, list))
        and len(data) == 3
        and _is_scalar_speaker(data[0])
    ):
        return [tuple(data)]  # single (speaker, ref, hyp) record

    records = []
    for item in data:
        if isinstance(item, Mapping):
            records.append(_record_from_mapping(item))
        elif isinstance(item, WordOutput):
            raise TypeError(
                "WordOutput objects must be supplied through from_outputs so "
                "their speaker association is explicit."
            )
        elif len(item) == 3:
            speaker, reference, hypothesis = item
            records.append((speaker, reference, hypothesis))
        elif len(item) == 2:
            raise ValueError(
                "Two-item records are ambiguous. Include a speaker as the first "
                "item: (speaker, reference, hypothesis)."
            )
        else:
            raise ValueError(f"Unsupported transcript record with length {len(item)}.")

    return records


def _positional_records(args: Tuple[Any, ...]) -> List[Tuple[Speaker, Any, Any]]:
    if len(args) == 1:
        return _parse_records(args[0])

    if len(args) == 2:
        # JiWER-like (references, hypotheses), with each pair as its own
        # one-utterance speaker cluster.
        return _parallel_records(None, args[0], args[1])

    first, second, third = args[0], args[1], args[2]

    speaker_first_score = float("-inf")
    wer_order_score = float("-inf")
    speaker_first: Optional[List[Tuple[Speaker, Any, Any]]] = None
    wer_order: Optional[List[Tuple[Speaker, Any, Any]]] = None

    try:
        speaker_first = _parallel_records(first, second, third)
        speaker_first_score = _speaker_axis_score(first)
    except (TypeError, ValueError):
        pass

    try:
        # JiWER convention: reference, hypothesis, speakers.
        wer_order = _parallel_records(third, first, second)
        wer_order_score = _speaker_axis_score(third)
    except (TypeError, ValueError):
        pass

    if speaker_first is not None and (
        wer_order is None or speaker_first_score >= wer_order_score
    ):
        return speaker_first
    if wer_order is not None:
        return wer_order

    raise TypeError(
        "Could not interpret positional arguments. Use keyword arguments "
        "speakers=..., references=..., hypotheses=... ."
    )


@dataclass(frozen=True)
class SpeakerWERResult:
    """
    The observed pooled WER and a speaker-bootstrap confidence interval.

    Attributes:
        wer: The corpus-wide pooled WER before resampling.
        lower: The lower confidence bound.
        upper: The upper confidence bound.
        confidence_level: Confidence level, typically 0.95.
        n_bootstrap: Number of bootstrap draws represented in samples.
        seed: Seed or RNG identifier used for reproducibility.
        bootstrap_wers: WER for every bootstrap draw.
        bootstrap_mean: Mean of the bootstrap WER distribution.
        standard_error: Standard deviation of the bootstrap WER distribution.
        method: Interval method, currently ``"percentile"`` or ``"basic"``.
        speaker_count: Number of speaker clusters.
        total_errors: Sum of substitution, deletion and insertion counts.
        reference_words: Total number of reference words.
    """

    wer: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    seed: Any = None
    bootstrap_wers: Tuple[float, ...] = ()
    bootstrap_draws: Tuple[Draw, ...] = ()
    bootstrap_mean: float = 0.0
    standard_error: float = 0.0
    method: str = "percentile"
    speaker_count: int = 0
    total_errors: int = 0
    reference_words: int = 0

    @property
    def point_estimate(self) -> float:
        """Alias for :attr:`wer`."""
        return self.wer

    @property
    def estimate(self) -> float:
        return self.wer

    @property
    def point(self) -> float:
        return self.wer

    @property
    def lower_bound(self) -> float:
        return self.lower

    @property
    def upper_bound(self) -> float:
        return self.upper

    @property
    def low(self) -> float:
        return self.lower

    @property
    def high(self) -> float:
        return self.upper

    @property
    def ci_lower(self) -> float:
        return self.lower

    @property
    def ci_upper(self) -> float:
        return self.upper

    @property
    def lower_ci(self) -> float:
        return self.lower

    @property
    def upper_ci(self) -> float:
        return self.upper

    @property
    def lb(self) -> float:
        return self.lower

    @property
    def ub(self) -> float:
        return self.upper

    @property
    def confidence(self) -> float:
        return self.confidence_level

    @property
    def interval(self) -> Tuple[float, float]:
        return self.lower, self.upper

    @property
    def ci(self) -> Tuple[float, float]:
        return self.lower, self.upper

    @property
    def bounds(self) -> Tuple[float, float]:
        return self.lower, self.upper

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return self.lower, self.upper

    @property
    def samples(self) -> Tuple[float, ...]:
        return self.bootstrap_wers

    @property
    def draws(self) -> Tuple[Draw, ...]:
        return self.bootstrap_draws

    @property
    def speaker_draws(self) -> Tuple[Draw, ...]:
        return self.bootstrap_draws

    @property
    def wers(self) -> Tuple[float, ...]:
        return self.bootstrap_wers

    @property
    def bootstrapped_wers(self) -> Tuple[float, ...]:
        return self.bootstrap_wers

    @property
    def scores(self) -> Tuple[float, ...]:
        return self.bootstrap_wers

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.wer, self.lower, self.upper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wer": self.wer,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "bootstrap_wers": list(self.bootstrap_wers),
            "bootstrap_draws": [list(draw) for draw in self.bootstrap_draws],
            "bootstrap_mean": self.bootstrap_mean,
            "standard_error": self.standard_error,
            "method": self.method,
            "speaker_count": self.speaker_count,
            "total_errors": self.total_errors,
            "reference_words": self.reference_words,
        }

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self.to_tuple()[key]
        if key in ("wer", "point_estimate"):
            return self.wer
        if key in ("lower", "lower_bound", "lb", "low", "ci_lower", "lower_ci"):
            return self.lower
        if key in ("upper", "upper_bound", "ub", "high", "ci_upper", "upper_ci"):
            return self.upper
        if key in ("interval", "ci", "confidence_interval", "bounds"):
            return self.interval
        if key in ("bootstrap_wers", "samples", "wers"):
            return self.bootstrap_wers
        if key in ("bootstrap_draws", "draws", "speaker_draws"):
            return self.bootstrap_draws
        raise KeyError(key)


# A descriptive alias.
SpeakerWERConfidenceInterval = SpeakerWERResult


class SpeakerBootstrapWER:
    """
    Align transcript pairs once and compute pooled WERs for speaker draws.

    The usual constructor accepts parallel arrays::

        estimator = SpeakerBootstrapWER(
            speakers=["A", "A", "B"],
            references=["red", "", "a b c"],
            hypotheses=["blue", "oops", "a b c"],
            seed=20261127,
        )

        estimator.wer_for_draw(["A", "B"])  # 0.5
        result = estimator.confidence_interval(n_bootstrap=2000)

    Positional arguments accept both ``(speakers, references, hypotheses)``
    and JiWER's ``(references, hypotheses, speakers)`` ordering.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        data = _pick_keyword(
            kwargs,
            ("data", "records", "corpus", "speaker_data", "utterances", "items"),
            None,
        )
        speakers = _pick_keyword(
            kwargs,
            ("speakers", "speaker_ids", "speaker", "speaker_id", "spk", "labels"),
            None,
        )
        references = _pick_keyword(
            kwargs,
            ("references", "reference", "refs", "ref"),
            None,
        )
        hypotheses = _pick_keyword(
            kwargs,
            ("hypotheses", "hypothesis", "hyps", "hyp"),
            None,
        )
        reference_transform = kwargs.pop("reference_transform", wer_default)
        hypothesis_transform = kwargs.pop("hypothesis_transform", wer_default)
        self.default_seed = _pick_keyword(
            kwargs,
            ("seed", "random_seed", "random_state", "rng"),
            None,
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

        self.reference_transform = reference_transform
        self.hypothesis_transform = hypothesis_transform

        if data is not None:
            if args:
                raise TypeError("Pass either positional data or data=..., not both.")
            records = _parse_records(data)
        elif speakers is not None or references is not None or hypotheses is not None:
            if args:
                raise TypeError(
                    "Do not mix positional arguments with named data arguments."
                )
            records = _parallel_records(speakers, references, hypotheses)
        elif args:
            records = _positional_records(args)
        else:
            records = []

        self.speaker_ids: Tuple[Speaker, ...] = ()
        self.speaker_order: List[Speaker] = []
        self.speaker_errors: Dict[Speaker, int] = {}
        self.speaker_reference_words: Dict[Speaker, int] = {}
        self.speaker_utterance_counts: Dict[Speaker, int] = {}
        self.outputs: List[WordOutput] = []
        self.alignment_count = 0

        for speaker, reference, hypothesis in records:
            self.add_transcript(speaker, reference, hypothesis)

        self.speaker_ids = tuple(self.speaker_order)

    def add_transcript(
        self,
        speaker: Speaker,
        reference: Any,
        hypothesis: Any,
    ) -> "SpeakerBootstrapWER":
        """Align and cache one additional transcript pair."""
        if speaker is None:
            speaker = len(self.speaker_order)
        try:
            hash(speaker)
        except TypeError as exc:
            raise TypeError("Speaker IDs must be hashable.") from exc

        output = _align_transcript(
            reference,
            hypothesis,
            self.reference_transform,
            self.hypothesis_transform,
        )
        errors = output.substitutions + output.deletions + output.insertions
        reference_words = output.hits + output.substitutions + output.deletions

        if speaker not in self.speaker_errors:
            self.speaker_order.append(speaker)
            self.speaker_errors[speaker] = 0
            self.speaker_reference_words[speaker] = 0
            self.speaker_utterance_counts[speaker] = 0

        self.speaker_errors[speaker] += errors
        self.speaker_reference_words[speaker] += reference_words
        self.speaker_utterance_counts[speaker] += 1
        self.outputs.append(output)
        self.alignment_count += 1
        self.speaker_ids = tuple(self.speaker_order)
        return self

    @classmethod
    def from_records(
        cls,
        records: Iterable[Union[Tuple[Speaker, Any, Any], Mapping[str, Any]]],
        **kwargs: Any,
    ) -> "SpeakerBootstrapWER":
        """Create an estimator from ``(speaker, ref, hyp)`` records/dicts."""
        return cls(data=records, **kwargs)

    @classmethod
    def from_outputs(
        cls,
        outputs: Union[
            Mapping[Speaker, Union[WordOutput, Iterable[WordOutput]]],
            Iterable[Tuple[Speaker, WordOutput]],
        ],
    ) -> "SpeakerBootstrapWER":
        """
        Build cached speaker counts from already-aligned :class:`WordOutput`
        objects without performing another alignment.
        """
        obj = cls.__new__(cls)
        obj.default_seed = None
        obj.reference_transform = wer_default
        obj.hypothesis_transform = wer_default
        obj.speaker_order = []
        obj.speaker_errors = {}
        obj.speaker_reference_words = {}
        obj.speaker_utterance_counts = {}
        obj.outputs = []
        obj.alignment_count = 0

        if isinstance(outputs, Mapping):
            iterator: Iterable[Tuple[Speaker, WordOutput]] = (
                (speaker, output)
                for speaker, output_or_list in outputs.items()
                for output in (
                    output_or_list
                    if isinstance(output_or_list, Iterable)
                    and not isinstance(output_or_list, WordOutput)
                    else [output_or_list]
                )
            )
        else:
            iterator = outputs

        for speaker, output in iterator:
            if speaker not in obj.speaker_errors:
                obj.speaker_order.append(speaker)
                obj.speaker_errors[speaker] = 0
                obj.speaker_reference_words[speaker] = 0
                obj.speaker_utterance_counts[speaker] = 0

            errors = output.substitutions + output.deletions + output.insertions
            reference_words = output.hits + output.substitutions + output.deletions
            obj.speaker_errors[speaker] += errors
            obj.speaker_reference_words[speaker] += reference_words
            obj.speaker_utterance_counts[speaker] += 1
            obj.outputs.append(output)

        obj.speaker_ids = tuple(obj.speaker_order)
        return obj

    @property
    def speaker_count(self) -> int:
        return len(self.speaker_order)

    @property
    def total_errors(self) -> int:
        return sum(self.speaker_errors.values())

    @property
    def reference_words(self) -> int:
        return sum(self.speaker_reference_words.values())

    @property
    def wer(self) -> float:
        """Pooled WER over every original transcript pair."""
        return _wer_from_counts(self.total_errors, self.reference_words)

    def _resolve_speaker(self, speaker: Union[Speaker, int]) -> Speaker:
        if speaker in self.speaker_errors:
            return speaker

        # Convenience for index-based draws such as [[0, 1], [0, 0]].
        if (
            isinstance(speaker, int)
            and not isinstance(speaker, bool)
            and 0 <= speaker < len(self.speaker_order)
        ):
            return self.speaker_order[speaker]

        raise KeyError(f"Unknown speaker or speaker index: {speaker!r}")

    def counts_for_draw(self, draw: Draw) -> Tuple[int, int]:
        """
        Return pooled ``(errors, reference_words)`` for one speaker draw.

        Repeated speaker IDs are counted with their full multiplicity.
        """
        errors = 0
        reference_words = 0
        for selected in draw:
            speaker = self._resolve_speaker(selected)
            errors += self.speaker_errors[speaker]
            reference_words += self.speaker_reference_words[speaker]
        return errors, reference_words

    def wer_for_draw(self, draw: Draw) -> float:
        """Return the pooled WER for one ordered/repeated speaker draw."""
        return _wer_from_counts(*self.counts_for_draw(draw))

    # Common aliases.
    pooled_wer = wer_for_draw
    score_draw = wer_for_draw
    draw_wer = wer_for_draw

    def _normalize_draws(
        self,
        draws: Union[Sequence[Any], Iterable[Draw]],
    ) -> List[Draw]:
        draw_list = list(draws)
        if not draw_list:
            return []

        first = draw_list[0]
        if _is_scalar_speaker(first):
            # A single draw such as ["A", "B"].
            return [draw_list]

        normalized: List[Draw] = []
        for draw in draw_list:
            if _is_scalar_speaker(draw):
                normalized.append((draw,))
            else:
                normalized.append(tuple(draw))
        return normalized

    def build_draws(
        self,
        n_bootstrap: int = 1000,
        seed: Any = None,
        draws: Optional[Iterable[Draw]] = None,
    ) -> Tuple[Draw, ...]:
        """Generate speaker draws or normalize an explicit replay sequence."""
        if draws is not None:
            return tuple(self._normalize_draws(draws))

        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")

        rng = _make_rng(self.default_seed if seed is None else seed)
        return tuple(
            tuple(rng.choices(self.speaker_ids, k=self.speaker_count))
            for _ in range(int(n_bootstrap))
        )

    def bootstrap_wers(
        self,
        n_bootstrap: int = 1000,
        seed: Any = None,
        draws: Optional[Iterable[Draw]] = None,
    ) -> Tuple[float, ...]:
        """Generate or replay draws and return their pooled WER values."""
        normalized_draws = self.build_draws(n_bootstrap, seed, draws)
        return tuple(self.wer_for_draw(draw) for draw in normalized_draws)

    def confidence_interval(
        self,
        n_bootstrap: int = 1000,
        confidence_level: Optional[float] = 0.95,
        seed: Any = None,
        *,
        alpha: Optional[float] = None,
        method: str = "percentile",
        draws: Optional[Iterable[Draw]] = None,
        **kwargs: Any,
    ) -> SpeakerWERResult:
        """
        Compute a percentile (default) or basic bootstrap confidence interval.

        Pass ``draws=[...]`` to replay an exact sequence of speaker samples.
        """
        if "n_resamples" in kwargs:
            n_bootstrap = kwargs.pop("n_resamples")
        if "num_bootstrap" in kwargs:
            n_bootstrap = kwargs.pop("num_bootstrap")
        if "num_resamples" in kwargs:
            n_bootstrap = kwargs.pop("num_resamples")
        if "iterations" in kwargs:
            n_bootstrap = kwargs.pop("iterations")
        if "random_seed" in kwargs:
            seed = kwargs.pop("random_seed")
        if "random_state" in kwargs:
            seed = kwargs.pop("random_state")
        if "rng" in kwargs:
            seed = kwargs.pop("rng")
        if kwargs:
            raise TypeError(
                f"Unexpected confidence_interval argument(s): {', '.join(kwargs)}"
            )

        level = _valid_confidence_level(confidence_level, alpha)
        draw_sequence = self.build_draws(
            n_bootstrap=n_bootstrap,
            seed=seed,
            draws=draws,
        )
        sample_wers = tuple(self.wer_for_draw(draw) for draw in draw_sequence)
        if not sample_wers:
            raise ValueError("At least one bootstrap draw is required.")

        alpha_value = 1.0 - level
        lower_quantile = alpha_value / 2.0
        upper_quantile = 1.0 - alpha_value / 2.0

        if method == "percentile":
            lower = _quantile(sample_wers, lower_quantile)
            upper = _quantile(sample_wers, upper_quantile)
        elif method in ("basic", "reverse-percentile"):
            lower = 2.0 * self.wer - _quantile(sample_wers, upper_quantile)
            upper = 2.0 * self.wer - _quantile(sample_wers, lower_quantile)
        else:
            raise ValueError(
                f"Unknown confidence interval method {method!r}. "
                "Expected 'percentile' or 'basic'."
            )

        mean = sum(sample_wers) / len(sample_wers)
        if len(sample_wers) > 1:
            variance = sum((value - mean) ** 2 for value in sample_wers) / (
                len(sample_wers) - 1
            )
            standard_error = math.sqrt(variance)
        else:
            standard_error = 0.0

        return SpeakerWERResult(
            wer=self.wer,
            lower=float(lower),
            upper=float(upper),
            confidence_level=level,
            n_bootstrap=len(sample_wers),
            seed=self.default_seed if seed is None else seed,
            bootstrap_wers=sample_wers,
            bootstrap_draws=draw_sequence,
            bootstrap_mean=mean,
            standard_error=standard_error,
            method="percentile" if method == "percentile" else "basic",
            speaker_count=self.speaker_count,
            total_errors=self.total_errors,
            reference_words=self.reference_words,
        )

    # Shorter/common aliases.
    ci = confidence_interval
    bootstrap = confidence_interval
    run_bootstrap = confidence_interval
    calculate_confidence_interval = confidence_interval
    get_confidence_interval = confidence_interval


# Descriptive class aliases.
SpeakerWERBootstrap = SpeakerBootstrapWER
BootstrapWER = SpeakerBootstrapWER
WERBootstrap = SpeakerBootstrapWER
ConfidenceInterval = SpeakerWERResult
WERConfidenceInterval = SpeakerWERResult


def speaker_wer_confidence_interval(*args: Any, **kwargs: Any) -> SpeakerWERResult:
    """
    Compute WER confidence intervals by resampling whole speakers.

    The function is intentionally flexible about argument order.  Prefer the
    keyword form in new code::

        speaker_wer_confidence_interval(
            speakers=["A", "B"],
            references=["red", "a b c"],
            hypotheses=["blue", "a b c"],
            seed=42,
            n_bootstrap=2000,
        )

    It also accepts ``speaker_wer_confidence_interval(records, seed=42)`` for
    ``(speaker, reference, hypothesis)`` records.
    """
    n_bootstrap = _pick_keyword(
        kwargs,
        (
            "n_bootstrap",
            "n_resamples",
            "num_bootstrap",
            "num_resamples",
            "bootstrap_iterations",
            "iterations",
        ),
        1000,
    )
    confidence_level = _pick_keyword(
        kwargs,
        ("confidence_level", "confidence", "ci_level"),
        0.95,
    )
    alpha = kwargs.pop("alpha", None)
    seed = _pick_keyword(
        kwargs,
        ("seed", "random_seed", "random_state", "rng"),
        None,
    )
    method = kwargs.pop("method", "percentile")
    draws = _pick_keyword(
        kwargs,
        ("draws", "resamples", "sample_draws", "bootstrap_draws"),
        None,
    )

    estimator = SpeakerBootstrapWER(*args, seed=seed, **kwargs)
    return estimator.confidence_interval(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        alpha=alpha,
        seed=seed,
        method=method,
        draws=draws,
    )


def wer_confidence_interval(
    reference: Any = None,
    hypothesis: Any = None,
    speakers: Any = None,
    *args: Any,
    **kwargs: Any,
) -> SpeakerWERResult:
    """
    JiWER-order alias: ``wer_confidence_interval(reference, hypothesis, speakers)``.
    """
    if args:
        raise TypeError("Too many positional arguments.")
    return speaker_wer_confidence_interval(
        speakers=speakers,
        references=reference,
        hypotheses=hypothesis,
        **kwargs,
    )


# Common alternative names.
bootstrap_wer_confidence_interval = speaker_wer_confidence_interval
speaker_bootstrap_wer = speaker_wer_confidence_interval
speaker_wer_ci = speaker_wer_confidence_interval
bootstrap_wer = wer_confidence_interval
wer_ci = wer_confidence_interval
speaker_wer = speaker_wer_confidence_interval
bootstrap_confidence_interval = speaker_wer_confidence_interval
speaker_bootstrap_confidence_interval = speaker_wer_confidence_interval
bootstrap_speaker_wer = speaker_wer_confidence_interval
bootstrap_speaker_wer_confidence_interval = speaker_wer_confidence_interval
speaker_wer_bootstrap = speaker_wer_confidence_interval
wer_bootstrap = wer_confidence_interval
wer_bootstrap_ci = wer_confidence_interval
clustered_wer_confidence_interval = speaker_wer_confidence_interval
speaker_confidence_interval = speaker_wer_confidence_interval
confidence_interval = speaker_wer_confidence_interval
calculate_wer_confidence_interval = speaker_wer_confidence_interval
bootstrap_ci = speaker_wer_confidence_interval


def speaker_wer_bounds(*args: Any, **kwargs: Any) -> Tuple[float, float]:
    """Return only the ``(lower, upper)`` speaker-bootstrap bounds."""
    return speaker_wer_confidence_interval(*args, **kwargs).interval


def wer_bounds(*args: Any, **kwargs: Any) -> Tuple[float, float]:
    """JiWER-order alias returning only ``(lower, upper)`` bounds."""
    return wer_confidence_interval(*args, **kwargs).interval
