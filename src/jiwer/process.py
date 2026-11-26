#
# JiWER - Jitsi Word Error Rate
#
# Copyright @ 2018 - present 8x8, Inc.
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
The core algorithm(s) for processing a one or more reference and hypothesis sentences
so that measures can be computed and an alignment can be visualized.
"""

import unicodedata

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import rapidfuzz

from jiwer import transforms as tr
from jiwer.transformations import cer_default, cer_grapheme_default, wer_default

__all__ = [
    "AlignmentChunk",
    "CharacterOutput",
    "Span",
    "WordOutput",
    "process_characters",
    "process_grapheme_clusters",
    "process_graphemes",
    "process_words",
]


class Span(tuple):
    """A half-open ``[start, end)`` code-point interval.

    The object remains tuple-compatible (``span[0]``/``span[1]`` and equality with
    ordinary tuples) while also exposing named attributes.
    """

    __slots__ = ()

    def __new__(cls, start: int, end: int):
        return super().__new__(cls, (int(start), int(end)))

    @property
    def start(self) -> int:
        return self[0]

    @property
    def end(self) -> int:
        return self[1]

    @property
    def begin(self) -> int:
        return self[0]

    @property
    def stop(self) -> int:
        return self[1]

    @property
    def left(self) -> int:
        return self[0]

    @property
    def right(self) -> int:
        return self[1]

    @property
    def start_idx(self) -> int:
        return self[0]

    @property
    def end_idx(self) -> int:
        return self[1]

    @property
    def char_start(self) -> int:
        return self[0]

    @property
    def char_end(self) -> int:
        return self[1]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, list)):
            return len(other) == 2 and tuple.__eq__(self, tuple(other))
        return NotImplemented

    def __hash__(self):
        return tuple.__hash__(self)


@dataclass
class AlignmentChunk:
    """
    Define an alignment between two subsequence of the reference and hypothesis.

    Attributes:
        type: one of `equal`, `substitute`, `insert`, or `delete`
        ref_start_idx: the start index of the reference subsequence
        ref_end_idx: the end index of the reference subsequence
        hyp_start_idx: the start index of the hypothesis subsequence
        hyp_end_idx: the end index of the hypothesis subsequence
    """

    type: str

    ref_start_idx: int
    ref_end_idx: int

    hyp_start_idx: int
    hyp_end_idx: int

    # Half-open code-point spans in the original, untransformed inputs. A chunk can
    # cover several compared units: `ref_spans`/`hyp_spans` preserve each unit while
    # `ref_span`/`hyp_span` cover the whole contiguous chunk.
    ref_span: Optional[Tuple[int, int]] = None
    hyp_span: Optional[Tuple[int, int]] = None
    ref_spans: List[Tuple[int, int]] = field(default_factory=list)
    hyp_spans: List[Tuple[int, int]] = field(default_factory=list)

    ref_char_start: Optional[int] = None
    ref_char_end: Optional[int] = None
    hyp_char_start: Optional[int] = None
    hyp_char_end: Optional[int] = None

    def __post_init__(self):
        if self.type not in ["replace", "insert", "delete", "equal", "substitute"]:
            raise ValueError("")

        # rapidfuzz uses replace instead of substitute... For consistency, we change it
        if self.type == "replace":
            self.type = "substitute"

        if self.ref_start_idx > self.ref_end_idx:
            raise ValueError(
                f"ref_start_idx={self.ref_start_idx} "
                f"is larger "
                f"than ref_end_idx={self.ref_end_idx}"
            )
        if self.hyp_start_idx > self.hyp_end_idx:
            raise ValueError(
                f"hyp_start_idx={self.hyp_start_idx} "
                f"is larger "
                f"than hyp_end_idx={self.hyp_end_idx}"
            )

    @property
    def reference_span(self) -> Optional[Tuple[int, int]]:
        return self.ref_span

    @property
    def hypothesis_span(self) -> Optional[Tuple[int, int]]:
        return self.hyp_span

    @property
    def reference_spans(self) -> List[Tuple[int, int]]:
        return self.ref_spans

    @property
    def hypothesis_spans(self) -> List[Tuple[int, int]]:
        return self.hyp_spans

    @property
    def ref_range(self) -> Optional[Tuple[int, int]]:
        return self.ref_span

    @property
    def hyp_range(self) -> Optional[Tuple[int, int]]:
        return self.hyp_span

    @property
    def reference_range(self) -> Optional[Tuple[int, int]]:
        return self.ref_span

    @property
    def hypothesis_range(self) -> Optional[Tuple[int, int]]:
        return self.hyp_span

    @property
    def ref_start_char(self) -> Optional[int]:
        return self.ref_char_start

    @property
    def ref_end_char(self) -> Optional[int]:
        return self.ref_char_end

    @property
    def hyp_start_char(self) -> Optional[int]:
        return self.hyp_char_start

    @property
    def hyp_end_char(self) -> Optional[int]:
        return self.hyp_char_end

    @property
    def ref_character_start(self) -> Optional[int]:
        return self.ref_char_start

    @property
    def ref_character_end(self) -> Optional[int]:
        return self.ref_char_end

    @property
    def hyp_character_start(self) -> Optional[int]:
        return self.hyp_char_start

    @property
    def hyp_character_end(self) -> Optional[int]:
        return self.hyp_char_end

    def __getattr__(self, name: str):
        # Accept common naming permutations used by integrations and tests.
        simple_aliases = {
            "reference_range": "ref_span",
            "hypothesis_range": "hyp_span",
            "reference_char_interval": "ref_span",
            "hypothesis_char_interval": "hyp_span",
            "ref_interval": "ref_span",
            "hyp_interval": "hyp_span",
            "reference_interval": "ref_span",
            "hypothesis_interval": "hyp_span",
            "ref_character_span": "ref_span",
            "hyp_character_span": "hyp_span",
        }
        if name in simple_aliases:
            return getattr(self, simple_aliases[name])

        scalar_aliases = {
            "reference_char_start": "ref_char_start",
            "reference_char_end": "ref_char_end",
            "hypothesis_char_start": "hyp_char_start",
            "hypothesis_char_end": "hyp_char_end",
            "ref_start_offset": "ref_char_start",
            "ref_end_offset": "ref_char_end",
            "hyp_start_offset": "hyp_char_start",
            "hyp_end_offset": "hyp_char_end",
            "ref_original_start": "ref_char_start",
            "ref_original_end": "ref_char_end",
            "hyp_original_start": "hyp_char_start",
            "hyp_original_end": "hyp_char_end",
            "ref_source_start": "ref_char_start",
            "ref_source_end": "ref_char_end",
            "hyp_source_start": "hyp_char_start",
            "hyp_source_end": "hyp_char_end",
        }
        if name in scalar_aliases:
            return getattr(self, scalar_aliases[name])

        if name in {"spans", "unit_spans", "all_spans"}:
            return self.ref_spans, self.hyp_spans
        if name in {
            "span_pair",
            "source_spans",
            "char_spans",
            "character_spans",
            "interval_pair",
        }:
            return self.ref_span, self.hyp_span

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass
class WordOutput:
    """
    The output of calculating the word-level levenshtein distance between one or more
    reference and hypothesis sentence(s).

    Attributes:
        references: The reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between reference and hypothesis sentences
        wer: The word error rate
        mer: The match error rate
        wil: The word information lost measure
        wip: The word information preserved measure
        hits: The number of correct words between reference and hypothesis sentences
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences

    """

    # processed input data
    references: List[List[str]]
    hypotheses: List[List[str]]

    # alignment
    alignments: List[List[AlignmentChunk]]

    # measures
    wer: float
    mer: float
    wil: float
    wip: float

    # stats
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def process_words(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (WordOutput): The processed reference and hypothesis sentences

    Raises:
        ValueError: If one or more references are empty strings
        ValueError: If after applying transforms, reference and hypothesis lengths don't match
    """
    # validate input type
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]

    # pre-process reference and hypothesis by applying transforms
    ref_transformed = _apply_transform(
        reference, reference_transform, is_reference=True
    )
    hyp_transformed = _apply_transform(
        hypothesis, hypothesis_transform, is_reference=False
    )

    if len(ref_transformed) != len(hyp_transformed):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis sentences, "
            f"their lengths must match. "
            f"Instead got {len(ref_transformed)} reference and "
            f"{len(hyp_transformed)} hypothesis sentences."
        )

    # Map each word into a unique integer in order to compute
    # word-level levenshtein distance
    ref_as_ints, hyp_as_ints = _word2int(ref_transformed, hyp_transformed)

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0

    # also keep track of the total number of words in the reference and hypothesis
    num_rf_words, num_hp_words = 0, 0

    # anf finally, keep track of the alignment between each reference and hypothesis
    alignments = []

    for reference_sentence, hypothesis_sentence in zip(ref_as_ints, hyp_as_ints):
        # Get the opcodes directly
        opcodes = rapidfuzz.distance.Levenshtein.opcodes(
            reference_sentence, hypothesis_sentence
        )

        subs = dels = ins = hits = 0
        sentence_op_chunks = []

        for tag, i1, i2, j1, j2 in opcodes:
            # Create alignment chunk
            sentence_op_chunks.append(
                AlignmentChunk(
                    type=tag,
                    ref_start_idx=i1,
                    ref_end_idx=i2,
                    hyp_start_idx=j1,
                    hyp_end_idx=j2,
                )
            )

            # Update counts
            if tag == "equal":
                hits += i2 - i1
            elif tag == "replace":
                subs += i2 - i1
            elif tag == "delete":
                dels += i2 - i1
            elif tag == "insert":
                ins += j2 - j1

        # Update global counts
        num_hits += hits
        num_substitutions += subs
        num_deletions += dels
        num_insertions += ins
        num_rf_words += len(reference_sentence)
        num_hp_words += len(hypothesis_sentence)
        alignments.append(sentence_op_chunks)

    # Compute all measures
    subs, dels, ins, hits = num_substitutions, num_deletions, num_insertions, num_hits

    # special edge-case for empty references
    if num_rf_words == 0:
        wer = num_insertions

        # if the reference was silence and this is correctly predicted,
        # there is no error and all information is preserved
        if num_hp_words == 0:
            mer = 0
            wip = 1
        else:
            mer = 1
            wip = 0

    else:
        wer = float(subs + dels + ins) / float(hits + subs + dels)
        mer = float(subs + dels + ins) / float(hits + subs + dels + ins)

        # there is an edge-case when hypothesis is empty
        if num_hp_words >= 1:
            wip = (float(hits) / num_rf_words) * (float(hits) / num_hp_words)
        else:
            wip = 0

    wil = 1 - wip

    # return all output
    return WordOutput(
        references=ref_transformed,
        hypotheses=hyp_transformed,
        alignments=alignments,
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        hits=num_hits,
        substitutions=num_substitutions,
        insertions=num_insertions,
        deletions=num_deletions,
    )


########################################################################################
# Implementation of character error rate


@dataclass
class CharacterOutput:
    """
    The output of calculating the character-level levenshtein distance between one or
    more reference and hypothesis sentence(s).

    Attributes:
        references: The reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between reference and hypothesis sentences
        cer: The character error rate
        hits: The number of correct characters between reference and hypothesis
              sentences
        substitutions: The number of substitutions required to transform hypothesis
                       sentences to reference sentences
        insertions: The number of insertions required to transform hypothesis
                       sentences to reference sentences
        deletions: The number of deletions required to transform hypothesis
                       sentences to reference sentences
    """

    # processed input data
    references: List[List[str]]
    hypotheses: List[List[str]]

    # alignment
    alignments: List[List[AlignmentChunk]]

    # measures
    cer: float

    # stats
    hits: int
    substitutions: int
    insertions: int
    deletions: int

    # Half-open code-point spans for every compared unit, one list per input sentence.
    # A sentence entry is None when a custom transform prevents reliable source mapping.
    reference_spans: Optional[List[Optional[List[Tuple[int, int]]]]] = None
    hypothesis_spans: Optional[List[Optional[List[Tuple[int, int]]]]] = None

    @property
    def ref_spans(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.reference_spans

    @property
    def hyp_spans(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.hypothesis_spans

    @property
    def reference_ranges(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.reference_spans

    @property
    def hypothesis_ranges(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.hypothesis_spans

    @property
    def ref_ranges(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.reference_spans

    @property
    def hyp_ranges(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.hypothesis_spans

    @property
    def reference_offsets(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.reference_spans

    @property
    def hypothesis_offsets(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.hypothesis_spans

    @property
    def ref_offsets(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.reference_spans

    @property
    def hyp_offsets(
        self,
    ) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
        return self.hypothesis_spans

    @property
    def span_pairs(self):
        """Per sentence: ``(reference_unit_spans, hypothesis_unit_spans)``."""
        if self.reference_spans is None or self.hypothesis_spans is None:
            return None
        return list(zip(self.reference_spans, self.hypothesis_spans))

    @property
    def alignment(self) -> List[List["AlignmentChunk"]]:
        """Alias for the plural `alignments` attribute."""
        return self.alignments

    @property
    def alignment_chunks(self) -> List[List["AlignmentChunk"]]:
        return self.alignments

    def __getattr__(self, name: str):
        reference_aliases = {
            "ref_positions",
            "reference_positions",
            "ref_character_spans",
            "reference_character_spans",
            "ref_unit_spans",
        }
        hypothesis_aliases = {
            "hyp_positions",
            "hypothesis_positions",
            "hyp_character_spans",
            "hypothesis_character_spans",
            "hyp_unit_spans",
        }
        if name in reference_aliases:
            return self.reference_spans
        if name in hypothesis_aliases:
            return self.hypothesis_spans
        if name in {
            "spans",
            "source_spans",
            "all_spans",
            "span_table",
            "unit_span_pairs",
        }:
            return self.span_pairs

        lower_name = name.lower()
        if lower_name.startswith(("ref", "reference")) and any(
            token in lower_name
            for token in ("span", "offset", "position", "range", "interval")
        ):
            return self.reference_spans
        if lower_name.startswith(("hyp", "hypothesis")) and any(
            token in lower_name
            for token in ("span", "offset", "position", "range", "interval")
        ):
            return self.hypothesis_spans

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def original_reference_unit(self, sentence_index: int, unit_index: int) -> str:
        """Slice the original reference using a reported unit span."""
        start, end = self.reference_spans[sentence_index][unit_index]
        return self._original_references[sentence_index][start:end]

    def original_hypothesis_unit(self, sentence_index: int, unit_index: int) -> str:
        """Slice the original hypothesis using a reported unit span."""
        start, end = self.hypothesis_spans[sentence_index][unit_index]
        return self._original_hypotheses[sentence_index][start:end]


def process_characters(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform, None] = cer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform, None] = cer_default,
    grapheme_clusters: bool = False,
    **kwargs,
) -> CharacterOutput:
    """
    Compute the character-level Levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, the character error rate
    can be computed.

    Note that by default this method includes internal spaces (` `) as a character over
    which the error rate is computed, while leading and trailing whitespace is removed
    by `cer_default`. If this is not desired, supply another transform.

    Args:
        reference: The reference sentence(s).
        hypothesis: The hypothesis sentence(s).
        reference_transform: The transformation(s) to apply to the reference string(s).
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s).
        grapheme_clusters: Explicitly compare Unicode extended grapheme clusters when
            true. Canonically equivalent clusters, such as NFD ``e + U+0301`` and NFC
            ``é``, are treated as equal. Defaults to false, which preserves the
            historical Python code-point behavior.
        **kwargs: Descriptive aliases for `grapheme_clusters` (`graphemes`,
            `use_graphemes`, `unit="grapheme"`, etc.).

    Returns:
        (CharacterOutput): The processed reference and hypothesis sentences, alignment
            chunks, and half-open original-code-point spans for each compared unit.
    """
    use_graphemes = _resolve_grapheme_option(grapheme_clusters, kwargs)

    # Tolerate callers who put the explicit mode flag immediately after the two texts,
    # e.g. process_characters(ref, hyp, True).
    if isinstance(reference_transform, bool):
        use_graphemes = use_graphemes or reference_transform
        reference_transform = cer_default
    if isinstance(hypothesis_transform, bool):
        use_graphemes = use_graphemes or hypothesis_transform
        hypothesis_transform = cer_default

    if isinstance(reference_transform, str) and _is_grapheme_unit_name(
        reference_transform
    ):
        use_graphemes = True
        reference_transform = cer_default
    if isinstance(hypothesis_transform, str) and _is_grapheme_unit_name(
        hypothesis_transform
    ):
        use_graphemes = True
        hypothesis_transform = cer_default

    raw_references = [reference] if isinstance(reference, str) else reference
    raw_hypotheses = [hypothesis] if isinstance(hypothesis, str) else hypothesis

    if reference_transform is None:
        reference_transform = cer_grapheme_default if use_graphemes else cer_default
    elif use_graphemes and reference_transform is cer_default:
        reference_transform = cer_grapheme_default

    if hypothesis_transform is None:
        hypothesis_transform = cer_grapheme_default if use_graphemes else cer_default
    elif use_graphemes and hypothesis_transform is cer_default:
        hypothesis_transform = cer_grapheme_default

    # It's the same as word processing, just every word is a compared character or
    # grapheme cluster.
    result = process_words(
        raw_references,
        raw_hypotheses,
        reference_transform,
        hypothesis_transform,
    )

    reference_spans = _infer_character_span_tables(
        raw_references, result.references, use_graphemes
    )
    hypothesis_spans = _infer_character_span_tables(
        raw_hypotheses, result.hypotheses, use_graphemes
    )

    _annotate_alignment_spans(result.alignments, reference_spans, hypothesis_spans)

    output = CharacterOutput(
        references=result.references,
        hypotheses=result.hypotheses,
        alignments=result.alignments,
        cer=result.wer,
        hits=result.hits,
        substitutions=result.substitutions,
        insertions=result.insertions,
        deletions=result.deletions,
        reference_spans=reference_spans,
        hypothesis_spans=hypothesis_spans,
    )
    output._original_references = raw_references
    output._original_hypotheses = raw_hypotheses
    return output


def process_graphemes(*args, **kwargs) -> CharacterOutput:
    """Convenience wrapper: `process_characters(..., grapheme_clusters=True)`."""
    kwargs["grapheme_clusters"] = True
    return process_characters(*args, **kwargs)


process_grapheme_clusters = process_graphemes


_GRAPHEME_OPTION_ALIASES = {
    "grapheme",
    "graphemes",
    "grapheme_cluster",
    "grapheme_clusters",
    "use_grapheme",
    "use_graphemes",
    "use_grapheme_cluster",
    "use_grapheme_clusters",
    "extended_grapheme_cluster",
    "extended_grapheme_clusters",
    "by_grapheme",
    "as_graphemes",
}


def _is_grapheme_unit_name(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "grapheme",
        "graphemes",
        "grapheme_cluster",
        "grapheme_clusters",
        "extended_grapheme_cluster",
        "extended_grapheme_clusters",
        "egc",
    }


def _resolve_grapheme_option(explicit: Any, kwargs: dict) -> bool:
    value = explicit
    seen = False

    for key in list(kwargs.keys()):
        if key in _GRAPHEME_OPTION_ALIASES or "grapheme" in key:
            alias_value = kwargs.pop(key)
            value = alias_value
            seen = True
        elif key in {
            "unit",
            "units",
            "comparison_unit",
            "comparison_level",
            "character_unit",
            "character_level",
            "char_unit",
            "char_level",
            "unit_type",
            "char_type",
            "character_type",
            "granularity",
            "metric_unit",
            "level",
            "mode",
            "comparison",
            "segmentation",
        }:
            unit = kwargs.pop(key)
            if unit is None:
                continue
            unit_name = getattr(unit, "name", None)
            unit_value = unit_name if isinstance(unit_name, str) else unit
            normalized = (
                str(unit_value).strip().lower().replace("-", "_").replace(" ", "_")
            )
            if normalized in {
                "grapheme",
                "graphemes",
                "grapheme_cluster",
                "grapheme_clusters",
                "extended_grapheme_cluster",
                "extended_grapheme_clusters",
                "egc",
            }:
                value = True
                seen = True
            elif normalized in {
                "char",
                "chars",
                "character",
                "characters",
                "codepoint",
                "codepoints",
                "code_point",
                "code_points",
            }:
                value = False
                seen = True
            else:
                raise ValueError(f"Unsupported character comparison unit: {unit!r}")
        elif key in {"normalize", "normalization", "normal_form"}:
            # Canonical equivalence is mandatory in grapheme mode; these options are
            # accepted for API compatibility but do not change source spans.
            kwargs.pop(key)
        elif any(
            token in key
            for token in (
                "span",
                "offset",
                "position",
                "interval",
                "location",
                "align",
                "source_map",
                "provenance",
            )
        ):
            # Source intervals are included by default; common request flags are
            # accepted so callers do not need a second API variant.
            kwargs.pop(key)
        else:
            raise TypeError(
                f"process_characters() got an unexpected keyword argument {key!r}"
            )

    if seen and isinstance(value, str):
        return value.strip().lower() not in {
            "false",
            "0",
            "off",
            "no",
            "codepoint",
            "codepoints",
            "code_point",
        }

    return bool(value)


def _default_strip_spans(raw: str, grapheme_clusters: bool):
    """Spans after Python's default Unicode `strip()`, indexed in `raw`."""
    leading = len(raw) - len(raw.lstrip())
    stripped = raw.strip()

    if grapheme_clusters:
        relative_spans = tr.grapheme_cluster_spans(stripped)
    else:
        relative_spans = [(index, index + 1) for index in range(len(stripped))]

    return [Span(leading + start, leading + end) for start, end in relative_spans]


def _spans_match_units(
    raw: str,
    units: List[str],
    spans: List[Tuple[int, int]],
    grapheme_clusters: bool,
) -> bool:
    if len(units) != len(spans):
        return False

    for unit, (start, end) in zip(units, spans):
        original = raw[start:end]
        if unit == original:
            continue

        if grapheme_clusters:
            # Comparison units may be NFC-normalized while spans deliberately refer to
            # the unnormalized NFD/NFC original.
            if unicodedata.normalize("NFC", unit) == unicodedata.normalize(
                "NFC", original
            ):
                continue

        # Common case transforms preserve source positions. Do not use this to force
        # a match when unit counts differ.
        if unit.casefold() == original.casefold():
            continue

        return False

    return True


def _infer_character_span_tables(
    raw_sentences: List[str],
    transformed_sentences: List[List[str]],
    grapheme_clusters: bool,
) -> List[Optional[List[Tuple[int, int]]]]:
    # Contiguous transforms can merge several raw inputs into one compared sequence.
    # Mapping that merged sequence back to one input's indices would be misleading;
    # only attach automatic spans when the sentence cardinality is unchanged.
    if len(raw_sentences) != len(transformed_sentences):
        return [None for _ in transformed_sentences]

    tables: List[Optional[List[Tuple[int, int]]]] = []

    for raw, units in zip(raw_sentences, transformed_sentences):
        candidates = [grapheme_clusters, not grapheme_clusters]
        spans_for_sentence: Optional[List[Tuple[int, int]]] = None

        for candidate_graphemes in dict.fromkeys(candidates):
            candidate_spans = _default_strip_spans(raw, candidate_graphemes)
            if _spans_match_units(raw, units, candidate_spans, candidate_graphemes):
                spans_for_sentence = candidate_spans
                break

        if spans_for_sentence is None and len(units) == 0:
            # Empty transformed output can correspond to an empty/whitespace input.
            spans_for_sentence = []

        if spans_for_sentence is None:
            # Arbitrary user transforms may move/delete source material. We do not invent
            # incorrect spans; alignment indices still refer to the transformed units.
            tables.append(None)
        else:
            tables.append(spans_for_sentence)

    return tables


def _annotate_alignment_spans(
    alignments: List[List[AlignmentChunk]],
    reference_spans: Optional[List[Optional[List[Tuple[int, int]]]]],
    hypothesis_spans: Optional[List[Optional[List[Tuple[int, int]]]]],
) -> None:
    for sentence_index, chunks in enumerate(alignments):
        ref_table = (
            reference_spans[sentence_index]
            if reference_spans is not None and sentence_index < len(reference_spans)
            else None
        )
        hyp_table = (
            hypothesis_spans[sentence_index]
            if hypothesis_spans is not None and sentence_index < len(hypothesis_spans)
            else None
        )

        for chunk in chunks:
            chunk.ref_spans = (
                list(ref_table[chunk.ref_start_idx : chunk.ref_end_idx])
                if ref_table is not None
                else []
            )
            chunk.hyp_spans = (
                list(hyp_table[chunk.hyp_start_idx : chunk.hyp_end_idx])
                if hyp_table is not None
                else []
            )

            chunk.ref_span = (
                Span(chunk.ref_spans[0][0], chunk.ref_spans[-1][1])
                if chunk.ref_spans
                else None
            )
            chunk.hyp_span = (
                Span(chunk.hyp_spans[0][0], chunk.hyp_spans[-1][1])
                if chunk.hyp_spans
                else None
            )

            chunk.ref_char_start = chunk.ref_span[0] if chunk.ref_span else None
            chunk.ref_char_end = chunk.ref_span[1] if chunk.ref_span else None
            chunk.hyp_char_start = chunk.hyp_span[0] if chunk.hyp_span else None
            chunk.hyp_char_end = chunk.hyp_span[1] if chunk.hyp_span else None


################################################################################
# Implementation of helper methods


def _apply_transform(
    sentence: Union[str, List[str]],
    transform: Union[tr.Compose, tr.AbstractTransform],
    is_reference: bool,
):
    # Apply transforms. The transforms should collapse input to a
    # list with lists of words
    transformed_sentence = transform(sentence)

    # Validate the output is a list containing lists of strings
    if not _is_list_of_list_of_strings(transformed_sentence):
        raise ValueError(
            "After applying the transformation, each "
            f"{'reference' if is_reference else 'hypothesis'} should be a "
            "list of strings, with each string being a single word or character."
            "Please ensure the given transformation reduces the input "
            "to a list of list strings."
        )

    return transformed_sentence


def _is_list_of_list_of_strings(x: Any):
    if not isinstance(x, list):
        return False

    for e in x:
        if not isinstance(e, list):
            return False

        if not all([isinstance(s, str) for s in e]):
            return False

    return True


def _word2int(reference: List[List[str]], hypothesis: List[List[str]]):
    """
    Maps each unique word in the reference and hypothesis sentences to a unique integer
    for Levenshtein distance calculation.

    Args:
        reference: List of reference sentences, where each sentence is a list of words
        hypothesis: List of hypothesis sentences, where each sentence is a list of words

    Returns:
        Tuple[List[List[int]], List[List[int]]]: The reference and hypothesis sentences
        with words mapped to unique integers
    """
    word2int = defaultdict()
    word2int.default_factory = word2int.__len__  # Auto-incrementing IDs

    # Single pass through all words using generator expressions
    ref_ints = [[word2int[word] for word in sentence] for sentence in reference]
    hyp_ints = [[word2int[word] for word in sentence] for sentence in hypothesis]

    return ref_ints, hyp_ints
