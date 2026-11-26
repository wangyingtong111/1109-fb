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
Recording-region alignment with line-level provenance.

A region is a continuous span such as one recording or fixed cut point.  Lines
inside a region are concatenated before word alignment, while different regions
are aligned independently.  This prevents a word deleted at the end of one
region from matching an identical word inserted at the beginning of another
region.
"""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from jiwer import transforms as tr
from jiwer.process import (
    AlignmentChunk,
    WordOutput,
    WordSource,
    _apply_transform,
    build_word_output,
)
from jiwer.transformations import wer_default

__all__ = [
    "Region",
    "Segment",
    "RegionEdit",
    "RegionResult",
    "RegionWordOutput",
    "SegmentResult",
    "SegmentWordOutput",
    "WordAttribution",
    "WordSource",
    "align_regions",
    "align_segments",
    "attribute_region_edits",
    "evaluate_regions",
    "process_contiguous_words",
    "process_continuous_words",
    "process_region_words",
    "process_segment_words",
    "process_words_with_regions",
    "process_words_with_segments",
    "process_regions",
    "process_segments",
    "process_words_by_region",
    "process_words_by_regions",
]


class LineRecord(NamedTuple):
    """A normalized input line."""

    index: int
    line_id: Any
    text: str


@dataclass
class Region:
    """
    A named continuous alignment region.

    ``reference`` and ``hypothesis`` are lists of lines.  A line may either be a
    string or a ``(line_id, text)`` tuple.  String lines receive their global
    zero-based input index as their source line identifier.
    """

    id: Any
    reference: Any
    hypothesis: Any


# A descriptive alias for callers who prefer "segment" terminology.
Segment = Region


@dataclass(frozen=True)
class RegionEdit:
    """
    A non-equal alignment interval in one region.

    ``ref_positions`` and ``hyp_positions`` contain one
    :class:`~jiwer.process.WordSource` for every transformed word in the chunk,
    even when rapidfuzz reports those words as one merged interval.
    """

    region: Any
    type: str
    ref_positions: Tuple[WordSource, ...] = ()
    hyp_positions: Tuple[WordSource, ...] = ()
    ref_start_idx: int = 0
    ref_end_idx: int = 0
    hyp_start_idx: int = 0
    hyp_end_idx: int = 0

    @property
    def kind(self) -> str:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def region_id(self) -> Any:
        """Alias for :attr:`region`."""
        return self.region

    @property
    def ref_sources(self) -> Tuple[WordSource, ...]:
        return self.ref_positions

    @property
    def hyp_sources(self) -> Tuple[WordSource, ...]:
        return self.hyp_positions

    @property
    def ref_origins(self) -> Tuple[WordSource, ...]:
        return self.ref_positions

    @property
    def hyp_origins(self) -> Tuple[WordSource, ...]:
        return self.hyp_positions

    @property
    def reference_positions(self) -> Tuple[WordSource, ...]:
        return self.ref_positions

    @property
    def hypothesis_positions(self) -> Tuple[WordSource, ...]:
        return self.hyp_positions

    @property
    def ref_lines(self) -> Tuple[Any, ...]:
        """Unique source line IDs touched on the reference side, in order."""
        return _unique(position.line_id for position in self.ref_positions)

    @property
    def hyp_lines(self) -> Tuple[Any, ...]:
        """Unique source line IDs touched on the hypothesis side, in order."""
        return _unique(position.line_id for position in self.hyp_positions)

    @property
    def reference_lines(self) -> Tuple[Any, ...]:
        return self.ref_lines

    @property
    def hypothesis_lines(self) -> Tuple[Any, ...]:
        return self.hyp_lines

    @property
    def ref_line(self) -> Any:
        """The only reference line, when this edit touches exactly one line."""
        lines = self.ref_lines
        if len(lines) != 1:
            return None
        return lines[0]

    @property
    def hyp_line(self) -> Any:
        """The only hypothesis line, when this edit touches exactly one line."""
        lines = self.hyp_lines
        if len(lines) != 1:
            return None
        return lines[0]

    @property
    def ref_word_indices(self) -> Tuple[int, ...]:
        return tuple(position.word_index for position in self.ref_positions)

    @property
    def hyp_word_indices(self) -> Tuple[int, ...]:
        return tuple(position.word_index for position in self.hyp_positions)

    def as_tuple(self) -> Tuple[Any, str, Tuple[Tuple[Any, int], ...], Tuple[Tuple[Any, int], ...]]:
        """
        Return the compact ``(region, type, ref_coordinates, hyp_coordinates)``
        representation.  Word sources are ordinary ``(line_id, word_index)``
        tuples and therefore compare equal to plain expected tuples.
        """
        return (
            self.region,
            self.type,
            tuple(tuple(position) for position in self.ref_positions),
            tuple(tuple(position) for position in self.hyp_positions),
        )


@dataclass(frozen=True)
class WordAttribution:
    """An individual edited word and its source-line coordinate."""

    region: Any
    type: str
    side: str
    position: WordSource
    counterpart: Optional[WordSource] = None

    @property
    def source(self) -> WordSource:
        return self.position

    def as_tuple(self) -> Tuple[Any, str, str, Tuple[Any, int], Optional[Tuple[Any, int]]]:
        counterpart = None if self.counterpart is None else tuple(self.counterpart)
        return (self.region, self.type, self.side, tuple(self.position), counterpart)


@dataclass
class RegionResult:
    """The independent alignment result for one region."""

    region: Any
    word_output: WordOutput
    alignment: List[AlignmentChunk]
    edits: Tuple[RegionEdit, ...]
    reference_sources: Tuple[WordSource, ...]
    hypothesis_sources: Tuple[WordSource, ...]

    @property
    def id(self) -> Any:
        return self.region

    @property
    def references(self) -> List[List[str]]:
        return self.word_output.references

    @property
    def hypotheses(self) -> List[List[str]]:
        return self.word_output.hypotheses

    @property
    def alignments(self) -> List[List[AlignmentChunk]]:
        return self.word_output.alignments

    @property
    def hits(self) -> int:
        return self.word_output.hits

    @property
    def substitutions(self) -> int:
        return self.word_output.substitutions

    @property
    def insertions(self) -> int:
        return self.word_output.insertions

    @property
    def deletions(self) -> int:
        return self.word_output.deletions

    @property
    def wer(self) -> float:
        return self.word_output.wer

    @property
    def mer(self) -> float:
        return self.word_output.mer

    @property
    def wil(self) -> float:
        return self.word_output.wil

    @property
    def wip(self) -> float:
        return self.word_output.wip

    @property
    def details(self) -> Tuple[Tuple[Any, str, Tuple, Tuple], ...]:
        """Compact plain-tuple edit details, excluding equal chunks."""
        return tuple(edit.as_tuple() for edit in self.edits)

    @property
    def word_edits(self) -> Tuple[WordAttribution, ...]:
        return tuple(_iter_word_attributions(self.edits))


@dataclass
class RegionWordOutput(WordOutput):
    """Aggregate counts and per-region results for region-aware alignment."""

    region_results: List[RegionResult] = field(default_factory=list)

    @property
    def regions(self) -> List[RegionResult]:
        """Per-region results."""
        return self.region_results

    @property
    def segments(self) -> List[RegionResult]:
        """Alias for :attr:`regions`."""
        return self.region_results

    @property
    def region_ids(self) -> Tuple[Any, ...]:
        return tuple(result.region for result in self.region_results)

    @property
    def edits(self) -> Tuple[RegionEdit, ...]:
        return tuple(edit for result in self.region_results for edit in result.edits)

    @property
    def errors(self) -> Tuple[RegionEdit, ...]:
        """Alias for :attr:`edits`."""
        return self.edits

    @property
    def details(self) -> Tuple[Tuple[Any, str, Tuple, Tuple], ...]:
        """Compact plain-tuple edit details for all regions."""
        return tuple(edit.as_tuple() for edit in self.edits)

    @property
    def word_edits(self) -> Tuple[WordAttribution, ...]:
        return tuple(attribution for result in self.region_results for attribution in result.word_edits)

    def as_case_tuple(self) -> Tuple[Tuple[int, int], Tuple]:
        """
        Return ``((deletions, insertions), details)``.

        This is a convenient assertion/reporting shape for the cross-region
        deletion/insertion scenarios addressed by this module.
        """
        return (self.deletions, self.insertions), self.details

    def __iter__(self):
        return iter(self.region_results)

    def __len__(self) -> int:
        return len(self.region_results)

    def __getitem__(self, item):
        return self.region_results[item]


def process_region_words(
    reference: Any = None,
    hypothesis: Any = None,
    regions: Any = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    *,
    reference_regions: Optional[Sequence[Any]] = None,
    hypothesis_regions: Optional[Sequence[Any]] = None,
    region_ids: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> RegionWordOutput:
    """
    Align words continuously inside each recording region while isolating regions.

    The easiest input is a sequence of ``(region_id, reference_lines,
    hypothesis_lines)`` triples::

        output = jiwer.process_region_words([
            ("A", [("a", "go now")], [("a1", "go")]),
            ("B", [("b", "next")], [("b1", "now next")]),
        ])

    Plain string lines are also accepted and receive integer line IDs.

    Alternative forms accepted for convenience include:

    * ``Region(id, reference_lines, hypothesis_lines)`` objects or dictionaries;
    * ``{region_id: (reference_lines, hypothesis_lines)}`` mappings;
    * nested ``reference=[region_a_ref_lines, ...]`` and matching
      ``hypothesis=[region_a_hyp_lines, ...]``;
    * flat ``reference``/``hypothesis`` lists with parallel
      ``reference_regions`` and ``hypothesis_regions`` labels.
    """
    # Accept common alias keyword names.
    if reference_regions is None:
        reference_regions = _first_kwarg(
            kwargs,
            "ref_regions",
            "reference_region_ids",
            "ref_region_ids",
            "reference_segments",
            "ref_segments",
        )
    if hypothesis_regions is None:
        hypothesis_regions = _first_kwarg(
            kwargs,
            "hyp_regions",
            "hypothesis_region_ids",
            "hyp_region_ids",
            "hypothesis_segments",
            "hyp_segments",
        )
    if regions is None:
        regions = _first_kwarg(kwargs, "segments", "segment", "region")
    if region_ids is None:
        region_ids = kwargs.get("segment_ids")

    # Convenience form: process_region_words([(region_id, refs, hyps), ...]).
    if (
        regions is None
        and hypothesis is None
        and (
            _looks_like_region_groups(reference)
            or isinstance(reference, MappingABC)
        )
    ):
        regions = reference
        reference = None

    _reject_unknown_kwargs(kwargs)

    raw_regions = _coerce_raw_regions(
        regions=regions,
        reference=reference,
        hypothesis=hypothesis,
        reference_regions=reference_regions,
        hypothesis_regions=hypothesis_regions,
        region_ids=region_ids,
    )

    ref_index = 0
    hyp_index = 0
    results: List[RegionResult] = []
    ref_streams: List[List[str]] = []
    hyp_streams: List[List[str]] = []

    for region_id, raw_reference, raw_hypothesis in raw_regions:
        ref_records, ref_index = _coerce_lines(raw_reference, ref_index)
        hyp_records, hyp_index = _coerce_lines(raw_hypothesis, hyp_index)

        ref_words, ref_sources = _transform_lines(
            ref_records, reference_transform, is_reference=True
        )
        hyp_words, hyp_sources = _transform_lines(
            hyp_records, hypothesis_transform, is_reference=False
        )

        word_output = build_word_output([ref_words], [hyp_words])
        chunks = word_output.alignments[0]

        edits: List[RegionEdit] = []
        for chunk in chunks:
            ref_positions = tuple(
                ref_sources[index]
                for index in range(chunk.ref_start_idx, chunk.ref_end_idx)
            )
            hyp_positions = tuple(
                hyp_sources[index]
                for index in range(chunk.hyp_start_idx, chunk.hyp_end_idx)
            )

            chunk.region = region_id
            chunk.ref_positions = ref_positions
            chunk.hyp_positions = hyp_positions

            if chunk.type != "equal":
                edits.append(
                    RegionEdit(
                        region=region_id,
                        type=chunk.type,
                        ref_positions=ref_positions,
                        hyp_positions=hyp_positions,
                        ref_start_idx=chunk.ref_start_idx,
                        ref_end_idx=chunk.ref_end_idx,
                        hyp_start_idx=chunk.hyp_start_idx,
                        hyp_end_idx=chunk.hyp_end_idx,
                    )
                )

        results.append(
            RegionResult(
                region=region_id,
                word_output=word_output,
                alignment=chunks,
                edits=tuple(edits),
                reference_sources=ref_sources,
                hypothesis_sources=hyp_sources,
            )
        )
        ref_streams.append(ref_words)
        hyp_streams.append(hyp_words)

    # Reuse the exact aggregate measure/count implementation, treating each
    # region as one independently aligned continuous stream.
    aggregate = build_word_output(ref_streams, hyp_streams)

    return RegionWordOutput(
        references=ref_streams,
        hypotheses=hyp_streams,
        alignments=[result.alignment for result in results],
        wer=aggregate.wer,
        mer=aggregate.mer,
        wil=aggregate.wil,
        wip=aggregate.wip,
        hits=aggregate.hits,
        substitutions=aggregate.substitutions,
        insertions=aggregate.insertions,
        deletions=aggregate.deletions,
        region_results=results,
    )


# Descriptive function aliases.
SegmentResult = RegionResult
SegmentWordOutput = RegionWordOutput
process_words_with_regions = process_region_words
process_words_with_segments = process_region_words
process_segment_words = process_region_words
process_segments = process_region_words
align_segments = process_region_words
process_contiguous_words = process_region_words
process_continuous_words = process_region_words
process_words_by_regions = process_region_words
process_words_by_region = process_region_words
process_regions = process_region_words
align_regions = process_region_words


def attribute_region_edits(
    regions: Any = None,
    reference: Any = None,
    hypothesis: Any = None,
    **kwargs: Any,
) -> Tuple[RegionEdit, ...]:
    """Align regions and return only their attributed non-equal edits."""
    return process_region_words(
        regions=regions,
        reference=reference,
        hypothesis=hypothesis,
        **kwargs,
    ).edits


def evaluate_regions(
    regions: Any = None,
    reference: Any = None,
    hypothesis: Any = None,
    **kwargs: Any,
) -> Tuple[Tuple[int, int], Tuple]:
    """
    Return the compact ``((deletions, insertions), details)`` acceptance shape.

    ``details`` contains ``(region, edit_type, ref_coordinates,
    hyp_coordinates)`` tuples and can be compared to plain expected tuples.
    """
    return process_region_words(
        regions=regions,
        reference=reference,
        hypothesis=hypothesis,
        **kwargs,
    ).as_case_tuple()


def maybe_process_words_with_regions(
    reference: Any,
    hypothesis: Any,
    reference_transform: Union[tr.Compose, tr.AbstractTransform],
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform],
    regions: Any,
    **kwargs: Any,
) -> Optional[RegionWordOutput]:
    """
    Delegate :func:`jiwer.process_words` to region processing when requested.

    Returns ``None`` for the historical positional sentence/list mode.
    """
    explicit_labels = any(
        kwargs.get(name) is not None
        for name in (
            "reference_regions",
            "hypothesis_regions",
            "ref_regions",
            "hyp_regions",
            "reference_region_ids",
            "hypothesis_region_ids",
            "reference_segments",
            "hypothesis_segments",
            "region_ids",
            "segment_ids",
        )
    )
    segment_arg = _first_kwarg(kwargs, "segments", "segment", "region")
    by_region = bool(kwargs.pop("by_region", False))
    continuous = any(
        bool(kwargs.pop(name, False))
        for name in (
            "continuous",
            "contiguous",
            "merge_lines",
            "merge_sentences",
        )
    )

    # process_words(reference=groups) as a one-argument convenience form.
    if (
        regions is None
        and segment_arg is None
        and not explicit_labels
        and not by_region
        and not continuous
        and hypothesis is None
        and (
            _looks_like_region_groups(reference)
            or isinstance(reference, MappingABC)
        )
    ):
        regions = reference
        reference = None

    nested_without_marker = (
        regions is None
        and segment_arg is None
        and not explicit_labels
        and not continuous
        and _is_nested_region_pair(reference, hypothesis)
    )

    if (
        regions is None
        and segment_arg is None
        and not explicit_labels
        and not by_region
        and not continuous
        and not nested_without_marker
    ):
        # Preserve the old **kwargs behaviour closely: process_words previously
        # did not accept arbitrary keywords beyond its explicit signature.
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"process_words() got an unexpected keyword argument {unexpected!r}"
            )
        return None

    if regions is None and segment_arg is not None:
        regions = segment_arg

    return process_region_words(
        regions=regions,
        reference=reference,
        hypothesis=hypothesis,
        reference_transform=reference_transform,
        hypothesis_transform=hypothesis_transform,
        **kwargs,
    )


################################################################################
# Input normalization


def _coerce_raw_regions(
    regions: Any,
    reference: Any,
    hypothesis: Any,
    reference_regions: Optional[Sequence[Any]],
    hypothesis_regions: Optional[Sequence[Any]],
    region_ids: Optional[Sequence[Any]],
) -> List[Tuple[Any, Any, Any]]:
    if regions is not None:
        # A single scalar region label puts every supplied line in that region.
        if (
            reference is not None
            and hypothesis is not None
            and _is_scalar_id(regions)
        ):
            ref_count = len(_as_line_list(reference))
            hyp_count = len(_as_line_list(hypothesis))
            return _groups_from_labels(
                reference,
                hypothesis,
                [regions] * ref_count,
                [regions] * hyp_count,
            )

        # A flat label sequence parallel to both line lists can be used for
        # both sides when their line counts agree.
        if (
            reference is not None
            and hypothesis is not None
            and _is_label_sequence(regions)
        ):
            ref_count = len(_as_line_list(reference))
            hyp_count = len(_as_line_list(hypothesis))
            labels = list(regions)
            if len(labels) == ref_count and len(labels) == hyp_count:
                return _groups_from_labels(reference, hypothesis, labels, labels)

        # A pair of flat label sequences can be passed as ``regions=(ref_labels,
        # hyp_labels)`` together with reference/hypothesis lines.
        if (
            reference is not None
            and hypothesis is not None
            and isinstance(regions, (tuple, list))
            and len(regions) == 2
            and _is_label_sequence(regions[0])
            and _is_label_sequence(regions[1])
        ):
            return _groups_from_labels(
                reference, hypothesis, regions[0], regions[1]
            )

        if isinstance(regions, MappingABC):
            raw_regions = []
            for region_id, value in regions.items():
                if isinstance(value, Region):
                    raw_regions.append((region_id, value.reference, value.hypothesis))
                elif isinstance(value, MappingABC):
                    raw_regions.append(
                        (
                            region_id,
                            _mapping_side(value, "reference"),
                            _mapping_side(value, "hypothesis"),
                        )
                    )
                elif isinstance(value, (tuple, list)) and len(value) == 2:
                    raw_regions.append((region_id, value[0], value[1]))
                else:
                    raise TypeError(
                        "Region mapping values must be (reference_lines, "
                        "hypothesis_lines) pairs or mappings."
                    )
            return raw_regions

        if _looks_like_region_groups(regions):
            raw: List[Tuple[Any, Any, Any]] = []
            for index, item in enumerate(regions):
                raw.append(_coerce_region_item(item, index))
            return raw

        # ``regions`` as labels for a single side is ambiguous; require the
        # explicit reference_regions/hypothesis_regions names instead.
        raise TypeError(
            "regions must be region triples/objects, a mapping, or a pair of "
            "reference/hypothesis label sequences."
        )

    if reference_regions is not None or hypothesis_regions is not None:
        if reference is None or hypothesis is None:
            raise ValueError(
                "reference and hypothesis lines are required when region labels are given."
            )
        if reference_regions is None or hypothesis_regions is None:
            raise ValueError(
                "Both reference_regions and hypothesis_regions must be provided."
            )
        return _groups_from_labels(
            reference, hypothesis, reference_regions, hypothesis_regions
        )

    if _is_nested_region_pair(reference, hypothesis):
        return _groups_from_nested(reference, hypothesis, region_ids)

    if reference is None and hypothesis is None:
        return []

    if region_ids is not None and reference is not None and hypothesis is not None:
        ref_count = len(_as_line_list(reference))
        hyp_count = len(_as_line_list(hypothesis))
        if len(region_ids) == ref_count and len(region_ids) == hyp_count:
            return _groups_from_labels(
                reference, hypothesis, region_ids, region_ids
            )

    # Direct call to process_region_words(reference=..., hypothesis=...) means
    # one continuous region.  Use a sentinel-ish stable id of 0.
    return [(0, reference if reference is not None else [], hypothesis if hypothesis is not None else [])]


def _coerce_region_item(item: Any, index: int) -> Tuple[Any, Any, Any]:
    if isinstance(item, Region):
        return item.id, item.reference, item.hypothesis

    if isinstance(item, MappingABC):
        region_id = _first_mapping_value(
            item, "id", "region", "region_id", "segment", "segment_id", "name",
            default=index,
        )
        return (
            region_id,
            _mapping_side(item, "reference"),
            _mapping_side(item, "hypothesis"),
        )

    if isinstance(item, (tuple, list)):
        if len(item) == 3:
            return item[0], item[1], item[2]
        if len(item) == 2:
            first, second = item
            # ("A", (reference_lines, hypothesis_lines))
            if _is_scalar_id(first) and isinstance(second, (tuple, list)) and len(second) == 2:
                return first, second[0], second[1]
            # (reference_lines, hypothesis_lines)
            return index, first, second

    raise TypeError(
        "Each region must be a (region_id, reference_lines, hypothesis_lines) "
        "tuple, Region object, or mapping."
    )


def _mapping_side(mapping: Mapping[str, Any], side: str) -> Any:
    if side == "reference":
        keys = (
            "reference",
            "references",
            "reference_lines",
            "ref",
            "ref_lines",
            "manual",
            "manual_transcript",
            "ground_truth",
        )
    else:
        keys = (
            "hypothesis",
            "hypotheses",
            "hypothesis_lines",
            "hyp",
            "hyp_lines",
            "asr",
            "asr_lines",
            "recognized",
            "recognized_lines",
        )

    for key in keys:
        if key in mapping:
            return mapping[key]

    raise KeyError(f"Region mapping is missing a {side!r} entry")


def _groups_from_labels(
    reference: Any,
    hypothesis: Any,
    reference_labels: Sequence[Any],
    hypothesis_labels: Sequence[Any],
) -> List[Tuple[Any, Any, Any]]:
    ref_lines = [
        _coerce_line(line, index)
        for index, line in enumerate(_as_line_list(reference))
    ]
    hyp_lines = [
        _coerce_line(line, index)
        for index, line in enumerate(_as_line_list(hypothesis))
    ]
    ref_labels = _broadcast_labels(reference_labels, len(ref_lines), "reference_regions")
    hyp_labels = _broadcast_labels(hypothesis_labels, len(hyp_lines), "hypothesis_regions")

    order: List[Any] = []
    ref_buckets: dict = {}
    hyp_buckets: dict = {}

    for line, label in zip(ref_lines, ref_labels):
        if label not in ref_buckets:
            ref_buckets[label] = []
            order.append(label)
        ref_buckets[label].append(line)

    for line, label in zip(hyp_lines, hyp_labels):
        if label not in hyp_buckets:
            hyp_buckets[label] = []
            if label not in ref_buckets:
                order.append(label)
        hyp_buckets.setdefault(label, []).append(line)

    return [
        (
            label,
            ref_buckets.get(label, []),
            hyp_buckets.get(label, []),
        )
        for label in order
    ]


def _groups_from_nested(
    reference: Any,
    hypothesis: Any,
    region_ids: Optional[Sequence[Any]] = None,
) -> List[Tuple[Any, Any, Any]]:
    ref_regions = list(reference)
    hyp_regions = list(hypothesis)

    if len(ref_regions) != len(hyp_regions):
        raise ValueError(
            "Nested reference and hypothesis must contain the same number of regions."
        )

    if region_ids is not None and len(region_ids) != len(ref_regions):
        raise ValueError("region_ids must have one identifier per nested region.")

    groups = []
    for index, (raw_ref, raw_hyp) in enumerate(zip(ref_regions, hyp_regions)):
        ref_id, ref_lines = _unwrap_nested_side(raw_ref)
        hyp_id, hyp_lines = _unwrap_nested_side(raw_hyp)
        if region_ids is not None:
            region_id = region_ids[index]
        elif ref_id is not None and hyp_id is not None and ref_id != hyp_id:
            raise ValueError(
                f"Nested region ids differ at index {index}: {ref_id!r} != {hyp_id!r}."
            )
        else:
            region_id = ref_id if ref_id is not None else hyp_id if hyp_id is not None else index

        groups.append((region_id, ref_lines, hyp_lines))

    return groups


def _unwrap_nested_side(value: Any) -> Tuple[Any, Any]:
    if isinstance(value, Region):
        return value.id, value.reference if hasattr(value, "reference") else value.hypothesis
    if isinstance(value, MappingABC):
        return _first_mapping_value(
            value, "id", "region", "region_id", "segment", "name", default=None
        ), _mapping_side(
            value,
            "reference" if any(k in value for k in ("reference", "references", "ref")) else "hypothesis",
        )

    # ("A", ["line one", ...])
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and _is_scalar_id(value[0])
        and isinstance(value[1], (str, list, tuple))
    ):
        return value[0], value[1]

    return None, value


def _coerce_lines(lines: Any, start_index: int) -> Tuple[List[LineRecord], int]:
    if lines is None:
        return [], start_index

    raw_lines = _as_line_list(lines)
    records: List[LineRecord] = []
    index = start_index

    for raw_line in raw_lines:
        records.append(_coerce_line(raw_line, index))
        index += 1

    return records, index


def _as_line_list(lines: Any) -> List[Any]:
    if isinstance(lines, str):
        return [lines]
    if isinstance(lines, (tuple, list)):
        # A single explicit (line_id, text) tuple should not be iterated as
        # multiple lines.
        if _is_explicit_line_tuple(lines):
            return [lines]
        return list(lines)
    return [lines]


def _coerce_line(line: Any, index: int) -> LineRecord:
    if isinstance(line, LineRecord):
        return line

    if isinstance(line, MappingABC):
        text = _first_mapping_value(
            line, "text", "transcript", "value", "content", default=None
        )
        if text is None:
            raise ValueError(f"Line mapping is missing text: {line!r}")
        line_id = _first_mapping_value(
            line, "line_id", "id", "name", "key", default=index
        )
        return LineRecord(index=index, line_id=line_id, text=str(text))

    if _is_explicit_line_tuple(line):
        return LineRecord(index=index, line_id=line[0], text=str(line[1]))

    if isinstance(line, str):
        return LineRecord(index=index, line_id=index, text=line)

    if isinstance(line, (list, tuple)) and all(isinstance(word, str) for word in line):
        return LineRecord(index=index, line_id=index, text=" ".join(line))

    raise TypeError(
        "Each transcript line must be a string, a (line_id, text) tuple, or a mapping."
    )


def _transform_lines(
    lines: Sequence[LineRecord],
    transform: Union[tr.Compose, tr.AbstractTransform],
    is_reference: bool,
) -> Tuple[List[str], Tuple[WordSource, ...]]:
    words: List[str] = []
    sources: List[WordSource] = []

    for region_line_index, line in enumerate(lines):
        transformed = _apply_transform([line.text], transform, is_reference=is_reference)
        # Standard transforms return one inner word list per input line.  Be
        # liberal with custom transforms and flatten any additional output
        # lists while keeping a continuous in-line word order.
        line_words = [word for sentence_words in transformed for word in sentence_words]

        for word_index, word in enumerate(line_words):
            words.append(word)
            sources.append(
                WordSource(
                    line.line_id,
                    word_index,
                    line_index=line.index,
                    region_line_index=region_line_index,
                    word=word,
                )
            )

    return words, tuple(sources)


################################################################################
# Small type/shape helpers


def _unique(values: Any) -> Tuple[Any, ...]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _iter_word_attributions(edits: Sequence[RegionEdit]):
    for edit in edits:
        if edit.type == "delete":
            for position in edit.ref_positions:
                yield WordAttribution(edit.region, edit.type, "reference", position)
        elif edit.type == "insert":
            for position in edit.hyp_positions:
                yield WordAttribution(edit.region, edit.type, "hypothesis", position)
        elif edit.type == "substitute":
            for index, ref_position in enumerate(edit.ref_positions):
                hyp_position = (
                    edit.hyp_positions[index]
                    if index < len(edit.hyp_positions)
                    else None
                )
                yield WordAttribution(
                    edit.region,
                    edit.type,
                    "reference",
                    ref_position,
                    hyp_position,
                )
            for index, hyp_position in enumerate(edit.hyp_positions):
                if index >= len(edit.ref_positions):
                    yield WordAttribution(
                        edit.region,
                        edit.type,
                        "hypothesis",
                        hyp_position,
                    )
        # Equal chunks are intentionally absent from RegionResult.edits.


def _first_kwarg(kwargs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return None


def _reject_unknown_kwargs(kwargs: Mapping[str, Any]) -> None:
    known = {
        "segments",
        "segment",
        "region",
        "segment_ids",
        "ref_regions",
        "hyp_regions",
        "reference_region_ids",
        "hypothesis_region_ids",
        "reference_segments",
        "hypothesis_segments",
        "ref_segments",
        "hyp_segments",
    }
    unexpected = [key for key in kwargs if key not in known]
    if unexpected:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(unexpected)}")


def _first_mapping_value(mapping: Mapping[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _is_scalar_id(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool)) or (
        hasattr(value, "__hash__") and not isinstance(value, (list, dict, set))
    )


def _is_explicit_line_tuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_scalar_id(value[0])
        and isinstance(value[1], str)
    )


def _as_label_list(labels: Any) -> List[Any]:
    if isinstance(labels, (str, int)):
        return [labels]
    return list(labels)


def _broadcast_labels(labels: Any, length: int, name: str) -> List[Any]:
    labels = _as_label_list(labels)
    if len(labels) == 1 and length != 1:
        return labels * length
    if len(labels) != length:
        raise ValueError(
            f"{name} has {len(labels)} labels for {length} input lines."
        )
    return labels


def _is_label_sequence(value: Any) -> bool:
    if isinstance(value, str):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False

    # Empty list is usable as a label sequence only in an empty-side context.
    items = list(iterator)
    return all(_is_scalar_id(item) and not isinstance(item, (tuple, list, dict)) for item in items)


def _looks_like_region_groups(value: Any) -> bool:
    if isinstance(value, (str, bytes, MappingABC, Region)):
        return False
    try:
        items = list(value)
    except TypeError:
        return False

    if not items:
        return True

    return all(_is_region_group_item(item) for item in items)


def _is_region_group_item(item: Any) -> bool:
    if isinstance(item, Region) or isinstance(item, MappingABC):
        return True
    if not isinstance(item, (tuple, list)):
        return False
    if len(item) == 3:
        return True
    if len(item) == 2:
        first, second = item
        if _is_scalar_id(first) and isinstance(second, (tuple, list)) and len(second) == 2:
            return True
        # (reference_lines, hypothesis_lines)
        return not _is_scalar_id(first) or isinstance(first, (list, tuple))
    return False


def _is_nested_region_pair(reference: Any, hypothesis: Any) -> bool:
    if reference is None or hypothesis is None:
        return False
    if isinstance(reference, str) or isinstance(hypothesis, str):
        return False
    try:
        ref_items = list(reference)
        hyp_items = list(hypothesis)
    except TypeError:
        return False

    if len(ref_items) == 0 or len(ref_items) != len(hyp_items):
        return False

    def looks_nested_side(item: Any) -> bool:
        if isinstance(item, (Region, MappingABC)):
            return True
        if isinstance(item, str):
            return False
        if isinstance(item, (tuple, list)):
            # ("region-id", ["line"]) or [line, line] where lines themselves
            # are strings/tuples.  A plain list of strings is not nested.
            if len(item) == 2 and _is_scalar_id(item[0]) and isinstance(item[1], (list, tuple)):
                return True
            return len(item) > 0 and all(
                isinstance(line, str) or _is_explicit_line_tuple(line) or isinstance(line, MappingABC)
                for line in item
            )
        return False

    return all(looks_nested_side(item) for item in ref_items + hyp_items)
