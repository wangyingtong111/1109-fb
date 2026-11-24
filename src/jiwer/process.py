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

import itertools

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import rapidfuzz

from jiwer import transforms as tr
from jiwer.transformations import cer_default, wer_default

__all__ = [
    "AlignmentChunk",
    "CharacterOutput",
    "MultiReferenceWordOutput",
    "WordOutput",
    "process_characters",
    "process_words",
    "process_words_multi_reference",
]

# Accepted identifiers for the per-sentence reference selection strategy
_SELECTION_EDIT = "edit"
_SELECTION_WER = "wer"

_SELECTION_ALIASES = {
    "edit": _SELECTION_EDIT,
    "edits": _SELECTION_EDIT,
    "edit_distance": _SELECTION_EDIT,
    "minimum_edits": _SELECTION_EDIT,
    "minimum_edit_distance": _SELECTION_EDIT,
    "wer": _SELECTION_WER,
    "batch_wer": _SELECTION_WER,
    "minimum_wer": _SELECTION_WER,
    "oracle_wer": _SELECTION_WER,
}


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


@dataclass
class MultiReferenceWordOutput(WordOutput):
    """
    The output of calculating the word-level levenshtein distance between one or
    more hypothesis sentences, where each hypothesis sentence can have one or
    more acceptable reference transcriptions.

    Exactly one of the available references is selected for every sentence.
    The selected reference sentences are exposed through `references` (in the
    same order as the hypothesis sentences), and the zero-based index of the
    selected reference within each sentence's candidate list is exposed through
    `reference_indices` (also called `reference_indexes`). The alignment chunks
    and all measures are computed relative to the selected references.

    Attributes:
        references: The selected reference sentences
        hypotheses: The hypothesis sentences
        alignments: The alignment between each selected reference and its
                    hypothesis sentence
        wer: The word error rate
        mer: The match error rate
        wil: The word information lost measure
        wip: The word information preserved measure
        hits: The number of correct words
        substitutions: The number of substitutions
        insertions: The number of insertions
        deletions: The number of deletions
        reference_indices: For each sentence, the index of the selected
                           reference transcription within that sentence's
                           list of candidates
    """

    # index of the selected reference for each sentence
    reference_indices: List[int] = None

    @property
    def reference_indexes(self) -> List[int]:
        """Alias of `reference_indices`."""
        return self.reference_indices

    @property
    def reference_index(self) -> List[int]:
        """Alias of `reference_indices`."""
        return self.reference_indices


def process_words(
    reference: Union[str, List[str], List[List[str]]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    method: str = _SELECTION_EDIT,
) -> Union[WordOutput, MultiReferenceWordOutput]:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate.

    If the same recording has multiple acceptable transcriptions, a nested list
    of references can be given. In that case one full reference transcription is
    selected for every sentence, and a :class:`MultiReferenceWordOutput` is
    returned which additionally contains the index of the selected reference.
    The regular, one-to-one entry point continues to work unchanged for string
    and flat list inputs.

    Args:
        reference: The reference sentence(s). Use a nested list (a list with one
                   list of candidate transcriptions per sentence) to provide
                   multiple references per sentence.
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        method: How to select a reference when multiple candidates are given.
                Either ``"edit"`` (default), which selects, per sentence, the
                candidate requiring the fewest word edits, or ``"wer"``, which
                selects the combination of references that minimizes the WER of
                the whole batch. Ignored when there is a single reference per
                sentence.

    Returns:
        (WordOutput): The processed reference and hypothesis sentences. A
        :class:`MultiReferenceWordOutput` is returned when multiple references
        are given for one or more sentences.

    Raises:
        ValueError: If one or more references are empty strings
        ValueError: If after applying transforms, reference and hypothesis lengths don't match
        ValueError: If an unknown selection `method` is given
    """
    # route nested references to the dedicated multi-reference implementation
    if _is_multi_reference_input(reference):
        return process_words_multi_reference(
            reference=reference,
            hypothesis=hypothesis,
            reference_transform=reference_transform,
            hypothesis_transform=hypothesis_transform,
            method=method,
        )

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

    alignments, stats = _align_sentences(ref_as_ints, hyp_as_ints)

    return _build_word_output(
        references=ref_transformed,
        hypotheses=hyp_transformed,
        alignments=alignments,
        num_rf_words=stats["ref_words"],
        num_hp_words=stats["hyp_words"],
        num_hits=stats["hits"],
        num_substitutions=stats["substitutions"],
        num_deletions=stats["deletions"],
        num_insertions=stats["insertions"],
    )


def process_words_multi_reference(
    reference: Union[List[str], List[List[str]]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    method: str = _SELECTION_EDIT,
) -> MultiReferenceWordOutput:
    """
    Compute the word-level levenshtein distance and alignment when one or more
    hypothesis sentences have multiple acceptable reference transcriptions.

    A complete reference transcription is selected for every sentence. The
    selection strategy is controlled with `method`:

    - ``"edit"``: for each sentence independently, the reference with the
      smallest number of word edits (levenshtein distance) is selected. Ties are
      broken by the order in which the candidate references are provided.
    - ``"wer"``: the combination of references minimizing the WER of the entire
      batch is selected. The total number of words in the selected references is
      used as the denominator. Ties on the batch WER are broken by, in corpus
      order, preferring the candidate that was provided first. When every
      selected reference would be empty, the existing zero-reference convention
      of :func:`process_words` is used.

    Args:
        reference: Either a list of candidate transcriptions for a single
                   sentence, or a nested list with a list of candidate
                   transcriptions for every sentence.
        hypothesis: The hypothesis sentence(s). A single string is treated as a
                    single sentence.
        reference_transform: The transformation(s) to apply to every reference string
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        method: ``"edit"`` for per-sentence minimum-edit selection, or ``"wer"``
                for minimum batch-WER selection.

    Returns:
        (MultiReferenceWordOutput): The selected references, their indices, the
        alignments with the hypothesis sentences, and all measures.

    Raises:
        ValueError: If no reference candidates are given for a sentence
        ValueError: If the number of reference and hypothesis sentences don't match
        ValueError: If an unknown selection `method` is given
    """
    selection = _normalize_selection_method(method)

    # Normalize the reference input into a list of lists of candidate strings
    if isinstance(reference, str):
        reference = [[reference]]
    elif len(reference) > 0 and all(isinstance(c, str) for c in reference):
        # a single sentence with multiple candidate transcriptions
        reference = [list(reference)]
    elif all(isinstance(candidates, (list, tuple)) for candidates in reference):
        # one list of candidate transcriptions per sentence
        reference = [list(candidates) for candidates in reference]
    else:
        raise ValueError(
            "The reference must be a list of candidate transcriptions for a single "
            "sentence, or a nested list containing one list of candidates per "
            "sentence. Please do not mix plain strings and lists of candidates."
        )

    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]

    if len(reference) == 0:
        raise ValueError(
            "At least one reference sentence with one or more candidates is required."
        )
    if any(len(candidates) == 0 for candidates in reference):
        raise ValueError(
            "Every sentence must have at least one reference transcription."
        )
    if len(reference) != len(hypothesis):
        raise ValueError(
            "The number of reference sentences and hypothesis sentences must match. "
            f"Instead got {len(reference)} reference and "
            f"{len(hypothesis)} hypothesis sentences."
        )

    # transform the hypothesis sentences once
    hyp_transformed = _apply_transform(
        hypothesis, hypothesis_transform, is_reference=False
    )

    # transform every candidate transcription independently
    candidates_transformed: List[List[List[str]]] = []
    for candidates in reference:
        transformed = _apply_transform(
            list(candidates), reference_transform, is_reference=True
        )
        candidates_transformed.append(transformed)

    # map every word to a unique integer over all references and hypotheses
    flat_refs = [
        words
        for sentence_candidates in candidates_transformed
        for words in sentence_candidates
    ]
    ref_as_ints, hyp_as_ints = _word2int(flat_refs, hyp_transformed)

    # regroup the integer-encoded references per sentence
    ref_int_groups: List[List[List[int]]] = []
    cursor = 0
    for sentence_candidates in candidates_transformed:
        group = ref_as_ints[cursor : cursor + len(sentence_candidates)]
        cursor += len(sentence_candidates)
        ref_int_groups.append(group)

    # compute the edit distance of every candidate against its hypothesis
    candidates_per_sentence: List[List[Tuple[int, int, int]]] = []
    for group, hyp_sentence in zip(ref_int_groups, hyp_as_ints):
        sentence_candidates = []
        for candidate_idx, ref_sentence in enumerate(group):
            edits = rapidfuzz.distance.Levenshtein.distance(
                ref_sentence, hyp_sentence
            )
            sentence_candidates.append(
                (candidate_idx, edits, len(ref_sentence))
            )
        candidates_per_sentence.append(sentence_candidates)

    if selection == _SELECTION_EDIT:
        selected_indices = _select_by_edit_distance(candidates_per_sentence)
    else:
        selected_indices = _select_by_batch_wer(candidates_per_sentence)

    # collect the selected references and compute their alignments
    selected_refs: List[List[str]] = []
    selected_ref_ints: List[List[int]] = []
    for sentence_idx, candidate_idx in enumerate(selected_indices):
        selected_refs.append(candidates_transformed[sentence_idx][candidate_idx])
        selected_ref_ints.append(ref_int_groups[sentence_idx][candidate_idx])

    alignments, stats = _align_sentences(selected_ref_ints, hyp_as_ints)

    output = _build_word_output(
        references=selected_refs,
        hypotheses=hyp_transformed,
        alignments=alignments,
        num_rf_words=stats["ref_words"],
        num_hp_words=stats["hyp_words"],
        num_hits=stats["hits"],
        num_substitutions=stats["substitutions"],
        num_deletions=stats["deletions"],
        num_insertions=stats["insertions"],
        output_cls=MultiReferenceWordOutput,
    )
    output.reference_indices = list(selected_indices)

    return output


def _normalize_selection_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError(
            f"Unknown reference selection method={method!r}. "
            f"Expected one of {sorted(set(_SELECTION_ALIASES))}."
        )

    key = method.lower()
    if key not in _SELECTION_ALIASES:
        raise ValueError(
            f"Unknown reference selection method={method!r}. "
            f"Expected one of {sorted(set(_SELECTION_ALIASES))}."
        )

    return _SELECTION_ALIASES[key]


def _is_multi_reference_input(reference: Any) -> bool:
    """
    Detect a nested list of references, i.e. a list containing one or more
    lists of candidate transcriptions.
    """
    return (
        isinstance(reference, list)
        and len(reference) > 0
        and any(isinstance(element, (list, tuple)) for element in reference)
    )


def _select_by_edit_distance(
    candidates_per_sentence: List[List[Tuple[int, int, int]]]
) -> List[int]:
    """
    Select, for every sentence, the candidate with the fewest edits.
    Candidates are visited in their provided order, so an equal number of edits
    keeps the first candidate.
    """
    selected_indices = []
    for sentence_candidates in candidates_per_sentence:
        best_idx = 0
        best_edits = None
        for candidate_idx, edits, _ref_len in sentence_candidates:
            if best_edits is None or edits < best_edits:
                best_idx = candidate_idx
                best_edits = edits
        selected_indices.append(best_idx)

    return selected_indices


def _select_by_batch_wer(
    candidates_per_sentence: List[List[Tuple[int, int, int]]]
) -> List[int]:
    """
    Select the combination of references minimizing the WER over the whole
    batch. The batch WER is the total number of edits divided by the total
    number of words in the selected references.

    Ratios are compared exactly with cross multiplication. Equal ratios keep the
    combination which is first in corpus order (the smallest candidate index in
    the earliest differing sentence). If no combination contains any reference
    words, the first candidate of every sentence is selected.
    """
    best_indices = None
    best_edits = None
    best_ref_words = None

    for combination in itertools.product(
        *[
            [candidate_idx for candidate_idx, _, _ in sentence_candidates]
            for sentence_candidates in candidates_per_sentence
        ]
    ):
        total_edits = 0
        total_ref_words = 0
        for sentence_idx, candidate_idx in enumerate(combination):
            _, edits, ref_len = candidates_per_sentence[sentence_idx][
                candidate_idx
            ]
            total_edits += edits
            total_ref_words += ref_len

        if total_ref_words == 0:
            # zero-reference combinations are only valid on their own
            if best_indices is None:
                best_indices = combination
                best_edits = total_edits
                best_ref_words = total_ref_words
            continue

        if best_indices is None or best_ref_words == 0:
            is_better = True
        else:
            # exact ratio comparison: total_edits / total_ref_words
            lhs = total_edits * best_ref_words
            rhs = best_edits * total_ref_words
            is_better = lhs < rhs or (
                lhs == rhs and list(combination) < list(best_indices)
            )

        if is_better:
            best_indices = combination
            best_edits = total_edits
            best_ref_words = total_ref_words

    return list(best_indices)


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


def process_characters(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
) -> CharacterOutput:
    """
    Compute the character-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, the character error rate
    can be computed.

    Note that the by default this method includes space (` `) as a
    character over which the error rate is computed. If this is not desired, the
    reference and hypothesis transform need to be modified.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (CharacterOutput): The processed reference and hypothesis sentences.

    """
    # it's the same as word processing, just every word is of length 1
    result = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return CharacterOutput(
        references=result.references,
        hypotheses=result.hypotheses,
        alignments=result.alignments,
        cer=result.wer,
        hits=result.hits,
        substitutions=result.substitutions,
        insertions=result.insertions,
        deletions=result.deletions,
    )


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


def _align_sentences(
    reference: List[List[int]], hypothesis: List[List[int]]
) -> Tuple[List[List[AlignmentChunk]], dict]:
    """
    Compute the levenshtein opcodes and aggregated error counts between
    matching reference and hypothesis sentences.
    """
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0
    num_rf_words, num_hp_words = 0, 0
    alignments = []

    for reference_sentence, hypothesis_sentence in zip(reference, hypothesis):
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

    stats = {
        "hits": num_hits,
        "substitutions": num_substitutions,
        "deletions": num_deletions,
        "insertions": num_insertions,
        "ref_words": num_rf_words,
        "hyp_words": num_hp_words,
    }

    return alignments, stats


def _build_word_output(
    references: List[List[str]],
    hypotheses: List[List[str]],
    alignments: List[List[AlignmentChunk]],
    num_rf_words: int,
    num_hp_words: int,
    num_hits: int,
    num_substitutions: int,
    num_deletions: int,
    num_insertions: int,
    output_cls=WordOutput,
) -> WordOutput:
    """
    Compute all measures from the aggregated error counts and construct the
    output object.
    """
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
    return output_cls(
        references=references,
        hypotheses=hypotheses,
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
