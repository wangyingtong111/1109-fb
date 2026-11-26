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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import rapidfuzz

from jiwer import transforms as tr
from jiwer.transformations import cer_default, wer_default

__all__ = [
    "AlignmentChunk",
    "CharacterOutput",
    "WordOutput",
    "process_characters",
    "process_words",
    "process_words_with_timestamps",
]


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
        hit_pairs: For each sentence, a list of `(ref_idx, hyp_idx)` tuples for every
                   word that is aligned as a hit (an equal-word match). The indices are
                   the positions of the words in the (transformed) reference and
                   hypothesis sentence. Only populated when word timestamps are given.
        delete_intervals: For each sentence, a list of `(ref_start_idx, ref_end_idx)`
                          half-open intervals of the reference words that are deleted.
                          The indices are the positions of the words in the
                          (transformed) reference sentence. Only populated when word
                          timestamps are given.

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

    # detailed word-level alignment (original word positions), only populated
    # when word timestamps are supplied to process_words
    hit_pairs: List[List[Tuple[int, int]]] = field(default_factory=list)
    delete_intervals: List[List[Tuple[int, int]]] = field(default_factory=list)


def process_words(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    reference_timestamps: Optional[Sequence[Sequence[float]]] = None,
    hypothesis_timestamps: Optional[Sequence[Sequence[float]]] = None,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate.

    When word timestamps are supplied for both the reference and the hypothesis,
    the alignment is additionally constrained by the timestamps:

    1. The edit (Levenshtein) distance always remains the first optimization
       objective.
    2. Among all alignments with the same, minimal edit distance, the alignment
       is chosen which minimizes the sum of the absolute timestamp differences
       over all hit (equal-word) pairs.
    3. When the timestamp sums are equal as well, a stable alignment based on the
       word indices is returned.

    The `hit_pairs` and `delete_intervals` attributes of the returned
    [jiwer.WordOutput][process.WordOutput] are only populated when timestamps are
    given. They report the hit word pairs and deleted reference word intervals
    using the original word positions.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        reference_timestamps: Optional timestamp (e.g. start time in seconds) of every
                              word for each reference sentence. Must have the same
                              number of sentences as the reference, and the number of
                              timestamps per sentence must match the number of words
                              after applying the reference transform.
        hypothesis_timestamps: Optional timestamp of every word for each hypothesis
                               sentence, with the same constraints as
                               `reference_timestamps`.

    Returns:
        (WordOutput): The processed reference and hypothesis sentences

    Raises:
        ValueError: If one or more references are empty strings
        ValueError: If after applying transforms, reference and hypothesis lengths don't match
        ValueError: If only one of the two timestamp arguments is given
        ValueError: If the timestamp lengths don't match the (transformed) sentence counts
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

    use_timestamps = reference_timestamps is not None or hypothesis_timestamps is not None

    if use_timestamps:
        if reference_timestamps is None or hypothesis_timestamps is None:
            raise ValueError(
                "Both reference_timestamps and hypothesis_timestamps must be "
                "provided, or neither must be provided."
            )

        ref_timestamps = _validate_timestamps(
            reference_timestamps, ref_transformed, is_reference=True
        )
        hyp_timestamps = _validate_timestamps(
            hypothesis_timestamps, hyp_transformed, is_reference=False
        )

        return _process_words_with_timestamps(
            ref_transformed=ref_transformed,
            hyp_transformed=hyp_transformed,
            ref_timestamps=ref_timestamps,
            hyp_timestamps=hyp_timestamps,
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
    wer, mer, wil, wip = _compute_measures(
        hits=num_hits,
        substitutions=num_substitutions,
        deletions=num_deletions,
        insertions=num_insertions,
        num_ref_words=num_rf_words,
        num_hyp_words=num_hp_words,
    )

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


def process_words_with_timestamps(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_timestamps: Sequence[Sequence[float]],
    hypothesis_timestamps: Sequence[Sequence[float]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences, using word timestamps to disambiguate
    alignments with an identical edit distance.

    This is a convenience wrapper around
    [jiwer.process_words][process.process_words] which requires the word
    timestamps to be present. See
    [jiwer.process_words][process.process_words] for the exact optimization
    criteria and the meaning of the `hit_pairs` and `delete_intervals` output
    attributes.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_timestamps: The timestamp of every word for each reference sentence.
        hypothesis_timestamps: The timestamp of every word for each hypothesis sentence.
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (WordOutput): The processed reference and hypothesis sentences
    """
    return process_words(
        reference=reference,
        hypothesis=hypothesis,
        reference_transform=reference_transform,
        hypothesis_transform=hypothesis_transform,
        reference_timestamps=reference_timestamps,
        hypothesis_timestamps=hypothesis_timestamps,
    )


def _process_words_with_timestamps(
    ref_transformed: List[List[str]],
    hyp_transformed: List[List[str]],
    ref_timestamps: List[List[float]],
    hyp_timestamps: List[List[float]],
) -> WordOutput:
    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0

    # also keep track of the total number of words in the reference and hypothesis
    num_rf_words, num_hp_words = 0, 0

    # keep track of the alignment between each reference and hypothesis
    alignments: List[List[AlignmentChunk]] = []

    # detailed hit/delete information at the original word positions
    all_hit_pairs: List[List[Tuple[int, int]]] = []
    all_delete_intervals: List[List[Tuple[int, int]]] = []

    for ref_sentence, hyp_sentence, ref_times, hyp_times in zip(
        ref_transformed, hyp_transformed, ref_timestamps, hyp_timestamps
    ):
        opcodes, hit_pairs = _timed_levenshtein_opcodes(
            reference=ref_sentence,
            hypothesis=hyp_sentence,
            reference_times=ref_times,
            hypothesis_times=hyp_times,
        )

        subs = dels = ins = hits = 0
        sentence_op_chunks = []
        sentence_delete_intervals: List[Tuple[int, int]] = []

        for tag, i1, i2, j1, j2 in opcodes:
            sentence_op_chunks.append(
                AlignmentChunk(
                    type=tag,
                    ref_start_idx=i1,
                    ref_end_idx=i2,
                    hyp_start_idx=j1,
                    hyp_end_idx=j2,
                )
            )

            if tag == "equal":
                hits += i2 - i1
            elif tag == "replace":
                subs += i2 - i1
            elif tag == "delete":
                dels += i2 - i1
                sentence_delete_intervals.append((i1, i2))
            elif tag == "insert":
                ins += j2 - j1

        num_hits += hits
        num_substitutions += subs
        num_deletions += dels
        num_insertions += ins
        num_rf_words += len(ref_sentence)
        num_hp_words += len(hyp_sentence)

        alignments.append(sentence_op_chunks)
        all_hit_pairs.append(hit_pairs)
        all_delete_intervals.append(sentence_delete_intervals)

    wer, mer, wil, wip = _compute_measures(
        hits=num_hits,
        substitutions=num_substitutions,
        deletions=num_deletions,
        insertions=num_insertions,
        num_ref_words=num_rf_words,
        num_hyp_words=num_hp_words,
    )

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
        hit_pairs=all_hit_pairs,
        delete_intervals=all_delete_intervals,
    )


def _timed_cost_lt(
    a: Tuple[int, float], b: Tuple[int, float], time_tolerance: float = 1e-9
) -> bool:
    """
    Return True if alignment cost `a` is strictly better than `b`.

    A cost is an `(edit_distance, timestamp_cost)` pair which is optimized
    lexicographically: the edit distance is the first objective and the sum of
    the absolute timestamp differences of the hits is the second. Timestamp
    costs whose difference is within `time_tolerance` are treated as equal, so
    that floating point round-off cannot disturb the stable index-based tie
    breaking.
    """
    if a[0] != b[0]:
        return a[0] < b[0]

    if abs(a[1] - b[1]) <= time_tolerance:
        return False

    return a[1] < b[1]


def _timed_levenshtein_opcodes(
    reference: List[str],
    hypothesis: List[str],
    reference_times: List[float],
    hypothesis_times: List[float],
) -> Tuple[List[Tuple[str, int, int, int, int]], List[Tuple[int, int]]]:
    """
    Compute the levenshtein alignment opcodes between a single reference and
    hypothesis sentence, using word timestamps to break ties.

    The optimization is lexicographic:

    1. minimize the number of edits (substitutions, deletions and insertions),
    2. minimize the sum of the absolute timestamp differences over all hits,
    3. use a stable index order when both are equal.

    Returns:
        A tuple of (opcodes, hit_pairs), where opcodes is a list of
        `(tag, ref_start, ref_end, hyp_start, hyp_end)` tuples in the same
        format as rapidfuzz, and hit_pairs is a list of
        `(ref_idx, hyp_idx)` tuples for every equal-word alignment.
    """
    n = len(reference)
    m = len(hypothesis)

    # dp[i][j] stores the best (edit_distance, timestamp_cost) pair to align
    # reference[:i] with hypothesis[:j]. The pairs are compared lexicographically
    # (edit distance first, timestamp cost second) via `_timed_cost_lt`, which
    # treats near-equal timestamp costs as a tie so that floating point noise can
    # never override the stable index order.
    dp: List[List[Tuple[int, float]]] = [
        [(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    # back[i][j] stores the edit used to reach state (i, j):
    # 0 = diagonal (equal or substitute), 1 = delete, 2 = insert
    back: List[List[int]] = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0.0)
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = (j, 0.0)
        back[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # candidate 1: diagonal move, either a hit or a substitution
            prev_edits, prev_time_cost = dp[i - 1][j - 1]
            if reference[i - 1] == hypothesis[j - 1]:
                diagonal = (
                    prev_edits,
                    prev_time_cost
                    + abs(reference_times[i - 1] - hypothesis_times[j - 1]),
                )
            else:
                diagonal = (prev_edits + 1, prev_time_cost)

            # candidate 2: delete reference word i-1
            delete = (dp[i - 1][j][0] + 1, dp[i - 1][j][1])

            # candidate 3: insert hypothesis word j-1
            insert = (dp[i][j - 1][0] + 1, dp[i][j - 1][1])

            # The candidates are optimized lexicographically:
            #   1. minimum edit distance,
            #   2. minimum summed absolute timestamp difference over the hits,
            #   3. a fixed operation order whenever the first two objectives
            #      are tied: delete, then insert, then diagonal.
            # Preferring delete/insert on a tie matches the timestamp-free
            # (rapidfuzz) alignment, where a substitution is only emitted when
            # it is strictly better than the equivalent delete plus insert.
            # This yields a deterministic, index-based alignment on fully tied
            # inputs, while distinct timestamps still drive the decision via
            # the second objective.
            best = delete
            best_move = 1
            if _timed_cost_lt(insert, best):
                best = insert
                best_move = 2
            if _timed_cost_lt(diagonal, best):
                best = diagonal
                best_move = 0

            dp[i][j] = best
            back[i][j] = best_move

    # backtrack to obtain the per-word edits in reverse order
    edits: List[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 0:
            if reference[i - 1] == hypothesis[j - 1]:
                edits.append("equal")
            else:
                edits.append("replace")
            i -= 1
            j -= 1
        elif move == 1:
            edits.append("delete")
            i -= 1
        else:
            edits.append("insert")
            j -= 1

    edits.reverse()

    opcodes: List[Tuple[str, int, int, int, int]] = []
    hit_pairs: List[Tuple[int, int]] = []

    ref_idx = 0
    hyp_idx = 0
    for edit in edits:
        if edit == "equal":
            tag, ref_span, hyp_span = "equal", 1, 1
            hit_pairs.append((ref_idx, hyp_idx))
        elif edit == "replace":
            tag, ref_span, hyp_span = "replace", 1, 1
        elif edit == "delete":
            tag, ref_span, hyp_span = "delete", 1, 0
        else:
            tag, ref_span, hyp_span = "insert", 0, 1

        if opcodes and opcodes[-1][0] == tag:
            prev_tag, prev_i1, prev_i2, prev_j1, prev_j2 = opcodes[-1]
            opcodes[-1] = (
                prev_tag,
                prev_i1,
                prev_i2 + ref_span,
                prev_j1,
                prev_j2 + hyp_span,
            )
        else:
            opcodes.append(
                (
                    tag,
                    ref_idx,
                    ref_idx + ref_span,
                    hyp_idx,
                    hyp_idx + hyp_span,
                )
            )

        ref_idx += ref_span
        hyp_idx += hyp_span

    return opcodes, hit_pairs


def _compute_measures(
    hits: int,
    substitutions: int,
    deletions: int,
    insertions: int,
    num_ref_words: int,
    num_hyp_words: int,
) -> Tuple[float, float, float, float]:
    # special edge-case for empty references
    if num_ref_words == 0:
        wer = insertions

        # if the reference was silence and this is correctly predicted,
        # there is no error and all information is preserved
        if num_hyp_words == 0:
            mer = 0
            wip = 1
        else:
            mer = 1
            wip = 0
    else:
        wer = float(substitutions + deletions + insertions) / float(
            hits + substitutions + deletions
        )
        mer = float(substitutions + deletions + insertions) / float(
            hits + substitutions + deletions + insertions
        )

        # there is an edge-case when hypothesis is empty
        if num_hyp_words >= 1:
            wip = (float(hits) / num_ref_words) * (
                float(hits) / num_hyp_words
            )
        else:
            wip = 0

    wil = 1 - wip

    return wer, mer, wil, wip


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


def _validate_timestamps(
    timestamps: Sequence[Sequence[float]],
    transformed_sentences: List[List[str]],
    is_reference: bool,
) -> List[List[float]]:
    """
    Validate that the timestamps form a sequence of numeric sequences with the
    same number of sentences as the transformed input and exactly one timestamp
    per transformed word.
    """
    side = "reference" if is_reference else "hypothesis"

    if not isinstance(timestamps, (list, tuple)):
        raise ValueError(
            f"The {side} timestamps must be a sequence with one sequence of "
            "timestamps for each sentence."
        )

    if len(timestamps) != len(transformed_sentences):
        raise ValueError(
            f"The number of {side} timestamp sentences ({len(timestamps)}) "
            "does not match the number of "
            f"{side} sentences ({len(transformed_sentences)})."
        )

    validated: List[List[float]] = []
    for sentence_idx, (sentence_times, words) in enumerate(
        zip(timestamps, transformed_sentences)
    ):
        if not isinstance(sentence_times, (list, tuple)):
            raise ValueError(
                f"The timestamps of {side} sentence {sentence_idx} must be a "
                "sequence of numbers."
            )

        if len(sentence_times) != len(words):
            raise ValueError(
                f"The number of {side} timestamps ({len(sentence_times)}) in "
                f"sentence {sentence_idx} does not match the number of words "
                f"({len(words)}) after applying the transform."
            )

        if not all(
            isinstance(t, (int, float)) and not isinstance(t, bool)
            for t in sentence_times
        ):
            raise ValueError(
                f"All {side} timestamps in sentence {sentence_idx} must be numbers."
            )

        validated.append([float(t) for t in sentence_times])

    return validated


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
