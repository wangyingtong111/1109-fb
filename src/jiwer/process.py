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
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import rapidfuzz

from jiwer import transforms as tr
from jiwer.transformations import cer_default, wer_default

__all__ = [
    "AlignmentChunk",
    "CharacterOutput",
    "WordOutput",
    "process_characters",
    "process_words",
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
        cost: The total edit cost of the alignment, computed as
              ``substitutions * substitution_cost``
              ``+ deletions * deletion_cost``
              ``+ insertions * insertion_cost``. With the default unit costs this
              equals the (integer) Levenshtein distance, while ``wer`` keeps its
              original interpretation as an error rate.
        substitution_cost: The per-substitution cost used to select the alignment
        deletion_cost: The per-deletion cost used to select the alignment
        insertion_cost: The per-insertion cost used to select the alignment

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

    # weighted edit cost of the chosen alignment
    cost: int = 0
    substitution_cost: int = 1
    deletion_cost: int = 1
    insertion_cost: int = 1


def process_words(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    substitution_cost: int = 1,
    deletion_cost: int = 1,
    insertion_cost: int = 1,
) -> WordOutput:
    """
    Compute the word-level levenshtein distance and alignment between one or more
    reference and hypothesis sentences. Based on the result, multiple measures
    can be computed, such as the word error rate.

    By default every edit operation (substitution, deletion and insertion) has a
    unit cost, which reproduces the classical Levenshtein alignment and WER.
    Alternatively, explicit per-operation costs can be given. The alignment is
    then selected to minimize the total weighted edit cost, and the resulting
    cost is reported on the output alongside the edit counts. The classical WER
    interpretation (with unit weights) is always preserved on `WordOutput.wer`.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        substitution_cost: The cost of substituting a reference word with a
                           hypothesis word. Must be a positive integer.
        deletion_cost: The cost of deleting a reference word. Must be a positive
                       integer.
        insertion_cost: The cost of inserting a hypothesis word. Must be a
                        positive integer.

    Returns:
        (WordOutput): The processed reference and hypothesis sentences

    Raises:
        ValueError: If one or more references are empty strings
        ValueError: If after applying transforms, reference and hypothesis lengths don't match
        ValueError: If any of the costs is not a positive integer
    """
    # validate the explicit edit costs
    for name, cost in [
        ("substitution_cost", substitution_cost),
        ("deletion_cost", deletion_cost),
        ("insertion_cost", insertion_cost),
    ]:
        if not isinstance(cost, int) or isinstance(cost, bool) or cost <= 0:
            raise ValueError(
                f"{name} must be a positive integer, got {cost!r} instead."
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

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    num_hits, num_substitutions, num_deletions, num_insertions = 0, 0, 0, 0

    # also keep track of the total number of words in the reference and hypothesis
    num_rf_words, num_hp_words = 0, 0

    # anf finally, keep track of the alignment between each reference and hypothesis
    alignments = []

    # With the default unit costs, the classical Levenshtein alignment is used.
    # With custom costs, a weighted alignment is computed which minimizes the
    # total weighted edit cost.
    use_unit_costs = (
        substitution_cost == 1 and deletion_cost == 1 and insertion_cost == 1
    )

    for reference_sentence, hypothesis_sentence in zip(ref_as_ints, hyp_as_ints):
        if use_unit_costs:
            # Get the opcodes directly
            opcodes = rapidfuzz.distance.Levenshtein.opcodes(
                reference_sentence, hypothesis_sentence
            )
        else:
            opcodes = _weighted_levenshtein_opcodes(
                reference_sentence,
                hypothesis_sentence,
                substitution_cost=substitution_cost,
                deletion_cost=deletion_cost,
                insertion_cost=insertion_cost,
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

    # total weighted edit cost of the selected alignment
    total_cost = (
        subs * substitution_cost
        + dels * deletion_cost
        + ins * insertion_cost
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
        cost=total_cost,
        substitution_cost=substitution_cost,
        deletion_cost=deletion_cost,
        insertion_cost=insertion_cost,
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


def _weighted_levenshtein_opcodes(
    reference: List[int],
    hypothesis: List[int],
    *,
    substitution_cost: int,
    deletion_cost: int,
    insertion_cost: int,
) -> List[Tuple[str, int, int, int, int]]:
    """
    Compute the opcodes of the minimum-cost alignment between two sequences under
    explicit per-operation costs.

    `rapidfuzz.distance.Levenshtein.opcodes` only supports unit costs, so the
    weighted dynamic program is implemented here. The returned opcodes use the
    same ``(tag, i1, i2, j1, j2)`` convention as rapidfuzz, with `tag` one of
    `equal`, `replace`, `delete`, or `insert`.

    When several alignments share the minimum total cost, the alignment which
    keeps the most matching words is preferred, mirroring the convention used
    by rapidfuzz for the unit-cost case. Consecutive operations of the same kind
    are merged into a single opcode.
    """
    m = len(reference)
    n = len(hypothesis)

    # dp[i][j] is the minimum cost of aligning reference[:i] with hypothesis[:j]
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    # hits[i][j] is the maximum number of matching words achievable among the
    # minimum-cost alignments of reference[:i] with hypothesis[:j]
    hits: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + deletion_cost
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + insertion_cost

    def _prefer(candidate_cost: int, candidate_hits: int, cell_i: int, cell_j: int):
        # choose the minimum cost, breaking ties in favor of more matches
        return (candidate_cost < dp[cell_i][cell_j]) or (
            candidate_cost == dp[cell_i][cell_j]
            and candidate_hits > hits[cell_i][cell_j]
        )

    for i in range(1, m + 1):
        ref_word = reference[i - 1]
        for j in range(1, n + 1):
            delete_cost_value = dp[i - 1][j] + deletion_cost
            delete_hits = hits[i - 1][j]

            insert_cost_value = dp[i][j - 1] + insertion_cost
            insert_hits = hits[i][j - 1]

            is_match = ref_word == hypothesis[j - 1]
            diagonal_cost = dp[i - 1][j - 1] + (
                0 if is_match else substitution_cost
            )
            diagonal_hits = hits[i - 1][j - 1] + (1 if is_match else 0)

            # seed with the deletion candidate
            dp[i][j] = delete_cost_value
            hits[i][j] = delete_hits

            if _prefer(insert_cost_value, insert_hits, i, j):
                dp[i][j] = insert_cost_value
                hits[i][j] = insert_hits

            if _prefer(diagonal_cost, diagonal_hits, i, j):
                dp[i][j] = diagonal_cost
                hits[i][j] = diagonal_hits

    # backtrack the minimum-cost path, from (m, n) to (0, 0)
    raw_ops: List[Tuple[str, int, int]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            if dp[i][j] == dp[i - 1][j - 1] and hits[i][j] == hits[i - 1][j - 1] + 1:
                raw_ops.append(("equal", i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + substitution_cost
            and hits[i][j] == hits[i - 1][j - 1]
        ):
            raw_ops.append(("replace", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + deletion_cost and hits[i][j] == hits[i - 1][j]:
            raw_ops.append(("delete", i - 1, j))
            i -= 1
        else:
            # the remaining step must be an insertion
            raw_ops.append(("insert", i, j - 1))
            j -= 1

    # the backtrack gives the operations in reverse order
    raw_ops.reverse()

    # merge consecutive operations of the same type into rapidfuzz-style opcodes
    opcodes: List[Tuple[str, int, int, int, int]] = []
    for tag, ref_idx, hyp_idx in raw_ops:
        if tag == "equal" or tag == "replace":
            i1, i2 = ref_idx, ref_idx + 1
            j1, j2 = hyp_idx, hyp_idx + 1
        elif tag == "delete":
            i1, i2 = ref_idx, ref_idx + 1
            j1 = j2 = hyp_idx
        else:  # insert
            i1 = i2 = ref_idx
            j1, j2 = hyp_idx, hyp_idx + 1

        if opcodes and opcodes[-1][0] == tag:
            prev_tag, prev_i1, _, prev_j1, _ = opcodes[-1]
            opcodes[-1] = (prev_tag, prev_i1, i2, prev_j1, j2)
        else:
            opcodes.append((tag, i1, i2, j1, j2))

    return opcodes
