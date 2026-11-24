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

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import rapidfuzz

from jiwer import transforms as tr
from jiwer.transformations import cer_default, wer_default

__all__ = [
    "AlignmentChunk",
    "CandidateOutput",
    "CharacterOutput",
    "PosteriorOutput",
    "TranscriptGroup",
    "WordOutput",
    "process_candidates",
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
# Implementation of probability-weighted candidate (decoding path) aggregation


@dataclass
class CandidateOutput:
    """
    The result of aligning a single hypothesis candidate (decoding path) against the
    reference, together with the weight that candidate received.

    Attributes:
        index: The position of the candidate in the input sequence.
        hypothesis: The original hypothesis string of this candidate.
        transcript: The hypothesis after the word-level transformation, as a tuple of
                    tokens. Candidates with an equal tuple are considered the same
                    transcript.
        log_weight: The weight of the candidate in log-space, when it was supplied as a
                    log weight (`None` otherwise).
        probability: The normalized probability of this candidate, i.e. its weight
                     divided by the sum of all candidate weights.
        output: The regular [jiwer.WordOutput][process.WordOutput] of this candidate,
                which holds its own alignment and error counts.
    """

    index: int
    hypothesis: str
    transcript: Tuple[str, ...]
    log_weight: Optional[float]
    probability: float
    output: WordOutput

    @property
    def wer(self) -> float:
        """The word error rate of this candidate on its own."""
        return self.output.wer

    @property
    def error_contribution(self) -> float:
        """
        The contribution of this candidate to the expected (posterior) WER,
        i.e. `probability * wer`.
        """
        return self.probability * self.output.wer

    @property
    def alignments(self) -> List[AlignmentChunk]:
        """The alignment chunks between the reference and this candidate."""
        return self.output.alignments[0]

    @property
    def hits(self) -> int:
        return self.output.hits

    @property
    def substitutions(self) -> int:
        return self.output.substitutions

    @property
    def deletions(self) -> int:
        return self.output.deletions

    @property
    def insertions(self) -> int:
        return self.output.insertions


@dataclass
class TranscriptGroup:
    """
    All candidates which, after the word-level transformation, yield the same
    transcript. Their probabilities are merged, which is what decides the top-1
    transcript.

    Attributes:
        transcript: The shared transcript as a tuple of tokens.
        probability: The merged probability of all candidates in this group.
        candidate_indices: The indices of the candidates in this group, in the order
                           they were given.
    """

    transcript: Tuple[str, ...]
    probability: float
    candidate_indices: List[int]

    @property
    def text(self) -> str:
        """The transcript as a single string, words separated by a space."""
        return " ".join(self.transcript)

    @property
    def first_candidate_index(self) -> int:
        """The index of the first candidate which produced this transcript."""
        return self.candidate_indices[0]


@dataclass
class PosteriorOutput:
    """
    The output of [jiwer.process_candidates][process.process_candidates].

    Attributes:
        reference: The reference sentence.
        candidates: One [jiwer.CandidateOutput][process.CandidateOutput] per input
                    candidate, keeping the input order.
        transcript_groups: The distinct transcripts after the word-level
                           transformation, with their merged probabilities. Groups are
                           ordered by the first appearance of their transcript.
        top1: The transcript group with the highest merged probability. Ties are
              resolved by the earliest candidate.
        wer: The expected (probability-weighted) word error rate over all candidates.
        log_weights: Whether the supplied weights were interpreted as log weights.
    """

    reference: str
    candidates: List[CandidateOutput]
    transcript_groups: List[TranscriptGroup]
    top1: TranscriptGroup
    wer: float
    log_weights: bool

    @property
    def expected_wer(self) -> float:
        """Alias for [wer][process.PosteriorOutput.wer]."""
        return self.wer

    @property
    def top1_transcript(self) -> str:
        """The top-1 transcript as a single string."""
        return self.top1.text

    @property
    def top1_candidate(self) -> CandidateOutput:
        """The first candidate which produced the top-1 transcript."""
        return self.candidates[self.top1.first_candidate_index]

    @property
    def top1_output(self) -> WordOutput:
        """
        The regular [jiwer.WordOutput][process.WordOutput] of the first candidate
        which produced the top-1 transcript, so its alignment can be inspected with
        the existing alignment and report entry points.
        """
        return self.top1_candidate.output

    def candidate_contributions(self) -> List[Tuple[int, float, float, float]]:
        """
        Retrieve, for every candidate in input order, a tuple of
        `(index, probability, wer, error_contribution)`.
        """
        return [
            (c.index, c.probability, c.wer, c.error_contribution)
            for c in self.candidates
        ]


def process_candidates(
    reference: Union[str, List[str]],
    hypotheses: Sequence[str],
    probabilities: Optional[Sequence[float]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    log_weights: bool = False,
) -> PosteriorOutput:
    """
    Aggregate the word error rate over several hypothesis candidates (decoding paths)
    of the same sentence, each carrying a probability.

    Every candidate is aligned individually, so its probability, its alignment and its
    own error counts remain available. The returned `wer` is the expected error rate,
    i.e. the sum of `probability * candidate.wer` over the candidates. The top-1
    transcript is chosen by merging the probabilities of all candidates which have the
    same hypothesis *after* the word-level transformation; ties are resolved in favour
    of the candidate which appeared first.

    Weights can be given either as probabilities or as log weights. Log weights may be
    shifted by an arbitrary common offset (for instance when they all have been
    decreased by 1000): the computation is shifted by the maximum weight before
    exponentiation, so no probability underflows because of that common offset.

    Args:
        reference: The reference sentence
        hypotheses: The hypothesis sentence of each candidate
        probabilities: The probability (or log weight) of each candidate. When omitted,
                       all candidates are weighted equally.
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        log_weights: If true, `probabilities` is interpreted as log weights
                     (e.g. log-probabilities), which need not be normalized.

    Returns:
        (PosteriorOutput): The per-candidate alignments, the merged transcript groups,
                           the top-1 transcript, and the expected WER.

    Raises:
        ValueError: If no hypotheses are given
        ValueError: If the number of weights does not match the number of hypotheses
        ValueError: If a weight is negative or all weights are zero
        ValueError: If log weights are not finite (and not `-inf`), or all are `-inf`

    Example:
        ```python
        import jiwer

        out = jiwer.process_candidates(
            "red green",
            ["red green", "red", "red blue green"],
            [0.25, 0.5, 0.25],
        )
        print(out.wer)                  # 0.375
        print(out.top1_transcript)      # red

        # merged probabilities of equal transcripts decide the top-1
        out = jiwer.process_candidates(
            "red green",
            ["red green", "red  green", "red"],
            [0.3, 0.3, 0.4],
            log_weights=False,
        )
        print(out.top1_transcript)      # red green
        ```
    """
    if isinstance(reference, list):
        if len(reference) != 1:
            raise ValueError(
                "process_candidates expects a single reference sentence, "
                f"got {len(reference)} reference sentences."
            )
        reference = reference[0]
    elif not isinstance(reference, str):
        raise ValueError(
            "process_candidates expects a single reference sentence, "
            f"got {reference!r}."
        )

    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    hypotheses = list(hypotheses)

    if len(hypotheses) == 0:
        raise ValueError("process_candidates requires at least one hypothesis.")

    if probabilities is None:
        weights = [1.0] * len(hypotheses)
        logs: List[Optional[float]] = [None] * len(hypotheses)
    else:
        weights = [float(p) for p in probabilities]
        if len(weights) != len(hypotheses):
            raise ValueError(
                "The number of weights "
                f"({len(weights)}) must match the number of hypotheses "
                f"({len(hypotheses)})."
            )

        for weight in weights:
            if math.isnan(weight) or weight == math.inf:
                raise ValueError(f"Weights must not be NaN or +inf, got {weight}.")
            if not log_weights and weight < 0:
                raise ValueError(f"Probabilities may not be negative, got {weight}.")

        if log_weights:
            logs = list(weights)
            offset = max(logs)
            if offset == -math.inf:
                raise ValueError("Weights may not all be zero.")
            # Shift by the maximum before exponentiating: a common offset on the
            # log weights (e.g. all lowered by 1000) then cancels exactly, instead
            # of underflowing towards zero.
            weights = [
                math.exp(w - offset) if w != -math.inf else 0.0 for w in logs
            ]
        else:
            logs = [None] * len(hypotheses)

    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("Weights may not all be zero.")

    normalized = [w / total for w in weights]

    candidates: List[CandidateOutput] = []
    for index, (hypothesis, probability, log_weight) in enumerate(
        zip(hypotheses, normalized, logs)
    ):
        output = process_words(
            reference,
            hypothesis,
            reference_transform,
            hypothesis_transform,
        )
        candidates.append(
            CandidateOutput(
                index=index,
                hypothesis=hypothesis,
                transcript=tuple(output.hypotheses[0]),
                log_weight=log_weight,
                probability=probability,
                output=output,
            )
        )

    # Merge candidates which result in the same transcript after the word-level
    # transformation. Groups are kept in order of first appearance, and a tie in the
    # merged probability is won by the group which appeared first.
    groups: Dict[Tuple[str, ...], TranscriptGroup] = {}
    for candidate in candidates:
        group = groups.get(candidate.transcript)
        if group is None:
            groups[candidate.transcript] = TranscriptGroup(
                transcript=candidate.transcript,
                probability=candidate.probability,
                candidate_indices=[candidate.index],
            )
        else:
            group.probability = math.fsum(
                [group.probability, candidate.probability]
            )
            group.candidate_indices.append(candidate.index)

    transcript_groups = list(groups.values())
    top1 = transcript_groups[0]
    for group in transcript_groups[1:]:
        if group.probability > top1.probability:
            top1 = group

    expected_wer = math.fsum(
        probability * candidate.wer
        for probability, candidate in zip(normalized, candidates)
    )

    return PosteriorOutput(
        reference=reference,
        candidates=candidates,
        transcript_groups=transcript_groups,
        top1=top1,
        wer=expected_wer,
        log_weights=log_weights,
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
