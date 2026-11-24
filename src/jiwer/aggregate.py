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
Aggregate word-level alignment results produced by (possibly many) workers.

Each worker aligns one or more recordings with
[jiwer.process_words][process.process_words]. Because a batch may be retried
and delivered twice, results are keyed by recording ID:

* Results with the same recording ID and identical transformed references,
  hypotheses and error counts are de-duplicated (counted only once).
* Results with different recording IDs are always counted separately, even
  when their content is identical.
* If two results share a recording ID but differ in content (even when the
  error counts happen to be equal), the entire merge is rejected and the
  previously aggregated results are kept untouched.

A summary only consumes already-aligned results. The existing conventions
for insertions into silence and for the WER of empty (zero-word) references
are preserved.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union

from jiwer.process import WordOutput

__all__ = [
    "MergeConflictError",
    "RecordingResult",
    "Summary",
    "load_summary",
    "merge_summaries",
    "save_summary",
    "summarize",
]


class MergeConflictError(ValueError):
    """
    Raised when a merge encounters two results with the same recording ID
    but different transformed references, hypotheses, or error counts.

    The merge is rejected as a whole: no result from the conflicting merge
    is added to the already aggregated results.
    """


@dataclass(frozen=True)
class RecordingResult:
    """
    The aligned result of a single recording, keyed by its recording ID.

    Attributes:
        recording_id: Unique identifier of the recording
        references: The transformed reference sentences (lists of words)
        hypotheses: The transformed hypothesis sentences (lists of words)
        hits: The number of correct words
        substitutions: The number of substitutions
        insertions: The number of insertions
        deletions: The number of deletions
    """

    recording_id: str

    references: List[List[str]]
    hypotheses: List[List[str]]

    hits: int
    substitutions: int
    insertions: int
    deletions: int

    @classmethod
    def from_output(
        cls, recording_id: str, output: WordOutput
    ) -> "RecordingResult":
        """
        Create a result for a single recording from the output of
        [jiwer.process_words][process.process_words].

        Args:
            recording_id: Unique identifier of the recording
            output: The word-level alignment output for that recording

        Returns:
            (RecordingResult): The aligned result, keyed by the recording ID
        """
        return cls(
            recording_id=recording_id,
            references=[list(s) for s in output.references],
            hypotheses=[list(s) for s in output.hypotheses],
            hits=output.hits,
            substitutions=output.substitutions,
            insertions=output.insertions,
            deletions=output.deletions,
        )

    def _content(self) -> Tuple:
        # everything that identifies a (re-)delivery of the same alignment
        return (
            _freeze(self.references),
            _freeze(self.hypotheses),
            self.hits,
            self.substitutions,
            self.insertions,
            self.deletions,
        )


def _freeze(sentences: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(s) for s in sentences)


class Summary:
    """
    An aggregate of aligned results, keyed by recording ID.

    The totals (hits, substitutions, insertions, deletions) and the WER are
    computed over the de-duplicated results. The WER follows the same
    convention as [jiwer.process_words][process.process_words]: when the
    references contain no words at all, the WER equals the number of
    insertions.
    """

    def __init__(self, results: Iterable[RecordingResult] = ()):
        self._results: Dict[str, RecordingResult] = {}
        self._add_all(self._results, results)

    @staticmethod
    def _add_all(
        acc: Dict[str, RecordingResult], results: Iterable[RecordingResult]
    ) -> None:
        for result in results:
            existing = acc.get(result.recording_id)
            if existing is None:
                acc[result.recording_id] = result
            elif existing._content() != result._content():
                raise MergeConflictError(
                    f"Conflicting results for recording "
                    f"id={result.recording_id!r}: the transformed references, "
                    f"hypotheses, or error counts differ. The merge is rejected."
                )
            # identical re-delivery: count only once

    @property
    def recording_ids(self) -> List[str]:
        """The recording IDs of the aggregated results."""
        return list(self._results.keys())

    def __len__(self) -> int:
        return len(self._results)

    def __contains__(self, recording_id: str) -> bool:
        return recording_id in self._results

    def __iter__(self) -> Iterator[RecordingResult]:
        return iter(self._results.values())

    def __getitem__(self, recording_id: str) -> RecordingResult:
        return self._results[recording_id]

    @property
    def hits(self) -> int:
        """The total number of correct words."""
        return sum(r.hits for r in self._results.values())

    @property
    def substitutions(self) -> int:
        """The total number of substitutions."""
        return sum(r.substitutions for r in self._results.values())

    @property
    def insertions(self) -> int:
        """The total number of insertions."""
        return sum(r.insertions for r in self._results.values())

    @property
    def deletions(self) -> int:
        """The total number of deletions."""
        return sum(r.deletions for r in self._results.values())

    @property
    def wer(self) -> float:
        """
        The word error rate over all aggregated results.

        Follows the convention of [jiwer.process_words][process.process_words]:
        if the references contain no words, the WER is the number of
        insertions.
        """
        ref_words = self.hits + self.substitutions + self.deletions
        if ref_words == 0:
            return float(self.insertions)
        return float(
            self.substitutions + self.deletions + self.insertions
        ) / float(ref_words)


def summarize(recording_id: str, output: WordOutput) -> Summary:
    """
    Create a summary holding the aligned result of a single recording.

    Args:
        recording_id: Unique identifier of the recording
        output: The word-level alignment output for that recording

    Returns:
        (Summary): A summary containing only this recording's result
    """
    return Summary([RecordingResult.from_output(recording_id, output)])


def merge_summaries(*summaries: Summary) -> Summary:
    """
    Merge one or more summaries into a new summary.

    Results with the same recording ID and identical transformed references,
    hypotheses, and error counts are counted only once. Results with
    different recording IDs are each counted, even if their content is the
    same.

    The merge is atomic: if a recording ID occurs with differing content,
    a [MergeConflictError][aggregate.MergeConflictError] is raised and the
    given summaries are left untouched, so the previously aggregated results
    are preserved. The merge result does not depend on the order of the
    given summaries.

    Args:
        summaries: The summaries to merge

    Returns:
        (Summary): A new summary with the merged, de-duplicated results

    Raises:
        MergeConflictError: If a recording ID occurs with differing content
    """
    merged: Dict[str, RecordingResult] = {}
    for summary in summaries:
        Summary._add_all(merged, summary)
    merged_summary = Summary()
    merged_summary._results = merged
    return merged_summary


def save_summary(summary: Summary, path: Union[str, Path]) -> None:
    """
    Save a summary to disk as JSON, keyed by recording ID.

    Args:
        summary: The summary to save
        path: The file path to write to
    """
    payload = {
        "version": 1,
        "results": [asdict(r) for r in summary],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_summary(path: Union[str, Path]) -> Summary:
    """
    Load a summary previously stored with
    [save_summary][aggregate.save_summary].

    Args:
        path: The file path to read from

    Returns:
        (Summary): The loaded summary
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = [
        RecordingResult(
            recording_id=entry["recording_id"],
            references=[list(s) for s in entry["references"]],
            hypotheses=[list(s) for s in entry["hypotheses"]],
            hits=entry["hits"],
            substitutions=entry["substitutions"],
            insertions=entry["insertions"],
            deletions=entry["deletions"],
        )
        for entry in payload["results"]
    ]
    return Summary(results)
