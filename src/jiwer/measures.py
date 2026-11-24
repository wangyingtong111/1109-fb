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
Convenience methods for calculating a number of similarity error
measures between one or more reference and hypothesis sentences.
These measures are commonly used to measure the performance of
an automatic speech recognition (ASR) system.

The following measures are implemented:

- Word Error Rate (WER), which is where this library got its name from. This
  has long been (and arguably still is) the de facto standard for computing
  ASR performance.
- Match Error Rate (MER)
- Word Information Lost (WIL)
- Word Information Preserved (WIP)
- Character Error Rate (CER)

Note that these functions merely call
[jiwer.process_words][process.process_words] and
[jiwer.process_characters][process.process_characters].
It is more efficient to call `process_words` or `process_characters` and access the
results from the
[jiwer.WordOutput][process.WordOutput] and
[jiwer.CharacterOutput][process.CharacterOutput]
classes.
"""

from typing import List, Union

from jiwer import transforms as tr
from jiwer.process import process_characters, process_words
from jiwer.transformations import cer_default, wer_default

__all__ = [
    "cer",
    "mer",
    "wer",
    "wil",
    "wip",
    "weighted_wer",
]

########################################################################################
# Implementation of the WER method and co, exposed publicly


def wer(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    substitution_cost: int = 1,
    deletion_cost: int = 1,
    insertion_cost: int = 1,
) -> float:
    """
    Calculate the word error rate (WER) between one or more reference and
    hypothesis sentences.

    With the default unit costs this is the classical WER. Explicit per-operation
    costs can be supplied to select the alignment on the minimum weighted edit
    cost; the returned WER is still computed with the original (unit-weight)
    formula from the edit counts of the selected alignment. To obtain the
    weighted total itself, use
    [jiwer.weighted_wer][measures.weighted_wer].

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        substitution_cost: The cost of substituting a reference word
        deletion_cost: The cost of deleting a reference word
        insertion_cost: The cost of inserting a hypothesis word

    Returns:
        (float): The word error rate of the given reference and
                 hypothesis sentence(s).
    """
    output = process_words(
        reference,
        hypothesis,
        reference_transform,
        hypothesis_transform,
        substitution_cost=substitution_cost,
        deletion_cost=deletion_cost,
        insertion_cost=insertion_cost,
    )

    return output.wer


def weighted_wer(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    substitution_cost: int = 1,
    deletion_cost: int = 1,
    insertion_cost: int = 1,
) -> int:
    """
    Calculate the total weighted edit cost between one or more reference and
    hypothesis sentences under explicit per-operation costs.

    The alignment is chosen to minimize the total cost, which is computed as

    ``substitutions * substitution_cost``
    ``+ deletions * deletion_cost``
    ``+ insertions * insertion_cost``.

    For example, with a substitution cost of 3 and deletion and insertion costs
    of 1, a single mismatched word is cheaper to explain as one deletion plus
    one insertion (total cost 2) than as one substitution (cost 3).

    Note that this returns an absolute edit cost, not a rate. The classical WER
    remains available through [jiwer.wer][measures.wer].

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        substitution_cost: The cost of substituting a reference word. Must be positive.
        deletion_cost: The cost of deleting a reference word. Must be positive.
        insertion_cost: The cost of inserting a hypothesis word. Must be positive.

    Returns:
        (int): The total weighted edit cost between the reference and
               hypothesis sentence(s).
    """
    output = process_words(
        reference,
        hypothesis,
        reference_transform,
        hypothesis_transform,
        substitution_cost=substitution_cost,
        deletion_cost=deletion_cost,
        insertion_cost=insertion_cost,
    )

    return output.cost


def mer(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> float:
    """
    Calculate the match error rate (MER) between one or more reference and
    hypothesis sentences.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (float): The match error rate of the given reference and
                 hypothesis sentence(s).
    """
    output = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return output.mer


def wip(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> float:
    """
    Calculate the word information preserved (WIP) between one or more reference and
    hypothesis sentences.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (float): The word information preserved of the given reference and
                 hypothesis sentence(s).
    """
    output = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return output.wip


def wil(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> float:
    """
    Calculate the word information lost (WIL) between one or more reference and
    hypothesis sentences.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (float): The word information lost of the given reference and
                 hypothesis sentence(s).
    """
    output = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return output.wil


########################################################################################
# Implementation of character-error-rate, exposed publicly


def cer(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = cer_default,
) -> float:
    """
    Calculate the character error rate (CER) between one or more reference and
    hypothesis sentences.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (float): The character error rate of the given reference and hypothesis
                 sentence(s).
    """
    output = process_characters(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return output.cer
