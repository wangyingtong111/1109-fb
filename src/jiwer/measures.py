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

from typing import List, Optional, Sequence, Union

from jiwer import transforms as tr
from jiwer.process import process_candidates, process_characters, process_words
from jiwer.transformations import cer_default, wer_default

__all__ = [
    "cer",
    "expected_wer",
    "mer",
    "wer",
    "wil",
    "wip",
]

########################################################################################
# Implementation of the WER method and co, exposed publicly


def wer(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
) -> float:
    """
    Calculate the word error rate (WER) between one or more reference and
    hypothesis sentences.

    Args:
        reference: The reference sentence(s)
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)

    Returns:
        (float): The word error rate of the given reference and
                 hypothesis sentence(s).
    """
    output = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

    return output.wer


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


def expected_wer(
    reference: Union[str, List[str]] = None,
    hypotheses: Sequence[str] = None,
    probabilities: Optional[Sequence[float]] = None,
    reference_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    log_weights: bool = False,
) -> float:
    """
    Calculate the probability-weighted (expected) word error rate over several
    hypothesis candidates, such as the different decoding paths of a single utterance.

    Each candidate is aligned on its own, so its probability, alignment and error
    counts remain available through
    [jiwer.process_candidates][process.process_candidates]. The returned value is the
    sum of `probability * wer` over the candidates.

    Args:
        reference: The reference sentence
        hypotheses: The hypothesis sentence of each candidate
        probabilities: The probability (or log weight) of each candidate. When omitted,
                       all candidates are weighted equally.
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        log_weights: If true, `probabilities` is interpreted as log weights, which
                     may be shifted by an arbitrary common offset (e.g. all lowered
                     by 1000) without underflowing.

    Returns:
        (float): The expected word error rate.

    Example:
        ```python
        import jiwer

        value = jiwer.expected_wer(
            "red green",
            ["red green", "red", "red blue green"],
            [0.25, 0.5, 0.25],
        )
        assert value == 0.375
        ```
    """
    output = process_candidates(
        reference,
        hypotheses,
        probabilities,
        reference_transform,
        hypothesis_transform,
        log_weights,
    )

    return output.wer


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
