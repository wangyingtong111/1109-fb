# JiWER

JiWER is a simple and fast python package to evaluate an automatic speech recognition system.
It supports the following measures:

1. word error rate (WER)
2. match error rate (MER)
3. word information lost (WIL) 
4. word information preserved (WIP) 
5. character error rate (CER)

These measures are computed with the use of the minimum-edit distance between one or more reference and hypothesis sentences.
The minimum-edit distance is calculated using [RapidFuzz](https://github.com/maxbachmann/RapidFuzz), which uses C++ under the hood, and is therefore faster than a pure python implementation.

## Documentation

For further info, see the documentation at [jitsi.github.io/jiwer](https://jitsi.github.io/jiwer).

## Installation

You should be able to install this package using [uv](https://docs.astral.sh/uv/): 

```
$ uv add jiwer
```

Or, if you prefer old-fashioned pip and you're using Python >= `3.8`:

```bash
$ pip install jiwer
```

## Usage

The most simple use-case is computing the word error rate between two strings:

```python
from jiwer import wer

reference = "hello world"
hypothesis = "hello duck"

error = wer(reference, hypothesis)
```

## Multiple references per sentence

A single recording may have more than one acceptable transcription. JiWER
accepts a nested list of references, where every sentence gets a list of
candidate transcriptions. One complete transcription is selected per sentence,
and the selected index is returned together with the regular alignment chunks.

```python3
import jiwer

# both references differ in one word, so the first one is selected
out = jiwer.process_words([["east gate", "west door"]], ["east door"])
assert out.reference_indices == [0]

# select the combination of references that minimizes the WER of the whole batch
references = [["a", "a b c d"], ["w x"]]
hypotheses = ["a b", "w y"]
out = jiwer.process_words(references, hypotheses, method="wer")
assert out.reference_indices == [1, 0]
assert out.wer == 1 / 2
```

Use `method="edit"` (the default) to pick the minimum-edit reference of each
sentence independently, or `method="wer"` to minimize the WER over the whole
batch. In batch mode the total number of words in the selected references is
used as the denominator, zero references follow the existing convention, and
ties are broken by the order in which the candidates are provided. The
dedicated `jiwer.process_words_multi_reference` and
`jiwer.wer_multi_reference` entry points provide the same functionality, while
the original one-to-one string/list entry point remains unchanged.

## A note on empty references

There is undefined behaviour when you apply an empty reference and hypothesis pair
to the WER formula, as you get a division by zero.

As of version 4.0, `jiwer` defines the behaviour as follows, and thereby also
lifts the requirement for reference strings to be non-empty.
This allows for testing whether models hallucinate on silent audio.
Note that usually, there are multiple reference and hypothesis pairs.
It now supported that one or more of these references are empty, but to test well,
most references should still be non-empty.

```python3
import jiwer

# when ref and hyp are both empty, there is no error as
# an ASR system correctly predicted silence/non-speech.
assert jiwer.wer('', '') == 0 
assert jiwer.mer('', '') == 0
assert jiwer.wip('', '') == 1
assert jiwer.wil('', '') == 0

assert jiwer.cer('', '') == 0
```

When the hypothesis is non-empty, every word or character counts as an insertion:
```python3
import jiwer

assert jiwer.wer('', 'silence') == 1
assert jiwer.wer('', 'peaceful silence') == 2
assert jiwer.process_words('', 'now defined behaviour').insertions == 3

assert jiwer.cer('', 'a') == 1
assert jiwer.cer('', 'abcde') == 5
```

## Licence

The jiwer package is released under the `Apache License, Version 2.0` licence by [8x8](https://www.8x8.com/).

For further information, see [`LICENCE`](./LICENSE).

## Reference

_For a comparison between WER, MER and WIL, see: \
Morris, Andrew & Maier, Viktoria & Green, Phil. (2004). [From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.](https://www.researchgate.net/publication/221478089_From_WER_and_RIL_to_MER_and_WIL_improved_evaluation_measures_for_connected_speech_recognition)_
