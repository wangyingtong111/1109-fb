# Usage

## Word error rate  

The most simple use-case is computing the word error rate between two strings:

```python
from jiwer import wer

reference = "hello world"
hypothesis = "hello duck"

error = wer(reference, hypothesis)
```

Similarly, to get other measures:

```python
import jiwer

reference = "hello world"
hypothesis = "hello duck"

wer = jiwer.wer(reference, hypothesis)
mer = jiwer.mer(reference, hypothesis)
wil = jiwer.wil(reference, hypothesis)

# faster, because `process_words` only needs to perform the heavy lifting once:
output = jiwer.process_words(reference, hypothesis)
wer = output.wer
mer = output.mer
wil = output.wil
```

You can also compute the WER over multiple sentences:

```python
from jiwer import wer

reference = ["hello world", "i like monthy python"]
hypothesis = ["hello duck", "i like python"]

error = wer(reference, hypothesis)
```

## Character error rate

We also provide the character error rate:

```python
import jiwer

reference = ["i can spell", "i hope"]
hypothesis = ["i kan cpell", "i hop"]

error = jiwer.cer(reference, hypothesis)

# if you also want the alignment
output = jiwer.process_characters(reference, hypothesis)
error = output.cer
```

## Alignment

With `jiwer.process_words` and `jiwer.process_characters`, you get the alignment between the reference and hypothesis.

We provide the alignment as a list of `AlignmentChunk` objects with attributes `type, ref_start_idx, ref_end_idx, hyp_start_idx, hyp_end_idx`, where `type` is one of `equal`, `substitute`, `delete`, or `insert`.

This looks like the following:

```python3
import jiwer

out = jiwer.process_words("short one here", "shoe order one")
print(out.alignments)
# [[[AlignmentChunk(type='insert', ref_start_idx=0, ref_end_idx=0, hyp_start_idx=0, hyp_end_idx=1), ...]]
```

To visualize the alignment, you can use `jiwer.visualize_alignment()`

For example:

```python3
import jiwer

out = jiwer.process_words(
    ["short one here", "quite a bit of longer sentence"],
    ["shoe order one", "quite bit of an even longest sentence here"],
)

print(jiwer.visualize_alignment(out))
```
Gives the following output
```text
=== SENTENCE 1 === 

REF: **** short one here
HYP: shoe order one ****
        I     S        D

=== SENTENCE 2 ===

REF: quite a bit of ** ****  longer sentence ****
HYP: quite * bit of an even longest sentence here
           D         I    I       S             I

=== SUMMARY ===
number of sentences: 2
substitutions=2 deletions=2 insertions=4 hits=5

mer=61.54%
wil=74.75%
wip=25.25%
wer=88.89%
```

Note that it also possible to visualize the character-level alignment, simply use the output of `jiwer.process_characters()` instead. 

## Alignment with word timestamps

When the same word is spoken (and recognized) multiple times, a plain edit-distance
alignment cannot tell *which* occurrence was missed. If you know the timestamp of every
word, you can pass them to `jiwer.process_words` (or use the convenience wrapper
`jiwer.process_words_with_timestamps`). The alignment is then chosen with the following
priority:

1. the edit (Levenshtein) distance always remains the first optimization objective,
2. among all alignments with the same edit distance, the one with the smallest sum of
   the absolute timestamp differences over all hit word pairs is chosen,
3. when the timestamp sums are equal as well, a stable, index-based alignment is returned.

The resulting `WordOutput` additionally exposes:

* `hit_pairs`: for each sentence, the `(reference_index, hypothesis_index)` pairs of
  every hit, and
* `delete_intervals`: for each sentence, the half-open
  `(reference_start_index, reference_end_index)` intervals of the deleted reference
  words.

Both use the (transformed) word positions. For example, with a reference containing
`go` at 0 and 5 seconds and a hypothesis only containing `go` at 5 seconds, the first
reference word is reported as the deletion:

```python3
import jiwer

out = jiwer.process_words(
    reference="go go",
    hypothesis="go",
    reference_timestamps=[[0.0, 5.0]],
    hypothesis_timestamps=[[5.0]],
)

print(out.hit_pairs)         # [[(1, 0)]]
print(out.delete_intervals)  # [[(0, 1)]]
```

Without timestamps, `process_words` keeps its original tokenization, edit counts, WER
and alignment order, and the two additional attributes stay empty.

## Error frequencies

You can list all the substitutions, insertions, and deletion, along with their frequencies:

```python3
import jiwer

out = jiwer.process_words(
    ["short one here", "quite a bit of longer sentence"],
    ["shoe order one", "quite bit of an even longest sentence here"],
)

print(jiwer.visualize_error_counts(out))
```

Will return
```text
=== SUBSTITUTIONS ===
short   --> order   = 1x
longer  --> longest = 1x

=== INSERTIONS ===
shoe    = 1x
an even = 1x
here    = 1x

=== DELETIONS ===
here = 1x
a    = 1x
```

## Transformations

You can apply transformations to reference or hypothesis strings before the calculation of various metrics
with the transform API. For all available, transformations, see [here](/jiwer/reference/transforms/).
For the default transformations, see [here](/jiwer/reference/transformations/).

An example of the transformation API:

```python3
import jiwer

tr = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.SubstituteWords({"I'm": 'i am'}),
    jiwer.ReduceToListOfListOfWords()
])

out = jiwer.process_words(
    "I'm good", 
    "i am bad", 
    reference_transform=tr, 
    hypothesis_transform=tr
)
```
