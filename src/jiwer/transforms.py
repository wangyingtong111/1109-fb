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
This file implements the building blocks for transforming a collection
of input strings to the desired format in order to calculate the WER of CER.

In principle, for word error rate calculations, every string of a sentence needs to be
collapsed into a list of strings, where each string is a *single* word.
This is done with [transforms.ReduceToListOfListOfWords][].
A composition of multiple transformations must therefore *always* end with
[transforms.ReduceToListOfListOfWords][].

For the character error rate, every string of a sentence also needs to be collapsed into
a list of strings, but here each string is a *single* character.
This is done with [transforms.ReduceToListOfListOfChars][]. Similarly, a
composition of multiple transformations must therefore also always end with
[transforms.ReduceToListOfListOfChars][].
"""

import functools
import re
import string
import sys
import unicodedata

from typing import Iterable, List, Mapping, Optional, Tuple, Union

__all__ = [
    "AbstractTransform",
    "Compose",
    "ExpandCommonEnglishContractions",
    "ReduceToListOfListOfChars",
    "ReduceToListOfListOfGraphemeClusters",
    "ReduceToListOfListOfGraphemes",
    "ReduceToListOfListOfWords",
    "ReduceToSingleSentence",
    "RemoveEmptyStrings",
    "RemoveKaldiNonWords",
    "RemoveMultipleSpaces",
    "RemovePunctuation",
    "RemoveSpecificWords",
    "RemoveWhiteSpace",
    "Strip",
    "SubstituteRegexes",
    "SubstituteWords",
    "ToLowerCase",
    "ToUpperCase",
    "grapheme_cluster_spans",
    "grapheme_spans",
    "split_grapheme_clusters",
    "split_graphemes",
]


class AbstractTransform(object):
    """
    The base class of a Transform.
    """

    def __call__(self, sentences: Union[str, List[str]]):
        """
        Transforms one or more strings.

        Args:
            sentences: The strings to transform.

        Returns:
            (Union[str, List[str]]): The transformed strings.

        """
        if isinstance(sentences, str):
            return self.process_string(sentences)
        elif isinstance(sentences, list):
            return self.process_list(sentences)
        else:
            raise ValueError(
                "input {} was expected to be a string or list of strings".format(
                    sentences
                )
            )

    def process_string(self, s: str):
        raise NotImplementedError()

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


class Compose(object):
    """
    Chain multiple transformations back-to-back to create a pipeline combining multiple
    transformations.

    Note that each transformation needs to end with either `ReduceToListOfListOfWords`
    or `ReduceToListOfListOfChars`, depending on whether word error rate,
    or character error rate is desired.

    Example:
        ```python3
        import jiwer

        jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords()
        ])
        ```
    """

    def __init__(self, transforms: List[AbstractTransform]):
        """

        Args:
            transforms: The list of transformations to chain.
        """
        self.transforms = transforms

    def __call__(self, text):
        for tr in self.transforms:
            text = tr(text)

        return text


class BaseRemoveTransform(AbstractTransform):
    def __init__(self, tokens_to_remove: Iterable[str], replace_token=""):
        self.tokens_to_remove = tokens_to_remove
        self.replace_token = replace_token

    def process_string(self, s: str):
        for w in self.tokens_to_remove:
            s = s.replace(w, self.replace_token)

        return s

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


class ReduceToListOfListOfWords(AbstractTransform):
    """
    Transforms a single input sentence, or a list of input sentences, into
    a list with lists of words, which is the expected format for calculating the
    edit operations between two input sentences on a word-level.

    A sentence is assumed to be a string, where words are delimited by a token
    (such as ` `, space). Each string is expected to contain only a single sentence.
    Empty strings (no output) are removed for the list.

    Example:
        ```python
        import jiwer

        sentences = ["hi", "this is an example"]

        print(jiwer.ReduceToListOfListOfWords()(sentences))
        # prints: [['hi'], ['this', 'is', 'an, 'example']]
        ```
    """

    def __init__(self, word_delimiter: str = " "):
        """
        Args:
            word_delimiter: the character which delimits words. Default is ` ` (space).
        """
        self.word_delimiter = word_delimiter

    def process_string(self, s: str):
        return [[w for w in s.split(self.word_delimiter) if len(w) >= 1]]

    def process_list(self, inp: List[str]):
        sentence_collection = []

        for sentence in inp:
            list_of_words = self.process_string(sentence)[0]
            sentence_collection.append(list_of_words)

        if len(sentence_collection) == 0:
            return [[]]

        return sentence_collection


class ReduceToListOfListOfChars(AbstractTransform):
    """
    Transforms a single input sentence, or a list of input sentences, into
    a list with lists of characters, which is the expected format for calculating the
    edit operations between two input sentences on a character-level.

    A sentence is assumed to be a string. Each string is expected to contain only a
    single sentence.

    Set `grapheme_clusters=True` to segment Unicode extended grapheme clusters rather
    than Python code points. When `normalize="NFC"` is used in grapheme mode,
    canonically equivalent clusters receive the same comparison representation while
    source spans can still point into the original, unnormalized string.

    Example:
        ```python
        import jiwer

        sentences = ["hi", "this is an example"]

        print(jiwer.ReduceToListOfListOfChars()(sentences))
        # prints: [['h', 'i'], ['t', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', 'n', ' ', 'e', 'x', 'a', 'm', 'p', 'l', 'e']]
        ```
    """

    def __init__(
        self,
        grapheme_clusters: bool = False,
        normalize: Union[str, bool, None] = None,
        **kwargs,
    ):
        # Accept a few descriptive aliases used by callers and integrations.
        aliases = {
            "grapheme": "grapheme_clusters",
            "graphemes": "grapheme_clusters",
            "use_grapheme": "grapheme_clusters",
            "use_graphemes": "grapheme_clusters",
            "use_grapheme_clusters": "grapheme_clusters",
            "extended_grapheme_clusters": "grapheme_clusters",
        }
        values = {"grapheme_clusters": grapheme_clusters}
        normalization = normalize
        for key, value in kwargs.items():
            if key in aliases or "grapheme" in key:
                values["grapheme_clusters"] = bool(value)
            elif key in {"normalization", "normal_form"}:
                normalization = value
            else:
                raise TypeError(
                    f"__init__() got an unexpected keyword argument {key!r}"
                )

        self.grapheme_clusters = values["grapheme_clusters"]
        if normalization is False:
            self.normalize = None
        elif normalization is None and self.grapheme_clusters:
            # Canonical equivalence is part of extended-grapheme comparison. Spans are
            # still computed from the original string, so NFC here does not shift them.
            self.normalize = "NFC"
        else:
            self.normalize = normalization

    def process_string(self, s: str):
        if self.grapheme_clusters:
            units = split_grapheme_clusters(s)
        else:
            units = [char for char in s]

        if self.normalize is not None:
            units = [unicodedata.normalize(self.normalize, unit) for unit in units]

        return [units]

    def process_list(self, inp: List[str]):
        sentence_collection = []

        for sentence in inp:
            list_of_words = self.process_string(sentence)[0]

            sentence_collection.append(list_of_words)

        if len(sentence_collection) == 0:
            return [[]]

        return sentence_collection


class ReduceToListOfListOfGraphemes(ReduceToListOfListOfChars):
    """
    Split each sentence into Unicode extended grapheme clusters.

    Clusters are normalized to NFC by default so decomposed and precomposed
    canonical equivalents compare as the same unit.
    """

    def __init__(self, normalize: Optional[str] = "NFC", **kwargs):
        kwargs.setdefault("grapheme_clusters", True)
        super().__init__(normalize=normalize, **kwargs)


# Descriptive alias.
ReduceToListOfListOfGraphemeClusters = ReduceToListOfListOfGraphemes


# Unicode extended grapheme cluster segmentation (UAX #29).
#
# Python intentionally does not expose a standard-library grapheme splitter. The
# tables below cover the Grapheme_Cluster_Break properties needed for common
# multilingual text, combining-mark sequences, CRLF, Hangul, regional indicators,
# skin-tone modifiers, and ZWJ emoji sequences.

_PREPEND_RANGES = (
    (0x0600, 0x0605),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
)

_EXTEND_CODEPOINTS = frozenset(
    {
        0x200C,  # zero width non-joiner
        0x1F3FB,
        0x1F3FC,
        0x1F3FD,
        0x1F3FE,
        0x1F3FF,
    }
)

_EXTEND_RANGES = (
    (0xFE00, 0xFE0F),  # variation selectors
    (0xE0100, 0xE01EF),  # supplementary variation selectors
)

_EXTENDED_PICTOGRAPHIC_RANGES = (
    (0x00A9, 0x00A9),
    (0x00AE, 0x00AE),
    (0x203C, 0x203C),
    (0x2049, 0x2049),
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x2194, 0x2199),
    (0x21A9, 0x21AA),
    (0x231A, 0x231B),
    (0x2328, 0x2328),
    (0x23CF, 0x23CF),
    (0x23E9, 0x23F3),
    (0x23F8, 0x23FA),
    (0x24C2, 0x24C2),
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x27BF),
    (0x2934, 0x2935),
    (0x2B05, 0x2B07),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F000, 0x1F0FF),
    (0x1F10D, 0x1F10F),
    (0x1F12F, 0x1F12F),
    (0x1F16C, 0x1F171),
    (0x1F17E, 0x1F17F),
    (0x1F18E, 0x1F18E),
    (0x1F191, 0x1F19A),
    (0x1F1AD, 0x1F1E5),
    (0x1F201, 0x1F20F),
    (0x1F21A, 0x1F21A),
    (0x1F22F, 0x1F22F),
    (0x1F232, 0x1F23A),
    (0x1F23C, 0x1F23F),
    (0x1F249, 0x1F53D),
    (0x1F546, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F774, 0x1F77F),
    (0x1F7D5, 0x1F7FF),
    (0x1F80C, 0x1F80F),
    (0x1F848, 0x1F84F),
    (0x1F85A, 0x1F85F),
    (0x1F888, 0x1F88F),
    (0x1F8AE, 0x1F8FF),
    (0x1F90C, 0x1F93A),
    (0x1F93C, 0x1F945),
    (0x1F947, 0x1FAFF),
    (0x1FC00, 0x1FFFD),
)


def _in_ranges(codepoint: int, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    for start, end in ranges:
        if codepoint < start:
            return False
        if codepoint <= end:
            return True
    return False


def _is_regional_indicator(codepoint: int) -> bool:
    return 0x1F1E6 <= codepoint <= 0x1F1FF


def _is_hangul_l(cp: int) -> bool:
    return 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C


def _is_hangul_v(cp: int) -> bool:
    return 0x1160 <= cp <= 0x11A7 or 0xD7B0 <= cp <= 0xD7C6


def _is_hangul_t(cp: int) -> bool:
    return 0x11A8 <= cp <= 0x11FF or 0xD7CB <= cp <= 0xD7FB


def _is_hangul_lv(cp: int) -> bool:
    return 0xAC00 <= cp <= 0xD7A3 and (cp - 0xAC00) % 28 == 0


def _is_hangul_lvt(cp: int) -> bool:
    return 0xAC00 <= cp <= 0xD7A3 and (cp - 0xAC00) % 28 != 0


def _is_extend(codepoint: int) -> bool:
    if codepoint in _EXTEND_CODEPOINTS or _in_ranges(codepoint, _EXTEND_RANGES):
        return True

    char = chr(codepoint)
    return unicodedata.category(char) in {"Mn", "Me"}


def _is_extended_pictographic(codepoint: int) -> bool:
    return _in_ranges(codepoint, _EXTENDED_PICTOGRAPHIC_RANGES)


def _grapheme_break_property(codepoint: int) -> str:
    if codepoint == 0x000D:
        return "CR"
    if codepoint == 0x000A:
        return "LF"
    if codepoint == 0x200D:
        return "ZWJ"
    if _is_regional_indicator(codepoint):
        return "Regional_Indicator"
    if _in_ranges(codepoint, _PREPEND_RANGES):
        return "Prepend"
    if _is_extend(codepoint):
        return "Extend"
    if unicodedata.category(chr(codepoint)) == "Mc":
        return "SpacingMark"
    if unicodedata.category(chr(codepoint)) in {"Cc", "Zl", "Zp"}:
        return "Control"
    return "Other"


def _is_grapheme_boundary(text: str, right_index: int) -> bool:
    """Return whether a UAX #29 boundary precedes `right_index`."""
    if right_index <= 0 or right_index >= len(text):
        return True

    prev_cp = ord(text[right_index - 1])
    next_cp = ord(text[right_index])
    prev_prop = _grapheme_break_property(prev_cp)
    next_prop = _grapheme_break_property(next_cp)

    # GB3: do not break CRLF.
    if prev_prop == "CR" and next_prop == "LF":
        return False

    # GB4/GB5.
    if prev_prop in {"Control", "CR", "LF"}:
        return True
    if next_prop in {"Control", "CR", "LF"}:
        return True

    # GB6-GB8: Hangul syllable composition.
    if _is_hangul_l(prev_cp) and (
        _is_hangul_l(next_cp)
        or _is_hangul_v(next_cp)
        or _is_hangul_lv(next_cp)
        or _is_hangul_lvt(next_cp)
    ):
        return False

    if (_is_hangul_lv(prev_cp) or _is_hangul_v(prev_cp)) and (
        _is_hangul_v(next_cp) or _is_hangul_t(next_cp)
    ):
        return False

    if (_is_hangul_lvt(prev_cp) or _is_hangul_t(prev_cp)) and _is_hangul_t(next_cp):
        return False

    # GB9/GB9a.
    if next_prop in {"Extend", "ZWJ", "SpacingMark"}:
        return False

    # GB9b.
    if prev_prop == "Prepend":
        return False

    # GB11: Extended_Pictographic Extend* ZWJ x Extended_Pictographic.
    if prev_cp == 0x200D and _is_extended_pictographic(next_cp):
        index = right_index - 2
        while index >= 0 and _is_extend(ord(text[index])):
            index -= 1
        if index >= 0 and _is_extended_pictographic(ord(text[index])):
            return False

    # GB12/GB13: keep paired regional indicators together.
    if _is_regional_indicator(prev_cp) and _is_regional_indicator(next_cp):
        count = 0
        index = right_index - 1
        while index >= 0 and _is_regional_indicator(ord(text[index])):
            count += 1
            index -= 1
        if count % 2 == 1:
            return False

    return True


def grapheme_cluster_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return half-open Python code-point spans of UAX #29 grapheme clusters.

    The spans index the original `text`; no normalization is performed before
    segmentation. For example, ``"e\\u0301"`` has span ``(0, 2)`` even though
    NFC normalization would represent the same cluster with one code point.
    """
    if not text:
        return []

    spans: List[Tuple[int, int]] = []
    start = 0

    for boundary in range(1, len(text) + 1):
        if boundary == len(text) or _is_grapheme_boundary(text, boundary):
            spans.append((start, boundary))
            start = boundary

    return spans


def split_grapheme_clusters(text: str) -> List[str]:
    """Split text into extended grapheme clusters without normalization."""
    return [text[start:end] for start, end in grapheme_cluster_spans(text)]


# Short aliases.
grapheme_spans = grapheme_cluster_spans
split_graphemes = split_grapheme_clusters


class ReduceToSingleSentence(AbstractTransform):
    """
    Transforms multiple sentences into a single sentence.
    This operation can be useful when the number of reference and hypothesis sentences
    differ, and you want to do a minimal alignment over these lists.
    Note that this creates an invariance: `wer([a, b], [a, b])` might not be equal to
    `wer([b, a], [b, a])`.

    Example:
        ```python3
        import jiwer

        sentences = ["hi", "this is an example"]

        print(jiwer.ReduceToSingleSentence()(sentences))
        # prints: ['hi this is an example']
        ```
    """

    def __init__(self, word_delimiter: str = " "):
        """
        :param word_delimiter: the character which delimits words. Default is ` ` (space).
        """
        self.word_delimiter = word_delimiter

    def process_string(self, s: str):
        return s

    def process_list(self, inp: List[str]):
        filtered_inp = [i for i in inp if len(i) >= 1]

        if len(filtered_inp) == 0:
            return []
        else:
            return ["{}".format(self.word_delimiter).join(filtered_inp)]


class SubstituteRegexes(AbstractTransform):
    r"""
    Transform strings by substituting substrings matching regex expressions into
    another substring.

    Example:
        ```python
        import jiwer

        sentences = ["is the world doomed or loved?", "edibles are allegedly cultivated"]

        # note: the regex string "\b(\w+)ed\b", matches every word ending in 'ed',
        # and "\1" stands for the first group ('\w+). It therefore removes 'ed' in every match.
        print(jiwer.SubstituteRegexes({r"doom": r"sacr", r"\b(\w+)ed\b": r"\1"})(sentences))

        # prints: ["is the world sacr or lov?", "edibles are allegedly cultivat"]
        ```
    """

    def __init__(self, substitutions: Mapping[str, str]):
        """

        Args:
            substitutions: a mapping of regex expressions to replacement strings.
        """
        self.substitutions = substitutions

    def process_string(self, s: str):
        for key, value in self.substitutions.items():
            s = re.sub(key, value, s)

        return s


class SubstituteWords(AbstractTransform):
    """
    This transform can be used to replace a word into another word.
    Note that the whole word is matched. If the word you're attempting to substitute
    is a substring of another word it will not be affected.
    For example, if you're substituting `foo` into `bar`, the word `foobar` will NOT
    be substituted into `barbar`.

    Example:
        ```python
        import jiwer

        sentences = ["you're pretty", "your book", "foobar"]

        print(jiwer.SubstituteWords({"pretty": "awesome", "you": "i", "'re": " am", 'foo': 'bar'})(sentences))

        # prints: ["i am awesome", "your book", "foobar"]
        ```

    """

    def __init__(self, substitutions: Mapping[str, str]):
        """
        Args:
            substitutions: A mapping of words to replacement words.
        """
        self.substitutions = substitutions

    def process_string(self, s: str):
        for key, value in self.substitutions.items():
            s = re.sub(r"\b{}\b".format(re.escape(key)), value, s)

        return s


class RemoveSpecificWords(SubstituteWords):
    """
    Can be used to filter out certain words.
    As words are replaced with a ` ` character, make sure to that
    `RemoveMultipleSpaces`, `Strip()` and `RemoveEmptyStrings` are present
    in the composition _after_ `RemoveSpecificWords`.

    Example:
        ```python
        import jiwer

        sentences = ["yhe awesome", "the apple is not a pear", "yhe"]

        print(jiwer.RemoveSpecificWords(["yhe", "the", "a"])(sentences))
        # prints: ['  awesome', '  apple is not   pear', ' ']
        # note the extra spaces
        ```
    """

    def __init__(self, words_to_remove: List[str]):
        """
        Args:
            words_to_remove: List of words to remove.
        """
        mapping = {word: " " for word in words_to_remove}

        super().__init__(mapping)


class RemoveWhiteSpace(BaseRemoveTransform):
    """
    This transform filters out white space characters.
    Note that by default space (` `) is also removed, which will make it impossible to
    split a sentence into a list of words by using `ReduceToListOfListOfWords` or
    `ReduceToSingleSentence`.
    This can be prevented by replacing all whitespace with the space character.
    If so, make sure that `jiwer.RemoveMultipleSpaces`,
    `Strip()` and `RemoveEmptyStrings` are present in the composition _after_
    `RemoveWhiteSpace`.

    Example:
        ```python
        import jiwer

        sentences = ["this is an example", "hello world\t"]

        print(jiwer.RemoveWhiteSpace()(sentences))
        # prints: ["thisisanexample", "helloworld"]

        print(jiwer.RemoveWhiteSpace(replace_by_space=True)(sentences))
        # prints: ["this is an example", "hello world  "]
        # note the trailing spaces
        ```
    """

    def __init__(self, replace_by_space: bool = False):
        """

        Args:
            replace_by_space: every white space character is replaced with a space (` `)
        """
        characters = [c for c in string.whitespace]

        if replace_by_space:
            replace_token = " "
        else:
            replace_token = ""

        super().__init__(characters, replace_token=replace_token)


@functools.lru_cache(1)
def _get_punctuation_characters():
    """Compute the punctuation characters only once and memoize."""
    codepoints = range(sys.maxunicode + 1)
    punctuation = set(
        chr(i) for i in codepoints if unicodedata.category(chr(i)).startswith("P")
    )
    return punctuation


class RemovePunctuation(BaseRemoveTransform):
    """
    This transform filters out punctuation. The punctuation characters are defined as
    all unicode characters whose category name starts with `P`.
    See [here](https://www.unicode.org/reports/tr44/#General_Category_Values) for more
    information.
    Example:
        ```python
        import jiwer

        sentences = ["this is an example!", "hello. goodbye"]

        print(jiwer.RemovePunctuation()(sentences))
        # prints: ['this is an example', "hello goodbye"]
        ```
    """

    def __init__(self):
        punctuation_characters = _get_punctuation_characters()
        super().__init__(punctuation_characters)


class RemoveMultipleSpaces(AbstractTransform):
    """
    Filter out multiple spaces between words.

    Example:
        ```python
        import jiwer

        sentences = ["this is   an   example ", "  hello goodbye  ", "  "]

        print(jiwer.RemoveMultipleSpaces()(sentences))
        # prints: ['this is an example ', " hello goodbye ", " "]
        # note that there are still trailing spaces
        ```

    """

    def process_string(self, s: str):
        return re.sub(r"\s\s+", " ", s)

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


class Strip(AbstractTransform):
    """
    Removes all leading and trailing spaces.

    Example:
        ```python
        import jiwer

        sentences = [" this is an example ", "  hello goodbye  ", "  "]

        print(jiwer.Strip()(sentences))
        # prints: ['this is an example', "hello goodbye", ""]
        # note that there is an empty string left behind which might need to be cleaned up
        ```
    """

    def process_string(self, s: str):
        return s.strip()


class RemoveEmptyStrings(AbstractTransform):
    """
    Remove empty strings from a list of strings.

    Example:
        ```python
        import jiwer

        sentences = ["", "this is an example", " ",  "                "]

        print(jiwer.RemoveEmptyStrings()(sentences))
        # prints: ['this is an example']
        ```
    """

    def process_string(self, s: str):
        return s.strip()

    def process_list(self, inp: List[str]):
        return [s for s in inp if self.process_string(s) != ""]


class ExpandCommonEnglishContractions(AbstractTransform):
    """
    Replace common contractions such as `let's` to `let us`.

    Currently, this method will perform the following replacements. Note that `␣` is
     used to indicate a space (` `) to get around markdown rendering constrains.

    | Contraction   | transformed into |
    | ------------- |:----------------:|
    | `won't`       | `␣will not`      |
    | `can't`       | `␣can not`       |
    | `let's`       | `␣let us`        |
    | `n't`         | `␣not`           |
    | `'re`         | `␣are`           |
    | `'s`          | `␣is`            |
    | `'d`          | `␣would`         |
    | `'ll`         | `␣will`          |
    | `'t`          | `␣not`           |
    | `'ve`         | `␣have`          |
    | `'m`          | `␣am`            |

    Example:
        ```python
        import jiwer

        sentences = ["she'll make sure you can't make it", "let's party!"]

        print(jiwer.ExpandCommonEnglishContractions()(sentences))
        # prints: ["she will make sure you can not make it", "let us party!"]
        ```

    """

    def process_string(self, s: str):
        # definitely a non exhaustive list

        # specific words
        s = re.sub(r"won't", "will not", s)
        s = re.sub(r"can\'t", "can not", s)
        s = re.sub(r"let\'s", "let us", s)

        # general attachments
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\'re", " are", s)
        s = re.sub(r"\'s", " is", s)
        s = re.sub(r"\'d", " would", s)
        s = re.sub(r"\'ll", " will", s)
        s = re.sub(r"\'t", " not", s)
        s = re.sub(r"\'ve", " have", s)
        s = re.sub(r"\'m", " am", s)

        return s


class ToLowerCase(AbstractTransform):
    """
    Convert every character into lowercase.
    Example:
        ```python
        import jiwer

        sentences = ["You're PRETTY"]

        print(jiwer.ToLowerCase()(sentences))

        # prints: ["you're pretty"]
        ```
    """

    def process_string(self, s: str):
        return s.lower()


class ToUpperCase(AbstractTransform):
    """
    Convert every character to uppercase.

    Example:
        ```python
        import jiwer

        sentences = ["You're amazing"]

        print(jiwer.ToUpperCase()(sentences))

        # prints: ["YOU'RE AMAZING"]
        ```
    """

    def process_string(self, s: str):
        return s.upper()


class RemoveKaldiNonWords(AbstractTransform):
    """
    Remove any word between `[]` and `<>`. This can be useful when working
    with hypotheses from the Kaldi project, which can output non-words such as
    `[laugh]` and `<unk>`.

    Example:
        ```python
        import jiwer

        sentences = ["you <unk> like [laugh]"]

        print(jiwer.RemoveKaldiNonWords()(sentences))

        # prints: ["you  like "]
        # note the extra spaces
        ```
    """

    def process_string(self, s: str):
        return re.sub(r"[<\[][^>\]]*[>\]]", "", s)
