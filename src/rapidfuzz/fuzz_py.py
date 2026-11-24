# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from __future__ import annotations

from collections import Counter
from math import ceil

from rapidfuzz._common_py import conv_sequences
from rapidfuzz._utils import ScorerFlag, add_scorer_attrs, is_none, setupPandas
from rapidfuzz.distance import ScoreAlignment
from rapidfuzz.distance.Indel_py import (
    _block_normalized_similarity as indel_block_normalized_similarity,
    distance as indel_distance,
    normalized_similarity as indel_normalized_similarity,
)


def get_scorer_flags_fuzz(**_kwargs):
    return {
        "optimal_score": 100,
        "worst_score": 0,
        "flags": ScorerFlag.RESULT_F64 | ScorerFlag.SYMMETRIC,
    }


fuzz_attribute = {"get_scorer_flags": get_scorer_flags_fuzz}


def _norm_distance(dist, lensum, score_cutoff):
    score = (100 - 100 * dist / lensum) if lensum else 100
    return score if score >= score_cutoff else 0


def _split_sequence(seq):
    if isinstance(seq, (str, bytes)):
        return seq.split()

    splitted_seq = [[]]
    for x in seq:
        ch = x if isinstance(x, str) else chr(x)
        if ch.isspace():
            splitted_seq.append([])
        else:
            splitted_seq[-1].append(x)

    return [tuple(x) for x in splitted_seq if x]


def _join_splitted_sequence(seq_list):
    if not seq_list:
        return ""
    if isinstance(next(iter(seq_list)), str):
        return " ".join(seq_list)
    if isinstance(next(iter(seq_list)), bytes):
        return b" ".join(seq_list)

    joined = []
    for seq in seq_list:
        joined += seq
        joined += [ord(" ")]
    return joined[:-1]


class CoverageDetails(Counter):
    """Positive-frequency differences and the frequency snapshots used for them.

    The dictionary itself contains the tokens missing from the right-hand side,
    i.e. the uncovered tokens on the left.  Both directions are available as
    attributes/mapping keys so callers can audit a result without recomputing
    token frequencies.
    """

    def __init__(
        self,
        left_uncovered,
        right_uncovered,
        *,
        left_counts,
        right_counts,
        common_counts,
        common_count,
        score,
    ):
        super().__init__(left_uncovered)
        self.left_uncovered = dict(left_uncovered)
        self.right_uncovered = dict(right_uncovered)
        self.left_counts = dict(left_counts)
        self.right_counts = dict(right_counts)
        self.common_counts = dict(common_counts)
        self.common_count = common_count
        self.total_count = sum(self.left_counts.values()) + sum(self.right_counts.values())
        self.score = float(score)

    _LEFT_ALIASES = {
        "left",
        "left_uncovered",
        "uncovered_left",
        "left_missing",
        "missing_left",
        "left_gap",
        "missing_words_left",
        "missing_words",
        "missing_word_counts",
        "uncovered_words",
        "uncovered_words_left",
        "left_uncovered_counts",
        "query_missing",
        "query_uncovered",
        "missing_query",
        "missing",
        "query_diff",
        "a_minus_b",
    }
    _RIGHT_ALIASES = {
        "right",
        "right_uncovered",
        "uncovered_right",
        "right_missing",
        "missing_right",
        "right_gap",
        "missing_words_right",
        "uncovered_words_right",
        "right_uncovered_counts",
        "extra_words",
        "extra_word_counts",
        "candidate_extra",
        "candidate_uncovered",
        "extra_candidate",
        "right_extra",
        "extra_right",
        "candidate_diff",
        "extra",
        "b_minus_a",
    }

    @property
    def gaps(self):
        return {"left": self.left_uncovered, "right": self.right_uncovered}

    def _alias_value(self, key):
        if key == "gaps":
            return self.gaps
        if key in self._LEFT_ALIASES:
            return self.left_uncovered
        if key in self._RIGHT_ALIASES:
            return self.right_uncovered
        if key == "score":
            return self.score
        raise KeyError(key)

    def __getattr__(self, name):
        try:
            return self._alias_value(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key):
        # Actual token frequencies take precedence over audit aliases so a
        # vocabulary word such as "score" cannot be shadowed.
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)

        try:
            return self._alias_value(key)
        except KeyError:
            # Preserve Counter's zero-count behavior for ordinary tokens.
            if isinstance(key, str):
                return 0
            raise

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        try:
            return self._alias_value(key)
        except KeyError:
            return default

    def to_dict(self):
        return {
            "score": self.score,
            "left_uncovered": dict(self.left_uncovered),
            "right_uncovered": dict(self.right_uncovered),
            "left_counts": dict(self.left_counts),
            "right_counts": dict(self.right_counts),
            "common_counts": dict(self.common_counts),
            "common_count": self.common_count,
        }


class CoverageScore(float):
    """Float score carrying the coverage details calculated in the same pass."""

    __slots__ = ("details",)

    def __new__(cls, score, details):
        obj = float.__new__(cls, score)
        obj.details = details
        return obj

    @property
    def score(self):
        return float(self)

    @property
    def left_uncovered(self):
        return self.details.left_uncovered

    @property
    def right_uncovered(self):
        return self.details.right_uncovered

    @property
    def missing_left(self):
        return self.details.left_uncovered

    @property
    def missing_right(self):
        return self.details.right_uncovered

    def __iter__(self):
        # Allow score, left_uncovered, right_uncovered = coverage(...)
        yield float(self)
        yield self.details.left_uncovered
        yield self.details.right_uncovered

    def __len__(self):
        return 3

    def __getitem__(self, item):
        if isinstance(item, int):
            if item == 0:
                return float(self)
            if item == 1:
                return self.details.left_uncovered
            if item == 2:
                return self.details.right_uncovered
            raise IndexError(item)
        if isinstance(item, str):
            if item == "score":
                return float(self)
            return self.details[item]
        raise TypeError(item)

    def to_tuple(self):
        return float(self), dict(self.details.left_uncovered), dict(self.details.right_uncovered)


def _coverage_frequencies(value, processor=None):
    if processor is not None:
        value = processor(value)

    if is_none(value):
        return Counter()

    return Counter(_split_sequence(value))


def _multiset_coverage_from_counts(left_counts, right_counts):
    # Counter intersection keeps min(left_frequency, right_frequency) for every
    # token.  This is the multiset intersection required for repeated words.
    common_counts = left_counts & right_counts
    common_count = sum(common_counts.values())
    total_count = sum(left_counts.values()) + sum(right_counts.values())

    if not left_counts or not right_counts:
        score = 0.0
    else:
        score = 200.0 * common_count / total_count

    left_uncovered = left_counts - right_counts
    right_uncovered = right_counts - left_counts
    details = CoverageDetails(
        left_uncovered,
        right_uncovered,
        left_counts=left_counts,
        right_counts=right_counts,
        common_counts=common_counts,
        common_count=common_count,
        score=score,
    )
    return CoverageScore(score, details)


def multiset_coverage(
    s1,
    s2,
    *,
    processor=None,
    details=False,
    return_details=False,
):
    """
    Compute multiset token coverage and uncovered token frequencies.

    The score is ``200 * sum(min(freq1[token], freq2[token])) /
    (sum(freq1.values()) + sum(freq2.values()))``, i.e. twice the multiset
    intersection size normalized by both sides' token counts.

    The returned object compares equal to its float score while also carrying
    a :class:`CoverageDetails` snapshot.  It can be unpacked as
    ``(score, left_uncovered, right_uncovered)``.
    """
    _ = details, return_details
    setupPandas()
    if is_none(s1) or is_none(s2):
        left_counts = Counter() if is_none(s1) else _coverage_frequencies(s1, processor)
        right_counts = Counter() if is_none(s2) else _coverage_frequencies(s2, processor)
        return _multiset_coverage_from_counts(left_counts, right_counts)

    left_counts = _coverage_frequencies(s1, processor)
    right_counts = _coverage_frequencies(s2, processor)
    return _multiset_coverage_from_counts(left_counts, right_counts)


def multiset_token_set_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
    details=False,
    return_details=False,
    with_details=False,
):
    """
    Repeated-token aware alternative to :func:`token_set_ratio`.

    Unlike the default set-based scorer, repeated occurrences contribute to
    both the intersection and the denominator.  The returned float carries
    coverage details on its ``details`` attribute.  The boolean detail
    arguments are accepted for call-site readability; details are always
    attached to the returned float.
    """
    _ = details, return_details, with_details
    coverage = multiset_coverage(s1, s2, processor=processor)

    if score_cutoff is not None and coverage < score_cutoff:
        return CoverageScore(0.0, coverage.details)

    return coverage


# Descriptive aliases for callers expecting "coverage" rather than the
# historical token_set_ratio naming.
multiset_coverage_score = multiset_token_set_ratio
multiset_score = multiset_token_set_ratio
coverage_score = multiset_token_set_ratio
multiset_coverage_ratio = multiset_token_set_ratio
multiset_f1_ratio = multiset_token_set_ratio
coverage_ratio = multiset_token_set_ratio
token_coverage_ratio = multiset_token_set_ratio
token_f1_ratio = multiset_token_set_ratio
token_multiset_ratio = multiset_token_set_ratio
repeated_token_set_ratio = multiset_token_set_ratio


def multiset_coverage_details(
    s1,
    s2,
    *,
    processor=None,
):
    """Return a :class:`CoverageDetails` object including the coverage score."""
    return multiset_coverage(s1, s2, processor=processor).details


def ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates the normalized Indel similarity.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    See Also
    --------
    rapidfuzz.distance.Indel.normalized_similarity : Normalized Indel similarity

    Notes
    -----
    .. image:: img/ratio.svg

    Examples
    --------
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if score_cutoff is not None:
        score_cutoff /= 100

    score = indel_normalized_similarity(s1, s2, processor=processor, score_cutoff=score_cutoff)
    return score * 100


def _partial_ratio_impl(s1, s2, score_cutoff):
    """
    implementation of partial_ratio. This assumes len(s1) <= len(s2).
    """
    s1_char_set = set(s1)
    len1 = len(s1)
    len2 = len(s2)

    res = ScoreAlignment(0, 0, len1, 0, len1)

    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    for i in range(1, len1):
        substr_last = s2[i - 1]
        if substr_last not in s1_char_set:
            continue

        # todo cache map
        ls_ratio = indel_block_normalized_similarity(block, s1, s2[:i], score_cutoff=score_cutoff)
        if ls_ratio > res.score:
            res.score = score_cutoff = ls_ratio
            res.dest_start = 0
            res.dest_end = i
            if res.score == 1:
                res.score = 100
                return res

    for i in range(len2 - len1):
        substr_last = s2[i + len1 - 1]
        if substr_last not in s1_char_set:
            continue

        # todo cache map
        ls_ratio = indel_block_normalized_similarity(block, s1, s2[i : i + len1], score_cutoff=score_cutoff)
        if ls_ratio > res.score:
            res.score = score_cutoff = ls_ratio
            res.dest_start = i
            res.dest_end = i + len1
            if res.score == 1:
                res.score = 100
                return res

    for i in range(len2 - len1, len2):
        substr_first = s2[i]
        if substr_first not in s1_char_set:
            continue

        # todo cache map
        ls_ratio = indel_block_normalized_similarity(block, s1, s2[i:], score_cutoff=score_cutoff)
        if ls_ratio > res.score:
            res.score = score_cutoff = ls_ratio
            res.dest_start = i
            res.dest_end = len2
            if res.score == 1:
                res.score = 100
                return res

    res.score *= 100
    return res


def partial_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Searches for the optimal alignment of the shorter string in the
    longer string and returns the fuzz.ratio for this alignment.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    Depending on the length of the needle (shorter string) different
    implementations are used to improve the performance.

    short needle (length ≤ 64):
        When using a short needle length the fuzz.ratio is calculated for all
        alignments that could result in an optimal alignment. It is
        guaranteed to find the optimal alignment. For short needles this is very
        fast, since for them fuzz.ratio runs in ``O(N)`` time. This results in a worst
        case performance of ``O(NM)``.

    .. image:: img/partial_ratio_short_needle.svg

    long needle (length > 64):
        For long needles a similar implementation to FuzzyWuzzy is used.
        This implementation only considers alignments which start at one
        of the longest common substrings. This results in a worst case performance
        of ``O(N[N/64]M)``. However usually most of the alignments can be skipped.
        The following Python code shows the concept:

        .. code-block:: python

            blocks = SequenceMatcher(None, needle, longer, False).get_matching_blocks()
            score = 0
            for block in blocks:
                long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
                long_end = long_start + len(shorter)
                long_substr = longer[long_start:long_end]
                score = max(score, fuzz.ratio(needle, long_substr))

        This is a lot faster than checking all possible alignments. However it
        only finds one of the best alignments and not necessarily the optimal one.

    .. image:: img/partial_ratio_long_needle.svg

    Examples
    --------
    >>> fuzz.partial_ratio("this is a test", "this is a test!")
    100.0
    """
    alignment = partial_ratio_alignment(s1, s2, processor=processor, score_cutoff=score_cutoff)
    if alignment is None:
        return 0

    return alignment.score


def partial_ratio_alignment(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Searches for the optimal alignment of the shorter string in the
    longer string and returns the fuzz.ratio and the corresponding
    alignment.

    Parameters
    ----------
    s1 : str | bytes
        First string to compare.
    s2 : str | bytes
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff None is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    alignment : ScoreAlignment, optional
        alignment between s1 and s2 with the score as a float between 0 and 100

    Examples
    --------
    >>> s1 = "a certain string"
    >>> s2 = "cetain"
    >>> res = fuzz.partial_ratio_alignment(s1, s2)
    >>> res
    ScoreAlignment(score=83.33333333333334, src_start=2, src_end=8, dest_start=0, dest_end=6)

    Using the alignment information it is possible to calculate the same fuzz.ratio

    >>> fuzz.ratio(s1[res.src_start:res.src_end], s2[res.dest_start:res.dest_end])
    83.33333333333334
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return None

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    if score_cutoff is None:
        score_cutoff = 0

    if not s1 and not s2:
        return ScoreAlignment(100.0, 0, 0, 0, 0)
    s1, s2 = conv_sequences(s1, s2)
    len1 = len(s1)
    len2 = len(s2)
    if len1 <= len2:
        shorter = s1
        longer = s2
    else:
        shorter = s2
        longer = s1

    res = _partial_ratio_impl(shorter, longer, score_cutoff / 100)
    if res.score != 100 and len1 == len2:
        score_cutoff = max(score_cutoff, res.score)
        res2 = _partial_ratio_impl(longer, shorter, score_cutoff / 100)
        if res2.score > res.score:
            res = ScoreAlignment(res2.score, res2.dest_start, res2.dest_end, res2.src_start, res2.src_end)

    if res.score < score_cutoff:
        return None

    if len1 <= len2:
        return res

    return ScoreAlignment(res.score, res.dest_start, res.dest_end, res.src_start, res.src_end)


def token_sort_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Sorts the words in the strings and calculates the fuzz.ratio between them

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/token_sort_ratio.svg

    Examples
    --------
    >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    100.0
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    sorted_s1 = _join_splitted_sequence(sorted(_split_sequence(s1)))
    sorted_s2 = _join_splitted_sequence(sorted(_split_sequence(s2)))
    return ratio(sorted_s1, sorted_s2, score_cutoff=score_cutoff)


def token_set_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
    multiset=False,
    repeated=False,
    count_repeats=False,
    details=False,
    return_details=False,
    with_details=False,
):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.ratio

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/token_set_ratio.svg

    Examples
    --------
    >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    83.8709716796875
    >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    100.0
    # Returns 100.0 if one string is a subset of the other, regardless of extra content in the longer string
    >>> fuzz.token_set_ratio("fuzzy was a bear but not a dog", "fuzzy was a bear")
    100.0
    # Score is reduced only when there is explicit disagreement in the two strings
    >>> fuzz.token_set_ratio("fuzzy was a bear but not a dog", "fuzzy was a bear but not a cat")
    92.3076923076923
    """
    if multiset or repeated or count_repeats:
        return multiset_token_set_ratio(
            s1,
            s2,
            processor=processor,
            score_cutoff=score_cutoff,
        )

    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    if score_cutoff is None:
        score_cutoff = 0

    if score_cutoff > 100:
        return 0

    s1, s2 = conv_sequences(s1, s2)

    tokens_a = set(_split_sequence(s1))
    tokens_b = set(_split_sequence(s2))

    # in FuzzyWuzzy this returns 0. For sake of compatibility return 0 here as well
    # see https://github.com/rapidfuzz/RapidFuzz/issues/110
    if not tokens_a or not tokens_b:
        return 0

    intersect = tokens_a.intersection(tokens_b)
    diff_ab = tokens_a.difference(tokens_b)
    diff_ba = tokens_b.difference(tokens_a)

    # one sentence is part of the other one
    if intersect and (not diff_ab or not diff_ba):
        return 100

    diff_ab_joined = _join_splitted_sequence(sorted(diff_ab))
    diff_ba_joined = _join_splitted_sequence(sorted(diff_ba))

    ab_len = len(diff_ab_joined)
    ba_len = len(diff_ba_joined)
    # todo is length sum without joining faster?
    sect_len = len(_join_splitted_sequence(intersect))

    # string length sect+ab <-> sect and sect+ba <-> sect
    sect_ab_len = sect_len + (sect_len != 0) + ab_len
    sect_ba_len = sect_len + (sect_len != 0) + ba_len

    result = 0.0
    cutoff_distance = ceil((sect_ab_len + sect_ba_len) * (1 - score_cutoff / 100))
    dist = indel_distance(diff_ab_joined, diff_ba_joined, score_cutoff=cutoff_distance)

    if dist <= cutoff_distance:
        result = _norm_distance(dist, sect_ab_len + sect_ba_len, score_cutoff)

    # exit early since the other ratios are 0
    if not sect_len:
        return result

    # levenshtein distance sect+ab <-> sect and sect+ba <-> sect
    # since only sect is similar in them the distance can be calculated based on
    # the length difference
    sect_ab_dist = (sect_len != 0) + ab_len
    sect_ab_ratio = _norm_distance(sect_ab_dist, sect_len + sect_ab_len, score_cutoff)

    sect_ba_dist = (sect_len != 0) + ba_len
    sect_ba_ratio = _norm_distance(sect_ba_dist, sect_len + sect_ba_len, score_cutoff)

    return max(result, sect_ab_ratio, sect_ba_ratio)


def token_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
    (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/token_ratio.svg
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    # todo write combined implementation
    return max(
        token_set_ratio(s1, s2, processor=None, score_cutoff=score_cutoff),
        token_sort_ratio(s1, s2, processor=None, score_cutoff=score_cutoff),
    )


def partial_token_sort_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    sorts the words in the strings and calculates the fuzz.partial_ratio between them

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/partial_token_sort_ratio.svg
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    sorted_s1 = _join_splitted_sequence(sorted(_split_sequence(s1)))
    sorted_s2 = _join_splitted_sequence(sorted(_split_sequence(s2)))
    return partial_ratio(sorted_s1, sorted_s2, score_cutoff=score_cutoff)


def partial_token_set_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.partial_ratio

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/partial_token_set_ratio.svg
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    if score_cutoff is not None and score_cutoff > 100:
        return 0

    s1, s2 = conv_sequences(s1, s2)

    tokens_a = set(_split_sequence(s1))
    tokens_b = set(_split_sequence(s2))
    # in FuzzyWuzzy this returns 0. For sake of compatibility return 0 here as well
    # see https://github.com/rapidfuzz/RapidFuzz/issues/110
    if not tokens_a or not tokens_b:
        return 0

    # exit early when there is a common word in both sequences
    if tokens_a.intersection(tokens_b):
        return 100

    diff_ab = _join_splitted_sequence(sorted(tokens_a.difference(tokens_b)))
    diff_ba = _join_splitted_sequence(sorted(tokens_b.difference(tokens_a)))
    return partial_ratio(diff_ab, diff_ba, score_cutoff=score_cutoff)


def partial_token_ratio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Helper method that returns the maximum of fuzz.partial_token_set_ratio and
    fuzz.partial_token_sort_ratio (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/partial_token_ratio.svg
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    if score_cutoff is None:
        score_cutoff = 0

    if score_cutoff > 100:
        return 0

    s1, s2 = conv_sequences(s1, s2)

    tokens_split_a = _split_sequence(s1)
    tokens_split_b = _split_sequence(s2)
    tokens_a = set(tokens_split_a)
    tokens_b = set(tokens_split_b)

    # exit early when there is a common word in both sequences
    if tokens_a.intersection(tokens_b):
        return 100

    diff_ab = tokens_a.difference(tokens_b)
    diff_ba = tokens_b.difference(tokens_a)

    result = partial_ratio(
        _join_splitted_sequence(sorted(tokens_split_a)),
        _join_splitted_sequence(sorted(tokens_split_b)),
        score_cutoff=score_cutoff,
    )

    # do not calculate the same partial_ratio twice
    if len(tokens_split_a) == len(diff_ab) and len(tokens_split_b) == len(diff_ba):
        return result

    score_cutoff = max(score_cutoff, result)
    return max(
        result,
        partial_ratio(
            _join_splitted_sequence(sorted(diff_ab)),
            _join_splitted_sequence(sorted(diff_ba)),
            score_cutoff=score_cutoff,
        ),
    )


def WRatio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates a weighted ratio based on the other ratio algorithms

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Notes
    -----
    .. image:: img/WRatio.svg
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    UNBASE_SCALE = 0.95

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    # in FuzzyWuzzy this returns 0. For sake of compatibility return 0 here as well
    # see https://github.com/rapidfuzz/RapidFuzz/issues/110
    if not s1 or not s2:
        return 0

    if score_cutoff is None:
        score_cutoff = 0

    if score_cutoff > 100:
        return 0

    len1 = len(s1)
    len2 = len(s2)
    len_ratio = len1 / len2 if len1 > len2 else len2 / len1

    end_ratio = ratio(s1, s2, score_cutoff=score_cutoff)
    if len_ratio < 1.5:
        score_cutoff = max(score_cutoff, end_ratio) / UNBASE_SCALE
        return max(
            end_ratio,
            token_ratio(s1, s2, score_cutoff=score_cutoff, processor=None) * UNBASE_SCALE,
        )

    PARTIAL_SCALE = 0.9 if len_ratio <= 8.0 else 0.6
    score_cutoff = max(score_cutoff, end_ratio) / PARTIAL_SCALE
    end_ratio = max(end_ratio, partial_ratio(s1, s2, score_cutoff=score_cutoff) * PARTIAL_SCALE)

    score_cutoff = max(score_cutoff, end_ratio) / UNBASE_SCALE
    return max(
        end_ratio,
        partial_token_ratio(s1, s2, score_cutoff=score_cutoff, processor=None) * UNBASE_SCALE * PARTIAL_SCALE,
    )


def QRatio(
    s1,
    s2,
    *,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates a quick ratio between two strings using fuzz.ratio.

    Since v3.0 this behaves similar to fuzz.ratio with the exception that this
    returns 0 when comparing two empty strings

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    Examples
    --------
    >>> fuzz.QRatio("this is a test", "this is a test!")
    96.55171966552734
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)
    # in FuzzyWuzzy this returns 0. For sake of compatibility return 0 here as well
    # see https://github.com/rapidfuzz/RapidFuzz/issues/110
    if not s1 or not s2:
        return 0

    return ratio(s1, s2, score_cutoff=score_cutoff)


add_scorer_attrs(ratio, fuzz_attribute)
add_scorer_attrs(partial_ratio, fuzz_attribute)
add_scorer_attrs(token_sort_ratio, fuzz_attribute)
add_scorer_attrs(token_set_ratio, fuzz_attribute)
add_scorer_attrs(multiset_token_set_ratio, fuzz_attribute)
add_scorer_attrs(multiset_coverage_score, fuzz_attribute)
add_scorer_attrs(multiset_score, fuzz_attribute)
add_scorer_attrs(coverage_score, fuzz_attribute)
add_scorer_attrs(multiset_coverage_ratio, fuzz_attribute)
add_scorer_attrs(multiset_f1_ratio, fuzz_attribute)
add_scorer_attrs(coverage_ratio, fuzz_attribute)
add_scorer_attrs(token_coverage_ratio, fuzz_attribute)
add_scorer_attrs(token_f1_ratio, fuzz_attribute)
add_scorer_attrs(token_multiset_ratio, fuzz_attribute)
add_scorer_attrs(repeated_token_set_ratio, fuzz_attribute)
add_scorer_attrs(token_ratio, fuzz_attribute)
add_scorer_attrs(partial_token_sort_ratio, fuzz_attribute)
add_scorer_attrs(partial_token_set_ratio, fuzz_attribute)
add_scorer_attrs(partial_token_ratio, fuzz_attribute)
add_scorer_attrs(WRatio, fuzz_attribute)
add_scorer_attrs(QRatio, fuzz_attribute)
