# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from __future__ import annotations

from rapidfuzz._common_py import conv_sequences
from rapidfuzz._utils import is_none, setupPandas
from rapidfuzz.distance._initialize_py import Editop, Editops

_WILDCARD_LENGTH_ERROR = "wildcard has to be a single character."


def _conv_wildcard(wildcard, s1, s2):
    """
    Bring the sentinel into the element domain of the already converted sequences.

    ``conv_sequences`` keeps two ``str`` inputs as strings and two ``bytes`` inputs as
    bytes, while every other input is a sequence of integer element identities. The
    sentinel therefore has to be converted the same way the elements are.
    """
    if isinstance(s1, str) and isinstance(s2, str):
        if isinstance(wildcard, str) and len(wildcard) == 1:
            return wildcard
        if isinstance(wildcard, (bytes, bytearray)) and len(wildcard) == 1:
            return chr(wildcard[0])
        if isinstance(wildcard, int) and not isinstance(wildcard, bool):
            return chr(wildcard)
        raise ValueError(_WILDCARD_LENGTH_ERROR)

    if isinstance(s1, (bytes, bytearray)) and isinstance(s2, (bytes, bytearray)):
        if isinstance(wildcard, (bytes, bytearray)) and len(wildcard) == 1:
            return wildcard[0]
        if isinstance(wildcard, str) and len(wildcard) == 1:
            return ord(wildcard)
        if isinstance(wildcard, int) and not isinstance(wildcard, bool):
            return wildcard
        raise ValueError(_WILDCARD_LENGTH_ERROR)

    if isinstance(wildcard, str):
        if len(wildcard) != 1:
            raise ValueError(_WILDCARD_LENGTH_ERROR)
        return ord(wildcard)

    if isinstance(wildcard, (bytes, bytearray)):
        if len(wildcard) != 1:
            raise ValueError(_WILDCARD_LENGTH_ERROR)
        return wildcard[0]

    if isinstance(wildcard, int) and not isinstance(wildcard, bool):
        return wildcard

    return hash(wildcard)


def _masked_prepare(s1, s2, wildcard, processor, pad):
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)

    if not pad and len(s1) != len(s2):
        msg = "Sequences are not the same length."
        raise ValueError(msg)

    return s1, s2, _conv_wildcard(wildcard, s1, s2)


def _masked_stats(s1, s2, wildcard):
    """
    Compare both sequences at their original positions.

    Only positions where both sides carry a real value take part, so a position that
    is masked in the query, in the candidate or in both is skipped instead of being
    counted as an equal or as a differing character. Padding is not a real value, so
    the comparison stops at the shorter sequence.
    """
    matches = 0
    comparable = 0
    for elem1, elem2 in zip(s1, s2):
        if elem1 == wildcard or elem2 == wildcard:
            continue

        comparable += 1
        matches += elem1 == elem2

    return matches, comparable


def comparable_count(
    s1,
    s2,
    *,
    wildcard="?",
    processor=None,
):
    """
    Number of positions that are compared in the masked Hamming mode.

    A position is comparable when both sequences contain a real value there, i.e.
    neither side is the ``wildcard`` sentinel. Padding is not a real value either, so
    only the overlapping part of both sequences is considered.

    This is the coverage that ``process.extract`` reports and uses for its threshold
    and for breaking ties between equal scores.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    wildcard : str, bytes or int, optional
        Sentinel characters marking positions without a real value. Default is ``"?"``.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    int
        amount of positions where both sequences carry a real value

    Examples
    --------
    >>> from rapidfuzz.distance import Hamming
    >>> Hamming.comparable_count("a?b", "axb")
    2
    >>> Hamming.comparable_count("a?b", "bxa")
    2
    >>> Hamming.comparable_count("a?b", "a?b")
    2
    """
    s1, s2, sentinel = _masked_prepare(s1, s2, wildcard, processor, True)
    _, comparable = _masked_stats(s1, s2, sentinel)
    return comparable


def comparable_positions(
    s1,
    s2,
    *,
    wildcard="?",
    processor=None,
):
    """
    Positions that are compared in the masked Hamming mode.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    wildcard : str, bytes or int, optional
        Sentinel characters marking positions without a real value. Default is ``"?"``.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    tuple[int, ...]
        ascending positions where both sequences carry a real value

    Examples
    --------
    >>> from rapidfuzz.distance import Hamming
    >>> Hamming.comparable_positions("a?b", "axb")
    (0, 2)
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    sentinel = _conv_wildcard(wildcard, s1, s2)

    return tuple(
        i
        for i, (elem1, elem2) in enumerate(zip(s1, s2))
        if elem1 != sentinel and elem2 != sentinel
    )


def distance(
    s1,
    s2,
    *,
    pad=True,
    wildcard=None,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates the Hamming distance between two strings.
    The hamming distance is defined as the number of positions
    where the two strings differ. It describes the minimum
    amount of substitutions required to transform s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    pad : bool, optional
       should strings be padded if there is a length difference.
       If pad is False and strings have a different length
       a ValueError is thrown instead. Defaults is True.
    wildcard : str, bytes or int, optional
        Enables the masked Hamming mode and marks positions without a real value.
        Only positions where both strings carry a real value are compared, so the
        sentinel is never counted as an equal or as a differing character. Positions
        beyond the shorter sequence carry a value on one side only and are skipped as
        well, therefore they are not part of a padded distance either. Default is
        None, which deactivates this behaviour and compares a literal ``"?"`` as a
        normal character.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int or None, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than score_cutoff,
        score_cutoff + 1 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If s1 and s2 have a different length while ``pad`` is False or the wildcard is
        not a single character.

    Examples
    --------
    >>> from rapidfuzz.distance import Hamming
    >>> Hamming.distance("a?b", "axb", wildcard="?")
    0
    >>> Hamming.distance("a?b", "bxa", wildcard="?")
    2
    """
    if wildcard is not None:
        s1, s2, sentinel = _masked_prepare(s1, s2, wildcard, processor, pad)
        matches, comparable = _masked_stats(s1, s2, sentinel)
        dist = comparable - matches
        return dist if (score_cutoff is None or dist <= score_cutoff) else score_cutoff + 1

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)

    if not pad and len(s1) != len(s2):
        msg = "Sequences are not the same length."
        raise ValueError(msg)

    min_len = min(len(s1), len(s2))
    dist = max(len(s1), len(s2))
    for i in range(min_len):
        dist -= s1[i] == s2[i]

    return dist if (score_cutoff is None or dist <= score_cutoff) else score_cutoff + 1


def similarity(
    s1,
    s2,
    *,
    pad=True,
    wildcard=None,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates the Hamming similarity between two strings.

    This is calculated as ``len1 - distance``. In the masked Hamming mode it is the
    amount of comparable positions holding the same value.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    pad : bool, optional
       should strings be padded if there is a length difference.
       If pad is False and strings have a different length
       a ValueError is thrown instead. Defaults is True.
    wildcard : str, bytes or int, optional
        Enables the masked Hamming mode and marks positions without a real value.
        Only positions where both strings carry a real value are compared. Default is
        None, which deactivates this behaviour.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the similarity is smaller than score_cutoff,
        0 is returned instead. Default is None, which deactivates
        this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If s1 and s2 have a different length while ``pad`` is False

    Examples
    --------
    >>> from rapidfuzz.distance import Hamming
    >>> Hamming.similarity("a?b", "axb", wildcard="?")
    2
    >>> Hamming.similarity("a?b", "bxa", wildcard="?")
    0
    """
    if wildcard is not None:
        s1, s2, sentinel = _masked_prepare(s1, s2, wildcard, processor, pad)
        matches, _ = _masked_stats(s1, s2, sentinel)
        return matches if (score_cutoff is None or matches >= score_cutoff) else 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    maximum = max(len(s1), len(s2))
    dist = distance(s1, s2, pad=pad)
    sim = maximum - dist

    return sim if (score_cutoff is None or sim >= score_cutoff) else 0


def normalized_distance(
    s1,
    s2,
    *,
    pad=True,
    wildcard=None,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates a normalized Hamming similarity in the range [1, 0].

    This is calculated as ``distance / (len1 + len2)``. In the masked Hamming mode the
    distance is divided by the amount of comparable positions. A pair without any
    comparable position is reported as a maximal distance of 1.0 instead of a perfect
    match.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    pad : bool, optional
       should strings be padded if there is a length difference.
       If pad is False and strings have a different length
       a ValueError is thrown instead. Defaults is True.
    wildcard : str, bytes or int, optional
        Enables the masked Hamming mode and marks positions without a real value.
        Default is None, which deactivates this behaviour.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For norm_dist > score_cutoff 1.0 is returned instead. Default is 1.0,
        which deactivates this behaviour.

    Returns
    -------
    norm_dist : float
        normalized distance between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If s1 and s2 have a different length while ``pad`` is False

    Examples
    --------
    >>> from rapidfuzz.distance import Hamming
    >>> Hamming.normalized_distance("a?b", "axb", wildcard="?")
    0.0
    >>> Hamming.normalized_distance("a?b", "bxa", wildcard="?")
    1.0
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 1.0

    if wildcard is not None:
        s1, s2, sentinel = _masked_prepare(s1, s2, wildcard, processor, pad)
        matches, comparable = _masked_stats(s1, s2, sentinel)
        norm_dist = (comparable - matches) / comparable if comparable else 1.0
        return norm_dist if (score_cutoff is None or norm_dist <= score_cutoff) else 1.0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    maximum = max(len(s1), len(s2))
    dist = distance(s1, s2, pad=pad)
    norm_dist = dist / maximum if maximum else 0

    return norm_dist if (score_cutoff is None or norm_dist <= score_cutoff) else 1.0


def normalized_similarity(
    s1,
    s2,
    *,
    pad=True,
    wildcard=None,
    processor=None,
    score_cutoff=None,
):
    """
    Calculates a normalized Hamming similarity in the range [0, 1].

    This is calculated as ``1 - normalized_distance``. In the masked Hamming mode it is
    ``matches / comparable``. A pair without any comparable position is reported as 0.0,
    so a fully masked pair can never become a perfect match.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    pad : bool, optional
       should strings be padded if there is a length difference.
       If pad is False and strings have a different length
       a ValueError is thrown instead. Defaults is True.
    wildcard : str, bytes or int, optional
        Enables the masked Hamming mode and marks positions without a real value.
        Default is None, which deactivates this behaviour.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For norm_sim < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    norm_sim : float
        normalized similarity between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If s1 and s2 have a different length while ``pad`` is False

    Examples
    --------
    >>> from rapidfuzz.distance import Hamming
    >>> Hamming.normalized_similarity("a?b", "axb", wildcard="?")
    1.0
    >>> Hamming.normalized_similarity("a?b", "bxa", wildcard="?")
    0.0
    """
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0.0

    if wildcard is not None:
        s1, s2, sentinel = _masked_prepare(s1, s2, wildcard, processor, pad)
        matches, comparable = _masked_stats(s1, s2, sentinel)
        norm_sim = matches / comparable if comparable else 0.0
        return norm_sim if (score_cutoff is None or norm_sim >= score_cutoff) else 0.0

    s1, s2 = conv_sequences(s1, s2)
    norm_dist = normalized_distance(s1, s2, pad=pad, processor=processor)
    norm_sim = 1 - norm_dist

    return norm_sim if (score_cutoff is None or norm_sim >= score_cutoff) else 0.0


def editops(
    s1,
    s2,
    *,
    pad=True,
    processor=None,
):
    """
    Return Editops describing how to turn s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    pad : bool, optional
       should strings be padded if there is a length difference.
       If pad is False and strings have a different length
       a ValueError is thrown instead. Defaults is True.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    editops : Editops
        edit operations required to turn s1 into s2
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)

    if not pad and len(s1) != len(s2):
        msg = "Sequences are not the same length."
        raise ValueError(msg)

    ops_list = []
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            ops_list.append(Editop("replace", i, i))

    for i in range(min_len, len(s1)):
        ops_list.append(Editop("delete", i, len(s2)))

    for i in range(min_len, len(s2)):
        ops_list.append(Editop("insert", len(s1), i))

    # sidestep input validation
    ops = Editops.__new__(Editops)
    ops._src_len = len(s1)
    ops._dest_len = len(s2)
    ops._editops = ops_list
    return ops


def opcodes(
    s1,
    s2,
    *,
    pad=True,
    processor=None,
):
    """
    Return Opcodes describing how to turn s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    pad : bool, optional
       should strings be padded if there is a length difference.
       If pad is False and strings have a different length
       a ValueError is thrown instead. Defaults is True.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.

    Returns
    -------
    opcodes : Opcodes
        edit operations required to turn s1 into s2
    """
    return editops(s1, s2, pad=pad, processor=processor).as_opcodes()


# ``process.extract`` uses this hook to reuse the very same set of comparable positions
# for score filtering, the coverage threshold and the tie breaking. It is only consulted
# when the masked mode is explicitly enabled through ``scorer_kwargs``.
for _scorer in (distance, similarity, normalized_distance, normalized_similarity):
    _scorer._RF_Coverage = comparable_count
del _scorer
