# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from __future__ import annotations

from rapidfuzz._common_py import common_affix, conv_sequences
from rapidfuzz._utils import is_none, setupPandas
from rapidfuzz.distance import Indel_py as Indel
from rapidfuzz.distance._initialize_py import Editop, Editops


def _levenshtein_maximum(s1, s2, weights):
    len1 = len(s1)
    len2 = len(s2)
    insert, delete, replace = weights

    max_dist = len1 * delete + len2 * insert

    if len1 >= len2:
        max_dist = min(max_dist, len2 * replace + (len1 - len2) * delete)
    else:
        max_dist = min(max_dist, len1 * replace + (len2 - len1) * insert)

    return max_dist


def _uniform_generic(s1, s2, weights, score_cutoff=None):
    len1 = len(s1)
    len2 = len(s2)
    insert, delete, replace = weights

    if score_cutoff is None or score_cutoff == float("inf") or min(weights) < 0:
        cache = [i * delete for i in range(len1 + 1)]

        for ch2 in s2:
            temp = cache[0]
            cache[0] += insert
            for i in range(len1):
                x = temp
                if s1[i] != ch2:
                    x = min(cache[i] + delete, cache[i + 1] + insert, temp + replace)
                temp = cache[i + 1]
                cache[i + 1] = x

        return cache[-1]

    # A path through cell (i, j) has offset j - i. The length difference to
    # the final cell forces the same number of unmatched characters on top of
    # the global length difference. This gives a weight-dependent diagonal band
    # containing every path which can stay within score_cutoff.
    length_difference = len2 - len1
    if length_difference >= 0:
        length_lower_bound = length_difference * insert
    else:
        length_lower_bound = -length_difference * delete

    if length_lower_bound > score_cutoff:
        return score_cutoff + 1

    indel_cost = insert + delete
    if indel_cost == 0:
        # All characters can be removed and inserted without cost.
        return 0

    extra_diagonals = int((score_cutoff - length_lower_bound) // indel_cost)
    min_offset = min(0, length_difference) - extra_diagonals
    max_offset = max(0, length_difference) + extra_diagonals

    inf = float("inf")
    cache = [inf] * (len1 + 1)

    # Initialize only row zero states inside the diagonal band.
    max_initial_index = min(len1, -min_offset)
    for i in range(max_initial_index + 1):
        cache[i] = i * delete

    for j, ch2 in enumerate(s2, 1):
        start = max(1, j - max_offset)
        end = min(len1, j - min_offset)
        row_has_possible_state = False

        if start > end:
            # This only occurs for an empty s1. Keeping cache[0] exact preserves
            # the normal boundary initialization.
            cache[0] += insert
            row_has_possible_state = cache[0] <= score_cutoff
        else:
            if start == 1:
                temp = cache[0]
                if j <= max_offset:
                    cache[0] = temp + insert
                else:
                    cache[0] = inf

                if cache[0] <= score_cutoff:
                    row_has_possible_state = True
            else:
                # cache[start - 1] has not been updated in this row yet, so it is
                # the diagonal predecessor from the previous row.
                temp = cache[start - 1]

            i = start
            above = cache[i]
            if s1[i - 1] == ch2:
                current = temp
            else:
                current = min(temp + replace, above + insert)
                if start == 1 and j <= max_offset:
                    current = min(current, cache[0] + delete)

            temp = above
            cache[i] = current
            if current <= score_cutoff:
                row_has_possible_state = True

            for i in range(start + 1, end + 1):
                above = cache[i]
                if s1[i - 1] == ch2:
                    current = temp
                else:
                    current = min(temp + replace, cache[i - 1] + delete, above + insert)

                temp = above
                cache[i] = current
                if current <= score_cutoff:
                    row_has_possible_state = True

        if not row_has_possible_state:
            return score_cutoff + 1

    return cache[-1] if cache[-1] <= score_cutoff else score_cutoff + 1


def _uniform_distance(s1, s2):
    if not s1:
        return len(s2)

    VP = (1 << len(s1)) - 1
    VN = 0
    currDist = len(s1)
    mask = 1 << (len(s1) - 1)

    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    for ch2 in s2:
        # Step 1: Computing D0
        PM_j = block_get(ch2, 0)
        X = PM_j
        D0 = (((X & VP) + VP) ^ VP) | X | VN
        # Step 2: Computing HP and HN
        HP = VN | ~(D0 | VP)
        HN = D0 & VP
        # Step 3: Computing the value D[m,j]
        currDist += (HP & mask) != 0
        currDist -= (HN & mask) != 0
        # Step 4: Computing Vp and VN
        HP = (HP << 1) | 1
        HN = HN << 1
        VP = HN | ~(D0 | HP)
        VN = HP & D0

    return currDist


def distance(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    """
    Calculates the minimum number of insertions, deletions, and substitutions
    required to change one sequence into the other according to Levenshtein with custom
    costs for insertion, deletion and substitution

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor : callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the distance is bigger than score_cutoff,
        score_cutoff + 1 is returned instead. Default is None, which deactivates
        this behaviour.
    score_hint : int, optional
        Expected distance between s1 and s2. This is used to select a
        faster implementation. Default is None, which deactivates this behaviour.

    Returns
    -------
    distance : int
        distance between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    Examples
    --------
    Find the Levenshtein distance between two strings:

    >>> from rapidfuzz.distance import Levenshtein
    >>> Levenshtein.distance("lewenstein", "levenshtein")
    2

    Setting a maximum distance allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.distance("lewenstein", "levenshtein", score_cutoff=1)
    2

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.distance("lewenstein", "levenshtein", weights=(1,1,2))
    3
    """
    _ = score_hint
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    if weights is None or weights == (1, 1, 1):
        dist = _uniform_distance(s1, s2)
    elif weights == (1, 1, 2):
        dist = Indel.distance(s1, s2)
    elif score_cutoff is None:
        dist = _uniform_generic(s1, s2, weights)
    else:
        dist = _uniform_generic(s1, s2, weights, score_cutoff)

    return dist if (score_cutoff is None or dist <= score_cutoff) else score_cutoff + 1


def similarity(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    """
    Calculates the levenshtein similarity in the range [max, 0] using custom
    costs for insertion, deletion and substitution.

    This is calculated as ``max - distance``, where max is the maximal possible
    Levenshtein distance given the lengths of the sequences s1/s2 and the weights.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor : callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : int, optional
        Maximum distance between s1 and s2, that is
        considered as a result. If the similarity is smaller than score_cutoff,
        0 is returned instead. Default is None, which deactivates
        this behaviour.
    score_hint : int, optional
        Expected similarity between s1 and s2. This is used to select a
        faster implementation. Default is None, which deactivates this behaviour.

    Returns
    -------
    similarity : int
        similarity between s1 and s2

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown
    """
    _ = score_hint
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    weights = weights or (1, 1, 1)
    maximum = _levenshtein_maximum(s1, s2, weights)
    dist = distance(s1, s2, weights=weights)
    sim = maximum - dist
    return sim if (score_cutoff is None or sim >= score_cutoff) else 0


def normalized_distance(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    """
    Calculates a normalized levenshtein distance in the range [1, 0] using custom
    costs for insertion, deletion and substitution.

    This is calculated as ``distance / max``, where max is the maximal possible
    Levenshtein distance given the lengths of the sequences s1/s2 and the weights.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor : callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For norm_dist > score_cutoff 1.0 is returned instead. Default is None,
        which deactivates this behaviour.
    score_hint : float, optional
        Expected normalized distance between s1 and s2. This is used to select a
        faster implementation. Default is None, which deactivates this behaviour.

    Returns
    -------
    norm_dist : float
        normalized distance between s1 and s2 as a float between 1.0 and 0.0

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown
    """
    _ = score_hint
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 1.0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    weights = weights or (1, 1, 1)
    maximum = _levenshtein_maximum(s1, s2, weights)
    dist = distance(s1, s2, weights=weights)
    norm_dist = dist / maximum if maximum else 0
    return norm_dist if (score_cutoff is None or norm_dist <= score_cutoff) else 1


def normalized_similarity(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    """
    Calculates a normalized levenshtein similarity in the range [0, 1] using custom
    costs for insertion, deletion and substitution.

    This is calculated as ``1 - normalized_distance``

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    weights : tuple[int, int, int] or None, optional
        The weights for the three operations in the form
        (insertion, deletion, substitution). Default is (1, 1, 1),
        which gives all three operations a weight of 1.
    processor : callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For norm_sim < score_cutoff 0 is returned instead. Default is None,
        which deactivates this behaviour.
    score_hint : int, optional
        Expected normalized similarity between s1 and s2. This is used to select a
        faster implementation. Default is None, which deactivates this behaviour.

    Returns
    -------
    norm_sim : float
        normalized similarity between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If unsupported weights are provided a ValueError is thrown

    Examples
    --------
    Find the normalized Levenshtein similarity between two strings:

    >>> from rapidfuzz.distance import Levenshtein
    >>> Levenshtein.normalized_similarity("lewenstein", "levenshtein")
    0.81818181818181

    Setting a score_cutoff allows the implementation to select
    a more efficient implementation:

    >>> Levenshtein.normalized_similarity("lewenstein", "levenshtein", score_cutoff=0.85)
    0.0

    It is possible to select different weights by passing a `weight`
    tuple.

    >>> Levenshtein.normalized_similarity("lewenstein", "levenshtein", weights=(1,1,2))
    0.85714285714285

    When a different processor is used s1 and s2 do not have to be strings

    >>> Levenshtein.normalized_similarity(["lewenstein"], ["levenshtein"], processor=lambda s: s[0])
    0.81818181818181
    """
    _ = score_hint
    setupPandas()
    if is_none(s1) or is_none(s2):
        return 0.0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    weights = weights or (1, 1, 1)
    norm_dist = normalized_distance(s1, s2, weights=weights)
    norm_sim = 1.0 - norm_dist
    return norm_sim if (score_cutoff is None or norm_sim >= score_cutoff) else 0


def _matrix(s1, s2):
    if not s1:
        return (len(s2), [], [])

    VP = (1 << len(s1)) - 1
    VN = 0
    currDist = len(s1)
    mask = 1 << (len(s1) - 1)

    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    matrix_VP = []
    matrix_VN = []
    for ch2 in s2:
        # Step 1: Computing D0
        PM_j = block_get(ch2, 0)
        X = PM_j
        D0 = (((X & VP) + VP) ^ VP) | X | VN
        # Step 2: Computing HP and HN
        HP = VN | ~(D0 | VP)
        HN = D0 & VP
        # Step 3: Computing the value D[m,j]
        currDist += (HP & mask) != 0
        currDist -= (HN & mask) != 0
        # Step 4: Computing Vp and VN
        HP = (HP << 1) | 1
        HN = HN << 1
        VP = HN | ~(D0 | HP)
        VN = HP & D0

        matrix_VP.append(VP)
        matrix_VN.append(VN)

    return (currDist, matrix_VP, matrix_VN)


def editops(
    s1,
    s2,
    *,
    processor=None,
    score_hint=None,
):
    """
    Return Editops describing how to turn s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor : callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_hint : int, optional
        Expected distance between s1 and s2. This is used to select a
        faster implementation. Default is None, which deactivates this behaviour.

    Returns
    -------
    editops : Editops
        edit operations required to turn s1 into s2

    Notes
    -----
    The alignment is calculated using an algorithm of Heikki Hyyrö, which is
    described [8]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [8] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.distance import Levenshtein
    >>> for tag, src_pos, dest_pos in Levenshtein.editops("qabxcd", "abycdf"):
    ...    print(("%7s s1[%d] s2[%d]" % (tag, src_pos, dest_pos)))
     delete s1[1] s2[0]
    replace s1[3] s2[2]
     insert s1[6] s2[5]
    """
    _ = score_hint
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)
    prefix_len, suffix_len = common_affix(s1, s2)
    s1 = s1[prefix_len : len(s1) - suffix_len]
    s2 = s2[prefix_len : len(s2) - suffix_len]
    dist, VP, VN = _matrix(s1, s2)

    editops = Editops([], 0, 0)
    editops._src_len = len(s1) + prefix_len + suffix_len
    editops._dest_len = len(s2) + prefix_len + suffix_len

    if dist == 0:
        return editops

    editop_list = [None] * dist
    col = len(s1)
    row = len(s2)
    while row != 0 and col != 0:
        # deletion
        if VP[row - 1] & (1 << (col - 1)):
            dist -= 1
            col -= 1
            editop_list[dist] = Editop("delete", col + prefix_len, row + prefix_len)
        else:
            row -= 1

            # insertion
            if row and (VN[row - 1] & (1 << (col - 1))):
                dist -= 1
                editop_list[dist] = Editop("insert", col + prefix_len, row + prefix_len)
            else:
                col -= 1

                # replace (Matches are not recorded)
                if s1[col] != s2[row]:
                    dist -= 1
                    editop_list[dist] = Editop("replace", col + prefix_len, row + prefix_len)

    while col != 0:
        dist -= 1
        col -= 1
        editop_list[dist] = Editop("delete", col + prefix_len, row + prefix_len)

    while row != 0:
        dist -= 1
        row -= 1
        editop_list[dist] = Editop("insert", col + prefix_len, row + prefix_len)

    editops._editops = editop_list
    return editops


def opcodes(
    s1,
    s2,
    *,
    processor=None,
    score_hint=None,
):
    """
    Return Opcodes describing how to turn s1 into s2.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor : callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_hint : int, optional
        Expected distance between s1 and s2. This is used to select a
        faster implementation. Default is None, which deactivates this behaviour.

    Returns
    -------
    opcodes : Opcodes
        edit operations required to turn s1 into s2

    Notes
    -----
    The alignment is calculated using an algorithm of Heikki Hyyrö, which is
    described [9]_. It has a time complexity and memory usage of ``O([N/64] * M)``.

    References
    ----------
    .. [9] Hyyrö, Heikki. "A Note on Bit-Parallel Alignment Computation."
           Stringology (2004).

    Examples
    --------
    >>> from rapidfuzz.distance import Levenshtein

    >>> a = "qabxcd"
    >>> b = "abycdf"
    >>> for tag, i1, i2, j1, j2 in Levenshtein.opcodes("qabxcd", "abycdf"):
    ...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
    ...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
     delete a[0:1] (q) b[0:0] ()
      equal a[1:3] (ab) b[0:2] (ab)
    replace a[3:4] (x) b[2:3] (y)
      equal a[4:6] (cd) b[3:5] (cd)
     insert a[6:6] () b[5:6] (f)
    """
    return editops(s1, s2, processor=processor, score_hint=score_hint).as_opcodes()
