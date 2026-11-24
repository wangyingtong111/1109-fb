from __future__ import annotations

import itertools

import pytest

from rapidfuzz.distance import Levenshtein_py


def _edit_cost(editops, weights):
    insert, delete, replace = weights
    costs = {"insert": insert, "delete": delete, "replace": replace}
    return sum(costs[tag] for tag, _src_pos, _dest_pos in editops)


@pytest.mark.parametrize("scorer", [Levenshtein_py], ids=["python"])
def test_cat_cut_prefers_delete_insert_over_expensive_replace(scorer):
    """
    with a substitution weight of 4 the optimal script deletes ``a`` and
    inserts ``u`` instead of substituting them: 1 + 1 = 2 < 4
    """
    weights = (1, 1, 4)
    ops = scorer.editops("cat", "cut", weights=weights)

    assert ops.as_list() == [("delete", 1, 1), ("insert", 2, 1)]
    assert ops.apply("cat", "cut") == "cut"
    assert _edit_cost(ops, weights) == 2
    assert _edit_cost(ops, weights) == scorer.distance("cat", "cut", weights=weights)


def test_default_editops_are_unchanged():
    """
    the default weights (1, 1, 1) keep using the original bit-parallel
    alignment and therefore produce the exact same edit script as before
    """
    assert Levenshtein_py.editops("cat", "cut").as_list() == [("replace", 1, 1)]
    assert Levenshtein_py.editops("qabxcd", "abycdf").as_list() == [
        ("delete", 0, 0),
        ("replace", 3, 2),
        ("insert", 6, 5),
    ]
    assert Levenshtein_py.editops("0", "").as_list() == [("delete", 0, 0)]
    assert Levenshtein_py.editops("", "0").as_list() == [("insert", 0, 0)]
    # explicitly passing the default weights takes the same code path
    assert (
        Levenshtein_py.editops("qabxcd", "abycdf", weights=(1, 1, 1)).as_list()
        == Levenshtein_py.editops("qabxcd", "abycdf").as_list()
    )
    # weights=None is equivalent to the default weights
    assert (
        Levenshtein_py.editops("cat", "cut", weights=None).as_list()
        == Levenshtein_py.editops("cat", "cut").as_list()
    )


@pytest.mark.parametrize(
    "weights",
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 4),
        (2, 3, 5),
        (3, 1, 2),
        (1, 7, 5),
        (5, 5, 5),
        (2, 2, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 0),
        (1, 0, 3),
        (0, 0, 5),
        (3, 0, 0),
        (0, 2, 7),
    ],
)
@pytest.mark.parametrize(
    ("s1", "s2"),
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("cat", "cut"),
        ("ac", "ca"),
        ("qabxcd", "abycdf"),
        ("kitten", "sitting"),
        ("aaabaaa", "abbaaabba"),
        ("Lorem ipsum.", "XYZLorem ABC iPsum"),
    ],
)
def test_weighted_editops_apply_and_match_distance(s1, s2, weights):
    ops = Levenshtein_py.editops(s1, s2, weights=weights)

    # the script transforms s1 into s2
    assert ops.apply(s1, s2) == s2
    # the total cost of the script equals the weighted Levenshtein distance
    assert _edit_cost(ops, weights) == Levenshtein_py.distance(s1, s2, weights=weights)

    # opcodes are an equivalent representation of the same edit script
    opcodes = Levenshtein_py.opcodes(s1, s2, weights=weights)
    assert opcodes == ops.as_opcodes()
    assert opcodes.as_editops() == ops
    assert opcodes.apply(s1, s2) == s2
    assert opcodes.as_matching_blocks() == ops.as_matching_blocks()


@pytest.mark.parametrize(
    "weights",
    [(1, 1, 4), (2, 3, 5), (3, 1, 2), (1, 7, 5), (2, 2, 0), (1, 0, 1), (0, 0, 0)],
)
@pytest.mark.parametrize(
    ("s1", "s2"),
    [
        ("cat", "cut"),
        ("ac", "ca"),
        ("qabxcd", "abycdf"),
        ("kitten", "sitting"),
        ("abc", ""),
    ],
)
def test_inverse_editops_and_reverse_cost(s1, s2, weights):
    """
    the inverse script transforms s2 back into s1. Because insertion and
    deletion may have different weights, evaluating the inverse requires
    swapping their weights; the resulting cost equals the distance of the
    reversed direction.
    """
    insert, delete, replace = weights
    reverse_weights = (delete, insert, replace)

    ops = Levenshtein_py.editops(s1, s2, weights=weights)
    inverse = ops.inverse()

    assert inverse.apply(s2, s1) == s1
    assert _edit_cost(inverse, reverse_weights) == Levenshtein_py.distance(
        s2, s1, weights=reverse_weights
    )
    # the inverse of the inverse is the original script
    assert inverse.inverse() == ops


@pytest.mark.parametrize(
    ("s1", "s2", "weights", "expected"),
    [
        # zero substitution cost: substitutions are still recorded, but
        # contribute nothing to the total cost
        ("cat", "cut", (1, 1, 0), [("replace", 1, 1)]),
        # zero deletion cost: deletions are recorded with a cost of zero
        ("abc", "ac", (1, 0, 1), [("delete", 1, 1)]),
        # zero insertion cost: insertions are recorded with a cost of zero
        ("ac", "abc", (0, 1, 1), [("insert", 1, 1)]),
        # all weights zero and equal strings: no edit operations at all
        ("abc", "abc", (0, 0, 0), []),
    ],
)
def test_zero_weights(s1, s2, weights, expected):
    ops = Levenshtein_py.editops(s1, s2, weights=weights)
    assert ops.as_list() == expected
    assert ops.apply(s1, s2) == s2
    assert _edit_cost(ops, weights) == Levenshtein_py.distance(s1, s2, weights=weights)


def test_all_zero_weights_different_strings():
    """
    with every weight being zero the distance is zero, but the edit script
    still has to describe how to transform one string into the other
    """
    weights = (0, 0, 0)
    ops = Levenshtein_py.editops("abc", "xyz", weights=weights)
    assert ops.apply("abc", "xyz") == "xyz"
    assert _edit_cost(ops, weights) == 0
    assert Levenshtein_py.distance("abc", "xyz", weights=weights) == 0


def test_equal_strings_have_no_editops():
    for weights in [(1, 1, 1), (2, 3, 5), (0, 0, 0), (1, 0, 1)]:
        ops = Levenshtein_py.editops("abc", "abc", weights=weights)
        assert ops.as_list() == []
        assert ops.src_len == 3
        assert ops.dest_len == 3
        assert _edit_cost(ops, weights) == 0


def _brute_force_distance(s1, s2, weights):
    insert, delete, replace = weights
    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        matrix[i][0] = i * delete
    for j in range(1, len2 + 1):
        matrix[0][j] = j * insert
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            matrix[i][j] = min(
                matrix[i - 1][j] + delete,
                matrix[i][j - 1] + insert,
                matrix[i - 1][j - 1] + (0 if s1[i - 1] == s2[j - 1] else replace),
            )
    return matrix[len1][len2]


def test_edit_script_is_optimal_on_small_alphabet():
    """
    exhaustively compare the cost of the generated edit script against an
    independent Wagner-Fischer implementation
    """
    alphabet = "ab"
    strings = [
        "".join(parts)
        for size in range(5)
        for parts in itertools.product(alphabet, repeat=size)
    ]
    weights_list = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 4),
        (2, 3, 5),
        (3, 1, 2),
        (1, 7, 5),
        (2, 2, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 0),
    ]

    for weights in weights_list:
        for s1 in strings:
            for s2 in strings:
                ops = Levenshtein_py.editops(s1, s2, weights=weights)
                assert ops.apply(s1, s2) == s2
                assert _edit_cost(ops, weights) == _brute_force_distance(
                    s1, s2, weights
                )


def test_weighted_editops_with_processor():
    ops = Levenshtein_py.editops("CAT", "cut", weights=(1, 1, 4), processor=str.lower)
    assert ops.apply("cat", "cut") == "cut"
    assert _edit_cost(ops, (1, 1, 4)) == Levenshtein_py.distance(
        "CAT", "cut", weights=(1, 1, 4), processor=str.lower
    )
