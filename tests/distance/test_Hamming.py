from __future__ import annotations

import pytest

from rapidfuzz import utils_cpp, utils_py
from rapidfuzz.distance import metrics_cpp, metrics_py
from tests.distance.common import Hamming


def hamming_editops(s1, s2):
    ops1 = metrics_cpp.hamming_editops(s1, s2)
    ops2 = metrics_py.hamming_editops(s1, s2)

    assert ops1.src_len == len(s1)
    assert ops1.dest_len == len(s2)
    assert ops2.src_len == len(s1)
    assert ops2.dest_len == len(s2)

    # todo currently the python and cpp implementation of Editops can
    # not be compared
    # assert ops1 == ops2
    assert ops1.as_list() == ops2.as_list()
    assert str(ops1) == str(ops2)
    assert repr(ops1) == repr(ops2)

    for op1, op2 in zip(ops1, ops2):
        assert op1 == op2
        assert str(op1) == str(op2)
        assert repr(op1) == repr(op2)

    return ops1


def test_basic():
    assert Hamming.distance("", "") == 0
    assert Hamming.distance("test", "test") == 0
    assert Hamming.distance("aaaa", "bbbb") == 4
    assert Hamming.distance("aaaa", "aaaaa") == 1


def test_disable_padding():
    assert Hamming.distance("", "", pad=False) == 0
    assert Hamming.distance("test", "test", pad=False) == 0
    assert Hamming.distance("aaaa", "bbbb", pad=False) == 4

    with pytest.raises(ValueError, match="Sequences are not the same length."):
        Hamming.distance("aaaa", "aaaaa", catch_exceptions=True, pad=False)

    # todo
    with pytest.raises(ValueError, match="Sequences are not the same length."):
        metrics_cpp.hamming_editops("aaaa", "aaaaa", pad=False)

    with pytest.raises(ValueError, match="Sequences are not the same length."):
        metrics_py.hamming_editops("aaaa", "aaaaa", pad=False)


def test_score_cutoff():
    """
    test whether score_cutoff works correctly
    """
    assert Hamming.distance("South Korea", "North Korea") == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=4) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=3) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=2) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=1) == 2
    assert Hamming.distance("South Korea", "North Korea", score_cutoff=0) == 1


def test_Editops():
    """
    basic test for Hamming.editops
    """
    assert hamming_editops("0", "").as_list() == [("delete", 0, 0)]
    assert hamming_editops("", "0").as_list() == [("insert", 0, 0)]

    assert hamming_editops("00", "0").as_list() == [("delete", 1, 1)]
    assert hamming_editops("0", "00").as_list() == [("insert", 1, 1)]

    assert hamming_editops("qabxcd", "abycdf").as_list() == [
        ("replace", 0, 0),
        ("replace", 1, 1),
        ("replace", 2, 2),
        ("replace", 3, 3),
        ("replace", 4, 4),
        ("replace", 5, 5),
    ]

    ops = hamming_editops("aaabaaa", "abbaaabba")
    assert ops.src_len == 7
    assert ops.dest_len == 9


def testCaseInsensitive():
    assert Hamming.distance("new york mets", "new YORK mets", processor=utils_cpp.default_process) == 0
    assert Hamming.distance("new york mets", "new YORK mets", processor=utils_py.default_process) == 0


def test_masked_mode():
    from rapidfuzz.distance import Hamming

    # only positions where both sides carry a real value take part
    assert Hamming.comparable_count("a?b", "axb") == 2
    assert Hamming.comparable_count("a?b", "bxa") == 2
    assert Hamming.comparable_positions("a?b", "axb") == (0, 2)

    assert Hamming.normalized_similarity("a?b", "axb", wildcard="?") == 1.0
    assert Hamming.normalized_similarity("a?b", "bxa", wildcard="?") == 0.0
    assert Hamming.normalized_distance("a?b", "axb", wildcard="?") == 0.0
    assert Hamming.normalized_distance("a?b", "bxa", wildcard="?") == 1.0
    assert Hamming.distance("a?b", "axb", wildcard="?") == 0
    assert Hamming.distance("a?b", "bxa", wildcard="?") == 2
    assert Hamming.similarity("a?b", "axb", wildcard="?") == 2
    assert Hamming.similarity("a?b", "bxa", wildcard="?") == 0

    # a candidate masked on the same positions is never a perfect match
    assert Hamming.normalized_similarity("???", "???", wildcard="?") == 0.0
    assert Hamming.normalized_distance("???", "???", wildcard="?") == 1.0
    assert Hamming.comparable_count("???", "???", wildcard="?") == 0

    # a masked position of one side never shifts the remaining positions
    assert Hamming.comparable_positions("a?b", "axc") == (0, 2)
    assert Hamming.normalized_similarity("a?b", "axc", wildcard="?") == 0.5


def test_masked_mode_legacy_unchanged():
    from rapidfuzz.distance import Hamming

    # without the new mode pad and a literal question mark keep their old meaning
    assert Hamming.distance("a?b", "axb") == 1
    assert Hamming.normalized_similarity("a?b", "axb") == pytest.approx(2 / 3)
    assert Hamming.distance("aaaa", "aaaaa") == 1
    with pytest.raises(ValueError, match="Sequences are not the same length."):
        Hamming.distance("aaaa", "aaaaa", pad=False)


def test_masked_mode_ranked_extract():
    from rapidfuzz import process
    from rapidfuzz.distance import Hamming

    kwargs = {"wildcard": "?"}

    # scores follow the comparable bits, not the raw string length
    hits = process.extract("a?b", ["axb", "bxa"], scorer=Hamming.normalized_similarity, scorer_kwargs=kwargs)
    assert [(score, key) for _, score, key in hits] == [(1.0, 0), (0.0, 1)]

    # candidates with fewer comparable bits are dropped, equal scores are ordered by
    # the amount of comparable bits and a full tie keeps the input order and the key
    ranked = process.extract(
        "a?b?c",
        ["a?bzc", "a????", "?ab?c", "a?b?c"],
        scorer=Hamming.normalized_similarity,
        scorer_kwargs={**kwargs, "min_comparable": 2},
    )
    assert [(score, key) for _, score, key in ranked] == [(1.0, 0), (1.0, 3), (1.0, 2)]

    assert process.extractOne(
        "a?b?c", ["?ab?c", "a?b?c"], scorer=Hamming.normalized_similarity, scorer_kwargs=kwargs
    )[2] == 1

    mapped = process.extract(
        "a?b", {"k1": "axb", "k2": "ayb"}, scorer=Hamming.normalized_similarity, scorer_kwargs=kwargs
    )
    assert [key for _, _, key in mapped] == ["k1", "k2"]
