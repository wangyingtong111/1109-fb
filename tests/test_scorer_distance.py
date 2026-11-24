# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Max Bachmann
"""Tests for the public distance declaration entry ``process.scorer_distance``."""
from __future__ import annotations

from functools import wraps

import pytest

from rapidfuzz import process_py
from rapidfuzz._utils import ScorerFlag

try:
    from rapidfuzz import process_cpp
except ImportError:  # pragma: no cover - C++ extension is optional
    process_cpp = None


modules = [process_py]
if process_cpp is not None:
    modules.append(process_cpp)


def _cost_scorer(costs, *, optimal_score=0, worst_score=10, flags=None, record=None):
    """Build a distance scorer honoring the ``score_cutoff`` contract."""

    @process_py.scorer_distance(
        optimal_score=optimal_score,
        worst_score=worst_score,
        flags=ScorerFlag.RESULT_I64 if flags is None else flags,
    )
    def scorer(s1, s2, score_cutoff=None, **scorer_kwargs):
        cost = costs[s2]
        if record is not None:
            record.append((s2, score_cutoff, dict(scorer_kwargs)))
        if score_cutoff is not None and cost > score_cutoff:
            return score_cutoff + 1
        return cost

    return scorer


@pytest.mark.parametrize("module", modules)
def test_scorer_distance_flags(module):
    scorer = module.scorer_distance(optimal_score=2, worst_score=5)(
        lambda s1, s2, score_cutoff=None: 0
    )
    flags = scorer._RF_ScorerPy["get_scorer_flags"]()
    assert flags["optimal_score"] == 2
    assert flags["worst_score"] == 5

    # bare decorator uses the builtin distance defaults
    bare = module.scorer_distance(lambda s1, s2, score_cutoff=None: 0)
    bare_flags = bare._RF_ScorerPy["get_scorer_flags"]()
    assert bare_flags["optimal_score"] == 0
    assert bare_flags["worst_score"] == 2**63 - 1
    assert bare_flags["flags"] == ScorerFlag.RESULT_SIZE_T | ScorerFlag.SYMMETRIC

    # fractional bounds (e.g. normalized distances) are supported as well
    normalized = module.scorer_distance(
        optimal_score=0.0, worst_score=1.0, flags=ScorerFlag.RESULT_F64
    )(lambda s1, s2, score_cutoff=None: 0.0)
    normalized_flags = normalized._RF_ScorerPy["get_scorer_flags"]()
    assert normalized_flags == {
        "optimal_score": 0.0,
        "worst_score": 1.0,
        "flags": ScorerFlag.RESULT_F64,
    }


@pytest.mark.parametrize("module", modules)
def test_scorer_distance_rejects_invalid_bounds(module):
    with pytest.raises(ValueError, match="optimal_score has to be smaller"):
        module.scorer_distance(optimal_score=5, worst_score=5)
    with pytest.raises(ValueError, match="optimal_score has to be smaller"):
        module.scorer_distance(optimal_score=6, worst_score=5)


@pytest.mark.parametrize("module", modules)
def test_extractOne_distance_contract(module):
    # costs 2 and 5 with a cutoff of 3: only the cost 2 choice is acceptable
    scorer = _cost_scorer({"a": 2, "b": 5})
    assert module.extractOne("q", ["a", "b"], scorer=scorer, score_cutoff=3) == ("a", 2, 0)
    assert module.extract("q", ["a", "b"], scorer=scorer, score_cutoff=3, limit=5) == [
        ("a", 2, 0)
    ]
    assert list(module.extract_iter("q", ["a", "b"], scorer=scorer, score_cutoff=3)) == [
        ("a", 2, 0)
    ]

    # without a cutoff the worst score acts as the initial upper bound
    assert module.extractOne("q", ["a", "b"], scorer=scorer) == ("a", 2, 0)
    assert module.extract("q", ["a", "b"], scorer=scorer) == [("a", 2, 0), ("b", 5, 1)]


@pytest.mark.parametrize("module", modules)
def test_extractOne_early_exit_and_narrowing_cutoff(module):
    # recorded call sequence: cost 4 first, then two cost 2 choices,
    # the declared optimal score is 2
    calls = []
    scorer = _cost_scorer(
        {"a": 4, "b": 2, "c": 2}, optimal_score=2, worst_score=10, record=calls
    )

    result = module.extractOne(
        "q", ["a", "b", "c"], scorer=scorer, scorer_kwargs={"factor": 3}
    )
    # the first optimal score (2) ends the search, ties keep the first match
    assert result == ("b", 2, 1)
    # the narrowest acceptable upper bound is handed over before every call:
    # first the worst score 10, then the cutoff narrowed down to the best cost 4
    assert calls == [
        ("a", 10, {"factor": 3}),
        ("b", 4, {"factor": 3}),
    ]


@pytest.mark.parametrize("module", modules)
def test_extract_keeps_traversal_and_tie_semantics(module):
    scorer = _cost_scorer({"a": 4, "b": 2, "c": 2}, optimal_score=2, worst_score=10)

    # top k keeps evaluating every choice and orders by the raw distance
    assert module.extract("q", ["a", "b", "c"], scorer=scorer, limit=2) == [
        ("b", 2, 1),
        ("c", 2, 2),
    ]
    assert module.extract("q", ["a", "b", "c"], scorer=scorer, limit=5) == [
        ("b", 2, 1),
        ("c", 2, 2),
        ("a", 4, 0),
    ]

    # the iterable interface keeps the original order and threshold semantics
    assert list(module.extract_iter("q", ["a", "b", "c"], scorer=scorer)) == [
        ("a", 4, 0),
        ("b", 2, 1),
        ("c", 2, 2),
    ]


@pytest.mark.parametrize("module", modules)
def test_declaration_on_wrapped_callable(module):
    calls = []
    scorer = _cost_scorer(
        {"a": 4, "b": 2, "c": 2}, optimal_score=2, worst_score=10, record=calls
    )

    @wraps(scorer)
    def wrapper(s1, s2, **kwargs):
        return scorer(s1, s2, **kwargs)

    declared = module.scorer_distance(
        optimal_score=2, worst_score=10, flags=ScorerFlag.RESULT_I64
    )(wrapper)
    assert hasattr(declared, "_RF_ScorerPy")

    calls.clear()
    assert module.extractOne("q", ["a", "b", "c"], scorer=declared) == ("b", 2, 1)
    assert [call[:2] for call in calls] == [("a", 10), ("b", 4)]


@pytest.mark.parametrize("module", modules)
def test_undeclared_scorer_keeps_similarity_contract(module):
    from rapidfuzz import fuzz

    def plain_scorer(s1, s2, score_cutoff=0):
        return fuzz.ratio(s1, s2, score_cutoff=score_cutoff)

    assert not hasattr(plain_scorer, "_RF_ScorerPy")
    result = module.extractOne(
        "new york mets", ["new", "new york mets"], scorer=plain_scorer
    )
    assert result == ("new york mets", 100.0, 1)

    @wraps(plain_scorer)
    def wrapped_scorer(*args, **kwargs):
        return plain_scorer(*args, **kwargs)

    assert not hasattr(wrapped_scorer, "_RF_ScorerPy")
    result = module.extractOne(
        "new york mets", ["new", "new york mets"], scorer=wrapped_scorer
    )
    assert result == ("new york mets", 100.0, 1)


@pytest.mark.parametrize("module", modules)
def test_distance_scorer_with_mapping(module):
    scorer = _cost_scorer({"a": 4, "b": 2}, optimal_score=2, worst_score=10)
    assert module.extractOne("q", {"ka": "a", "kb": "b"}, scorer=scorer) == (
        "b",
        2,
        "kb",
    )
    assert list(module.extract_iter("q", {"ka": "a", "kb": "b"}, scorer=scorer)) == [
        ("a", 4, "ka"),
        ("b", 2, "kb"),
    ]


@pytest.mark.parametrize("module", modules)
def test_normalized_float_distance(module):
    costs = {"a": 0.4, "b": 0.2, "c": 0.2}

    @module.scorer_distance(
        optimal_score=0.0, worst_score=1.0, flags=ScorerFlag.RESULT_F64
    )
    def scorer(s1, s2, score_cutoff=None):
        cost = costs[s2]
        if score_cutoff is not None and cost > score_cutoff:
            return 1.0
        return cost

    assert module.extractOne("q", ["a", "b", "c"], scorer=scorer) == ("b", 0.2, 1)
    assert module.extract("q", ["a", "b", "c"], scorer=scorer, limit=2) == [
        ("b", 0.2, 1),
        ("c", 0.2, 2),
    ]
    assert module.extractOne(
        "q", ["a", "b", "c"], scorer=scorer, score_cutoff=0.3
    ) == ("b", 0.2, 1)
    assert module.extract(
        "q", ["a", "b", "c"], scorer=scorer, score_cutoff=0.3
    ) == [("b", 0.2, 1), ("c", 0.2, 2)]
    assert module.extractOne(
        "q", ["a"], scorer=scorer, score_cutoff=0.1
    ) is None


def test_cdist_uses_declared_result_type():
    np = pytest.importorskip("numpy")

    @process_py.scorer_distance(
        optimal_score=0, worst_score=10, flags=ScorerFlag.RESULT_I64
    )
    def scorer(s1, s2, score_cutoff=None):
        return abs(int(s1) - int(s2))

    matrix = process_py.cdist(["1", "5"], ["2", "5"], scorer=scorer)
    assert matrix.dtype == np.int32
    assert matrix.tolist() == [[1, 4], [3, 0]]

    vector = process_py.cpdist(["1", "5"], ["2", "5"], scorer=scorer)
    assert vector.dtype == np.int32
    assert vector.tolist() == [1, 0]
