# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

import sys
from math import isnan
from typing import Any, Callable

pandas_NA = None


def setupPandas() -> None:
    global pandas_NA  # noqa: PLW0603
    if pandas_NA is None:
        pandas = sys.modules.get("pandas")
        if hasattr(pandas, "NA"):
            pandas_NA = pandas.NA


setupPandas()


class ScorerFlag:
    RESULT_F64 = 1 << 5
    RESULT_I64 = 1 << 6
    RESULT_SIZE_T = 1 << 7
    SYMMETRIC = 1 << 11


def _get_scorer_flags_distance(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 0,
        "worst_score": 2**63 - 1,
        "flags": ScorerFlag.RESULT_SIZE_T | ScorerFlag.SYMMETRIC,
    }


def _get_scorer_flags_similarity(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 2**63 - 1,
        "worst_score": 0,
        "flags": ScorerFlag.RESULT_SIZE_T | ScorerFlag.SYMMETRIC,
    }


def _get_scorer_flags_normalized_distance(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 0,
        "worst_score": 1,
        "flags": ScorerFlag.RESULT_F64 | ScorerFlag.SYMMETRIC,
    }


def _get_scorer_flags_normalized_similarity(**_kwargs: Any) -> dict[str, Any]:
    return {
        "optimal_score": 1,
        "worst_score": 0,
        "flags": ScorerFlag.RESULT_F64 | ScorerFlag.SYMMETRIC,
    }


def is_none(s: Any) -> bool:
    if s is None or s is pandas_NA:
        return True

    return isinstance(s, float) and isnan(s)


def add_scorer_attrs(func: Any, cached_scorer_call: dict[str, Callable[..., dict[str, Any]]]) -> None:
    func._RF_ScorerPy = cached_scorer_call
    # used to detect the function hasn't been wrapped afterwards
    func._RF_OriginalScorer = func


def scorer_distance(
    func: Callable[..., Any] | None = None,
    *,
    optimal_score: int | float = 0,
    worst_score: int | float = 2**63 - 1,
    flags: int | None = None,
) -> Any:
    """
    Declare a custom scorer as an edit distance.

    A distance scorer returns a score for which smaller values represent better
    matches. ``optimal_score`` is the best possible score (returned for a perfect
    match) and ``worst_score`` is the worst possible score. When no
    ``score_cutoff`` is provided the ``worst_score`` is used as the initial
    threshold and every score up to this threshold is accepted.

    The declaration can either be used as a decorator factory with keyword
    arguments::

        @scorer_distance(optimal_score=2, worst_score=10)
        def my_distance(s1, s2, score_cutoff=None): ...

    or directly as a decorator, in which case the default declaration for an
    edit distance (``optimal_score=0``, ``worst_score=2**63 - 1``) is used::

        @scorer_distance
        def my_distance(s1, s2, score_cutoff=None): ...

    Attaching the declaration to a wrapped callable is supported as well:
    simply apply the returned decorator to the wrapper function. Scorers without
    any declaration keep the legacy similarity contract.

    Parameters
    ----------
    func : Callable, optional
        the scorer to declare. When omitted a decorator is returned instead.
    optimal_score : int or float, optional
        the best possible score the scorer can return. Has to be smaller than
        ``worst_score``. Defaults to ``0``.
    worst_score : int or float, optional
        the worst possible score the scorer can return. Defaults to ``2**63 - 1``.
    flags : int, optional
        optional combination of :class:`ScorerFlag` values describing the result
        type (e.g. :data:`ScorerFlag.RESULT_I64`). This is mainly used to select
        the dtype of the arrays returned by ``cdist``/``cpdist``.

    Returns
    -------
    Callable
        the declared scorer or a decorator declaring the passed scorer.
    """
    if optimal_score >= worst_score:
        msg = "optimal_score has to be smaller than worst_score for a distance scorer"
        raise ValueError(msg)

    if flags is None:
        flags = ScorerFlag.RESULT_SIZE_T | ScorerFlag.SYMMETRIC

    def _decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        def _get_scorer_flags(**_kwargs: Any) -> dict[str, Any]:
            return {
                "optimal_score": optimal_score,
                "worst_score": worst_score,
                "flags": flags,
            }

        add_scorer_attrs(inner_func, {"get_scorer_flags": _get_scorer_flags})
        return inner_func

    if func is not None:
        return _decorator(func)

    return _decorator


default_distance_attribute: dict[str, Callable[..., dict[str, Any]]] = {"get_scorer_flags": _get_scorer_flags_distance}
default_similarity_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _get_scorer_flags_similarity
}
default_normalized_distance_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _get_scorer_flags_normalized_distance
}
default_normalized_similarity_attribute: dict[str, Callable[..., dict[str, Any]]] = {
    "get_scorer_flags": _get_scorer_flags_normalized_similarity
}
