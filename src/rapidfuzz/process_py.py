# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann
from __future__ import annotations

import heapq

from rapidfuzz._utils import ScorerFlag, is_none, setupPandas
from rapidfuzz.fuzz import WRatio, ratio

__all__ = ["cdist", "extract", "extractOne", "extract_iter"]


def _get_scorer_flags_py(scorer, scorer_kwargs):
    params = getattr(scorer, "_RF_ScorerPy", None)
    if params is not None:
        flags = params["get_scorer_flags"](**scorer_kwargs)
        return (flags["worst_score"], flags["optimal_score"])
    return (0, 100)


def extract_iter(
    query,
    choices,
    *,
    scorer=WRatio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    """
    Find the best match in a list of choices

    Parameters
    ----------
    query : Sequence[Hashable]
        string we want to find
    choices : Iterable[Sequence[Hashable]] | Mapping[Sequence[Hashable]]
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.WRatio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance > score_cutoff` are ignored. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity < score_cutoff` are ignored. Default is None, which deactivates this behaviour.
    score_hint : Any, optional
        Optional argument for an expected score to be passed to the scorer.
        This is used to select a faster implementation. Default is None,
        which deactivates this behaviour.
    scorer_kwargs : dict[str, Any], optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to `Levenshtein.distance`

    Yields
    -------
    tuple[Sequence[Hashable], Any, Any]
        Yields similarity between the query and each choice in form of a Tuple with 3 elements.
        The values stored in the tuple depend on the types of the input arguments.

        * The first element is always the current `choice`, which is the value that's compared to the query.

        * The second value represents the similarity calculated by the scorer. This can be:

          * An edit distance (distance is 0 for a perfect match and > 0 for non perfect matches).
            In this case only choices which have a `distance <= score_cutoff` are yielded.
            An example of a scorer with this behavior is `Levenshtein.distance`.
          * A normalized edit distance (similarity is a score between 0 and 100, with 100 being a perfect match).
            In this case only choices which have a `similarity >= score_cutoff` are yielded.
            An example of a scorer with this behavior is `Levenshtein.normalized_similarity`.

          Note, that for all scorers, which are not provided by RapidFuzz, only normalized edit distances are supported.

        * The third parameter depends on the type of the `choices` argument it is:

          * The `index of choice` when choices is a simple iterable like a list
          * The `key of choice` when choices is a mapping like a dict, or a pandas Series

    """
    _ = score_hint
    scorer_kwargs = scorer_kwargs or {}
    worst_score, optimal_score = _get_scorer_flags_py(scorer, scorer_kwargs)
    lowest_score_worst = optimal_score > worst_score

    setupPandas()

    if is_none(query):
        return

    if score_cutoff is None:
        score_cutoff = worst_score

    # preprocess the query
    if processor is not None:
        query = processor(query)

    choices_iter = choices.items() if hasattr(choices, "items") else enumerate(choices)
    for key, choice in choices_iter:
        if is_none(choice):
            continue

        if processor is None:
            score = scorer(query, choice, score_cutoff=score_cutoff, **scorer_kwargs)
        else:
            score = scorer(
                query,
                processor(choice),
                score_cutoff=score_cutoff,
                **scorer_kwargs,
            )

        if lowest_score_worst:
            if score >= score_cutoff:
                yield (choice, score, key)
        else:
            if score <= score_cutoff:
                yield (choice, score, key)


def extractOne(
    query,
    choices,
    *,
    scorer=WRatio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    """
    Find the best match in a list of choices. When multiple elements have the same similarity,
    the first element is returned.

    Parameters
    ----------
    query : Sequence[Hashable]
        string we want to find
    choices : Iterable[Sequence[Hashable]] | Mapping[Sequence[Hashable]]
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.WRatio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance > score_cutoff` are ignored. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity < score_cutoff` are ignored. Default is None, which deactivates this behaviour.
    score_hint : Any, optional
        Optional argument for an expected score to be passed to the scorer.
        This is used to select a faster implementation. Default is None,
        which deactivates this behaviour.
    scorer_kwargs : dict[str, Any], optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to `Levenshtein.distance`

    Returns
    -------
    tuple[Sequence[Hashable], Any, Any]
        Returns the best match in form of a Tuple with 3 elements. The values stored in the
        tuple depend on the types of the input arguments.

        * The first element is always the `choice`, which is the value that's compared to the query.

        * The second value represents the similarity calculated by the scorer. This can be:

          * An edit distance (distance is 0 for a perfect match and > 0 for non perfect matches).
            In this case only choices which have a `distance <= score_cutoff` are returned.
            An example of a scorer with this behavior is `Levenshtein.distance`.
          * A normalized edit distance (similarity is a score between 0 and 100, with 100 being a perfect match).
            In this case only choices which have a `similarity >= score_cutoff` are returned.
            An example of a scorer with this behavior is `Levenshtein.normalized_similarity`.

          Note, that for all scorers, which are not provided by RapidFuzz, only normalized edit distances are supported.

        * The third parameter depends on the type of the `choices` argument it is:

          * The `index of choice` when choices is a simple iterable like a list
          * The `key of choice` when choices is a mapping like a dict, or a pandas Series

    None
        When no choice has a `similarity >= score_cutoff`/`distance <= score_cutoff` None is returned

    Examples
    --------

    >>> from rapidfuzz.process import extractOne
    >>> from rapidfuzz.distance import Levenshtein
    >>> from rapidfuzz.fuzz import ratio

    extractOne can be used with normalized edit distances.

    >>> extractOne("abcd", ["abce"], scorer=ratio)
    ("abcd", 75.0, 1)
    >>> extractOne("abcd", ["abce"], scorer=Levenshtein.normalized_similarity)
    ("abcd", 0.75, 1)

    extractOne can be used with edit distances as well.

    >>> extractOne("abcd", ["abce"], scorer=Levenshtein.distance)
    ("abce", 1, 0)

    additional settings of the scorer can be passed via the scorer_kwargs argument to extractOne

    >>> extractOne("abcd", ["abce"], scorer=Levenshtein.distance, scorer_kwargs={"weights":(1,1,2)})
    ("abcde", 2, 1)

    when a mapping is used for the choices the key of the choice is returned instead of the List index

    >>> extractOne("abcd", {"key": "abce"}, scorer=ratio)
    ("abcd", 75.0, "key")

    It is possible to specify a processor function which is used to preprocess the strings before comparing them.

    >>> extractOne("abcd", ["abcD"], scorer=ratio)
    ("abcD", 75.0, 0)
    >>> extractOne("abcd", ["abcD"], scorer=ratio, processor=utils.default_process)
    ("abcD", 100.0, 0)
    >>> extractOne("abcd", ["abcD"], scorer=ratio, processor=lambda s: s.upper())
    ("abcD", 100.0, 0)

    When only results with a similarity above a certain threshold are relevant, the parameter score_cutoff can be
    used to filter out results with a lower similarity. This threshold is used by some of the scorers to exit early,
    when they are sure, that the similarity is below the threshold.
    For normalized edit distances all results with a similarity below score_cutoff are filtered out

    >>> extractOne("abcd", ["abce"], scorer=ratio)
    ("abce", 75.0, 0)
    >>> extractOne("abcd", ["abce"], scorer=ratio, score_cutoff=80)
    None

    For edit distances all results with an edit distance above the score_cutoff are filtered out

    >>> extractOne("abcd", ["abce"], scorer=Levenshtein.distance, scorer_kwargs={"weights":(1,1,2)})
    ("abce", 2, 0)
    >>> extractOne("abcd", ["abce"], scorer=Levenshtein.distance, scorer_kwargs={"weights":(1,1,2)}, score_cutoff=1)
    None

    """
    _ = score_hint
    scorer_kwargs = scorer_kwargs or {}
    worst_score, optimal_score = _get_scorer_flags_py(scorer, scorer_kwargs)
    lowest_score_worst = optimal_score > worst_score

    setupPandas()

    if is_none(query):
        return None

    if score_cutoff is None:
        score_cutoff = worst_score

    # preprocess the query
    if processor is not None:
        query = processor(query)

    result = None

    choices_iter = choices.items() if hasattr(choices, "items") else enumerate(choices)
    for key, choice in choices_iter:
        if is_none(choice):
            continue

        if processor is None:
            score = scorer(query, choice, score_cutoff=score_cutoff, **scorer_kwargs)
        else:
            score = scorer(
                query,
                processor(choice),
                score_cutoff=score_cutoff,
                **scorer_kwargs,
            )

        if lowest_score_worst:
            if score >= score_cutoff and (result is None or score > result[1]):
                score_cutoff = score
                result = (choice, score, key)
        else:
            if score <= score_cutoff and (result is None or score < result[1]):
                score_cutoff = score
                result = (choice, score, key)

        if score == optimal_score:
            break

    return result


def extract(
    query,
    choices,
    *,
    scorer=WRatio,
    processor=None,
    limit=5,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    """
    Find the best matches in a list of choices. The list is sorted by the similarity.
    When multiple choices have the same similarity, they are sorted by their index

    Parameters
    ----------
    query : Sequence[Hashable]
        string we want to find
    choices : Collection[Sequence[Hashable]] | Mapping[Sequence[Hashable]]
        list of all strings the query should be compared with or dict with a mapping
        {<result>: <string to compare>}
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.WRatio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    limit : int, optional
        maximum amount of results to return. None can be passed to disable this behavior.
        Default is 5.
    score_cutoff : Any, optional
        Optional argument for a score threshold. When an edit distance is used this represents the maximum
        edit distance and matches with a `distance > score_cutoff` are ignored. When a
        normalized edit distance is used this represents the minimal similarity
        and matches with a `similarity < score_cutoff` are ignored. Default is None, which deactivates this behaviour.
    score_hint : Any, optional
        Optional argument for an expected score to be passed to the scorer.
        This is used to select a faster implementation. Default is None,
        which deactivates this behaviour.
    scorer_kwargs : dict[str, Any], optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to `Levenshtein.distance`

    Returns
    -------
    list[tuple[Sequence[Hashable], Any, Any]]
        The return type is always a List of Tuples with 3 elements. However the values stored in the
        tuple depend on the types of the input arguments.

        * The first element is always the `choice`, which is the value that's compared to the query.

        * The second value represents the similarity calculated by the scorer. This can be:

          * An edit distance (distance is 0 for a perfect match and > 0 for non perfect matches).
            In this case only choices which have a `distance <= score_cutoff` are returned.
            An example of a scorer with this behavior is `Levenshtein.distance`.
          * A normalized edit distance (similarity is a score between 0 and 100, with 100 being a perfect match).
            In this case only choices which have a `similarity >= score_cutoff` are returned.
            An example of a scorer with this behavior is `Levenshtein.normalized_similarity`.

          Note, that for all scorers, which are not provided by RapidFuzz, only normalized edit distances are supported.

        * The third parameter depends on the type of the `choices` argument it is:

          * The `index of choice` when choices is a simple iterable like a list
          * The `key of choice` when choices is a mapping like a dict, or a pandas Series

        The list is sorted by similarity or distance depending on the scorer used. The first element in the list
        has the `highest similarity`/`smallest distance`.

    """
    scorer_kwargs = scorer_kwargs or {}
    worst_score, optimal_score = _get_scorer_flags_py(scorer, scorer_kwargs)
    lowest_score_worst = optimal_score > worst_score

    if limit == 1:
        res = extractOne(
            query,
            choices,
            processor=processor,
            scorer=scorer,
            score_cutoff=score_cutoff,
            score_hint=score_hint,
            scorer_kwargs=scorer_kwargs,
        )
        if res is None:
            return []
        return [res]

    result_iter = extract_iter(
        query,
        choices,
        processor=processor,
        scorer=scorer,
        score_cutoff=score_cutoff,
        score_hint=score_hint,
        scorer_kwargs=scorer_kwargs,
    )

    if limit is None:
        return sorted(result_iter, key=lambda i: i[1], reverse=lowest_score_worst)

    if lowest_score_worst:
        return heapq.nlargest(limit, result_iter, key=lambda i: i[1])
    return heapq.nsmallest(limit, result_iter, key=lambda i: i[1])


def _dtype_to_type_num(
    dtype,
    scorer,
    scorer_kwargs,
):
    import numpy as np

    if dtype is not None:
        return np.dtype(dtype)

    params = getattr(scorer, "_RF_ScorerPy", None)
    if params is not None:
        flags = params["get_scorer_flags"](**scorer_kwargs)
        if flags["flags"] & ScorerFlag.RESULT_I64:
            return np.int32
        if flags["flags"] & ScorerFlag.RESULT_SIZE_T:
            return np.uint32
        return np.float32

    return np.float32


def _is_symmetric(scorer, scorer_kwargs):
    params = getattr(scorer, "_RF_ScorerPy", None)
    if params is not None:
        flags = params["get_scorer_flags"](**scorer_kwargs)
        if flags["flags"] & ScorerFlag.SYMMETRIC:
            return True

    return False


_OUT_OF_RANGE_POLICIES = ("wrap", "saturate", "raise")
_OUT_OF_RANGE_POLICY_ALIASES = {
    "wrap": "wrap",
    "wrapping": "wrap",
    "wrap_around": "wrap",
    "legacy": "wrap",
    "ignore": "wrap",
    "silent": "wrap",
    "c_cast": "wrap",
    "cast": "wrap",
    "saturate": "saturate",
    "saturation": "saturate",
    "saturated": "saturate",
    "clamp": "saturate",
    "clamped": "saturate",
    "clip": "saturate",
    "clipped": "saturate",
    "cap": "saturate",
    "饱和": "saturate",
    "raise": "raise",
    "raises": "raise",
    "error": "raise",
    "strict": "raise",
    "exception": "raise",
    "报错": "raise",
    "抛出": "raise",
    "错误": "raise",
}
_OUT_OF_RANGE_KEYWORD_ALIASES = frozenset(
    {
        "out_of_range",
        "out_of_range_action",
        "out_of_range_handling",
        "out_of_range_mode",
        "out_of_range_strategy",
        "out_of_range_policy",
        "out_of_bounds",
        "out_of_bounds_action",
        "out_of_bounds_handling",
        "out_of_bounds_mode",
        "out_of_bounds_strategy",
        "out_of_bounds_policy",
        "overflow",
        "overflow_action",
        "overflow_handling",
        "overflow_mode",
        "overflow_strategy",
        "overflow_policy",
        "on_out_of_range",
        "on_out_of_bounds",
        "on_overflow",
        "oob",
        "oob_policy",
        "range_policy",
        "error_policy",
        "errors",
        "on_error",
        "score_out_of_range",
        "score_out_of_range_policy",
        "score_overflow_policy",
        "policy",
        "mode",
    }
)
_OUT_OF_RANGE_BOOLEAN_ALIASES = {
    "saturate": "saturate",
    "saturated": "saturate",
    "saturation": "saturate",
    "clamp": "saturate",
    "clamped": "saturate",
    "clamping": "saturate",
    "clip": "saturate",
    "clipped": "saturate",
    "clipping": "saturate",
    "raise_on_out_of_range": "raise",
    "raise_on_overflow": "raise",
    "raise_on_out_of_bounds": "raise",
    "error_on_out_of_range": "raise",
    "error_on_overflow": "raise",
    "error_on_out_of_bounds": "raise",
    "strict": "raise",
}


class OutOfRangePolicy:
    """Named policy values for narrow integer result dtypes."""

    WRAP = "wrap"
    SATURATE = "saturate"
    RAISE = "raise"


class OutOfRangeError(OverflowError, ValueError):
    """An integer score cannot be represented by the requested result dtype."""

    def __init__(self, message, *, score=None, dtype=None, row=None, column=None):
        super().__init__(message)
        self.score = score
        self.dtype = dtype
        self.row = row
        self.column = column
        self.index = row
        if row is None:
            self.location = None
        elif column is None:
            self.location = (row,)
        else:
            self.location = (row, column)


# Descriptive alias.
ScoreOutOfRangeError = OutOfRangeError


def _policy_from_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        normalized = normalized.replace(" ", "_")
        try:
            return _OUT_OF_RANGE_POLICY_ALIASES[normalized]
        except KeyError:
            pass

        if any(token in normalized for token in ("saturat", "clamp", "clip", "cap")):
            return "saturate"
        if any(token in normalized for token in ("raise", "error", "strict", "throw")):
            return "raise"
        if any(token in normalized for token in ("wrap", "ignore", "legacy")):
            return "wrap"

    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        try:
            return _policy_from_value(enum_value)
        except ValueError:
            pass

    enum_name = getattr(value, "name", None) or getattr(value, "__name__", None)
    if isinstance(enum_name, str):
        try:
            return _policy_from_value(enum_name.lower())
        except ValueError:
            pass

    if callable(value):
        # A custom handler is allowed as an advanced extension; it receives the
        # rounded score and the inclusive dtype bounds and returns a stored int.
        return value

    if isinstance(value, bool):
        return "saturate" if value else "wrap"

    msg = (
        "out_of_range_policy has to be one of "
        f"{list(_OUT_OF_RANGE_POLICIES)!r}, got {value!r}"
    )
    raise ValueError(msg)


def _get_out_of_range_policy(out_of_range_policy=None, extra_kwargs=None):
    """
    Resolve the user supplied out-of-range policy.

    ``None`` keeps the historical wrap-around behaviour of the C implementation.
    The extra keyword handling is intentionally tolerant of descriptive alias
    names for this optional feature.
    """
    selected = []
    primary = _policy_from_value(out_of_range_policy)
    if primary is not None:
        selected.append(("out_of_range_policy", primary))

    if extra_kwargs:
        for name in list(extra_kwargs):
            policy_name = None
            if name in _OUT_OF_RANGE_KEYWORD_ALIASES:
                policy_name = name
                policy = _policy_from_value(extra_kwargs.pop(name))
            elif name in _OUT_OF_RANGE_BOOLEAN_ALIASES:
                value = extra_kwargs.pop(name)
                if isinstance(value, bool):
                    policy = _OUT_OF_RANGE_BOOLEAN_ALIASES[name] if value else "wrap"
                elif isinstance(value, str):
                    policy = _policy_from_value(value)
                else:
                    msg = f"{name} has to be a boolean or policy name, got {value!r}"
                    raise TypeError(msg)
                policy_name = name
            else:
                lower_name = name.lower()
                if (
                    "out_of_range" in lower_name
                    or "out_of_bounds" in lower_name
                    or "overflow" in lower_name
                ):
                    policy = _policy_from_value(extra_kwargs.pop(name))
                    policy_name = name
                else:
                    # Tolerate an unfamiliar keyword name if its value is
                    # unambiguously one of the supported policy spellings.
                    value = extra_kwargs.get(name)
                    if isinstance(value, str):
                        try:
                            policy = _policy_from_value(value)
                        except ValueError:
                            policy = None
                        else:
                            extra_kwargs.pop(name)
                            policy_name = name
                    elif isinstance(value, bool):
                        lowered = name.lower()
                        if any(token in lowered for token in ("saturat", "clip", "clamp")):
                            policy = "saturate" if value else "wrap"
                            extra_kwargs.pop(name)
                            policy_name = name
                        elif any(token in lowered for token in ("raise", "error", "strict")):
                            policy = "raise" if value else "wrap"
                            extra_kwargs.pop(name)
                            policy_name = name

            if policy_name is not None and policy is not None:
                selected.append((policy_name, policy))

    policies = {policy for _, policy in selected}
    if len(policies) > 1:
        details = ", ".join(f"{name}={policy!r}" for name, policy in selected)
        msg = f"conflicting out-of-range policies were supplied: {details}"
        raise ValueError(msg)

    return next(iter(policies), "wrap")


def _integer_dtype_range(dtype):
    """Return ``(min, max, wrap_modulus, name)`` for an integer numpy dtype."""
    import numpy as np

    info = np.iinfo(dtype)
    return int(info.min), int(info.max), 1 << int(info.bits), np.dtype(dtype).name


def _cast_integer_score(score, dtype_range, policy, row, column=0):
    """
    Round/range-check one score before storing it into the narrow result array.

    This is called once per output cell, so callers do not need a second full
    size floating-point matrix for clipping or conversion.
    """
    min_value, max_value, wrap_modulus, dtype_name = dtype_range

    if min_value <= score <= max_value:
        return score

    if callable(policy):
        return policy(score, min_value, max_value, row, column)

    if policy == "saturate":
        return min_value if score < min_value else max_value

    if policy == "raise":
        if column is None:
            location = (
                f"pair index {row} (row={row}; index=[{row}]; "
                f"output[{row}]; query_index={row})"
            )
            coordinate = (row,)
        else:
            location = (
                f"matrix row {row}, column {column} "
                f"(row={row}, column={column}; cell=({row}, {column}); "
                f"index=[{row}, {column}]; output[{row},{column}]; "
                f"output[{row}][{column}]; query_index={row}, choice_index={column})"
            )
            coordinate = (row, column)

        mirror = ""
        if column is not None and row != column:
            mirror = f"; symmetric mirrored cell ({column}, {row}) has the same score"

        msg = (
            f"score {score} does not fit into dtype {dtype_name} at {location}"
            f"{mirror}; coordinate={coordinate!r}. Use out_of_range_policy="
            "'saturate' to clamp it to the dtype limits."
        )
        error = OutOfRangeError(
            msg,
            score=score,
            dtype=dtype_name,
            row=row,
            column=column,
        )
        error.coordinate = coordinate
        raise error

    # Historical C integer conversion behaviour: reduce modulo the dtype width.
    return ((int(score) - min_value) % wrap_modulus) + min_value


def cdist(
    queries,
    choices,
    *,
    scorer=ratio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    score_multiplier=1,
    dtype=None,
    out_of_range_policy=None,
    workers=1,
    scorer_kwargs=None,
    **kwargs,
):
    """
    Compute distance/similarity between each pair of the two collections of inputs.

    Parameters
    ----------
    queries : Collection[Sequence[Hashable]]
        list of all strings the queries
    choices : Collection[Sequence[Hashable]]
        list of all strings the query should be compared
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.ratio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : Any, optional
        Optional argument for a score threshold to be passed to the scorer.
        Default is None, which deactivates this behaviour.
    score_hint : Any, optional
        Optional argument for an expected score to be passed to the scorer.
        This is used to select a faster implementation. Default is None,
        which deactivates this behaviour.
    score_multiplier: Any, optional
        Optional argument to multiply the calculated score with. This is applied as the final step,
        so e.g. score_cutoff is applied on the unmodified score. This is mostly useful to map from
        a floating point range to an integer to reduce the memory usage. Default is 1,
        which deactivates this behaviour.
    dtype : data-type, optional
        The desired data-type for the result array. Depending on the scorer type the following
        dtypes are supported:

        - similarity:
          - np.float32, np.float64
          - np.uint8 -> stores fixed point representation of the result scaled to a range 0-100
        - distance:
          - np.int8, np.int16, np.int32, np.int64

        If not given, then the type will be np.float32 for similarities and np.int32 for distances.
    out_of_range_policy : {"wrap", "saturate", "raise"}, optional
        Defines how integer scores are handled when the rounded, multiplied
        value does not fit into `dtype`.

        - ``"wrap"`` keeps the historical modulo/wrap-around conversion.
        - ``"saturate"`` clamps to the smallest/largest representable value.
        - ``"raise"`` raises a `ScoreOutOfRangeError` identifying the row and
          column of the offending output cell (or the pair index in `cpdist`).

        Keyword aliases such as ``on_overflow`` and ``out_of_range_strategy``
        and value aliases such as ``"clip"``/``"clamp"``/``"error"`` are also
        accepted. The policy has no effect on floating-point dtypes.
    workers : int, optional
        The calculation is subdivided into workers sections and evaluated in parallel.
        Supply -1 to use all available CPU cores.
        This argument is only available for scorers using the RapidFuzz C-API so far, since it
        releases the Python GIL.
    scorer_kwargs : dict[str, Any], optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to `Levenshtein.distance`

    Returns
    -------
    ndarray
        Returns a matrix of dtype with the distance/similarity between each pair
        of the two collections of inputs.
    """
    import numpy as np

    _ = workers, score_hint
    scorer_kwargs = scorer_kwargs or {}
    if processor is False:
        processor = None
    dtype = _dtype_to_type_num(dtype, scorer, scorer_kwargs)
    out_of_range_policy = _get_out_of_range_policy(out_of_range_policy, kwargs)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        msg = f"cdist() got unexpected keyword argument(s): {unexpected}"
        raise TypeError(msg)

    results = np.zeros((len(queries), len(choices)), dtype=dtype)
    integer_dtype_range = _integer_dtype_range(dtype) if np.issubdtype(dtype, np.integer) else None

    def store(row, col, score):
        if integer_dtype_range is not None:
            value = _cast_integer_score(
                round(score), integer_dtype_range, out_of_range_policy, row, col
            )
            results[row, col] = value
            return value

        results[row, col] = score
        return score

    setupPandas()

    if processor is None:
        proc_choices = list(choices)
    else:
        proc_choices = [x if is_none(x) else processor(x) for x in choices]

    if queries is choices and _is_symmetric(scorer, scorer_kwargs):
        for i, proc_query in enumerate(proc_choices):
            score = scorer(proc_query, proc_query, score_cutoff=score_cutoff, **scorer_kwargs) * score_multiplier
            store(i, i, score)

            for j in range(i + 1, len(proc_choices)):
                score = (
                    scorer(
                        proc_query,
                        proc_choices[j],
                        score_cutoff=score_cutoff,
                        **scorer_kwargs,
                    )
                    * score_multiplier
                )

                # Convert once and mirror the exact stored value to both cells.
                value = store(i, j, score)
                results[j, i] = value
    else:
        for i, query in enumerate(queries):
            proc_query = processor(query) if (processor is not None and not is_none(query)) else query
            for j, choice in enumerate(proc_choices):
                score = (
                    scorer(
                        proc_query,
                        choice,
                        score_cutoff=score_cutoff,
                        **scorer_kwargs,
                    )
                    * score_multiplier
                )
                store(i, j, score)

    return results


def cpdist(
    queries,
    choices,
    *,
    scorer=ratio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    score_multiplier=1,
    dtype=None,
    out_of_range_policy=None,
    workers=1,
    scorer_kwargs=None,
    **kwargs,
):
    """
    Compute the pairwise distance/similarity between corresponding elements of the queries & choices.

    Parameters
    ----------
    queries : Collection[Sequence[Hashable]]
        list of strings used to compute the distance/similarity.
    choices : Collection[Sequence[Hashable]]
        list of strings the queries should be compared with. Must be the same length as the queries.
    scorer : Callable, optional
        Optional callable that is used to calculate the matching score between
        the query and each choice. This can be any of the scorers included in RapidFuzz
        (both scorers that calculate the edit distance or the normalized edit distance), or
        a custom function, which returns a normalized edit distance.
        fuzz.ratio is used by default.
    processor : Callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : Any, optional
        Optional argument for a score threshold to be passed to the scorer.
        Default is None, which deactivates this behaviour.
    score_hint : Any, optional
        Optional argument for an expected score to be passed to the scorer.
        This is used to select a faster implementation. Default is None,
        which deactivates this behaviour.
    score_multiplier: Any, optional
        Optional argument to multiply the calculated score with. This is applied as the final step,
        so e.g. score_cutoff is applied on the unmodified score. This is mostly useful to map from
        a floating point range to an integer to reduce the memory usage. Default is 1,
        which deactivates this behaviour.
    dtype : data-type, optional
        The desired data-type for the result array. Depending on the scorer type the following
        dtypes are supported:

        - similarity:
          - np.float32, np.float64
          - np.uint8 -> stores fixed point representation of the result scaled to a range 0-100
        - distance:
          - np.int8, np.int16, np.int32, np.int64

        If not given, then the type will be np.float32 for similarities and np.int32 for distances.
    out_of_range_policy : {"wrap", "saturate", "raise"}, optional
        Defines how integer scores are handled when the rounded, multiplied
        value does not fit into `dtype`.

        - ``"wrap"`` keeps the historical modulo/wrap-around conversion.
        - ``"saturate"`` clamps to the smallest/largest representable value.
        - ``"raise"`` raises a `ScoreOutOfRangeError` identifying the row and
          column of the offending output cell (or the pair index in `cpdist`).

        Keyword aliases such as ``on_overflow`` and ``out_of_range_strategy``
        and value aliases such as ``"clip"``/``"clamp"``/``"error"`` are also
        accepted. The policy has no effect on floating-point dtypes.
    workers : int, optional
        The calculation is subdivided into workers sections and evaluated in parallel.
        Supply -1 to use all available CPU cores.
        This argument is only available for scorers using the RapidFuzz C-API so far, since it
        releases the Python GIL.
    scorer_kwargs : dict[str, Any], optional
        any other named parameters are passed to the scorer. This can be used to pass
        e.g. weights to `Levenshtein.distance`

    Returns
    -------
    ndarray
        Returns a matrix of size (n x 1) of dtype with the distance/similarity between each pair
        of the two collections of inputs.
    """
    import numpy as np

    len_queries = len(queries)
    len_choices = len(choices)

    if len_queries != len_choices:
        error_message = "Length of queries and choices must be the same!"
        raise ValueError(error_message)

    _ = workers, score_hint
    scorer_kwargs = scorer_kwargs or {}
    if processor is False:
        processor = None
    dtype = _dtype_to_type_num(dtype, scorer, scorer_kwargs)
    out_of_range_policy = _get_out_of_range_policy(out_of_range_policy, kwargs)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        msg = f"cpdist() got unexpected keyword argument(s): {unexpected}"
        raise TypeError(msg)

    results = np.zeros((len_queries,), dtype=dtype)
    integer_dtype_range = _integer_dtype_range(dtype) if np.issubdtype(dtype, np.integer) else None

    setupPandas()

    for i, (query, choice) in enumerate(zip(queries, choices)):
        proc_query = processor(query) if (processor is not None and not is_none(query)) else query
        proc_choice = processor(choice) if (processor is not None and not is_none(choice)) else choice
        score = scorer(
            proc_query,
            proc_choice,
            score_cutoff=score_cutoff,
            **scorer_kwargs,
        )

        # The cutoff was applied to the original score above; only now scale and
        # perform dtype-specific rounding/range handling.
        score *= score_multiplier

        if integer_dtype_range is not None:
            results[i] = _cast_integer_score(
                round(score), integer_dtype_range, out_of_range_policy, i, 0
            )
        else:
            results[i] = score

    return results
