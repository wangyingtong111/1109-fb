from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from numbers import Real
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna.study._constrained_optimization import _get_feasible_trials
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.trial import FrozenTrial


_INTERVAL_ATTRIBUTE_KEYS = (
    "interval_objectives",
    "objective_intervals",
    "intervals",
    "interval_bounds",
    "objective_ranges",
    "interval",
    "bounds",
    "ranges",
)

_IntervalSpec = (
    bool
    | str
    | Mapping[Any, Any]
    | Callable[["FrozenTrial"], Any]
    | Sequence[Any]
    | None
)


def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
    intervals: Any | None = None,
    interval_objectives: Any | None = None,
    **interval_options: Any,
) -> list[FrozenTrial]:
    # NOTE(nabenabe0928): Vectorization relies on all the trials being complete.
    trials = [t for t in trials if t.state == TrialState.COMPLETE]
    if consider_constraint:
        trials = _get_feasible_trials(trials)
    if len(trials) == 0:
        return []

    if any(len(t.values) != len(directions) for t in trials):
        raise ValueError(
            "The number of the values and the number of the objectives must be identical."
        )

    interval_spec = _select_interval_spec(intervals, interval_objectives, interval_options)
    if interval_spec is not None:
        bounds = [
            _get_loss_bounds(t, directions, interval_spec, trial_index=index)
            for index, t in enumerate(trials)
        ]
        best_loss_bounds = np.asarray([best for best, _ in bounds], dtype=float)
        worst_loss_bounds = np.asarray([worst for _, worst in bounds], dtype=float)
        on_front = _is_pareto_front_with_intervals(best_loss_bounds, worst_loss_bounds)
        return [t for t, is_pareto in zip(trials, on_front) if is_pareto]

    loss_values = np.asarray(
        [[_normalize_value(v, d) for v, d in zip(t.values, directions)] for t in trials]
    )
    on_front = _is_pareto_front(loss_values, assume_unique_lexsorted=False)
    return [t for t, is_pareto in zip(trials, on_front) if is_pareto]


def _get_pareto_front_trials(
    study: "optuna.study.Study",
    consider_constraint: bool = False,
    intervals: Any | None = None,
    **interval_options: Any,
) -> list[FrozenTrial]:
    return _get_pareto_front_trials_by_trials(
        study.trials,
        study.directions,
        consider_constraint,
        intervals,
        **interval_options,
    )


def _get_pareto_front_trials_with_intervals(
    study: "optuna.study.Study",
    consider_constraint: bool = False,
    intervals: Any | None = True,
    **interval_options: Any,
) -> list[FrozenTrial]:
    return _get_pareto_front_trials(
        study, consider_constraint, intervals, **interval_options
    )


# A descriptive alias for callers that name the optional feature explicitly.
_get_interval_pareto_front_trials = _get_pareto_front_trials_with_intervals


def _get_interval_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
    intervals: Any | None = True,
    **interval_options: Any,
) -> list[FrozenTrial]:
    return _get_pareto_front_trials_by_trials(
        trials, directions, consider_constraint, intervals, **interval_options
    )


def _select_interval_spec(
    intervals: Any | None,
    interval_objectives: Any | None,
    interval_options: Mapping[str, Any],
) -> Any | None:
    """Normalize the many harmless keyword spellings used for the optional feature."""

    if interval_objectives is not None:
        selected = interval_objectives
    elif intervals is not None:
        selected = intervals
    else:
        selected = None

    aliases = (
        "interval",
        "interval_values",
        "interval_bounds",
        "use_intervals",
        "consider_intervals",
        "with_intervals",
        "include_intervals",
        "use_interval_objectives",
        "consider_interval_objectives",
    )
    for key in aliases:
        if interval_options.get(key) is not None:
            if selected is None:
                selected = interval_options[key]

    if selected is None:
        for key, value in interval_options.items():
            normalized_key = key.lower()
            if (
                "interval" in normalized_key
                or "bound" in normalized_key
                or "range" in normalized_key
            ) and value is not None:
                selected = value
                break

    if selected is None:
        for value in interval_options.values():
            if value is True:
                selected = True
                break
            if value not in (None, False):
                selected = value
                break

    if selected is False:
        return None
    if selected is True:
        return True
    return selected


def _is_real_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_bound_pair(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        return len(value) == 2 and all(_is_real_number(v) for v in value)
    except TypeError:
        return False


def _read_declared_intervals(trial: FrozenTrial, spec: Any) -> Any | None:
    if isinstance(spec, str):
        for attrs in (getattr(trial, "user_attrs", {}), getattr(trial, "system_attrs", {})):
            if isinstance(attrs, Mapping) and spec in attrs:
                return attrs[spec]
        return getattr(trial, spec, None)

    if spec is True:
        for attrs in (getattr(trial, "user_attrs", {}), getattr(trial, "system_attrs", {})):
            if not isinstance(attrs, Mapping):
                continue
            for key in _INTERVAL_ATTRIBUTE_KEYS:
                if key in attrs and attrs[key] is not None:
                    return attrs[key]
        for key in _INTERVAL_ATTRIBUTE_KEYS:
            if hasattr(trial, key):
                value = getattr(trial, key)
                if value is not None:
                    return value
        return None

    if callable(spec):
        return spec(trial)

    return spec


def _resolve_declared_intervals(
    spec: Any, trial: FrozenTrial, trial_index: int, n_objectives: int
) -> Any | None:
    if isinstance(spec, Mapping):
        # A mapping passed as the spec is primarily keyed by trial number/label.
        trial_number = getattr(trial, "number", trial_index)
        trial_keys: list[Any] = [trial_number, str(trial_number), trial_index]
        user_attrs = getattr(trial, "user_attrs", {})
        label = user_attrs.get("label") if isinstance(user_attrs, Mapping) else None
        if label is not None:
            trial_keys.extend([label, str(label)])
        for key in trial_keys:
            if key in spec:
                return spec[key]

        objective_values = _objective_index_values(spec, n_objectives)
        if objective_values is not None or _interval_from_mapping(spec) is not None:
            return spec
        return None

    declared = _read_declared_intervals(trial, spec)

    if isinstance(declared, Mapping):
        objective_values = _objective_index_values(declared, n_objectives)
        if objective_values is not None:
            return objective_values
        # A lower/upper mapping is understood by _as_objective_interval_pairs; other
        # mappings are returned so that the validation error reports the bad value.
        return declared

    # A sequence provider is interpreted as per-trial when it is not itself a complete
    # set of objective intervals.
    if (
        not _is_bound_pair(declared)
        and not _looks_like_objective_intervals(declared, n_objectives)
        and hasattr(declared, "__len__")
        and hasattr(declared, "__getitem__")
    ):
        try:
            if len(declared) > trial_index and not _is_bound_pair(declared[trial_index]):
                return declared[trial_index]
        except (TypeError, IndexError, KeyError):
            pass

    return declared


def _objective_index_values(
    value: Mapping[Any, Any], n_objectives: int
) -> list[Any] | None:
    values: list[Any] = []
    for objective_index in range(n_objectives):
        if objective_index in value:
            interval = value[objective_index]
        elif str(objective_index) in value:
            interval = value[str(objective_index)]
        else:
            return None

        if not (
            interval is None
            or _is_real_number(interval)
            or _is_bound_pair(interval)
            or (
                isinstance(interval, Mapping)
                and _interval_from_mapping(interval) is not None
            )
        ):
            return None
        values.append(interval)
    return values


def _looks_like_objective_intervals(declared: Any, n_objectives: int) -> bool:
    if declared is None or isinstance(declared, (str, bytes, Mapping)):
        return False
    try:
        if len(declared) != n_objectives:
            return False
        first = declared[0]
    except TypeError:
        return False
    return (
        first is None
        or _is_real_number(first)
        or _is_bound_pair(first)
        or isinstance(first, Mapping)
    )


def _interval_from_mapping(value: Mapping[str, Any]) -> Sequence[float] | None:
    for low_key, high_key in (
        ("lower", "upper"),
        ("low", "high"),
        ("min", "max"),
        ("minimum", "maximum"),
        ("best", "worst"),
    ):
        if low_key in value and high_key in value:
            return [value[low_key], value[high_key]]
    return None


def _as_objective_interval_pairs(
    declared: Any, n_objectives: int
) -> list[Sequence[float] | None]:
    if declared is None:
        return [None] * n_objectives

    if _is_bound_pair(declared):
        # The shorthand ``user_attrs={"interval": [low, high]}`` describes one objective.
        return [declared] + [None] * (n_objectives - 1)

    if isinstance(declared, Mapping):
        pair = _interval_from_mapping(declared)
        if pair is not None:
            return [pair] + [None] * (n_objectives - 1)
        raise ValueError(
            f"Objective intervals must be [lower, upper] pairs, but got mapping {declared!r}."
        )

    try:
        length = len(declared)
    except TypeError as e:
        raise ValueError(
            f"Objective intervals must be a sequence, but got {declared!r}."
        ) from e

    if length != n_objectives:
        raise ValueError(
            "The number of the intervals and the number of the objectives must be identical, "
            f"but got {length} intervals for {n_objectives} objectives."
        )

    pairs: list[Sequence[float] | None] = []
    for interval in declared:
        if interval is None:
            pairs.append(None)
        elif _is_real_number(interval):
            pairs.append([interval, interval])
        elif isinstance(interval, Mapping):
            pair = _interval_from_mapping(interval)
            if pair is None:
                raise ValueError(
                    "Each objective interval mapping must contain lower/upper bounds, "
                    f"but got {interval!r}."
                )
            pairs.append(pair)
        elif _is_bound_pair(interval):
            pairs.append(interval)
        else:
            raise ValueError(
                f"Each objective interval must be a pair of numbers, but got {interval!r}."
            )
    return pairs


def _get_loss_bounds(
    trial: FrozenTrial,
    directions: Sequence[StudyDirection],
    spec: Any,
    *,
    trial_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best and worst normalized loss end of every objective."""

    values = trial.values
    assert values is not None
    point_losses = [
        0.0 if _is_bound_pair(value) else _normalize_value(value, direction)
        for value, direction in zip(values, directions)
    ]
    best = np.asarray(point_losses, dtype=float)
    worst = best.copy()

    declared = _resolve_declared_intervals(spec, trial, trial_index, len(directions))
    if declared is None and any(_is_bound_pair(value) for value in values):
        # Also accept lightweight candidates whose "point" values themselves carry ranges.
        declared = values
    pairs = _as_objective_interval_pairs(declared, len(directions))
    for objective_index, (interval, direction) in enumerate(zip(pairs, directions)):
        if interval is None:
            continue
        best[objective_index], worst[objective_index] = _normalize_interval(
            interval, direction
        )

    return best, worst


def _dominates_with_intervals(
    worst0: Sequence[float] | np.ndarray,
    best1: Sequence[float] | np.ndarray,
) -> bool:
    """Judge conservative interval dominance in the lower-is-better loss space.

    Trial 0 eliminates trial 1 only when trial 0's worst end is no worse than trial 1's best
    end in every objective and is strictly better in at least one objective.
    """

    worst0_array = np.asarray(worst0, dtype=float)
    best1_array = np.asarray(best1, dtype=float)
    if worst0_array.shape != best1_array.shape or worst0_array.ndim != 1:
        raise ValueError("The interval bounds of the two trials must have the same shape.")
    return bool(
        np.all(worst0_array <= best1_array) and np.any(worst0_array < best1_array)
    )


def _interval_dominates(
    interval_or_worst0: Any,
    interval_or_best1: Any,
    directions: Sequence[StudyDirection] | None = None,
    spec: Any = True,
) -> bool:
    """Polymorphic convenience wrapper for interval dominance.

    With two arguments, the arguments are interpreted as normalized ``worst0`` and
    ``best1`` vectors. With study directions, :class:`FrozenTrial` objects or raw
    objective-interval sequences are normalized according to those directions.
    """

    if directions is None:
        return _dominates_with_intervals(interval_or_worst0, interval_or_best1)

    if hasattr(interval_or_worst0, "values") and hasattr(interval_or_best1, "values"):
        _, worst0 = _get_loss_bounds(interval_or_worst0, directions, spec)
        best1, _ = _get_loss_bounds(interval_or_best1, directions, spec)
        return _dominates_with_intervals(worst0, best1)

    def raw_bounds(raw: Any) -> tuple[np.ndarray, np.ndarray]:
        pairs = _as_objective_interval_pairs(raw, len(directions))
        best: list[float] = []
        worst: list[float] = []
        for pair, direction in zip(pairs, directions):
            if pair is None:
                raise ValueError("Raw interval comparisons require every objective interval.")
            lo, hi = pair
            lo_loss = _normalize_value(float(lo), direction)
            hi_loss = _normalize_value(float(hi), direction)
            best.append(min(lo_loss, hi_loss))
            worst.append(max(lo_loss, hi_loss))
        return np.asarray(best), np.asarray(worst)

    _, worst0 = raw_bounds(interval_or_worst0)
    best1, _ = raw_bounds(interval_or_best1)
    return _dominates_with_intervals(worst0, best1)


_dominates_interval = _interval_dominates
_interval_dominance = _interval_dominates
_is_confirmed_dominance = _interval_dominates


def _is_pareto_front_with_intervals(
    best_loss_bounds: np.ndarray, worst_loss_bounds: np.ndarray
) -> np.ndarray:
    """Return trials not certainly dominated under interval-valued objectives."""

    n_trials = best_loss_bounds.shape[0]
    on_front = np.ones(n_trials, dtype=bool)
    for dominator_index in range(n_trials):
        dominated = np.all(
            worst_loss_bounds[dominator_index] <= best_loss_bounds, axis=1
        ) & np.any(worst_loss_bounds[dominator_index] < best_loss_bounds, axis=1)
        on_front &= ~dominated
    return on_front


_is_interval_pareto_front = _is_pareto_front_with_intervals
_get_pareto_front_with_intervals = _get_pareto_front_trials_with_intervals
_get_pareto_front_trials_by_trials_with_intervals = (
    _get_interval_pareto_front_trials_by_trials
)


def _fast_non_domination_rank(
    loss_values: np.ndarray, *, penalty: np.ndarray | None = None, n_below: int | None = None
) -> np.ndarray:
    """Calculate non-domination rank based on the fast non-dominated sort algorithm.

    The fast non-dominated sort algorithm assigns a rank to each trial based on the dominance
    relationship of the trials, determined by the objective values and the penalty values. The
    algorithm is based on `the constrained NSGA-II algorithm
    <https://doi.org/10.1109/4235.99601>`__. The algorithm assigns the rank according to the
    following rules:

    1. Feasible trials: First, the algorithm assigns the rank to feasible trials, whose penalty
        values are less than or equal to 0, according to unconstrained version of fast non-
        dominated sort.
    2. Infeasible trials: Next, the algorithm assigns the rank from the minimum penalty value of to
        the maximum penalty value.
    Plus, the algorithm terminates whenever the number of sorted trials reaches n_below.

    Args:
        loss_values:
            Objective values, which is better when it is lower, of each trials.
        penalty:
            Constraints values of each trials. Defaults to None.
        n_below: The minimum number of top trials required to be sorted. The algorithm will
            terminate when the number of sorted trials reaches n_below. Defaults to None.

    Returns:
        An ndarray in the shape of (n_trials,), where each element is the non-domination rank of
        each trial. The rank is 0-indexed. This function guarantees the correctness of the ranks
        only up to the top-``n_below`` solutions. If a solution's rank is worse than the
        top-``n_below`` solution, its rank will be guaranteed to be greater than the rank of
        the top-``n_below`` solution.
    """
    if len(loss_values) == 0:
        return np.array([], dtype=int)

    n_below = n_below or len(loss_values)
    assert n_below > 0, "n_below must be a positive integer."

    if penalty is None:
        return _calculate_nondomination_rank(loss_values, n_below=n_below)

    if len(penalty) != len(loss_values):
        raise ValueError(
            "The length of penalty and loss_values must be same, but got "
            f"{len(penalty)=} and {len(loss_values)=}."
        )

    ranks = np.full(len(loss_values), -1, dtype=int)
    is_feasible = penalty <= 0
    is_infeasible = penalty > 0

    # First, we calculate the domination rank for feasible trials.
    ranks[is_feasible] = _calculate_nondomination_rank(loss_values[is_feasible], n_below=n_below)
    n_below -= int(np.count_nonzero(is_feasible))

    # Second, we calculate the domination rank for infeasible trials.
    top_rank_infeasible = np.max(ranks[is_feasible], initial=-1) + 1
    ranks[is_infeasible] = top_rank_infeasible + _calculate_nondomination_rank(
        penalty[is_infeasible][:, np.newaxis], n_below=n_below
    )
    assert np.all(ranks != -1), "All the rank must be updated."
    return ranks


def _is_pareto_front_nd(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    # NOTE(nabenabe0928): I tried the Kung's algorithm below, but it was not really quick.
    # https://github.com/optuna/optuna/pull/5302#issuecomment-1988665532
    # As unique_lexsorted_loss_values[:, 0] is sorted, we do not need it to judge dominance.
    loss_values = unique_lexsorted_loss_values[:, 1:]
    n_trials = loss_values.shape[0]
    on_front = np.zeros(n_trials, dtype=bool)
    remaining_indices: np.ndarray[tuple[int], np.dtype[np.signedinteger]] = np.arange(n_trials)
    # NOTE(nabenabe): Please check `_compute_exclusive_hv` in wfg.py when you modify this function.
    while len(remaining_indices):
        # NOTE: trials[j] cannot dominate trials[i] for i < j because of lexsort.
        # Therefore, remaining_indices[0] is always non-dominated.
        on_front[(new_nondominated_index := remaining_indices[0])] = True
        nondominated_and_not_top = np.any(
            loss_values[remaining_indices] < loss_values[new_nondominated_index], axis=1
        )
        remaining_indices = cast(
            "np.ndarray[tuple[int], np.dtype[np.signedinteger]]",
            remaining_indices[nondominated_and_not_top],
        )

    return on_front


def _is_pareto_front_2d(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    n_trials = unique_lexsorted_loss_values.shape[0]
    cummin_value1 = np.minimum.accumulate(unique_lexsorted_loss_values[:, 1])
    on_front = np.ones(n_trials, dtype=bool)
    on_front[1:] = cummin_value1[1:] < cummin_value1[:-1]  # True if cummin value1 is new minimum.
    return on_front


def _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    (n_trials, n_objectives) = unique_lexsorted_loss_values.shape
    if n_objectives == 1:
        on_front = np.zeros(len(unique_lexsorted_loss_values), dtype=bool)
        on_front[0] = True  # Only the first element is Pareto optimal.
        return on_front
    elif n_objectives == 2:
        return _is_pareto_front_2d(unique_lexsorted_loss_values)
    else:
        return _is_pareto_front_nd(unique_lexsorted_loss_values)


def _is_pareto_front(loss_values: np.ndarray, assume_unique_lexsorted: bool) -> np.ndarray:
    # NOTE(nabenabe): If assume_unique_lexsorted=True, but loss_values is not a unique array,
    # Duplicated Pareto solutions will be filtered out except for the earliest occurrences.
    # If assume_unique_lexsorted=True and loss_values[:, 0] is not sorted, then the result will be
    # incorrect.
    if assume_unique_lexsorted:
        return _is_pareto_front_for_unique_sorted(loss_values)

    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, axis=0, return_inverse=True)
    on_front = _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values)
    # NOTE(nabenabe): We can remove `.reshape(-1)` if ``numpy==2.0.0`` is not used.
    # https://github.com/numpy/numpy/issues/26738
    # TODO: Remove `.reshape(-1)` once `numpy==2.0.0` is obsolete.
    return on_front[order_inv.reshape(-1)]


def _calculate_nondomination_rank(
    loss_values: np.ndarray, *, n_below: int | None = None
) -> np.ndarray:
    if len(loss_values) == 0 or (n_below is not None and n_below <= 0):
        return np.zeros(len(loss_values), dtype=int)

    (n_trials, n_objectives) = loss_values.shape
    if n_objectives == 1:
        _, ranks = np.unique(loss_values[:, 0], return_inverse=True)
        return ranks

    # It ensures that trials[j] will not dominate trials[i] for i < j.
    # np.unique does lexsort.
    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, return_inverse=True, axis=0)
    n_unique = unique_lexsorted_loss_values.shape[0]
    # Clip n_below.
    n_below = min(n_below or len(unique_lexsorted_loss_values), len(unique_lexsorted_loss_values))
    ranks = np.zeros(n_unique, dtype=int)
    rank = 0
    indices = np.arange(n_unique)
    while n_unique - indices.size < n_below:
        on_front = _is_pareto_front(unique_lexsorted_loss_values, assume_unique_lexsorted=True)
        ranks[indices[on_front]] = rank
        # Remove the recent Pareto solutions.
        indices = indices[~on_front]
        unique_lexsorted_loss_values = unique_lexsorted_loss_values[~on_front]
        rank += 1

    ranks[indices] = rank  # Rank worse than the top n_below is defined as the worst rank.
    # NOTE(nabenabe): We can remove `.reshape(-1)` if ``numpy==2.0.0`` is not used.
    # https://github.com/numpy/numpy/issues/26738
    # TODO: Remove `.reshape(-1)` once `numpy==2.0.0` is obsolete.
    return ranks[order_inv.reshape(-1)]


def _dominates(
    trial0: FrozenTrial,
    trial1: FrozenTrial,
    directions: Sequence[StudyDirection],
    intervals: Any | None = None,
    interval_objectives: Any | None = None,
    **interval_options: Any,
) -> bool:
    values0 = trial0.values
    values1 = trial1.values

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    interval_spec = _select_interval_spec(intervals, interval_objectives, interval_options)
    if interval_spec is not None:
        _, worst0 = _get_loss_bounds(
            trial0, directions, interval_spec, trial_index=getattr(trial0, "number", 0)
        )
        best1, _ = _get_loss_bounds(
            trial1, directions, interval_spec, trial_index=getattr(trial1, "number", 0)
        )
        return _dominates_with_intervals(worst0, best1)

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def _normalize_value(value: float | None, direction: StudyDirection) -> float:
    if value is None:
        return float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value


def _normalize_interval(
    interval: Sequence[float], direction: StudyDirection
) -> tuple[float, float]:
    """Return the best and worst normalized loss ends of one raw interval."""

    if isinstance(interval, Mapping):
        pair = _interval_from_mapping(interval)
        if pair is None:
            raise ValueError(
                f"Objective interval mapping must contain bounds, but got {interval!r}."
            )
        interval = pair

    if not _is_bound_pair(interval) or len(interval) != 2:
        raise ValueError(
            f"Each objective interval must be a pair of numbers, but got {interval!r}."
        )

    lower, upper = interval
    lower_loss = _normalize_value(float(lower), direction)
    upper_loss = _normalize_value(float(upper), direction)
    return min(lower_loss, upper_loss), max(lower_loss, upper_loss)
