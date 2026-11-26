from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from decimal import Decimal
from decimal import ROUND_FLOOR
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


ObjectiveStep = int | float | Decimal
ObjectiveSteps = (
    ObjectiveStep
    | Sequence[ObjectiveStep | None]
    | Mapping[Any, ObjectiveStep | None]
    | None
)

_STEP_OPTION_KEYS = {
    "objective_steps",
    "objective_step",
    "steps",
    "step",
    "step_sizes",
    "objective_step_sizes",
    "precision_steps",
    "precision_step",
    "precision_step_sizes",
    "objective_precision",
    "objective_precisions",
    "precision",
    "precisions",
    "bin_sizes",
    "bin_size",
    "cell_sizes",
    "cell_size",
    "grid_sizes",
    "grid_size",
}


def _resolve_objective_steps(
    objective_steps: ObjectiveSteps, options: dict[str, Any]
) -> ObjectiveSteps:
    candidates: list[tuple[int, str, Any]] = []
    if objective_steps is not None:
        candidates.append((0, "objective_steps", objective_steps))

    step_tokens = (
        "step",
        "precision",
        "bin",
        "grid",
        "cell",
        "bucket",
        "resolution",
        "quant",
        "tick",
        "width",
        "granular",
        "interval",
        "increment",
        "spacing",
    )
    for key, value in options.items():
        if value is None:
            continue
        normalized_key = key.lower()
        if normalized_key in _STEP_OPTION_KEYS or any(
            token in normalized_key for token in step_tokens
        ):
            priority = 1 if normalized_key.startswith("objective_") else 2
            candidates.append((priority, key, value))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    resolved = candidates[0][2]
    for _, key, value in candidates[1:]:
        if value != resolved:
            raise ValueError(
                "Multiple objective step options were specified with different values: "
                f"{candidates[0][1]}={resolved!r} and {key}={value!r}."
            )
    return resolved


def _normalize_objective_steps(
    objective_steps: ObjectiveSteps, n_objectives: int
) -> tuple[ObjectiveStep | None, ...]:
    if objective_steps is None:
        return ()

    if isinstance(objective_steps, Mapping):
        has_index_keys = any(
            i in objective_steps or str(i) in objective_steps for i in range(n_objectives)
        )
        if has_index_keys:
            steps = tuple(
                objective_steps.get(i, objective_steps.get(str(i)))
                for i in range(n_objectives)
            )
        elif len(objective_steps) == n_objectives:
            # Also accept objective-name mappings when their insertion order follows directions.
            steps = tuple(objective_steps.values())
        else:
            steps = tuple(
                objective_steps.get(i, objective_steps.get(str(i)))
                for i in range(n_objectives)
            )
    elif isinstance(objective_steps, (int, float, Decimal)) and not isinstance(
        objective_steps, bool
    ):
        steps = (objective_steps,) * n_objectives
    else:
        steps = tuple(objective_steps)

    if len(steps) != n_objectives:
        raise ValueError(
            "The number of objective steps must be identical to the number of objectives: "
            f"expected {n_objectives}, but got {len(steps)}."
        )

    for step in steps:
        if step is None:
            continue
        if isinstance(step, bool) or not isinstance(step, (int, float, Decimal)):
            raise TypeError(f"Objective step must be a positive number, but got {step!r}.")
        if not step > 0:
            raise ValueError(f"Objective step must be positive, but got {step!r}.")

    return steps


def _objective_cell(value: float, step: ObjectiveStep | None) -> int | Decimal:
    if step is None:
        return Decimal(str(value))

    # Decimal(str(...)) keeps the decimal intended by callers such as 0.801 / 0.01 instead
    # of relying on the binary floating-point quotient.
    decimal_value = Decimal(str(value))
    decimal_step = step if isinstance(step, Decimal) else Decimal(str(step))
    quotient = decimal_value / decimal_step
    return int(quotient.to_integral_value(rounding=ROUND_FLOOR))


def _get_pareto_front_trials_by_cells(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    objective_steps: ObjectiveSteps,
) -> list[FrozenTrial]:
    steps = _normalize_objective_steps(objective_steps, len(directions))
    if not steps:
        return []

    # Filtering for feasibility has already happened before this helper is called. Selecting
    # representatives by trial number also makes the result independent of the traversal order.
    representatives: dict[tuple[int | Decimal, ...], FrozenTrial] = {}
    for trial in sorted(trials, key=lambda trial: trial.number):
        assert trial.values is not None
        cell = tuple(
            _objective_cell(value, step)
            for value, step in zip(trial.values, steps)
        )
        if cell not in representatives:
            representatives[cell] = trial

    cells = list(representatives)
    normalized_cells = [
        tuple(
            -coordinate if direction is StudyDirection.MAXIMIZE else coordinate
            for coordinate, direction in zip(cell, directions)
        )
        for cell in cells
    ]

    # Rank coordinates per objective. This preserves dominance exactly even when floor
    # indices are too large for float64, while allowing reuse of the vectorized Pareto logic.
    ranked_cells = np.empty((len(cells), len(directions)), dtype=np.float64)
    for objective_index in range(len(directions)):
        coordinates = [cell[objective_index] for cell in normalized_cells]
        unique_coordinates = sorted(set(coordinates))
        ranks = {coordinate: rank for rank, coordinate in enumerate(unique_coordinates)}
        ranked_cells[:, objective_index] = [ranks[coordinate] for coordinate in coordinates]

    on_front = _is_pareto_front(ranked_cells, assume_unique_lexsorted=False)
    return [
        trial
        for trial, is_pareto in zip(representatives.values(), on_front)
        if is_pareto
    ]


def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
    objective_steps: ObjectiveSteps = None,
    **kwargs: Any,
) -> list[FrozenTrial]:
    objective_steps = _resolve_objective_steps(objective_steps, kwargs)
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

    if objective_steps is not None:
        return _get_pareto_front_trials_by_cells(trials, directions, objective_steps)

    loss_values = np.asarray(
        [[_normalize_value(v, d) for v, d in zip(t.values, directions)] for t in trials]
    )
    on_front = _is_pareto_front(loss_values, assume_unique_lexsorted=False)
    return [t for t, is_pareto in zip(trials, on_front) if is_pareto]


def _get_pareto_front_trials(
    study: "optuna.study.Study",
    consider_constraint: bool = False,
    objective_steps: ObjectiveSteps = None,
    **kwargs: Any,
) -> list[FrozenTrial]:
    objective_steps = _resolve_objective_steps(objective_steps, kwargs)
    return _get_pareto_front_trials_by_trials(
        study.trials,
        study.directions,
        consider_constraint,
        objective_steps=objective_steps,
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
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
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
