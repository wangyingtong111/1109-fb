from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optuna.pruners._base import BasePruner
from optuna.study._multi_objective import _dominates_values
from optuna.trial._intermediate import get_intermediate_vectors
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    import optuna
    from optuna.trial import FrozenTrial


class ParetoPruner(BasePruner):
    """Pruner based on strict Pareto dominance of the intermediate objective vectors.

    In a multi-objective study, the objective values are only known at the end of a trial, so the
    trade-off between the objectives cannot be taken into account by the scalar pruners before the
    trial finishes. This pruner enables pruning during a trial by comparing the *complete* vector
    reported at the current step with the vectors reported by the completed trials at the very same
    step.

    A trial is pruned as soon as one of the completed trials strictly dominates its vector. A
    vector ``v`` strictly dominates another vector ``w`` when ``v`` is no worse than ``w`` on every
    objective and strictly better on at least one objective, where "better" follows the
    :attr:`~optuna.study.Study.directions` of the study. A candidate that only trades off one
    objective against another is never pruned, so useful Pareto points remain in the study.

    Vectors are provided through the vector reporting path, i.e.
    :func:`optuna.trial.Trial.report` with a sequence of values in a multi-objective study. Only
    the vectors reported at the exact same step are compared: a vector reported at another step is
    never used as a substitute for the current step, because intermediate values at different steps
    are not comparable.

    The scalar reporting path (a single float) is left untouched: it is still used by
    single-objective optimization and by the existing scalar pruners.

    .. note::
        This pruner is intended for multi-objective optimization. In a single-objective study it
        never prunes, so a vector-aware pruner can be configured unconditionally.

    Example:

        Prune a trial as soon as its intermediate (accuracy, latency) pair is dominated at the same
        step by a completed trial.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0.0, 1.0)
                for step in range(10):
                    accuracy = x * (step + 1) / 10.0
                    latency = (1.0 - x) * (step + 1) / 10.0
                    trial.report([accuracy, latency], step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return x, 1.0 - x


            study = optuna.create_study(
                directions=["maximize", "minimize"],
                pruner=optuna.pruners.ParetoPruner(),
            )
            study.optimize(objective, n_trials=20)

    Args:
        n_startup_trials:
            Pruning is disabled until the given number of trials have finished in the study. The
            candidate trial itself is not counted. Defaults to ``0``.
        n_warmup_steps:
            Pruning is disabled while the current step is smaller than ``n_warmup_steps``; the
            earliest a trial can be pruned is at ``step == n_warmup_steps``. Defaults to ``0``.
    """

    def __init__(self, n_startup_trials: int = 0, n_warmup_steps: int = 0) -> None:
        if n_startup_trials < 0:
            raise ValueError(
                f"Number of startup trials cannot be negative, but got {n_startup_trials=}."
            )
        if n_warmup_steps < 0:
            raise ValueError(
                f"Number of warmup steps cannot be negative, but got {n_warmup_steps=}."
            )

        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps

    def prune(self, study: "optuna.study.Study", trial: "FrozenTrial") -> bool:
        directions = study.directions
        if len(directions) <= 1:
            # Single-objective studies never report vectors and rely on the scalar pruners.
            return False

        vectors = get_intermediate_vectors(trial)
        if not vectors:
            return False

        # Compare the vectors reported at the latest step of the candidate only. Vectors reported
        # at another step cannot substitute for the current step.
        step = max(vectors)
        if step < self._n_warmup_steps:
            return False

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if len(completed_trials) < self._n_startup_trials:
            return False

        candidate = vectors[step]
        if len(candidate) != len(directions) or any(math.isnan(v) for v in candidate):
            # A vector of the wrong length or containing NaN cannot be compared by dominance.
            return False

        for completed_trial in completed_trials:
            reference_vectors = get_intermediate_vectors(completed_trial)
            # Only the vector reported at the exact same step is a valid reference.
            if step not in reference_vectors:
                continue
            reference = reference_vectors[step]
            if len(reference) != len(candidate):
                continue
            if _dominates_values(reference, candidate, directions):
                return True

        return False
