from __future__ import annotations

from collections.abc import Sequence
import decimal
import itertools
from numbers import Integral
from numbers import Real
from typing import Any
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

from optuna._warnings import optuna_warn
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Mapping

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


GridValueType = Union[str, float, int, bool, None]


_logger = get_logger(__name__)


class StepGrid(Sequence[int]):
    """Grid points of a parameter declared by step numbers of a discretized distribution.

    This object is used as a value of the ``search_space`` argument of
    :class:`~optuna.samplers.GridSampler`. Instead of listing parameter values directly,
    :class:`~optuna.samplers.StepGrid` lists integer step numbers on the ladder defined by
    the ``step`` argument of a suggest method: step number ``k`` corresponds to the value
    ``low + k * step``.

    Declaring a grid by step numbers avoids floating-point representation issues. For
    example, ``0.1 + 0.2`` computed from increments and ``0.3`` read from a configuration
    file are different floats, but both are unambiguously represented by step number ``3``
    of ``trial.suggest_float("x", 0.0, 1.0, step=0.1)``. Duplicated step numbers are
    deduplicated so that the corresponding point is scheduled only once.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0.0, 1.0, step=0.1)
                return x**2


            # Step numbers 3 and 4 correspond to the values 0.3 and 0.4.
            search_space = {"x": optuna.samplers.StepGrid([3, 3, 4])}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective)

            # The duplicated step number 3 is evaluated only once.
            assert len(study.trials) == 2

    .. note::
        Each step number must be an integer. Values that do not fall exactly on the
        ladder's steps, e.g. ``0.3`` itself or ``2.5``, are rejected with a
        :exc:`ValueError`. Please pass the corresponding integer step number instead.

    .. note::
        The parameter must be suggested with a discretized distribution, i.e.
        :func:`~optuna.trial.Trial.suggest_float` with the ``step`` argument or
        :func:`~optuna.trial.Trial.suggest_int` without ``log=True``. Otherwise, a
        :exc:`ValueError` is raised when the parameter is suggested. A step number out of
        the ladder's range is rejected with a :exc:`ValueError` as well.

    Args:
        steps:
            A sequence of integer step numbers. ``0`` corresponds to the lower endpoint of
            the distribution. Duplicated step numbers are deduplicated, keeping the order
            of their first occurrences.
    """

    def __init__(self, steps: Sequence[int]) -> None:
        deduplicated_steps: list[int] = []
        seen_steps: set[int] = set()
        for step in steps:
            if isinstance(step, bool) or not isinstance(step, Integral):
                raise ValueError(
                    f"Each value in `StepGrid` must be an integer step number, but got "
                    f"{step!r} of type {type(step)}. Values that do not fall exactly on "
                    "the distribution's steps are not allowed; please pass the "
                    "corresponding integer step number instead, e.g. `3` for the value "
                    "`0.3` of `suggest_float(..., low=0.0, step=0.1)`."
                )
            step_number = int(step)
            if step_number not in seen_steps:
                seen_steps.add(step_number)
                deduplicated_steps.append(step_number)
        self._steps = tuple(deduplicated_steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        return self._steps[index]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StepGrid):
            return NotImplemented
        return self._steps == other._steps

    def __repr__(self) -> str:
        return f"StepGrid(steps={list(self._steps)!r})"


class GridSampler(BaseSampler):
    """Sampler that performs exhaustive search over the define-and-run user-specified grids.

    With :class:`~optuna.samplers.GridSampler`, the trials suggest all combinations of parameters
    in the given search space during the study.

    .. note::
        :class:`~optuna.samplers.BruteForceSampler` also provides the exhaustive search feature.
        Unlike :class:`~optuna.samplers.GridSampler`, :class:`~optuna.samplers.BruteForceSampler`
        does not ask users to provide search space and it supports hierarchical, aka dynamic search
        space, enabling users to seamlessly enjoy the exhaustive search.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_int("y", -100, 100)
                return x**2 + y**2


            search_space = {"x": [-50, 0, 50], "y": [-99, 0, 99]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective)

    Example:

        A grid can also be declared by step numbers of a discretized distribution with
        :class:`~optuna.samplers.StepGrid`. This avoids floating-point representation
        issues such as ``0.1 + 0.2 != 0.3``; both are unambiguously represented by step
        number ``3`` and are scheduled as the same grid point.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0.0, 1.0, step=0.1)
                return x**2


            # Step numbers 0, 3, and 10 correspond to the values 0.0, 0.3, and 1.0.
            search_space = {"x": optuna.samplers.StepGrid([0, 3, 10])}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective)

    Note:

        This sampler with :ref:`ask_and_tell` raises :exc:`RuntimeError` just after evaluating
        the final grid. This is because :class:`~optuna.samplers.GridSampler` automatically
        stops the optimization if all combinations in the passed ``search_space`` have already
        been evaluated, internally invoking the :func:`~optuna.study.Study.stop` method.
        As a workaround, we need to handle the error manually as in
        https://github.com/optuna/optuna/issues/4121#issuecomment-1305289910.

    Note:

        :class:`~optuna.samplers.GridSampler` does not take care of a parameter's quantization
        specified by discrete suggest methods but just samples one of values specified in the
        search space. E.g., in the following code snippet, either of ``-0.5`` or ``0.5`` is
        sampled as ``x`` instead of an integer point.

        .. testcode::

            import optuna


            def objective(trial):
                # The following suggest method specifies integer points between -5 and 5.
                x = trial.suggest_float("x", -5, 5, step=1)
                return x**2


            # Non-int points are specified in the grid.
            search_space = {"x": [-0.5, 0.5]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective, n_trials=2)

    Note:
        A parameter configuration in the grid is not considered finished until its trial is
        finished. Therefore, during distributed optimization where trials run concurrently,
        different workers will occasionally suggest the same parameter configuration.
        The total number of actual trials may therefore exceed the size of the grid.

    Note:
        All parameters must be specified when using :class:`~optuna.samplers.GridSampler` with
        :meth:`~optuna.study.Study.enqueue_trial`.

    Args:
        search_space:
            A dictionary whose key and value are a parameter name and the corresponding
            candidates of values, respectively. A :class:`~optuna.samplers.StepGrid` can
            be used as a value to declare the candidates by step numbers of a discretized
            distribution instead of listing the values directly.
        seed:
            A seed to fix the order of trials as the grid is randomly shuffled. This shuffle is
            beneficial when the number of grids is larger than ``n_trials`` in
            :meth:`~optuna.Study.optimize` to suppress suggesting similar grids. Please note
            that fixing ``seed`` for each process is strongly recommended in distributed
            optimization to avoid duplicated suggestions.
    """

    def __init__(
        self,
        search_space: Mapping[str, Sequence[GridValueType] | StepGrid],
        seed: int | None = None,
    ) -> None:
        self._step_grid_params: set[str] = set()
        self._search_space = {}
        for param_name, param_values in sorted(search_space.items()):
            if isinstance(param_values, StepGrid):
                # The grid of this parameter is declared by step numbers of the
                # distribution. The deduplicated integer step numbers are stored as the
                # canonical grid values, which are free of floating-point representation
                # issues and safely round-trip through persistent storages.
                self._step_grid_params.add(param_name)
                self._search_space[param_name] = list(param_values)
            else:
                for value in param_values:
                    self._check_value(param_name, value)
                self._search_space[param_name] = list(param_values)

        self._all_grids = list(itertools.product(*self._search_space.values()))
        self._param_names = sorted(search_space.keys())
        self._n_min_trials = len(self._all_grids)
        self._rng = LazyRandomState(seed or 0)
        self._rng.rng.shuffle(self._all_grids)  # type: ignore[arg-type]

    def reseed_rng(self) -> None:
        self._rng.rng.seed()

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Instead of returning param values, GridSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        # When the trial is created by RetryHeartbeatStaleTrialCallback or enqueue_trial, we
        # should not assign a new grid_id.
        if "grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return

        if 0 <= trial.number and trial.number < self._n_min_trials:
            study._storage.set_trial_system_attr(
                trial._trial_id, "search_space", self._search_space
            )
            study._storage.set_trial_system_attr(trial._trial_id, "grid_id", trial.number)
            return

        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            # This case may occur with distributed optimization or trial queue. If there is no
            # target grid, `GridSampler` evaluates a visited, duplicated point with the current
            # trial. After that, the optimization stops.

            _logger.warning(
                "`GridSampler` is re-evaluating a configuration because the grid has been "
                "exhausted. This may happen due to a timing issue during distributed optimization "
                "or when re-running optimizations on already finished studies."
            )

            # One of all grids is randomly picked up in this case.
            target_grids = list(range(len(self._all_grids)))

        # In distributed optimization, multiple workers may simultaneously pick up the same grid.
        # To make the conflict less frequent, the grid is chosen randomly.
        grid_id = int(self._rng.rng.choice(target_grids))

        study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if "grid_id" not in trial.system_attrs:
            message = "All parameters must be specified when using GridSampler with enqueue_trial."
            raise ValueError(message)

        if param_name not in self._search_space:
            message = f"The parameter name, {param_name}, is not found in the given grid."
            raise ValueError(message)

        grid_id = trial.system_attrs["grid_id"]
        param_value = self._all_grids[grid_id][self._param_names.index(param_name)]
        if param_name in self._step_grid_params:
            param_value = self._convert_step_number_to_value(
                param_name, param_value, param_distribution
            )
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            optuna_warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            study.stop()
        elif len(target_grids) == 1:
            # Enqueued trials have no grid_id and must not stop the remaining grid.
            grid_id = study._storage.get_trial_system_attrs(trial._trial_id).get("grid_id")
            if grid_id == target_grids[0]:
                study.stop()

    @staticmethod
    def _convert_step_number_to_value(
        param_name: str, step_number: int, param_distribution: BaseDistribution
    ) -> int | float:
        if isinstance(param_distribution, IntDistribution) and not param_distribution.log:
            max_step_number = (
                param_distribution.high - param_distribution.low
            ) // param_distribution.step
            value: int | float = param_distribution.low + step_number * param_distribution.step
        elif (
            isinstance(param_distribution, FloatDistribution)
            and param_distribution.step is not None
        ):
            # Use decimal arithmetic to obtain the canonical value of each step, e.g.
            # exactly `0.3` for step number 3 of `suggest_float("x", 0.0, 1.0, step=0.1)`.
            d_low = decimal.Decimal(str(param_distribution.low))
            d_high = decimal.Decimal(str(param_distribution.high))
            d_step = decimal.Decimal(str(param_distribution.step))
            max_step_number = int((d_high - d_low) // d_step)
            value = float(d_low + step_number * d_step)
        else:
            raise ValueError(
                f"`StepGrid` is used for the parameter `{param_name}`, but its distribution "
                f"`{param_distribution}` does not define a ladder of steps. `StepGrid` "
                "requires a distribution discretized by `step`, i.e. `suggest_float` with "
                "the `step` argument or `suggest_int` without `log=True`."
            )

        if not 0 <= step_number <= max_step_number:
            raise ValueError(
                f"The step number {step_number} for the parameter `{param_name}` is out of "
                f"the valid range [0, {max_step_number}] of the distribution "
                f"`{param_distribution}`. Values that do not fall on the distribution's "
                "steps are not allowed."
            )

        return value

    @staticmethod
    def _check_value(param_name: str, param_value: Any) -> None:
        if param_value is None or isinstance(param_value, (str, int, float, bool)):
            return

        message = (
            f"{param_name} contains a value with the type of {type(param_value)}, "
            "which is not supported by `GridSampler`. Please make sure a value is `str`, "
            "`int`, `float`, `bool` or `None` for persistent storage."
        )
        optuna_warn(message)

    def _get_unvisited_grid_ids(self, study: Study) -> list[int]:
        # List up unvisited grids based on already finished ones.
        visited_grids = []
        running_grids = []

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for t in trials:
            if "grid_id" in t.system_attrs and self._same_search_space(
                t.system_attrs["search_space"]
            ):
                if t.state.is_finished():
                    visited_grids.append(t.system_attrs["grid_id"])
                elif t.state == TrialState.RUNNING:
                    running_grids.append(t.system_attrs["grid_id"])

        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids) - set(running_grids)

        # If evaluations for all grids have been started, return grids that have not yet finished
        # because all grids should be evaluated before stopping the optimization.
        if len(unvisited_grids) == 0:
            unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)

    @staticmethod
    def _grid_value_equal(value1: GridValueType, value2: GridValueType) -> bool:
        value1_is_nan = isinstance(value1, Real) and np.isnan(float(value1))
        value2_is_nan = isinstance(value2, Real) and np.isnan(float(value2))
        return (value1 == value2) or (value1_is_nan and value2_is_nan)

    def _same_search_space(self, search_space: Mapping[str, Sequence[GridValueType]]) -> bool:
        if search_space == self._search_space:
            return True
        if set(search_space.keys()) != set(self._search_space.keys()):
            return False

        for param_name in search_space.keys():
            if len(search_space[param_name]) != len(self._search_space[param_name]):
                return False

            for i, param_value in enumerate(search_space[param_name]):
                if not self._grid_value_equal(param_value, self._search_space[param_name][i]):
                    return False

        return True

    def is_exhausted(self, study: Study) -> bool:
        """
        Return True if all the possible params are evaluated, otherwise return False.
        """
        return len(self._get_unvisited_grid_ids(study)) == 0
