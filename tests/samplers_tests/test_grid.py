from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import ValuesView
import itertools

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.samplers._grid import GridValueType
from optuna.storages import RetryHeartbeatStaleTrialCallback
from optuna.testing.objectives import fail_objective
from optuna.testing.objectives import pruned_objective
from optuna.testing.storages import StorageSupplier
from optuna.trial import Trial


def test_study_optimize_with_single_search_space() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 100)
        b = trial.suggest_float("b", -0.1, 0.1)
        c = trial.suggest_categorical("c", ("x", "y", None, 1, 2.0))
        d = trial.suggest_float("d", -5, 5, step=1)
        e = trial.suggest_float("e", 0.0001, 1, log=True)

        if c == "x":
            return a * d
        else:
            return b * e

    # Test that all combinations of the grid is sampled.
    search_space = {
        "b": np.arange(-0.1, 0.1, 0.05),
        "c": ("x", "y", None, 1, 2.0),
        "d": [-5.0, 5.0],
        "e": [0.1],
        "a": list(range(0, 100, 20)),
    }
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))  # type: ignore
    study.optimize(objective)

    def sorted_values(
        d: Mapping[str, Sequence[GridValueType]],
    ) -> ValuesView[Sequence[GridValueType]]:
        return dict(sorted(d.items())).values()

    all_grids = itertools.product(*sorted_values(search_space))  # type: ignore
    all_suggested_values = [tuple([p for p in sorted_values(t.params)]) for t in study.trials]
    assert set(all_grids) == set(all_suggested_values)

    # Test a non-existing parameter name in the grid.
    search_space = {"a": list(range(0, 100, 20))}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))  # type: ignore
    with pytest.raises(ValueError):
        study.optimize(objective)

    # Test a value with out of range.
    search_space = {
        "a": [110],  # 110 is out of range specified by the suggest method.
        "b": [0],
        "c": ["x"],
        "d": [0],
        "e": [0.1],
    }
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))  # type: ignore
    with pytest.warns(UserWarning):
        study.optimize(objective)


def test_study_optimize_with_exceeding_number_of_trials() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_int("a", 0, 100)

    # When `n_trials` is `None`, the optimization stops just after all grids are evaluated.
    search_space: dict[str, list[GridValueType]] = {"a": [0, 50]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=None)
    assert len(study.trials) == 2

    # If the optimization is triggered after all grids are evaluated, an additional trial runs.
    study.optimize(objective, n_trials=None)
    assert len(study.trials) == 3


def test_study_optimize_with_pruning() -> None:
    # Pruned trials should count towards grid consumption.
    search_space: dict[str, list[GridValueType]] = {"a": [0, 50]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(pruned_objective, n_trials=None)
    assert len(study.trials) == 2


def test_study_optimize_with_fail() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_int("a", 0, 100)

    # Failed trials should count towards grid consumption.
    search_space: dict[str, list[GridValueType]] = {"a": [0, 50]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(fail_objective, n_trials=1, catch=ValueError)
    study.optimize(objective, n_trials=None)
    assert len(study.trials) == 2


def test_study_optimize_with_numpy_related_search_space() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 0, 10)
        b = trial.suggest_float("b", -0.1, 0.1)

        return a + b

    # Test that all combinations of the grid is sampled.
    search_space = {
        "a": np.linspace(0, 10, 11),
        "b": np.arange(-0.1, 0.1, 0.05),
    }
    with StorageSupplier("sqlite") as storage:
        study = optuna.create_study(
            sampler=samplers.GridSampler(search_space),  # type: ignore
            storage=storage,
        )
        study.optimize(objective, n_trials=None)


def test_study_optimize_with_multiple_search_spaces() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 100)
        b = trial.suggest_float("b", -100, 100)

        return a * b

    # Run 3 trials with a search space.
    search_space_0 = {"a": [0, 50], "b": [-50, 0, 50]}
    sampler_0 = samplers.GridSampler(search_space_0)
    study = optuna.create_study(sampler=sampler_0)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    for t in study.trials:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])

    # Run 2 trials with another space.
    search_space_1 = {"a": [0, 25], "b": [-50]}
    sampler_1 = samplers.GridSampler(search_space_1)
    study.sampler = sampler_1
    study.optimize(objective, n_trials=2)

    assert not sampler_0._same_search_space(sampler_1._search_space)
    assert len(study.trials) == 5
    for t in study.trials[:3]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[3:5]:
        assert sampler_1._same_search_space(t.system_attrs["search_space"])

    # Run 3 trials with the first search space again.
    study.sampler = sampler_0
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 8
    for t in study.trials[:3]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[3:5]:
        assert sampler_1._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[5:]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])


def test_cast_value() -> None:
    samplers.GridSampler._check_value("x", None)
    samplers.GridSampler._check_value("x", True)
    samplers.GridSampler._check_value("x", False)
    samplers.GridSampler._check_value("x", -1)
    samplers.GridSampler._check_value("x", -1.5)
    samplers.GridSampler._check_value("x", float("nan"))
    samplers.GridSampler._check_value("x", "foo")
    samplers.GridSampler._check_value("x", "")

    with pytest.warns(UserWarning):
        samplers.GridSampler._check_value("x", [1])


def test_has_same_search_space() -> None:
    search_space: dict[str, list[int | str]] = {"x": [3, 2, 1], "y": ["a", "b", "c"]}
    sampler = samplers.GridSampler(search_space)
    assert sampler._same_search_space(search_space)
    assert sampler._same_search_space({"x": [3, 2, 1], "y": ["a", "b", "c"]})

    assert not sampler._same_search_space({"y": ["c", "a", "b"], "x": [1, 2, 3]})
    assert not sampler._same_search_space({"x": [3, 2, 1, 0], "y": ["a", "b", "c"]})
    assert not sampler._same_search_space({"x": [3, 2], "y": ["a", "b", "c"]})


def test_retried_trial() -> None:
    sampler = samplers.GridSampler({"a": [0, 50]})
    study = optuna.create_study(sampler=sampler)
    trial = study.ask()
    trial.suggest_int("a", 0, 100)

    callback = RetryHeartbeatStaleTrialCallback()
    callback(study, study.trials[0])

    study.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    assert len(study.trials) == 3
    assert study.trials[0].params["a"] == study.trials[1].params["a"]
    assert study.trials[0].system_attrs["grid_id"] == study.trials[1].system_attrs["grid_id"]


@pytest.mark.parametrize(
    ("grid_values", "n_initial_trials"), [([0], 0), ([0, 50], 0), ([0, 50], 1)]
)
def test_enqueued_trial(grid_values: list[int], n_initial_trials: int) -> None:
    sampler = samplers.GridSampler({"a": grid_values})
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: trial.suggest_int("a", 0, 100), n_trials=n_initial_trials)
    study.enqueue_trial({"a": 100})

    study.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    trials = study.trials
    assert len(trials) == len(grid_values) + 1
    assert trials[n_initial_trials].params["a"] == 100
    grid_trials = trials[:n_initial_trials] + trials[n_initial_trials + 1 :]
    assert sorted(trial.params["a"] for trial in grid_trials) == grid_values


def test_same_seed_trials() -> None:
    grid_values = [0, 20, 40, 60, 80, 100]
    seed = 0

    sampler1 = samplers.GridSampler({"a": grid_values}, seed)
    study1 = optuna.create_study(sampler=sampler1)
    study1.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    sampler2 = samplers.GridSampler({"a": grid_values}, seed)
    study2 = optuna.create_study(sampler=sampler2)
    study2.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    for i in range(len(grid_values)):
        assert study1.trials[i].params["a"] == study2.trials[i].params["a"]


def test_enqueued_insufficient_trial() -> None:
    sampler = samplers.GridSampler({"a": [0, 50]})
    study = optuna.create_study(sampler=sampler)
    study.enqueue_trial({})

    with pytest.raises(ValueError):
        study.optimize(lambda trial: trial.suggest_int("a", 0, 100))


def test_nan() -> None:
    sampler = optuna.samplers.GridSampler({"x": [0, float("nan")]})
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda trial: 1 if np.isnan(trial.suggest_categorical("x", [0, float("nan")])) else 0
    )
    assert len(study.get_trials()) == 2


def test_is_exhausted() -> None:
    search_space = {"a": [0, 50]}
    sampler = samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler)
    assert not sampler.is_exhausted(study)
    study.optimize(lambda trial: trial.suggest_categorical("a", [0, 50]))
    assert sampler.is_exhausted(study)

def test_step_grid_sequence_interface() -> None:
    # Duplicated step numbers are deduplicated, keeping the first occurrences' order.
    step_grid = samplers.StepGrid([3, 1, 3, 2])
    assert list(step_grid) == [3, 1, 2]
    assert len(step_grid) == 3
    assert step_grid[0] == 3
    assert 1 in step_grid
    assert step_grid == samplers.StepGrid([3, 1, 2])
    assert step_grid != samplers.StepGrid([1, 2, 3])
    assert repr(step_grid) == "StepGrid(steps=[3, 1, 2])"

    # NumPy integers are accepted as step numbers as well.
    assert list(samplers.StepGrid(np.arange(0, 3))) == [0, 1, 2]


def test_step_grid_deduplicates_equivalent_step_numbers() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0, step=0.1)

    # `0.1 + 0.2` computed from increments and `0.3` read from a configuration file are
    # different floats, but on the ladder defined by `step=0.1` both are unambiguously
    # represented by step number 3.
    assert 0.1 + 0.2 != 0.3
    step_from_increment = round((0.1 + 0.2) / 0.1)
    step_from_config = round(0.3 / 0.1)

    search_space = {"x": samplers.StepGrid([step_from_increment, step_from_config, 4])}
    sampler = samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective)

    # Only the two distinct points, step numbers 3 and 4, are scheduled.
    assert len(study.trials) == 2
    suggested_values = sorted(t.params["x"] for t in study.trials)
    assert suggested_values == [0.3, 0.4]
    # The canonical value is suggested, not a float artifact such as `0.1 + 0.2`.
    assert 0.1 + 0.2 not in suggested_values


def test_step_grid_with_int_distribution() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_int("x", 0, 10, step=2)

    search_space = {"x": samplers.StepGrid([0, 3, 5, 5])}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(objective)

    assert sorted(t.params["x"] for t in study.trials) == [0, 6, 10]


def test_step_grid_mixed_with_plain_values() -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0, step=0.1)
        y = trial.suggest_categorical("y", ["a", "b"])
        return x if y == "a" else -x

    search_space = {"x": samplers.StepGrid([1, 2]), "y": ["a", "b"]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(objective)

    assert len(study.trials) == 4
    assert sorted(t.params["x"] for t in study.trials) == [0.1, 0.1, 0.2, 0.2]


def test_step_grid_resume_optimization() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0, step=0.1)

    search_space = {"x": samplers.StepGrid([3, 3, 4])}
    with StorageSupplier("sqlite") as storage:
        sampler = samplers.GridSampler(search_space, seed=0)
        study = optuna.create_study(sampler=sampler, storage=storage, study_name="step-grid")
        study.optimize(objective, n_trials=1)

        assert len(study.trials) == 1
        assert not sampler.is_exhausted(study)
        # The unfinished step is still visible as an unvisited grid.
        assert len(sampler._get_unvisited_grid_ids(study)) == 1

        # Resume the study on the same storage with a new sampler instance.
        new_sampler = samplers.GridSampler(search_space, seed=0)
        resumed_study = optuna.create_study(
            sampler=new_sampler, storage=storage, study_name="step-grid", load_if_exists=True
        )
        resumed_study.optimize(objective)

        # Each step's completion is still visible after the resume.
        assert len(resumed_study.trials) == 2
        assert new_sampler.is_exhausted(resumed_study)
        assert new_sampler._get_unvisited_grid_ids(resumed_study) == []
        assert sorted(t.params["x"] for t in resumed_study.trials) == [0.3, 0.4]
        for t in resumed_study.trials:
            assert new_sampler._same_search_space(t.system_attrs["search_space"])


@pytest.mark.parametrize("steps", [[0.3], [2.5], [0.1 + 0.2], ["3"], [True], [None]])
def test_step_grid_rejects_non_integer_step_numbers(steps: list) -> None:
    # Values that do not fall exactly on the distribution's steps are rejected by an
    # explicit rule: step numbers must be integers.
    with pytest.raises(ValueError):
        samplers.StepGrid(steps)


@pytest.mark.parametrize("steps", [[-1], [11]])
def test_step_grid_rejects_out_of_range_step_numbers(steps: list) -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0, step=0.1)

    study = optuna.create_study(sampler=samplers.GridSampler({"x": samplers.StepGrid(steps)}))
    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1)


def test_step_grid_requires_discretized_distribution() -> None:
    # A continuous float distribution has no ladder of steps.
    study = optuna.create_study(sampler=samplers.GridSampler({"x": samplers.StepGrid([0, 1])}))
    with pytest.raises(ValueError):
        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    # Neither does a categorical distribution.
    study = optuna.create_study(sampler=samplers.GridSampler({"x": samplers.StepGrid([0, 1])}))
    with pytest.raises(ValueError):
        study.optimize(lambda trial: 0.0 if trial.suggest_categorical("x", ["a", "b"]) else 1.0)

    # Log-scaled distributions are not ladders with equal steps.
    study = optuna.create_study(sampler=samplers.GridSampler({"x": samplers.StepGrid([0, 1])}))
    with pytest.raises(ValueError):
        study.optimize(lambda trial: trial.suggest_int("x", 1, 100, log=True), n_trials=1)


def test_plain_float_and_non_uniform_grids_are_unchanged() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0, step=0.1)

    # Plain float grids keep the existing behavior: values are used as-is and are neither
    # snapped to steps nor deduplicated, so `0.1 + 0.2` and `0.3` remain two distinct
    # grid points.
    search_space: dict[str, list[GridValueType]] = {"x": [0.1 + 0.2, 0.3]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(objective)
    assert len(study.trials) == 2

    # Non-uniform grids keep working as before; off-ladder values are only warned about.
    search_space = {"x": [0.0, 0.25, 0.5, 1.0]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    with pytest.warns(UserWarning):
        study.optimize(objective)
    assert sorted(t.params["x"] for t in study.trials) == [0.0, 0.25, 0.5, 1.0]
