from __future__ import annotations

import math
from typing import Literal

import pytest

import optuna
from optuna.pruners import _percentile
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import TrialState


def test_percentile_pruner_percentile() -> None:
    optuna.pruners.PercentilePruner(0.0)
    optuna.pruners.PercentilePruner(25.0)
    optuna.pruners.PercentilePruner(100.0)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(-0.1)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(100.1)


def test_percentile_pruner_n_startup_trials() -> None:
    optuna.pruners.PercentilePruner(25.0, n_startup_trials=0)
    optuna.pruners.PercentilePruner(25.0, n_startup_trials=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, n_startup_trials=-1)


def test_percentile_pruner_n_warmup_steps() -> None:
    optuna.pruners.PercentilePruner(25.0, n_warmup_steps=0)
    optuna.pruners.PercentilePruner(25.0, n_warmup_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, n_warmup_steps=-1)


def test_percentile_pruner_interval_steps() -> None:
    optuna.pruners.PercentilePruner(25.0, interval_steps=1)
    optuna.pruners.PercentilePruner(25.0, interval_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, interval_steps=-1)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, interval_steps=0)


def test_percentile_pruner_with_one_trial() -> None:
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)

    # A pruner is not activated at a first trial.
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "direction_value", [("minimize", [1, 2, 3, 4, 5], 2.1), ("maximize", [1, 2, 3, 4, 5], 3.9)]
)
def test_25_percentile_pruner_intermediate_values(
    direction_value: tuple[Literal["minimize", "maximize"], list[float], float],
) -> None:
    direction, intermediate_values, latest_value = direction_value
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(direction=direction, pruner=pruner)

    for v in intermediate_values:
        trial = study.ask()
        trial.report(v, 1)
        study.tell(trial, v)

    trial = study.ask()
    # A pruner is not activated if a trial has no intermediate values.
    assert not trial.should_prune()

    trial.report(latest_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert trial.should_prune()


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_25_percentile_pruner_intermediate_values_nan() -> None:
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(pruner=pruner)

    trial = study.ask()
    trial.report(float("nan"), 1)
    # A pruner is not activated if the study does not have any previous trials.
    assert not trial.should_prune()
    study.tell(trial, -1)

    trial = study.ask()
    trial.report(float("nan"), 1)
    # A pruner is activated if the best intermediate value of this trial is NaN.
    assert trial.should_prune()
    study.tell(trial, -1)

    trial = study.ask()
    trial.report(1, 1)
    # A pruner is not activated if the 25 percentile intermediate value is NaN.
    assert not trial.should_prune()


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "direction_expected", [(StudyDirection.MINIMIZE, 0.1), (StudyDirection.MAXIMIZE, 0.2)]
)
def test_get_best_intermediate_result_over_steps(
    direction_expected: tuple[StudyDirection, float],
) -> None:
    direction, expected = direction_expected

    if direction == StudyDirection.MINIMIZE:
        study = optuna.study.create_study(direction="minimize")
    else:
        study = optuna.study.create_study(direction="maximize")

    # FrozenTrial.intermediate_values has no elements.
    trial_id_empty = study._storage.create_new_trial(study._study_id)
    trial_empty = study._storage.get_trial(trial_id_empty)

    with pytest.raises(ValueError):
        _percentile._get_best_intermediate_result_over_steps(trial_empty, direction)

    # Input value has no NaNs but float values.
    trial_float = study.ask()
    trial_float.report(0.1, step=0)
    trial_float.report(0.2, step=1)
    frozen_trial_float = study._storage.get_trial(trial_float._trial_id)
    assert expected == _percentile._get_best_intermediate_result_over_steps(
        frozen_trial_float, direction
    )

    # Input value has a float value and a NaN.
    trial_float_nan = study.ask()
    trial_float_nan.report(0.3, step=0)
    trial_float_nan.report(float("nan"), step=1)
    frozen_trial_float_nan = study._storage.get_trial(trial_float_nan._trial_id)
    assert 0.3 == _percentile._get_best_intermediate_result_over_steps(
        frozen_trial_float_nan, direction
    )

    # Input value has a NaN only.
    trial_nan = study.ask()
    trial_nan.report(float("nan"), step=0)
    frozen_trial_nan = study._storage.get_trial(trial_nan._trial_id)
    assert math.isnan(
        _percentile._get_best_intermediate_result_over_steps(frozen_trial_nan, direction)
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_get_percentile_intermediate_result_over_trials() -> None:
    def setup_study(trial_num: int, _intermediate_values: list[list[float]]) -> Study:
        _study = optuna.study.create_study(direction="minimize")
        trial_ids = [_study._storage.create_new_trial(_study._study_id) for _ in range(trial_num)]

        for step, values in enumerate(_intermediate_values):
            # Study does not have any complete trials.
            with pytest.raises(ValueError):
                completed_trials = _study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                _direction = _study.direction
                _percentile._get_percentile_intermediate_result_over_trials(
                    completed_trials, _direction, step, 25, 1
                )

            for i in range(trial_num):
                trial_id = trial_ids[i]
                value = values[i]
                _study._storage.set_trial_intermediate_value(trial_id, step, value)

        # Set trial states complete because this method ignores incomplete trials.
        for trial_id in trial_ids:
            _study._storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE)

        return _study

    # Input value has no NaNs but float values (step=0).
    intermediate_values = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    study = setup_study(9, intermediate_values)
    all_trials = study.get_trials()
    direction = study.direction
    assert 0.3 == _percentile._get_percentile_intermediate_result_over_trials(
        all_trials, direction, 0, 25.0, 1
    )

    # Input value has a float value and NaNs (step=1).
    intermediate_values.append(
        [0.1, 0.2, 0.3, 0.4, 0.5, float("nan"), float("nan"), float("nan"), float("nan")]
    )
    study = setup_study(9, intermediate_values)
    all_trials = study.get_trials()
    direction = study.direction
    assert 0.2 == _percentile._get_percentile_intermediate_result_over_trials(
        all_trials, direction, 1, 25.0, 1
    )

    # Input value has NaNs only (step=2).
    intermediate_values.append(
        [
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ]
    )
    study = setup_study(9, intermediate_values)
    all_trials = study.get_trials()
    direction = study.direction
    assert math.isnan(
        _percentile._get_percentile_intermediate_result_over_trials(
            all_trials, direction, 2, 75, 1
        )
    )

    # n_min_trials = 2.
    assert math.isnan(
        _percentile._get_percentile_intermediate_result_over_trials(
            all_trials, direction, 2, 75, 2
        )
    )


def _create_phase_isolated_study(
    direction: Literal["minimize", "maximize"] = "minimize",
    n_min_trials: int = 1,
) -> Study:
    pruner = optuna.pruners.PercentilePruner(
        25.0,
        n_startup_trials=0,
        n_warmup_steps=0,
        interval_steps=1,
        n_min_trials=n_min_trials,
        phase_isolation=True,
    )
    return optuna.study.create_study(direction=direction, pruner=pruner)


def test_percentile_pruner_phase_isolation_protects_only_within_phase() -> None:
    # Main phase (phase=0) of the reference trial has a different number of steps (3)
    # than that of the current trial (1), and its fine-tune phase (phase=1) reports 5
    # at phase-local step 0.
    study = _create_phase_isolated_study()
    reference = study.ask()
    reference.report(2, 0, phase=0)
    reference.report(1, 1, phase=0)
    reference.report(1, 2, phase=0)
    reference.report(5, 3, phase=1)
    study.tell(reference, 5)

    current = study.ask()
    current.report(1, 0, phase=0)  # Best result of the main phase.
    current.report(8, 1, phase=1)  # Fine-tune phase-local step 0.

    # The main-phase result (1) must not protect the trial in the fine-tune phase.
    assert current.should_prune()


def test_percentile_pruner_without_phase_isolation_keeps_legacy_rule() -> None:
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0, n_min_trials=1)
    study = optuna.study.create_study(direction="minimize", pruner=pruner)

    reference = study.ask()
    reference.report(5, 0)
    study.tell(reference, 5)

    current = study.ask()
    current.report(1, 0)  # Good historical result protects the trial.
    current.report(8, 1)
    assert not current.should_prune()


def test_percentile_pruner_phase_isolation_ignored_without_phases() -> None:
    # Even with phase isolation enabled, unphased reports follow the legacy rule.
    study = _create_phase_isolated_study()
    reference = study.ask()
    reference.report(5, 0)
    study.tell(reference, 5)

    current = study.ask()
    current.report(1, 0)
    current.report(8, 1)
    assert not current.should_prune()


def test_percentile_pruner_phase_isolation_compares_local_steps() -> None:
    # Trials may have main phases of different lengths; local steps inside the
    # fine-tune phase are still compared.
    study = _create_phase_isolated_study()

    first = study.ask()
    first.report(3, 0, phase=0)
    first.report(5, 1, phase=1)  # phase-1 local step 0
    study.tell(first, 5)

    second = study.ask()
    second.report(3, 0, phase=0)
    second.report(2, 1, phase=0)
    second.report(4, 2, phase=1)  # phase-1 local step 0 (global step 2)
    study.tell(second, 4)

    current = study.ask()
    current.report(9, 0, phase=0)
    current.report(8, 1, phase=1)  # phase-1 local step 0
    assert current.should_prune()


def test_percentile_pruner_phase_isolation_respects_n_min_trials() -> None:
    study = _create_phase_isolated_study(n_min_trials=2)
    current = study.ask()
    current.report(1, 0, phase=0)
    current.report(8, 1, phase=1)
    # No completed trial reports in phase 1, so the minimum sample count is unmet.
    assert not current.should_prune()


def test_percentile_pruner_phase_isolation_within_phase_protection() -> None:
    study = _create_phase_isolated_study()
    reference = study.ask()
    reference.report(9, 0, phase=0)
    study.tell(reference, 9)

    current = study.ask()
    current.report(1, 0, phase=0)  # Best within the same phase still protects.
    assert not current.should_prune()


def test_get_local_step_and_phase_mapping() -> None:
    study = _create_phase_isolated_study()
    trial = study.ask()
    trial.report(3, 0, phase=0)
    trial.report(2, 1, phase=0)
    trial.report(5, 2, phase=1)
    trial.report(4, 3, phase=1)
    frozen = study._storage.get_trial(trial._trial_id)

    assert _percentile._get_phase_mapping(frozen) == {0: 0, 1: 0, 2: 1, 3: 1}
    assert _percentile._get_local_step(frozen, 0) == 0
    assert _percentile._get_local_step(frozen, 1) == 1
    assert _percentile._get_local_step(frozen, 2) == 0
    assert _percentile._get_local_step(frozen, 3) == 1

