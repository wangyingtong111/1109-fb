from __future__ import annotations

import pytest

import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState


def _create_study_with_history(
    history: dict[int, tuple[float, float]],
    directions: list[str] | None = None,
) -> optuna.Study:
    study = optuna.create_study(
        directions=directions or ["maximize", "minimize"],
        pruner=optuna.pruners.ParetoPruner(),
    )
    study.add_trial(
        optuna.trial.create_trial(
            values=[0.8, 10.0],
            intermediate_values={step: values for step, values in history.items()},
        )
    )
    return study


@pytest.mark.parametrize(
    "history,reported,expected",
    [
        # The completed trial dominates the candidate at the same step.
        ({3: (0.8, 10.0)}, (0.7, 12.0), True),
        # A trade-off: better accuracy but worse latency must not be pruned.
        ({3: (0.8, 10.0)}, (0.9, 13.0), False),
        # A vector reported at another step cannot substitute for the current step.
        ({2: (0.8, 10.0)}, (0.7, 12.0), False),
    ],
)
def test_pareto_pruner_acceptance_cases(
    history: dict[int, tuple[float, float]],
    reported: tuple[float, float],
    expected: bool,
) -> None:
    study = _create_study_with_history(history)
    trial = study.ask()
    trial.report(list(reported), 3)

    assert trial.should_prune() is expected


def test_pareto_pruner_respects_study_directions() -> None:
    study = optuna.create_study(
        directions=["minimize", "minimize"], pruner=optuna.pruners.ParetoPruner()
    )
    study.add_trial(
        optuna.trial.create_trial(values=[0.8, 10.0], intermediate_values={3: (0.8, 10.0)})
    )

    worse = study.ask()
    worse.report((0.9, 11.0), 3)
    assert worse.should_prune()

    better = study.ask()
    better.report((0.7, 9.0), 3)
    assert not better.should_prune()


def test_pareto_pruner_compares_only_the_latest_step() -> None:
    study = _create_study_with_history({3: (0.8, 10.0)})

    # An earlier dominated step must not prune once the latest step is a trade-off.
    trade_off = study.ask()
    trade_off.report((0.7, 12.0), 2)
    trade_off.report((0.9, 13.0), 3)
    assert not trade_off.should_prune()

    # A dominated latest step prunes even if an earlier step was a trade-off.
    dominated = study.ask()
    dominated.report((0.9, 13.0), 2)
    dominated.report((0.7, 12.0), 3)
    assert dominated.should_prune()


def test_pareto_pruner_does_not_prune_on_equal_vector() -> None:
    study = _create_study_with_history({3: (0.8, 10.0)})
    trial = study.ask()
    trial.report((0.8, 10.0), 3)

    assert not trial.should_prune()


def test_pareto_pruner_disabled_without_completed_trials() -> None:
    study = optuna.create_study(
        directions=["maximize", "minimize"], pruner=optuna.pruners.ParetoPruner()
    )
    trial = study.ask()
    trial.report((0.1, 100.0), 0)

    assert not trial.should_prune()


def test_pareto_pruner_warmup_and_startup() -> None:
    warmup_study = optuna.create_study(
        directions=["maximize", "minimize"],
        pruner=optuna.pruners.ParetoPruner(n_warmup_steps=5),
    )
    warmup_study.add_trial(
        optuna.trial.create_trial(values=[0.8, 10.0], intermediate_values={3: (0.8, 10.0)})
    )
    trial = warmup_study.ask()
    trial.report((0.7, 12.0), 3)
    assert not trial.should_prune()

    startup_study = optuna.create_study(
        directions=["maximize", "minimize"],
        pruner=optuna.pruners.ParetoPruner(n_startup_trials=2),
    )
    startup_study.add_trial(
        optuna.trial.create_trial(values=[0.8, 10.0], intermediate_values={3: (0.8, 10.0)})
    )
    trial = startup_study.ask()
    trial.report((0.7, 12.0), 3)
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "n_startup_trials,n_warmup_steps",
    [(-1, 0), (0, -1)],
)
def test_pareto_pruner_invalid_arguments(n_startup_trials: int, n_warmup_steps: int) -> None:
    with pytest.raises(ValueError):
        optuna.pruners.ParetoPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
        )


def test_pareto_pruner_ignores_nan_and_mismatched_vectors() -> None:
    study = _create_study_with_history({3: (0.8, 10.0)})

    nan_trial = study.ask()
    nan_trial.report((float("nan"), 12.0), 3)
    assert not nan_trial.should_prune()

    short_trial = study.ask()
    with pytest.raises(ValueError):
        short_trial.report((0.7,), 3)


def test_pareto_pruner_never_prunes_single_objective_study() -> None:
    study = optuna.create_study(pruner=optuna.pruners.ParetoPruner())
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
    trial = study.ask()
    trial.report(1.0, 0)

    assert not trial.should_prune()


def test_default_pruner_of_multi_objective_study_is_vector_aware() -> None:
    study = optuna.create_study(directions=["maximize", "minimize"])
    assert isinstance(study.pruner, optuna.pruners.ParetoPruner)

    single_objective_study = optuna.create_study()
    assert isinstance(single_objective_study.pruner, optuna.pruners.MedianPruner)


def test_vector_reporting_is_kept_apart_from_scalar_intermediate_values() -> None:
    from optuna.trial._intermediate import get_intermediate_vectors

    study = optuna.create_study(directions=["maximize", "minimize"])
    trial = study.ask()
    trial.report([0.5, 20.0], 2)
    trial.report((0.8, 10.0), 3)

    assert get_intermediate_vectors(trial._cached_frozen_trial) == {
        2: [0.5, 20.0],
        3: [0.8, 10.0],
    }
    assert trial._cached_frozen_trial.intermediate_values == {}


@pytest.mark.parametrize("storage_mode", ["inmemory", "sqlite"])
def test_pareto_pruner_persists_vectors(storage_mode: str) -> None:
    if storage_mode == "inmemory":
        storage: str | optuna.storages.BaseStorage = optuna.storages.InMemoryStorage()
    else:
        from optuna.testing.tempfile_pool import NamedTemporaryFilePool

        with NamedTemporaryFilePool() as fp:
            study = optuna.create_study(
                storage=f"sqlite:///{fp.name}",
                study_name="pareto",
                directions=["maximize", "minimize"],
                pruner=optuna.pruners.ParetoPruner(),
            )
            completed = study.ask()
            completed.report([0.8, 10.0], 3)
            study.tell(completed, [0.8, 10.0])

            reloaded = optuna.load_study(
                study_name="pareto",
                storage=f"sqlite:///{fp.name}",
                pruner=optuna.pruners.ParetoPruner(),
            )
            trial = reloaded.ask()
            trial.report([0.7, 12.0], 3)
            assert trial.should_prune()
        return

    study = optuna.create_study(
        storage=storage, directions=["maximize", "minimize"], pruner=optuna.pruners.ParetoPruner()
    )
    completed = study.ask()
    completed.report([0.8, 10.0], 3)
    study.tell(completed, [0.8, 10.0])

    trial = study.ask()
    trial.report([0.7, 12.0], 3)
    assert trial.should_prune()


def test_pareto_pruner_end_to_end_keeps_pareto_candidates() -> None:
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0.0, 1.0)
        for step in range(5):
            trial.report([x * (step + 1) / 5.0, (1.0 - x) * (step + 1) / 5.0], step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return x, 1.0 - x

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=6)

    assert any(t.state == TrialState.COMPLETE for t in study.trials)
    # The vector path is optional for the scalar pruners: they never see the vectors.
    assert all(t.intermediate_values == {} for t in study.trials)


def test_pareto_pruner_different_number_of_objectives_is_not_pruned() -> None:
    pruner = optuna.pruners.ParetoPruner()
    study = _create_study_with_history({3: (0.8, 10.0)})
    trial = study.ask()
    trial.report((0.7, 12.0), 3)

    # A single-objective direction list is not comparable with 2-element vectors.
    fake_study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    assert not pruner.prune(fake_study, trial._get_latest_trial())


def test_pareto_pruner_direction_type_is_normalized() -> None:
    from optuna.study._multi_objective import _dominates_values

    assert _dominates_values(
        [0.9, 9.0], [0.8, 10.0], [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    )
    assert not _dominates_values(
        [0.9, 11.0], [0.8, 10.0], [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    )
