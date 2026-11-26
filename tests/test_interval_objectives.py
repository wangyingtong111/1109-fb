from __future__ import annotations

import optuna
from optuna.study._multi_objective import _dominates
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials


def _study(with_c: bool = False, reverse: bool = False) -> optuna.Study:
    rows = [
        ("A", [0.77, 0.91], 8),
        ("B", [0.79, 0.82], 9),
    ]
    if with_c:
        rows.append(("C", [0.68, 0.77], 12))
    rows.extend(
        [
            ("D", [0.82, 0.82], 10),
            ("E", [0.82, 0.82], 10),
        ]
    )
    if reverse:
        rows.reverse()

    study = optuna.create_study(directions=["maximize", "minimize"])
    for label, quality, latency in rows:
        study.add_trial(
            optuna.trial.create_trial(
                values=[(quality[0] + quality[1]) / 2, latency],
                user_attrs={
                    "label": label,
                    "intervals": [quality, [latency, latency]],
                },
            )
        )
    return study


def _labels(study: optuna.Study, **kwargs: object) -> list[str]:
    return sorted(
        t.user_attrs["label"]
        for t in _get_pareto_front_trials_by_trials(
            study.trials, study.directions, intervals=True, **kwargs
        )
    )


def test_interval_front_keeps_overlapping_ranges() -> None:
    study = _study()
    assert _labels(study) == ["A", "B", "D", "E"]


def test_interval_front_eliminates_c_but_keeps_overlapping_a_b() -> None:
    study = _study(with_c=True)
    assert _labels(study) == ["A", "B", "D", "E"]
    assert _labels(_study(with_c=True, reverse=True)) == ["A", "B", "D", "E"]


def test_identical_degenerate_intervals_do_not_dominate() -> None:
    study = _study()
    assert not _dominates(study.trials[2], study.trials[3], study.directions, True)


def test_old_best_trials_still_compares_points() -> None:
    study = _study(with_c=True)
    assert [t.user_attrs["label"] for t in study.best_trials] == ["A"]


def test_study_interval_report_has_same_trial_return_type() -> None:
    study = _study(with_c=True)
    front = study.get_pareto_front_trials(interval_objectives=True)
    assert isinstance(front, list)
    assert all(isinstance(t, optuna.trial.FrozenTrial) for t in front)
    assert sorted(t.user_attrs["label"] for t in front) == ["A", "B", "D", "E"]
