from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.trial import FrozenTrial


# System attribute that keeps the objective vectors reported at intermediate steps of a
# multi-objective study. The stored value maps ``str(step)`` to the reported vector so that it is
# JSON serializable in every storage backend. It is deliberately kept apart from
# ``FrozenTrial.intermediate_values``: the scalar reporting path and the scalar pruners keep their
# original behavior and never see the vectors.
_INTERMEDIATE_VECTORS_KEY = "intermediate_values:vectors"


def get_intermediate_vectors(trial: FrozenTrial) -> dict[int, list[float]]:
    """Return the objective vectors reported at intermediate steps of ``trial``.

    Vectors reported through :func:`optuna.trial.Trial.report` are read from the dedicated trial
    system attribute. As a compatibility convenience, an intermediate value stored directly on the
    trial that is itself a sequence (e.g. a trial built with
    :func:`optuna.trial.create_trial` from a ``{step: (value0, value1)}`` mapping) is also treated
    as a vector. Scalar intermediate values remain untouched.

    Args:
        trial:
            A :class:`~optuna.trial.FrozenTrial` object.

    Returns:
        A mapping from step to the objective vector reported at that step. Trials that only use the
        scalar reporting path return an empty mapping.
    """

    vectors: dict[int, list[float]] = {}
    for step, value in trial.intermediate_values.items():
        if isinstance(value, (str, bytes, bytearray)) or isinstance(value, Mapping):
            continue
        if not isinstance(value, Iterable):
            continue
        try:
            vectors[int(step)] = [float(v) for v in value]
        except (TypeError, ValueError):
            # Not a numeric objective vector; leave the entry to the scalar reporting path.
            continue

    raw = trial.system_attrs.get(_INTERMEDIATE_VECTORS_KEY)
    if raw:
        # JSON (de)serialization turns the integer steps into string keys.
        vectors.update({int(step): [float(v) for v in vector] for step, vector in raw.items()})

    return vectors


def set_intermediate_vector(
    trial: FrozenTrial, storage: Any, trial_id: int, step: int, values: Sequence[float]
) -> None:
    """Persist an objective vector reported at ``step`` and mirror it on ``trial``.

    Args:
        trial:
            The cached :class:`~optuna.trial.FrozenTrial` of the running trial.
        storage:
            The storage of the study that owns the trial.
        trial_id:
            The trial id of the running trial.
        step:
            The step at which the vector is reported.
        values:
            The objective vector reported at ``step``.
    """

    vectors = get_intermediate_vectors(trial)
    vectors[step] = [float(v) for v in values]
    # String keys are required so that the mapping stays JSON serializable in every storage.
    serialized = {str(step): vector for step, vector in sorted(vectors.items())}
    storage.set_trial_system_attr(trial_id, _INTERMEDIATE_VECTORS_KEY, serialized)
    trial.system_attrs[_INTERMEDIATE_VECTORS_KEY] = serialized
