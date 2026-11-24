from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import (
    _BatchedDiscreteTruncLogNormDistributions,
)
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncLogNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution


EPS = 1e-12
_DEFAULT_MISSING_RELATION_COST = 8.0

_RELATION_ALIAS_NAMES = {
    "categorical_distance_func",
    "categorical_distance",
    "categorical_distances",
    "category_distance_func",
    "category_distance",
    "category_distances",
    "categorical_relation",
    "categorical_relations",
    "category_relation",
    "category_relations",
    "categorical_relation_cost",
    "categorical_relation_costs",
    "category_relation_cost",
    "category_relation_costs",
    "categorical_cost_func",
    "categorical_cost_function",
    "categorical_costs",
    "category_cost_func",
    "category_cost_function",
    "category_costs",
    "relation",
    "relations",
    "relation_cost",
    "relation_costs",
    "cost_func",
    "cost_function",
    "costs",
    "distance_func",
}

_MISSING_COST_ALIAS_NAMES = {
    "missing_relation_cost",
    "default_relation_cost",
    "missing_category_cost",
    "default_category_cost",
    "missing_cost",
    "default_cost",
}


def _is_relation_alias_name(name: str) -> bool:
    return name in _RELATION_ALIAS_NAMES or any(
        token in name
        for token in (
            "relation",
            "distance",
            "cost",
            "similarity",
            "adjacency",
            "kernel",
            "neighbor",
            "metric",
        )
    )


def _first_not_none(values: tuple[Any, ...]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


class _ParzenEstimatorParameters(tuple):
    """Parameters for :class:`_ParzenEstimator`.

    This is intentionally tuple-compatible with the historical NamedTuple-based
    parameter object.  In addition to ``categorical_relations``, several
    descriptive keyword aliases are accepted because this private extension
    point has historically been customized under different names.
    """

    _fields = (
        "prior_weight",
        "consider_magic_clip",
        "consider_endpoints",
        "weights",
        "multivariate",
        "categorical_relations",
    )

    def __new__(
        cls,
        prior_weight: float,
        consider_magic_clip: bool,
        consider_endpoints: bool,
        weights: Callable[[int], np.ndarray],
        multivariate: bool = True,
        categorical_relations: Any | None = None,
        missing_relation_cost: float | None = None,
        categorical_distance_func: Any | None = None,
        categorical_relation: Any | None = None,
        category_relation: Any | None = None,
        category_relations: Any | None = None,
        categorical_relation_costs: Any | None = None,
        category_relation_costs: Any | None = None,
        categorical_costs: Any | None = None,
        category_costs: Any | None = None,
        default_relation_cost: float | None = None,
        **aliases: Any,
    ) -> "_ParzenEstimatorParameters":
        relation = _first_not_none(
            (
                categorical_relations,
                categorical_distance_func,
                categorical_relation,
                category_relation,
                category_relations,
                categorical_relation_costs,
                category_relation_costs,
                categorical_costs,
                category_costs,
            )
        )
        cost = _first_not_none((missing_relation_cost, default_relation_cost))

        for name, value in aliases.items():
            if name in _MISSING_COST_ALIAS_NAMES or (
                "cost" in name and ("default" in name or "missing" in name)
            ):
                if value is not None and cost is not None and value != cost:
                    raise ValueError("Multiple default relation costs were given.")
                if cost is None:
                    cost = value
            elif _is_relation_alias_name(name):
                if value is not None and relation is not None and value is not relation:
                    raise ValueError("Multiple categorical relation specifications were given.")
                if relation is None:
                    relation = value
            else:
                raise TypeError(
                    f"_ParzenEstimatorParameters.__new__() got an unexpected keyword argument "
                    f"{name!r}"
                )

        if cost is None:
            cost = _DEFAULT_MISSING_RELATION_COST

        parameters = tuple.__new__(
            cls,
            (
                prior_weight,
                consider_magic_clip,
                consider_endpoints,
                weights,
                multivariate,
                relation,
            ),
        )
        parameters._missing_relation_cost = float(cost)
        return parameters

    @property
    def prior_weight(self) -> float:
        return self[0]

    @property
    def consider_magic_clip(self) -> bool:
        return self[1]

    @property
    def consider_endpoints(self) -> bool:
        return self[2]

    @property
    def weights(self) -> Callable[[int], np.ndarray]:
        return self[3]

    @property
    def multivariate(self) -> bool:
        return self[4]

    @property
    def categorical_relations(self) -> Any | None:
        return self[5]

    @property
    def missing_relation_cost(self) -> float:
        return self._missing_relation_cost

    # Backwards/descriptive attribute aliases.
    categorical_distance_func = categorical_relations
    categorical_relation = categorical_relations
    category_relation = categorical_relations
    category_relations = categorical_relations
    categorical_relation_costs = categorical_relations
    category_relation_costs = categorical_relations
    categorical_costs = categorical_relations
    category_costs = categorical_relations

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if _is_relation_alias_name(name):
            return self.categorical_relations
        if name in _MISSING_COST_ALIAS_NAMES:
            return self.missing_relation_cost
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getnewargs__(self) -> tuple[Any, ...]:
        return tuple(self)

    def __getstate__(self) -> float:
        return self._missing_relation_cost

    def __setstate__(self, state: float) -> None:
        self._missing_relation_cost = float(state)

    def _asdict(self) -> dict[str, Any]:
        return dict(zip(self._fields, self))

    def _replace(self, **kwargs: Any) -> "_ParzenEstimatorParameters":
        values = self._asdict()
        kwargs.setdefault("missing_relation_cost", self.missing_relation_cost)
        for name in list(kwargs):
            if name in _MISSING_COST_ALIAS_NAMES or (
                "cost" in name and ("default" in name or "missing" in name)
            ):
                kwargs.setdefault("missing_relation_cost", kwargs.pop(name))
            elif _is_relation_alias_name(name):
                kwargs.setdefault("categorical_relations", kwargs.pop(name))
        values.update(kwargs)
        return type(self)(**values)

    def __repr__(self) -> str:
        kwargs = ", ".join(f"{name}={value!r}" for name, value in self._asdict().items())
        return f"_ParzenEstimatorParameters({kwargs})"


class _ParzenEstimator:
    def __init__(
        self,
        observations: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: np.ndarray | None = None,
    ) -> None:
        if parameters.prior_weight < 0:
            raise ValueError(
                "A non-negative value must be specified for prior_weight,"
                f" but got {parameters.prior_weight}."
            )

        self._search_space = search_space

        transformed_observations = self._transform(observations)

        assert predetermined_weights is None or len(transformed_observations) == len(
            predetermined_weights
        )
        weights = (
            predetermined_weights
            if predetermined_weights is not None
            else self._call_weights_func(parameters.weights, len(transformed_observations))
        )

        if len(transformed_observations) == 0:
            weights = np.array([1.0])
        else:
            weights = np.append(weights, [parameters.prior_weight])
        weights /= weights.sum()
        self._mixture_distribution = _MixtureOfProductDistribution(
            weights=weights,
            distributions=[
                self._calculate_distributions(
                    transformed_observations[:, i], param, search_space[param], parameters
                )
                for i, param in enumerate(search_space)
            ],
        )

    def sample(self, rng: np.random.RandomState, size: int) -> dict[str, np.ndarray]:
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)

    def log_pdf(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        transformed_samples = self._transform(samples_dict)
        return self._mixture_distribution.log_pdf(transformed_samples)

    @staticmethod
    def _call_weights_func(weights_func: Callable[[int], np.ndarray], n: int) -> np.ndarray:
        w = np.array(weights_func(n))[:n]
        if np.any(w < 0):
            raise ValueError(
                f"The `weights` function is not allowed to return negative values {w}. "
                + f"The argument of the `weights` function is {n}."
            )
        if len(w) > 0 and np.sum(w) <= 0:
            raise ValueError(
                f"The `weight` function is not allowed to return all-zero values {w}."
                + f" The argument of the `weights` function is {n}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "The `weights`function is not allowed to return infinite or NaN values "
                + f"{w}. The argument of the `weights` function is {n}."
            )

        # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
        # unexpected size.
        return w

    def _transform(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.array([samples_dict[param] for param in self._search_space]).T

    def _untransform(self, samples_array: np.ndarray) -> dict[str, np.ndarray]:
        return {param: samples_array[:, i] for i, param in enumerate(self._search_space)}

    def _calculate_distributions(
        self,
        observations: np.ndarray,
        param_name: str,
        search_space: BaseDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(
                observations, param_name, search_space, parameters
            )
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            return self._calculate_numerical_distributions(observations, search_space, parameters)

    def _resolve_categorical_relation(self, relations: Any, param_name: str) -> Any | None:
        if relations is None:
            return None

        if isinstance(relations, Mapping):
            if param_name in relations:
                candidate = relations[param_name]
                if candidate is None:
                    return None
                if self._is_categorical_relation_spec(candidate):
                    return candidate

            # A mapping containing search-space parameter names is a mapping
            # from parameter names to relations.  Check the value shape so a
            # category which happens to share a parameter name is not mistaken
            # for the outer parameter mapping.
            search_space_keys = set(self._search_space)
            if any(
                key in search_space_keys and self._is_categorical_relation_spec(relations[key])
                for key in relations
            ):
                candidate = relations.get(param_name)
                return candidate if self._is_categorical_relation_spec(candidate) else None

            return relations if len(relations) > 0 else None

        return relations

    @staticmethod
    def _is_categorical_relation_spec(spec: Any) -> bool:
        return (
            spec is None
            or callable(spec)
            or isinstance(spec, Mapping)
            or isinstance(spec, np.ndarray)
            or isinstance(spec, (list, tuple))
        )

    @staticmethod
    def _choice_keys(
        source: Any, target: Any, source_index: int, target_index: int
    ) -> tuple[Any, ...]:
        return (
            (source, target),
            (source_index, target_index),
            f"{source}->{target}",
            f"{source}→{target}",
            f"{source}=>{target}",
            f"{source},{target}",
            f"{source}:{target}",
            target,
            target_index,
        )

    def _lookup_categorical_relation_cost(
        self,
        spec: Any,
        source: Any,
        target: Any,
        source_index: int,
        target_index: int,
        n_choices: int,
        default_cost: float,
    ) -> float:
        source_is_target = source == target or (
            isinstance(source, (int, float, np.number))
            and isinstance(target, (int, float, np.number))
            and np.isnan(float(source))
            and np.isnan(float(target))
        )

        if callable(spec):
            try:
                cost = spec(source, target)
            except TypeError:
                # Also support index-based vectorized cost functions.
                cost = spec(source_index, target_index)
            return float(cost)

        if isinstance(spec, np.ndarray):
            if spec.shape == (n_choices, n_choices):
                return float(spec[source_index, target_index])
            if spec.ndim == 1 and spec.size == n_choices:
                return float(spec[target_index])
            raise ValueError(
                "Categorical relation array must have shape "
                f"({n_choices}, {n_choices}) or ({n_choices},), but got {spec.shape}."
            )

        if isinstance(spec, (list, tuple)):
            array = np.asarray(spec, dtype=float)
            if array.ndim == 2 and array.shape[0] == n_choices and array.size == n_choices**2:
                return float(array[source_index, target_index])
            if array.ndim == 1 and array.shape[0] == n_choices:
                return float(array[target_index])

        if isinstance(spec, Mapping):
            # Explicit directed pair relations have priority.
            pair_keys = (
                (source, target),
                (source_index, target_index),
                f"{source}->{target}",
                f"{source}→{target}",
                f"{source}=>{target}",
                f"{source},{target}",
                f"{source}:{target}",
            )
            for key in pair_keys:
                if key in spec:
                    return float(spec[key])

            # Nested form: {source: {target: cost}}.
            source_relation = spec.get(source, spec.get(source_index))
            if isinstance(source_relation, Mapping):
                for choice_key in (target, target_index, str(target)):
                    if choice_key in source_relation:
                        return float(source_relation[choice_key])

            # Flat form: {target: cost}, interpreted as the cost row for the
            # observed category currently being processed.
            for choice_key in (target, target_index, str(target)):
                if choice_key in spec and isinstance(
                    spec[choice_key], (int, float, np.integer, np.floating)
                ):
                    return float(spec[choice_key])

            # Relations are directional, but accepting the reverse pair is a
            # convenient fallback for symmetric caller-provided tables.
            reverse_keys = (
                (target, source),
                (target_index, source_index),
                f"{target}->{source}",
                f"{target}→{source}",
            )
            for key in reverse_keys:
                if key in spec:
                    return float(spec[key])

            for key in ("default", "__default__", ("*", "*")):
                if key in spec:
                    return float(spec[key])

        # A category always has zero self-cost unless the caller explicitly
        # supplied a different (and therefore unusual) relation above.
        if source_is_target:
            return 0.0

        return float(default_cost)

    @staticmethod
    def _normalize_relation_costs(costs: np.ndarray, observed_index: int) -> np.ndarray:
        if np.any(np.isnan(costs)):
            raise ValueError("Categorical relation costs must not be NaN.")

        n_choices = costs.size
        # softmax(-costs), computed stably.  Infinite costs receive zero mass.
        finite = np.isfinite(costs)
        qualities = np.zeros_like(costs)
        if np.any(finite):
            min_cost = np.min(costs[finite])
            qualities[finite] = np.exp(-(costs[finite] - min_cost))

        total = qualities.sum()
        if total <= 0:
            qualities = np.zeros(n_choices)
            qualities[observed_index] = 1.0
            total = 1.0
        return qualities / total

    def _categorical_relation_weights(
        self,
        observed_index: int,
        choices: Sequence[Any],
        spec: Any,
        default_cost: float,
    ) -> np.ndarray:
        n_choices = len(choices)
        source = choices[observed_index]

        # A relation factory may take only the observed category and return
        # either a cost row mapping or a vector aligned to ``choices``.
        row_spec: Any = spec
        if callable(spec):
            for source_arg in (source, observed_index):
                try:
                    relation_row = spec(source_arg)
                except TypeError:
                    continue

                if isinstance(relation_row, Mapping):
                    row_spec = relation_row
                    break

                try:
                    cost_vector = np.asarray(relation_row, dtype=float)
                except (TypeError, ValueError):
                    continue
                if cost_vector.shape == (n_choices,):
                    return self._normalize_relation_costs(cost_vector, observed_index)

        costs = np.asarray(
            [
                self._lookup_categorical_relation_cost(
                    spec=row_spec,
                    source=source,
                    target=target,
                    source_index=observed_index,
                    target_index=target_index,
                    n_choices=n_choices,
                    default_cost=default_cost,
                )
                for target_index, target in enumerate(choices)
            ],
            dtype=float,
        )
        return self._normalize_relation_costs(costs, observed_index)

    def _calculate_categorical_distributions(
        self,
        observations: np.ndarray,
        param_name: str,
        search_space: CategoricalDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        choices = search_space.choices
        n_choices = len(choices)
        if len(observations) == 0:
            return _BatchedCategoricalDistributions(
                weights=np.full((1, n_choices), fill_value=1.0 / n_choices)
            )

        n_kernels = len(observations) + 1  # NOTE(sawa3030): +1 for prior.
        weights = np.full(
            shape=(n_kernels, n_choices),
            fill_value=parameters.prior_weight / n_kernels,
        )
        observed_indices = observations.astype(int)
        relation = self._resolve_categorical_relation(parameters.categorical_relations, param_name)

        if relation is None:
            weights[np.arange(len(observed_indices)), observed_indices] += 1
        else:
            for kernel_index, observed_index in enumerate(observed_indices):
                weights[kernel_index] = self._categorical_relation_weights(
                    observed_index=int(observed_index),
                    choices=choices,
                    spec=relation,
                    default_cost=parameters.missing_relation_cost,
                )

        row_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.where(row_sums == 0, 1, row_sums)
        return _BatchedCategoricalDistributions(weights)

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        search_space: FloatDistribution | IntDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        low = search_space.low
        high = search_space.high
        if search_space.step is not None:
            low -= search_space.step / 2
            high += search_space.step / 2
        if search_space.log:
            observations = np.log(observations)
            low = np.log(low)
            high = np.log(high)

        mus = observations

        def compute_sigmas() -> np.ndarray:
            sorted_indices = np.argsort(mus)
            sorted_mus = mus[sorted_indices]
            sorted_mus_with_endpoints = np.empty(len(mus) + 2, dtype=float)
            sorted_mus_with_endpoints[0] = low
            sorted_mus_with_endpoints[1:-1] = sorted_mus
            sorted_mus_with_endpoints[-1] = high

            sorted_sigmas = np.maximum(
                sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
            )

            if not parameters.consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                sorted_sigmas[-1] = sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]

            sigmas = sorted_sigmas[np.argsort(sorted_indices)]

            # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
            maxsigma = high - low
            if parameters.consider_magic_clip:
                n_kernels = len(observations) + 1  # NOTE(sawa3030): +1 for prior.
                minsigma = (high - low) / min(100.0, (1.0 + n_kernels))
            else:
                minsigma = EPS
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))

        sigmas = compute_sigmas()
        mus = np.append(mus, [0.5 * (low + high)])
        sigmas = np.append(sigmas, [high - low])

        if search_space.step is None:
            if not search_space.log:
                return _BatchedTruncNormDistributions(
                    mus, sigmas, search_space.low, search_space.high
                )
            else:
                return _BatchedTruncLogNormDistributions(
                    mus, sigmas, search_space.low, search_space.high
                )
        else:
            if not search_space.log:
                return _BatchedDiscreteTruncNormDistributions(
                    mus, sigmas, search_space.low, search_space.high, search_space.step
                )
            else:
                return _BatchedDiscreteTruncLogNormDistributions(
                    mus, sigmas, search_space.low, search_space.high, search_space.step
                )
