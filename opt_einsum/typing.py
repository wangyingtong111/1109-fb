"""Types used in the opt_einsum package."""

from collections import namedtuple
from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal, NamedTuple

TensorShapeType = tuple[int, ...]
PathType = Collection[TensorShapeType]

ArrayType = Any

ArrayIndexType = frozenset[str]
ArrayShaped = namedtuple("ArrayShaped", ["shape"])


class FixedNodeStep(NamedTuple):
    """A single step of a fixed-node contraction plan.

    Each step explicitly names the static node ids it consumes and the single
    node id produced. Input arrays occupy node ids ``0`` to ``num_inputs - 1``,
    intermediate nodes may use any other integer ids, and node ids are never
    recycled - once a node is consumed it cannot be referenced again.

    Parameters:
        inputs: The fixed node ids consumed by this contraction step.
        output: The fixed node id produced by this contraction step.
    """

    inputs: tuple[int, ...]
    output: int


FixedNodePathType = Sequence[FixedNodeStep]

ContractionListType = list[tuple[Any, ArrayIndexType, str, tuple[str, ...] | None, str | bool]]
PathSearchFunctionType = Callable[[list[ArrayIndexType], ArrayIndexType, dict[str, int], int | None], PathType]

# Contract kwargs
OptimizeKind = (
    None
    | int
    | Literal["optimal"]
    | Literal["dp"]
    | Literal["greedy"]
    | Literal["random-greedy"]
    | Literal["random-greedy-128"]
    | Literal["branch-all"]
    | Literal["branch-2"]
    | Literal["auto"]
    | Literal["auto-hq"]
    | PathSearchFunctionType
    | PathType
    | FixedNodePathType
)
BackendType = Literal["auto", "object", "autograd", "cupy", "dask", "jax", "theano", "tensorflow", "torch", "libjax"]
