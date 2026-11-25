"""Types used in the opt_einsum package."""

from collections import namedtuple
from collections.abc import Callable, Collection
from typing import Any, Literal, TypedDict

TensorShapeType = tuple[int, ...]
PathType = Collection[TensorShapeType]

ArrayType = Any

ArrayIndexType = frozenset[str]
ArrayShaped = namedtuple("ArrayShaped", ["shape"])

ContractionListType = list[tuple[Any, ArrayIndexType, str, tuple[str, ...] | None, str | bool]]
PathSearchFunctionType = Callable[[list[ArrayIndexType], ArrayIndexType, dict[str, int], int | None], PathType]


class PeakMemoryStep(TypedDict):
    """A single entry of the peak (simultaneously live) memory budget ledger.

    Attributes:
        step: ``0`` marks the initial state before any contraction; every
            subsequent entry numbers the contraction it describes from 1.
        contraction: The (linear) contraction indices of the step, or an
            empty tuple for the initial state entry.
        operands: Number of logical elements held by the operands of the step.
        result: Number of logical elements the result of the step occupies.
        live: Number of logical elements simultaneously live at the step,
            counting the new result before the step operands are released.
        peak: Running maximum of ``live`` up to and including this entry.
        peak_memory_limit: The active peak budget, or ``None`` if unlimited.
    """

    step: int
    contraction: tuple[int, ...]
    operands: int
    result: int
    live: int
    peak: int
    peak_memory_limit: int | None

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
)
BackendType = Literal["auto", "object", "autograd", "cupy", "dask", "jax", "theano", "tensorflow", "torch", "libjax"]
