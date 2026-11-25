"""Contains the primary optimization and contraction routines."""

from collections.abc import Collection, Iterable, Mapping, Sequence
from decimal import Decimal
from functools import lru_cache
from numbers import Integral
from typing import Any, Literal, overload

from opt_einsum import backends, blas, helpers, parser, paths, sharing
from opt_einsum.typing import (
    ArrayIndexType,
    ArrayShaped,
    ArrayType,
    BackendType,
    ContractionListType,
    OptimizeKind,
    PathType,
    TensorShapeType,
)

__all__ = [
    "contract_path",
    "contract",
    "format_const_einsum_str",
    "ContractExpression",
    "ShapeContract",
    "shape_only",
]

## Common types

_MemoryLimit = None | int | Decimal | Literal["max_input"]
_DynamicAxes = None | bool | str | Collection[Any] | Mapping[Any, Any]
_ShapeContractOption = bool | str | Collection[Any] | Mapping[str, Any] | None

_SCALAR_TYPES = (bool, int, float, complex, str, bytes)


class PathInfo:
    """A printable object to contain information about a contraction path."""

    def __init__(
        self,
        contraction_list: ContractionListType,
        input_subscripts: str,
        output_subscript: str,
        indices: ArrayIndexType,
        path: PathType,
        scale_list: Sequence[int],
        naive_cost: int,
        opt_cost: int,
        size_list: Sequence[int],
        size_dict: dict[str, int],
    ):
        self.contraction_list = contraction_list
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        self.path = path
        self.indices = indices
        self.scale_list = scale_list
        self.naive_cost = Decimal(naive_cost)
        self.opt_cost = Decimal(opt_cost)
        self.speedup = self.naive_cost / max(self.opt_cost, Decimal(1))
        self.size_list = size_list
        self.size_dict = size_dict

        self.shapes = [tuple(size_dict[k] for k in ks) for ks in input_subscripts.split(",")]
        self.eq = f"{input_subscripts}->{output_subscript}"
        self.largest_intermediate = Decimal(max(size_list, default=1))

    def __repr__(self) -> str:
        # Return the path along with a nice string representation
        header = ("scaling", "BLAS", "current", "remaining")

        path_print = [
            f"  Complete contraction:  {self.eq}\n",
            f"         Naive scaling:  {len(self.indices)}\n",
            f"     Optimized scaling:  {max(self.scale_list, default=0)}\n",
            f"      Naive FLOP count:  {self.naive_cost:.3e}\n",
            f"  Optimized FLOP count:  {self.opt_cost:.3e}\n",
            f"   Theoretical speedup:  {self.speedup:.3e}\n",
            f"  Largest intermediate:  {self.largest_intermediate:.3e} elements\n",
            "-" * 80 + "\n",
            "{:>6} {:>11} {:>22} {:>37}\n".format(*header),
            "-" * 80,
        ]

        for n, contraction in enumerate(self.contraction_list):
            _, _, einsum_str, remaining, do_blas = contraction

            if remaining is not None:
                remaining_str = ",".join(remaining) + "->" + self.output_subscript
            else:
                remaining_str = "..."
            size_remaining = max(0, 56 - max(22, len(einsum_str)))

            path_run = (
                self.scale_list[n],
                do_blas,
                einsum_str,
                remaining_str,
                size_remaining,
            )
            path_print.append("\n{:>4} {:>14} {:>22}    {:>{}}".format(*path_run))

        return "".join(path_print)


def _choose_memory_arg(memory_limit: _MemoryLimit, size_list: list[int]) -> int | None:
    if memory_limit == "max_input":
        return max(size_list)

    if isinstance(memory_limit, str):
        raise ValueError("memory_limit must be None, int, or the string Literal['max_input'].")

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        else:
            raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


_EinsumDefaultKeys = Literal["order", "casting", "dtype", "out"]


def _filter_einsum_defaults(kwargs: dict[_EinsumDefaultKeys, Any]) -> dict[_EinsumDefaultKeys, Any]:
    """Filters out default contract kwargs to pass to various backends."""
    kwargs = kwargs.copy()
    ret: dict[_EinsumDefaultKeys, Any] = {}
    if (order := kwargs.pop("order", "K")) != "K":
        ret["order"] = order

    if (casting := kwargs.pop("casting", "safe")) != "safe":
        ret["casting"] = casting

    if (dtype := kwargs.pop("dtype", None)) is not None:
        ret["dtype"] = dtype

    if (out := kwargs.pop("out", None)) is not None:
        ret["out"] = out

    ret.update(kwargs)
    return ret


def _interleaved_subscript_lists(operands: Sequence[Any]) -> list[list[Any]]:
    """Return the subscript lists of an 'interleaved' einsum input."""
    tmp_operands = list(operands)
    subscript_lists: list[list[Any]] = []
    for _ in range(len(tmp_operands) // 2):
        tmp_operands.pop(0)  # the operand itself
        subscript_lists.append(tmp_operands.pop(0))
    return subscript_lists


def _normalize_dims(dims: Iterable[Any]) -> tuple[int | None, ...]:
    """Convert dimensions to ints, keeping symbolic/unknown ones as ``None``.

    Some backends expose symbolic sizes (for example TensorFlow's ``None`` for a
    dynamic axis); those are left alone so the contract can skip their precise
    check rather than wrongly rejecting the call.
    """
    normalized: list[int | None] = []
    for dim in dims:
        if dim is None:
            normalized.append(None)
            continue
        try:
            normalized.append(int(dim))
        except (TypeError, ValueError):
            normalized.append(None)
    return tuple(normalized)


def _build_shape(shape: Any) -> tuple[int | None, ...]:
    """Normalize a *build-time* shape (a tuple of ints, or an actual array)."""
    if hasattr(shape, "shape"):
        return _normalize_dims(shape.shape)

    if isinstance(shape, _SCALAR_TYPES):
        return ()

    if isinstance(shape, Sequence):
        return _normalize_dims(shape)

    return _normalize_dims(parser.get_shape(shape))


def _call_shape(array: Any) -> tuple[int | None, ...] | None:
    """Normalize an operand supplied to a compiled expression to a shape tuple.

    Follows the repository's data semantics: ``shape`` is used when present,
    otherwise nested sequences are data (their lengths are the dimensions). An
    empty sequence is treated as a scalar-like object, mirroring ``get_shape``.
    Returns ``None`` if the shape simply cannot be inspected (for example a
    symbolic framework shape), so the contract can skip that operand.
    """
    try:
        if hasattr(array, "shape"):
            return _normalize_dims(array.shape)

        if isinstance(array, _SCALAR_TYPES):
            return ()

        if isinstance(array, Sequence) and len(array) == 0:
            return ()

        return _normalize_dims(parser.get_shape(array))
    except TypeError:
        return None


def _coerce_dims(item: Any) -> tuple[int | None, ...] | None:
    """Normalize an entry of :meth:`ShapeContract.check` to a shape tuple.

    Explicit shape tuples (and flat lists of integers) are taken as shapes, while
    arrays and nested data use the ordinary call-time shape semantics.
    """
    is_dimension_sequence = (
        isinstance(item, Sequence)
        and not isinstance(item, (str, bytes))
        and all(dim is None or isinstance(dim, Integral) for dim in item)
    )
    if is_dimension_sequence:
        return _normalize_dims(item)

    return _call_shape(item)


def _infer_variable_labels(
    input_subscripts: str,
    output_subscript: str,
    ellipsis_labels: Collection[str] = (),
) -> frozenset[str]:
    """Infer the labels that naturally vary between calls.

    These are the ellipsis broadcast batch labels and, for matmul-like equations,
    the label shared by *every* operand and the output (such as ``b`` in
    ``bij,bjk->bik``).
    """
    labels = set(ellipsis_labels)

    terms = input_subscripts.split(",")
    if len(terms) >= 2:
        shared = set(terms[0])
        for term in terms[1:]:
            shared.intersection_update(term)
        shared.intersection_update(output_subscript)
        labels.update(shared)

    return frozenset(labels)


def _resolve_variable_labels(
    dynamic_axes: _DynamicAxes,
    input_subscripts: str,
    output_subscript: str,
    ellipsis_labels: Collection[str],
    symbol_map: Mapping[Any, str] | None = None,
    has_ellipsis: bool = True,
) -> frozenset[str]:
    """Resolve the user's ``dynamic_axes`` declaration into subscript labels."""
    if dynamic_axes is None or dynamic_axes is True or (isinstance(dynamic_axes, str) and dynamic_axes == "auto"):
        return _infer_variable_labels(input_subscripts, output_subscript, ellipsis_labels)

    if dynamic_axes is False:
        return frozenset()

    if isinstance(dynamic_axes, str):
        if dynamic_axes == "...":
            dynamic_axes = [Ellipsis]
        else:
            dynamic_axes = list(dynamic_axes)

    if isinstance(dynamic_axes, Mapping):
        keys = list(dynamic_axes)
        if keys and all(isinstance(key, str) and len(key) == 1 for key in keys):
            entries: Collection[Any] = keys
        else:
            # Also accept the ``{operand_index: axis_index_or_indexes}`` form.
            terms = input_subscripts.split(",")
            positional: set[str] = set()
            for operand_index, axes in dynamic_axes.items():
                if not isinstance(operand_index, int) or not 0 <= operand_index < len(terms):
                    raise ValueError(f"`dynamic_axes` operand index {operand_index!r} is out of range.")
                if isinstance(axes, Mapping):
                    axis_indices = list(axes)
                elif isinstance(axes, int):
                    axis_indices = [axes]
                else:
                    axis_indices = list(axes)
                for axis_index in axis_indices:
                    if not isinstance(axis_index, int):
                        raise ValueError("Positional `dynamic_axes` entries must be integer axis indexes.")
                    positional.add(terms[operand_index][axis_index])
            return frozenset(positional)
    elif isinstance(dynamic_axes, int):
        raise ValueError(
            "`dynamic_axes` must name subscript label(s), not a single axis index; use a one-element sequence."
        )
    else:
        entries = dynamic_axes

    labels: set[str] = set()
    wants_ellipsis = False
    for entry in entries:
        if entry is Ellipsis or (isinstance(entry, str) and entry == "..."):
            wants_ellipsis = True
            continue

        if symbol_map is not None:
            if entry not in symbol_map:
                raise ValueError(
                    f"`dynamic_axes` contains the label {entry!r} which does not appear in the expression."
                )
            labels.add(symbol_map[entry])
            continue

        if not isinstance(entry, str) or len(entry) != 1:
            raise ValueError(
                f"`dynamic_axes` entries must be single subscript characters or `Ellipsis`, got {entry!r}."
            )
        labels.add(entry)

    if wants_ellipsis:
        if not has_ellipsis:
            raise ValueError(
                "`dynamic_axes` requested the ellipsis batch dimensions but the subscripts contain no ellipsis."
            )
        labels.update(ellipsis_labels)

    available = set(input_subscripts.replace(",", "").replace("->", "")) | set(ellipsis_labels)
    unknown = labels - available
    if unknown:
        raise ValueError(
            f"`dynamic_axes` contains {sorted(unknown)!r} which do not appear in the expression's subscripts."
        )

    return frozenset(labels)


class ShapeContract:
    """A compiled expression's shape contract: locked labels and variable labels.

    The contract is built once from the expression's subscripts and build shapes.
    Every label is *locked* to the size it had in the corresponding operand
    (contraction and channel dimensions, for example), unless it was declared
    variable via ``dynamic_axes``. Only variable labels - typically the
    broadcast batch dimensions - may change size between calls, and then only in
    a broadcast compatible way: every operand that shares the label must agree,
    with size ``1`` broadcasting against any other size.

    Both checks are performed purely from the call-time shapes. That means a
    violation is always reported *before* the backend is entered and before a
    caller supplied ``out`` buffer can be written to.

    Examples:
        ```python
        contract = ShapeContract.from_expression("...ij,...jk->...ik", [(1, 2, 3), (1, 3, 4)])
        contract.dynamic
        #> frozenset({'c'})

        contract.check([(5, 2, 3), (1, 3, 4)])
        contract.check([(5, 2, 6), (1, 6, 4)])
        #> ValueError: ... static label 'j' ...
        ```
    """

    __slots__ = ("input_subscripts", "output_subscript", "locked", "dynamic", "strict")

    def __init__(
        self,
        input_subscripts: str,
        output_subscript: str,
        locked: list[tuple[int | None, ...]],
        dynamic: Collection[str],
        strict: bool = True,
    ) -> None:
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        #: ``locked[i][axis]`` is the build-time size for operand ``i``, or ``None``
        #: if the corresponding label was declared variable.
        self.locked = locked
        self.dynamic = frozenset(dynamic)
        #: Whether the contract is enforced on *every* call (or only when ``out``
        #: is supplied, which keeps plain calls of legacy expressions permissive).
        self.strict = strict

    @classmethod
    def from_expression(
        cls,
        subscripts: str,
        shapes: Sequence[Any],
        dynamic_axes: _DynamicAxes = None,
        strict: bool = True,
        symbol_map: Mapping[Any, str] | None = None,
    ) -> "ShapeContract":
        """Build a :class:`ShapeContract` for ``subscripts`` and ``shapes``."""
        normalized = [_build_shape(shape) for shape in shapes]
        input_subscripts, output_subscript, _ = parser.parse_einsum_input([subscripts] + list(normalized), shapes=True)

        ellipsis_labels = parser.find_ellipsis_labels(subscripts, normalized)
        dynamic = _resolve_variable_labels(
            dynamic_axes,
            input_subscripts,
            output_subscript,
            ellipsis_labels,
            symbol_map=symbol_map,
            has_ellipsis="..." in subscripts,
        )

        locked = [
            tuple(None if label in dynamic else dim for label, dim in zip(term, shape))
            for term, shape in zip(input_subscripts.split(","), normalized)
        ]

        return cls(input_subscripts, output_subscript, locked, dynamic, strict=strict)

    def check(self, shapes: Sequence[Any], ranks_are_fatal: bool = True) -> bool:
        """Validate call-time ``shapes`` against this contract.

        Each entry may be an array (its ``shape`` is used) or an explicit shape
        tuple. Raises ``ValueError`` naming the offending operand and subscript
        label, and does so before any contraction or array write can happen.

        If ``ranks_are_fatal`` is ``False`` then a rank mismatch simply returns
        ``False`` instead of raising, leaving the call to the backend's own error
        handling - used by the implicit contract to keep legacy behaviour intact.
        """
        terms = self.input_subscripts.split(",")
        if len(shapes) != len(terms):
            raise ValueError(
                "Shape contract check failed before backend dispatch or writing `out`: this expression was built "
                f"for {len(terms)} operands but received {len(shapes)}."
            )

        dims_by_operand = [_coerce_dims(shape) for shape in shapes]
        # operands whose shape can not be inspected are left to the backend
        checkable = [dims is not None for dims in dims_by_operand]
        rank_problems = [
            f"Operand {i} has rank {len(dims)} (shape {dims}), but this expression's shape contract "
            f"expects rank {len(term)} (subscript {term!r})."
            for i, (term, dims) in enumerate(zip(terms, dims_by_operand))
            if dims is not None and len(dims) != len(term)
        ]
        if rank_problems:
            if not ranks_are_fatal:
                return False
            raise ValueError(
                "Shape contract check failed before backend dispatch or writing `out`: " + " ".join(rank_problems)
            )

        problems: list[str] = []
        observed: dict[str, list[tuple[int, int]]] = {label: [] for label in self.dynamic}

        for i, (term, locked_term, dims, check) in enumerate(zip(terms, self.locked, dims_by_operand, checkable)):
            if not check:
                continue
            for axis, (label, dim, expected) in enumerate(zip(term, dims, locked_term)):
                if dim is None:
                    # a symbolic / unknown call-time dimension can not be checked
                    continue
                if label in self.dynamic:
                    observed[label].append((i, dim))
                elif expected is not None and dim != expected:
                    problems.append(
                        f"Operand {i} has size {dim} for static label {label!r} at axis {axis} of subscript "
                        f"{term!r}, but this expression was built with size {expected} for that position. "
                        "Static labels are locked by the expression's shape contract."
                    )

        for label in sorted(self.dynamic):
            sizes = observed[label]
            non_unit = [(i, dim) for i, dim in sizes if dim != 1]
            if len(non_unit) < 2:
                continue
            first_index, first_dim = non_unit[0]
            for other_index, other_dim in non_unit[1:]:
                if other_dim != first_dim:
                    problems.append(
                        f"Dynamic label {label!r} is {first_dim} in operand {first_index} but {other_dim} in "
                        f"operand {other_index}. Variable labels may change together, but every operand must "
                        "remain broadcast compatible (equal sizes, or 1 on all but one operand)."
                    )

        if problems:
            raise ValueError(
                "Shape contract check failed before backend dispatch or writing `out`: " + " ".join(problems)
            )

        return True

    def __repr__(self) -> str:
        dynamic = "".join(sorted(self.dynamic)) or "-"
        return f"<ShapeContract('{self.input_subscripts}->{self.output_subscript}', dynamic={dynamic!r})>"


def _resolve_shape_contract(
    shape_contract: _ShapeContractOption,
    dynamic_axes: _DynamicAxes,
) -> tuple[bool | None, _DynamicAxes]:
    """Normalize the ``shape_contract`` / ``dynamic_axes`` options.

    Returns whether the contract is enforced on every call (``True``), only when
    a caller supplied ``out`` is written to (``None``), or not at all (``False``),
    together with the (possibly inferred) variable-axis declaration.
    """
    if shape_contract is None:
        # No explicit request: keep legacy permissive calls, but still guard
        # writes into a caller supplied ``out`` buffer.
        return (True if dynamic_axes is not None else None), dynamic_axes

    if shape_contract is False:
        return False, dynamic_axes

    if isinstance(shape_contract, Mapping):
        options = dict(shape_contract)
        enabled = options.pop("enabled", True)
        strict = options.pop("strict", True)
        if "dynamic_axes" in options:
            if dynamic_axes is None:
                dynamic_axes = options.pop("dynamic_axes")
        if options:
            raise TypeError(f"Did not understand the following `shape_contract` options: {options.keys()}")
        if not enabled:
            return False, dynamic_axes
        # ``strict=False`` keeps the legacy permissive plain calls, while still
        # validating before a caller supplied ``out`` buffer is written.
        return (True if strict else None), dynamic_axes

    if dynamic_axes is None:
        # ``shape_contract=True`` enables the contract; a label or collection of
        # labels passed as ``shape_contract`` doubles as the declaration.
        dynamic_axes = True if shape_contract is True else shape_contract

    return True, dynamic_axes


# Overlaod for contract(einsum_string, *operands)
@overload
def contract_path(
    subscripts: str,
    *operands: ArrayType,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    **kwargs: Any,
) -> tuple[PathType, PathInfo]: ...


# Overlaod for contract(operand, indices, operand, indices, ....)
@overload
def contract_path(
    subscripts: ArrayType,
    *operands: ArrayType | Collection[int],
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    **kwargs: Any,
) -> tuple[PathType, PathInfo]: ...


def contract_path(
    subscripts: Any,
    *operands: Any,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    shapes: bool = False,
    **kwargs: Any,
) -> tuple[PathType, PathInfo]:
    """Find a contraction order `path`, without performing the contraction.

    Parameters:
          subscripts: Specifies the subscripts for summation.
          *operands: These are the arrays for the operation.
          use_blas: Do you use BLAS for valid operations, may use extra memory for more intermediates.
          optimize: Choose the type of path the contraction will be optimized with.
                - if a list is given uses this as the path.
                - `'optimal'` An algorithm that explores all possible ways of
                contracting the listed tensors. Scales factorially with the number of
                terms in the contraction.
                - `'dp'` A faster (but essentially optimal) algorithm that uses
                dynamic programming to exhaustively search all contraction paths
                without outer-products.
                - `'greedy'` An cheap algorithm that heuristically chooses the best
                pairwise contraction at each step. Scales linearly in the number of
                terms in the contraction.
                - `'random-greedy'` Run a randomized version of the greedy algorithm
                32 times and pick the best path.
                - `'random-greedy-128'` Run a randomized version of the greedy
                algorithm 128 times and pick the best path.
                - `'branch-all'` An algorithm like optimal but that restricts itself
                to searching 'likely' paths. Still scales factorially.
                - `'branch-2'` An even more restricted version of 'branch-all' that
                only searches the best two options at each step. Scales exponentially
                with the number of terms in the contraction.
                - `'auto'` Choose the best of the above algorithms whilst aiming to
                keep the path finding time below 1ms.
                - `'auto-hq'` Aim for a high quality contraction, choosing the best
                of the above algorithms whilst aiming to keep the path finding time
                below 1sec.

          memory_limit: Give the upper bound of the largest intermediate tensor contract will build.
                - None or -1 means there is no limit
                - `max_input` means the limit is set as largest input tensor
                - a positive integer is taken as an explicit limit on the number of elements

                The default is None. Note that imposing a limit can make contractions
                exponentially slower to perform.

          shapes: Whether ``contract_path`` should assume arrays (the default) or array shapes have been supplied.

    Returns:
          path: The optimized einsum contraciton path
          PathInfo: A printable object containing various information about the path found.

    Notes:
          The resulting path indicates which terms of the input contraction should be
          contracted first, the result of this contraction is then appended to the end of
          the contraction list.

    Examples:
          We can begin with a chain dot example. In this case, it is optimal to
          contract the b and c tensors represented by the first element of the path (1,
          2). The resulting tensor is added to the end of the contraction and the
          remaining contraction, `(0, 1)`, is then executed.

      ```python
      a = np.random.rand(2, 2)
      b = np.random.rand(2, 5)
      c = np.random.rand(5, 2)
      path_info = opt_einsum.contract_path('ij,jk,kl->il', a, b, c)
      print(path_info[0])
      #> [(1, 2), (0, 1)]
      print(path_info[1])
      #>   Complete contraction:  ij,jk,kl->il
      #>          Naive scaling:  4
      #>      Optimized scaling:  3
      #>       Naive FLOP count:  1.600e+02
      #>   Optimized FLOP count:  5.600e+01
      #>    Theoretical speedup:  2.857
      #>   Largest intermediate:  4.000e+00 elements
      #> -------------------------------------------------------------------------
      #> scaling                  current                                remaining
      #> -------------------------------------------------------------------------
      #>    3                   kl,jk->jl                                ij,jl->il
      #>    3                   jl,ij->il                                   il->il
      ```

      A more complex index transformation example.

      ```python
      I = np.random.rand(10, 10, 10, 10)
      C = np.random.rand(10, 10)
      path_info = oe.contract_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)

      print(path_info[0])
      #> [(0, 2), (0, 3), (0, 2), (0, 1)]
      print(path_info[1])
      #>   Complete contraction:  ea,fb,abcd,gc,hd->efgh
      #>          Naive scaling:  8
      #>      Optimized scaling:  5
      #>       Naive FLOP count:  8.000e+08
      #>   Optimized FLOP count:  8.000e+05
      #>    Theoretical speedup:  1000.000
      #>   Largest intermediate:  1.000e+04 elements
      #> --------------------------------------------------------------------------
      #> scaling                  current                                remaining
      #> --------------------------------------------------------------------------
      #>    5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
      #>    5               bcde,fb->cdef                         gc,hd,cdef->efgh
      #>    5               cdef,gc->defg                            hd,defg->efgh
      #>    5               defg,hd->efgh                               efgh->efgh
      ```
    """
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    # Hidden option, only einsum should call this
    einsum_call_arg = kwargs.pop("einsum_call", False)
    if len(kwargs):
        raise TypeError(f"Did not understand the following kwargs: {kwargs.keys()}")

    # Python side parsing
    operands_ = [subscripts] + list(operands)
    input_subscripts, output_subscript, operands_prepped = parser.parse_einsum_input(operands_, shapes=shapes)

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    input_sets = [frozenset(x) for x in input_list]
    if shapes:
        input_shapes = list(operands_prepped)
    else:
        input_shapes = [parser.get_shape(x) for x in operands_prepped]
    output_set = frozenset(output_subscript)
    indices = frozenset(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict: dict[str, int] = {}
    for tnum, term in enumerate(input_list):
        sh = input_shapes[tnum]

        if len(sh) != len(term):
            raise ValueError(
                f"Einstein sum subscript '{input_list[tnum]}' does not contain the "
                f"correct number of indices for operand {tnum}."
            )
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        f"Size of label '{char}' for operand {tnum} ({size_dict[char]}) does not match previous "
                        f"terms ({dim})."
                    )
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [helpers.compute_size_by_dict(term, size_dict) for term in input_list + [output_subscript]]
    memory_arg = _choose_memory_arg(memory_limit, size_list)

    num_ops = len(input_list)

    # Compute naive cost
    # This is not quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, num_ops, size_dict)

    # Compute the path
    if optimize is False:
        path_tuple: PathType = [tuple(range(num_ops))]
    elif not isinstance(optimize, (str, paths.PathOptimizer)):
        # Custom path supplied
        path_tuple = optimize  # type: ignore
    elif num_ops <= 2:
        # Nothing to be optimized
        path_tuple = [tuple(range(num_ops))]
    elif isinstance(optimize, paths.PathOptimizer):
        # Custom path optimizer supplied
        path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
    else:
        path_optimizer = paths.get_path_fn(optimize)
        path_tuple = path_optimizer(input_sets, output_set, size_dict, memory_arg)

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path_tuple):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract_tuple = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size
        cost = helpers.flop_count(idx_contract, bool(idx_removed), len(contract_inds), size_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shapes.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = blas.can_blas(tmp_inputs, "".join(out_inds), idx_removed, tmp_shapes)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path_tuple)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        input_shapes.append(shp_result)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining: tuple[str, ...] | None = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return operands_prepped, contraction_list  # type: ignore

    path_print = PathInfo(
        contraction_list,
        input_subscripts,
        output_subscript,
        indices,
        path_tuple,
        scale_list,
        naive_cost,
        opt_cost,
        size_list,
        size_dict,
    )

    return path_tuple, path_print


@sharing.einsum_cache_wrap
def _einsum(*operands: Any, **kwargs: Any) -> ArrayType:
    """Base einsum, but with pre-parse for valid characters if a string is given."""
    fn = backends.get_func("einsum", kwargs.pop("backend", "numpy"))

    if not isinstance(operands[0], str):
        return fn(*operands, **kwargs)

    einsum_str, operands = operands[0], operands[1:]

    # Do we need to temporarily map indices into [a-z,A-Z] range?
    if not parser.has_valid_einsum_chars_only(einsum_str):
        # Explicitly find output str first so as to maintain order
        if "->" not in einsum_str:
            einsum_str += "->" + parser.find_output_str(einsum_str)

        einsum_str = parser.convert_to_valid_einsum_chars(einsum_str)

    kwargs = _filter_einsum_defaults(kwargs)  # type: ignore
    return fn(einsum_str, *operands, **kwargs)


def _default_transpose(x: ArrayType, axes: tuple[int, ...]) -> ArrayType:
    #  most libraries implement a method version
    return x.transpose(axes)


@sharing.transpose_cache_wrap
def _transpose(x: ArrayType, axes: tuple[int, ...], backend: str = "numpy") -> ArrayType:
    """Base transpose."""
    fn = backends.get_func("transpose", backend, _default_transpose)
    return fn(x, axes)


@sharing.tensordot_cache_wrap
def _tensordot(x: ArrayType, y: ArrayType, axes: tuple[int, ...], backend: str = "numpy") -> ArrayType:
    """Base tensordot."""
    fn = backends.get_func("tensordot", backend)
    return fn(x, y, axes=axes)


# Rewrite einsum to handle different cases


@overload
def contract(
    subscripts: str,
    *operands: ArrayType,
    out: ArrayType = ...,
    use_blas: bool = ...,
    optimize: OptimizeKind = ...,
    memory_limit: _MemoryLimit = ...,
    backend: BackendType = ...,
    **kwargs: Any,
) -> ArrayType: ...


@overload
def contract(
    subscripts: ArrayType,
    *operands: ArrayType | Collection[int],
    out: ArrayType = ...,
    use_blas: bool = ...,
    optimize: OptimizeKind = ...,
    memory_limit: _MemoryLimit = ...,
    backend: BackendType = ...,
    **kwargs: Any,
) -> ArrayType: ...


def contract(
    subscripts: str | ArrayType,
    *operands: ArrayType | Collection[int],
    out: ArrayType | None = None,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    backend: BackendType = "auto",
    **kwargs: Any,
) -> ArrayType:
    """Evaluates the Einstein summation convention on the operands. A drop in
    replacement for NumPy's einsum function that optimizes the order of contraction
    to reduce overall scaling at the cost of several intermediate arrays.

    Parameters:
        subscripts: Specifies the subscripts for summation.
        *operands: These are the arrays for the operation.
        out: A output array in which set the resulting output.
        use_blas: Do you use BLAS for valid operations, may use extra memory for more intermediates.
        optimize:- Choose the type of path the contraction will be optimized with
            - if a list is given uses this as the path.
            - `'optimal'` An algorithm that explores all possible ways of
            contracting the listed tensors. Scales factorially with the number of
            terms in the contraction.
            - `'dp'` A faster (but essentially optimal) algorithm that uses
            dynamic programming to exhaustively search all contraction paths
            without outer-products.
            - `'greedy'` An cheap algorithm that heuristically chooses the best
            pairwise contraction at each step. Scales linearly in the number of
            terms in the contraction.
            - `'random-greedy'` Run a randomized version of the greedy algorithm
            32 times and pick the best path.
            - `'random-greedy-128'` Run a randomized version of the greedy
            algorithm 128 times and pick the best path.
            - `'branch-all'` An algorithm like optimal but that restricts itself
            to searching 'likely' paths. Still scales factorially.
            - `'branch-2'` An even more restricted version of 'branch-all' that
            only searches the best two options at each step. Scales exponentially
            with the number of terms in the contraction.
            - `'auto', None, True` Choose the best of the above algorithms whilst aiming to
            keep the path finding time below 1ms.
            - `'auto-hq'` Aim for a high quality contraction, choosing the best
            of the above algorithms whilst aiming to keep the path finding time
            below 1sec.
            - `False` will not optimize the contraction.

        memory_limit:- Give the upper bound of the largest intermediate tensor contract will build.
            - None or -1 means there is no limit.
            - `max_input` means the limit is set as largest input tensor.
            - A positive integer is taken as an explicit limit on the number of elements.

            The default is None. Note that imposing a limit can make contractions
            exponentially slower to perform.

        backend: Which library to use to perform the required ``tensordot``, ``transpose``
            and ``einsum`` calls. Should match the types of arrays supplied, See
            `contract_expression` for generating expressions which convert
            numpy arrays to and from the backend library automatically.

    Returns:
        The result of the einsum expression.

    Notes:
        This function should produce a result identical to that of NumPy's einsum
        function. The primary difference is ``contract`` will attempt to form
        intermediates which reduce the overall scaling of the given einsum contraction.
        By default the worst intermediate formed will be equal to that of the largest
        input array. For large einsum expressions with many input arrays this can
        provide arbitrarily large (1000 fold+) speed improvements.

        For contractions with just two tensors this function will attempt to use
        NumPy's built-in BLAS functionality to ensure that the given operation is
        performed optimally. When NumPy is linked to a threaded BLAS, potential
        speedups are on the order of 20-100 for a six core machine.
    """
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    operands_list = [subscripts] + list(operands)

    # If no optimization, run pure einsum
    if optimize is False:
        input_subscripts, output_subscript, arrays = parser.parse_einsum_input(operands_list)
        _backend = parse_backend(arrays, backend)
        return _einsum(f"{input_subscripts}->{output_subscript}", *arrays, backend=_backend, out=out, **kwargs)

    # Grab non-einsum kwargs
    gen_expression = kwargs.pop("_gen_expression", False)
    constants_dict = kwargs.pop("_constants_dict", {})
    shape_contract = kwargs.pop("_shape_contract", None)

    if gen_expression:
        full_str = operands_list[0]

    # Build the contraction list and operand
    contraction_list: ContractionListType
    operands, contraction_list = contract_path(  # type: ignore
        *operands_list, optimize=optimize, memory_limit=memory_limit, einsum_call=True, use_blas=use_blas
    )

    # check if performing contraction or just building expression
    if gen_expression:
        return ContractExpression(
            full_str, contraction_list, constants_dict, _shape_contract=shape_contract, **kwargs
        )

    return _core_contract(operands, contraction_list, backend=backend, out=out, **kwargs)


@lru_cache(None)
def _infer_backend_class_cached(cls: type) -> str:
    return cls.__module__.split(".")[0]


def infer_backend(x: Any) -> str:
    return _infer_backend_class_cached(x.__class__)


def parse_backend(arrays: Sequence[ArrayType], backend: str | None) -> str:
    """Find out what backend we should use, dipatching based on the first
    array if ``backend='auto'`` is specified.
    """
    if (backend != "auto") and (backend is not None):
        return backend
    backend = infer_backend(arrays[0])

    # some arrays will be defined in modules that don't implement tensordot
    # etc. so instead default to numpy
    if not backends.has_tensordot(backend):
        return "numpy"

    return backend


def _core_contract(
    operands_: Sequence[ArrayType],
    contraction_list: ContractionListType,
    backend: str | None = "auto",
    evaluate_constants: bool = False,
    out: ArrayType | None = None,
    **kwargs: Any,
) -> ArrayType:
    """Inner loop used to perform an actual contraction given the output
    from a ``contract_path(..., einsum_call=True)`` call.
    """
    # Special handling if out is specified
    specified_out = out is not None

    operands = list(operands_)
    backend = parse_backend(operands, backend)

    # try and do as much as possible without einsum if not available
    no_einsum = not backends.has_einsum(backend)

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, _, blas_flag = contraction

        # check if we are performing the pre-pass of an expression with constants,
        #     if so, break out upon finding first non-constant (None) operand
        if evaluate_constants and any(operands[x] is None for x in inds):
            return operands, contraction_list[num:]

        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # Call tensordot (check if should prefer einsum, but only if available)
        if blas_flag and ("EINSUM" not in blas_flag or no_einsum):  # type: ignore
            # Checks have already been handled
            input_str, results_index = einsum_str.split("->")
            input_left, input_right = input_str.split(",")

            tensor_result = "".join(s for s in input_left + input_right if s not in idx_rm)

            if idx_rm:
                # Find indices to contract over
                left_pos, right_pos = [], []
                for s in idx_rm:
                    left_pos.append(input_left.find(s))
                    right_pos.append(input_right.find(s))

                # Construct the axes tuples in a canonical order
                axes = tuple(zip(*sorted(zip(left_pos, right_pos))))
            else:
                # Ensure axes is always pair of tuples
                axes = ((), ())

            # Contract!
            new_view = _tensordot(*tmp_operands, axes=axes, backend=backend, **kwargs)

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:
                transpose = tuple(map(tensor_result.index, results_index))
                new_view = _transpose(new_view, axes=transpose, backend=backend)

                if handle_out:
                    out[...] = new_view  # type: ignore

        else:
            # Call einsum
            out_kwarg: None | ArrayType = None
            if handle_out:
                out_kwarg = out
            new_view = _einsum(einsum_str, *tmp_operands, backend=backend, out=out_kwarg, **kwargs)

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out
    else:
        return operands[0]


def format_const_einsum_str(einsum_str: str, constants: Iterable[int]) -> str:
    """Add brackets to the constant terms in ``einsum_str``. For example:

        >>> format_const_einsum_str('ab,bc,cd->ad', [0, 2])
        'bc,[ab,cd]->ad'

    No-op if there are no constants.
    """
    if not constants:
        return einsum_str

    if "->" in einsum_str:
        lhs, rhs = einsum_str.split("->")
        arrow = "->"
    else:
        lhs, rhs, arrow = einsum_str, "", ""

    wrapped_terms = [f"[{t}]" if i in constants else t for i, t in enumerate(lhs.split(","))]

    formatted_einsum_str = "{}{}{}".format(",".join(wrapped_terms), arrow, rhs)

    # merge adjacent constants
    formatted_einsum_str = formatted_einsum_str.replace("],[", ",")
    return formatted_einsum_str


class ContractExpression:
    """Helper class for storing an explicit ``contraction_list`` which can
    then be repeatedly called solely with the array arguments.
    """

    def __init__(
        self,
        contraction: str,
        contraction_list: ContractionListType,
        constants_dict: dict[int, ArrayType],
        **kwargs: Any,
    ):
        # the compiled shape contract is kept separate from the backend kwargs
        self._shape_contract: ShapeContract | None = kwargs.pop("_shape_contract", None)

        self.contraction = format_const_einsum_str(contraction, constants_dict.keys())
        self.contraction_list = contraction_list
        self.kwargs = kwargs

        # need to know _full_num_args to parse constants with, and num_args to call with
        self._full_num_args = contraction.count(",") + 1
        self.num_args = self._full_num_args - len(constants_dict)

        # likewise need to know full contraction list
        self._full_contraction_list = contraction_list

        self._constants_dict = constants_dict
        self._evaluated_constants: dict[str, Any] = {}
        self._backend_expressions: dict[str, Any] = {}

    @property
    def shape_contract(self) -> ShapeContract | None:
        """The compiled :class:`ShapeContract`, if one is attached to the expression."""
        return self._shape_contract

    def _call_shapes(self, arrays: Sequence[ArrayType]) -> list[TensorShapeType]:
        """Build the full per-operand shape list for a call.

        Constant operands are substituted with their build-time shapes so the shape
        contract can be checked for every operand without evaluating any constants
        or touching the backend.
        """
        var_shapes = iter(_call_shape(array) for array in arrays)
        return [
            _build_shape(self._constants_dict[i]) if i in self._constants_dict else next(var_shapes)
            for i in range(self._full_num_args)
        ]

    def _validate_shapes(self, arrays: Sequence[ArrayType], out: ArrayType | None) -> None:
        """Run the shape contract, if it applies to this call.

        A contract that was explicitly requested is enforced on every call. The
        implicit contract attached to every expression is only enforced when a
        caller supplied ``out`` buffer is about to be written, which keeps plain
        legacy calls permissive.
        """
        contract = self._shape_contract
        if contract is None:
            return
        if not (contract.strict or out is not None):
            return

        contract.check(self._call_shapes(arrays), ranks_are_fatal=contract.strict)

    def evaluate_constants(self, backend: str | None = "auto") -> None:
        """Convert any constant operands to the correct backend form, and
        perform as many contractions as possible to create a new list of
        operands, stored in ``self._evaluated_constants[backend]``. This also
        makes sure ``self.contraction_list`` only contains the remaining,
        non-const operations.
        """
        # prepare a list of operands, with `None` for non-consts
        tmp_const_ops = [self._constants_dict.get(i, None) for i in range(self._full_num_args)]
        backend = parse_backend(tmp_const_ops, backend)

        # get the new list of operands with constant operations performed, and remaining contractions
        try:
            new_ops, new_contraction_list = backends.evaluate_constants(backend, tmp_const_ops, self)
        except KeyError:
            new_ops, new_contraction_list = self(*tmp_const_ops, backend=backend, evaluate_constants=True)

        self._evaluated_constants[backend] = new_ops
        self.contraction_list = new_contraction_list

    def _get_evaluated_constants(self, backend: str) -> list[ArrayType | None]:
        """Retrieve or generate the cached list of constant operators (mixed
        in with None representing non-consts) and the remaining contraction
        list.
        """
        try:
            return self._evaluated_constants[backend]
        except KeyError:
            self.evaluate_constants(backend)
            return self._evaluated_constants[backend]

    def _get_backend_expression(self, arrays: Sequence[ArrayType], backend: str) -> Any:
        try:
            return self._backend_expressions[backend]
        except KeyError:
            fn = backends.build_expression(backend, arrays, self)
            self._backend_expressions[backend] = fn
            return fn

    def _contract(
        self,
        arrays: Sequence[ArrayType],
        out: ArrayType | None = None,
        backend: str | None = "auto",
        evaluate_constants: bool = False,
    ) -> ArrayType:
        """The normal, core contraction."""
        contraction_list = self._full_contraction_list if evaluate_constants else self.contraction_list

        return _core_contract(
            list(arrays),
            contraction_list,
            out=out,
            backend=backend,
            evaluate_constants=evaluate_constants,
            **self.kwargs,
        )

    def _contract_with_conversion(
        self,
        arrays: Sequence[ArrayType],
        out: ArrayType | None,
        backend: str,
        evaluate_constants: bool = False,
    ) -> ArrayType:
        """Special contraction, i.e., contraction with a different backend
        but converting to and from that backend. Retrieves or generates a
        cached expression using ``arrays`` as templates, then calls it
        with ``arrays``.

        If ``evaluate_constants=True``, perform a partial contraction that
        prepares the constant tensors and operations with the right backend.
        """
        # convert consts to correct type & find reduced contraction list
        if evaluate_constants:
            return backends.evaluate_constants(backend, arrays, self)

        result = self._get_backend_expression(arrays, backend)(*arrays)

        if out is not None:
            out[()] = result
            return out

        return result

    def __call__(
        self,
        *arrays: ArrayType,
        out: None | ArrayType = None,
        backend: str = "auto",
        evaluate_constants: bool = False,
    ) -> ArrayType:
        """Evaluate this expression with a set of arrays.

        Parameters:
            arrays: The arrays to supply as input to the expression.
            out: If specified, output the result into this array.
            backend: Perform the contraction with this backend library. If numpy arrays
                are supplied then try to convert them to and from the correct
                backend array type.
            evaluate_constants: Pre-evaluates constants with the appropriate backend.

        Returns:
            The contracted result.
        """
        correct_num_args = self._full_num_args if evaluate_constants else self.num_args

        if len(arrays) != correct_num_args:
            raise ValueError(
                f"This `ContractExpression` takes exactly {self.num_args} array arguments but received {len(arrays)}."
            )

        if not evaluate_constants:
            # Validate before backend inference/dispatch, constant evaluation, or
            # any possible write into a caller supplied ``out`` buffer.
            self._validate_shapes(arrays, out)

        backend = parse_backend(arrays, backend)

        if self._constants_dict and not evaluate_constants:
            # fill in the missing non-constant terms with newly supplied arrays
            ops_var, ops_const = iter(arrays), self._get_evaluated_constants(backend)
            ops: Sequence[ArrayType] = [next(ops_var) if op is None else op for op in ops_const]
        else:
            ops = arrays

        try:
            # Check if the backend requires special preparation / calling
            #   but also ignore non-numpy arrays -> assume user wants same type back
            if backends.has_backend(backend) and all(infer_backend(x) == "numpy" for x in arrays):
                return self._contract_with_conversion(ops, out, backend, evaluate_constants=evaluate_constants)

            return self._contract(ops, out=out, backend=backend, evaluate_constants=evaluate_constants)

        except ValueError as err:
            original_msg = str(err.args) if err.args else ""
            msg = (
                "Internal error while evaluating `ContractExpression`. Note that few checks are performed"
                " - the number and rank of the array arguments must match the original expression. "
                f"The internal error was: '{original_msg}'",
            )
            err.args = msg
            raise

    def __repr__(self) -> str:
        if self._constants_dict:
            constants_repr = f", constants={sorted(self._constants_dict)}"
        else:
            constants_repr = ""
        return f"<ContractExpression('{self.contraction}'{constants_repr})>"

    def __str__(self) -> str:
        s = [self.__repr__()]
        for i, c in enumerate(self.contraction_list):
            s.append(f"\n  {i + 1}.  ")
            s.append(f"'{c[2]}'" + (f" [{c[-1]}]" if c[-1] else ""))
            s.append(f"\neinsum_kwargs={self.kwargs}")
        return "".join(s)


def shape_only(shape: TensorShapeType) -> ArrayShaped:
    """Dummy ``numpy.ndarray`` which has a shape only - for generating
    contract expressions.
    """
    return ArrayShaped(shape)


# Overlaod for contract(einsum_string, *operands)
@overload
def contract_expression(
    subscripts: str,
    *operands: ArrayType | TensorShapeType,
    constants: Collection[int] | None = ...,
    use_blas: bool = ...,
    optimize: OptimizeKind = ...,
    memory_limit: _MemoryLimit = ...,
    dynamic_axes: _DynamicAxes = ...,
    shape_contract: _ShapeContractOption = ...,
    **kwargs: Any,
) -> ContractExpression: ...


# Overlaod for contract(operand, indices, operand, indices, ....)
@overload
def contract_expression(
    subscripts: ArrayType | TensorShapeType,
    *operands: ArrayType | TensorShapeType | Collection[int],
    constants: Collection[int] | None = ...,
    use_blas: bool = ...,
    optimize: OptimizeKind = ...,
    memory_limit: _MemoryLimit = ...,
    dynamic_axes: _DynamicAxes = ...,
    shape_contract: _ShapeContractOption = ...,
    **kwargs: Any,
) -> ContractExpression: ...


def contract_expression(
    subscripts: str | ArrayType | TensorShapeType,
    *shapes: ArrayType | TensorShapeType | Collection[int],
    constants: Collection[int] | None = None,
    use_blas: bool = True,
    optimize: OptimizeKind = True,
    memory_limit: _MemoryLimit = None,
    dynamic_axes: _DynamicAxes = None,
    shape_contract: _ShapeContractOption = None,
    **kwargs: Any,
) -> ContractExpression:
    """Generate a reusable expression for a given contraction with
    specific shapes, which can, for example, be cached.

    Parameters:

        subscripts: Specifies the subscripts for summation.
        shapes: Shapes of the arrays to optimize the contraction for.
        constants: The indices of any constant arguments in `shapes`, in which case the
            actual array should be supplied at that position rather than just a
            shape. If these are specified, then constant parts of the contraction
            between calls will be reused. Additionally, if a GPU-enabled backend is
            used for example, then the constant tensors will be kept on the GPU,
            minimizing transfers.
        dynamic_axes: Declare which subscript labels are allowed to change size
            between calls - typically the broadcast batch dimensions. Every other
            label is *locked* to the size it had in `shapes`, so a call that
            changes a contraction or channel dimension is rejected. Supplying this
            argument enables the expression's shape contract on every call.
            Accepts:

            - ``None`` (the default): no explicit declaration. Plain calls keep
              the legacy permissive behaviour, but shapes are still validated
              before a caller supplied ``out`` buffer is written.
            - ``True`` or ``"auto"``: infer the variable labels - the ellipsis
              broadcast batch labels and, for matmul-like equations, the label
              shared by every input and the output (e.g. ``b`` in
              ``bij,bjk->bik``).
            - ``False``: lock every label.
            - a subscript character or iterable of characters: only those labels
              may change size. ``Ellipsis`` or ``"..."`` additionally marks the
              ellipsis batch dimensions as variable.
            - ``{operand_index: axis_index_or_indexes}``: declare labels by the
              axis positions they occupy in the expression.

            Whatever the declaration, every operand that shares a variable label
            must remain broadcast compatible (equal sizes, or ``1`` on all but one
            operand). Both the locked-label and the broadcast checks run *before*
            the backend is entered or ``out`` is written to, and a rejected call
            leaves a constant cache untouched.
        shape_contract: Enable the shape contract without declaring anything
            explicit; ``True`` behaves like ``dynamic_axes="auto"``. A label or
            collection of labels may also be passed here as a shorthand for
            ``dynamic_axes``. ``False`` disables the contract entirely, restoring
            completely unchecked calls.
        kwargs: Passed on to `contract_path` or `einsum`. See `contract`.

    Returns:
        Callable with signature `expr(*arrays, out=None, backend='numpy')` where the array's shapes should match `shapes`.

    Notes:
        The `out` keyword argument should be supplied to the generated expression
        rather than this function.
        The `backend` keyword argument should also be supplied to the generated
        expression. If numpy arrays are supplied, if possible they will be
        converted to and back from the correct backend array type.
        Without an explicitly enabled shape contract the generated expression will
        work with any arrays which have the same rank (number of dimensions) as
        the original shapes, however, if the actual sizes are different, the
        expression may no longer be optimal. With a contract, the locked labels
        and the broadcast compatibility of the variable labels are validated
        first.
        Constant operations will be computed upon the first call with a particular
        backend, then subsequently reused.

    Examples:
    Basic usage:

    ```python
    expr = contract_expression("ab,bc->ac", (3, 4), (4, 5))
    a, b = np.random.rand(3, 4), np.random.rand(4, 5)
    c = expr(a, b)
    np.allclose(c, a @ b)
    #> True
    ```

    Supply `a` as a constant:

    ```python
    expr = contract_expression("ab,bc->ac", a, (4, 5), constants=[0])
    expr
    #> <ContractExpression('[ab],bc->ac', constants=[0])>

    c = expr(b)
    np.allclose(c, a @ b)
    #> True
    ```

    A batched matmul with a variable batch dimension, but locked contraction and
    channel dimensions:

    ```python
    expr = contract_expression(
        "...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes=["..."]
    )

    expr(np.ones((5, 2, 3)), np.ones((1, 3, 4))).shape
    #> (5, 2, 4)

    # the contraction and channel labels stay locked - even though the backend
    # could multiply the new sizes, this raises before any work is done:
    expr(np.ones((5, 2, 6)), np.ones((1, 6, 4)))
    #> ValueError: Operand 0 has size 6 for static label 'j' ...
    ```

    Without ``dynamic_axes`` or ``shape_contract`` plain calls remain permissive.
    """
    if not optimize:
        raise ValueError("Can only generate expressions for optimized contractions.")

    for arg in ("out", "backend"):
        if kwargs.get(arg, None) is not None:
            raise ValueError(
                f"'{arg}' should only be specified when calling a `ContractExpression`, not when building it."
            )

    if not isinstance(subscripts, str):
        interleaved = (subscripts,) + shapes
        symbol_map: dict[Any, str] | None = parser.build_symbol_map(_interleaved_subscript_lists(interleaved))
        subscripts, shapes = parser.convert_interleaved_input(interleaved)
    else:
        symbol_map = None

    enable_contract, dynamic_axes = _resolve_shape_contract(shape_contract, dynamic_axes)

    kwargs["_gen_expression"] = True

    # build dict of constant indices mapped to arrays
    constants = constants or ()
    constants_dict = {i: shapes[i] for i in constants}
    kwargs["_constants_dict"] = constants_dict

    # Compile the shape contract. Every expression carries the implicit contract
    # (used to guard writes into a caller supplied ``out``); an explicitly
    # requested contract is enforced on every call as well, while
    # ``shape_contract=False`` disables it entirely.
    if enable_contract is False:
        kwargs["_shape_contract"] = None
    else:
        kwargs["_shape_contract"] = ShapeContract.from_expression(
            subscripts,
            shapes,
            dynamic_axes=dynamic_axes,
            strict=bool(enable_contract),
            symbol_map=symbol_map,
        )

    # apart from constant arguments, make dummy arrays
    dummy_arrays = [s if i in constants else shape_only(s) for i, s in enumerate(shapes)]  # type: ignore

    return contract(
        subscripts, *dummy_arrays, use_blas=use_blas, optimize=optimize, memory_limit=memory_limit, **kwargs
    )
