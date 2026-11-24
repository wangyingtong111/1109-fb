"""Conjugate-aware contraction kernels for NumPy.

Older NumPy releases do not provide an ``einsum`` (or ``tensordot``) kernel
that can conjugate operands as part of the contraction, so the only portable
way to express e.g. a complex inner product ``sum(conj(x) * y)`` was to
materialize a full conjugate copy of ``x`` first.  The routines in this module
fuse the conjugation into the contraction instead:

- when the installed NumPy natively supports the ``conjugate`` keyword, it is
  used directly, keeping everything inside the compiled kernel;
- otherwise the contraction is evaluated in blocks over the summed indices,
  so that at most a block-sized (never operand-sized) conjugate is formed.

Neither path mutates or replaces the supplied arrays.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from opt_einsum.parser import get_symbol
from opt_einsum.sharing import conjugate_einsum_cache_wrap, conjugate_tensordot_cache_wrap

__all__ = [
    "einsum_supports_conjugate",
    "tensordot_supports_conjugate",
    "conjugate",
    "conjugate_einsum",
    "conjugate_tensordot",
]

# Size of the (summed-index) blocks used by the pure-python fallback.  Only
# block-sized temporary conjugate views are ever materialized, independently of
# the full operand size.
_BLOCK_SIZE = 256

# capability of the installed NumPy is probed at most once
_EINSUM_SUPPORTS_CONJUGATE: bool | None = None
_TENSORDOT_SUPPORTS_CONJUGATE: bool | None = None


def _normalize_conjugate(conjugate_arg: bool | Sequence[bool] | None, n: int) -> tuple[bool, ...]:
    if conjugate_arg is None:
        return (False,) * n
    if isinstance(conjugate_arg, bool):
        flags = (conjugate_arg,) * n
    else:
        flags = tuple(conjugate_arg)
    if len(flags) != n:
        raise ValueError(
            f"`conjugate` must specify one flag per operand (expected {n}, got {len(flags)})."
        )
    if any(not isinstance(flag, bool) for flag in flags):
        raise TypeError("`conjugate` flags must be booleans.")
    return flags


def conjugate(x: Any) -> Any:
    """Conjugate ``x`` without assuming a particular array/scalar type."""
    conjugate_method = getattr(x, "conjugate", None)
    if conjugate_method is not None:
        return conjugate_method()
    return x.conj()


def einsum_supports_conjugate() -> bool:
    """Whether ``numpy.einsum`` natively accepts a ``conjugate`` argument."""
    global _EINSUM_SUPPORTS_CONJUGATE
    if _EINSUM_SUPPORTS_CONJUGATE is None:
        try:
            # newer NumPy releases expose conjugation as a native einsum option
            np.einsum("i->", np.zeros(1, dtype=np.complex128), conjugate=(False,))  # type: ignore[call-overload]
        except TypeError:
            _EINSUM_SUPPORTS_CONJUGATE = False
        else:
            _EINSUM_SUPPORTS_CONJUGATE = True
    return _EINSUM_SUPPORTS_CONJUGATE


def tensordot_supports_conjugate() -> bool:
    """Whether ``numpy.tensordot`` natively accepts a ``conjugate`` argument."""
    global _TENSORDOT_SUPPORTS_CONJUGATE
    if _TENSORDOT_SUPPORTS_CONJUGATE is None:
        try:
            np.tensordot(  # type: ignore[call-overload]
                np.zeros(1, dtype=np.complex128),
                np.zeros(1, dtype=np.complex128),
                axes=1,
                conjugate=(False, False),
            )
        except TypeError:
            _TENSORDOT_SUPPORTS_CONJUGATE = False
        else:
            _TENSORDOT_SUPPORTS_CONJUGATE = True
    return _TENSORDOT_SUPPORTS_CONJUGATE


def _try_blas_conjugate(
    equation: str,
    arrays: Sequence[np.ndarray],
    conjugate_flags: tuple[bool, ...],
    out: np.ndarray | None,
) -> np.ndarray | None:
    """Fast path for the standard two-operand contractions (dot / GEMM /
    matrix-vector / outer product) expressible with ``matmul``/``tensordot``.

    Conjugation is applied per block of the contracted axes, so only
    block-sized conjugate temporaries are ever formed. Returns ``None`` when
    the equation needs the general (repeated-index, broadcasting, ...) kernel.
    """
    if len(arrays) != 2:
        return None

    lhs, rhs = equation.split("->")
    left_term, right_term = lhs.split(",")
    x, y = arrays
    conj_x, conj_y = conjugate_flags

    # repeated labels within one operand cannot be expressed by matmul
    if len(set(left_term)) != len(left_term) or len(set(right_term)) != len(right_term):
        return None

    left_axes, right_axes = list(left_term), list(right_term)
    contracted = [label for label in left_axes if label in right_axes]
    if not contracted:
        # outer product: no reduction axis, a single multiply suffices
        xc = conjugate(x) if conj_x else x
        yc = conjugate(y) if conj_y else y
        return np.einsum(equation, xc, yc, out=out)

    # the blocked matmul kernel below assumes a single contracted axis;
    # equations with several (e.g. ``ij,ij->``) use the general blocked kernel
    if len(contracted) != 1:
        return None

    primary_left = left_axes.index(contracted[0])
    primary_right = right_axes.index(contracted[0])
    other_left = [i for i in range(x.ndim) if i != primary_left]
    other_right = [i for i in range(y.ndim) if i != primary_right]
    xm = np.transpose(x, other_left + [primary_left])
    ym = np.transpose(y, [primary_right] + other_right)

    # map each label to its dimension after the axis permutation
    left_kept_labels = [label for label in left_axes if label not in contracted]
    right_kept_labels = [label for label in right_axes if label not in contracted]
    xm_label_sizes = dict(zip(other_left + [primary_left], xm.shape))
    ym_label_sizes = dict(zip([primary_right] + other_right, ym.shape))
    free_shape = tuple(xm_label_sizes[left_axes.index(label)] for label in left_kept_labels)
    free_shape += tuple(ym_label_sizes[right_axes.index(label)] for label in right_kept_labels)
    result_dtype = np.result_type(*arrays)
    result = np.zeros(free_shape, dtype=result_dtype)

    k_total = xm.shape[-1]
    for start in range(0, k_total, _BLOCK_SIZE):
        stop = min(start + _BLOCK_SIZE, k_total)
        xb = xm[..., start:stop]
        yb = ym[start:stop, ...]
        if conj_x:
            xb = conjugate(xb)
        if conj_y:
            yb = conjugate(yb)
        result += xb @ yb

    target_labels = list(rhs)
    output_labels = left_kept_labels + right_kept_labels
    if output_labels != target_labels:
        perm = tuple(output_labels.index(label) for label in target_labels)
        result = np.ascontiguousarray(np.transpose(result, perm))

    if out is not None:
        out[...] = result
        return out
    return result


def _blocked_einsum(
    equation: str,
    arrays: Sequence[np.ndarray],
    conjugate_flags: tuple[bool, ...],
    out: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Fallback conjugate einsum which only ever forms block-sized conjugates.

    The summed (contracted) indices are traversed in blocks: each per-block
    contraction is delegated to regular ``numpy.einsum`` with block slices, so
    a conjugated operand only requires a conjugate of the current block rather
    than of the whole operand.
    """
    lhs, rhs = equation.split("->")
    inputs = lhs.split(",")

    sizes: dict[str, int] = {}
    for term, array in zip(inputs, arrays):
        for label, dim in zip(term, array.shape):
            dim = int(dim)
            if label in sizes:
                # broadcasting: size-1 dims may heal against larger ones
                if sizes[label] == 1:
                    sizes[label] = dim
            else:
                sizes[label] = dim

    out_shape = tuple(sizes[label] for label in rhs)
    dtype = kwargs.pop("dtype", None)
    result_dtype = np.result_type(*arrays) if dtype is None else np.dtype(dtype)
    if out is None:
        order = kwargs.pop("order", "K")
        out = np.zeros(out_shape, dtype=result_dtype, order=order if order in ("C", "F") else "C")
    else:
        kwargs.pop("order", None)

    summed = [label for label in sizes if label not in rhs]

    if not summed:
        # nothing to reduce over: conjugate each requested operand once (these
        # arrays flow directly into the output, so no extra full-size reduction
        # temporary exists anyway)
        blocked_arrays = [conjugate(array) if flag else array for array, flag in zip(arrays, conjugate_flags)]
        return np.einsum(equation, *blocked_arrays, out=out, **kwargs)

    block_starts = {label: list(range(0, sizes[label], _BLOCK_SIZE)) for label in summed}
    label_order = list(block_starts)
    for starts in np.ndindex(*[len(block_starts[label]) for label in label_order]):
        slices = {
            label: (
                block_starts[label][idx],
                min(block_starts[label][idx] + _BLOCK_SIZE, sizes[label]),
            )
            for label, idx in zip(label_order, starts)
        }
        blocked_arrays = []
        for term, array, flag in zip(inputs, arrays, conjugate_flags):
            selector = tuple(slice(*slices[label]) if label in slices else slice(None) for label in term)
            blocked_arrays.append(conjugate(array[selector]) if flag else array[selector])
        out += np.einsum(equation, *blocked_arrays, **kwargs)

    return out


@conjugate_einsum_cache_wrap
def conjugate_einsum(
    equation: str,
    *arrays: np.ndarray,
    conjugate: bool | Sequence[bool] | None = None,
    out: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Evaluate ``np.einsum(equation, *arrays, **kwargs)`` with per-operand
    conjugation, without materializing a full conjugate copy of any operand.

    Uses NumPy's native ``conjugate`` support when available, and a blocked
    fallback otherwise.
    """
    flags = _normalize_conjugate(conjugate, len(arrays))

    if not any(flags):
        return np.einsum(equation, *arrays, out=out, **kwargs)

    if einsum_supports_conjugate():
        return np.einsum(  # type: ignore[call-overload]
            equation, *arrays, conjugate=flags, out=out, **kwargs
        )

    # the common dot/GEMM family gets a blocked BLAS fast path; only block-sized
    # conjugate temporaries are formed there
    if not kwargs:
        blas_result = _try_blas_conjugate(equation, arrays, flags, out)
        if blas_result is not None:
            return blas_result

    return _blocked_einsum(equation, arrays, flags, out=out, **kwargs)


@conjugate_tensordot_cache_wrap
def conjugate_tensordot(
    x: np.ndarray,
    y: np.ndarray,
    axes: Any,
    *,
    conjugate: tuple[bool, bool] = (False, False),
    **kwargs: Any,
) -> np.ndarray:
    """Conjugate-aware counterpart of ``numpy.tensordot``.

    Falls back to :func:`conjugate_einsum` (which itself never builds a full
    conjugate copy) when the installed ``numpy.tensordot`` lacks native
    ``conjugate`` support.
    """
    flags = _normalize_conjugate(conjugate, 2)

    if not any(flags):
        return np.tensordot(x, y, axes=axes, **kwargs)

    if tensordot_supports_conjugate():
        return np.tensordot(  # type: ignore[call-overload]
            x, y, axes=axes, conjugate=flags, **kwargs
        )

    # translate the tensordot call into an explicit einsum equation so the
    # blocked conjugate kernel can handle it without full-size temporaries
    x_ndim, y_ndim = x.ndim, y.ndim
    if isinstance(axes, (tuple, list)) and len(axes) == 2 and isinstance(axes[0], (tuple, list)):
        x_axes, y_axes = tuple(axes[0]), tuple(axes[1])
    else:
        if isinstance(axes, (tuple, list)):
            n = len(axes)
            x_axes = tuple(range(x_ndim - n, x_ndim))
            y_axes = tuple(axes)
        else:
            n = int(axes)
            x_axes = tuple(range(x_ndim - n, x_ndim))
            y_axes = tuple(range(n))

    labels = [get_symbol(i) for i in range(x_ndim + y_ndim)]
    x_term = list(labels[:x_ndim])
    y_term = list(labels[x_ndim : x_ndim + y_ndim])
    contracted = set()
    for x_axis, y_axis in zip(x_axes, y_axes):
        y_term[y_axis] = x_term[x_axis]
        contracted.add(x_term[x_axis])
    # tensordot orders the result as x's free axes followed by y's free axes
    output = [label for label in x_term if label not in contracted]
    output += [label for label in y_term if label not in contracted]
    equation = "{},{}->{}".format("".join(x_term), "".join(y_term), "".join(output))

    return conjugate_einsum(equation, x, y, conjugate=flags, **kwargs)
