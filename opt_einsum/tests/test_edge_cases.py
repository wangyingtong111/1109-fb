"""
Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
"""

from typing import Any

import pytest

from opt_einsum import contract, contract_expression, contract_path
from opt_einsum.typing import PathType

# NumPy is required for the majority of this file
np = pytest.importorskip("numpy")


def test_contract_expression_checks() -> None:
    # check optimize needed
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), optimize=False)

    # check sizes are still checked
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), (42, 42))

    # check if out given
    out = np.empty((2, 4))
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), out=out)

    # check still get errors when wrong ranks supplied to expression
    expr = contract_expression("ab,bc->ac", (2, 3), (3, 4))

    # too few arguments
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3))
    assert "`ContractExpression` takes exactly 2" in str(err.value)

    # too many arguments
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3))
    assert "`ContractExpression` takes exactly 2" in str(err.value)

    # wrong shapes
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3, 4), np.random.rand(3, 4))
    assert "Internal error while evaluating `ContractExpression`" in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 4), np.random.rand(3, 4, 5))
    assert "Internal error while evaluating `ContractExpression`" in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), out=np.random.rand(2, 4, 6))
    assert "Internal error while evaluating `ContractExpression`" in str(err.value)

    # should only be able to specify out
    with pytest.raises(TypeError) as err_type:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), order="F")  # type: ignore
    assert "got an unexpected keyword" in str(err_type.value)


def test_broadcasting_contraction() -> None:
    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(10)

    ein_scalar = contract("ijk,kl,jl", a, b, c, optimize=False)
    opt_scalar = contract("ijk,kl,jl", a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)

    result = ein_scalar * d

    ein = contract("ijk,kl,jl,i->i", a, b, c, d, optimize=False)
    opt = contract("ijk,kl,jl,i->i", a, b, c, d, optimize=True)

    assert np.allclose(ein, result)
    assert np.allclose(opt, result)


def test_broadcasting_contraction2() -> None:
    a = np.random.rand(1, 1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)

    ein_scalar = contract("abjk,kl,jl", a, b, c, optimize=False)
    opt_scalar = contract("abjk,kl,jl", a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)

    result = ein_scalar * d

    ein = contract("abjk,kl,jl,ab->ab", a, b, c, d, optimize=False)
    opt = contract("abjk,kl,jl,ab->ab", a, b, c, d, optimize=True)

    assert np.allclose(ein, result)
    assert np.allclose(opt, result)


def test_broadcasting_contraction3() -> None:
    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 1, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)

    ein = contract("ajk,kbl,jl,ab->ab", a, b, c, d, optimize=False)
    opt = contract("ajk,kbl,jl,ab->ab", a, b, c, d, optimize=True)

    assert np.allclose(ein, opt)


def test_broadcasting_contraction4() -> None:
    a = np.arange(64).reshape(2, 4, 8)
    ein = contract("obk,ijk->ioj", a, a, optimize=False)
    opt = contract("obk,ijk->ioj", a, a, optimize=True)

    assert np.allclose(ein, opt)


def test_zero_length_broadcasting_contraction() -> None:
    operands = (np.ones(1), np.empty(0), np.ones(1))
    expected = np.einsum("i,i,i->", *operands)
    actual = contract("i,i,i->", *operands)
    assert actual == expected


def test_can_blas_on_healed_broadcast_dimensions() -> None:
    expr = contract_expression("ab,bc,bd->acd", (5, 4), (1, 5), (4, 20))
    # first contraction involves broadcasting
    assert expr.contraction_list[0][2] == "bc,ab->bca"
    assert expr.contraction_list[0][-1] is False
    # but then is healed GEMM is usable
    assert expr.contraction_list[1][2] == "bca,bd->acd"
    assert expr.contraction_list[1][-1] == "GEMM"


def test_pathinfo_for_empty_contraction() -> None:
    eq = "->"
    arrays = (1.0,)
    path: PathType = []
    _, info = contract_path(eq, *arrays, optimize=path)
    # some info is built lazily, so check repr
    assert repr(info)
    assert info.largest_intermediate == 1


class _CannotMultiply:
    """Array element which fails loudly if it ever takes part in a product."""

    def __mul__(self, other: Any) -> Any:
        raise AssertionError("An explicitly empty output should never multiply operands.")

    __rmul__ = __mul__

    def __add__(self, other: Any) -> Any:
        raise AssertionError("An explicitly empty output should never add operands.")

    __radd__ = __add__


def test_empty_output_skips_intermediate_multiplication() -> None:
    # 'ab,bc' are the (large) weights, 'd' is a zero-length axis in the output.
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    d = np.empty(0)

    expected = np.einsum("ab,bc,d->acd", a, b, d)

    # both an automatically found path and an explicit one which would
    # otherwise multiply 'ab' and 'bc' first must short circuit.
    for optimize in (True, [(0, 1), (0, 1)], [(0, 2), (0, 1)]):
        actual = contract("ab,bc,d->acd", a, b, d, optimize=optimize)
        assert actual.shape == (2, 4, 0)
        assert actual.dtype == expected.dtype
        assert np.array_equal(actual, expected)

    # prove that no product between the weights is performed at all
    poison = np.empty((2, 3), dtype=object)
    poison[...] = [[_CannotMultiply() for _ in range(3)] for _ in range(2)]
    other = np.empty((3, 4), dtype=object)
    other[...] = [[_CannotMultiply() for _ in range(4)] for _ in range(3)]
    empty = np.empty(0, dtype=object)

    result = contract("ab,bc,d->acd", poison, other, empty, optimize=[(0, 1), (0, 1)], backend="object")
    assert result.shape == (2, 4, 0)
    assert result.dtype == object


def test_empty_output_expression_is_single_step() -> None:
    expr = contract_expression("ab,bc,d->acd", (2, 3), (3, 4), (0,))
    # a single einsum taking all the original operands, no intermediate GEMM
    assert len(expr.contraction_list) == 1
    _, idx_removed, einsum_str, _, blas_flag = expr.contraction_list[0]
    assert einsum_str.endswith("->acd")
    assert idx_removed == frozenset("b")
    assert blas_flag is False

    a, b, d = np.random.rand(2, 3), np.random.rand(3, 4), np.empty(0)
    expected = np.einsum("ab,bc,d->acd", a, b, d)
    assert np.array_equal(expr(a, b, d), expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.complex128, np.int32])
def test_empty_output_preserves_dtype_and_out(dtype: Any) -> None:
    a = np.random.rand(2, 3).astype(dtype)
    b = np.random.rand(3, 4).astype(dtype)
    d = np.empty(0, dtype=dtype)

    expected = np.einsum("ab,bc,d->acd", a, b, d)
    actual = contract("ab,bc,d->acd", a, b, d)
    assert actual.dtype == expected.dtype
    assert actual.shape == (2, 4, 0)

    out = np.empty((2, 4, 0), dtype=dtype)
    returned = contract("ab,bc,d->acd", a, b, d, out=out)
    assert returned is out
    assert out.dtype == dtype
    assert out.shape == (2, 4, 0)


def test_empty_output_explicit_backend() -> None:
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    d = np.empty(0)

    # supply the (only-supported-here) numpy backend explicitly
    actual = contract("ab,bc,d->acd", a, b, d, backend="numpy")
    assert np.array_equal(actual, np.einsum("ab,bc,d->acd", a, b, d))
    assert actual.shape == (2, 4, 0)
    assert actual.dtype == np.einsum("ab,bc,d->acd", a, b, d).dtype


def test_empty_reduction_axis_keeps_non_empty_output() -> None:
    # 'b' is summed over and has length 0, but every output index is non-empty:
    # the result is a non-empty tensor of zeros which must use the normal path.
    a = np.random.rand(2, 0)
    b = np.random.rand(0, 4)

    expected = np.einsum("ab,bc->ac", a, b)
    actual = contract("ab,bc->ac", a, b)
    assert actual.shape == (2, 4)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)

    path, info = contract_path("ab,bc->ac", a, b)
    # no short circuit: a normal pairwise GEMM plan is still produced
    assert list(path) == [(0, 1)]
    assert info.contraction_list[0][2] == "bc,ab->ac"

    out = np.empty((2, 4), dtype=a.dtype)
    returned = contract("ab,bc->ac", a, b, out=out)
    assert returned is out
    assert np.array_equal(out, expected)


def test_empty_output_does_not_change_normal_planning() -> None:
    # a single non-zero length axis is planned exactly as before
    a, b, c = (np.random.rand(*s) for s in [(2, 3), (3, 4), (4, 5)])
    path, info = contract_path("ab,bc,cd->ad", a, b, c)
    assert list(path) == [(0, 1), (0, 1)]
    assert info.contraction_list[0][2] == "bc,ab->ca"
    assert np.allclose(contract("ab,bc,cd->ad", a, b, c), np.einsum("ab,bc,cd->ad", a, b, c))

    # a two operand empty output is already a single contraction, so it keeps
    # its usual (BLAS-able) plan rather than being collapsed.
    x = np.random.rand(0, 3)
    y = np.random.rand(3, 4)
    path, info = contract_path("ab,bc->ac", x, y)
    assert list(path) == [(0, 1)]
    assert info.contraction_list[0][2] == "bc,ab->ac"
    assert np.array_equal(contract("ab,bc->ac", x, y), np.einsum("ab,bc->ac", x, y))


@pytest.mark.parametrize(
    "expression, operands",
    [
        [",,->", (5, 5.0, 2.0j)],
        ["ab,->", ([[5, 5], [2.0, 1]], 2.0j)],
        ["ab,bc->ac", ([[5, 5], [2.0, 1]], [[2.0, 1], [3.0, 4]])],
        ["ab,->", ([[5, 5], [2.0, 1]], True)],
    ],
)
def test_contract_with_assumed_shapes(expression: str, operands: tuple[Any]) -> None:
    """Test that we can contract with assumed shapes, and that the output is correct. This is required as we need to infer intermediate shape sizes."""

    benchmark = np.einsum(expression, *operands)
    result = contract(expression, *operands, optimize=True)
    assert np.allclose(benchmark, result)
