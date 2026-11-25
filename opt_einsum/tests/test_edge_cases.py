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
    # no contraction steps means no actual intermediates
    assert info.intermediate_shapes == ()
    assert info.actual_size_list == ()


def test_pathinfo_actual_sizes_resolve_broadcasting() -> None:
    # the first contraction produces ``ac`` with the size-1 ``a`` axis not yet
    # broadcast against the ``ad`` operand: it really holds 3 elements, not 21
    _, info = contract_path(
        "ab,bc,ad->acd",
        (1, 2),
        (2, 3),
        (7, 4),
        shapes=True,
        optimize=[(0, 1), (0, 1)],
    )

    # step 1: (1, 3) -> 3 elements; step 2: (7, 3, 4) -> 84 elements
    assert info.actual_size_list == (3, 84)
    assert tuple(info.intermediate_shapes[0]) == (3, 1)
    assert tuple(info.intermediate_shapes[1]) == (7, 3, 4)

    # the broadcast-inflated cost model is unchanged
    assert tuple(info.size_list) == (21, 84)
    assert int(info.largest_intermediate) == 84

    # report string matches the concrete shapes
    report = info.actual_sizes_report()
    assert "elements=3" in report and "elements=84" in report

    # and the real contraction produces the advertised element count
    operands = np.ones((1, 2)), np.ones((2, 3)), np.ones((7, 4))
    result = contract("ab,bc,ad->acd", *operands, optimize=[(0, 1), (0, 1)])
    assert result.shape == (7, 3, 4)
    assert result.size == info.actual_size_list[-1] == 84


def test_pathinfo_actual_sizes_are_order_independent() -> None:
    # swapping the operand order must leave the actual per-step element counts
    # consistent with the concrete output shape
    _, info = contract_path(
        "ad,ab,bc->acd",
        (7, 4),
        (1, 2),
        (2, 3),
        shapes=True,
        optimize=[(1, 2), (0, 1)],
    )
    assert info.actual_size_list == (3, 84)

    operands = np.ones((7, 4)), np.ones((1, 2)), np.ones((2, 3))
    result = contract("ad,ab,bc->acd", *operands, optimize=[(1, 2), (0, 1)])
    assert result.shape == (7, 3, 4) and result.size == 84


def test_pathinfo_actual_sizes_see_late_zero_length_axis() -> None:
    # ``a`` has size 1 in the operands contracted first and only becomes
    # zero-length in the third operand, which is met at the second step
    _, info = contract_path(
        "ab,bc,ad->acd",
        (1, 2),
        (2, 3),
        (0, 4),
        shapes=True,
        optimize=[(0, 1), (0, 1)],
    )

    # first intermediate still physically contains 3 elements; final one is empty
    assert info.actual_size_list == (3, 0)
    assert tuple(info.intermediate_shapes[0]) == (3, 1)
    assert tuple(info.intermediate_shapes[1]) == (0, 3, 4)

    # the dict-based cost model cannot distinguish these and stays untouched
    assert tuple(info.size_list) == (0, 0)

    operands = np.ones((1, 2)), np.ones((2, 3)), np.ones((0, 4))
    result = contract("ab,bc,ad->acd", *operands, optimize=[(0, 1), (0, 1)])
    assert result.shape == (0, 3, 4) and result.size == 0


def test_pathinfo_actual_sizes_match_model_without_broadcasting() -> None:
    # with no size-1 / zero-length axes the actual report must agree exactly
    # with the existing cost-model sizes
    _, info = contract_path("ij,jk,kl->il", (2, 2), (2, 5), (5, 2), shapes=True)
    assert info.actual_size_list == tuple(info.size_list)

    # scalar intermediates contain a single element
    _, scalar_info = contract_path("ab,ab->", (2, 3), (2, 3), shapes=True)
    assert tuple(scalar_info.intermediate_shapes[0]) == ()
    assert scalar_info.actual_size_list == (1,)


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
