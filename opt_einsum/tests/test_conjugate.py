"""Tests for per-operand conjugation declared at the ``contract`` /
``ContractExpression`` entry points.

The key properties exercised here are:

- conjugated contractions match an explicit ``np.conj`` based reference;
- the supplied operands are never modified;
- the NumPy fused path never materializes a full operand-sized conjugate;
- conjugated and plain contractions coexist inside one shared-intermediate
  block and remain individually correct and repeatable;
- backends without a fused kernel (here the object backend) still return the
  same numerical value in their own array type;
- legacy calls, ``out`` and interleaved (explicit multi-operand) inputs keep
  their existing contracts.
"""

from typing import Any

import pytest

from opt_einsum import contract, contract_expression, shared_intermediates
from opt_einsum.backends import numpy_conj
from opt_einsum.sharing import count_cached_ops

np = pytest.importorskip("numpy")
Array = Any


def test_vector_reduction_conjugate_left_only() -> None:
    a = np.array([1 + 1j, 2 + 0j])
    b = np.array([1 + 0j, 1j])

    result = contract("i,i->", a, b, conjugate=(True, False))
    assert result == 1 + 1j

    # inputs are untouched
    assert np.array_equal(a, np.array([1 + 1j, 2 + 0j]))
    assert np.array_equal(b, np.array([1 + 0j, 1j]))


def test_vector_reduction_matches_reference_for_every_pattern() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(7) + 1j * rng.standard_normal(7)
    b = rng.standard_normal(7) + 1j * rng.standard_normal(7)

    patterns: list[tuple[bool, bool]] = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    for cj in patterns:
        expected = np.dot(a.conj() if cj[0] else a, b.conj() if cj[1] else b)
        # optimized (BLAS) path
        assert np.allclose(contract("i,i->", a, b, conjugate=cj), expected)
        # unoptimized pure-einsum style path
        assert np.allclose(contract("i,i->", a, b, conjugate=cj, optimize=False), expected)
        # single boolean applies to every operand
        scalar = contract("i,i->", a, b, conjugate=any(cj) and all(cj))
        if all(cj):
            assert np.allclose(scalar, expected)

    # default (no conjugate kwarg) unchanged
    assert np.allclose(contract("i,i->", a, b), np.dot(a, b))


def test_conjugate_validation() -> None:
    a = np.array([1 + 1j, 2 + 0j])
    b = np.array([1 + 0j, 1j])

    with pytest.raises(ValueError):
        contract("i,i->", a, b, conjugate=(True,))
    with pytest.raises(TypeError):
        contract("i,i->", a, b, conjugate=(1, 0))  # type: ignore[arg-type]

    expr = contract_expression("i,i->", a.shape, b.shape)
    with pytest.raises(ValueError):
        expr(a, b, conjugate=(True,))


@pytest.mark.parametrize("optimize", ["greedy", "optimal", True, False])
@pytest.mark.parametrize("use_blas", [True, False])
def test_conjugate_matches_explicit_conj_reference(optimize: object, use_blas: bool) -> None:
    rng = np.random.default_rng(1)
    eq = "ij,jk,kl->il"
    arrays = [
        rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3)),
        rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)),
        rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5)),
    ]
    originals = [a.copy() for a in arrays]

    for conj_index in range(3):
        flags = tuple(i == conj_index for i in range(3))
        explicit = [x.conj() if flag else x for x, flag in zip(arrays, flags)]
        expected = contract(eq, *explicit, optimize=False, use_blas=False)
        got = contract(
            eq,
            *arrays,
            conjugate=flags,
            optimize=optimize,  # type: ignore[call-overload]
            use_blas=use_blas,
        )
        assert np.allclose(got, expected)

        # all operands unchanged afterwards
    for original, current in zip(originals, arrays):
        assert np.array_equal(original, current)


def test_gemm_and_outer_product_conjugation() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    b = rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))

    assert np.allclose(
        contract("ij,jk->ik", a, b, conjugate=(True, False)),
        a.conj() @ b,
    )
    assert np.allclose(
        contract("ij,jk->ik", a, b, conjugate=(False, True)),
        a @ b.conj(),
    )
    v = rng.standard_normal(3) + 1j
    w = rng.standard_normal(4) + 1j
    assert np.allclose(
        contract("i,j->ij", v, w, conjugate=(True, False)),
        np.einsum("i,j->ij", v.conj(), w),
    )


def test_fused_path_does_not_materialize_full_operand(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    # non-contiguous complex constant with a contracted dimension much larger
    # than the fused kernel's block size
    x = (rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000))[::2]
    y = rng.standard_normal(10_000) + 1j * rng.standard_normal(10_000)
    assert not x.flags["C_CONTIGUOUS"]

    largest = [0]
    true_conjugate = numpy_conj.conjugate

    def tracking_conjugate(array: Array) -> Array:
        if np.iscomplexobj(array):
            largest[0] = max(largest[0], array.size)
        return true_conjugate(array)

    monkeypatch.setattr(numpy_conj, "conjugate", tracking_conjugate)
    result = contract("i,i->", x, y, conjugate=(True, False))
    assert np.allclose(result, np.dot(x.conj(), y))

    # the biggest conjugate ever formed is block-sized, far below operand size
    assert 0 < largest[0] <= numpy_conj._BLOCK_SIZE
    assert largest[0] < x.size

    # a matrix contraction keeps the same guarantee: no conjugate is ever as
    # large as the full left operand
    matrix = (rng.standard_normal(600 * 2_000) + 1j * rng.standard_normal(600 * 2_000))
    matrix = matrix.reshape(600, 2_000)[:, ::2]
    right = rng.standard_normal((1_000, 100)) + 1j * rng.standard_normal((1_000, 100))
    largest[0] = 0
    monkeypatch.setattr(numpy_conj, "conjugate", tracking_conjugate)
    matrix_result = contract("ij,jk->ik", matrix, right, conjugate=(True, False))
    assert np.allclose(matrix_result, matrix.conj() @ right)
    assert largest[0] < matrix.size


def test_two_expressions_share_one_block_with_distinct_results() -> None:
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(2_000) + 1j * rng.standard_normal(2_000))[::2]
    y = rng.standard_normal(1_000) + 1j * rng.standard_normal(1_000)
    assert not x.flags["C_CONTIGUOUS"]

    conj_expr = contract_expression("i,i->", x.shape, y.shape, conjugate=(True, False))
    plain_expr = contract_expression("i,i->", x.shape, y.shape, conjugate=(False, False))

    expected_conj = np.dot(x.conj(), y)
    expected_plain = np.dot(x, y)
    assert not np.allclose(expected_conj, expected_plain)

    with shared_intermediates() as cache:
        conj_1 = conj_expr(x, y)
        plain_1 = plain_expr(x, y)
        conj_2 = conj_expr(x, y)
        plain_2 = plain_expr(x, y)

    assert np.allclose(conj_1, expected_conj)
    assert np.allclose(plain_1, expected_plain)
    assert np.allclose(conj_2, expected_conj)
    assert np.allclose(plain_2, expected_plain)
    assert not np.allclose(conj_1, plain_1)

    # the conjugated and plain kernels are cached under separate op names
    op_counts = count_cached_ops(cache)
    assert op_counts.get("conjugate_einsum", 0) + op_counts.get("conjugate_tensordot", 0) >= 1
    assert op_counts.get("tensordot", 0) >= 1

    # the shared constant is still the original, non-contiguous array
    assert not x.flags["C_CONTIGUOUS"]

    # repeatable outside the shared block as well
    for _ in range(3):
        assert np.allclose(conj_expr(x, y), expected_conj)
        assert np.allclose(plain_expr(x, y), expected_plain)


def test_per_call_conjugate_override() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    y = rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))

    expr = contract_expression("ij,jk->ik", x.shape, y.shape)

    assert np.allclose(expr(x, y), x @ y)
    assert np.allclose(expr(x, y, conjugate=(True, False)), x.conj() @ y)
    assert np.allclose(expr(x, y, conjugate=(False, True)), x @ y.conj())
    # the expression itself keeps its default (plain) behavior
    assert np.allclose(expr(x, y), x @ y)

    with shared_intermediates():
        left = expr(x, y, conjugate=(True, False))
        right = expr(x, y, conjugate=(False, True))
        again = expr(x, y, conjugate=(True, False))
    assert np.allclose(left, again)
    assert not np.allclose(left, right)


def test_conjugate_with_constants() -> None:
    rng = np.random.default_rng(7)
    eq = "ij,jk,kl->il"
    a = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    c = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))

    expected = contract(eq, a.conj(), b, c, optimize=False, use_blas=False)

    expr = contract_expression(
        eq, a, b.shape, c.shape, constants=[0], conjugate=(True, False, False)
    )
    assert "conjugate=" in repr(expr)
    for _ in range(2):
        assert np.allclose(expr(b, c), expected)

    # overriding a constant-bearing expression still works
    plain = contract_expression(eq, a, b.shape, c.shape, constants=[0])
    assert np.allclose(plain(b, c, conjugate=(True, False, False)), expected)

    a_before = a.copy()
    expr(b, c)
    assert np.array_equal(a, a_before)


def test_per_call_override_with_constants() -> None:
    rng = np.random.default_rng(71)
    eq = "ij,jk,kl->il"
    a = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    b = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    c = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))

    expr = contract_expression(eq, a.shape, b, c.shape, constants=[1])

    # conjugate the constant position via a per-call override
    expected_const = contract(eq, a, b.conj(), c, optimize=False, use_blas=False)
    assert np.allclose(expr(a, c, conjugate=(False, True, False)), expected_const)

    # conjugate a non-constant position instead
    expected_var = contract(eq, a.conj(), b, c, optimize=False, use_blas=False)
    assert np.allclose(expr(a, c, conjugate=(True, False, False)), expected_var)

    # the cached default (plain) contraction is unaffected by the overrides
    expected_plain = contract(eq, a, b, c, optimize=False, use_blas=False)
    assert np.allclose(expr(a, c), expected_plain)
    for _ in range(2):
        assert np.allclose(expr(a, c, conjugate=(False, True, False)), expected_const)


def test_object_backend_preserves_value_and_type() -> None:
    a = np.array([1 + 1j, 2 + 0j], dtype=object)
    b = np.array([1 + 0j, 1j], dtype=object)

    result = contract("i,i->", a, b, conjugate=(True, False), backend="object")
    assert result.dtype == object
    assert result == 1 + 1j

    expr = contract_expression("i,i->", a.shape, b.shape, conjugate=(True, False))
    result_expr = expr(a, b, backend="object")
    assert result_expr.dtype == object
    assert result_expr == 1 + 1j

    # operands untouched
    assert a[0] == 1 + 1j and b[1] == 1j


def test_legacy_out_and_interleaved_contracts() -> None:
    a = np.array([1 + 1j, 2 + 0j])
    b = np.array([1 + 0j, 1j])

    # out keyword honored on every entry path
    for optimized in (True, False):
        out = np.zeros((), dtype=np.complex128)
        result = contract("i,i->", a, b, conjugate=(True, False), out=out, optimize=optimized)
        assert result is out
        assert out == 1 + 1j

    expr = contract_expression("i,i->", a.shape, b.shape, conjugate=(True, False))
    out_expr = np.zeros((), dtype=np.complex128)
    assert expr(a, b, out=out_expr) is out_expr
    assert out_expr == 1 + 1j

    # interleaved (explicit index) multi-operand input style
    interleaved = contract(a, [0], b, [0], [], conjugate=(True, False))
    assert interleaved == 1 + 1j
    interleaved_expr = contract_expression(
        a.shape, [0], b.shape, [0], [], conjugate=(True, False)
    )
    assert interleaved_expr(a, b) == 1 + 1j

    # legacy calls without conjugate keep working exactly as before
    assert np.allclose(contract("i,i->", a, b), np.einsum("i,i->", a, b))


def test_native_kernel_capability_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    real_einsum = np.einsum
    received: list[tuple[bool, ...] | None] = []

    class FakeEinsum:
        def __call__(
            self,
            equation: str,
            *arrays: Array,
            conjugate: tuple[bool, ...] | None = None,
            out: object = None,
            **kwargs: object,
        ):
            received.append(conjugate)
            if conjugate is None:
                return real_einsum(equation, *arrays, out=out, **kwargs)
            converted = [a.conj() if flag else a for a, flag in zip(arrays, conjugate)]
            return real_einsum(equation, *converted, out=out, **kwargs)

    monkeypatch.setattr(numpy_conj.np, "einsum", FakeEinsum())
    monkeypatch.setattr(numpy_conj, "_EINSUM_SUPPORTS_CONJUGATE", True)

    a = np.array([1 + 1j, 2 + 0j])
    b = np.array([1 + 0j, 1j])
    result = numpy_conj.conjugate_einsum.__wrapped__("i,i->", a, b, conjugate=(True, False))
    assert result == 1 + 1j
    assert received[-1] == (True, False)


def test_float_operands_with_conjugate_flag() -> None:
    rng = np.random.default_rng(8)
    a = rng.standard_normal((4, 5))
    b = rng.standard_normal((5, 6))
    # conjugation of real data is a no-op numerically
    assert np.allclose(
        contract("ij,jk->ik", a, b, conjugate=(True, False)),
        a @ b,
    )


def test_non_contiguous_multidimensional_constant() -> None:
    rng = np.random.default_rng(9)
    a = (rng.standard_normal(48) + 1j * rng.standard_normal(48)).reshape(6, 8)[::2]
    b = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    assert not a.flags["C_CONTIGUOUS"]

    got = contract("ij,jk->ik", a, b, conjugate=(True, False))
    assert np.allclose(got, a.conj() @ b)

    expr = contract_expression("ij,jk->ik", a.shape, b.shape, conjugate=(True, False))
    assert np.allclose(expr(a, b), a.conj() @ b)
    assert np.allclose(expr(a, b), a.conj() @ b)
