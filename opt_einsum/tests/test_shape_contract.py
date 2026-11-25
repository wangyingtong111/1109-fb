"""
Tests the optional shape contract of compiled expressions: locked static labels,
variable broadcast batch labels, early rejection and constant cache reuse.
"""

from importlib import import_module
from unittest.mock import patch

import pytest

from opt_einsum import contract_expression
from opt_einsum.contract import ShapeContract
from opt_einsum.parser import find_ellipsis_labels

# NumPy is required for this file
np = pytest.importorskip("numpy")

contract_module = import_module("opt_einsum.contract")


@pytest.fixture
def no_backend_calls():
    """Patch the contraction primitives so we can assert nothing was dispatched."""
    calls = []

    dot = contract_module._tensordot
    einsum = contract_module._einsum

    def record_dot(*args, **kwargs):
        calls.append("tensordot")
        return dot(*args, **kwargs)

    def record_einsum(*args, **kwargs):
        calls.append("einsum")
        return einsum(*args, **kwargs)

    with patch.object(contract_module, "_tensordot", record_dot), patch.object(
        contract_module, "_einsum", record_einsum
    ):
        yield calls


def test_dynamic_batch_runs_and_static_labels_are_locked(no_backend_calls) -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes=["..."])

    result = expr(np.ones((5, 2, 3)), np.ones((1, 3, 4)))
    assert result.shape == (5, 2, 4)
    np.testing.assert_allclose(result, np.full((5, 2, 4), 3.0))
    assert no_backend_calls

    # both contraction dimensions change together - the backend could multiply
    # them, but the contract rejects the call before anything is dispatched
    no_backend_calls.clear()
    with pytest.raises(ValueError) as err:
        expr(np.ones((5, 2, 6)), np.ones((1, 6, 4)))
    message = str(err.value)
    assert "Operand 0" in message and "Operand 1" in message
    assert "'j'" in message
    assert no_backend_calls == []


def test_static_channel_dimension_is_locked() -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes=True)

    with pytest.raises(ValueError, match=r"(?s)Operand 0 .* static label 'i'"):
        expr(np.ones((5, 6, 3)), np.ones((1, 3, 4)))
    with pytest.raises(ValueError, match=r"(?s)Operand 1 .* static label 'k'"):
        expr(np.ones((5, 2, 3)), np.ones((1, 3, 6)))


def test_ellipsis_follows_labels_not_axis_positions() -> None:
    # the batch label is axis 1 of the first operand but axis 0 of the second
    expr = contract_expression("i...j,...jk->i...k", (2, 1, 3), (1, 3, 4), dynamic_axes=["..."])

    result = expr(np.ones((2, 5, 3)), np.ones((1, 3, 4)))
    assert result.shape == (2, 5, 4)

    with pytest.raises(ValueError, match=r"(?s)Dynamic label .* broadcast"):
        expr(np.ones((2, 2, 3)), np.ones((5, 3, 4)))


def test_named_batch_label_can_be_dynamic() -> None:
    expr = contract_expression("bij,bjk->bik", (1, 2, 3), (1, 3, 4), dynamic_axes=["b"])

    assert expr(np.ones((5, 2, 3)), np.ones((1, 3, 4))).shape == (5, 2, 4)

    with pytest.raises(ValueError, match=r"(?s)Dynamic label 'b'"):
        expr(np.ones((2, 2, 3)), np.ones((5, 3, 4)))


def test_inferred_labels_from_shape_contract_shortcut() -> None:
    expr = contract_expression("bij,bjk->bik", (1, 2, 3), (1, 3, 4), shape_contract=True)
    assert expr.shape_contract is not None
    assert "b" in expr.shape_contract.dynamic

    assert expr(np.ones((5, 2, 3)), np.ones((1, 3, 4))).shape == (5, 2, 4)


def test_dynamic_conflict_fails_before_backend_or_out_write(no_backend_calls) -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes=True)
    out = np.full((5, 2, 4), -11.0)

    with pytest.raises(ValueError, match=r"(?s)Dynamic label"):
        expr(np.ones((5, 2, 3)), np.ones((3, 3, 4)), out=out)

    assert no_backend_calls == []
    np.testing.assert_allclose(out, -11.0)


def test_constant_batch_one_cache_is_reused_for_different_legal_batches() -> None:
    constant = np.ones((1, 3, 4))
    expr = contract_expression(
        "...ij,...jk->...ik",
        (1, 2, 3),
        constant,
        constants=[1],
        dynamic_axes=["..."],
    )

    first = expr(np.ones((2, 2, 3)))
    cache = expr._evaluated_constants["numpy"]
    second = expr(np.ones((5, 2, 3)))

    assert first.shape == (2, 2, 4)
    assert second.shape == (5, 2, 4)
    np.testing.assert_allclose(first, 3.0)
    np.testing.assert_allclose(second, 3.0)
    assert cache is expr._evaluated_constants["numpy"]

    # the constant's own channel dimension is still locked
    with pytest.raises(ValueError, match=r"(?s)Operand 0 .* static label 'j'"):
        expr(np.ones((5, 2, 6)))


def test_unenabled_contract_keeps_legacy_permissive_calls() -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4))

    result = expr(np.ones((5, 2, 6)), np.ones((1, 6, 4)))
    assert result.shape == (5, 2, 4)
    np.testing.assert_allclose(result, 6.0)
    assert len(expr.shape_contract.dynamic) == 1  # implicit, not enforced on plain calls


def test_implicit_contract_still_guards_out_writes(no_backend_calls) -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4))
    out = np.full((5, 2, 4), -11.0)

    with pytest.raises(ValueError, match=r"(?s)static label 'j'"):
        expr(np.ones((5, 2, 6)), np.ones((1, 6, 4)), out=out)

    assert no_backend_calls == []
    np.testing.assert_allclose(out, -11.0)


def test_shape_contract_disabled_explicitly() -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), shape_contract=False)
    assert expr.shape_contract is None

    assert expr(np.ones((5, 2, 6)), np.ones((1, 6, 4))).shape == (5, 2, 4)


def test_interleaved_input_gets_contract() -> None:
    expr = contract_expression(
        (1, 2, 3),
        [Ellipsis, "i", "j"],
        (1, 3, 4),
        [Ellipsis, "j", "k"],
        [Ellipsis, "i", "k"],
        dynamic_axes=["..."],
    )

    assert expr(np.ones((5, 2, 3)), np.ones((1, 3, 4))).shape == (5, 2, 4)
    with pytest.raises(ValueError, match=r"(?s)Operand 0 .* static label"):
        expr(np.ones((5, 2, 6)), np.ones((1, 6, 4)))


def test_shape_contract_helpers() -> None:
    assert find_ellipsis_labels("...ij,...jk->...ik", [(1, 2, 3), (1, 3, 4)]) == ("c",)
    assert find_ellipsis_labels("i...j,...jk->i...k", [(2, 1, 3), (1, 3, 4)]) == ("c",)
    assert find_ellipsis_labels("ij,jk->ik", [(2, 3), (3, 4)]) == ()

    contract = ShapeContract.from_expression("...ij,...jk->...ik", [(1, 2, 3), (1, 3, 4)], dynamic_axes=True)
    assert contract.dynamic == frozenset("c")
    contract.check([(5, 2, 3), (1, 3, 4)])
    with pytest.raises(ValueError):
        contract.check([(5, 2, 6), (1, 6, 4)])
    assert "dynamic='c'" in repr(contract)


def test_unknown_dynamic_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="do not appear"):
        contract_expression("ij,jk->ik", (2, 3), (3, 4), dynamic_axes=["z"])


def test_positional_dynamic_axes_mapping() -> None:
    expr = contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes={0: 0, 1: 0})
    assert expr.shape_contract is not None
    assert expr(np.ones((5, 2, 3)), np.ones((1, 3, 4))).shape == (5, 2, 4)
