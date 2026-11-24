"""Tests for fixed-node contraction plans.

A fixed-node plan names, for every step, both the static node ids it consumes
and the node id it produces. Unlike the legacy linear position paths the node
ids never get recycled, so the plan can be serialized with arbitrary (for
example non-contiguous) numbering.
"""

import pytest

import opt_einsum as oe
from opt_einsum import contract, contract_expression, contract_path
from opt_einsum.paths import (
    fixed_node_dependencies,
    fixed_node_to_linear,
    is_fixed_node_path,
    linear_to_ssa,
    validate_fixed_node_path,
)
from opt_einsum.testing import build_views
from opt_einsum.typing import FixedNodeStep

np = pytest.importorskip("numpy")

EQUATION = "ab,bc,cd,da->a"
SHAPES = [(2, 3), (3, 4), (4, 5), (5, 2)]

# A legal plan using deliberately non-contiguous, non-SSA node ids:
#   step 0: unary reduction of input 2            -> node 9
#   step 1: combine inputs 0 and 3 with node 9    -> node 12
#   step 2: merge input 1 with node 12            -> node 13
VALID_FIXED_PLAN = [
    ((2,), 9),
    ((0, 3, 9), 12),
    ((1, 12), 13),
]
EXPECTED_LINEAR_PATH = [(2,), (0, 2, 3), (0, 1)]


def test_is_fixed_node_path_detection() -> None:
    # fixed-node format - tuple pairs or explicit FixedNodeStep objects
    assert is_fixed_node_path(VALID_FIXED_PLAN)
    assert is_fixed_node_path([FixedNodeStep((0, 2), 9)])

    # an empty sequence keeps the legacy empty-path semantics
    assert not is_fixed_node_path([])

    # legacy position lists must keep their original meaning
    assert not is_fixed_node_path([(0, 2), (0, 1)])
    assert not is_fixed_node_path([(2,), (0, 2, 3), (0, 1)])
    assert not is_fixed_node_path([tuple(range(4))])

    # malformed objects are not silently treated as fixed-node plans
    assert not is_fixed_node_path("abc")
    assert not is_fixed_node_path([(0, 2)])
    assert not is_fixed_node_path([((0, 2), "9")])
    assert not is_fixed_node_path([(0, (9,))])


def test_fixed_node_to_linear_conversion() -> None:
    assert fixed_node_to_linear(VALID_FIXED_PLAN, 4) == EXPECTED_LINEAR_PATH

    # the same plan expressed with the explicit named tuple
    named_plan = [
        FixedNodeStep((2,), 9),
        FixedNodeStep((0, 3, 9), 12),
        FixedNodeStep((1, 12), 13),
    ]
    assert fixed_node_to_linear(named_plan, 4) == EXPECTED_LINEAR_PATH

    # ids need not be contiguous or increasing - the old ssa converter cannot do this
    sparse_plan = [
        ((1, 3), 42),
        ((0, 42), 7),
        ((2, 7), 100),
    ]
    assert fixed_node_to_linear(sparse_plan, 4) == [(1, 3), (0, 2), (0, 1)]


def test_fixed_node_dependencies_can_be_traced() -> None:
    deps = fixed_node_dependencies(VALID_FIXED_PLAN, 4)
    assert deps == {9: (2,), 12: (0, 3, 9), 13: (1, 12)}

    # imported (input) nodes can be looked up through the produced nodes
    assert deps[9] == (2,)
    assert 9 in deps[12]
    assert 12 in deps[13]


@pytest.mark.parametrize(
    "plan, match",
    [
        # the exact example from the specification: node 2 reused in step 1
        ([[(0, 2), 9], [(9, 2), 11]], "step 1.*node 2.*already consumed"),
        # referencing a node that was never created
        ([[(0, 9), 5]], "step 0.*node 9 does not exist"),
        # produced node id clashes with an input node
        ([[(0, 1), 2]], "step 0.*node 2 collides"),
        # produced node id clashes with an earlier produced node
        ([[(0, 1), 9], [(9, 2), 9]], "step 1.*node 9 collides"),
        # early termination leaves inputs orphaned - must not look like fewer operands
        ([[(0, 1), 9]], "left input node.*\\[2, 3\\]"),
        # a step that consumes nothing
        ([[(), 9]], "step 0.*does not consume any nodes"),
        # the same node listed twice within one step
        ([[(0, 0), 9], [(9, 1), 10]], "step 0.*node.*\\[0\\].*more than once"),
        # malformed step shape
        ([[(0, 2)]], "step 0.*\\(inputs, output\\) pair"),
    ],
)
def test_invalid_fixed_node_plans_raise_first_bad_step(plan: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_fixed_node_path(plan, 4)


def test_orphaned_inputs_use_equation_operand_count() -> None:
    # the plan only touches two of the four operands; the validator must use the
    # equation's operand count rather than inferring a smaller contraction
    with pytest.raises(ValueError, match="all 4 input operand"):
        validate_fixed_node_path([[(0, 1), 9]], 4)

    shapes = SHAPES
    with pytest.raises(ValueError, match="all 4 input operand"):
        contract_path(EQUATION, *shapes, shapes=True, optimize=[[(0, 1), 9]])


def test_contract_path_validates_before_array_data_is_read() -> None:
    bad_plan = [[(0, 2), 9], [(9, 2), 11]]

    # shape-only call never touches array data but must still fail
    with pytest.raises(ValueError, match="step 1.*node 2.*already consumed"):
        contract_path(EQUATION, *SHAPES, shapes=True, optimize=bad_plan)

    # a lazy array-like whose data can never be materialized: only the shape is
    # inspected while building the plan, element data must never be requested
    class LazyArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __array__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
            raise AssertionError("array data was read before the fixed-node plan was validated")

    with pytest.raises(ValueError, match="step 1.*node 2.*already consumed"):
        contract_path(
            EQUATION,
            LazyArray(SHAPES[0]),
            LazyArray(SHAPES[1]),
            LazyArray(SHAPES[2]),
            LazyArray(SHAPES[3]),
            optimize=bad_plan,
        )


def test_contract_path_records_fixed_node_information() -> None:
    path, info = contract_path(EQUATION, *SHAPES, shapes=True, optimize=VALID_FIXED_PLAN)
    assert path == EXPECTED_LINEAR_PATH
    assert info.fixed_node_path == [FixedNodeStep(*step) for step in VALID_FIXED_PLAN]
    assert info.node_dependencies == {9: (2,), 12: (0, 3, 9), 13: (1, 12)}
    # the printable PathInfo still works
    assert repr(info)

    # legacy position plans do not populate the fixed-node attributes
    _, legacy_info = contract_path(EQUATION, *SHAPES, shapes=True, optimize=EXPECTED_LINEAR_PATH)
    assert legacy_info.fixed_node_path == []
    assert legacy_info.node_dependencies == {}


def test_contract_with_fixed_node_plan_matches_naive_einsum() -> None:
    views = build_views(EQUATION)
    expected = contract(EQUATION, *views, optimize=False)

    fixed_result = contract(EQUATION, *views, optimize=VALID_FIXED_PLAN)
    linear_result = contract(EQUATION, *views, optimize=EXPECTED_LINEAR_PATH)

    np.testing.assert_allclose(fixed_result, expected)
    np.testing.assert_allclose(linear_result, expected)


def test_contract_expression_with_fixed_node_plan() -> None:
    views = build_views(EQUATION)
    expected = contract(EQUATION, *views, optimize=False)

    expr = contract_expression(EQUATION, *SHAPES, optimize=VALID_FIXED_PLAN)
    np.testing.assert_allclose(expr(*views), expected)


def test_fixed_node_plan_with_interleaved_input() -> None:
    views = build_views(EQUATION)
    expected = np.einsum(
        views[0], [0, 1], views[1], [1, 2], views[2], [2, 3], views[3], [3, 0], [0]
    )
    result = contract(
        views[0],
        [0, 1],
        views[1],
        [1, 2],
        views[2],
        [2, 3],
        views[3],
        [3, 0],
        [0],
        optimize=VALID_FIXED_PLAN,
    )
    np.testing.assert_allclose(result, expected)


def test_fixed_node_plan_roundtrips_with_ssa_paths() -> None:
    # plans produced by the existing SSA machinery remain valid fixed-node plans
    ssa_path = linear_to_ssa(EXPECTED_LINEAR_PATH)
    fixed_from_ssa = [(tuple(ids), 4 + i) for i, ids in enumerate(ssa_path)]

    assert fixed_node_to_linear(fixed_from_ssa, 4) == EXPECTED_LINEAR_PATH

    views = build_views(EQUATION)
    expected = contract(EQUATION, *views, optimize=False)
    np.testing.assert_allclose(contract(EQUATION, *views, optimize=fixed_from_ssa), expected)


def test_legacy_position_paths_keep_original_semantics() -> None:
    views = build_views(EQUATION)
    expected = contract(EQUATION, *views, optimize=False)

    for path in [
        [(1, 3), (0, 2), (0, 1)],
        [(2,), (0, 2, 3), (0, 1)],
        [tuple(range(4))],
    ]:
        np.testing.assert_allclose(contract(EQUATION, *views, optimize=path), expected)


def test_duplicate_node_name_does_not_change_operand_count() -> None:
    # producing node 2 again is an error even though 2 is a valid input id,
    # and it must not be misread as a two-operand plan
    with pytest.raises(ValueError, match="step 0.*node 2 collides"):
        contract_path(EQUATION, *SHAPES, shapes=True, optimize=[[(0, 1), 2]])


def test_fixed_node_plan_with_unary_final_reduction() -> None:
    # single-operand steps (e.g. reductions) are supported anywhere in the plan
    equation = "ab,bc->ac"
    shapes = [(2, 3), (3, 4)]
    plan = [[(1,), 7], [(0, 7), 8]]
    assert fixed_node_to_linear(plan, 2) == [(1,), (0, 1)]

    views = build_views(equation)
    np.testing.assert_allclose(
        contract(equation, *views, optimize=plan), contract(equation, *views, optimize=False)
    )
