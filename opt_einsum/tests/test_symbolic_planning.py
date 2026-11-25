import pytest

import opt_einsum as oe
from opt_einsum.parser import find_output_shape


def test_symbolic_batch_size_is_retained_and_bindable() -> None:
    path, info = oe.contract_path(
        "bij,bjk->bik",
        ("B", 2, 3),
        (1, 3, 4),
        shapes=True,
        optimize=[(0, 1)],
    )

    assert path == [(0, 1)]
    assert len(info.size_list) == 1
    assert info.size_list[0] == "8×B"
    assert info.size_list[0] == "8*B"
    assert isinstance(info.size_list[0], str)

    assert info.size_list[0].bind(B=5) == 40
    assert info.bind(B=5).size_list == [40]
    assert oe.bind_plan(info, {"B": 5}).size_list == [40]

    _, integer_info = oe.contract_path(
        "bij,bjk->bik",
        (5, 2, 3),
        (1, 3, 4),
        shapes=True,
        optimize=[(0, 1)],
    )
    assert integer_info.size_list == [40]


def test_distinct_symbols_under_same_label_are_not_equated() -> None:
    _, info = oe.contract_path(
        "bij,bjk->bik",
        ("B", 2, 3),
        ("C", 3, 4),
        shapes=True,
        optimize=[(0, 1)],
    )

    assert info.size_list[0] == "8×broadcast(B,C)"
    assert info.size_list[0].bind(B=1, C=5) == 40
    assert info.size_list[0].bind(B=5, C=5) == 40

    with pytest.raises(ValueError, match="conflict|does not match|冲突"):
        info.bind(B=2, C=5)

    with pytest.raises(ValueError, match="conflict|does not match|冲突"):
        oe.contract_path(
            "bij,bjk->bik",
            ("B", 2, 3),
            ("C", 3, 4),
            shapes=True,
            optimize="greedy",
            bindings={"B": 2, "C": 5},
        )


def test_repeated_symbol_and_zero_one_broadcast_share_binding() -> None:
    assert find_output_shape(["b", "b"], [(0,), (1,)], "b") == (0,)
    symbolic = oe.broadcast_dimensions(["B", "C"], label="b")
    assert symbolic.bind(B=0, C=1) == 0

    _, info = oe.contract_path(
        "bij,bjk->bik",
        ("B", 2, 3),
        ("C", 3, 4),
        shapes=True,
        optimize=[(0, 1)],
        bindings={"B": 0, "C": 1},
    )
    assert info.size_list == [0]


def test_unresolved_priority_and_budget_comparisons_are_explicit() -> None:
    with pytest.raises((TypeError, ValueError), match="symbolic|unbound|binding"):
        oe.contract_path(
            "ab,bc,cd->ad",
            ("B", 2),
            (2, 3),
            (3, 4),
            shapes=True,
            optimize="greedy",
        )

    with pytest.raises((TypeError, ValueError), match="Cannot compare|unbound|symbol"):
        oe.contract_path(
            "bij,bjk->bik",
            ("B", 2, 3),
            ("C", 3, 4),
            shapes=True,
            optimize=[(0, 1)],
            memory_limit="max_input",
        )[1].largest_intermediate < 100


def test_plain_integer_planning_is_unchanged() -> None:
    path, info = oe.contract_path(
        "bij,bjk->bik",
        (5, 2, 3),
        (1, 3, 4),
        shapes=True,
        optimize=[(0, 1)],
    )
    assert path == [(0, 1)]
    assert info.size_list == [40]
    assert int(info.largest_intermediate) == 40
