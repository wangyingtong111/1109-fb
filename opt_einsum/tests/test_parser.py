"""
Directly tests various parser utility functions.
"""

from typing import Any

import pytest

from opt_einsum.parser import (
    find_output_shape,
    gen_unused_symbols,
    get_shape,
    get_symbol,
    parse_einsum_input,
    sort_user_symbols,
)
from opt_einsum.testing import build_arrays_from_tuples


def test_get_symbol() -> None:
    assert get_symbol(2) == "c"
    assert get_symbol(200000) == "\U000315cc"
    # Ensure we skip surrogates '[\uD800-\uDFFF]'
    assert get_symbol(55155) == "\ud7ff"
    assert get_symbol(55156) == "\ue000"
    assert get_symbol(55295) == "\ue08b"
    assert get_symbol(55296) == "\ue08c"
    assert get_symbol(57343) == "\ue88b"


@pytest.mark.parametrize("n", [52, 55156, 55157, 60000])
def test_symbols_utf8(n: int) -> None:
    symbols = "".join(get_symbol(i) for i in range(n))
    assert len(set(symbols)) == n
    assert symbols.encode("utf-8").decode("utf-8") == symbols


def test_unused_symbols_utf8() -> None:
    used = "ab\ud7ff\ue000"
    symbols = "".join(gen_unused_symbols(used, 60000))
    assert not set(symbols).intersection(used)
    assert len(set(symbols)) == 60000
    assert symbols.encode("utf-8").decode("utf-8") == symbols


def test_parse_einsum_input() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *ops])
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert operands == ops


def test_parse_einsum_input_shapes_error() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])

    with pytest.raises(ValueError):
        _ = parse_einsum_input([eq, *ops], shapes=True)


def test_parse_einsum_input_shapes() -> None:
    eq = "ab,bc,cd"
    shapes = [(2, 3), (3, 4), (4, 5)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert shapes == operands


def test_parse_with_ellisis() -> None:
    eq = "...a,ab"
    shapes = [(2, 3), (3, 4)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == "da,ab"
    assert output_subscript == "db"
    assert shapes == operands


def test_sort_user_symbols() -> None:
    # comparable labels retain their implicit sorted order
    assert sort_user_symbols({2, 0, 1}) == [0, 1, 2]
    assert sort_user_symbols({"time", "channel"}) == ["channel", "time"]

    # incomparable types are grouped by type name and sorted within each
    # group, so 7 and '7' remain distinct axes
    assert sort_user_symbols({"time", 7, "channel"}) == [7, "channel", "time"]
    assert sort_user_symbols({"7", 7}) == [7, "7"]

    # genuinely uncomparable labels still raise
    class Uncomparable:
        __hash__ = object.__hash__

        def __lt__(self, other: object) -> bool:
            raise TypeError("uncomparable")

    with pytest.raises(TypeError):
        sort_user_symbols({Uncomparable(), Uncomparable()})


def test_parse_einsum_input_label_map() -> None:
    shapes = [(2, 3, 5, 4, 4), (3, 7), (7, 8)]
    operands = [
        shapes[0],
        [Ellipsis, 7, "trace", "trace"],
        shapes[1],
        [Ellipsis, "time"],
        shapes[2],
        ["time", "feature"],
        [Ellipsis, 7, "feature"],
    ]
    input_subscripts, output_subscript, ops, label_map = parse_einsum_input(
        operands, shapes=True, return_label_map=True
    )
    assert input_subscripts == "hiadd,ic,cb"
    assert output_subscript == "hiab"
    assert ops == tuple(shapes)
    # integers sort before strings, and the ellipsis expands from the right
    assert label_map == {
        "a": 7,
        "b": "feature",
        "c": "time",
        "d": "trace",
        "h": (Ellipsis, 0),
        "i": (Ellipsis, 1),
    }


def test_parse_einsum_input_label_map_string_equation() -> None:
    eq = "...a,ab->...b"
    shapes = [(2, 3), (3, 4)]
    input_subscripts, output_subscript, operands, label_map = parse_einsum_input(
        [eq, *shapes], shapes=True, return_label_map=True
    )
    assert input_subscripts == "da,ab"
    assert output_subscript == "db"
    # string equations simply map every symbol to itself
    assert label_map == {"a": "a", "b": "b", "d": (Ellipsis, 0)}


@pytest.mark.parametrize(
    "array, shape",
    [
        [[5], (1,)],
        [[5, 5], (2,)],
        [(5, 5), (2,)],
        [[[[[[5, 2]]]]], (1, 1, 1, 1, 2)],
        [[[[[["abcdef", "b"]]]]], (1, 1, 1, 1, 2)],
        ["A", ()],
        [b"A", ()],
        [True, ()],
        [5, ()],
        [5.0, ()],
        [5.0 + 0j, ()],
    ],
)
def test_get_shapes(array: Any, shape: tuple[int]) -> None:
    assert get_shape(array) == shape


def test_find_output_shape_with_zero_length_broadcast() -> None:
    assert find_output_shape(["i", "i"], [(0,), (1,)], "i") == (0,)
