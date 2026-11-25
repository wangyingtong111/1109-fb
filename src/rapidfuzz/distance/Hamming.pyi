# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Callable, TypeVar, overload

from rapidfuzz.distance import Editops, Opcodes

_UnprocessedType1 = TypeVar("_UnprocessedType1")
_UnprocessedType2 = TypeVar("_UnprocessedType2")
_Wildcard = str | bytes | int

@overload
def comparable_count(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    wildcard: _Wildcard = "?",
    processor: None = None,
) -> int: ...
@overload
def comparable_count(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    wildcard: _Wildcard = "?",
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
) -> int: ...
@overload
def comparable_positions(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    wildcard: _Wildcard = "?",
    processor: None = None,
) -> tuple[int, ...]: ...
@overload
def comparable_positions(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    wildcard: _Wildcard = "?",
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
) -> tuple[int, ...]: ...
@overload
def distance(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: None = None,
    score_cutoff: int | None = None,
) -> int: ...
@overload
def distance(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
    score_cutoff: int | None = None,
) -> int: ...
@overload
def normalized_distance(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: None = None,
    score_cutoff: float | None = 0,
) -> float: ...
@overload
def normalized_distance(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
    score_cutoff: float | None = 0,
) -> float: ...
@overload
def similarity(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: None = None,
    score_cutoff: int | None = None,
) -> int: ...
@overload
def similarity(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
    score_cutoff: int | None = None,
) -> int: ...
@overload
def normalized_similarity(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: None = None,
    score_cutoff: float | None = 0,
) -> float: ...
@overload
def normalized_similarity(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    pad: bool = True,
    wildcard: _Wildcard | None = None,
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
    score_cutoff: float | None = 0,
) -> float: ...
@overload
def editops(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: None = None,
) -> Editops: ...
@overload
def editops(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
) -> Editops: ...
@overload
def opcodes(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    *,
    processor: None = None,
) -> Opcodes: ...
@overload
def opcodes(
    s1: _UnprocessedType1,
    s2: _UnprocessedType2,
    *,
    processor: Callable[[_UnprocessedType1 | _UnprocessedType2], Sequence[Hashable]],
) -> Opcodes: ...
