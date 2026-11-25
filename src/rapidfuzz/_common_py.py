# SPDX-License-Identifier: MIT
# Copyright (C) 2023 Max Bachmann

from __future__ import annotations

from array import array
from collections.abc import Hashable, Sequence
from typing import Any


class _TokenCoder:
    """Preserve historical hash codes while resolving actual hash collisions."""

    # Supplementary private-use area.  It stays convertible with chr() and does
    # not collide with whitespace code points used by fuzz token helpers.
    _COLLISION_CODE_BASE = 0x100000

    __slots__ = ("_collision_count", "_groups", "_used_codes")

    def __init__(self) -> None:
        self._groups: dict[int, list[tuple[Any, Any]]] = {}
        self._used_codes: set[Any] = set()
        self._collision_count = 0

    @staticmethod
    def _equal(left: Any, right: Any) -> bool | None:
        try:
            result = left == right
        except Exception:
            result = NotImplemented

        if result is not NotImplemented:
            return bool(result)

        try:
            result = right == left
        except Exception:
            return None

        if result is NotImplemented:
            # Python would try identity comparison after both reflected calls fail.
            return left is right

        return bool(result)

    def _free_code(self) -> int:
        while True:
            code = self._COLLISION_CODE_BASE + self._collision_count
            self._collision_count += 1
            if code not in self._used_codes:
                self._used_codes.add(code)
                return code

    def encode(self, token: Any, historical_code: Any, token_hash: int) -> Any:
        group = self._groups.get(token_hash)

        if group is not None:
            equality_unavailable = False
            for known_token, token_code in group:
                if known_token is token:
                    return token_code

                equal = self._equal(known_token, token)
                if equal is True:
                    return token_code
                if equal is None:
                    equality_unavailable = True

            if equality_unavailable:
                # Historical custom hashables may expose __hash__ but no usable
                # __eq__. Their hash-only behaviour treated the whole bucket as
                # the same token.
                return group[0][1]

            token_code = self._free_code()
            group.append((token, token_code))
            return token_code

        # Avoid reusing a code already assigned to a different equality group.
        if historical_code not in self._used_codes:
            token_code = historical_code
        else:
            token_code = self._free_code()

        self._used_codes.add(token_code)
        self._groups[token_hash] = [(token, token_code)]
        return token_code


def _converted_token(elem: Any) -> tuple[Any, Any, int]:
    # this is required so e.g. a list of char can be compared to a string
    if isinstance(elem, str) and len(elem) == 1:
        code = ord(elem)
        return code, code, code

    # Keep the historical conversion result for -1. Equality-aware coding also
    # distinguishes it from -2 when their hashes collide.
    if isinstance(elem, int) and elem == -1:
        return -1, -1, hash(-1)

    token_hash = hash(elem)
    return elem, token_hash, token_hash


def conv_sequence(s: Sequence[Hashable]) -> Sequence[Hashable]:
    if isinstance(s, str):
        return [ord(x) for x in s]

    if isinstance(s, bytes):
        return s

    if isinstance(s, array):
        if s.typecode in ("u", "w"):
            return [ord(x) for x in s]

        return s

    if s is None:
        return s

    res = []
    for elem in s:
        if isinstance(elem, str) and len(elem) == 1:
            res.append(ord(elem))
        elif isinstance(elem, int) and elem == -1:
            res.append(-1)
        else:
            res.append(hash(elem))

    return res


def _conv_sequence_shared(s: Sequence[Hashable], coder: _TokenCoder) -> list[Any] | None:
    # Strings and bytes already have an equality-preserving integer representation.
    # Normalize arrays through the historical path before assigning shared codes.
    if isinstance(s, (str, bytes, array)):
        return [coder.encode(elem, elem, hash(elem)) for elem in conv_sequence(s)]

    if s is None:
        return None

    result = []
    for elem in s:
        token, historical_code, token_hash = _converted_token(elem)
        result.append(coder.encode(token, historical_code, token_hash))

    return result


def conv_sequences(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> tuple[Sequence[Hashable], Sequence[Hashable]]:
    if isinstance(s1, str) and isinstance(s2, str):
        return s1, s2

    if isinstance(s1, bytes) and isinstance(s2, bytes):
        return s1, s2

    coder = _TokenCoder()
    return _conv_sequence_shared(s1, coder), _conv_sequence_shared(s2, coder)


def common_prefix(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> int:
    prefix_len = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2:
            break

        prefix_len += 1

    return prefix_len


def common_suffix(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> int:
    suffix_len = 0
    for ch1, ch2 in zip(reversed(s1), reversed(s2)):
        if ch1 != ch2:
            break

        suffix_len += 1

    return suffix_len


def common_affix(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> tuple[int, int]:
    prefix_len = common_prefix(s1, s2)
    suffix_len = common_suffix(s1[prefix_len:], s2[prefix_len:])
    return (prefix_len, suffix_len)
