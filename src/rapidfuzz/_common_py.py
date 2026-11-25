# SPDX-License-Identifier: MIT
# Copyright (C) 2023 Max Bachmann

from __future__ import annotations

from array import array
from collections.abc import Hashable, Sequence


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


def conv_sequences(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> tuple[Sequence[Hashable], Sequence[Hashable]]:
    if isinstance(s1, str) and isinstance(s2, str):
        return s1, s2

    if isinstance(s1, bytes) and isinstance(s2, bytes):
        return s1, s2

    return conv_sequence(s1), conv_sequence(s2)


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


def _resulting_length(ops: Sequence[tuple[str, int, int]], src_len: int) -> int:
    """
    Length of the string the edit operations produce when they are applied to
    a string of length `src_len`
    """
    inserted = 0
    deleted = 0
    for tag, _src_pos, _dest_pos in ops:
        if tag == "insert":
            inserted += 1
        elif tag == "delete":
            deleted += 1

    return src_len - deleted + inserted


def _invalid(msg: str) -> ValueError:
    return ValueError(f"Editops can not be composed: {msg}")


def compose_editops(
    first: Sequence[tuple[str, int, int]],
    first_src_len: int,
    first_dest_len: int,
    second: Sequence[tuple[str, int, int]],
    second_src_len: int,
    second_dest_len: int,
) -> list[tuple[str, int, int]]:
    """
    Compose two consecutive edit operation sequences into a single one.

    `first` turns a string of length `first_src_len` into the intermediate
    string of length `first_dest_len`. `second` turns this intermediate string
    of length `second_src_len` into the final string of length `second_dest_len`.
    The returned list of `(tag, src_pos, dest_pos)` tuples turns the original
    string into the final string.

    The composition tracks the identity of every character instead of only
    comparing the resulting texts. A character which is replaced by the first
    sequence and removed by the second one is reported as a deletion of the
    original character, even when another character with the same value would
    be available. An insertion which is removed again by the second sequence
    cancels out, while a character touched by only one of the two sequences
    keeps its position in the original string.
    """
    if first_dest_len != second_src_len:
        msg = (
            "dest_len of the first Editops "
            f"({first_dest_len}) does not match src_len of the second Editops ({second_src_len})"
        )
        raise _invalid(msg)

    if _resulting_length(first, first_src_len) != first_dest_len:
        msg = (
            "the first edit operations do not transform a string of length "
            f"{first_src_len} into a string of length {first_dest_len}"
        )
        raise _invalid(msg)

    if _resulting_length(second, second_src_len) != second_dest_len:
        msg = (
            "the second edit operations do not transform a string of length "
            f"{second_src_len} into a string of length {second_dest_len}"
        )
        raise _invalid(msg)

    # --- locate every character of the intermediate string in the original string ---
    # src_of_mid[i] is the index of the original character the intermediate
    # character i was created from, -1 when it was inserted by the first sequence
    src_of_mid = [-1] * first_dest_len
    # position in the original string the intermediate character i belongs to.
    # This is also defined for inserted characters, where it fixes the position
    # the insertion has to be replayed at.
    anchor_of_mid = [0] * first_dest_len
    # whether the intermediate character equals the character it was created from.
    # It is cleared when either sequence replaces the character.
    kept = [True] * first_dest_len
    mid_of_src = [-1] * first_src_len

    src_pos = 0
    mid_pos = 0
    for tag, _src_pos, op_mid_pos in first:
        # characters between the last and the current edit operation are kept
        while mid_pos < op_mid_pos:
            if mid_pos >= first_dest_len or src_pos >= first_src_len:
                raise _invalid("the first edit operations are out of range")
            mid_of_src[src_pos] = mid_pos
            src_of_mid[mid_pos] = src_pos
            anchor_of_mid[mid_pos] = src_pos
            src_pos += 1
            mid_pos += 1

        if tag == "delete":
            src_pos += 1
        elif tag == "insert":
            anchor_of_mid[mid_pos] = src_pos
            mid_pos += 1
        else:  # replace
            mid_of_src[src_pos] = mid_pos
            src_of_mid[mid_pos] = src_pos
            anchor_of_mid[mid_pos] = src_pos
            kept[mid_pos] = False
            src_pos += 1
            mid_pos += 1

    # characters after the last edit operation are kept
    while mid_pos < first_dest_len:
        if src_pos >= first_src_len:
            raise _invalid("the first edit operations are out of range")
        mid_of_src[src_pos] = mid_pos
        src_of_mid[mid_pos] = src_pos
        anchor_of_mid[mid_pos] = src_pos
        src_pos += 1
        mid_pos += 1

    # --- locate every character of the final string in the intermediate string ---
    dest_of_mid = [-1] * first_dest_len
    mid_of_dest = [-1] * second_dest_len
    # position in the original string an inserted character of the final string
    # has to be inserted at
    anchor_of_dest = [-1] * second_dest_len

    mid_pos = 0
    dest_pos = 0
    for tag, op_mid_pos, _dest_pos in second:
        while mid_pos < op_mid_pos:
            if mid_pos >= first_dest_len or dest_pos >= second_dest_len:
                raise _invalid("the second edit operations are out of range")
            mid_of_dest[dest_pos] = mid_pos
            dest_of_mid[mid_pos] = dest_pos
            mid_pos += 1
            dest_pos += 1

        if tag == "delete":
            mid_pos += 1
        elif tag == "insert":
            anchor_of_dest[dest_pos] = anchor_of_mid[mid_pos] if mid_pos < first_dest_len else first_src_len
            dest_pos += 1
        else:  # replace
            mid_of_dest[dest_pos] = mid_pos
            dest_of_mid[mid_pos] = dest_pos
            kept[mid_pos] = False
            mid_pos += 1
            dest_pos += 1

    while mid_pos < first_dest_len:
        if dest_pos >= second_dest_len:
            raise _invalid("the second edit operations are out of range")
        mid_of_dest[dest_pos] = mid_pos
        dest_of_mid[mid_pos] = dest_pos
        mid_pos += 1
        dest_pos += 1

    # characters inserted by the second sequence which originate from a character
    # inserted by the first sequence are ordered like that insertion
    for dest_index in range(second_dest_len):
        mid_index = mid_of_dest[dest_index]
        if mid_index != -1 and src_of_mid[mid_index] == -1:
            anchor_of_dest[dest_index] = anchor_of_mid[mid_index]

    # --- link the original characters with the characters of the final string ---
    dest_of_src = [-1] * first_src_len
    for src_index in range(first_src_len):
        mid_index = mid_of_src[src_index]
        if mid_index != -1:
            dest_of_src[src_index] = dest_of_mid[mid_index]

    # every character inserted by the second sequence has to know the position
    # in the original string it belongs to
    for dest_index in range(second_dest_len):
        if mid_of_dest[dest_index] == -1 and anchor_of_dest[dest_index] == -1:
            raise _invalid("the second edit operations are out of range")

    # --- describe the result from the perspective of the original string ---
    result = []
    src_pos = 0
    dest_pos = 0
    while src_pos < first_src_len or dest_pos < second_dest_len:
        # insertions belong to the position they were inserted at
        if dest_pos < second_dest_len and anchor_of_dest[dest_pos] == src_pos:
            result.append(("insert", src_pos, dest_pos))
            dest_pos += 1
            continue

        if src_pos >= first_src_len:
            raise _invalid("the edit operations are out of order")

        dest_index = dest_of_src[src_pos]
        if dest_index == -1:
            result.append(("delete", src_pos, dest_pos))
        else:
            if dest_index != dest_pos:
                raise _invalid("the edit operations are out of order")

            mid_index = mid_of_src[src_pos]
            if not kept[mid_index]:
                result.append(("replace", src_pos, dest_pos))

            dest_pos += 1

        src_pos += 1

    return result
