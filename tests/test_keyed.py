import pytest

import jiwer


def test_aligns_by_id_and_orders_output():
    # reference: a=red, b=green; hypothesis: b=green, c=blue
    out = jiwer.process_words_with_ids(
        reference=[("a", "red"), ("b", "green")],
        hypothesis=[("b", "green"), ("c", "blue")],
    )

    # output order: reference order first, then hypothesis-only IDs
    assert out.ids == ["a", "b", "c"]

    # missing hypothesis for "a" aligns with an empty sentence -> deletion
    # "b" is a hit
    # missing reference for "c" aligns with an empty sentence -> insertion
    assert out.deletions == 1
    assert out.hits == 1
    assert out.insertions == 1
    assert out.substitutions == 0

    assert out.wer == pytest.approx(1.0)

    assert out.references == [["red"], ["green"], []]
    assert out.hypotheses == [[], ["green"], ["blue"]]


def test_duplicate_hypothesis_id_fails_entirely():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        jiwer.process_words_with_ids(
            reference=[("a", "red"), ("b", "green")],
            hypothesis=[("b", "green"), ("c", "blue"), ("b", "rose")],
        )


def test_duplicate_reference_id_fails_entirely():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        jiwer.process_words_with_ids(
            reference=[("a", "red"), ("a", "blue")],
            hypothesis=[("a", "red")],
        )


def test_accepts_one_shot_iterables():
    out = jiwer.process_words_with_ids(
        reference=iter([("a", "red"), ("b", "green")]),
        hypothesis=(record for record in [("b", "green"), ("c", "blue")]),
    )

    assert out.ids == ["a", "b", "c"]
    assert out.wer == pytest.approx(1.0)


def test_empty_transcripts_on_both_sides():
    out = jiwer.process_words_with_ids(
        reference=[("a", ""), ("b", "hello world")],
        hypothesis=[("a", ""), ("b", "hello world")],
    )

    assert out.ids == ["a", "b"]
    assert out.hits == 2
    assert out.deletions == 0
    assert out.insertions == 0
    assert out.substitutions == 0
    assert out.wer == 0


def test_empty_transcript_counts_like_missing():
    # an empty hypothesis transcript contributes deletions just like a
    # missing hypothesis record does
    out_empty = jiwer.process_words_with_ids(
        reference=[("a", "hello")],
        hypothesis=[("a", "")],
    )
    out_missing = jiwer.process_words_with_ids(
        reference=[("a", "hello")],
        hypothesis=[],
    )

    for out in (out_empty, out_missing):
        assert out.deletions == 1
        assert out.wer == pytest.approx(1.0)


def test_default_process_words_still_pairs_by_position():
    out = jiwer.process_words(
        reference=["red", "green"],
        hypothesis=["green", "blue"],
    )

    # positional pairing: red->green (sub), green->blue (sub)
    assert out.substitutions == 2
    assert out.hits == 0
    assert out.wer == pytest.approx(1.0)
