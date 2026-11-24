import pytest

import jiwer


def _output(reference, hypothesis):
    return jiwer.process_words(reference=reference, hypothesis=hypothesis)


@pytest.fixture
def output_a():
    # recording A: empty reference (silence), hypothesis inserts two words
    return _output("", "red blue")


@pytest.fixture
def output_b():
    # recording B: four perfectly correct words
    return _output("the cat sat down", "the cat sat down")


@pytest.fixture
def output_b_changed():
    # same recording B, but re-aligned with four different (still correct) words
    return _output("a dog runs fast", "a dog runs fast")


@pytest.fixture
def output_c():
    return _output("hello there", "hello there")


def test_merge_saved_summaries_is_order_independent(tmp_path, output_a, output_b):
    summary_ab = jiwer.merge_summaries(
        jiwer.summarize("A", output_a),
        jiwer.summarize("B", output_b),
    )
    summary_b = jiwer.summarize("B", output_b)

    path_ab = tmp_path / "ab.json"
    path_b = tmp_path / "b.json"
    jiwer.save_summary(summary_ab, path_ab)
    jiwer.save_summary(summary_b, path_b)

    loaded_ab = jiwer.load_summary(path_ab)
    loaded_b = jiwer.load_summary(path_b)

    for merged in (
        jiwer.merge_summaries(loaded_ab, loaded_b),
        jiwer.merge_summaries(loaded_b, loaded_ab),
    ):
        assert merged.hits == 4
        assert merged.insertions == 2
        assert merged.substitutions == 0
        assert merged.deletions == 0
        assert merged.wer == pytest.approx(0.5)
        assert sorted(merged.recording_ids) == ["A", "B"]


def test_duplicate_delivery_counted_once(output_b):
    merged = jiwer.merge_summaries(
        jiwer.summarize("B", output_b),
        jiwer.summarize("B", output_b),
    )

    assert len(merged) == 1
    assert merged.hits == 4
    assert merged.wer == 0


def test_same_content_different_ids_counted_separately(output_b):
    merged = jiwer.merge_summaries(
        jiwer.summarize("B1", output_b),
        jiwer.summarize("B2", output_b),
    )

    assert len(merged) == 2
    assert merged.hits == 8
    assert merged.wer == 0


def test_conflicting_recording_rejects_whole_merge(
    tmp_path, output_a, output_b, output_b_changed, output_c
):
    summary_ab = jiwer.merge_summaries(
        jiwer.summarize("A", output_a),
        jiwer.summarize("B", output_b),
    )
    path_ab = tmp_path / "ab.json"
    jiwer.save_summary(summary_ab, path_ab)
    loaded_ab = jiwer.load_summary(path_ab)

    # a new summary with a new recording C and a changed (still correct) B
    summary_cb = jiwer.merge_summaries(
        jiwer.summarize("C", output_c),
        jiwer.summarize("B", output_b_changed),
    )

    with pytest.raises(jiwer.MergeConflictError, match="'B'"):
        jiwer.merge_summaries(loaded_ab, summary_cb)

    # the previously aggregated results are preserved: B is unchanged and
    # C was not added
    assert sorted(loaded_ab.recording_ids) == ["A", "B"]
    assert "C" not in loaded_ab
    assert loaded_ab.hits == 4
    assert loaded_ab.insertions == 2
    assert loaded_ab.wer == pytest.approx(0.5)
    assert loaded_ab["B"].references == [["the", "cat", "sat", "down"]]


def test_conflict_even_when_counts_coincide(output_b, output_b_changed):
    # both versions of B have 4 hits and no errors, but the content differs
    assert output_b.hits == output_b_changed.hits == 4

    with pytest.raises(jiwer.MergeConflictError):
        jiwer.merge_summaries(
            jiwer.summarize("B", output_b),
            jiwer.summarize("B", output_b_changed),
        )


def test_empty_reference_conventions_preserved(output_a):
    summary = jiwer.summarize("A", output_a)

    # silence with insertions: wer equals the number of insertions
    assert summary.insertions == 2
    assert summary.wer == 2

    # complete silence correctly predicted: no error
    silent = jiwer.summarize("S", _output("", ""))
    assert silent.wer == 0
    assert silent.insertions == 0


def test_save_and_load_roundtrip(tmp_path, output_a, output_b):
    summary = jiwer.merge_summaries(
        jiwer.summarize("A", output_a),
        jiwer.summarize("B", output_b),
    )

    path = tmp_path / "summary.json"
    jiwer.save_summary(summary, path)
    loaded = jiwer.load_summary(path)

    assert sorted(loaded.recording_ids) == ["A", "B"]
    assert loaded["A"].references == [[]]
    assert loaded["A"].hypotheses == [["red", "blue"]]
    assert loaded.hits == summary.hits
    assert loaded.insertions == summary.insertions
    assert loaded.wer == pytest.approx(summary.wer)

