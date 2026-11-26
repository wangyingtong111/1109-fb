import jiwer


NORMAL_REGIONS = [
    ("A", [("a", "go now")], [("a1", "go")]),
    ("B", [("b", "next")], [("b1", "now next")]),
]

CRITICAL_REGIONS = [
    (
        "A",
        [("a", "go"), ("a2", "now"), ("a3", "then")],
        [("a1", "go")],
    ),
    (
        "B",
        [("b", "next")],
        [("b1", "now"), ("b2", "then next")],
    ),
]


def test_region_alignment_is_isolated_and_attributes_chunks_to_source_lines():
    output = jiwer.process_region_words(NORMAL_REGIONS)

    assert (output.deletions, output.insertions) == (1, 1)
    assert output.details == (
        ("A", "delete", (("a", 1),), ()),
        ("B", "insert", (), (("b1", 0),)),
    )
    assert output.as_case_tuple() == (
        (1, 1),
        (
            ("A", "delete", (("a", 1),), ()),
            ("B", "insert", (), (("b1", 0),)),
        ),
    )

    delete_edit, insert_edit = output.edits
    assert delete_edit.ref_line == "a"
    assert delete_edit.ref_word_indices == (1,)
    assert insert_edit.hyp_line == "b1"
    assert insert_edit.hyp_word_indices == (0,)

    # The same groups are available through process_words(regions=...).
    via_process_words = jiwer.process_words(regions=NORMAL_REGIONS)
    assert via_process_words.details == output.details


def test_merged_same_type_chunk_still_records_every_word_source():
    output = jiwer.process_region_words(CRITICAL_REGIONS)

    assert (output.deletions, output.insertions) == (2, 2)
    assert output.details == (
        ("A", "delete", (("a2", 0), ("a3", 0)), ()),
        ("B", "insert", (), (("b1", 0), ("b2", 0))),
    )

    delete_edit, insert_edit = output.edits
    assert delete_edit.ref_lines == ("a2", "a3")
    assert delete_edit.ref_word_indices == (0, 0)
    assert insert_edit.hyp_lines == ("b1", "b2")
    assert insert_edit.hyp_word_indices == (0, 0)

    # Each word is also available as an individual attribution.
    reference_words = [
        attribution
        for attribution in output.word_edits
        if attribution.side == "reference"
    ]
    hypothesis_words = [
        attribution
        for attribution in output.word_edits
        if attribution.side == "hypothesis"
    ]
    assert [tuple(item.position) for item in reference_words] == [
        ("a2", 0),
        ("a3", 0),
    ]
    assert [tuple(item.position) for item in hypothesis_words] == [
        ("b1", 0),
        ("b2", 0),
    ]

    # Every region reports independently, even though concatenating all words
    # across the fixed region boundary yields equal reference/hypothesis texts.
    assert output.region_ids == ("A", "B")
    assert output.region_results[0].deletions == 2
    assert output.region_results[0].insertions == 0
    assert output.region_results[1].deletions == 0
    assert output.region_results[1].insertions == 2

    # Without the fixed region boundary, contiguous merging cancels the same
    # words across regions and incorrectly reports zero edits.
    merged = jiwer.process_words(
        ["go", "now", "then", "next"],
        ["go", "now", "then next"],
        jiwer.wer_contiguous,
        jiwer.wer_contiguous,
    )
    assert merged.deletions == 0
    assert merged.insertions == 0


def test_plain_lines_receive_global_zero_based_line_ids():
    output = jiwer.process_region_words(
        [
            ("A", ["go now"], ["go"]),
            ("B", ["next"], ["now next"]),
        ]
    )

    # Reference lines are global line 0 (A) and 1 (B); hypothesis "now" is on
    # global hyp line 1 because A's "go" occupies hyp line 0.
    assert output.details == (
        ("A", "delete", ((0, 1),), ()),
        ("B", "insert", (), ((1, 0),)),
    )

    deleted_source = output.edits[0].ref_positions[0]
    inserted_source = output.edits[1].hyp_positions[0]
    assert deleted_source.line_index == 0
    assert deleted_source.region_line_index == 0
    assert inserted_source.line_index == 1
    assert inserted_source.region_line_index == 0


def test_flat_lines_can_be_grouped_by_region_labels():
    output = jiwer.process_words(
        ["go now", "next"],
        ["go", "now next"],
        reference_regions=["A", "B"],
        hypothesis_regions=["A", "B"],
    )

    assert output.region_ids == ("A", "B")
    assert output.deletions == 1
    assert output.insertions == 1


def test_nested_region_lists_are_supported():
    output = jiwer.process_words(
        [("A", ["go now"]), ("B", ["next"])],
        [("A", ["go"]), ("B", ["now next"])],
    )

    assert output.region_ids == ("A", "B")
    assert output.details == (
        ("A", "delete", ((0, 1),), ()),
        ("B", "insert", (), ((1, 0),)),
    )


def test_process_words_without_regions_remains_positional_sentence_alignment():
    output = jiwer.process_words("go now", "go")
    assert (output.hits, output.substitutions, output.deletions, output.insertions) == (
        1,
        0,
        1,
        0,
    )

    lines = jiwer.process_words(["go now", "next"], ["go", "now next"])
    assert lines.deletions == 1
    assert lines.insertions == 1
    # Provenance is attached by the opt-in region entry point; historical
    # process_words chunks keep their old five-coordinate interpretation.
    assert lines.alignments[0][1].ref_positions == ()

    assert jiwer.wer("go now", "go") == 0.5
