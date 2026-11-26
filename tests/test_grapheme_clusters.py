import unicodedata

import jiwer

REF_CONFLICT = " é👩‍🔬\r\nx "
HYP_CONFLICT = " é👩‍🔬\r\ny "


def test_default_character_spans_use_original_indices():
    out = jiwer.process_characters(" oak ", " oat ")

    assert out.cer == 1 / 3
    assert out.references == [["o", "a", "k"]]
    assert out.reference_spans == [[(1, 2), (2, 3), (3, 4)]]
    assert out.hypothesis_spans == [[(1, 2), (2, 3), (3, 4)]]

    chunk = out.alignments[0][-1]
    assert chunk.type == "substitute"
    assert chunk.ref_span == (3, 4)
    assert chunk.hyp_span == (3, 4)


def test_extended_grapheme_clusters_and_original_spans():
    out = jiwer.process_characters(REF_CONFLICT, HYP_CONFLICT, grapheme_clusters=True)

    assert out.cer == 0.25
    assert out.references == [["é", "👩‍🔬", "\r\n", "x"]]
    assert out.hypotheses == [["é", "👩‍🔬", "\r\n", "y"]]
    assert out.reference_spans == [[(1, 3), (3, 6), (6, 8), (8, 9)]]
    assert out.hypothesis_spans == [[(1, 2), (2, 5), (5, 7), (7, 8)]]

    chunk = out.alignments[0][-1]
    assert chunk.type == "substitute"
    assert chunk.ref_start_idx == 3
    assert chunk.ref_end_idx == 4
    assert chunk.hyp_start_idx == 3
    assert chunk.hyp_end_idx == 4
    assert chunk.ref_span == (8, 9)
    assert chunk.hyp_span == (7, 8)
    assert chunk.ref_char_start == 8
    assert chunk.ref_char_end == 9
    assert chunk.hyp_char_start == 7
    assert chunk.hyp_char_end == 8

    assert REF_CONFLICT[slice(*chunk.ref_span)] == "x"
    assert HYP_CONFLICT[slice(*chunk.hyp_span)] == "y"
    assert out.original_reference_unit(0, 0) == "é"
    assert out.original_hypothesis_unit(0, 0) == "é"


def test_grapheme_cer_alias():
    assert jiwer.cer(REF_CONFLICT, HYP_CONFLICT, graphemes=True) == 0.25
    assert (
        jiwer.cer(REF_CONFLICT, HYP_CONFLICT, unit="extended_grapheme_clusters") == 0.25
    )


def test_codepoint_mode_is_unchanged():
    ref = "q̲x"
    hyp = "q̇x"

    out = jiwer.process_characters(ref, hyp)

    assert out.cer == 1 / 3
    assert out.references == [["q", "̲", "x"]]
    assert out.reference_spans == [[(0, 1), (1, 2), (2, 3)]]
    assert jiwer.cer(ref, hyp) == 1 / 3


def test_grapheme_mode_counts_combined_q_cluster_once():
    ref = "q̲x"
    hyp = "q̇x"

    out = jiwer.process_characters(ref, hyp, grapheme_clusters=True)

    assert out.cer == 0.5
    assert out.references == [["q̲", "x"]]
    assert out.substitutions == 1
    assert out.hits == 1


def test_grapheme_splitter_contract():
    assert jiwer.split_grapheme_clusters(REF_CONFLICT) == [
        " ",
        "é",
        "👩‍🔬",
        "\r\n",
        "x",
        " ",
    ]
    assert jiwer.grapheme_cluster_spans(REF_CONFLICT) == [
        (0, 1),
        (1, 3),
        (3, 6),
        (6, 8),
        (8, 9),
        (9, 10),
    ]
    assert unicodedata.normalize(
        "NFC", "".join(jiwer.split_grapheme_clusters(REF_CONFLICT))
    ) == unicodedata.normalize("NFC", REF_CONFLICT)
