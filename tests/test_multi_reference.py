import pytest

import jiwer


class TestMultiReferenceSingleSentence:
    def test_east_gate_west_door_selects_first_on_tie(self):
        # both references differ in exactly one word -> the first one is kept
        out = jiwer.process_words_multi_reference(
            ["east gate", "west door"], "east door"
        )

        assert isinstance(out, jiwer.MultiReferenceWordOutput)
        assert out.reference_indices == [0]
        assert out.reference_indexes == [0]
        assert out.reference_index == [0]
        assert out.references == [["east", "gate"]]
        assert out.hypotheses == [["east", "door"]]
        assert out.wer == pytest.approx(0.5)
        assert out.substitutions == 1
        assert out.hits == 1

        # the original alignment chunks are returned for the selected reference
        assert out.alignments == [
            [
                jiwer.AlignmentChunk(
                    type="equal",
                    ref_start_idx=0,
                    ref_end_idx=1,
                    hyp_start_idx=0,
                    hyp_end_idx=1,
                ),
                jiwer.AlignmentChunk(
                    type="substitute",
                    ref_start_idx=1,
                    ref_end_idx=2,
                    hyp_start_idx=1,
                    hyp_end_idx=2,
                ),
            ]
        ]

    def test_second_reference_wins_when_better(self):
        out = jiwer.process_words_multi_reference(
            ["east gate", "west door"], "west door"
        )

        assert out.reference_indices == [1]
        assert out.references == [["west", "door"]]
        assert out.wer == 0

    def test_nested_routing_through_process_words(self):
        out = jiwer.process_words([["east gate", "west door"]], ["east door"])

        assert isinstance(out, jiwer.MultiReferenceWordOutput)
        assert out.reference_indices == [0]
        assert out.alignments is not None

    def test_tuple_candidates(self):
        out = jiwer.process_words_multi_reference(
            [("east gate", "west door")], ["east door"]
        )
        assert out.reference_indices == [0]

    def test_flat_single_sentence_candidates(self):
        out = jiwer.process_words_multi_reference(
            ["a b", "a c"], "a c"
        )
        assert out.reference_indices == [1]
        assert out.wer == 0

    def test_empty_reference_convention(self):
        out = jiwer.process_words_multi_reference([""], "hello")
        assert out.reference_indices == [0]
        assert out.insertions == 1
        assert out.wer == 1
        assert out.mer == 1
        assert out.wip == 0
        assert out.wil == 1

        out = jiwer.process_words_multi_reference([""], "")
        assert out.wer == 0
        assert out.mer == 0
        assert out.wip == 1
        assert out.wil == 0

    def test_empty_reference_candidate_vs_non_empty(self):
        out = jiwer.process_words_multi_reference(
            [["", "a b"]], ["a b"], method="wer"
        )
        assert out.reference_indices == [1]
        assert out.wer == 0

    def test_transforms_are_applied_per_candidate(self):
        transform = jiwer.Compose(
            [jiwer.ToLowerCase(), jiwer.ReduceToListOfListOfWords()]
        )
        out = jiwer.process_words_multi_reference(
            [["EAST GATE", "west door"]],
            ["east gate"],
            reference_transform=transform,
        )
        assert out.reference_indices == [0]
        assert out.wer == 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            jiwer.process_words_multi_reference(["a"], "a", method="unknown")

    def test_method_aliases(self):
        refs = [["a", "a b c d"], ["w x"]]
        hyps = ["a b", "w y"]

        for alias in ("edit", "edits", "minimum_edit_distance"):
            out = jiwer.process_words_multi_reference(refs, hyps, method=alias)
            assert out.reference_indices == [0, 0]

        for alias in ("wer", "batch_wer", "minimum_wer", "oracle_wer"):
            out = jiwer.process_words_multi_reference(refs, hyps, method=alias)
            assert out.reference_indices == [1, 0]

    def test_malformed_mixed_reference_raises(self):
        with pytest.raises(ValueError):
            jiwer.process_words_multi_reference([["a b"], "c d"], ["x", "y"])

    def test_empty_reference_collection_raises(self):
        with pytest.raises(ValueError):
            jiwer.process_words_multi_reference([], ["a"])

    def test_empty_candidate_list_raises(self):
        with pytest.raises(ValueError):
            jiwer.process_words_multi_reference([["a"], []], ["a", "b"])

    def test_sentence_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            jiwer.process_words_multi_reference([["a"], ["b"]], ["a"])


class TestMultiReferenceBatch:
    def test_batch_wer_prefers_long_reference(self):
        # first recording: "a" or "a b c d", recognized as "a b"
        # second recording: "w x", recognized as "w y"
        refs = [["a", "a b c d"], ["w x"]]
        hyps = ["a b", "w y"]

        out = jiwer.process_words_multi_reference(refs, hyps, method="wer")

        assert out.reference_indices == [1, 0]
        assert out.references == [
            ["a", "b", "c", "d"],
            ["w", "x"],
        ]
        # long ref: 2 deletions ; second: 1 substitution -> 3 / 6 = 1/2
        assert out.wer == pytest.approx(1 / 2)
        assert out.deletions == 2
        assert out.substitutions == 1

    def test_batch_wer_prefers_short_reference_after_correct_recording(self):
        # the second recording is now a fully correct four-word utterance
        refs = [["a", "a b c d"], ["w x", "w x y z"]]
        hyps = ["a b", "w x y z"]

        out = jiwer.process_words_multi_reference(refs, hyps, method="wer")

        assert out.reference_indices == [0, 1]
        # first sentence: 1 insertion on the one-word reference
        # second sentence: perfect match on the four-word reference
        # total: 1 / 5
        assert out.wer == pytest.approx(1 / 5)
        assert out.insertions == 1

    def test_batch_wer_through_process_words_and_wer_function(self):
        refs = [["a", "a b c d"], ["w x"]]
        hyps = ["a b", "w y"]

        out = jiwer.process_words(refs, hyps, method="wer")
        assert out.reference_indices == [1, 0]
        assert out.wer == pytest.approx(1 / 2)

        score = jiwer.wer(refs, hyps, method="wer")
        assert score == pytest.approx(1 / 2)

        score = jiwer.wer_multi_reference(refs, hyps, method="wer")
        assert score == pytest.approx(1 / 2)

    def test_edit_mode_differs_from_wer_mode(self):
        refs = [["a", "a b c d"], ["w x"]]
        hyps = ["a b", "w y"]

        # per sentence, both first-sentence refs need a single edit
        out = jiwer.process_words_multi_reference(refs, hyps, method="edit")
        assert out.reference_indices == [0, 0]
        # 1 insertion (1 ref word) + 1 substitution (2 ref words) -> 2/3
        assert out.wer == pytest.approx(2 / 3)

    def test_batch_wer_tie_breaks_on_corpus_order(self):
        # construct two combinations with an identical batch WER
        # s1: "a" -> hyp "x" = 1/1,  "a b" -> hyp "x" = 2/2
        # s2: "c d e" -> hyp "f g" = 3/3, "f g" -> hyp "f g" = 0/2
        # combos: (0,0)=4/4=1.0, (0,1)=1/3, (1,0)=5/5=1.0, (1,1)=2/4=0.5
        # best is (0,1); verify ordering logic on an actual tie below
        refs = [["a", "a b"], ["c d", "e f"]]
        hyps = ["x", "y"]
        # every reference is fully wrong:
        # (0,0): (1+2)/ (1+2) = 1.0, (0,1): (1+2)/(1+2)=1.0,
        # (1,0): (2+2)/(2+2)=1.0, (1,1): (2+2)/(2+2)=1.0 -> all tie -> (0, 0)
        out = jiwer.process_words_multi_reference(refs, hyps, method="wer")
        assert out.reference_indices == [0, 0]
        assert out.wer == pytest.approx(1.0)

    def test_batch_wer_tie_picks_earliest_differing_sentence(self):
        # s2 candidates have identical stats, so the tie between full combos
        # must be resolved in favour of the first candidate of sentence 1
        refs = [["a b", "c d"], ["e f", "g h"]]
        hyps = ["a x", "e y"]
        # all combos have 2 edits on 2 ref words per sentence -> 4/4 = 1.0
        out = jiwer.process_words_multi_reference(refs, hyps, method="wer")
        assert out.reference_indices == [0, 0]

    def test_all_empty_references_keep_first_candidate(self):
        out = jiwer.process_words_multi_reference(
            [["", ""], [""]], ["a", "b c"], method="wer"
        )
        assert out.reference_indices == [0, 0]
        assert out.wer == 3  # zero-reference convention: insertions count
        assert out.insertions == 3

    def test_visualization_and_error_counts(self):
        out = jiwer.process_words_multi_reference(
            [["east gate", "west door"]], ["east door"]
        )

        visualization = jiwer.visualize_alignment(out, show_measures=True)
        assert "REF: east gate" in visualization
        assert "wer=50.00%" in visualization

        substitutions, insertions, deletions = jiwer.collect_error_counts(out)
        assert dict(substitutions) == {("gate", "door"): 1}
        assert dict(insertions) == {}
        assert dict(deletions) == {}


class TestBackwardCompatibility:
    def test_flat_string_entry_unchanged(self):
        out = jiwer.process_words("this is a test", "this test")
        assert type(out) is jiwer.WordOutput
        assert not isinstance(out, jiwer.MultiReferenceWordOutput)
        assert out.wer == 0.5

    def test_flat_list_entry_unchanged(self):
        reference = ["hello world", "i like monthy python"]
        hypothesis = ["hello duck", "i like python"]

        out = jiwer.process_words(reference, hypothesis)
        assert type(out) is jiwer.WordOutput
        # 2 substitutions over 6 reference words
        assert out.wer == pytest.approx(2 / 6)

    def test_measure_functions_accept_method_keyword(self):
        # one substitution over two words
        assert jiwer.wer("a b", "a c") == pytest.approx(0.5)
        assert jiwer.mer("a b", "a c") == pytest.approx(0.5)
        assert jiwer.wil("a b", "a c") == pytest.approx(0.75)
        assert jiwer.wip("a b", "a c") == pytest.approx(0.25)

    def test_cer_unchanged(self):
        assert jiwer.cer("abc", "abd") == pytest.approx(1 / 3)
        out = jiwer.process_characters("abc", "abd")
        assert type(out) is jiwer.CharacterOutput
