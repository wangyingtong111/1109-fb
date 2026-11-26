import unittest

import jiwer


class TestTimestampAlignment(unittest.TestCase):
    def test_repeated_word_deletes_earliest_occurrence(self):
        # reference "go go" at 0s and 5s, hypothesis only has "go" at 5s.
        # The first reference word is deleted, the second is a hit.
        out = jiwer.process_words_with_timestamps(
            reference="go go",
            hypothesis="go",
            reference_timestamps=[[0.0, 5.0]],
            hypothesis_timestamps=[[5.0]],
        )

        self.assertEqual(out.hit_pairs, [[(1, 0)]])
        self.assertEqual(out.delete_intervals, [[(0, 1)]])

        chunks = out.alignments[0]
        self.assertEqual(
            [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
             for c in chunks],
            [("delete", 0, 1, 0, 0), ("equal", 1, 2, 0, 1)],
        )

        self.assertEqual(out.hits, 1)
        self.assertEqual(out.deletions, 1)
        self.assertEqual(out.substitutions, 0)
        self.assertEqual(out.insertions, 0)
        self.assertEqual(out.wer, 0.5)

    def test_legal_alignment_for_two_hypothesis_words(self):
        # reference "go go go" at 0, 5, 10s, hypothesis "go go" at 9, 10s.
        # Only the first reference word is deleted; the remaining two are hit
        # in order, even though the first hypothesis word is locally closest
        # to the last reference word.
        out = jiwer.process_words_with_timestamps(
            reference="go go go",
            hypothesis="go go",
            reference_timestamps=[[0.0, 5.0, 10.0]],
            hypothesis_timestamps=[[9.0, 10.0]],
        )

        self.assertEqual(out.hit_pairs, [[(1, 0), (2, 1)]])
        self.assertEqual(out.delete_intervals, [[(0, 1)]])

        chunks = out.alignments[0]
        self.assertEqual(
            [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
             for c in chunks],
            [("delete", 0, 1, 0, 0), ("equal", 1, 3, 0, 2)],
        )

        self.assertEqual(out.hits, 2)
        self.assertEqual(out.deletions, 1)
        self.assertAlmostEqual(out.wer, 1 / 3)

    def test_edit_distance_has_priority_over_timestamps(self):
        # A locally closer timestamp may never increase the edit distance.
        # Reference "a b", hypothesis "b": the only single-edit alignment
        # deletes "a" and hits "b", even though "b" is temporally far away.
        out = jiwer.process_words_with_timestamps(
            reference="a b",
            hypothesis="b",
            reference_timestamps=[[0.0, 100.0]],
            hypothesis_timestamps=[[1.0]],
        )

        self.assertEqual(out.hit_pairs, [[(1, 0)]])
        self.assertEqual(out.delete_intervals, [[(0, 1)]])
        self.assertEqual(out.deletions, 1)
        self.assertEqual(out.hits, 1)

    def test_minimum_time_difference_breaks_edit_tie(self):
        # Both alignments have one deletion. The timestamp cost decides which
        # reference occurrence is deleted.
        closer = jiwer.process_words_with_timestamps(
            reference="go go",
            hypothesis="go",
            reference_timestamps=[[0.0, 5.0]],
            hypothesis_timestamps=[[4.0]],
        )
        self.assertEqual(closer.hit_pairs, [[(1, 0)]])
        self.assertEqual(closer.delete_intervals, [[(0, 1)]])

        farther = jiwer.process_words_with_timestamps(
            reference="go go",
            hypothesis="go",
            reference_timestamps=[[4.0, 10.0]],
            hypothesis_timestamps=[[5.0]],
        )
        self.assertEqual(farther.hit_pairs, [[(0, 0)]])
        self.assertEqual(farther.delete_intervals, [[(1, 2)]])

    def test_equal_timestamps_give_stable_index_order(self):
        # With identical timestamps there is nothing to distinguish the
        # occurrences, so a stable, deterministic index order is returned.
        # The diagonal (hit) move is preferred, hence the earliest reference
        # words are hit in order and the last occurrence is deleted.
        out = jiwer.process_words_with_timestamps(
            reference="go go go",
            hypothesis="go go",
            reference_timestamps=[[5.0, 5.0, 5.0]],
            hypothesis_timestamps=[[5.0, 5.0]],
        )

        self.assertEqual(out.hit_pairs, [[(0, 0), (1, 1)]])
        self.assertEqual(out.delete_intervals, [[(2, 3)]])

        # running the same computation again must give the identical result
        again = jiwer.process_words_with_timestamps(
            reference="go go go",
            hypothesis="go go",
            reference_timestamps=[[5.0, 5.0, 5.0]],
            hypothesis_timestamps=[[5.0, 5.0]],
        )
        self.assertEqual(again.hit_pairs, out.hit_pairs)
        self.assertEqual(again.delete_intervals, out.delete_intervals)

    def test_insertions_and_substitutions(self):
        out = jiwer.process_words_with_timestamps(
            reference="a b",
            hypothesis="a x b",
            reference_timestamps=[[0.0, 2.0]],
            hypothesis_timestamps=[[0.0, 1.0, 2.0]],
        )

        self.assertEqual(out.hit_pairs, [[(0, 0), (1, 2)]])
        self.assertEqual(out.delete_intervals, [[]])
        self.assertEqual(out.insertions, 1)
        self.assertEqual(out.hits, 2)
        self.assertEqual(out.wer, 0.5)

        substitution = jiwer.process_words_with_timestamps(
            reference="a b",
            hypothesis="a c",
            reference_timestamps=[[0.0, 1.0]],
            hypothesis_timestamps=[[0.0, 1.0]],
        )
        self.assertEqual(substitution.hit_pairs, [[(0, 0)]])
        self.assertEqual(substitution.delete_intervals, [[]])
        self.assertEqual(substitution.substitutions, 1)

    def test_multiple_sentences(self):
        out = jiwer.process_words_with_timestamps(
            reference=["go go", "a b"],
            hypothesis=["go", "a b"],
            reference_timestamps=[[0.0, 5.0], [0.0, 1.0]],
            hypothesis_timestamps=[[5.0], [0.0, 1.0]],
        )

        self.assertEqual(out.hit_pairs, [[(1, 0)], [(0, 0), (1, 1)]])
        self.assertEqual(out.delete_intervals, [[(0, 1)], []])
        self.assertEqual(out.deletions, 1)
        self.assertEqual(out.hits, 3)

    def test_empty_sentences(self):
        both_empty = jiwer.process_words_with_timestamps(
            reference="", hypothesis="",
            reference_timestamps=[[]], hypothesis_timestamps=[[]],
        )
        self.assertEqual(both_empty.hit_pairs, [[]])
        self.assertEqual(both_empty.delete_intervals, [[]])
        self.assertEqual(both_empty.wer, 0)
        self.assertEqual(both_empty.wip, 1)

        empty_ref = jiwer.process_words_with_timestamps(
            reference="", hypothesis="x",
            reference_timestamps=[[]], hypothesis_timestamps=[[1.0]],
        )
        self.assertEqual(empty_ref.insertions, 1)
        self.assertEqual(empty_ref.hit_pairs, [[]])
        self.assertEqual(empty_ref.delete_intervals, [[]])

        empty_hyp = jiwer.process_words_with_timestamps(
            reference="x", hypothesis="",
            reference_timestamps=[[1.0]], hypothesis_timestamps=[[]],
        )
        self.assertEqual(empty_hyp.deletions, 1)
        self.assertEqual(empty_hyp.delete_intervals, [[(0, 1)]])
        self.assertEqual(empty_hyp.hit_pairs, [[]])

    def test_integer_timestamps_and_tuple_input(self):
        out = jiwer.process_words(
            reference="go go",
            hypothesis="go",
            reference_timestamps=((0, 5),),
            hypothesis_timestamps=((5,),),
        )
        self.assertEqual(out.hit_pairs, [[(1, 0)]])
        self.assertEqual(out.delete_intervals, [[(0, 1)]])

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            # only one side has timestamps
            jiwer.process_words(
                reference="a", hypothesis="a",
                reference_timestamps=[[0.0]],
            )

        with self.assertRaises(ValueError):
            # timestamp count does not match word count
            jiwer.process_words_with_timestamps(
                reference="a b", hypothesis="a",
                reference_timestamps=[[0.0, 1.0]],
                hypothesis_timestamps=[[0.0, 2.0]],
            )

        with self.assertRaises(ValueError):
            # number of timestamp sentences does not match
            jiwer.process_words_with_timestamps(
                reference=["a", "b"], hypothesis=["a", "b"],
                reference_timestamps=[[0.0]],
                hypothesis_timestamps=[[0.0], [1.0]],
            )

        with self.assertRaises(ValueError):
            # non-numeric timestamp
            jiwer.process_words_with_timestamps(
                reference="a", hypothesis="a",
                reference_timestamps=[["now"]],
                hypothesis_timestamps=[[0.0]],
            )


class TestBackwardCompatibility(unittest.TestCase):
    def test_no_timestamps_keeps_existing_behavior(self):
        out = jiwer.process_words("go go go", "go go")

        # detailed fields are not populated without timestamps
        self.assertEqual(out.hit_pairs, [])
        self.assertEqual(out.delete_intervals, [])

        # the existing rapidfuzz alignment (last reference word deleted) is
        # kept untouched on the timestamp-free code path
        chunks = out.alignments[0]
        self.assertEqual(
            [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
             for c in chunks],
            [("equal", 0, 2, 0, 2), ("delete", 2, 3, 2, 2)],
        )
        self.assertEqual(out.hits, 2)
        self.assertEqual(out.deletions, 1)

    def test_measures_identical_with_and_without_timestamps(self):
        # use sentences without repeated words so that the optimal alignment
        # is unique regardless of the timestamps
        without = jiwer.process_words(
            "alpha beta gamma delta", "alpha beta epsilon delta"
        )
        with_ts = jiwer.process_words_with_timestamps(
            "alpha beta gamma delta",
            "alpha beta epsilon delta",
            reference_timestamps=[[0.0, 1.0, 2.0, 3.0]],
            hypothesis_timestamps=[[0.0, 1.0, 2.5, 3.0]],
        )

        self.assertEqual(without.hits, with_ts.hits)
        self.assertEqual(without.substitutions, with_ts.substitutions)
        self.assertEqual(without.deletions, with_ts.deletions)
        self.assertEqual(without.insertions, with_ts.insertions)
        self.assertAlmostEqual(without.wer, with_ts.wer)
        self.assertAlmostEqual(without.mer, with_ts.mer)
        self.assertAlmostEqual(without.wil, with_ts.wil)
        self.assertAlmostEqual(without.wip, with_ts.wip)
        self.assertEqual(with_ts.hit_pairs, [[(0, 0), (1, 1), (3, 3)]])
        self.assertEqual(with_ts.delete_intervals, [[]])

    def test_visualization_still_works_with_timestamps(self):
        out = jiwer.process_words_with_timestamps(
            "go go", "go",
            reference_timestamps=[[0.0, 5.0]],
            hypothesis_timestamps=[[5.0]],
        )
        visualization = jiwer.visualize_alignment(out, show_measures=False)
        self.assertIn("REF:", visualization)
        self.assertIn("HYP:", visualization)


if __name__ == "__main__":
    unittest.main()
