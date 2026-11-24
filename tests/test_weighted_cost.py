import unittest

import jiwer


class TestWeightedWordCost(unittest.TestCase):
    def test_substitution_more_expensive_than_delete_plus_insert(self):
        # reference "red", hypothesis "blue"
        # with substitution cost 3 and deletion/insertion cost 1, the minimum
        # cost alignment is one deletion plus one insertion (total cost 2)
        out = jiwer.process_words(
            "red",
            "blue",
            substitution_cost=3,
            deletion_cost=1,
            insertion_cost=1,
        )

        self.assertEqual(out.cost, 2)
        self.assertEqual(out.substitutions, 0)
        self.assertEqual(out.deletions, 1)
        self.assertEqual(out.insertions, 1)
        self.assertEqual(out.hits, 0)

        chunk_types = [chunk.type for chunk in out.alignments[0]]
        self.assertIn("delete", chunk_types)
        self.assertIn("insert", chunk_types)
        self.assertNotIn("substitute", chunk_types)

        # the classical WER interpretation is preserved on the output
        self.assertEqual(out.wer, 2.0)
        self.assertEqual(
            out.cost,
            out.substitutions * out.substitution_cost
            + out.deletions * out.deletion_cost
            + out.insertions * out.insertion_cost,
        )

    def test_default_entry_keeps_single_substitution(self):
        # with unit costs the same pair aligns as a single substitution
        out = jiwer.process_words("red", "blue")

        self.assertEqual(out.substitutions, 1)
        self.assertEqual(out.deletions, 0)
        self.assertEqual(out.insertions, 0)
        self.assertEqual(out.cost, 1)
        self.assertEqual(out.wer, 1.0)
        self.assertEqual(
            [chunk.type for chunk in out.alignments[0]], ["substitute"]
        )

        # the top-level convenience function is unchanged by default
        self.assertEqual(jiwer.wer("red", "blue"), 1.0)

    def test_weighted_wer_convenience_function(self):
        cost = jiwer.weighted_wer(
            "red",
            "blue",
            substitution_cost=3,
            deletion_cost=1,
            insertion_cost=1,
        )
        self.assertEqual(cost, 2)

        # with unit costs the weighted cost is the ordinary edit distance
        self.assertEqual(jiwer.weighted_wer("red", "blue"), 1)

    def test_wer_accepts_costs_but_keeps_rate_interpretation(self):
        # the WER itself is always a rate based on the edit counts, while the
        # weighted entry selects the alignment
        wer_value = jiwer.wer(
            "red",
            "blue",
            substitution_cost=3,
            deletion_cost=1,
            insertion_cost=1,
        )
        self.assertEqual(wer_value, 2.0)

    def test_costs_are_recorded_on_output(self):
        out = jiwer.process_words(
            "a b c",
            "a x c",
            substitution_cost=4,
            deletion_cost=2,
            insertion_cost=3,
        )
        self.assertEqual(out.substitution_cost, 4)
        self.assertEqual(out.deletion_cost, 2)
        self.assertEqual(out.insertion_cost, 3)
        self.assertEqual(
            out.cost,
            out.substitutions * 4
            + out.deletions * 2
            + out.insertions * 3,
        )

    def test_multiple_sentences_aggregate_cost(self):
        out = jiwer.process_words(
            ["red", "a b c"],
            ["blue", "a x c"],
            substitution_cost=3,
            deletion_cost=1,
            insertion_cost=1,
        )

        # both sentences prefer delete + insert over substitution
        self.assertEqual(out.substitutions, 0)
        self.assertEqual(out.deletions, 2)
        self.assertEqual(out.insertions, 2)
        self.assertEqual(out.cost, 4)

    def test_substitution_preferred_when_cheapest(self):
        # when a substitution is cheaper than deleting and inserting, the
        # alignment should contain a substitution
        out = jiwer.process_words(
            "red",
            "blue",
            substitution_cost=1,
            deletion_cost=5,
            insertion_cost=5,
        )

        self.assertEqual(out.substitutions, 1)
        self.assertEqual(out.deletions, 0)
        self.assertEqual(out.insertions, 0)
        self.assertEqual(out.cost, 1)

    def test_empty_reference_and_hypothesis(self):
        out = jiwer.process_words(
            "",
            "a b",
            substitution_cost=3,
            deletion_cost=2,
            insertion_cost=4,
        )
        self.assertEqual(out.insertions, 2)
        self.assertEqual(out.cost, 8)

        out = jiwer.process_words(
            "a b",
            "",
            substitution_cost=3,
            deletion_cost=2,
            insertion_cost=4,
        )
        self.assertEqual(out.deletions, 2)
        self.assertEqual(out.cost, 4)

        out = jiwer.process_words(
            "",
            "",
            substitution_cost=3,
            deletion_cost=2,
            insertion_cost=4,
        )
        self.assertEqual(out.cost, 0)

    def test_invalid_costs_raise(self):
        for kwargs in [
            {"substitution_cost": 0},
            {"deletion_cost": -1},
            {"insertion_cost": 0},
            {"substitution_cost": 1.5},
            {"deletion_cost": True},
            {"insertion_cost": "1"},
        ]:
            with self.assertRaises(ValueError):
                jiwer.process_words("a", "b", **kwargs)

    def test_unit_costs_match_default_alignment_counts(self):
        # explicitly passing unit costs must behave like the default entry
        default = jiwer.process_words(
            "the quick brown fox", "a slow brown fox"
        )
        explicit = jiwer.process_words(
            "the quick brown fox",
            "a slow brown fox",
            substitution_cost=1,
            deletion_cost=1,
            insertion_cost=1,
        )

        self.assertEqual(default.substitutions, explicit.substitutions)
        self.assertEqual(default.deletions, explicit.deletions)
        self.assertEqual(default.insertions, explicit.insertions)
        self.assertEqual(default.hits, explicit.hits)
        self.assertEqual(default.cost, explicit.cost)
        self.assertEqual(default.wer, explicit.wer)

    def test_visualization_reports_cost(self):
        out = jiwer.process_words(
            "red",
            "blue",
            substitution_cost=3,
            deletion_cost=1,
            insertion_cost=1,
        )
        text = jiwer.visualize_alignment(out)
        self.assertIn("cost=2", text)
        self.assertIn("substitution_cost=3", text)
        self.assertIn("deletion_cost=1", text)
        self.assertIn("insertion_cost=1", text)

        # default (unit cost) visualization keeps its original format
        default_text = jiwer.visualize_alignment(
            jiwer.process_words("red", "blue")
        )
        self.assertNotIn("cost=", default_text)


if __name__ == "__main__":
    unittest.main()
