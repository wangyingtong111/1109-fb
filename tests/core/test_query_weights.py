import unittest

import numpy as np

import bm25s


class TestQueryWeights(unittest.TestCase):
    """
    Term-level weights applied at query time. The query may keep duplicate terms
    and may contain terms that are not in the vocabulary. For BM25+/BM25L, each
    weighted term scales its *complete* contribution, including the
    non-occurrence base score received by documents that do not contain it.
    """

    def make_retriever(self, method="bm25+", delta=1):
        # The last document contains neither "apple" nor "banana".
        corpus = [
            ["apple", "banana"],
            ["apple", "apple"],
            ["banana", "banana"],
            ["cherry", "date"],
        ]
        retriever = bm25s.BM25(method=method, delta=delta, backend="numpy")
        retriever.index(corpus, show_progress=False)
        return retriever

    def test_weighted_query_is_sum_of_weighted_full_term_scores(self):
        # apple appears twice with weights 0.5 and 1.5 -> total weight 2
        # banana appears once with weight 2 -> total weight 2
        # ghost is not indexed and must be ignored together with its weight 9
        query = ["apple", "banana", "ghost", "apple"]
        weights = [0.5, 2, 9, 1.5]

        for method in ("bm25+", "bm25l"):
            with self.subTest(method=method):
                retriever = self.make_retriever(method=method)
                weighted = retriever.get_scores(query, query_weights=weights)
                expected = (
                    2 * retriever.get_scores(["apple"])
                    + 2 * retriever.get_scores(["banana"])
                )
                np.testing.assert_allclose(weighted, expected, rtol=1e-6)

    def test_empty_match_document_keeps_weighted_nonoccurrence_base(self):
        retriever = self.make_retriever(method="bm25+")
        query = ["apple", "banana", "ghost", "apple"]
        weights = [0.5, 2, 9, 1.5]

        weighted = retriever.get_scores(query, query_weights=weights)
        expected = (
            2 * retriever.get_scores(["apple"])
            + 2 * retriever.get_scores(["banana"])
        )

        # Document 3 contains neither query term, so its score is entirely made
        # of weighted non-occurrence base scores; the relation must still hold.
        self.assertGreater(weighted[3], 0)
        np.testing.assert_allclose(weighted[3], expected[3], rtol=1e-6)
        np.testing.assert_allclose(weighted, expected, rtol=1e-6)

    def test_unindexed_term_and_its_weight_are_dropped(self):
        retriever = self.make_retriever()

        ids, aligned_weights = retriever.get_tokens_ids_and_weights(
            ["apple", "ghost", "apple"], [0.5, 9, 1.5]
        )
        # "ghost" must not produce a token id nor a surviving weight.
        self.assertEqual(len(ids), 2)
        np.testing.assert_allclose(aligned_weights, [0.5, 1.5])

        # A query made only of out-of-vocabulary terms scores zero everywhere.
        ghost_scores = retriever.get_scores(["ghost"], query_weights=[9])
        np.testing.assert_array_equal(
            ghost_scores, np.zeros(4, dtype=ghost_scores.dtype)
        )

    def test_default_weights_reproduce_unweighted_retrieval(self):
        retriever = self.make_retriever()
        query = ["apple", "banana", "ghost", "apple"]

        unweighted = retriever.get_scores(query)
        # Duplicate query terms still accumulate: apple twice, banana once.
        expected = (
            2 * retriever.get_scores(["apple"])
            + retriever.get_scores(["banana"])
        )
        np.testing.assert_allclose(unweighted, expected, rtol=1e-6)

        # Explicit unit weights are equivalent to passing nothing.
        explicit = retriever.get_scores(query, query_weights=[1, 1, 1, 1])
        np.testing.assert_allclose(explicit, unweighted, rtol=1e-6)

    def test_explicit_weights_match_duplicated_unweighted_terms(self):
        retriever = self.make_retriever()
        # A weight of 2 on a single apple equals two unweighted apple occurrences.
        weighted = retriever.get_scores(["apple"], query_weights=[2])
        duplicated = retriever.get_scores(["apple", "apple"])
        np.testing.assert_allclose(weighted, duplicated, rtol=1e-6)

    def test_query_weights_work_for_every_method(self):
        query = ["apple", "banana", "ghost", "apple"]
        weights = [0.5, 2, 9, 1.5]

        for method in ("lucene", "robertson", "atire", "bm25l", "bm25+"):
            with self.subTest(method=method):
                retriever = self.make_retriever(method=method)
                weighted = retriever.get_scores(query, query_weights=weights)
                expected = (
                    2 * retriever.get_scores(["apple"])
                    + 2 * retriever.get_scores(["banana"])
                )
                np.testing.assert_allclose(weighted, expected, rtol=1e-6)

    def test_token_id_path_uses_aligned_weights(self):
        retriever = self.make_retriever()
        query = ["apple", "banana", "ghost", "apple"]
        weights = [0.5, 2, 9, 1.5]

        ids, aligned_weights = retriever.get_tokens_ids_and_weights(
            query, weights
        )
        via_ids = retriever.get_scores_from_ids(
            ids, query_weights=aligned_weights
        )
        via_tokens = retriever.get_scores(query, query_weights=weights)
        np.testing.assert_allclose(via_ids, via_tokens, rtol=1e-6)

    def test_document_weight_mask_composes_with_query_weights(self):
        retriever = self.make_retriever()
        query = ["apple", "banana", "ghost", "apple"]
        weights = [0.5, 2, 9, 1.5]
        mask = np.array([1.0, 0.5, 1.0, 0.0], dtype=np.float32)

        expected = retriever.get_scores(query, query_weights=weights) * mask
        combined = retriever.get_scores(
            query, query_weights=weights, weight_mask=mask
        )
        np.testing.assert_allclose(combined, expected, rtol=1e-6)

    def test_retrieve_applies_query_weights_before_top_k(self):
        retriever = self.make_retriever()
        query = ["apple", "banana", "ghost", "apple"]
        weights = [0.5, 2, 9, 1.5]
        expected = retriever.get_scores(query, query_weights=weights)

        result = retriever.retrieve(
            [query],
            query_weights=[weights],
            k=4,
            show_progress=False,
            backend_selection="numpy",
        )
        # Compare by document id so tied score order does not matter.
        order = np.argsort(result.documents[0])
        np.testing.assert_allclose(
            result.scores[0][order], expected[result.documents[0][order]], rtol=1e-6
        )

        # The highest-scoring document must come first when results are sorted.
        self.assertEqual(result.documents[0][0], int(np.argmax(expected)))

    def test_invalid_weight_lengths_raise(self):
        retriever = self.make_retriever()
        query = ["apple", "banana", "ghost", "apple"]

        with self.assertRaises(ValueError):
            retriever.get_scores(query, query_weights=[1, 2, 3])

        with self.assertRaises(ValueError):
            retriever.get_scores_from_ids(
                retriever.get_tokens_ids(query), query_weights=[1, 2]
            )

        with self.assertRaises(ValueError):
            retriever.retrieve(
                [query], query_weights=[[1, 2, 3]], k=1, show_progress=False
            )

        with self.assertRaises(ValueError):
            retriever.retrieve(
                [query, ["apple"]],
                query_weights=[[1, 1, 1, 1]],
                k=1,
                show_progress=False,
            )

    def test_weights_do_not_mutate_index_or_inputs(self):
        retriever = self.make_retriever()
        query = ["apple", "banana", "ghost", "apple"]
        weights = np.array([0.5, 2, 9, 1.5], dtype=np.float32)
        original_weights = weights.copy()
        before = retriever.get_scores(["apple"])

        retriever.get_scores(query, query_weights=weights)

        np.testing.assert_array_equal(weights, original_weights)
        np.testing.assert_allclose(
            retriever.get_scores(["apple"]), before, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
