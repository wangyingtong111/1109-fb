import unittest

import numpy as np

import bm25s


class TestCandidateIds(unittest.TestCase):
    def setUp(self):
        # Four documents containing alpha, alpha, beta, gamma in that order.
        self.corpus = [["alpha"], ["alpha"], ["beta"], ["gamma"]]
        self.retriever = bm25s.BM25(backend="numpy")
        self.retriever.index(self.corpus, show_progress=False)

    def retrieve(self, **kwargs):
        kwargs.setdefault("backend_selection", "numpy")
        kwargs.setdefault("show_progress", False)
        return self.retriever.retrieve([["alpha"]], **kwargs)

    def test_candidate_set_restricts_recall_without_padding(self):
        # Only docs 0 and 2 are allowed; the query is alpha with k=3.
        # Doc 0 matches alpha, while doc 2 contains beta and is a legitimate
        # zero-score candidate. Excluded docs 1 and 3 must never pad the result.
        result = self.retrieve(k=3, candidate_ids=np.array([0, 2]))

        np.testing.assert_array_equal(result.documents, np.array([[0, 2]]))
        self.assertGreater(result.scores[0, 0], 0.0)
        self.assertEqual(result.scores[0, 1], 0.0)
        self.assertEqual(result.documents.shape, (1, 2))

    def test_candidate_ids_accepts_python_list(self):
        result = self.retrieve(k=3, candidate_ids=[0, 2])
        np.testing.assert_array_equal(result.documents, np.array([[0, 2]]))

    def test_empty_candidate_set_returns_empty_result(self):
        # Nothing allowed: the result is empty, with no excluded docs filling k.
        result = self.retrieve(k=3, candidate_ids=np.array([], dtype=np.int32))

        self.assertEqual(result.documents.shape, (1, 0))
        self.assertEqual(result.scores.shape, (1, 0))
        np.testing.assert_array_equal(result.documents, np.empty((1, 0), dtype=np.int32))
        np.testing.assert_array_equal(result.scores, np.empty((1, 0), dtype=np.float32))

    def test_single_candidate_is_not_padded(self):
        # Even when the lone candidate has no query-term hit, it is the only result.
        result = self.retrieve(k=3, candidate_ids=np.array([2]))

        np.testing.assert_array_equal(result.documents, np.array([[2]]))
        np.testing.assert_array_equal(result.scores, np.array([[0.0]]))

    def test_no_candidate_restriction_preserves_legacy_top_k(self):
        # The old call without a hard restriction still returns three candidates.
        result = self.retrieve(k=3)

        self.assertEqual(result.documents.shape, (1, 3))
        # alpha appears in docs 0 and 1, so both must be present in the top-3.
        self.assertEqual(set(result.documents[0].tolist()) & {0, 1}, {0, 1})

    def test_candidate_set_is_separate_from_weight_mask(self):
        # weight_mask scales the complete score vector; candidate_ids hard-restricts
        # recall. Together they still never return an excluded document.
        mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        unweighted = self.retriever.get_scores(["alpha"])
        result = self.retrieve(
            k=3, candidate_ids=np.array([0, 2]), weight_mask=mask
        )

        np.testing.assert_array_equal(result.documents, np.array([[0, 2]]))
        np.testing.assert_allclose(
            result.scores,
            (unweighted * mask)[np.array([0, 2])].reshape(1, -1),
            rtol=1e-6,
        )

    def test_zero_weight_mask_alone_still_pads_but_candidate_set_does_not(self):
        # Documents excluded only by a zero weight still pad the hard top-k,
        # demonstrating why a separate candidate set is needed.
        mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        masked = self.retrieve(k=3, weight_mask=mask)
        self.assertEqual(masked.documents.shape, (1, 3))

        restricted = self.retrieve(k=3, candidate_ids=np.array([0, 2]))
        self.assertEqual(restricted.documents.shape, (1, 2))

    def test_candidate_set_applies_per_query_and_supports_multi_query(self):
        result = self.retriever.retrieve(
            [["alpha"], ["beta"], ["gamma"]],
            k=3,
            candidate_ids=np.array([0, 2]),
            backend_selection="numpy",
            show_progress=False,
        )

        self.assertEqual(result.documents.shape, (3, 2))
        for row in result.documents:
            self.assertEqual(set(row.tolist()), {0, 2})

    def test_corpus_argument_maps_candidate_document_ids(self):
        labels = np.array(["d0", "d1", "d2", "d3"])
        documents = self.retriever.retrieve(
            [["alpha"]],
            k=3,
            candidate_ids=np.array([0, 2]),
            corpus=labels,
            return_as="documents",
            backend_selection="numpy",
            show_progress=False,
        )
        np.testing.assert_array_equal(documents, np.array([["d0", "d2"]]))

    def test_unmatched_query_keeps_zero_score_candidates(self):
        result = self.retriever.retrieve(
            [["nomatch"]],
            k=3,
            candidate_ids=np.array([3, 2]),
            backend_selection="numpy",
            show_progress=False,
        )

        # Both candidates stay recallable as zero-score docs; their tie order is
        # not guaranteed by the top-k selection.
        self.assertEqual(set(result.documents[0].tolist()), {2, 3})
        np.testing.assert_array_equal(result.scores, np.zeros((1, 2)))

    def test_invalid_candidate_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "1-D"):
            self.retrieve(k=2, candidate_ids=np.array([[0, 1]]))
        with self.assertRaisesRegex(ValueError, "valid document IDs"):
            self.retrieve(k=2, candidate_ids=np.array([0, 4]))
        with self.assertRaisesRegex(ValueError, "integer document IDs"):
            self.retrieve(k=2, candidate_ids=np.array([0.0, 2.0]))

    def test_k_larger_than_corpus_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "larger than the number"):
            self.retrieve(k=10, candidate_ids=np.array([0, 2]))


if __name__ == "__main__":
    unittest.main()
