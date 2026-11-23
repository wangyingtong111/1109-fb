import unittest

import numpy as np

import bm25s


@unittest.skipUnless(bm25s.NUMBA_AVAILABLE, "Numba is not installed")
class TestCandidateIdsBackendParity(unittest.TestCase):
    def setUp(self):
        self.corpus = [["alpha"], ["alpha"], ["beta"], ["gamma"]]

    def make_retriever(self, method):
        retriever = bm25s.BM25(method=method, delta=1, backend="numpy")
        retriever.index(self.corpus, show_progress=False)
        return retriever

    def test_candidate_set_matches_across_backends(self):
        for method in ("lucene", "bm25l", "bm25+"):
            retriever = self.make_retriever(method)
            candidates = np.array([0, 2], dtype=np.int32)
            expected_docs = np.array([[0, 2]])
            expected_scores = retriever.get_scores(["alpha"])[candidates]

            for backend, n_threads in (
                ("numpy", 0),
                ("numba", 1),
                ("numba", 2),
            ):
                with self.subTest(method=method, backend=backend, n_threads=n_threads):
                    retriever.backend = backend
                    result = retriever.retrieve(
                        [["alpha"]],
                        k=3,
                        candidate_ids=candidates,
                        n_threads=n_threads,
                        backend_selection=backend,
                        show_progress=False,
                    )
                    np.testing.assert_array_equal(result.documents, expected_docs)
                    np.testing.assert_allclose(
                        result.scores, expected_scores.reshape(1, -1), rtol=1e-6
                    )

    def test_empty_and_single_candidate_sets_across_backends(self):
        retriever = self.make_retriever("lucene")
        cases = (
            np.array([], dtype=np.int32),
            np.array([2], dtype=np.int32),
        )
        for candidates in cases:
            expected_len = len(candidates)
            for backend, n_threads in (("numpy", 0), ("numba", 1)):
                with self.subTest(
                    candidates=candidates.tolist(), backend=backend
                ):
                    retriever.backend = backend
                    result = retriever.retrieve(
                        [["alpha"]],
                        k=3,
                        candidate_ids=candidates,
                        n_threads=n_threads,
                        backend_selection=backend,
                        show_progress=False,
                    )
                    self.assertEqual(result.documents.shape, (1, expected_len))
                    self.assertEqual(result.scores.shape, (1, expected_len))
                    if expected_len:
                        np.testing.assert_array_equal(
                            result.documents, candidates.reshape(1, -1)
                        )


if __name__ == "__main__":
    unittest.main()
