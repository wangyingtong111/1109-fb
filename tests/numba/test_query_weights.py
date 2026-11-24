import unittest

import numpy as np

import bm25s


@unittest.skipUnless(bm25s.NUMBA_AVAILABLE, "Numba is not installed")
class TestQueryWeightsBackendParity(unittest.TestCase):
    def make_retriever(self, method):
        corpus = [
            ["apple", "banana"],
            ["apple", "apple"],
            ["banana", "banana"],
            ["cherry", "date"],
        ]
        retriever = bm25s.BM25(method=method, delta=1, backend="numpy")
        retriever.index(corpus, show_progress=False)
        return retriever

    def test_weighted_scores_match_across_backends(self):
        queries = [
            ["apple", "banana", "ghost", "apple"],
            ["banana", "ghost"],
        ]
        query_weights = [
            [0.5, 2, 9, 1.5],
            [3, 9],
        ]

        for method in ("bm25l", "bm25+"):
            retriever = self.make_retriever(method)
            expected = np.array(
                [
                    retriever.get_scores(query, query_weights=weights)
                    for query, weights in zip(queries, query_weights)
                ]
            )
            for backend, n_threads in (
                ("numpy", 0),
                ("numba", 1),
                ("numba", 2),
            ):
                with self.subTest(
                    method=method, backend=backend, n_threads=n_threads
                ):
                    retriever.backend = backend
                    result = retriever.retrieve(
                        queries,
                        query_weights=query_weights,
                        k=4,
                        n_threads=n_threads,
                        backend_selection=backend,
                        show_progress=False,
                    )
                    # Compare by document ID so tied scores need no tie order.
                    scores_by_id = np.take_along_axis(
                        result.scores,
                        np.argsort(result.documents, axis=1),
                        axis=1,
                    )
                    np.testing.assert_allclose(
                        scores_by_id, expected, rtol=1e-6
                    )

    def test_default_weights_keep_backends_aligned(self):
        retriever = self.make_retriever("bm25+")
        queries = [["apple", "banana", "apple"]]
        numpy_result = retriever.retrieve(
            queries,
            k=4,
            backend_selection="numpy",
            show_progress=False,
        )
        for n_threads in (1, 2):
            with self.subTest(n_threads=n_threads):
                retriever.backend = "numba"
                numba_result = retriever.retrieve(
                    queries,
                    k=4,
                    n_threads=n_threads,
                    backend_selection="numba",
                    show_progress=False,
                )
                np.testing.assert_array_equal(
                    numpy_result.documents, numba_result.documents
                )
                np.testing.assert_allclose(
                    numpy_result.scores, numba_result.scores, rtol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
