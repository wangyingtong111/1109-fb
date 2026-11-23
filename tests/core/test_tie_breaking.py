import shutil
import tempfile
import unittest

import numpy as np

import bm25s
from bm25s.selection import topk as topk_np


class TestStableTieBreaking(unittest.TestCase):
    """Tied scores must be broken by the original document index (lower index
    first), with the score remaining the primary criterion. This makes the
    first screen of results compatible across different values of k on the
    same index, and stable across save/load."""

    def setUp(self):
        # five completely identical short documents -> identical scores
        self.corpus = ["hello world"] * 5
        self.corpus_tokens = bm25s.tokenize(self.corpus)
        self.query_tokens = bm25s.tokenize(["hello"])

        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokens)

        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _doc_ids(self, retriever, k):
        results = retriever.retrieve(
            self.query_tokens, k=k, return_as="tuple", show_progress=False
        )
        return results.documents[0].tolist(), results.scores[0].tolist()

    def test_k2_and_k3_are_prefix_compatible(self):
        ids_k2, scores_k2 = self._doc_ids(self.retriever, k=2)
        ids_k3, scores_k3 = self._doc_ids(self.retriever, k=3)

        self.assertEqual(ids_k2, [0, 1])
        self.assertEqual(ids_k3, [0, 1, 2])
        # the k=2 first screen is a prefix of the k=3 one
        self.assertEqual(ids_k3[:2], ids_k2)
        # all five documents tie on score
        self.assertTrue(np.allclose(scores_k2, scores_k2[0]))
        self.assertTrue(np.allclose(scores_k3, scores_k3[0]))

    def test_results_stable_after_save_and_reload(self):
        ids_k2_before, _ = self._doc_ids(self.retriever, k=2)
        ids_k3_before, _ = self._doc_ids(self.retriever, k=3)

        self.retriever.save(self.tmpdir)
        reloaded = bm25s.BM25.load(self.tmpdir, load_corpus=False)

        ids_k2_after, _ = self._doc_ids(reloaded, k=2)
        ids_k3_after, _ = self._doc_ids(reloaded, k=3)

        self.assertEqual(ids_k2_after, [0, 1])
        self.assertEqual(ids_k3_after, [0, 1, 2])
        self.assertEqual(ids_k2_after, ids_k2_before)
        self.assertEqual(ids_k3_after, ids_k3_before)

    def test_higher_score_ranks_before_all_tied_docs(self):
        # add a document mentioning the query term more often: it must rank
        # ahead of every tied document even though it has the highest index
        corpus = self.corpus + ["hello hello hello hello"]
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus))

        ids_k3, scores_k3 = self._doc_ids(retriever, k=3)
        self.assertEqual(ids_k3, [5, 0, 1])
        self.assertGreater(scores_k3[0], scores_k3[1])

        ids_all, scores_all = self._doc_ids(retriever, k=6)
        self.assertEqual(ids_all, [5, 0, 1, 2, 3, 4])
        # scores are non-increasing and the tied block keeps index order
        self.assertTrue(all(a >= b for a, b in zip(scores_all, scores_all[1:])))


class TestTopkTieBreakingSelection(unittest.TestCase):
    """The numpy selection itself must break ties by document index and stay
    consistent across k, without fully sorting the whole score vector."""

    def test_identical_scores_select_lowest_indices(self):
        scores = np.ones(5, dtype=np.float32)
        for k, expected in ((1, [0]), (2, [0, 1]), (3, [0, 1, 2]), (5, [0, 1, 2, 3, 4])):
            sel_scores, sel_ind = topk_np(scores, k=k, backend="numpy", sorted=True)
            self.assertEqual(sel_ind.tolist(), expected)
            self.assertEqual(sel_scores.tolist(), [1.0] * k)

    def test_mixed_scores_tie_breaking(self):
        scores = np.array([1.0, 3.0, 1.0, 2.0, 1.0], dtype=np.float32)
        sel_scores, sel_ind = topk_np(scores, k=3, backend="numpy", sorted=True)
        self.assertEqual(sel_ind.tolist(), [1, 3, 0])
        self.assertEqual(sel_scores.tolist(), [3.0, 2.0, 1.0])

    def test_unsorted_selection_is_deterministic_too(self):
        scores = np.ones(10, dtype=np.float32)
        _, ind_a = topk_np(scores, k=4, backend="numpy", sorted=False)
        _, ind_b = topk_np(scores, k=4, backend="numpy", sorted=False)
        self.assertEqual(sorted(ind_a.tolist()), [0, 1, 2, 3])
        self.assertEqual(ind_a.tolist(), ind_b.tolist())


if __name__ == "__main__":
    unittest.main()
