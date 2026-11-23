"""
Tests for indexing corpora whose integer token IDs are caller-managed, fixed
external IDs that are neither zero-based nor contiguous (for example hand-assigned
IDs such as ``17`` and ``4001``).

With ``BM25.index(..., external_ids=True)``:

* the external IDs stay queryable as the keys of ``vocab_dict``;
* the index internally only allocates one compact column per distinct token, so the
  number of sparse columns does not grow with the magnitude of an external ID;
* queries keep using the original external IDs;
* the same mapping is restored after a save/load cycle.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import bm25s


class TestExternalTokenIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Token 17 appears in docs 0 and 2; token 4001 appears in docs 1 and 2.
        cls.corpus = [[17, 17], [4001], [17, 4001]]
        cls.num_distinct_tokens = 2

    def _build(self, create_empty_token=True, **kwargs):
        retriever = bm25s.BM25()
        retriever.index(
            self.corpus,
            external_ids=True,
            create_empty_token=create_empty_token,
            show_progress=False,
            **kwargs,
        )
        return retriever

    def test_vocab_maps_external_ids_to_compact_columns(self):
        retriever = self._build(create_empty_token=False)
        # The two external IDs map bijectively to compact columns 0 and 1.
        self.assertEqual(
            set(retriever.vocab_dict.keys()), {17, 4001}
        )
        self.assertEqual(
            sorted(retriever.vocab_dict.values()), [0, 1]
        )
        self.assertEqual(retriever.external_token_ids_set, {17, 4001})
        self.assertTrue(retriever._external_ids_enabled)

    def test_allocated_columns_are_compact(self):
        # One column per distinct token, no extra columns spanning up to 4001.
        retriever = self._build(create_empty_token=False)
        n_cols = len(retriever.scores["indptr"]) - 1
        self.assertEqual(n_cols, self.num_distinct_tokens)
        self.assertLess(n_cols, 4001)

        # With the empty token reserved, exactly one more column is allocated.
        retriever_empty = self._build(create_empty_token=True)
        n_cols_empty = len(retriever_empty.scores["indptr"]) - 1
        self.assertEqual(n_cols_empty, self.num_distinct_tokens + 1)

    def test_retrieve_external_id_17_only_matches_docs_0_and_2(self):
        retriever = self._build()
        results = retriever.retrieve([[17]], k=2, show_progress=False)
        self.assertEqual(sorted(results.documents[0].tolist()), [0, 2])
        # Both retrieved documents must have a positive score.
        self.assertTrue((results.scores[0] > 0).all())

    def test_retrieve_external_id_4001_only_matches_docs_1_and_2(self):
        retriever = self._build()
        results = retriever.retrieve([[4001]], k=2, show_progress=False)
        self.assertEqual(sorted(results.documents[0].tolist()), [1, 2])
        self.assertTrue((results.scores[0] > 0).all())

    def test_retrieve_with_both_external_ids(self):
        retriever = self._build()
        results = retriever.retrieve([[17, 4001]], k=3, show_progress=False)
        # Document 2 contains both tokens and must rank first.
        self.assertEqual(results.documents[0, 0], 2)
        self.assertEqual(sorted(results.documents[0].tolist()), [0, 1, 2])

    def test_get_scores_uses_external_ids(self):
        retriever = self._build(create_empty_token=False)
        scores_17 = retriever.get_scores([17])
        scores_4001 = retriever.get_scores([4001])
        self.assertEqual(len(scores_17), len(self.corpus))
        self.assertGreater(scores_17[0], 0)
        self.assertEqual(scores_17[1], 0)
        self.assertGreater(scores_17[2], 0)
        self.assertEqual(scores_4001[0], 0)
        self.assertGreater(scores_4001[1], 0)
        self.assertGreater(scores_4001[2], 0)

    def test_unknown_external_id_is_filtered(self):
        retriever = self._build()
        # The OOV-only query falls back to the reserved empty-token column, giving
        # every document a zero score instead of indexing a huge column.
        results = retriever.retrieve([[999999]], k=2, show_progress=False)
        self.assertTrue((results.scores[0] == 0).all())

    def test_unknown_external_id_without_empty_token_raises(self):
        retriever = self._build(create_empty_token=False)
        with self.assertRaises(ValueError) as ctx:
            retriever.retrieve([[999999]], k=2, show_progress=False)
        self.assertIn("does not contain any tokens", str(ctx.exception))

    def test_mixed_known_and_unknown_external_ids(self):
        retriever = self._build(create_empty_token=False)
        results = retriever.retrieve([[17, 999999]], k=2, show_progress=False)
        self.assertEqual(sorted(results.documents[0].tolist()), [0, 2])

    def test_save_and_load_preserves_external_mapping(self):
        retriever = self._build()
        save_dir = tempfile.mkdtemp()
        try:
            retriever.save(save_dir, show_progress=False)

            # The on-disk vocab must serialize integer external IDs as JSON strings.
            with open(os.path.join(save_dir, "vocab.index.json"), "r") as f:
                on_disk_vocab = json.load(f)
            self.assertTrue(all(isinstance(k, str) for k in on_disk_vocab))

            loaded = bm25s.BM25.load(save_dir)
            # Keys are restored to integers and the mapping is identical.
            self.assertEqual(loaded.vocab_dict, retriever.vocab_dict)
            self.assertTrue(
                all(isinstance(k, int) for k in loaded.vocab_dict)
            )
            self.assertTrue(loaded._external_ids_enabled)
            self.assertEqual(loaded.external_token_ids_set, {0, 17, 4001})
            self.assertEqual(loaded.empty_external_token_id, 0)

            # Queries still use the same external IDs after reload.
            r17 = loaded.retrieve([[17]], k=2, show_progress=False)
            r4001 = loaded.retrieve([[4001]], k=2, show_progress=False)
            self.assertEqual(sorted(r17.documents[0].tolist()), [0, 2])
            self.assertEqual(sorted(r4001.documents[0].tolist()), [1, 2])

            # The compact column count is also preserved.
            self.assertEqual(
                len(loaded.scores["indptr"]) - 1,
                len(retriever.scores["indptr"]) - 1,
            )
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)

    def test_default_mode_keeps_legacy_integer_behavior(self):
        # Without external_ids=True, integers keep being treated as internal column
        # indices; the external-ID bookkeeping must stay disabled.
        contiguous = [[0, 0], [1], [0, 1]]
        retriever = bm25s.BM25()
        retriever.index(contiguous, create_empty_token=False, show_progress=False)
        self.assertFalse(retriever._external_ids_enabled)
        self.assertIsNone(retriever.external_token_ids_set)
        results = retriever.retrieve([[0]], k=2, show_progress=False)
        self.assertEqual(sorted(results.documents[0].tolist()), [0, 2])

    def test_string_corpus_unaffected_by_external_mode(self):
        # A string corpus ignores external_ids and keeps string vocab keys.
        corpus = [["cat", "cat"], ["dog"], ["cat", "dog"]]
        retriever = bm25s.BM25()
        retriever.index(
            corpus, external_ids=True, create_empty_token=False, show_progress=False
        )
        self.assertFalse(retriever._external_ids_enabled)
        self.assertTrue(all(isinstance(k, str) for k in retriever.vocab_dict))
        results = retriever.retrieve([["cat"]], k=2, show_progress=False)
        self.assertEqual(sorted(results.documents[0].tolist()), [0, 2])

    def test_external_ids_with_large_gap_and_duplicates(self):
        # A wider gap and repeated tokens must still produce a compact index.
        corpus = [[5, 5, 900000], [900000], [5]]
        retriever = bm25s.BM25()
        retriever.index(
            corpus, external_ids=True, create_empty_token=False, show_progress=False
        )
        self.assertEqual(
            set(retriever.vocab_dict.keys()), {5, 900000}
        )
        self.assertEqual(len(retriever.scores["indptr"]) - 1, 2)
        results = retriever.retrieve([[900000]], k=2, show_progress=False)
        self.assertEqual(sorted(results.documents[0].tolist()), [0, 1])

    def test_inverted_postings_per_scoring_method(self):
        # The sparse columns must point at exactly the documents that contain each
        # external token. This is checked structurally (via the CSC indptr/indices) so it
        # stays valid even for IDF variants whose scores truncate to zero.
        for method in ("lucene", "robertson", "atire", "bm25l", "bm25+"):
            retriever = bm25s.BM25(method=method)
            retriever.index(
                self.corpus,
                external_ids=True,
                create_empty_token=False,
                show_progress=False,
            )
            indptr = retriever.scores["indptr"]
            indices = retriever.scores["indices"]

            def column_docs(column):
                return sorted(
                    indices[indptr[column] : indptr[column + 1]].tolist()
                )

            self.assertEqual(
                column_docs(retriever.vocab_dict[17]), [0, 2], method
            )
            self.assertEqual(
                column_docs(retriever.vocab_dict[4001]), [1, 2], method
            )

    @unittest.skipUnless(
        __import__("bm25s").NUMBA_AVAILABLE, "numba is not installed"
    )
    def test_numba_backend_translates_external_ids(self):
        retriever = bm25s.BM25(backend="numba")
        retriever.index(
            self.corpus, external_ids=True, show_progress=False
        )
        r17 = retriever.retrieve([[17]], k=2, show_progress=False)
        r4001 = retriever.retrieve([[4001]], k=2, show_progress=False)
        self.assertEqual(sorted(r17.documents[0].tolist()), [0, 2])
        self.assertEqual(sorted(r4001.documents[0].tolist()), [1, 2])
        # The numba backend pre-translates external IDs, but internal state still uses
        # compact columns, so a reload behaves identically.
        save_dir = tempfile.mkdtemp()
        try:
            retriever.save(save_dir, show_progress=False)
            loaded = bm25s.BM25.load(save_dir)
            self.assertEqual(
                sorted(
                    loaded.retrieve([[17]], k=2, show_progress=False)
                    .documents[0]
                    .tolist()
                ),
                [0, 2],
            )
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
