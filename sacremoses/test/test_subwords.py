# -*- coding: utf-8 -*-

"""
Tests for subwords.py / SubwordTokenizer (BPE learning).
"""

import os
import shutil
import tempfile
import unittest
from collections import Counter

from sacremoses.subwords import SubwordTokenizer

EOW = ""  # end-of-word marker appended to the last symbol


def _recompute(vocab):
    """Full recomputation of pair stats/indices from the current vocabulary."""
    stats = Counter()
    indices = {}
    for i, (word, freq) in enumerate(vocab):
        for first, second in zip(word, word[1:]):
            stats[first, second] += freq
            indices.setdefault((first, second), Counter())[i] += 1
    return stats, indices


class TestSubwordTokenizer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _train(self, text):
        path = os.path.join(self.tmpdir, "corpus.txt")
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(text)
        return SubwordTokenizer(path)

    def _merge(self, tokenizer, pair):
        changes = tokenizer.replace_pair(pair)
        tokenizer.update_pair_statistics(pair, changes)

    def _assert_consistent(self, tokenizer):
        """Incremental stats/indices must equal a full recompute."""
        stats, indices = _recompute(tokenizer.vocab)
        self.assertEqual({k: v for k, v in tokenizer.stats.items() if v}, dict(stats))
        incremental = {
            pair: {i: c for i, c in row.items() if c}
            for pair, row in tokenizer.indices.items()
        }
        incremental = {pair: row for pair, row in incremental.items() if row}
        recomputed = {pair: dict(row) for pair, row in indices.items()}
        self.assertEqual(incremental, recomputed)

    def test_plain_merge_updates_counts(self):
        tokenizer = self._train("abc abc")
        self._merge(tokenizer, ("a", "b"))
        # "ab" is followed by the end-of-word "c", twice, in a single row.
        self.assertEqual(tokenizer.stats[("ab", "c" + EOW)], 2)
        self.assertEqual(tokenizer.indices[("ab", "c" + EOW)], Counter({0: 1}))
        self._assert_consistent(tokenizer)

    def test_backslash_is_literal_in_merged_symbol(self):
        tokenizer = self._train("a\\b a\\b")
        self._merge(tokenizer, ("a", "\\"))
        (word, freq), = tokenizer.vocab
        # One real backslash, not an escaped two-character "\\\\".
        self.assertEqual(word, ("a\\", "b" + EOW))
        self.assertEqual(freq, 2)
        self.assertEqual(tokenizer.stats[("a\\", "b" + EOW)], 2)
        self.assertEqual(tokenizer.indices[("a\\", "b" + EOW)], Counter({0: 1}))
        self._assert_consistent(tokenizer)

    def test_backslash_merge_continues_on_real_symbol(self):
        tokenizer = self._train("a\\b a\\b a\\\\b")
        self._merge(tokenizer, ("a", "\\"))
        self.assertEqual(
            sorted(word for word, _ in tokenizer.vocab),
            sorted([("a\\", "b" + EOW), ("a\\", "\\", "b" + EOW)]),
        )
        self._assert_consistent(tokenizer)

        # Continue on the real symbol: "a\\" + "b<EOW>" -> "a\\b<EOW>".
        self._merge(tokenizer, ("a\\", "b" + EOW))
        self.assertEqual(
            sorted(word for word, _ in tokenizer.vocab),
            sorted([("a\\b" + EOW,), ("a\\", "\\", "b" + EOW)]),
        )
        # The two-backslash word keeps its own pair count, index row and
        # vocabulary row; it is not swallowed by the previous merges.
        self.assertEqual(tokenizer.stats[("\\", "b" + EOW)], 1)
        self.assertEqual(tokenizer.stats[("a\\", "\\")], 1)
        self.assertEqual(tokenizer.indices[("a\\", "\\")], Counter({1: 1}))
        self.assertEqual(len(tokenizer.vocab), 2)
        self._assert_consistent(tokenizer)


if __name__ == "__main__":
    unittest.main()
