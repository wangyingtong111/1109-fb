# -*- coding: utf-8 -*-

"""
Tests for sacremoses.subwords.

The merge loop used to stop after a single merge: it zeroed the merged pair's
own count and then compared that zero against ``min_freq``, returning early
even when a freshly created pair was still above the threshold.  These tests
pin the fixed contract:

* a merge that creates a new high-frequency pair feeds the next merge;
* ties are broken deterministically by the pair, not by dict order;
* ``jump`` is a prefetch hint only -- every value yields the same sequence.
"""

import os
import random
import shutil
import tempfile
import unittest
from collections import Counter

from sacremoses.subwords import SubwordTokenizer

#: End-of-word marker appended to the final symbol of every vocabulary entry.
EOW = "\ue000"


def eow(word):
    """A single-symbol vocabulary entry for ``word``, end-of-word included."""
    return (word + EOW,)


class SubwordLearnTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.count = 0

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _tokenizer(self, text):
        self.count += 1
        path = os.path.join(self.tmpdir, "corpus-%d.txt" % self.count)
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(text)
        return SubwordTokenizer(path)

    def _sorted_vocab(self, tokenizer):
        return sorted(tokenizer.vocab, key=lambda wf: (-wf[1], wf[0]))

    def test_second_merge_happens_after_first(self):
        """abc x3, two merge slots, min_freq=2 -> the whole word merges."""
        tokenizer = self._tokenizer("abc abc abc ")
        tokenizer.learn(2, min_freq=2)
        self.assertEqual(tokenizer.vocab, [(eow("abc"), 3)])

    def test_stops_early_when_nothing_reaches_min_freq(self):
        tokenizer = self._tokenizer("ab ab ")
        tokenizer.learn(5, min_freq=3)
        self.assertEqual(tokenizer.vocab, [(("a", "b" + EOW), 2)])

    def test_respects_num_symbols(self):
        tokenizer = self._tokenizer("abc abc abc ")
        tokenizer.learn(1, min_freq=2)
        # ('a','b') and ('b','c<EOW>') tie at 3; like upstream subword-nmt the
        # larger pair under the deterministic tie-break wins, so 'bc' merges.
        self.assertEqual(tokenizer.vocab, [(("a", "bc" + EOW), 3)])

    def test_zero_symbols_is_a_noop(self):
        tokenizer = self._tokenizer("abc abc abc ")
        before = list(tokenizer.vocab)
        tokenizer.learn(0, min_freq=2)
        self.assertEqual(tokenizer.vocab, before)

    def test_tie_break_is_deterministic(self):
        """Equal-frequency pairs merge in a fixed order, not dict order."""
        corpus = "ab ab cd cd "
        results = []
        for _ in range(5):
            tokenizer = self._tokenizer(corpus)
            tokenizer.learn(2, min_freq=2)
            results.append(self._sorted_vocab(tokenizer))
        self.assertTrue(all(result == results[0] for result in results))
        # Ties are broken by the pair itself (max wins), so ('c','d') beats
        # ('a','b') and the independent 'cd' pair merges first while the
        # 'ab' word is left untouched.
        first = self._tokenizer(corpus)
        first.learn(1, min_freq=2)
        self.assertEqual(
            self._sorted_vocab(first),
            [(("a", "b" + EOW), 2), (eow("cd"), 2)],
        )

    def test_jump_does_not_change_merge_order(self):
        """jump is a prefetch hint: same result for every value, jump=1 included."""
        corpora = [
            "abc abc abc ",
            "abcd abc abc abcd ",
            "aaab aaab baaa ",
            "abab abab abab ",
            "aaabbb aaabbb bbbaaa ",
        ]
        for corpus in corpora:
            expected = None
            for jump in (1, 2, 3, 5, 100):
                tokenizer = self._tokenizer(corpus)
                tokenizer.learn(4, min_freq=2, jump=jump)
                vocab = self._sorted_vocab(tokenizer)
                if expected is None:
                    expected = vocab
                else:
                    self.assertEqual(
                        vocab,
                        expected,
                        "jump=%d diverged on %r" % (jump, corpus),
                    )

    def test_batch_does_not_trust_stale_counts(self):
        """A snapshot can list a pair that an earlier merge in the same batch
        wipes out; the loop must re-check the live count instead of merging
        it again off the snapshotted count."""
        tokenizer = self._tokenizer("abab abab ")
        # ('a','b'), ('b','a') and ('a','b<EOW>') all start at 2.
        self.assertEqual(tokenizer.stats[("a", "b")], 2)
        tokenizer.learn(2, min_freq=2, jump=3)
        # Both merges complete; no pair is counted or merged twice.
        self.assertEqual(tokenizer.vocab, [(("a", "bab" + EOW), 2)])

    def test_agrees_with_recount_reference(self):
        """Randomised differential against an independent re-counting BPE."""

        def reference(text, num, min_freq):
            words = Counter()
            for word in text.split():
                words[tuple(word[:-1]) + (word[-1] + EOW,)] += 1
            for _ in range(num):
                counts = Counter()
                for word, freq in words.items():
                    for prev, curr in zip(word, word[1:]):
                        counts[prev, curr] += freq
                counts = {p: c for p, c in counts.items() if c > 0}
                if not counts:
                    break
                best = max(sorted(counts), key=lambda p: (counts[p], p))
                if counts[best] < min_freq:
                    break
                merged = Counter()
                for word, freq in words.items():
                    out, i = [], 0
                    while i < len(word):
                        if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                            out.append(word[i] + word[i + 1])
                            i += 2
                        else:
                            out.append(word[i])
                            i += 1
                    merged[tuple(out)] += freq
                words = merged
            return sorted(words.items(), key=lambda wf: (-wf[1], wf[0]))

        rng = random.Random(20261125)
        for _ in range(120):
            text = " ".join(
                "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
                for _ in range(rng.randint(1, 5))
            )
            num = rng.randint(0, 5)
            min_freq = rng.randint(1, 3)
            expected = reference(text, num, min_freq)
            for jump in (1, 2, 4):
                tokenizer = self._tokenizer(text + " ")
                tokenizer.learn(num, min_freq=min_freq, jump=jump)
                self.assertEqual(
                    self._sorted_vocab(tokenizer),
                    expected,
                    "text=%r num=%d min_freq=%d jump=%d"
                    % (text, num, min_freq, jump),
                )


if __name__ == "__main__":
    unittest.main()
