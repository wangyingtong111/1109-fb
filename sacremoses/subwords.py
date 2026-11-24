# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
from collections import Counter, defaultdict
from functools import reduce
from heapq import nlargest

from sacremoses.util import pairwise


class SubwordTokenizer(object):
    """
    This is a Python port of the Subword NMT from
    https://github.com/rsennrich/subword-nmt
    """

    def __init__(self, filename):
        #
        self.vocab = self.get_vocabulary(filename)
        self.stats, self.indices = self.get_pair_statistics()
        self.big_stats = copy.deepcopy(self.stats)

    def get_vocabulary(self, filename, is_dict=False):
        vocab = Counter()
        # Vocabulary/BPE files are routinely non-ASCII; do not let the
        # platform locale decide how they decode.
        with open(filename, encoding="utf-8") as fin:
            if is_dict:
                for line in fin:
                    word, count = line.strip().split(" ")
                    vocab[word] += int(count)
            else:
                vocab.update(fin.read().split())
        # Converts the string keys to tuples of characters,
        # adds "\uE000" to the last character.
        vocab = Counter(
            {tuple(k[:-1]) + (k[-1] + "\uE000",): v for (k, v) in vocab.items()}
        )
        return vocab.most_common()

    def get_pair_statistics(self):
        """Count frequency of all symbol pairs, and create index"""
        # Data structure of pair frequencies
        stats = Counter()
        # Index from pairs to words
        indices = defaultdict(lambda: Counter())

        for i, (word, freq) in enumerate(self.vocab):
            for prev, curr in pairwise(word):
                stats[prev, curr] += freq
                indices[prev, curr][i] += 1

        return stats, indices

    def modify_token(self, token, pair):
        """
        From https://stackoverflow.com/a/40367074/610569
            >>> modify_token(('s', 'h', 'e', 'r', 'l', 'o', 'c', 'k'), ('h', 'e'))
            ('S', 'he', 'r', 'l', 'o', 'c', 'k')
        """
        first, second = pair
        pair_str = "".join(pair).replace("\\", "\\\\")
        f = (
            lambda acc, e: acc[:-1] + (pair_str,)
            if acc[-1] == first and e == second
            else acc + (e,)
        )
        return reduce(f, token[1:], (token[0],))

    def replace_pair(self, pair):
        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
        changes = []
        for j, freq in self.indices[pair].items():
            if freq < 1:
                continue
            word, freq = self.vocab[j]
            new_word = self.modify_token(word, pair)
            self.vocab[j] = (new_word, freq)
            changes.append((j, new_word, word, freq))
        return changes

    def update_pair_statistics(self, pair, changed):
        """
        Minimally update the indices and frequency of symbol pairs
        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.
        """
        self.stats[pair] = 0
        self.indices[pair] = Counter()
        first, second = pair
        new_pair = first + second
        for j, word, old_word, freq in changed:

            # Find all instances of pair in the old_word, and update frequency/indices around it
            i = 0
            # Keep moving down the old_word string until we cannot find
            # the first char in the new_pair.
            while True:
                try:
                    # Find the next occurence of the first character in the new_pair.
                    i = old_word.index(first, i)
                except ValueError:
                    break
                # Checks that old_word[i:i+1] is the same as new_pair.
                # (i) `i < len(old_word)-1` checks that the index i is not the last character.
                # (ii) `old_word[i+1]` checks that the char after the index is the second char in the new_pair.
                if i < len(old_word) - 1 and old_word[i + 1] == second:
                    # `if i` checks that i is non-zero.
                    # We can skip the first char since there's no previous bigram.
                    if i:
                        # Find the previous bigram and reduce its count.
                        prev = old_word[i - 1 : i + 1]
                        self.stats[prev] -= freq
                        self.indices[prev][j] -= 1
                    # `if < len(old_word)-2` checks that the new_pair is not at the end of the old_word.
                    if i < len(old_word) - 2:
                        # The multiple if conditions that follows checks that the bigram after i and i+1
                        # is not the same as new_pair to avoid double-counting consecutive pairs.
                        # (i)   `old_word[i+2] != first` checks that two chars after i, it isn't the same as
                        #                          the first char in the new_pair.
                        # (ii)  `old_word[i+3] != second` checks that three chars after i, it isn't the same
                        #                         as the second char in the new_pair.
                        # (iii) `i >= len(old_word)-3` checks that the i index is one of the last 4 chars in old_word.
                        # @rico: Is the `i >= len(old_word)-3` check to avoid IndexError?
                        if (
                            old_word[i + 2] != first
                            or i >= len(old_word) - 3
                            or old_word[i + 3] != second
                        ):
                            # Find the next bigram and reduce its count.
                            # `nex` is the next bigram after new_pair.
                            nex = old_word[i + 1 : i + 3]
                            self.stats[nex] -= freq
                            self.indices[nex][j] -= 1
                    # Now we move the ith index to two chars to the right when
                    # old_word[i:i+1] is the same as new_pair.
                    i += 2
                else:  # Otherwise, we move one char to the right.
                    i += 1

            # Find all instances of pair in the new *word*, and update frequency/indices around it
            # Reset the index to the start of the string.
            i = 0
            # Similarly, we keep moving down the new *word* string until we cannot find
            # the first char in the new_pair.
            while True:
                try:
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                # We are sure that the new_pair is in the new *word* so there's no need to
                # do an outer check as what was done in the old_word.
                if (
                    i
                ):  # `if i` checks that i is non-zero, skip the first char since there's no previous bigram.
                    prev = word[i - 1 : i + 1]
                    # This time, we add the frequency back to the statistics and indices.
                    self.stats[prev] += freq
                    self.indices[prev][j] += 1
                # The multiple if conditions that follows checks that the bigram after i and i+1
                # is not the same as new_pair to avoid double-counting consecutive pairs.
                # `i < len(word)-1` checks if i is not the last character.
                # `word[i+1]` checks that the next char is not the new_pair.
                if i < len(word) - 1 and word[i + 1] != new_pair:
                    # `nex` is the next bigram after new_pair.
                    nex = word[i : i + 2]
                    # We add the frequency back to the statistics and indices.
                    self.stats[nex] += freq
                    self.indices[nex][j] += 1
                # We move one char down the new *word*
                i += 1

    def _pair_rank(self, pair):
        """Sort key for a pair: (frequency, pair).

        The pair itself breaks ties, so the merge order is fully determined by
        the corpus and never depends on dict insertion order or hash seed.
        This is the tie-break used by upstream subword-nmt:
        ``max(stats, key=lambda x: (stats[x], x))``.
        """
        return self.stats.get(pair, 0), pair

    def _best_pair(self):
        """The pair with the highest live frequency, ties broken by the pair."""
        if not self.stats:
            return None
        return max(self.stats, key=self._pair_rank)

    def learn(self, num_symbols, min_freq=2, jump=1, is_dict=None):
        """Learn up to ``num_symbols`` merges, stopping early only once no pair
        still reaches ``min_freq``.

        A pair that has just been merged has its own count zeroed, which must
        not be mistaken for "the corpus is exhausted": the merges it created
        are frequently the most frequent pairs left.  So the loop keeps going
        until a *freshly selected* candidate is below ``min_freq``.

        ``jump`` is a prefetch hint: several candidates are picked in one pass,
        but those counts go stale the moment anything is merged, so the batch
        only decides *which* pairs to look at.  Before a candidate is applied
        its count is re-read and it is confirmed to still be the live best;
        otherwise the batch is dropped and the selection is redone from the
        live statistics.  ``jump`` therefore never changes the result: every
        value produces exactly the same merges, in the same order, as
        ``jump=1``.
        """
        num_symbols = max(0, int(num_symbols))
        jump = max(1, int(jump))
        merged = 0
        while merged < num_symbols:
            # Snapshot a batch of candidates, best first.
            batch = nlargest(jump, self.stats, key=self._pair_rank)
            if not batch:
                break
            progressed = False
            for position, pair in enumerate(batch):
                if merged >= num_symbols:
                    break
                # Re-read the live count: an earlier merge in this batch may
                # have consumed this pair or dropped it below the threshold.
                # The snapshotted count is never trusted for the decision.
                live = self.stats.get(pair, 0)
                if live <= 0 or live < min_freq:
                    continue
                # `batch[0]` is the live best by construction; any later
                # position must still be the live best or the ranking has
                # moved and the batch is stale.
                if position and self._best_pair() != pair:
                    break
                changes = self.replace_pair(pair)
                self.update_pair_statistics(pair, changes)
                self.stats[pair] = 0
                merged += 1
                progressed = True
            if not progressed:
                # Nothing in the batch was applicable: every remaining pair is
                # already merged or below min_freq.
                break
