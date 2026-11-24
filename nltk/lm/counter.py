# Natural Language Toolkit
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
Language Model Counter
----------------------
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from operator import index as integer_index

from nltk.probability import ConditionalFreqDist, FreqDist

#: Upper bound on the number of *distinct* ngrams a single ``NgramCounter`` may
#: store. The counter keeps every distinct ngram of every order in a nested
#: dictionary tree, so training a language model on an untrusted corpus of
#: distinct tokens grows memory without limit (~order * tokens) and OOM-kills the
#: worker (CWE-770; CVE-2026-12928). Once this many distinct ngrams have been
#: stored, ``update`` raises ``ValueError`` instead of growing unbounded. The
#: default is generous enough for NLTK's documented training corpora (e.g. Brown
#: at the usual orders); raise it if you train on a genuinely larger corpus.
MAX_NGRAMS = 10_000_000


class _ChangeTrackingFreqDist(FreqDist):
    """FreqDist that records changes to its positive/zero topology.

    Kneser-Ney continuation counts depend on whether an ngram type has a
    positive count, not on its raw frequency.  The class-level revision lets
    cached continuation indexes notice direct ``counts[...][...] -= 1`` edits
    as well as calls to :meth:`NgramCounter.subtract`.
    """

    _topology_revision = 0

    @classmethod
    def _bump_topology_revision(cls):
        cls._topology_revision += 1

    def __setitem__(self, key, value):
        was_positive = self.get(key, 0) > 0
        super().__setitem__(key, value)
        if was_positive != (value > 0):
            type(self)._bump_topology_revision()

    def __delitem__(self, key):
        was_positive = self.get(key, 0) > 0
        super().__delitem__(key)
        if was_positive:
            type(self)._bump_topology_revision()


def _new_order_counts():
    """Create a CFD whose frequency distributions track topology changes."""
    counts = ConditionalFreqDist()
    counts.default_factory = _ChangeTrackingFreqDist
    return counts


class NgramCounter:
    """Class for counting ngrams.

    Will count any ngram sequence you give it ;)

    First we need to make sure we are feeding the counter sentences of ngrams.

    >>> text = [["a", "b", "c", "d"], ["a", "c", "d", "c"]]
    >>> from nltk.util import ngrams
    >>> text_bigrams = [ngrams(sent, 2) for sent in text]
    >>> text_unigrams = [ngrams(sent, 1) for sent in text]

    The counting itself is very simple.

    >>> from nltk.lm import NgramCounter
    >>> ngram_counts = NgramCounter(text_bigrams + text_unigrams)

    You can conveniently access ngram counts using standard python dictionary notation.
    String keys will give you unigram counts.

    >>> ngram_counts['a']
    2
    >>> ngram_counts['aliens']
    0

    If you want to access counts for higher order ngrams, use a list or a tuple.
    These are treated as "context" keys, so what you get is a frequency distribution
    over all continuations after the given context.

    >>> sorted(ngram_counts[['a']].items())
    [('b', 1), ('c', 1)]
    >>> sorted(ngram_counts[('a',)].items())
    [('b', 1), ('c', 1)]

    This is equivalent to specifying explicitly the order of the ngram (in this case
    2 for bigram) and indexing on the context.

    >>> ngram_counts[2][('a',)] is ngram_counts[['a']]
    True

    Note that the keys in `ConditionalFreqDist` cannot be lists, only tuples!
    It is generally advisable to use the less verbose and more flexible square
    bracket notation.

    To get the count of the full ngram "a b", do this:

    >>> ngram_counts[['a']]['b']
    1

    Specifying the ngram order as a number can be useful for accessing all ngrams
    in that order.

    >>> ngram_counts[2]
    <ConditionalFreqDist with 4 conditions>

    The keys of this `ConditionalFreqDist` are the contexts we discussed earlier.
    Unigrams can also be accessed with a human-friendly alias.

    >>> ngram_counts.unigrams is ngram_counts[1]
    True

    Similarly to `collections.Counter`, you can update counts after initialization.

    >>> ngram_counts['e']
    0
    >>> ngram_counts.update([ngrams(["d", "e", "f"], 1)])
    >>> ngram_counts['e']
    1

    """

    def __init__(self, ngram_text=None):
        """Creates a new NgramCounter.

        If `ngram_text` is specified, counts ngrams from it, otherwise waits for
        `update` method to be called explicitly.

        :param ngram_text: Optional text containing sentences of ngrams, as for `update` method.
        :type ngram_text: Iterable(Iterable(tuple(str))) or None

        """
        self._counts = defaultdict(_new_order_counts)
        self.unigrams = _ChangeTrackingFreqDist()
        self._counts[1] = self.unigrams
        #: Number of distinct ngrams stored, tracked for the ``MAX_NGRAMS`` guard.
        self._distinct = 0
        #: Monotonic cache-generation token for positive ngram types.
        self._version = 0

        if ngram_text:
            self.update(ngram_text)

    def update(self, ngram_text):
        """Updates ngram counts from `ngram_text`.

        Expects `ngram_text` to be a sequence of sentences (sequences).
        Each sentence consists of ngrams as tuples of strings.

        :param Iterable(Iterable(tuple(str))) ngram_text: Text containing sentences of ngrams.
        :raises TypeError: if the ngrams are not tuples.

        """

        for sent in ngram_text:
            for ngram in sent:
                if not isinstance(ngram, tuple):
                    raise TypeError(f"Ngram <{ngram}> isn't a tuple, but {type(ngram)}")

                ngram_order = len(ngram)
                if ngram_order == 1:
                    if ngram[0] not in self.unigrams:
                        self._note_new_ngram()
                    self.unigrams[ngram[0]] += 1
                    continue

                context, word = ngram[:-1], ngram[-1]
                # Probe with .get() so testing whether this (context, word) pair
                # is new does not eagerly create empty nested entries through the
                # defaultdicts; otherwise an ngram refused by the MAX_NGRAMS guard
                # would still leave new contexts/orders behind, defeating the
                # memory bound. The real entries are created only once the guard
                # below has passed.
                order_counts = self._counts.get(ngram_order)
                context_counts = (
                    order_counts.get(context) if order_counts is not None else None
                )
                if context_counts is None or word not in context_counts:
                    self._note_new_ngram()
                self[ngram_order][context][word] += 1

    def _note_new_ngram(self):
        """Account for one newly-stored distinct ngram and enforce the bound.

        The counter retains every distinct ngram, so without a bound an untrusted
        corpus of distinct tokens grows memory without limit and OOM-kills the
        worker (CWE-770; CVE-2026-12928). Refuse once ``MAX_NGRAMS`` distinct
        ngrams have been stored. The bound is checked *before* incrementing so a
        refused ngram (which is never stored) does not inflate the counter.
        """
        if self._distinct >= MAX_NGRAMS:
            raise ValueError(
                "Refusing to count further: NgramCounter exceeded the limit of "
                "%d distinct ngrams (CWE-770). Training on this corpus would grow "
                "memory without bound. Use a smaller corpus or order, or raise "
                "nltk.lm.counter.MAX_NGRAMS." % MAX_NGRAMS
            )
        self._distinct += 1
        self._version = getattr(self, "_version", 0) + 1

    @staticmethod
    def _add_requested_ngram(requested, ngram, amount=1):
        """Validate one requested subtraction and aggregate repeated requests."""
        if not isinstance(ngram, tuple):
            raise TypeError(f"Ngram <{ngram}> isn't a tuple, but {type(ngram)}")
        try:
            amount = integer_index(amount)
        except TypeError as exc:
            raise ValueError(
                f"Ngram subtraction amount must be an integer, got {amount!r}"
            ) from exc
        if amount < 0:
            raise ValueError(
                f"Ngram subtraction amount cannot be negative, got {amount!r}"
            )
        if amount == 0:
            return
        requested[ngram] += amount

    @staticmethod
    def _looks_like_ngram(item):
        # Ngrams are tuples of string tokens; a sentence contains ngram tuples.
        return isinstance(item, tuple) and all(isinstance(token, str) for token in item)

    def _current_ngram_count(self, ngram):
        if len(ngram) == 1:
            return self.unigrams.get(ngram[0], 0)

        order_counts = self._counts.get(len(ngram))
        if order_counts is None:
            return 0
        context_counts = order_counts.get(ngram[:-1])
        if context_counts is None:
            return 0
        return context_counts.get(ngram[-1], 0)

    def subtract(self, ngram_text=None, **kwargs):
        """Subtract previously counted ngrams without rebuilding the counter.

        ``ngram_text`` accepts the same nested sentence/ngram format as
        :meth:`update`, a flat iterable of ngram tuples, or a mapping from
        ngram tuples (or unigram strings) to amounts.  The whole request is
        checked before any count is changed: requesting more occurrences than
        are currently stored raises ``ValueError`` and leaves the counter
        untouched.

        Zero-count ngram types and their now-empty contexts are removed.
        Vocabulary objects are intentionally not changed by this method.
        """
        requested = Counter()

        if ngram_text is not None:
            if isinstance(ngram_text, NgramCounter):
                for order, order_counts in ngram_text._counts.items():
                    if order == 1:
                        for ngram, amount in order_counts.items():
                            self._add_requested_ngram(requested, (ngram,), amount)
                    else:
                        for context, context_counts in order_counts.items():
                            for word, amount in context_counts.items():
                                self._add_requested_ngram(
                                    requested, (*context, word), amount
                                )
            elif isinstance(ngram_text, Mapping):
                for ngram, amount in ngram_text.items():
                    if isinstance(ngram, str):
                        ngram = (ngram,)
                    self._add_requested_ngram(requested, ngram, amount)
            elif self._looks_like_ngram(ngram_text):
                self._add_requested_ngram(requested, ngram_text)
            else:
                for sentence in ngram_text:
                    if self._looks_like_ngram(sentence):
                        self._add_requested_ngram(requested, sentence)
                    else:
                        for ngram in sentence:
                            self._add_requested_ngram(requested, ngram)

        for token, amount in kwargs.items():
            self._add_requested_ngram(requested, (token,), amount)

        # Validate every request before mutating anything, so an overdraw in
        # a later ngram cannot partially subtract an earlier one.
        for ngram, amount in requested.items():
            current = self._current_ngram_count(ngram)
            if amount > current:
                raise ValueError(
                    f"Cannot subtract {amount} occurrence(s) of ngram {ngram!r}; "
                    f"only {current} are currently counted."
                )

        removed_type = False
        for ngram, amount in requested.items():
            order = len(ngram)
            if order == 1:
                context_counts = self.unigrams
                key = ngram[0]
            else:
                context = ngram[:-1]
                key = ngram[-1]
                context_counts = self[order][context]

            new_count = context_counts[key] - amount
            if new_count == 0:
                del context_counts[key]
                self._distinct -= 1
                removed_type = True

                if order > 1 and not any(
                    count > 0 for count in context_counts.values()
                ):
                    order_counts = self._counts[order]
                    del order_counts[context]
                    if not order_counts:
                        del self._counts[order]
            else:
                # Validation guarantees this remains positive.
                context_counts[key] = new_count

        if removed_type:
            # getattr keeps counters pickled before this field existed usable.
            self._version = getattr(self, "_version", 0) + 1

    def subtract_ngram(self, ngram, amount=1):
        """Subtract ``amount`` occurrences of one ngram tuple."""
        self.subtract({ngram: amount})

    #: Descriptive aliases for callers using update/remove terminology.
    subtract_counts = subtract
    subtract_ngrams = subtract
    deduct_counts = subtract
    remove = subtract
    remove_ngram = subtract_ngram
    downdate = subtract
    deduct = subtract
    remove_ngrams = subtract

    def N(self):
        """Returns grand total number of ngrams stored.

        This includes ngrams from all orders, so some duplication is expected.
        :rtype: int

        >>> from nltk.lm import NgramCounter
        >>> counts = NgramCounter([[("a", "b"), ("c",), ("d", "e")]])
        >>> counts.N()
        3

        """
        return sum(val.N() for val in self._counts.values())

    def _order_counts(self, order):
        """Return one order's CFD and restore tracking after unpickling."""
        order_counts = self._counts[order]
        if isinstance(order_counts, ConditionalFreqDist):
            order_counts.default_factory = _ChangeTrackingFreqDist
        return order_counts

    def __getitem__(self, item):
        """User-friendly access to ngram counts."""
        if isinstance(item, int):
            return self._order_counts(item)
        elif isinstance(item, str):
            return self._counts.__getitem__(1)[item]
        elif isinstance(item, Sequence):
            return self._order_counts(len(item) + 1)[tuple(item)]

    def __str__(self):
        return "<{} with {} ngram orders and {} ngrams>".format(
            self.__class__.__name__, len(self._counts), self.N()
        )

    def __len__(self):
        return self._counts.__len__()

    def __contains__(self, item):
        return item in self._counts
