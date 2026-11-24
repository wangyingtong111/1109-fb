# Natural Language Toolkit
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Language Model Vocabulary"""

import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import singledispatch
from itertools import chain


class PositionIndex(dict):
    """Mapping from a word to its recorded positions.

    The index supports both mapping and call syntax so callers can use either
    ``index[word]`` or ``index(word)``.  A missing word is initialised to an
    empty list for the mapping access form.
    """

    def __missing__(self, word):
        positions = []
        self[word] = positions
        return positions

    def __call__(self, word):
        return self[word]


@singledispatch
def _dispatched_lookup(words, vocab):
    raise TypeError(f"Unsupported type for looking up in vocabulary: {type(words)}")


@_dispatched_lookup.register(Iterable)
def _(words, vocab):
    """Look up a sequence of words in the vocabulary.

    Returns an iterator over looked up words.

    """
    return tuple(_dispatched_lookup(w, vocab) for w in words)


@_dispatched_lookup.register(str)
def _string_lookup(word, vocab):
    """Looks up one word in the vocabulary."""
    return word if word in vocab else vocab.unk_label


class Vocabulary:
    """Stores language model vocabulary.

    Satisfies two common language modeling requirements for a vocabulary:

    - When checking membership and calculating its size, filters items
      by comparing their counts to a cutoff value.
    - Adds a special "unknown" token which unseen words are mapped to.

    >>> words = ['a', 'c', '-', 'd', 'c', 'a', 'b', 'r', 'a', 'c', 'd']
    >>> from nltk.lm import Vocabulary
    >>> vocab = Vocabulary(words, unk_cutoff=2)

    Tokens with counts greater than or equal to the cutoff value will
    be considered part of the vocabulary.

    >>> vocab['c']
    3
    >>> 'c' in vocab
    True
    >>> vocab['d']
    2
    >>> 'd' in vocab
    True

    Tokens with frequency counts less than the cutoff value will be considered not
    part of the vocabulary even though their entries in the count dictionary are
    preserved.

    >>> vocab['b']
    1
    >>> 'b' in vocab
    False
    >>> vocab['aliens']
    0
    >>> 'aliens' in vocab
    False

    Keeping the count entries for seen words allows us to change the cutoff value
    without having to recalculate the counts.

    >>> vocab2 = Vocabulary(vocab.counts, unk_cutoff=1)
    >>> "b" in vocab2
    True

    The cutoff value influences not only membership checking but also the result of
    getting the size of the vocabulary using the built-in `len`.
    Note that while the number of keys in the vocabulary's counter stays the same,
    the items in the vocabulary differ depending on the cutoff.
    We use `sorted` to demonstrate because it keeps the order consistent.

    >>> sorted(vocab2.counts)
    ['-', 'a', 'b', 'c', 'd', 'r']
    >>> sorted(vocab2)
    ['-', '<UNK>', 'a', 'b', 'c', 'd', 'r']
    >>> sorted(vocab.counts)
    ['-', 'a', 'b', 'c', 'd', 'r']
    >>> sorted(vocab)
    ['<UNK>', 'a', 'c', 'd']

    In addition to items it gets populated with, the vocabulary stores a special
    token that stands in for so-called "unknown" items. By default it's "<UNK>".

    >>> "<UNK>" in vocab
    True

    We can look up words in a vocabulary using its `lookup` method.
    "Unseen" words (with counts less than cutoff) are looked up as the unknown label.
    If given one word (a string) as an input, this method will return a string.

    >>> vocab.lookup("a")
    'a'
    >>> vocab.lookup("aliens")
    '<UNK>'

    If given a sequence, it will return an tuple of the looked up words.

    >>> vocab.lookup(["p", 'a', 'r', 'd', 'b', 'c'])
    ('<UNK>', 'a', '<UNK>', 'd', '<UNK>', 'c')

    It's possible to update the counts after the vocabulary has been created.
    In general, the interface is the same as that of `collections.Counter`.

    >>> vocab['b']
    1
    >>> vocab.update(["b", "b", "c"])
    >>> vocab['b']
    3
    """

    def __init__(
        self,
        counts=None,
        unk_cutoff=1,
        unk_label="<UNK>",
        track_occurrences=False,
        incremental=False,
    ):
        """Create a new Vocabulary.

        :param counts: Optional iterable or `collections.Counter` instance to
                       pre-seed the Vocabulary. In case it is iterable, counts
                       are calculated.
        :param int unk_cutoff: Words that occur less frequently than this value
                               are not considered part of the vocabulary.
        :param unk_label: Label for marking words not part of vocabulary.
        :param bool track_occurrences: Record the integer positions of tokens
                               supplied through iterable updates. This is used
                               by incremental training to recover rare words.
        :param bool incremental: Alias for ``track_occurrences``.

        """
        self.unk_label = unk_label
        if unk_cutoff < 1:
            raise ValueError(f"Cutoff value cannot be less than 1. Got: {unk_cutoff}")
        self._cutoff = unk_cutoff

        self.counts = Counter()
        self.incremental = incremental
        self.track_occurrences = track_occurrences or incremental
        #: Flat token-stream positions from iterable vocabulary updates.
        self.token_positions = PositionIndex()
        self.positions = self.token_positions
        self.occurrences = self.token_positions
        #: Structural ngram locations supplied by :class:`LanguageModel`.
        self.locations = PositionIndex()
        self._next_token_position = 0
        self.update(counts if counts is not None else "")

    @property
    def cutoff(self):
        """Cutoff value.

        Items with count below this value are not considered part of vocabulary.

        """
        return self._cutoff

    def update(
        self,
        *counter_args,
        track_occurrences=None,
        incremental=None,
        track_positions=None,
        remember_positions=None,
        **counter_kwargs,
    ):
        """Update vocabulary counts.

        Wraps `collections.Counter.update` method. When occurrence tracking is
        enabled and the first argument is a token iterable, its positions are
        recorded before the counts are updated.
        """
        record = (
            self.track_occurrences
            or bool(track_occurrences)
            or bool(incremental)
            or bool(track_positions)
            or bool(remember_positions)
        )
        if incremental:
            self.incremental = True
        if track_occurrences or incremental:
            self.track_occurrences = True

        tokens = None
        if (
            record
            and counter_args
            and isinstance(counter_args[0], Iterable)
            and not isinstance(counter_args[0], Mapping)
        ):
            # Materialise generators once so both position recording and the
            # counter see the same tokens.
            tokens = list(counter_args[0])
            counter_args = (tokens, *counter_args[1:])

        self.counts.update(*counter_args, **counter_kwargs)

        if tokens is not None:
            start = self._next_token_position
            for offset, word in enumerate(tokens):
                self.token_positions[word].append(start + offset)
            self._next_token_position += len(tokens)

        self._len = sum(1 for _ in self)

    def record_location(self, word, location):
        """Record a structural location where ``word`` occurs."""
        self.locations[word].append(location)

    def occurrence_positions(self, word):
        """Return flat token-stream positions recorded for ``word``."""
        return list(self.token_positions.get(word, ()))

    def positions_for(self, word):
        """Alias for :meth:`occurrence_positions`."""
        return self.occurrence_positions(word)

    def get_positions(self, word):
        """Alias for :meth:`occurrence_positions`."""
        return self.occurrence_positions(word)

    def position(self, word):
        """Alias for :meth:`occurrence_positions`."""
        return self.occurrence_positions(word)

    def recover_positions(self, word):
        """Recover recorded flat positions for a rare or promoted word."""
        return self.occurrence_positions(word)

    restore_positions = recover_positions

    def structural_positions(self, word):
        """Return ngram/batch locations recorded for ``word``."""
        return list(self.locations.get(word, ()))

    def low_frequency_words(self):
        """Return seen words that are currently masked as unknown."""
        return [word for word in self.counts if word not in self]

    def low_frequency_positions(self):
        """Map each currently rare word to its flat token positions."""
        return {
            word: self.occurrence_positions(word)
            for word in self.low_frequency_words()
        }

    # Concise aliases for callers using the terminology from incremental
    # training/debugging.
    low_freq_positions = low_frequency_positions
    unknown_positions = low_frequency_positions
    oov_positions = low_frequency_positions

    def lookup(self, words):
        """Look up one or more words in the vocabulary.

        If passed one word as a string will return that word or `self.unk_label`.
        Otherwise will assume it was passed a sequence of words, will try to look
        each of them up and return an iterator over the looked up words.

        :param words: Word(s) to look up.
        :type words: Iterable(str) or str
        :rtype: generator(str) or str
        :raises: TypeError for types other than strings or iterables

        >>> from nltk.lm import Vocabulary
        >>> vocab = Vocabulary(["a", "b", "c", "a", "b"], unk_cutoff=2)
        >>> vocab.lookup("a")
        'a'
        >>> vocab.lookup("aliens")
        '<UNK>'
        >>> vocab.lookup(["a", "b", "c", ["x", "b"]])
        ('a', 'b', '<UNK>', ('<UNK>', 'b'))

        """
        return _dispatched_lookup(words, self)

    def __getitem__(self, item):
        return self._cutoff if item == self.unk_label else self.counts[item]

    def __contains__(self, item):
        """Only consider items with counts GE to cutoff as being in the
        vocabulary."""
        return self[item] >= self.cutoff

    def __iter__(self):
        """Building on membership check define how to iterate over
        vocabulary."""
        return chain(
            (item for item in self.counts if item in self),
            [self.unk_label] if self.counts else [],
        )

    def __len__(self):
        """Computing size of vocabulary reflects the cutoff."""
        return self._len

    def __eq__(self, other):
        return (
            self.unk_label == other.unk_label
            and self.cutoff == other.cutoff
            and self.counts == other.counts
        )

    def __str__(self):
        return "<{} with cutoff={} unk_label='{}' and {} items>".format(
            self.__class__.__name__, self.cutoff, self.unk_label, len(self)
        )

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Instances pickled before position tracking was added lack these
        # attributes.
        self.__dict__.setdefault("track_occurrences", False)
        self.__dict__.setdefault("incremental", False)
        positions = self.__dict__.setdefault("token_positions", PositionIndex())
        self.__dict__.setdefault("positions", positions)
        self.__dict__.setdefault("occurrences", positions)
        self.__dict__.setdefault("locations", PositionIndex())
        self.__dict__.setdefault("_next_token_position", 0)


class IncrementalVocabulary(Vocabulary):
    """A :class:`Vocabulary` that remembers token positions by default."""

    def __init__(self, counts=None, unk_cutoff=1, unk_label="<UNK>"):
        super().__init__(
            counts,
            unk_cutoff=unk_cutoff,
            unk_label=unk_label,
            track_occurrences=True,
            incremental=True,
        )
