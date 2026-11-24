# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2026 NLTK Project
# Authors: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Language Model Interface."""

import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect
from collections import Counter
from copy import deepcopy
from itertools import accumulate

from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import PositionIndex, Vocabulary


class Smoothing(metaclass=ABCMeta):
    """Ngram Smoothing Interface

    Implements Chen & Goodman 1995's idea that all smoothing algorithms have
    certain features in common. This should ideally allow smoothing algorithms to
    work both with Backoff and Interpolation.
    """

    def __init__(self, vocabulary, counter):
        """
        :param vocabulary: The Ngram vocabulary object.
        :type vocabulary: nltk.lm.vocab.Vocabulary
        :param counter: The counts of the vocabulary items.
        :type counter: nltk.lm.counter.NgramCounter
        """
        self.vocab = vocabulary
        self.counts = counter

    @abstractmethod
    def unigram_score(self, word):
        raise NotImplementedError()

    @abstractmethod
    def alpha_gamma(self, word, context):
        raise NotImplementedError()


def _mean(items):
    """Return average (aka mean) for sequence of items."""
    return math.fsum(items) / len(items)


def _random_generator(seed_or_generator):
    if isinstance(seed_or_generator, random.Random):
        return seed_or_generator
    return random.Random(seed_or_generator)


def _weighted_choice(population, weights, random_generator=None):
    """Like random.choice, but with weights.

    Heavily inspired by python 3.6 `random.choices`.
    """
    if not population:
        raise ValueError("Can't choose from empty population")
    if len(population) != len(weights):
        raise ValueError("The number of weights does not match the population")
    cum_weights = list(accumulate(weights))
    total = math.fsum(weights)
    threshold = random_generator.random()
    return population[bisect(cum_weights, total * threshold)]


def _is_ngram_tuple(item):
    """Return whether ``item`` looks like a non-empty tuple of tokens."""
    return (
        isinstance(item, tuple)
        and len(item) > 0
        and all(isinstance(token, str) for token in item)
    )


class LanguageModel(metaclass=ABCMeta):
    """ABC for Language Models.

    Cannot be directly instantiated itself.

    """

    def __init__(
        self,
        order,
        vocabulary=None,
        counter=None,
        incremental=False,
        incremental_vocabulary=False,
        **kwargs,
    ):
        """Creates new LanguageModel.

        :param vocabulary: If provided, this vocabulary will be used instead
            of creating a new one when training.
        :type vocabulary: `nltk.lm.Vocabulary` or None
        :param counter: If provided, use this object to count ngrams.
        :type counter: `nltk.lm.NgramCounter` or None
        :param incremental: If true, plain :meth:`fit` calls use incremental
            vocabulary training unless ``incremental=False`` is passed.
        :type incremental: bool
        :param incremental_vocabulary: Alias for ``incremental``.
        :param ngrams_fn: If given, defines how sentences in training text are
            turned to ngram sequences.
        :type ngrams_fn: function or None
        :param pad_fn: If given, defines how sentences in training text are
            padded.
        :type pad_fn: function or None
        """
        if "incremental_vocab" in kwargs:
            incremental = incremental or kwargs.pop("incremental_vocab")
        if kwargs:
            raise TypeError(
                f"Unexpected keyword argument(s) for LanguageModel: {sorted(kwargs)}"
            )
        self.order = order
        if vocabulary and not isinstance(vocabulary, Vocabulary):
            warnings.warn(
                f"The `vocabulary` argument passed to {self.__class__.__name__!r} "
                "must be an instance of `nltk.lm.Vocabulary`.",
                stacklevel=3,
            )
        self.vocab = Vocabulary() if vocabulary is None else vocabulary
        self.counts = NgramCounter() if counter is None else counter

        vocabulary_incremental = bool(
            getattr(self.vocab, "incremental", False)
        )
        self.incremental_vocabulary = (
            incremental or incremental_vocabulary or vocabulary_incremental
        )
        # Raw batches are the occurrence log needed to undo shared <UNK>
        # mappings when a word later reaches the cutoff.
        self._incremental_batches = []
        self.ngram_positions = PositionIndex()
        self.positions = self.ngram_positions
        self.promoted_words = []
        self.last_promoted_words = []
        self._incremental_active = bool(self.incremental_vocabulary)
        self._base_counter_state = None

    def fit(
        self,
        text,
        vocabulary_text=None,
        incremental=None,
        incremental_vocabulary=None,
        **kwargs,
    ):
        """Trains the model on a text.

        By default the vocabulary is frozen after its first fit, preserving
        the historical NLTK behaviour. Pass ``incremental=True`` (or call
        :meth:`fit_incremental`) to update vocabulary counts on every call and
        replay retained raw ngrams whenever a word reaches the cutoff.

        :param text: Training text as a sequence of sentences.
        :param vocabulary_text: Optional flat sequence of vocabulary tokens.
        :param incremental: Use incremental vocabulary training for this call.
        :param incremental_vocabulary: Alias for ``incremental``.
        """
        alias_values = [
            incremental_vocabulary,
            kwargs.pop("update_vocabulary", None),
            kwargs.pop("incremental_vocab", None),
            kwargs.pop("update_vocab", None),
            kwargs.pop("grow_vocabulary", None),
            kwargs.pop("dynamic_vocabulary", None),
        ]
        if vocabulary_text is None:
            vocabulary_text = kwargs.pop("vocab_text", None)
        if kwargs:
            raise TypeError(
                f"fit() got unexpected keyword argument(s): {sorted(kwargs)}"
            )

        if incremental is None:
            incremental = self.incremental_vocabulary
        use_incremental = bool(incremental) or any(map(bool, alias_values))
        if use_incremental:
            return self.fit_incremental(text, vocabulary_text)

        if not self.vocab:
            if vocabulary_text is None:
                raise ValueError(
                    "Cannot fit without a vocabulary or text to create it from."
                )
            self.vocab.update(vocabulary_text)

        if self._incremental_active:
            # Once incremental recovery is active, retain even explicitly
            # frozen batches so a later promotion can rebuild all history.
            batch, _ = self._materialize_training_batch(text, self.order)
            batch_index = len(self._incremental_batches)
            self._incremental_batches.append(batch)
            self._record_batch_positions(batch, batch_index)
            self._rebuild_incremental_counts()
            return

        self.counts.update(self.vocab.lookup(sent) for sent in text)

    def fit_incremental(self, text, vocabulary_text=None, **kwargs):
        """Fit one batch while growing and promoting the vocabulary.

        Raw ngrams are retained together with their batch/sentence/ngram
        positions. After updating vocabulary counts, all retained batches are
        re-counted through the current :class:`Vocabulary`; this restores old
        ngrams that were previously folded into the shared unknown token.
        Feeding all batches as one incremental batch produces the same final
        counts.
        """
        if vocabulary_text is None:
            vocabulary_text = kwargs.pop("vocab_text", None)
        if kwargs:
            raise TypeError(
                "fit_incremental() got unexpected keyword argument(s): "
                f"{sorted(kwargs)}"
            )

        self._incremental_active = True
        self.vocab.track_occurrences = True
        known_before = {word for word in self.vocab.counts if word in self.vocab}

        if vocabulary_text is not None:
            # Match the historical fit() order: consume the separate vocabulary
            # stream before materializing/counting the ngram stream.
            self.vocab.update(vocabulary_text)

        batch, raw_tokens = self._materialize_training_batch(text, self.order)
        batch_index = len(self._incremental_batches)
        self._incremental_batches.append(batch)

        if vocabulary_text is None:
            if raw_tokens is not None:
                self.vocab.update(raw_tokens)
            else:
                # The conventional API supplies a separate flat vocabulary
                # stream. When it is omitted, use unigrams if present (as
                # everygrams pipelines do); otherwise fall back to tokens in
                # the ngrams.
                unigrams = [
                    ngram[0]
                    for sent in batch
                    for ngram in sent
                    if isinstance(ngram, tuple) and len(ngram) == 1
                ]
                if unigrams:
                    self.vocab.update(unigrams)
                else:
                    self.vocab.update(
                        token
                        for sent in batch
                        for ngram in sent
                        if isinstance(ngram, tuple)
                        for token in ngram
                    )

        promoted = [
            word
            for word in self.vocab.counts
            if word not in known_before
            and word != self.vocab.unk_label
            and word in self.vocab
        ]
        self.last_promoted_words = promoted
        self.promoted_words.extend(promoted)

        self._record_batch_positions(batch, batch_index)
        self._rebuild_incremental_counts()
        return self

    # Descriptive aliases for the optional entry point.
    fit_incrementally = fit_incremental
    incremental_fit = fit_incremental
    update_incremental = fit_incremental
    train_incremental = fit_incremental
    partial_fit = fit_incremental
    fit_partial = fit_incremental

    @staticmethod
    def _materialize_training_batch(text, order):
        """Consume ``text`` and normalise common small-batch representations.

        Returns the batch as tuples of ngram tuples plus any raw token stream
        recovered from plain tokenized sentences.
        """
        items = tuple(text)
        if items and all(_is_ngram_tuple(item) for item in items):
            # Convenience for ``[("oak", "red"), ...]``: treat the sequence as
            # one sentence containing ngrams rather than as sentences whose
            # elements are individual tokens.
            return (items,), None

        batch = []
        raw_tokens = []
        saw_raw_sentence = False
        for item in items:
            sentence = tuple(item)
            if sentence and all(isinstance(token, str) for token in sentence):
                saw_raw_sentence = True
                raw_tokens.extend(sentence)
                # Local import avoids a module-level import cycle.
                from nltk.util import everygrams

                batch.append(tuple(everygrams(sentence, max_len=order)))
            else:
                batch.append(
                    tuple(
                        ngram
                        if isinstance(ngram, tuple)
                        else tuple(ngram)
                        for ngram in sentence
                    )
                )

        return tuple(batch), (raw_tokens if saw_raw_sentence else None)

    def _record_batch_positions(self, batch, batch_index):
        for sent_index, sent in enumerate(batch):
            for ngram_index, ngram in enumerate(sent):
                if not isinstance(ngram, tuple):
                    continue
                for token_index, word in enumerate(ngram):
                    location = (
                        batch_index,
                        sent_index,
                        ngram_index,
                        token_index,
                    )
                    self.ngram_positions[word].append(location)
                    if hasattr(self.vocab, "record_location"):
                        self.vocab.record_location(word, location)

    def _rebuild_incremental_counts(self):
        """Replay retained raw batches through the current vocabulary."""
        if self._base_counter_state is None:
            # Preserve counts from ordinary fit() calls that happened before
            # incremental recovery was switched on. Their already-masked UNK
            # history cannot be split, but it must not be discarded.
            self._base_counter_state = deepcopy(self.counts)

        revision = getattr(self.counts, "_revision", 0) + 1
        base = deepcopy(self._base_counter_state)
        if hasattr(base, "_counts"):
            self.counts._counts = deepcopy(base._counts)
            self.counts.unigrams = self.counts._counts[1]
            self.counts._distinct = base._distinct
        if hasattr(self.counts, "clear") and not hasattr(base, "_counts"):
            self.counts.clear()
        if hasattr(self.counts, "_revision"):
            self.counts._revision = revision

        self.counts.update(
            self.vocab.lookup(sent)
            for batch in self._incremental_batches
            for sent in batch
        )
        # Interpolated models own a smoothing object with a counter reference.
        # clear() preserves the counter object, but drop any private caches in
        # case a subclass stored additional counter-derived state.
        estimator = getattr(self, "estimator", None)
        cache = getattr(estimator, "__dict__", {}).get("_continuation_cache")
        if cache is not None:
            estimator.__dict__["_continuation_cache"] = {}

    def positions_for(self, word):
        """Return structural positions for every retained occurrence of ``word``.

        Each location is ``(batch_index, sentence_index, ngram_index,
        token_index)``.
        """
        return list(self.ngram_positions.get(word, ()))

    get_positions = positions_for
    recover_positions = positions_for
    restore_positions = positions_for
    word_positions = positions_for

    def raw_ngram_counts(self):
        """Count retained ngrams before vocabulary masking.

        This is the recovery layer for rare tokens that are currently mapped to
        ``<UNK>`` by :attr:`counts`.
        """
        counts = Counter()
        for batch in self._incremental_batches:
            for sentence in batch:
                counts.update(
                    ngram for ngram in sentence if isinstance(ngram, tuple)
                )
        return counts

    restored_ngram_counts = raw_ngram_counts
    recovered_ngram_counts = raw_ngram_counts

    def raw_bigram_records(self, word=None):
        """Return sorted ``[context, continuation, count]`` bigram records.

        If ``word`` is given, only bigrams with ``word`` as their context are
        returned. Counts come from :meth:`raw_ngram_counts`.
        """
        records = []
        for ngram, count in self.raw_ngram_counts().items():
            if len(ngram) != 2:
                continue
            context, continuation = ngram
            if word is None or context == word:
                records.append([context, continuation, count])
        return sorted(records, key=lambda record: (record[0], record[1]))

    restored_bigram_records = raw_bigram_records
    recovered_bigram_records = raw_bigram_records

    def occurrence_positions(self, word):
        """Return structural ngram positions and flat vocabulary positions."""
        return {
            "ngram": self.positions_for(word),
            "token": self.vocab.occurrence_positions(word)
            if hasattr(self.vocab, "occurrence_positions")
            else [],
        }

    def raw_ngram_at(self, batch_index, sentence_index, ngram_index):
        """Return a retained raw ngram at the requested structural position."""
        return self._incremental_batches[batch_index][sentence_index][ngram_index]

    def unknown_positions(self):
        """Map currently rare words to their retained structural positions."""
        return {
            word: self.positions_for(word)
            for word in self.vocab.low_frequency_words()
        }

    low_frequency_positions = unknown_positions

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("incremental_vocabulary", False)
        self.__dict__.setdefault("_incremental_active", False)
        self.__dict__.setdefault("_base_counter_state", None)
        self.__dict__.setdefault("promoted_words", [])
        self.__dict__.setdefault("last_promoted_words", [])
        batches = self.__dict__.setdefault("_incremental_batches", [])
        positions = self.__dict__.setdefault("ngram_positions", PositionIndex())
        self.__dict__.setdefault("positions", positions)
        # Rebuild the position index for models pickled with batches but before
        # the index itself was part of the state.
        if not positions and batches:
            for batch_index, batch in enumerate(batches):
                self._record_batch_positions(batch, batch_index)

    def score(self, word, context=None):
        """Masks out of vocab (OOV) words and computes their model score.

        For model-specific logic of calculating scores, see the `unmasked_score`
        method.
        """
        return self.unmasked_score(
            self.vocab.lookup(word), self.vocab.lookup(context) if context else None
        )

    @abstractmethod
    def unmasked_score(self, word, context=None):
        """Score a word given some optional context.

        Concrete models are expected to provide an implementation.
        Note that this method does not mask its arguments with the OOV label.
        Use the `score` method for that.

        :param str word: Word for which we want the score
        :param tuple(str) context: Context the word is in.
            If `None`, compute unigram score.
        :param context: tuple(str) or None
        :rtype: float
        """
        raise NotImplementedError()

    def logscore(self, word, context=None):
        """Evaluate the log score of this word in this context.

        The arguments are the same as for `score` and `unmasked_score`.

        """
        return log_base2(self.score(word, context))

    def context_counts(self, context):
        """Helper method for retrieving counts for a given context.

        Assumes context has been checked and oov words in it masked.
        :type context: tuple(str) or None

        """
        return (
            self.counts[len(context) + 1][context] if context else self.counts.unigrams
        )

    def entropy(self, text_ngrams):
        """Calculate cross-entropy of model for given evaluation text.

        This implementation is based on the Shannon-McMillan-Breiman theorem,
        as used and referenced by Dan Jurafsky and Jordan Boyd-Graber.

        :param Iterable(tuple(str)) text_ngrams: A sequence of ngram tuples.
        :rtype: float

        """
        return -1 * _mean(
            [self.logscore(ngram[-1], ngram[:-1]) for ngram in text_ngrams]
        )

    def perplexity(self, text_ngrams):
        """Calculates the perplexity of the given text.

        This is simply 2 ** cross-entropy for the text, so the arguments are the same.

        """
        return pow(2.0, self.entropy(text_ngrams))

    def generate(self, num_words=1, text_seed=None, random_seed=None):
        """Generate words from the model.

        :param int num_words: How many words to generate. By default 1.
        :param text_seed: Generation can be conditioned on preceding context.
        :param random_seed: A random seed or an instance of `random.Random`. If provided,
            makes the random sampling part of generation reproducible.
        :return: One (str) word or a list of words generated from model.

        Examples:

        >>> from nltk.lm import MLE
        >>> lm = MLE(2)
        >>> lm.fit([[("a", "b"), ("b", "c")]], vocabulary_text=['a', 'b', 'c'])
        >>> lm.fit([[("a",), ("b",), ("c",)]])
        >>> lm.generate(random_seed=3)
        'a'
        >>> lm.generate(text_seed=['a'])
        'b'

        """
        text_seed = [] if text_seed is None else list(text_seed)
        random_generator = _random_generator(random_seed)
        # This is the base recursion case.
        if num_words == 1:
            context = (
                text_seed[-self.order + 1 :]
                if len(text_seed) >= self.order
                else text_seed
            )
            samples = self.context_counts(self.vocab.lookup(context))
            while context and not samples:
                context = context[1:] if len(context) > 1 else []
                samples = self.context_counts(self.vocab.lookup(context))
            # Sorting samples achieves two things:
            # - reproducible randomness when sampling
            # - turns Mapping into Sequence which `_weighted_choice` expects
            samples = sorted(samples)
            return _weighted_choice(
                samples,
                tuple(self.score(w, context) for w in samples),
                random_generator,
            )
        # We build up text one word at a time using the preceding context.
        generated = []
        for _ in range(num_words):
            generated.append(
                self.generate(
                    num_words=1,
                    text_seed=text_seed + generated,
                    random_seed=random_generator,
                )
            )
        return generated
