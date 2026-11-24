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
from dataclasses import dataclass
from itertools import accumulate

from nltk.lm.counter import NgramCounter
from nltk.lm.util import NEG_INF, log_base2
from nltk.lm.vocabulary import Vocabulary


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


@dataclass(frozen=True)
class SentenceScore:
    """Diagnostic score of a single sentence.

    Besides the sentence-level entropy and perplexity, it keeps every piece
    of information needed to track down why an MLE model returns an infinite
    score: the ngrams in evaluation order, their log scores and the ngrams
    whose (un-smoothed) model score is zero.

    A sentence whose history crosses a sentence boundary is scored on its own:
    the ngrams of a sentence never use words from another sentence as context.
    Zero-probability ngrams are reported as-is rather than being patched with a
    non-zero probability, so their log score remains ``-inf``.

    :param tuple(tuple(str)) ngrams: The ngrams the sentence was scored on.
    :param tuple(float) logscores: The log (base 2) score of each ngram, in the
        same order as `ngrams`.
    """

    ngrams: tuple
    logscores: tuple

    def __post_init__(self):
        if not self.ngrams:
            raise ValueError(
                "Cannot score a sentence without any ngrams. "
                "Preprocess the sentence with the same pipeline used for "
                "training (e.g. `padded_everygram_pipeline`)."
            )
        if len(self.ngrams) != len(self.logscores):
            raise ValueError(
                "The number of ngrams and log scores must match. "
                f"Got {len(self.ngrams)} ngrams and {len(self.logscores)} scores."
            )

    def __len__(self):
        """Number of words (scored ngrams) in the sentence."""
        return len(self.ngrams)

    def __iter__(self):
        """Iterate over ``(ngram, logscore)`` pairs."""
        return iter(zip(self.ngrams, self.logscores))

    @property
    def total_logscore(self):
        """Sum of the per-word log scores of the sentence."""
        return math.fsum(self.logscores)

    @property
    def entropy(self):
        """Cross-entropy of the sentence (mean negative log score)."""
        return -self.total_logscore / len(self)

    @property
    def perplexity(self):
        """Perplexity of the sentence (``2 ** entropy``)."""
        return pow(2.0, self.entropy)

    @property
    def zero_probability_ngrams(self):
        """Ngrams (in evaluation order) assigned a zero model score."""
        return tuple(
            ngram
            for ngram, logscore in zip(self.ngrams, self.logscores)
            if logscore == NEG_INF
        )

    @property
    def is_infinite(self):
        """Whether the sentence contains at least one zero-probability ngram."""
        return bool(self.zero_probability_ngrams)


@dataclass(frozen=True)
class TextScore:
    """Aggregate and per-sentence diagnostic scores of an evaluation text.

    Returned by `LanguageModel.evaluate`, it preserves the original
    per-word overall figures (computed over the whole corpus exactly as
    `entropy` and `perplexity` do) alongside the contribution of every
    individual sentence.

    :param tuple(SentenceScore) sentences: One `SentenceScore` per input
        sentence, in evaluation order.
    """

    sentences: tuple

    def __post_init__(self):
        if not self.sentences:
            raise ValueError(
                "Cannot evaluate an empty text: at least one scored sentence "
                "is required."
            )

    def __iter__(self):
        """Iterate over the per-sentence `SentenceScore` objects."""
        return iter(self.sentences)

    def __len__(self):
        """Number of sentences in the evaluation text."""
        return len(self.sentences)

    @property
    def num_words(self):
        """Total number of scored words across all sentences."""
        return sum(len(sentence) for sentence in self.sentences)

    @property
    def total_logscore(self):
        """Sum of every per-word log score in the text."""
        return math.fsum(
            sentence.total_logscore for sentence in self.sentences
        )

    @property
    def entropy(self):
        """Cross-entropy over the whole text (the original per-word figure).

        Identical to calling `LanguageModel.entropy` on the flat stream of
        ngrams.
        """
        return -self.total_logscore / self.num_words

    @property
    def perplexity(self):
        """Perplexity over the whole text (the original per-word figure).

        Identical to calling `LanguageModel.perplexity` on the flat stream of
        ngrams; it stays infinite whenever any ngram has zero probability.
        """
        return pow(2.0, self.entropy)

    @property
    def zero_probability_ngrams(self):
        """All zero-probability ngrams, paired with their sentence index.

        :return: Tuples of ``(sentence_index, ngram)`` in evaluation order,
            allowing the offending word and its context to be located.
        :rtype: tuple(tuple(int, tuple(str)))
        """
        return tuple(
            (index, ngram)
            for index, sentence in enumerate(self.sentences)
            for ngram in sentence.zero_probability_ngrams
        )

    @property
    def is_infinite(self):
        """Whether any sentence contains a zero-probability ngram."""
        return any(sentence.is_infinite for sentence in self.sentences)


class LanguageModel(metaclass=ABCMeta):
    """ABC for Language Models.

    Cannot be directly instantiated itself.

    """

    def __init__(self, order, vocabulary=None, counter=None):
        """Creates new LanguageModel.

        :param vocabulary: If provided, this vocabulary will be used instead
            of creating a new one when training.
        :type vocabulary: `nltk.lm.Vocabulary` or None
        :param counter: If provided, use this object to count ngrams.
        :type counter: `nltk.lm.NgramCounter` or None
        :param ngrams_fn: If given, defines how sentences in training text are turned to ngram
            sequences.
        :type ngrams_fn: function or None
        :param pad_fn: If given, defines how sentences in training text are padded.
        :type pad_fn: function or None
        """
        self.order = order
        if vocabulary and not isinstance(vocabulary, Vocabulary):
            warnings.warn(
                f"The `vocabulary` argument passed to {self.__class__.__name__!r} "
                "must be an instance of `nltk.lm.Vocabulary`.",
                stacklevel=3,
            )
        self.vocab = Vocabulary() if vocabulary is None else vocabulary
        self.counts = NgramCounter() if counter is None else counter

    def fit(self, text, vocabulary_text=None):
        """Trains the model on a text.

        :param text: Training text as a sequence of sentences.

        """
        if not self.vocab:
            if vocabulary_text is None:
                raise ValueError(
                    "Cannot fit without a vocabulary or text to create it from."
                )
            self.vocab.update(vocabulary_text)
        self.counts.update(self.vocab.lookup(sent) for sent in text)

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

    def evaluate(self, text) -> TextScore:
        """Diagnostically score a text sentence by sentence.

        Unlike `entropy` and `perplexity`, which only return a single (often
        infinite) figure, this keeps the contribution of every sentence so that
        the exact ngram responsible for an infinite score can be located
        together with its context.

        Each sentence is scored independently: the ngrams of one sentence never
        use words from another sentence as context, i.e. the history resets at
        every sentence boundary. Zero-probability ngrams are left untouched
        (no smoothing or backoff is added on top of the model), so the overall
        perplexity stays infinite while the remaining sentences keep their
        finite contributions.

        The input may be a single-pass generator: each sentence generator is
        consumed exactly once and its ngrams are materialized while scoring.

        :param Iterable(Iterable(tuple(str))) text: An iterable of sentences,
            where each sentence is itself an iterable of ngram tuples, as
            produced by `nltk.lm.preprocessing.padded_everygram_pipeline`.
        :return: A `TextScore` exposing both the original per-word overall
            entropy/perplexity and the per-sentence contributions.
        :raises ValueError: If the text or any sentence contains no ngrams.
        """

        def score_sentence(sentence_ngrams) -> SentenceScore:
            ngrams = tuple(sentence_ngrams)
            logscores = tuple(
                self.logscore(ngram[-1], ngram[:-1]) for ngram in ngrams
            )
            return SentenceScore(ngrams, logscores)

        sentences = tuple(score_sentence(sentence_ngrams) for sentence_ngrams in text)
        return TextScore(sentences)

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
