# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Tests for per-sentence diagnostic evaluation (`LanguageModel.evaluate`)."""

import math

import pytest

from nltk.lm import MLE, Lidstone, SentenceScore, TextScore
from nltk.lm.preprocessing import padded_everygram_pipeline, padded_everygrams


@pytest.fixture(scope="session")
def mle_model():
    # A bigram MLE trained on the single sentence "a b".
    train, vocab = padded_everygram_pipeline(2, [["a", "b"]])
    model = MLE(2)
    model.fit(train, vocab)
    # "c" is a known word but never appears as a continuation during training.
    model.vocab.update(["c"])
    return model


@pytest.fixture(scope="session")
def test_text():
    return [["a", "b"], ["a", "c"]]


def _pipeline(text):
    ngrams, _ = padded_everygram_pipeline(2, text)
    return ngrams


def test_evaluate_returns_text_score_with_sentence_scores(mle_model, test_text):
    result = mle_model.evaluate(_pipeline(test_text))

    assert isinstance(result, TextScore)
    assert len(result) == 2
    assert all(isinstance(sentence, SentenceScore) for sentence in result)
    # Every padded sentence of "x y" yields 7 scored ngrams (everygrams up to 2).
    assert [len(sentence) for sentence in result] == [7, 7]
    assert result.num_words == 14


def test_first_sentence_keeps_finite_contribution(mle_model, test_text):
    result = mle_model.evaluate(_pipeline(test_text))
    first = result.sentences[0]

    assert not first.is_infinite
    assert first.zero_probability_ngrams == ()
    assert math.isfinite(first.entropy)
    assert math.isfinite(first.perplexity)
    # The seen bigram "a b" has probability 1 -> log score 0.
    assert dict(first)[("a", "b")] == 0.0


def test_second_sentence_locates_zero_probability_ngram(mle_model, test_text):
    result = mle_model.evaluate(_pipeline(test_text))
    second = result.sentences[1]

    assert second.is_infinite
    assert math.isinf(second.entropy)
    assert math.isinf(second.perplexity)
    # The first offending ngram is the unseen continuation "a" -> "c",
    # complete with the context that led to it.
    assert ("a", "c") in second.zero_probability_ngrams
    assert result.zero_probability_ngrams[0] == (1, ("a", "c"))
    # Every reported location carries the index of the sentence it belongs to.
    assert all(index == 1 for index, _ in result.zero_probability_ngrams)


def test_overall_figures_stay_infinite_and_match_flat_call(mle_model, test_text):
    result = mle_model.evaluate(_pipeline(test_text))
    flat = [
        ngram
        for sentence in test_text
        for ngram in padded_everygrams(2, sentence)
    ]

    assert result.is_infinite
    assert math.isinf(result.entropy)
    assert math.isinf(result.perplexity)
    # Same per-word overall figures as the original flat ngram call.
    assert result.entropy == mle_model.entropy(list(flat))
    assert result.perplexity == mle_model.perplexity(list(flat))


def test_evaluate_accepts_single_pass_generator(mle_model, test_text):
    ngrams, _ = padded_everygram_pipeline(2, test_text)

    def single_pass():
        yield from ngrams

    result = mle_model.evaluate(single_pass())
    # Consuming the generator once must be enough to build the full report.
    assert len(result) == 2
    assert result.sentences[0].ngrams
    assert result.sentences[1].is_infinite


def test_history_resets_at_sentence_boundary(mle_model):
    # Each sentence is scored on its own padded ngrams; no ngram may cross a
    # sentence boundary (there is never an "</s>" -> "<s>" continuation).
    result = mle_model.evaluate(_pipeline([["a", "b"], ["a", "b"]]))
    for sentence in result:
        assert ("</s>", "<s>") not in sentence.ngrams
        # Every sentence is independently padded with its own start marker.
        assert sentence.ngrams[0] == ("<s>",)
    assert not result.is_infinite
    # Both isolated sentences are identical, so their contributions match.
    assert result.sentences[0].total_logscore == result.sentences[1].total_logscore


def test_clean_text_matches_flat_entropy_and_is_finite(mle_model):
    text = [["a", "b"]]
    result = mle_model.evaluate(_pipeline(text))
    flat = list(padded_everygrams(2, text[0]))

    assert math.isfinite(result.perplexity)
    assert result.entropy == mle_model.entropy(flat)
    assert result.perplexity == mle_model.perplexity(flat)


def test_smoothed_model_has_no_zero_probabilities(test_text):
    # A smoothed model never assigns zero, so the report must be fully finite.
    train, vocab = padded_everygram_pipeline(2, [["a", "b"]])
    model = Lidstone(0.1, order=2)
    model.fit(train, vocab)
    model.vocab.update(["c"])

    result = model.evaluate(_pipeline(test_text))
    assert not result.is_infinite
    assert result.zero_probability_ngrams == ()
    assert all(math.isfinite(s.perplexity) for s in result)
    flat = [
        ngram
        for sentence in test_text
        for ngram in padded_everygrams(2, sentence)
    ]
    assert result.entropy == pytest.approx(model.entropy(flat))


def test_sentence_score_iterates_ngram_logscore_pairs(mle_model):
    result = mle_model.evaluate(_pipeline([["a", "b"]]))
    pairs = list(result.sentences[0])
    assert pairs == list(zip(result.sentences[0].ngrams, result.sentences[0].logscores))


def test_empty_text_raises(mle_model):
    with pytest.raises(ValueError):
        mle_model.evaluate(iter([]))


def test_empty_sentence_raises(mle_model):
    with pytest.raises(ValueError):
        mle_model.evaluate(iter([iter([])]))
