# Natural Language Toolkit: Language Model Unit Tests
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
import math
import random
from math import fsum as sum
from operator import itemgetter

import pytest

from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,
    Laplace,
    Lidstone,
    StupidBackoff,
    Vocabulary,
    WittenBellInterpolated,
)
from nltk.lm.preprocessing import padded_everygrams


@pytest.fixture(scope="session")
def vocabulary():
    return Vocabulary(["a", "b", "c", "d", "z", "<s>", "</s>"], unk_cutoff=1)


@pytest.fixture(scope="session")
def training_data():
    return [["a", "b", "c", "d"], ["e", "g", "a", "d", "b", "e"]]


@pytest.fixture(scope="session")
def bigram_training_data(training_data):
    return [list(padded_everygrams(2, sent)) for sent in training_data]


@pytest.fixture(scope="session")
def trigram_training_data(training_data):
    return [list(padded_everygrams(3, sent)) for sent in training_data]


@pytest.fixture
def mle_bigram_model(vocabulary, bigram_training_data):
    model = MLE(2, vocabulary=vocabulary)
    model.fit(bigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        ("d", ["c"], 1),
        # Unseen ngrams should yield 0
        ("d", ["e"], 0),
        # Unigrams should also be 0
        ("z", None, 0),
        # N unigrams = 14
        # count('a') = 2
        ("a", None, 2.0 / 14),
        # count('y') = 3
        ("y", None, 3.0 / 14),
    ],
)
def test_mle_bigram_scores(mle_bigram_model, word, context, expected_score):
    assert pytest.approx(mle_bigram_model.score(word, context), 1e-4) == expected_score


def test_mle_bigram_logscore_for_zero_score(mle_bigram_model):
    assert math.isinf(mle_bigram_model.logscore("d", ["e"]))


def test_mle_bigram_entropy_perplexity_seen(mle_bigram_model):
    # ngrams seen during training
    trained = [
        ("<s>", "a"),
        ("a", "b"),
        ("b", "<UNK>"),
        ("<UNK>", "a"),
        ("a", "d"),
        ("d", "</s>"),
    ]
    # Ngram = Log score
    # <s>, a    = -1
    # a, b      = -1
    # b, UNK    = -1
    # UNK, a    = -1.585
    # a, d      = -1
    # d, </s>   = -1
    # TOTAL logscores   = -6.585
    # - AVG logscores   = 1.0975
    H = 1.0975
    perplexity = 2.1398
    assert pytest.approx(mle_bigram_model.entropy(trained), 1e-4) == H
    assert pytest.approx(mle_bigram_model.perplexity(trained), 1e-4) == perplexity


def test_mle_bigram_entropy_perplexity_unseen(mle_bigram_model):
    # In MLE, even one unseen ngram should make entropy and perplexity infinite
    untrained = [("<s>", "a"), ("a", "c"), ("c", "d"), ("d", "</s>")]

    assert math.isinf(mle_bigram_model.entropy(untrained))
    assert math.isinf(mle_bigram_model.perplexity(untrained))


def test_mle_bigram_entropy_perplexity_unigrams(mle_bigram_model):
    # word = score, log score
    # <s>   = 0.1429, -2.8074
    # a     = 0.1429, -2.8074
    # c     = 0.0714, -3.8073
    # UNK   = 0.2143, -2.2224
    # d     = 0.1429, -2.8074
    # c     = 0.0714, -3.8073
    # </s>  = 0.1429, -2.8074
    # TOTAL logscores = -21.6243
    # - AVG logscores = 3.0095
    H = 3.0095
    perplexity = 8.0529

    text = [("<s>",), ("a",), ("c",), ("-",), ("d",), ("c",), ("</s>",)]

    assert pytest.approx(mle_bigram_model.entropy(text), 1e-4) == H
    assert pytest.approx(mle_bigram_model.perplexity(text), 1e-4) == perplexity


@pytest.fixture
def mle_trigram_model(trigram_training_data, vocabulary):
    model = MLE(order=3, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # count(d | b, c) = 1
        # count(b, c) = 1
        ("d", ("b", "c"), 1),
        # count(d | c) = 1
        # count(c) = 1
        ("d", ["c"], 1),
        # total number of tokens is 18, of which "a" occurred 2 times
        ("a", None, 2.0 / 18),
        # in vocabulary but unseen
        ("z", None, 0),
        # out of vocabulary should use "UNK" score
        ("y", None, 3.0 / 18),
    ],
)
def test_mle_trigram_scores(mle_trigram_model, word, context, expected_score):
    assert pytest.approx(mle_trigram_model.score(word, context), 1e-4) == expected_score


@pytest.fixture
def lidstone_bigram_model(bigram_training_data, vocabulary):
    model = Lidstone(0.1, order=2, vocabulary=vocabulary)
    model.fit(bigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # count(d | c) = 1
        # *count(d | c) = 1.1
        # Count(w | c for w in vocab) = 1
        # *Count(w | c for w in vocab) = 1.8
        ("d", ["c"], 1.1 / 1.8),
        # Total unigrams: 14
        # Vocab size: 8
        # Denominator: 14 + 0.8 = 14.8
        # count("a") = 2
        # *count("a") = 2.1
        ("a", None, 2.1 / 14.8),
        # in vocabulary but unseen
        # count("z") = 0
        # *count("z") = 0.1
        ("z", None, 0.1 / 14.8),
        # out of vocabulary should use "UNK" score
        # count("<UNK>") = 3
        # *count("<UNK>") = 3.1
        ("y", None, 3.1 / 14.8),
    ],
)
def test_lidstone_bigram_score(lidstone_bigram_model, word, context, expected_score):
    assert (
        pytest.approx(lidstone_bigram_model.score(word, context), 1e-4)
        == expected_score
    )


def test_lidstone_entropy_perplexity(lidstone_bigram_model):
    text = [
        ("<s>", "a"),
        ("a", "c"),
        ("c", "<UNK>"),
        ("<UNK>", "d"),
        ("d", "c"),
        ("c", "</s>"),
    ]
    # Unlike MLE this should be able to handle completely novel ngrams
    # Ngram = score, log score
    # <s>, a    = 0.3929, -1.3479
    # a, c      = 0.0357, -4.8074
    # c, UNK    = 0.0(5), -4.1699
    # UNK, d    = 0.0263,  -5.2479
    # d, c      = 0.0357, -4.8074
    # c, </s>   = 0.0(5), -4.1699
    # TOTAL logscore: −24.5504
    # - AVG logscore: 4.0917
    H = 4.0917
    perplexity = 17.0504
    assert pytest.approx(lidstone_bigram_model.entropy(text), 1e-4) == H
    assert pytest.approx(lidstone_bigram_model.perplexity(text), 1e-4) == perplexity


@pytest.fixture
def lidstone_trigram_model(trigram_training_data, vocabulary):
    model = Lidstone(0.1, order=3, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # Logic behind this is the same as for bigram model
        ("d", ["c"], 1.1 / 1.8),
        # if we choose a word that hasn't appeared after (b, c)
        ("e", ["c"], 0.1 / 1.8),
        # Trigram score now
        ("d", ["b", "c"], 1.1 / 1.8),
        ("e", ["b", "c"], 0.1 / 1.8),
    ],
)
def test_lidstone_trigram_score(lidstone_trigram_model, word, context, expected_score):
    assert (
        pytest.approx(lidstone_trigram_model.score(word, context), 1e-4)
        == expected_score
    )


@pytest.fixture
def laplace_bigram_model(bigram_training_data, vocabulary):
    model = Laplace(2, vocabulary=vocabulary)
    model.fit(bigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # basic sanity-check:
        # count(d | c) = 1
        # *count(d | c) = 2
        # Count(w | c for w in vocab) = 1
        # *Count(w | c for w in vocab) = 9
        ("d", ["c"], 2.0 / 9),
        # Total unigrams: 14
        # Vocab size: 8
        # Denominator: 14 + 8 = 22
        # count("a") = 2
        # *count("a") = 3
        ("a", None, 3.0 / 22),
        # in vocabulary but unseen
        # count("z") = 0
        # *count("z") = 1
        ("z", None, 1.0 / 22),
        # out of vocabulary should use "UNK" score
        # count("<UNK>") = 3
        # *count("<UNK>") = 4
        ("y", None, 4.0 / 22),
    ],
)
def test_laplace_bigram_score(laplace_bigram_model, word, context, expected_score):
    assert (
        pytest.approx(laplace_bigram_model.score(word, context), 1e-4) == expected_score
    )


def test_laplace_bigram_entropy_perplexity(laplace_bigram_model):
    text = [
        ("<s>", "a"),
        ("a", "c"),
        ("c", "<UNK>"),
        ("<UNK>", "d"),
        ("d", "c"),
        ("c", "</s>"),
    ]
    # Unlike MLE this should be able to handle completely novel ngrams
    # Ngram = score, log score
    # <s>, a    = 0.2, -2.3219
    # a, c      = 0.1, -3.3219
    # c, UNK    = 0.(1), -3.1699
    # UNK, d    = 0.(09), 3.4594
    # d, c      = 0.1 -3.3219
    # c, </s>   = 0.(1), -3.1699
    # Total logscores: −18.7651
    # - AVG logscores: 3.1275
    H = 3.1275
    perplexity = 8.7393
    assert pytest.approx(laplace_bigram_model.entropy(text), 1e-4) == H
    assert pytest.approx(laplace_bigram_model.perplexity(text), 1e-4) == perplexity


def test_laplace_gamma(laplace_bigram_model):
    assert laplace_bigram_model.gamma == 1


@pytest.fixture
def wittenbell_trigram_model(trigram_training_data, vocabulary):
    model = WittenBellInterpolated(3, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # For unigram scores by default revert to regular MLE
        # Total unigrams: 18
        # Vocab Size = 7
        # count('c'): 1
        ("c", None, 1.0 / 18),
        # in vocabulary but unseen
        # count("z") = 0
        ("z", None, 0 / 18),
        # out of vocabulary should use "UNK" score
        # count("<UNK>") = 3
        ("y", None, 3.0 / 18),
        # 2 words follow b and b occurred a total of 2 times
        # gamma(['b']) = 2 / (2 + 2) = 0.5
        # mle.score('c', ['b']) = 0.5
        # mle('c') = 1 / 18 = 0.055
        # (1 - gamma) * mle + gamma * mle('c') ~= 0.27 + 0.055
        ("c", ["b"], (1 - 0.5) * 0.5 + 0.5 * 1 / 18),
        # building on that, let's try 'a b c' as the trigram
        # 1 word follows 'a b' and 'a b' occurred 1 time
        # gamma(['a', 'b']) = 1 / (1 + 1) = 0.5
        # mle("c", ["a", "b"]) = 1
        ("c", ["a", "b"], (1 - 0.5) + 0.5 * ((1 - 0.5) * 0.5 + 0.5 * 1 / 18)),
        # P(c|zb)
        # The ngram 'zbc' was not seen, so we use P(c|b). See issue #2332.
        ("c", ["z", "b"], ((1 - 0.5) * 0.5 + 0.5 * 1 / 18)),
    ],
)
def test_wittenbell_trigram_score(
    wittenbell_trigram_model, word, context, expected_score
):
    assert (
        pytest.approx(wittenbell_trigram_model.score(word, context), 1e-4)
        == expected_score
    )


###############################################################################
#                              Notation Explained                             #
###############################################################################
# For all subsequent calculations we use the following notation:
# 1. '*': Placeholder for any word/character. E.g. '*b' stands for
#    all bigrams that end in 'b'. '*b*' stands for all trigrams that
#    contain 'b' in the middle.
# 1. count(ngram): Count all instances (tokens) of an ngram.
# 1. unique(ngram): Count unique instances (types) of an ngram.


@pytest.fixture
def kneserney_trigram_model(trigram_training_data, vocabulary):
    model = KneserNeyInterpolated(order=3, discount=0.75, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # P(c) = count('*c') / unique('**')
        #      = 1 / 14
        ("c", None, 1.0 / 14),
        # P(z) = count('*z') / unique('**')
        #      = 0 / 14
        # 'z' is in the vocabulary, but it was not seen during training.
        ("z", None, 0.0 / 14),
        # P(y)
        # Out of vocabulary should use "UNK" score.
        # P(y) = P(UNK) = count('*UNK') / unique('**')
        ("y", None, 3 / 14),
        # We start with P(c|b)
        # P(c|b) = alpha('bc') + gamma('b') * P(c)
        # alpha('bc') = max(unique('*bc') - discount, 0) / unique('*b*')
        #             = max(1 - 0.75, 0) / 2
        #             = 0.125
        # gamma('b')  = discount * unique('b*') / unique('*b*')
        #             = (0.75 * 2) / 2
        #             = 0.75
        ("c", ["b"], (0.125 + 0.75 * (1 / 14))),
        # Building on that, let's try P(c|ab).
        # P(c|ab) = alpha('abc') + gamma('ab') * P(c|b)
        # alpha('abc') = max(count('abc') - discount, 0) / count('ab*')
        #              = max(1 - 0.75, 0) / 1
        #              = 0.25
        # gamma('ab')  = (discount * unique('ab*')) / count('ab*')
        #              = 0.75 * 1 / 1
        ("c", ["a", "b"], 0.25 + 0.75 * (0.125 + 0.75 * (1 / 14))),
        # P(c|zb)
        # The ngram 'zbc' was not seen, so we use P(c|b). See issue #2332.
        ("c", ["z", "b"], (0.125 + 0.75 * (1 / 14))),
    ],
)
def test_kneserney_trigram_score(
    kneserney_trigram_model, word, context, expected_score
):
    assert (
        pytest.approx(kneserney_trigram_model.score(word, context), 1e-4)
        == expected_score
    )


def test_kneserney_scores_follow_later_fit(vocabulary, trigram_training_data):
    # Continuation counts are cached between scores; a second fit must refresh them.
    model = KneserNeyInterpolated(order=3, discount=0.75, vocabulary=vocabulary)
    model.fit(trigram_training_data[:1])
    model.score("c", ["a", "b"])
    model.fit(trigram_training_data[1:])

    expected = KneserNeyInterpolated(order=3, discount=0.75, vocabulary=vocabulary)
    expected.fit(trigram_training_data)
    for word, context in [
        ("c", ["a", "b"]),
        ("d", ["c"]),
        ("b", ["a", "d"]),
        ("e", []),
    ]:
        assert model.score(word, context) == expected.score(word, context)


@pytest.fixture
def absolute_discounting_trigram_model(trigram_training_data, vocabulary):
    model = AbsoluteDiscountingInterpolated(order=3, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # For unigram scores revert to uniform
        # P(c) = count('c') / count('**')
        ("c", None, 1.0 / 18),
        # in vocabulary but unseen
        # count('z') = 0
        ("z", None, 0.0 / 18),
        # out of vocabulary should use "UNK" score
        # count('<UNK>') = 3
        ("y", None, 3 / 18),
        # P(c|b) = alpha('bc') + gamma('b') * P(c)
        # alpha('bc') = max(count('bc') - discount, 0) / count('b*')
        #             = max(1 - 0.75, 0) / 2
        #             = 0.125
        # gamma('b')  = discount * unique('b*') / count('b*')
        #             = (0.75 * 2) / 2
        #             = 0.75
        ("c", ["b"], (0.125 + 0.75 * (2 / 2) * (1 / 18))),
        # Building on that, let's try P(c|ab).
        # P(c|ab) = alpha('abc') + gamma('ab') * P(c|b)
        # alpha('abc') = max(count('abc') - discount, 0) / count('ab*')
        #              = max(1 - 0.75, 0) / 1
        #              = 0.25
        # gamma('ab')  = (discount * unique('ab*')) / count('ab*')
        #              = 0.75 * 1 / 1
        ("c", ["a", "b"], 0.25 + 0.75 * (0.125 + 0.75 * (2 / 2) * (1 / 18))),
        # P(c|zb)
        # The ngram 'zbc' was not seen, so we use P(c|b). See issue #2332.
        ("c", ["z", "b"], (0.125 + 0.75 * (2 / 2) * (1 / 18))),
    ],
)
def test_absolute_discounting_trigram_score(
    absolute_discounting_trigram_model, word, context, expected_score
):
    assert (
        pytest.approx(absolute_discounting_trigram_model.score(word, context), 1e-4)
        == expected_score
    )


@pytest.fixture
def stupid_backoff_trigram_model(trigram_training_data, vocabulary):
    model = StupidBackoff(order=3, vocabulary=vocabulary)
    model.fit(trigram_training_data)
    return model


@pytest.mark.parametrize(
    "word, context, expected_score",
    [
        # For unigram scores revert to uniform
        # total bigrams = 18
        ("c", None, 1.0 / 18),
        # in vocabulary but unseen
        # bigrams ending with z = 0
        ("z", None, 0.0 / 18),
        # out of vocabulary should use "UNK" score
        # count('<UNK>'): 3
        ("y", None, 3 / 18),
        # c follows 1 time out of 2 after b
        ("c", ["b"], 1 / 2),
        # c always follows ab
        ("c", ["a", "b"], 1 / 1),
        # The ngram 'z b c' was not seen, so we backoff to
        # the score of the ngram 'b c' * smoothing factor
        ("c", ["z", "b"], (0.4 * (1 / 2))),
    ],
)
def test_stupid_backoff_trigram_score(
    stupid_backoff_trigram_model, word, context, expected_score
):
    assert (
        pytest.approx(stupid_backoff_trigram_model.score(word, context), 1e-4)
        == expected_score
    )


###############################################################################
#               Probability Distributions Should Sum up to Unity              #
###############################################################################


@pytest.fixture(scope="session")
def kneserney_bigram_model(bigram_training_data, vocabulary):
    model = KneserNeyInterpolated(order=2, vocabulary=vocabulary)
    model.fit(bigram_training_data)
    return model


@pytest.mark.parametrize(
    "model_fixture",
    [
        "mle_bigram_model",
        "mle_trigram_model",
        "lidstone_bigram_model",
        "laplace_bigram_model",
        "wittenbell_trigram_model",
        "absolute_discounting_trigram_model",
        "kneserney_bigram_model",
        pytest.param(
            "stupid_backoff_trigram_model",
            marks=pytest.mark.xfail(
                reason="Stupid Backoff is not a valid distribution"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "context",
    [("a",), ("c",), ("<s>",), ("b",), ("<UNK>",), ("d",), ("e",), ("r",), ("w",)],
    ids=itemgetter(0),
)
def test_sums_to_1(model_fixture, context, request):
    model = request.getfixturevalue(model_fixture)
    scores_for_context = sum(model.score(w, context) for w in model.vocab)
    assert pytest.approx(scores_for_context, 1e-7) == 1.0


###############################################################################
#                               Generating Text                               #
###############################################################################


def test_generate_one_no_context(mle_trigram_model):
    assert mle_trigram_model.generate(random_seed=3) == "<UNK>"


def test_generate_one_from_limiting_context(mle_trigram_model):
    # We don't need random_seed for contexts with only one continuation
    assert mle_trigram_model.generate(text_seed=["c"]) == "d"
    assert mle_trigram_model.generate(text_seed=["b", "c"]) == "d"
    assert mle_trigram_model.generate(text_seed=["a", "c"]) == "d"


def test_generate_one_from_varied_context(mle_trigram_model):
    # When context doesn't limit our options enough, seed the random choice
    assert mle_trigram_model.generate(text_seed=("a", "<s>"), random_seed=2) == "a"


def test_generate_cycle(mle_trigram_model):
    # Add a cycle to the model: bd -> b, db -> d
    more_training_text = [padded_everygrams(mle_trigram_model.order, list("bdbdbd"))]

    mle_trigram_model.fit(more_training_text)
    # Test that we can escape the cycle
    assert mle_trigram_model.generate(7, text_seed=("b", "d"), random_seed=5) == [
        "b",
        "d",
        "b",
        "d",
        "b",
        "d",
        "</s>",
    ]


def test_generate_with_text_seed(mle_trigram_model):
    assert mle_trigram_model.generate(5, text_seed=("<s>", "e"), random_seed=3) == [
        "<UNK>",
        "a",
        "d",
        "b",
        "<UNK>",
    ]


def test_generate_oov_text_seed(mle_trigram_model):
    assert mle_trigram_model.generate(
        text_seed=("aliens",), random_seed=3
    ) == mle_trigram_model.generate(text_seed=("<UNK>",), random_seed=3)


def test_generate_None_text_seed(mle_trigram_model):
    # should crash with type error when we try to look it up in vocabulary
    with pytest.raises(TypeError):
        mle_trigram_model.generate(text_seed=(None,))

    # This will work
    assert mle_trigram_model.generate(
        text_seed=None, random_seed=3
    ) == mle_trigram_model.generate(random_seed=3)


###############################################################################
#                Generating Text Restricted to Given Candidates               #
###############################################################################


@pytest.fixture
def restricted_vocabulary():
    # "b" and "c" are vocabulary items; "x" below is out of vocabulary and so
    # gets looked up as the unknown label while fitting.
    return Vocabulary(["a", "b", "c", "<s>", "</s>"], unk_cutoff=1)


@pytest.fixture
def restricted_training_data():
    # "a" is only ever followed by "b".
    return [list(padded_everygrams(2, ["a", "b", "x", "b"]))]


@pytest.fixture
def restricted_wb_model(restricted_vocabulary, restricted_training_data):
    model = WittenBellInterpolated(2, vocabulary=restricted_vocabulary)
    model.fit(restricted_training_data)
    return model


@pytest.fixture
def restricted_mle_model(restricted_vocabulary, restricted_training_data):
    model = MLE(2, vocabulary=restricted_vocabulary)
    model.fit(restricted_training_data)
    return model


def test_generate_candidates_single_continuation(
    restricted_wb_model, restricted_mle_model
):
    # Allowing only the single observed continuation must always produce it,
    # independently of the random stream.
    for model in (restricted_wb_model, restricted_mle_model):
        for seed in range(10):
            assert (
                model.generate(text_seed=["a"], random_seed=seed, candidates=["b"])
                == "b"
            )


def test_generate_candidates_wb_unseen_in_vocab_word(restricted_wb_model):
    # "c" is a vocabulary item but never seen after "a"; Witten-Bell reserves
    # unseen probability mass for the unknown label rather than for "c".
    assert restricted_wb_model.score("c", ["a"]) == 0
    with pytest.raises(ValueError):
        restricted_wb_model.generate(
            text_seed=["a"], random_seed=1, candidates=["c"]
        )


def test_generate_candidates_mle_zero_score_raises(restricted_mle_model):
    # MLE assigns zero to the unseen continuation "c".
    assert restricted_mle_model.score("c", ["a"]) == 0
    with pytest.raises(ValueError):
        restricted_mle_model.generate(
            text_seed=["a"], random_seed=1, candidates=["c"]
        )


def test_generate_candidates_does_not_shorten_context(restricted_mle_model):
    # The model must not quietly drop the history to make "c" sampleable.
    with pytest.raises(ValueError):
        restricted_mle_model.generate(
            text_seed=["a"], random_seed=1, candidates=["c"]
        )

    # An unseen context is treated the same way: no backoff to a shorter
    # history that would give the candidate a positive MLE score.
    model = MLE(2, vocabulary=restricted_mle_model.vocab)
    model.fit([list(padded_everygrams(2, ["c", "b"]))])
    with pytest.raises(ValueError):
        model.generate(text_seed=["z"], random_seed=1, candidates=["c"])


def test_generate_candidates_wb_oov_uses_unknown_label(restricted_wb_model):
    # The unseen word receives the unknown label's smoothed probability mass
    # and the generated token is the model's unknown label.
    unk_label = restricted_wb_model.vocab.unk_label
    assert restricted_wb_model.score("mars", ["a"]) > 0
    assert (
        restricted_wb_model.generate(
            text_seed=["a"], random_seed=1, candidates=["mars"]
        )
        == unk_label
    )


def test_generate_candidates_oov_collapses_to_single_unknown(restricted_wb_model):
    # Two unseen candidates both look up to the same unknown symbol, so they
    # carry a single share of the probability mass and the output is the
    # model's unknown label.
    unk_label = restricted_wb_model.vocab.unk_label
    for seed in range(10):
        assert (
            restricted_wb_model.generate(
                text_seed=["a"],
                random_seed=seed,
                candidates=["mars", "venus"],
            )
            == unk_label
        )


def test_generate_candidates_mle_oov_raises(restricted_mle_model):
    # MLE has no probability mass for unknown continuations.
    assert restricted_mle_model.score("mars", ["a"]) == 0
    with pytest.raises(ValueError):
        restricted_mle_model.generate(
            text_seed=["a"], random_seed=1, candidates=["mars"]
        )


def test_generate_candidates_mixed_sample_both(restricted_wb_model):
    unk_label = restricted_wb_model.vocab.unk_label
    outcomes = {
        restricted_wb_model.generate(
            text_seed=["a"], random_seed=seed, candidates=["b", "mars"]
        )
        for seed in range(40)
    }
    assert outcomes == {"b", unk_label}


def test_generate_candidates_empty_raises(restricted_wb_model):
    with pytest.raises(ValueError):
        restricted_wb_model.generate(text_seed=["a"], candidates=[])


def test_generate_candidates_reproducible(restricted_wb_model):
    first = restricted_wb_model.generate(
        5, text_seed=["a"], random_seed=42, candidates=["b", "mars"]
    )
    second = restricted_wb_model.generate(
        5, text_seed=["a"], random_seed=42, candidates=["b", "mars"]
    )
    assert first == second


def test_generate_candidates_multi_word_uses_single_random_stream(restricted_wb_model):
    # Reproduce the multi-word call step by step while sharing one Random,
    # relocating the history from the words already generated at each step.
    generator = random.Random(7)
    manual, history = [], ["a"]
    for _ in range(4):
        word = restricted_wb_model.generate(
            text_seed=history,
            random_seed=generator,
            candidates=["b", "mars"],
        )
        manual.append(word)
        history = history + [word]

    assert restricted_wb_model.generate(
        4, text_seed=["a"], random_seed=7, candidates=["b", "mars"]
    ) == manual


def test_generate_without_candidates_keeps_observed_range(
    restricted_wb_model, restricted_mle_model
):
    # Without candidates, generation still samples observed continuations and
    # never reaches the smoothed unknown mass.
    for model in (restricted_wb_model, restricted_mle_model):
        for seed in range(10):
            assert model.generate(text_seed=["a"], random_seed=seed) == "b"


def test_stupid_backoff_unigram_long_context_matches_empty_context(vocabulary):
    model = StupidBackoff(order=1, vocabulary=vocabulary)
    model.fit([[("a",)], [("b",)], [("a",)]])

    long_context = tuple("x" for _ in range(10000))
    assert model.score("a", long_context) == model.score("a", ())


def test_witten_bell_unigram_long_context_matches_empty_context(vocabulary):
    model = WittenBellInterpolated(order=1, vocabulary=vocabulary)
    model.fit([[("a",)], [("b",)], [("a",)]])

    long_context = tuple("x" for _ in range(10000))
    assert model.score("a", long_context) == model.score("a", ())
