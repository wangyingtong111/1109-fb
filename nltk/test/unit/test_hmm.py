import pytest

from nltk.tag import hmm


def _wikipedia_example_hmm():
    # Example from wikipedia
    # (https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm)

    states = ["rain", "no rain"]
    symbols = ["umbrella", "no umbrella"]

    A = [[0.7, 0.3], [0.3, 0.7]]  # transition probabilities
    B = [[0.9, 0.1], [0.2, 0.8]]  # emission probabilities
    pi = [0.5, 0.5]  # initial probabilities

    seq = ["umbrella", "umbrella", "no umbrella", "umbrella", "umbrella"]
    seq = list(zip(seq, [None] * len(seq)))

    model = hmm._create_hmm_tagger(states, symbols, A, B, pi)
    return model, states, symbols, seq


def test_forward_probability():
    from numpy.testing import assert_array_almost_equal

    # example from p. 385, Huang et al
    model, states, symbols = hmm._market_hmm_example()
    seq = [("up", None), ("up", None)]
    expected = [[0.35, 0.02, 0.09], [0.1792, 0.0085, 0.0357]]

    fp = 2 ** model._forward_probability(seq)

    assert_array_almost_equal(fp, expected)


def test_forward_probability2():
    from numpy.testing import assert_array_almost_equal

    model, states, symbols, seq = _wikipedia_example_hmm()
    fp = 2 ** model._forward_probability(seq)

    # examples in wikipedia are normalized
    fp = (fp.T / fp.sum(axis=1)).T

    wikipedia_results = [
        [0.8182, 0.1818],
        [0.8834, 0.1166],
        [0.1907, 0.8093],
        [0.7308, 0.2692],
        [0.8673, 0.1327],
    ]

    assert_array_almost_equal(wikipedia_results, fp, 4)


def test_backward_probability():
    from numpy.testing import assert_array_almost_equal

    model, states, symbols, seq = _wikipedia_example_hmm()

    bp = 2 ** model._backward_probability(seq)
    # examples in wikipedia are normalized

    bp = (bp.T / bp.sum(axis=1)).T

    wikipedia_results = [
        # Forward-backward algorithm doesn't need b0_5,
        # so .backward_probability doesn't compute it.
        # [0.6469, 0.3531],
        [0.5923, 0.4077],
        [0.3763, 0.6237],
        [0.6533, 0.3467],
        [0.6273, 0.3727],
        [0.5, 0.5],
    ]

    assert_array_almost_equal(wikipedia_results, bp, 4)


def test_best_path_matches_simple():
    model, states, symbols = hmm._market_hmm_example()
    for seq in (["up"], ["up", "up", "down"], ["up", "down", "down", "up"]):
        assert model.best_path(seq) == model.best_path_simple(seq)


def test_tag_repeated_unseen_word():
    from nltk.probability import LidstoneProbDist

    train = [
        [("the", "DT"), ("dog", "NN"), ("runs", "VB")],
        [("a", "DT"), ("cat", "NN"), ("sleeps", "VB")],
    ]
    tagger = hmm.HiddenMarkovModelTrainer().train_supervised(
        train, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)
    )
    words = ["the", "blorp", "blorp", "runs"]
    expected = tagger.best_path_simple(words)
    assert [tag for _, tag in tagger.tag(words)] == expected
    assert tagger._symbols.count("blorp") == 1


def _bounded_training_data():
    # the training alphabet is deliberately restricted to the symbols a, b
    return [
        [("a", "A"), ("b", "B")],
        [("b", "B"), ("a", "A")],
        [("a", "A"), ("a", "A"), ("b", "B")],
    ]


def _lidstone_estimator():
    from nltk.probability import LidstoneProbDist

    return lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)


def test_unknown_cache_size_validation():
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(
        _bounded_training_data(), estimator=_lidstone_estimator()
    )
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            hmm.HiddenMarkovModelTagger(
                tagger._symbols,
                tagger._states,
                tagger._transitions,
                tagger._outputs,
                tagger._priors,
                unknown_cache_size=bad,
            )


def test_unknown_cache_keeps_symbol_table_fixed_and_bounds_columns():
    tagger = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(),
        estimator=_lidstone_estimator(),
        unknown_cache_size=2,
    )
    assert tagger._symbols == ["a", "b"]

    sentence = [f"u{i}" for i in range(1, 21)]
    tagger.best_path(sentence)

    # the training symbol table must never absorb prediction-time words
    assert tagger._symbols == ["a", "b"]
    # at most two unseen words stay resident in the cache
    assert len(tagger._unknown_symbol_order) <= 2
    assert len(tagger._unknown_symbol_map) <= 2
    # the emission matrix keeps exactly two reusable unknown-word columns
    assert tagger._cache[1].shape[1] == len(tagger._symbols) + 2


def test_unknown_cache_path_matches_uncached_reference():
    bounded = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(),
        estimator=_lidstone_estimator(),
        unknown_cache_size=2,
    )
    # an unbounded tagger exercises the pre-existing cached Viterbi path,
    # while best_path_simple never uses the cache at all
    unbounded = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(), estimator=_lidstone_estimator()
    )

    sentence = [f"u{i}" for i in range(1, 21)]
    expected = unbounded.best_path(sentence)
    assert bounded.best_path(sentence) == expected
    assert bounded.best_path(sentence) == bounded.best_path_simple(sentence)


def test_unknown_cache_repeated_word_shares_column_and_survives_eviction():
    tagger = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(),
        estimator=_lidstone_estimator(),
        unknown_cache_size=2,
    )
    sentence = ["u1", "u1", "u2", "u1", "u3", "u1"]
    expected = tagger.best_path_simple(sentence)

    assert tagger.best_path(sentence) == expected
    # every occurrence of u1 resolves to the same resident column
    u1_columns = {tagger._cached_column("u1") for _ in sentence}
    assert len(u1_columns) == 1
    assert len(tagger._unknown_symbol_order) <= 2


def test_unknown_cache_preserves_estimator_scores():
    import numpy as np

    tagger = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(),
        estimator=_lidstone_estimator(),
        unknown_cache_size=2,
    )
    # fill the cache, then force an eviction and refill a recycled column
    tagger.best_path(["u1", "u2", "u3"])
    column = tagger._cache_unknown_symbol("u4")

    output_matrix = tagger._cache[1]
    for i, state in enumerate(tagger._states):
        live_score = tagger._output_logprob(state, "u4")
        cached_score = output_matrix[i, column]
        assert np.isclose(live_score, cached_score)


def test_unknown_cache_reset_rebuilds_consistently():
    tagger = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(),
        estimator=_lidstone_estimator(),
        unknown_cache_size=2,
    )
    sentence = ["u1", "u1", "u2", "u1", "u3", "u1"]
    first_path = tagger.best_path(sentence)

    tagger.reset_cache()
    assert tagger._cache is None
    assert tagger._unknown_symbol_map == {}
    assert tagger._symbols == ["a", "b"]

    assert tagger.best_path(sentence) == first_path
    assert tagger._symbols == ["a", "b"]


def test_default_cache_still_grows_symbol_table():
    tagger = hmm.HiddenMarkovModelTagger.train(
        _bounded_training_data(), estimator=_lidstone_estimator()
    )
    tagger.best_path(["u1", "u2", "u3"])
    assert "u1" in tagger._symbols and "u3" in tagger._symbols


def setup_module(module):
    pytest.importorskip("numpy")
