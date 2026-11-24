# Natural Language Toolkit: Language Model Unit Tests
import pickle

from nltk.lm import MLE, IncrementalVocabulary, Vocabulary


def test_incremental_fit_recovers_promoted_words_old_ngrams():
    model = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    first_batch = [[("oak", "red"), ("elm", "red")]]
    first_vocab = ["oak", "elm", "red", "red", "blue", "blue"]
    second_batch = [[("oak", "blue")]]
    second_vocab = ["oak", "blue"]

    model.fit(first_batch, first_vocab, incremental=True)
    assert "oak" not in model.vocab
    assert "elm" not in model.vocab

    model.fit(second_batch, second_vocab, incremental=True)

    assert "oak" in model.vocab
    assert "elm" not in model.vocab
    assert model.counts[["oak"]]["red"] == 1
    assert model.counts[["oak"]]["blue"] == 1
    assert model.counts[["<UNK>"]]["red"] == 1
    assert model.counts[["<UNK>"]]["blue"] == 0


def test_incremental_batches_equal_one_merged_incremental_fit():
    first_batch = [[("oak", "red"), ("elm", "red")]]
    second_batch = [[("oak", "blue")]]
    first_vocab = ["oak", "elm", "red", "red", "blue", "blue"]
    second_vocab = ["oak", "blue"]

    incremental = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    incremental.fit(first_batch, first_vocab, incremental=True)
    incremental.fit(second_batch, second_vocab, incremental=True)

    merged = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    merged.fit_incremental(first_batch + second_batch, first_vocab + second_vocab)

    assert dict(incremental.counts[["oak"]]) == dict(merged.counts[["oak"]])
    assert dict(incremental.counts[["<UNK>"]]) == dict(merged.counts[["<UNK>"]])
    assert incremental.vocab.counts == merged.vocab.counts


def test_merged_default_fit_has_same_counts():
    first_batch = [[("oak", "red"), ("elm", "red")]]
    second_batch = [[("oak", "blue")]]
    first_vocab = ["oak", "elm", "red", "red", "blue", "blue"]
    second_vocab = ["oak", "blue"]

    incremental = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    incremental.fit_incremental(first_batch, first_vocab)
    incremental.fit_incremental(second_batch, second_vocab)

    frozen = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    frozen.fit(first_batch + second_batch, first_vocab + second_vocab)

    assert dict(incremental.counts[["oak"]]) == dict(frozen.counts[["oak"]])
    assert incremental.vocab.counts == frozen.vocab.counts


def test_default_fit_remains_frozen():
    model = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    model.fit(
        [[("oak", "red"), ("elm", "red")]],
        ["oak", "elm", "red", "red", "blue", "blue"],
    )
    model.fit([[("oak", "blue")]], ["oak", "blue"])

    assert "oak" not in model.vocab
    assert model.counts[["<UNK>"]]["red"] == 2
    assert model.counts[["<UNK>"]]["blue"] == 1
    assert model.counts[["oak"]]["red"] == 0


def test_positions_allow_recovering_each_occurrence():
    vocab = IncrementalVocabulary(unk_cutoff=2)
    model = MLE(2, vocabulary=vocab)

    model.fit_incremental([[("oak", "red")]], ["oak", "red"])
    model.fit_incremental(
        [[("oak", "blue")], [("elm", "green")]],
        ["oak", "blue", "elm", "green"],
    )

    assert model.positions_for("oak") == [(0, 0, 0, 0), (1, 0, 0, 0)]
    first_location = model.positions_for("oak")[0]
    assert model.raw_ngram_at(*first_location[:3]) == ("oak", "red")
    assert model.raw_bigram_records("oak") == [
        ["oak", "blue", 1],
        ["oak", "red", 1],
    ]
    assert "elm" not in model.vocab
    assert model.vocab.recover_positions("elm") == [4]


def test_abstract_batch_records_match_expected_oak_bigrams():
    model = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    model.fit_incremental([["oak", "red"]])
    model.fit_incremental([["oak", "blue"], ["elm", "green"]])

    assert model.restored_bigram_records("oak") == [
        ["oak", "blue", 1],
        ["oak", "red", 1],
    ]
    assert "elm" not in model.vocab


def test_pickle_preserves_incremental_history():
    model = MLE(2, vocabulary=Vocabulary(unk_cutoff=2))
    model.fit_incremental(
        [[("oak", "red"), ("elm", "red")]],
        ["oak", "elm", "red", "red", "blue", "blue"],
    )
    model.fit_incremental([[("oak", "blue")]], ["oak", "blue"])

    restored = pickle.loads(pickle.dumps(model))
    assert restored.counts[["oak"]]["red"] == 1
    assert restored.counts[["oak"]]["blue"] == 1
    assert restored.positions_for("oak") == model.positions_for("oak")

    restored.fit_incremental(
        [[("oak", "green")]], ["oak", "green", "green"]
    )
    assert restored.counts[["oak"]]["green"] == 1

