import pytest

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

## Test bigram counters with discontinuous bigrams and repeated words

_EPSILON = 1e-8
SENT = "this this is is a a test test".split()


def close_enough(x, y):
    """Verify that two sequences of n-gram association values are within
    _EPSILON of each other.
    """

    return all(abs(x1[1] - y1[1]) <= _EPSILON for x1, y1 in zip(x, y))


def test_bigram2():
    b = BigramCollocationFinder.from_words(SENT)

    assert sorted(b.ngram_fd.items()) == [
        (("a", "a"), 1),
        (("a", "test"), 1),
        (("is", "a"), 1),
        (("is", "is"), 1),
        (("test", "test"), 1),
        (("this", "is"), 1),
        (("this", "this"), 1),
    ]
    assert sorted(b.word_fd.items()) == [("a", 2), ("is", 2), ("test", 2), ("this", 2)]

    assert len(SENT) == sum(b.word_fd.values()) == sum(b.ngram_fd.values()) + 1
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        [
            (("a", "a"), 1.0),
            (("a", "test"), 1.0),
            (("is", "a"), 1.0),
            (("is", "is"), 1.0),
            (("test", "test"), 1.0),
            (("this", "is"), 1.0),
            (("this", "this"), 1.0),
        ],
    )


def test_bigram3():
    b = BigramCollocationFinder.from_words(SENT, window_size=3)
    assert sorted(b.ngram_fd.items()) == sorted(
        [
            (("a", "test"), 3),
            (("is", "a"), 3),
            (("this", "is"), 3),
            (("a", "a"), 1),
            (("is", "is"), 1),
            (("test", "test"), 1),
            (("this", "this"), 1),
        ]
    )

    assert sorted(b.word_fd.items()) == sorted(
        [("a", 2), ("is", 2), ("test", 2), ("this", 2)]
    )

    assert (
        len(SENT) == sum(b.word_fd.values()) == (sum(b.ngram_fd.values()) + 2 + 1) / 2.0
    )
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        sorted(
            [
                (("a", "test"), 1.584962500721156),
                (("is", "a"), 1.584962500721156),
                (("this", "is"), 1.584962500721156),
                (("a", "a"), 0.0),
                (("is", "is"), 0.0),
                (("test", "test"), 0.0),
                (("this", "this"), 0.0),
            ]
        ),
    )


def test_bigram5():
    b = BigramCollocationFinder.from_words(SENT, window_size=5)
    assert sorted(b.ngram_fd.items()) == sorted(
        [
            (("a", "test"), 4),
            (("is", "a"), 4),
            (("this", "is"), 4),
            (("is", "test"), 3),
            (("this", "a"), 3),
            (("a", "a"), 1),
            (("is", "is"), 1),
            (("test", "test"), 1),
            (("this", "this"), 1),
        ]
    )
    assert sorted(b.word_fd.items()) == sorted(
        [("a", 2), ("is", 2), ("test", 2), ("this", 2)]
    )
    n_word_fd = sum(b.word_fd.values())
    n_ngram_fd = (sum(b.ngram_fd.values()) + 4 + 3 + 2 + 1) / 4.0
    assert len(SENT) == n_word_fd == n_ngram_fd
    assert close_enough(
        sorted(b.score_ngrams(BigramAssocMeasures.pmi)),
        sorted(
            [
                (("a", "test"), 1.0),
                (("is", "a"), 1.0),
                (("this", "is"), 1.0),
                (("is", "test"), 0.5849625007211562),
                (("this", "a"), 0.5849625007211562),
                (("a", "a"), -1.0),
                (("is", "is"), -1.0),
                (("test", "test"), -1.0),
                (("this", "this"), -1.0),
            ]
        ),
    )


## Windowed, document-aware bigram counting

# First document [a, b, c] (window size 3) split before b into two chunks;
# the second document is the short title [a, b].
WINDOW_SIZE = 3
DOC1_CHUNKS = [["a"], ["b", "c"]]
DOC2_CHUNKS = [["a", "b"]]
WINDOWED_DOCUMENTS = [DOC1_CHUNKS, DOC2_CHUNKS]

# Ordered pairs that actually exist within each document's windows:
#   doc1: a->b, a->c, b->c          (3 opportunities)
#   doc2: a->b                        (1 opportunity)
EXPECTED_BIGRAM_FD = {("a", "b"): 2, ("a", "c"): 1, ("b", "c"): 1}
EXPECTED_WORD_FD = {"a": 3, "b": 1}
EXPECTED_OPPORTUNITIES = 4


def _windowed_finder():
    return BigramCollocationFinder.from_windowed_documents(
        WINDOWED_DOCUMENTS, window_size=WINDOW_SIZE
    )


def test_windowed_documents_counts_only_existing_pairs():
    finder = _windowed_finder()
    assert dict(finder.ngram_fd) == EXPECTED_BIGRAM_FD
    assert dict(finder.word_fd) == EXPECTED_WORD_FD
    # Short titles contribute only the ordered position pairs that exist.
    assert finder.n_opportunities == EXPECTED_OPPORTUNITIES
    assert sum(finder.word_fd.values()) == EXPECTED_OPPORTUNITIES
    # Windows never cross the boundary between separate documents:
    # c (end of doc 1) is never paired with a (start of doc 2).
    assert ("c", "a") not in finder.ngram_fd
    assert finder.window_size == WINDOW_SIZE


def test_windowed_chunks_behave_like_unsplit_document():
    split_finder = _windowed_finder()
    whole_finder = BigramCollocationFinder.from_windowed_documents(
        [[["a", "b", "c"]], [["a", "b"]]], window_size=WINDOW_SIZE
    )
    # The window reaching from one chunk into the next is preserved.
    assert dict(whole_finder.ngram_fd) == dict(split_finder.ngram_fd)
    assert dict(whole_finder.word_fd) == dict(split_finder.word_fd)
    assert whole_finder.n_opportunities == split_finder.n_opportunities
    assert whole_finder.n_opportunities == EXPECTED_OPPORTUNITIES


def test_windowed_documents_do_not_cross_boundaries():
    # A flattened word stream of the same tokens would create cross-document
    # pairs such as c->a; the document-aware entry must not.
    finder = _windowed_finder()
    flattened = BigramCollocationFinder.from_words(
        ["a", "b", "c", "a", "b"], window_size=WINDOW_SIZE
    )
    assert flattened.ngram_fd[("c", "a")] == 1
    assert ("c", "a") not in finder.ngram_fd
    assert flattened.ngram_fd[("b", "a")] == 1
    assert ("b", "a") not in finder.ngram_fd


def test_windowed_batches_merge_like_single_pass():
    first_batch = BigramCollocationFinder.from_windowed_documents(
        [DOC1_CHUNKS], window_size=WINDOW_SIZE
    )
    second_batch = BigramCollocationFinder.from_windowed_documents(
        [DOC2_CHUNKS], window_size=WINDOW_SIZE
    )
    merged = BigramCollocationFinder.merge(first_batch, second_batch)

    single_pass = _windowed_finder()
    assert dict(merged.ngram_fd) == dict(single_pass.ngram_fd)
    assert dict(merged.word_fd) == dict(single_pass.word_fd)
    assert merged.n_opportunities == single_pass.n_opportunities
    assert merged.n_opportunities == EXPECTED_OPPORTUNITIES
    assert merged.window_size == WINDOW_SIZE


def test_windowed_filter_keeps_common_denominator():
    finder = _windowed_finder()
    scores_before = dict(finder.score_ngrams(BigramAssocMeasures.raw_freq))
    denominator = finder.n_opportunities

    # Remove one output candidate.
    finder.apply_ngram_filter(lambda w1, w2: (w1, w2) == ("a", "b"))
    assert ("a", "b") not in finder.ngram_fd

    # The common denominator is unaffected by filtering...
    assert finder.n_opportunities == denominator == EXPECTED_OPPORTUNITIES
    # ...so the surviving candidates keep exactly their old scores.
    scores_after = dict(finder.score_ngrams(BigramAssocMeasures.raw_freq))
    assert set(scores_after) == {("a", "c"), ("b", "c")}
    for ngram, score in scores_after.items():
        assert abs(score - scores_before[ngram]) <= _EPSILON


def test_windowed_opportunity_denominator_used_for_scoring():
    finder = _windowed_finder()
    # raw_freq scales the co-occurrence count by 1/(window_size - 1) and
    # divides by the total number of bigram opportunities (4 here).
    expected_ab = (2 / (WINDOW_SIZE - 1.0)) / EXPECTED_OPPORTUNITIES
    assert close_enough(
        finder.score_ngrams(BigramAssocMeasures.raw_freq),
        [
            (("a", "b"), expected_ab),
            (("a", "c"), expected_ab / 2),
            (("b", "c"), expected_ab / 2),
        ],
    )


def test_windowed_documents_reject_small_window():
    with pytest.raises(ValueError):
        BigramCollocationFinder.from_windowed_documents(
            WINDOWED_DOCUMENTS, window_size=1
        )


def test_merge_requires_matching_window_size():
    first = BigramCollocationFinder.from_windowed_documents(
        [DOC1_CHUNKS], window_size=WINDOW_SIZE
    )
    second = BigramCollocationFinder.from_windowed_documents(
        [DOC2_CHUNKS], window_size=2
    )
    with pytest.raises(ValueError):
        BigramCollocationFinder.merge(first, second)
    with pytest.raises(ValueError):
        BigramCollocationFinder.merge()


def test_from_words_keeps_token_denominator():
    # The original word-stream entry and its scoring remain unchanged.
    for window_size in (2, 3, 5):
        finder = BigramCollocationFinder.from_words(SENT, window_size=window_size)
        assert finder.n_opportunities == finder.N == len(SENT)


def test_windowed_empty_and_single_token_documents():
    # Documents with fewer than two tokens contribute no ordered pairs.
    finder = BigramCollocationFinder.from_windowed_documents(
        [[[]], [[], ["x"]], [["y"], []]], window_size=3
    )
    assert finder.n_opportunities == 0
    assert dict(finder.ngram_fd) == {}
    assert dict(finder.word_fd) == {}

    empty = BigramCollocationFinder.from_windowed_documents([], window_size=3)
    assert empty.n_opportunities == 0
    assert dict(empty.ngram_fd) == {}


def test_windowed_streaming_chunks_match_single_chunk():
    tokens = list("abcde")
    whole = BigramCollocationFinder.from_windowed_documents(
        [[tokens]], window_size=4
    )
    # Handing the document over one token per chunk keeps the windows that
    # reach across chunk boundaries.
    streamed = BigramCollocationFinder.from_windowed_documents(
        [[[token] for token in tokens]], window_size=4
    )
    assert dict(streamed.ngram_fd) == dict(whole.ngram_fd)
    assert dict(streamed.word_fd) == dict(whole.word_fd)
    assert streamed.n_opportunities == whole.n_opportunities
    # 3 + 3 + 3 full-window pairs plus the 2- and 1-pair short tails.
    assert whole.n_opportunities == 9
