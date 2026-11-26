"""Tests for the optional exact tie-breaking rule in
``nltk.translate.gleu_score.corpus_gleu``.

When several references give the same GLEU ratio, the original implementation
keeps whichever reference is encountered first because ``max()`` returns the
first maximal element, so the corpus score can change merely by reordering the
references. With ``precise_tie_break=True`` the ratios are compared exactly on
the integer counts, exact ties are resolved in favour of the larger
denominator, and the raw per-sentence ``(n_match, n_all)`` counts are reported
alongside the aggregate corpus score.
"""

import pytest

from nltk.translate.gleu_score import corpus_gleu, sentence_gleu


# ---------------------------------------------------------------------------
# Fixtures: unigram-only data in which the first sentence has two tied
# references contributing 1/2 and 2/4, and the second contributes 2/2.
# ---------------------------------------------------------------------------

HYP1 = ["a", "b"]
REF1A = ["a", "x"]  # 1 match out of 2 -> 1/2
REF1B = ["a", "b", "x", "y"]  # 2 matches out of 4 -> 2/4
HYP2 = ["a", "b"]
REF2 = ["a", "b"]  # 2 matches out of 2 -> 2/2


def _refs(flipped=False):
    first = [REF1B, REF1A] if flipped else [REF1A, REF1B]
    return [first, [REF2]]


def _hyps():
    return [HYP1, HYP2]


# ---------------------------------------------------------------------------
# Default behaviour: unchanged interface (a bare float) and original
# reference-order dependence on ties.
# ---------------------------------------------------------------------------

def test_default_returns_a_bare_float():
    result = corpus_gleu(_refs(), _hyps(), max_len=1)
    assert isinstance(result, float)


def test_default_keeps_original_traversal_order_dependence():
    # First sentence keeps the first tied reference: 1/2 + 2/2 = 3/4.
    assert corpus_gleu(_refs(), _hyps(), max_len=1) == pytest.approx(0.75)
    # Flipped order keeps the other tied reference: 2/4 + 2/2 = 4/6.
    assert corpus_gleu(_refs(flipped=True), _hyps(), max_len=1) == pytest.approx(
        4 / 6
    )


def test_default_explicit_false_matches_original_behavior():
    assert corpus_gleu(
        _refs(), _hyps(), max_len=1, precise_tie_break=False
    ) == pytest.approx(0.75)


def test_sentence_gleu_interface_unchanged():
    # sentence_gleu must keep returning a bare float.
    result = sentence_gleu([REF1A, REF1B], HYP1, max_len=1)
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Precise tie-breaking: exact ratio comparison, larger denominator on ties,
# order independence and reported per-sentence counts.
# ---------------------------------------------------------------------------

def test_precise_tie_break_selects_larger_denominator():
    score, counts = corpus_gleu(
        _refs(), _hyps(), max_len=1, precise_tie_break=True
    )
    assert counts == [(2, 4), (2, 2)]
    # Raw per-sentence counts are preserved in the aggregate: 4/6.
    assert score == pytest.approx(4 / 6)


def test_precise_tie_break_is_order_independent():
    score, counts = corpus_gleu(
        _refs(), _hyps(), max_len=1, precise_tie_break=True
    )
    flipped_score, flipped_counts = corpus_gleu(
        _refs(flipped=True), _hyps(), max_len=1, precise_tie_break=True
    )
    assert flipped_score == score
    assert flipped_counts == counts == [(2, 4), (2, 2)]


def test_precise_tie_break_alias():
    score, counts = corpus_gleu(
        _refs(), _hyps(), max_len=1, exact_tie_break=True
    )
    assert counts == [(2, 4), (2, 2)]
    assert score == pytest.approx(4 / 6)


def test_precise_tie_break_accepts_positional_flag():
    result = corpus_gleu(_refs(), _hyps(), 1, 1, True)
    assert isinstance(result, tuple)
    score, counts = result
    assert counts == [(2, 4), (2, 2)]
    assert score == pytest.approx(4 / 6)


def test_larger_ratio_wins_even_with_smaller_denominator():
    # A clearly larger ratio (1/1 = 1.0) beats the tied 1/2 vs 2/4 pair even
    # though its denominator is the smallest.
    ref_perfect = ["a", "b"]  # 2 matches out of 2 -> 2/2 = 1.0
    refs = [[REF1A, REF1B, ref_perfect], [REF2]]
    score, counts = corpus_gleu(
        refs, _hyps(), max_len=1, precise_tie_break=True
    )
    assert counts == [(2, 2), (2, 2)]
    assert score == pytest.approx(1.0)


def test_tied_ratio_same_denominator_counts_are_identical():
    # Two distinct references giving the very same (n_match, n_all) counts.
    ref_same_a = ["a", "x"]
    ref_same_b = ["a", "z"]  # also 1 match out of 2
    refs = [[ref_same_a, ref_same_b], [REF2]]
    score_a, counts_a = corpus_gleu(
        refs, _hyps(), max_len=1, precise_tie_break=True
    )
    score_b, counts_b = corpus_gleu(
        [list(reversed(refs[0])), refs[1]],
        _hyps(),
        max_len=1,
        precise_tie_break=True,
    )
    assert counts_a == counts_b == [(1, 2), (2, 2)]
    assert score_a == score_b == pytest.approx(3 / 4)


def test_corpus_aggregate_uses_raw_counts_not_average():
    # 4/6 from the two-sentence fixture must not equal the mean of the
    # per-sentence ratios ((2/4 + 2/2) / 2 = 0.75).
    score, counts = corpus_gleu(
        _refs(), _hyps(), max_len=1, precise_tie_break=True
    )
    total_match = sum(m for m, _ in counts)
    total_all = sum(d for _, d in counts)
    assert (total_match, total_all) == (4, 6)
    assert score == pytest.approx(total_match / total_all)
    assert score != pytest.approx(
        sum(m / d for m, d in counts) / len(counts)
    )


def test_zero_denominator_corpus_returns_zero_in_both_modes():
    # Empty hypothesis and empty references: every denominator is zero.
    refs = [[[]]]
    hyps = [[]]
    assert corpus_gleu(refs, hyps, max_len=1) == 0.0
    score, counts = corpus_gleu(
        refs, hyps, max_len=1, precise_tie_break=True
    )
    assert score == 0.0
    assert counts == [(0, 0)]


def test_sentence_without_usable_reference_reports_zero_counts():
    # An empty hypothesis paired with an empty reference has a zero
    # denominator and therefore no usable (n_match, n_all) pair; it must
    # still get a reported entry while a normal sentence contributes normally.
    refs = [[[]], [REF2]]
    hyps = [[], HYP2]
    score, counts = corpus_gleu(
        refs, hyps, max_len=1, precise_tie_break=True
    )
    assert counts == [(0, 0), (2, 2)]
    assert score == pytest.approx(1.0)


def test_empty_reference_with_nonempty_hypothesis_keeps_raw_count():
    # n_all = max(n_all_output, n_all_target): an empty reference still
    # leaves the hypothesis length as denominator, giving a raw (0, tpfp)
    # count in both modes (this mirrors the original implementation).
    refs = [[[]], [REF2]]
    hyps = [HYP1, HYP2]
    default_score = corpus_gleu(refs, hyps, max_len=1)
    precise_score, counts = corpus_gleu(
        refs, hyps, max_len=1, precise_tie_break=True
    )
    assert counts == [(0, 2), (2, 2)]
    assert precise_score == default_score == pytest.approx(0.5)


def test_default_mode_does_not_collect_sentence_counts():
    # Sanity check that the new bookkeeping path is not taken by default.
    result = corpus_gleu(_refs(), _hyps(), max_len=1)
    assert not isinstance(result, tuple)


def test_unexpected_keyword_argument_raises_typeerror():
    with pytest.raises(TypeError):
        corpus_gleu(_refs(), _hyps(), max_len=1, bogus_flag=True)
