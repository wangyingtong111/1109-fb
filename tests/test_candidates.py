import math

import pytest

import jiwer


def test_weighted_wer_and_top1():
    out = jiwer.process_candidates(
        "red green",
        ["red green", "red", "red blue green"],
        [0.25, 0.5, 0.25],
    )

    assert out.wer == pytest.approx(0.375)
    assert out.expected_wer == pytest.approx(0.375)
    assert out.top1_transcript == "red"
    assert len(out.candidates) == 3


def test_per_candidate_probability_alignment_and_contribution():
    out = jiwer.process_candidates(
        "red green",
        ["red green", "red", "red blue green"],
        [0.25, 0.5, 0.25],
    )

    contributions = out.candidate_contributions()
    assert [c[0] for c in contributions] == [0, 1, 2]

    assert out.candidates[0].probability == pytest.approx(0.25)
    assert out.candidates[0].wer == pytest.approx(0.0)
    assert out.candidates[0].error_contribution == pytest.approx(0.0)
    assert [c.type for c in out.candidates[0].alignments] == ["equal"]

    assert out.candidates[1].probability == pytest.approx(0.5)
    assert out.candidates[1].wer == pytest.approx(0.5)
    assert out.candidates[1].error_contribution == pytest.approx(0.25)
    assert [c.type for c in out.candidates[1].alignments] == ["equal", "delete"]

    assert out.candidates[2].probability == pytest.approx(0.25)
    assert out.candidates[2].wer == pytest.approx(0.5)
    assert out.candidates[2].error_contribution == pytest.approx(0.125)

    assert sum(c[3] for c in contributions) == pytest.approx(0.375)

    # the raw per-candidate alignments are usable with the existing report entry
    report = jiwer.visualize_alignment(out.candidates[1].output)
    assert "REF: red green" in report
    assert "HYP: red *****" in report


def test_equal_transcripts_merge_probability_for_top1():
    out = jiwer.process_candidates(
        "red green",
        ["red green", "red  green", "red"],
        [0.3, 0.3, 0.4],
    )

    assert out.top1_transcript == "red green"
    assert out.top1.probability == pytest.approx(0.6)
    assert out.top1.candidate_indices == [0, 1]
    assert len(out.transcript_groups) == 2
    assert out.wer == pytest.approx(0.2)


def test_shifted_log_weights_stay_stable():
    probs = [0.3, 0.3, 0.4]
    log_weights = [math.log(p) - 1000 for p in probs]

    out = jiwer.process_candidates(
        "red green",
        ["red green", "red  green", "red"],
        log_weights,
        log_weights=True,
    )

    assert out.log_weights is True
    assert out.top1_transcript == "red green"
    assert out.wer == pytest.approx(0.2)

    # each path keeps its own probability and contribution
    assert [c.probability for c in out.candidates] == pytest.approx(probs)
    assert out.candidates[2].error_contribution == pytest.approx(0.2)
    assert [c[0] for c in out.candidate_contributions()] == [0, 1, 2]


def test_log_weights_without_shift_match_probabilities():
    probs = [0.25, 0.5, 0.25]
    shifted = jiwer.process_candidates(
        "red green",
        ["red green", "red", "red blue green"],
        [math.log(p) for p in probs],
        log_weights=True,
    )
    plain = jiwer.process_candidates(
        "red green",
        ["red green", "red", "red blue green"],
        probs,
    )

    assert shifted.wer == pytest.approx(plain.wer)
    assert shifted.top1_transcript == plain.top1_transcript


def test_huge_common_offset_does_not_underflow():
    probs = [0.25, 0.5, 0.25]
    out = jiwer.process_candidates(
        "red green",
        ["red green", "red", "red blue green"],
        [math.log(p) - 10_000 for p in probs],
        log_weights=True,
    )

    assert out.wer == pytest.approx(0.375)
    assert [c.probability for c in out.candidates] == pytest.approx(probs)


def test_top1_tie_uses_first_candidate():
    out = jiwer.process_candidates("red green", ["red green", "red"], [0.5, 0.5])

    assert out.wer == pytest.approx(0.25)
    assert out.top1_transcript == "red green"
    assert out.top1_candidate.index == 0


def test_top1_tie_between_merged_groups_uses_first_appearance():
    out = jiwer.process_candidates(
        "red green",
        ["red", "red green", "red", "red green"],
        [0.25, 0.25, 0.25, 0.25],
    )

    assert out.transcript_groups[0].probability == pytest.approx(0.5)
    assert out.transcript_groups[1].probability == pytest.approx(0.5)
    assert out.top1_transcript == "red"


def test_expected_wer_convenience_function():
    value = jiwer.expected_wer(
        "red green",
        ["red green", "red", "red blue green"],
        [0.25, 0.5, 0.25],
    )

    assert value == pytest.approx(0.375)
    assert value == pytest.approx(
        jiwer.process_candidates(
            "red green",
            ["red green", "red", "red blue green"],
            [0.25, 0.5, 0.25],
        ).wer
    )


def test_unweighted_candidates_are_averaged():
    out = jiwer.process_candidates("red green", ["red green", "red"])

    assert out.wer == pytest.approx(0.25)
    assert [c.probability for c in out.candidates] == pytest.approx([0.5, 0.5])


def test_single_alignment_entry_points_still_work():
    out = jiwer.process_words("green fruit", "green red fruit")
    assert out.insertions == 1 and out.hits == 2 and out.wer == 0.5
    assert jiwer.cer("ab", "ax") == 0.5
    assert jiwer.wer("red green", "red green") == 0.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        jiwer.process_candidates("red green", [])

    with pytest.raises(ValueError):
        jiwer.process_candidates("red green", ["red"], [0.5, 0.5])

    with pytest.raises(ValueError):
        jiwer.process_candidates("red green", ["red"], [0.0])

    with pytest.raises(ValueError):
        jiwer.process_candidates("red green", ["red", "red green"], [0.5, -0.5])

    with pytest.raises(ValueError):
        jiwer.process_candidates(
            "red green",
            ["red", "red green"],
            [float("-inf"), float("-inf")],
            log_weights=True,
        )

    with pytest.raises(ValueError):
        jiwer.process_candidates(["a", "b"], ["red"])
