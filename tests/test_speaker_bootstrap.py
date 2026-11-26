from random import Random
from unittest.mock import patch

import jiwer

from jiwer.bootstrap import SpeakerBootstrapWER, speaker_wer_confidence_interval

NORMAL = [
    ("A", "red", "blue"),
    ("B", "a b c", "a b c"),
]

CONFLICT = [
    ("A", "red", "blue"),
    ("A", "", "oops"),
    ("B", "a b c", "a b c"),
]


def test_pooled_speaker_draw_is_not_macro_average():
    estimator = SpeakerBootstrapWER(NORMAL)

    assert estimator.alignment_count == 2
    assert estimator.wer == 0.25
    assert estimator.wer_for_draw(["A", "B"]) == 0.25

    # A repeated cluster contributes both its numerator and denominator twice.
    assert estimator.wer_for_draw(["A", "A"]) == 1.0
    assert estimator.wer_for_draw(["B", "B"]) == 0.0


def test_added_silence_pair_and_exact_replays():
    estimator = SpeakerBootstrapWER(CONFLICT)

    assert estimator.alignment_count == 3
    assert estimator.total_errors == 2
    assert estimator.reference_words == 4
    assert estimator.wer == 0.5

    draws = [("A", "B"), ("A", "A"), ("B", "B")]
    assert [estimator.wer_for_draw(draw) for draw in draws] == [0.5, 2.0, 0.0]

    result = estimator.confidence_interval(draws=draws)
    assert result.bootstrap_wers == (0.5, 2.0, 0.0)
    assert result.n_bootstrap == 3


def test_index_draws_and_parallel_arguments():
    estimator = SpeakerBootstrapWER(
        speakers=["A", "A", "B"],
        references=["red", "", "a b c"],
        hypotheses=["blue", "oops", "a b c"],
    )

    assert estimator.wer_for_draw([[0, 1], [0, 0], [1, 1]][0]) == 0.5
    assert [
        estimator.wer_for_draw([0, 1]),
        estimator.wer_for_draw([0, 0]),
        estimator.wer_for_draw([1, 1]),
    ] == [0.5, 2.0, 0.0]

    result = jiwer.wer_confidence_interval(
        ["red", "", "a b c"],
        ["blue", "oops", "a b c"],
        ["A", "A", "B"],
        draws=[["A", "B"], ["A", "A"], ["B", "B"]],
    )
    assert result.bootstrap_wers == (0.5, 2.0, 0.0)


def test_grouped_dictionary_input():
    estimator = SpeakerBootstrapWER(
        {
            "A": [("red", "blue"), ("", "oops")],
            "B": [("a b c", "a b c")],
        }
    )

    assert estimator.speaker_ids == ("A", "B")
    assert estimator.alignment_count == 3
    assert estimator.wer_for_draw(["A", "B"]) == 0.5
    assert estimator.wer_for_draw(["A", "A"]) == 2.0


def test_alignment_runs_once_per_transcript():
    with patch(
        "jiwer.process.process_words", wraps=jiwer.process_words
    ) as process_words:
        estimator = SpeakerBootstrapWER(CONFLICT)
        assert process_words.call_count == 3

        first = estimator.confidence_interval(n_bootstrap=50, seed=1)
        second = estimator.confidence_interval(n_bootstrap=500, seed=2)

        assert process_words.call_count == 3
        assert estimator.alignment_count == 3
        assert len(first.bootstrap_wers) == 50
        assert len(second.bootstrap_wers) == 500


def test_seed_is_reusable_and_interval_is_returned():
    first = speaker_wer_confidence_interval(
        CONFLICT, n_bootstrap=200, confidence_level=0.95, seed=20261127
    )
    second = speaker_wer_confidence_interval(
        CONFLICT, n_bootstrap=200, confidence_level=0.95, seed=20261127
    )

    assert first.bootstrap_wers == second.bootstrap_wers
    assert first.interval == second.interval
    assert first.lower <= first.wer <= first.upper
    assert 0.0 <= first.lower <= first.upper

    # A different seed should normally produce a different finite sample path.
    third = speaker_wer_confidence_interval(CONFLICT, n_bootstrap=200, seed=1)
    assert third.bootstrap_wers != first.bootstrap_wers


def test_cached_draws_match_fully_expanded_wer():
    random = Random(20261127)
    texts = [
        "red blue",
        "green fruit",
        "a b c",
        "one two three four",
        "",
        "go now",
    ]
    records = []
    for speaker in "ABCDE":
        for _ in range(random.randint(1, 4)):
            records.append(
                (
                    speaker,
                    random.choice(texts),
                    random.choice(texts + ["extra spoken words"]),
                )
            )

    estimator = SpeakerBootstrapWER(records)

    for _ in range(50):
        draw = [
            random.choice(estimator.speaker_ids) for _ in range(estimator.speaker_count)
        ]
        expanded = [
            (reference, hypothesis)
            for speaker in draw
            for record_speaker, reference, hypothesis in records
            if record_speaker == speaker
        ]
        references, hypotheses = zip(*expanded)

        assert estimator.wer_for_draw(draw) == jiwer.wer(
            list(references), list(hypotheses)
        )


def test_plain_wer_is_unchanged():
    assert jiwer.wer("red", "blue") == 1.0
    assert jiwer.wer("", "oops") == 1.0
    assert jiwer.wer(["red", "a b c"], ["blue", "a b c"]) == 0.25
