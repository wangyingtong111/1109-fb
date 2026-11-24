from collections import Counter

from rapidfuzz import fuzz, process


def test_multiset_coverage_counts_repeated_words():
    score = fuzz.multiset_coverage("red red tea", "red tea")
    assert score == 80.0
    assert score.left_uncovered == {"red": 1}
    assert score.right_uncovered == {}

    full = fuzz.multiset_coverage("red red tea", "tea red red")
    assert full == 100.0
    assert full.left_uncovered == {}
    assert full.right_uncovered == {}


def test_default_set_score_is_unchanged():
    assert fuzz.token_set_ratio("red red tea", "red tea") == 100.0
    assert fuzz.token_set_ratio("red red tea", "red tea", multiset=True) == 80.0


def test_extract_coverage_uses_one_first_pass_snapshot():
    calls = Counter()

    def processor(value):
        calls[value] += 1
        if value == "alias":
            return "red red tea" if calls[value] == 1 else "red tea"
        if value == "late":
            return "tea red red"
        return value

    choices = {"plain": "red tea", "earliest": "alias", "latest": "late"}
    hits = process.extract_coverage(
        "red red tea",
        choices,
        processor=processor,
        score_cutoff=80,
        limit=2,
    )

    assert [hit.key for hit in hits] == ["earliest", "latest"]
    assert [(hit.score, hit.key) for hit in hits] == [(100.0, "earliest"), (100.0, "latest")]
    assert hits.explanations == {"earliest": {}, "latest": {}}
    assert calls == Counter({"red red tea": 1, "red tea": 1, "alias": 1, "late": 1})


def test_extract_coverage_accepts_one_shot_iterator_and_keeps_input_tie_order():
    calls = []

    def processor(value):
        calls.append(value)
        return {"alias": "red red tea", "late": "tea red red"}.get(value, value)

    choices = iter(["red tea", "alias", "late"])
    hits = process.extract_coverage(
        "red red tea",
        choices,
        processor=processor,
        score_cutoff=80,
        k=2,
    )

    assert [hit.key for hit in hits] == [1, 2]
    assert calls == ["red red tea", "red tea", "alias", "late"]
    assert list(hits.explanations) == [1, 2]
