from __future__ import annotations

import random

from rapidfuzz import fuzz, process
from rapidfuzz.process_py import _BoundedEntityRanking, extract


def _cost_scorer(costs, *, higher_better=False):
    worst, optimal = (0, 100) if higher_better else (10**18, 0)

    def scorer(query, choice, score_cutoff=None, **kwargs):
        return costs[choice]

    scorer._RF_ScorerPy = {
        "get_scorer_flags": lambda **kwargs: {
            "worst_score": worst,
            "optimal_score": optimal,
        }
    }
    return scorer


def test_entity_top_k_promotes_second_entity():
    choices = {
        "a1": "oak green",
        "a2": "oak-green",
        "b1": "oak gray",
    }
    entities = {"a1": "A", "a2": "A", "b1": "B"}

    result = extract(
        "oak green",
        choices,
        scorer=fuzz.ratio,
        entities=entities,
        k=2,
    )

    assert [item[2] for item in result] == ["a1", "b1"]
    assert [entities[item[2]] for item in result] == ["A", "B"]
    assert result[0][1] == 100.0

    public_result = process.extract(
        "oak green",
        choices,
        scorer=fuzz.ratio,
        entity_key=entities,
        limit=2,
    )
    assert public_result == result


def test_evicted_entity_can_return_and_ties_are_stable():
    costs = {"a1": 4, "b1": 3, "c1": 2, "a2": 1, "a3": 1, "d1": 2}
    entities = {
        "a1": "A",
        "a2": "A",
        "a3": "A",
        "b1": "B",
        "c1": "C",
        "d1": "D",
    }
    choices = {key: key for key in costs}

    result = extract(
        "query",
        choices,
        scorer=_cost_scorer(costs),
        entity_key=entities,
        limit=2,
    )

    assert [item[2] for item in result] == ["a2", "c1"]
    assert [entities[item[2]] for item in result] == ["A", "C"]


def test_ranking_state_never_exceeds_k():
    hits = [
        ("A", 4, "a1"),
        ("B", 3, "b1"),
        ("C", 2, "c1"),
        ("A", 1, "a2"),
        ("A", 1, "a3"),
        ("D", 2, "d1"),
    ]
    ranking = _BoundedEntityRanking(2, False)
    peak = 0

    for order, (entity, score, key) in enumerate(hits):
        ranking.add(entity, order, (key, score, key))
        peak = max(peak, len(ranking.groups))

    winners = [item[2] for item in ranking.results()]
    assert peak == 2
    assert winners == ["a2", "c1"]


def test_missing_entity_labels_are_independent():
    costs = {"a1": 1, "a2": 1, "b1": 2}
    entities = {"a1": "A", "a2": "A"}

    grouped = extract(
        "query",
        {key: key for key in costs},
        scorer=_cost_scorer(costs),
        entities=entities,
        k=3,
    )
    assert [item[2] for item in grouped] == ["a1", "b1"]

    unlabelled = extract(
        "query",
        {key: key for key in costs},
        scorer=_cost_scorer(costs),
        entities={},
        k=2,
    )
    assert [item[2] for item in unlabelled] == ["a1", "a2"]


def test_one_shot_iterator_and_sequence_labels():
    costs = [4, 3, 2, 1, 1, 2]
    labels = ["A", "B", "C", "A", "A", "D"]
    choices = iter([f"c{i}" for i in range(len(costs))])
    scorer = _cost_scorer({f"c{i}": score for i, score in enumerate(costs)})

    result = extract("query", choices, scorer=scorer, entities=labels, k=2)

    assert [item[2] for item in result] == [3, 2]


def test_ungrouped_extract_keeps_all_aliases():
    choices = {
        "a1": "oak green",
        "a2": "oak-green",
        "b1": "oak gray",
    }

    result = extract("oak green", choices, scorer=fuzz.ratio, limit=2)

    assert [item[2] for item in result] == ["a1", "a2"]


def test_bounded_ranking_matches_full_group_oracle():
    for seed in range(500):
        rng = random.Random(seed)
        size = rng.randrange(1, 25)
        k = rng.choice([1, 2, 3, 5, None])
        higher_better = rng.choice([False, True])
        labels = [rng.choice(["A", "B", "C", None]) for _ in range(size)]
        scores = [rng.randrange(6) for _ in range(size)]
        cutoff = rng.choice([None, rng.randrange(6)])
        scorer = _cost_scorer(
            {f"c{i}": score for i, score in enumerate(scores)},
            higher_better=higher_better,
        )

        result = extract(
            "q",
            iter(f"c{i}" for i in range(size)),
            scorer=scorer,
            entities=labels,
            k=k,
            **({} if cutoff is None else {"score_cutoff": cutoff}),
        )

        best = {}
        for index, (label, score) in enumerate(zip(labels, scores)):
            if cutoff is not None and (
                score < cutoff if higher_better else score > cutoff
            ):
                continue
            entity = object() if label is None else label
            rank = (-score if higher_better else score, index)
            if entity not in best or rank < best[entity][0]:
                best[entity] = (rank, (f"c{index}", score, index))

        expected = [item[1] for item in sorted(best.values(), key=lambda x: x[0])]
        if k is not None:
            expected = expected[:k]
        assert result == expected
