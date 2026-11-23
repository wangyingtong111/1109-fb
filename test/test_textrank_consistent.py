# -*- coding: utf-8 -*-
"""
Tests for the round-consistent iteration mode of the TextRank graph.

A small symmetric graph (a path of three nodes with unit-weight edges) is
built twice: once as a--m--z, and once after renaming every node and
swapping the edge insertion order.  In the round-consistent mode the
normalized scores must follow the graph position of a node instead of its
label, so the two rounds agree node by node through the renaming map.
"""
from __future__ import absolute_import, unicode_literals

import jieba.analyse
from jieba.analyse.textrank import UndirectWeightedGraph


# renaming map used between the two rounds: a -> x, m -> y, z -> w
RENAME = {'a': 'x', 'm': 'y', 'z': 'w'}


def build_round_one():
    """Edges a--m, m--z, both of weight 1, added in natural order."""
    g = UndirectWeightedGraph()
    g.addEdge('a', 'm', 1)
    g.addEdge('m', 'z', 1)
    return g


def build_round_two():
    """Same graph after renaming (m->y is the center), edges added in
    reversed order so insertion order cannot drive the result either."""
    g = UndirectWeightedGraph()
    g.addEdge(RENAME['z'], RENAME['m'], 1)  # w--y first
    g.addEdge(RENAME['m'], RENAME['a'], 1)  # then y--x
    return g


def test_round_consistent_center_wins_and_ends_tie():
    ws = build_round_one().rank(consistent=True)

    # the center node occupies the central position, so it takes top score
    assert ws['m'] > ws['a']
    assert ws['m'] > ws['z']
    # the two endpoints are structurally identical, their weights must tie
    assert ws['a'] == ws['z']


def test_round_consistent_scores_follow_node_position_after_rename():
    first = build_round_one().rank(consistent=True)
    second = build_round_two().rank(consistent=True)

    # the center stays the highest-scoring node after renaming
    assert second['y'] > second['x']
    assert second['y'] > second['w']
    # endpoints still tie even though their labels now sort around 'y'
    assert second['x'] == second['w']

    # normalized scores correspond one node at a time through the rename map
    for old_name, new_name in RENAME.items():
        assert second[new_name] == first[old_name]


def test_round_consistent_alias_and_keyword_arguments():
    g1 = build_round_one()
    g2 = build_round_one()
    assert dict(g1.rank(consistent=True)) == dict(g2.rank(round_consistent=True))


def test_default_iteration_is_left_untouched():
    # the historical in-place iteration remains the default and keeps its
    # known (order-sensitive) normalized values on this graph
    historical = {
        'a': 0.49968078822404843,
        'm': 1.0,
        'z': 0.5047545483540439,
    }
    default_ws = dict(build_round_one().rank())
    explicit_ws = dict(build_round_one().rank(consistent=False))

    assert default_ws == historical
    assert explicit_ws == historical
    # without the new mode the symmetric endpoints do not tie, which is
    # exactly the label/order dependence the consistent mode removes
    assert default_ws['a'] != default_ws['z']


def test_textrank_high_level_accepts_consistent_flag():
    sentence = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，" \
               "增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。"

    default_tags = jieba.analyse.textrank(sentence, topK=5, withWeight=True)
    consistent_tags = jieba.analyse.textrank(
        sentence, topK=5, withWeight=True, consistent=True)
    alias_tags = jieba.analyse.textrank(
        sentence, topK=5, withWeight=True, round_consistent=True)

    assert len(default_tags) == 5
    assert len(consistent_tags) == 5
    assert alias_tags == consistent_tags
    # the two modes pick the same keywords; only their weights (and thus the
    # ordering of tied labels) may differ
    assert {w for w, _ in consistent_tags} == {w for w, _ in default_tags}
    # every reported weight stays within the normalized [0, 1] range
    assert all(0.0 <= weight <= 1.0 for _, weight in consistent_tags)
