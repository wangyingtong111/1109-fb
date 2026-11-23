#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jieba import Tokenizer


def _build_tokenizer(entries):
    """Build an isolated Tokenizer from a small static dictionary."""
    fd, path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'wb') as fh:
        for word, freq in entries:
            fh.write(('%s %d\n' % (word, freq)).encode('utf-8'))
    tokenizer = Tokenizer(path)
    tokenizer.initialize()
    return tokenizer


BASE_ENTRIES = [('甲', 50), ('乙', 50), ('甲乙', 10), ('词', 10)]
WITHOUT_CI_ENTRIES = [('甲', 50), ('乙', 50), ('甲乙', 10)]


def test_idempotent_add_keeps_total_and_unrelated_dag():
    """Re-pushing the identical word/freq must not change total or DAG."""
    # Stage 1: a single static load is the reference state.
    static = _build_tokenizer(BASE_ENTRIES)
    assert static.total == 120
    assert static.lcut('甲乙', HMM=False) == ['甲', '乙']

    dynamic = _build_tokenizer(BASE_ENTRIES)
    assert dynamic.total == 120

    # Overwrite the original freq of 「词」 with the same value 30 times.
    for _ in range(30):
        dynamic.add_word('词', 10)

    # Idempotent updates neither inflate total nor perturb the DAG route
    # of the unrelated word 「甲乙」.
    assert dynamic.total == 120
    assert dynamic.FREQ['词'] == 10
    assert dynamic.lcut('甲乙', HMM=False) == static.lcut('甲乙', HMM=False)
    assert dynamic.lcut('甲乙', HMM=False) == ['甲', '乙']


def test_overwrite_freq_nets_old_contribution():
    """Overwriting with a different freq replaces rather than accumulates."""
    tokenizer = _build_tokenizer(BASE_ENTRIES)
    tokenizer.add_word('词', 25)
    assert tokenizer.total == 135
    assert tokenizer.FREQ['词'] == 25
    tokenizer.add_word('词', 10)
    assert tokenizer.total == 120
    assert tokenizer.FREQ['词'] == 10


def test_del_word_subtracts_old_contribution_keeps_prefixes():
    """Deleting a word removes its old contribution but keeps valid prefixes."""
    # Stage 3 reference: a static dictionary that never contained 「词」.
    static = _build_tokenizer(WITHOUT_CI_ENTRIES)
    assert static.total == 110
    assert static.lcut('甲乙', HMM=False) == ['甲', '乙']

    dynamic = _build_tokenizer(BASE_ENTRIES)
    dynamic.del_word('词')

    # total loses exactly the deleted word's old contribution (120 -> 110).
    assert dynamic.total == 110
    # The word is suppressed (freq 0) but its prefix entry still exists.
    assert dynamic.FREQ.get('词') == 0
    # Segmentation is identical to loading the pruned static dictionary once.
    assert dynamic.lcut('甲乙', HMM=False) == static.lcut('甲乙', HMM=False)
    assert dynamic.lcut('甲乙', HMM=False) == ['甲', '乙']

    # Deleting a multi-char word keeps its valid prefixes.
    dynamic.del_word('甲乙')
    assert dynamic.total == 100
    assert dynamic.FREQ['甲乙'] == 0
    assert dynamic.FREQ['甲'] == 50 and dynamic.FREQ['乙'] == 50

    # Re-adding a deleted (zero-freq) word restores its contribution once.
    dynamic.add_word('甲乙', 10)
    assert dynamic.total == 110
    assert dynamic.FREQ['甲乙'] == 10


if __name__ == '__main__':
    test_idempotent_add_keeps_total_and_unrelated_dag()
    test_overwrite_freq_nets_old_contribution()
    test_del_word_subtracts_old_contribution_keeps_prefixes()
    print('all tests passed')
