#encoding=utf-8
from __future__ import print_function, unicode_literals
import sys
sys.path.append("../")
import jieba

# Build an isolated tokenizer: main path is 量子/计算框架,
# the dictionary additionally contains the cross-boundary word 量子计算.
tk = jieba.Tokenizer()
tk.initialize()
tk.add_word('量子', 100)
tk.add_word('计算框架', 100000)
tk.add_word('量子计算', 100)

sentence = '量子计算框架。量子计算框架'


def test_main_path():
    assert tk.lcut(sentence) == ['量子', '计算框架', '。', '量子', '计算框架']


def test_extra_words_with_max_len():
    tokens = list(tk.tokenize(sentence, mode="search", extra_max_len=4))
    # the two copies of the附加词 each carry their own half-open span
    extras = [t for t in tokens if t[0] == '量子计算']
    assert [(t[1], t[2]) for t in extras] == [(0, 4), (7, 11)], extras
    # main word 计算框架 appears exactly once per occurrence
    mains = [t for t in tokens if t[0] == '计算框架']
    assert [(t[1], t[2]) for t in mains] == [(2, 6), (9, 13)], mains
    # punctuation is a hard boundary: nothing spans across 。 at [6, 7)
    for w, s, e in tokens:
        assert not (s < 6 < e or s < 7 < e), (w, s, e)
    # cut_for_search and tokenize yield words in the same order
    assert tk.lcut_for_search(sentence, extra_max_len=4) == \
        [t[0] for t in tokens]


def test_extra_max_len_is_a_cap():
    # 量子计算 is 4 chars long, so a cap of 3 excludes it
    assert '量子计算' not in tk.lcut_for_search(sentence, extra_max_len=3)
    assert '量子计算' in tk.lcut_for_search(sentence, extra_max_len=4)


def test_default_behavior_unchanged():
    # without extra_max_len no cross-boundary words are emitted
    assert '量子计算' not in tk.lcut_for_search(sentence)
    # main words are never dropped
    result = tk.lcut_for_search(sentence, extra_max_len=4)
    for w in ('量子', '计算框架'):
        assert result.count(w) == 2, (w, result)


if __name__ == "__main__":
    test_main_path()
    test_extra_words_with_max_len()
    test_extra_max_len_is_a_cap()
    test_default_behavior_unchanged()
    print("all tests passed")
