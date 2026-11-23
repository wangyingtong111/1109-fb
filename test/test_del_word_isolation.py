#encoding=utf-8
from __future__ import print_function, unicode_literals
import os
import shutil
import sys
import tempfile

try:
    from io import open
except ImportError:
    pass

sys.path.append("../")
import jieba


def main():
    tmp_dir = tempfile.mkdtemp(prefix="jieba_del_word_")
    dict_path = os.path.join(tmp_dir, "mini_dict.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        # Only a handful of common characters, so that "杭研" is not a
        # dictionary word and can only be produced by HMM new word discovery.
        f.write("的 100000\n是 50000\n我 40000\n杭 100\n研 100\n")

    try:
        a = jieba.Tokenizer(dict_path)
        b = jieba.Tokenizer(dict_path)

        # Both tokenizers start from the same dictionary: "杭研" is unknown
        # to the prefix dict, so the HMM joins it into one word.
        assert a.FREQ.get("杭研") is None
        assert b.FREQ.get("杭研") is None
        assert a.lcut("杭研") == ["杭研"], a.lcut("杭研")
        assert b.lcut("杭研") == ["杭研"], b.lcut("杭研")

        # Deleting the word on A must not leak into B.
        a.del_word("杭研")
        assert a.lcut("杭研") == ["杭", "研"], a.lcut("杭研")
        assert b.lcut("杭研") == ["杭研"], b.lcut("杭研")

        # A high frequency add brings the word back on A only.
        a.add_word("杭研", 100000)
        assert a.lcut("杭研") == ["杭研"], a.lcut("杭研")
        assert b.lcut("杭研") == ["杭研"], b.lcut("杭研")

        # B keeps deciding by its own dictionary/HMM state.
        b.del_word("杭研")
        assert b.lcut("杭研") == ["杭", "研"], b.lcut("杭研")
        assert a.lcut("杭研") == ["杭研"], a.lcut("杭研")

        # Re-adding on B also restores the word there.
        b.add_word("杭研", 100000)
        assert b.lcut("杭研") == ["杭研"], b.lcut("杭研")
        assert a.lcut("杭研") == ["杭研"], a.lcut("杭研")

        print("del_word isolation test passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
