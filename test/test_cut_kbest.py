#encoding=utf-8
from __future__ import print_function, unicode_literals
import io
import os
import sys
import tempfile

sys.path.append("../")
import jieba


def build_tokenizer():
    path = os.path.join(tempfile.mkdtemp(), "kbest_dict.txt")
    with io.open(path, "w", encoding="utf-8") as f:
        for word, freq in [("甲", 10), ("乙", 10), ("丙", 10),
                           ("甲乙", 20), ("乙丙", 20), ("甲乙丙", 5)]:
            f.write("%s %d\n" % (word, freq))
    tokenizer = jieba.Tokenizer(path)
    tokenizer.initialize()
    return tokenizer


def main():
    tokenizer = build_tokenizer()
    sent = "甲乙丙"

    top3 = tokenizer.cut_kbest(sent, 3)
    assert top3 == [["甲乙丙"], ["甲乙", "丙"], ["甲", "乙丙"]], top3
    assert tokenizer.cut_kbest(sent, 2) == top3[:2]
    assert tokenizer.cut_kbest(sent, 1) == [["甲乙丙"]]
    assert len(tokenizer.cut_kbest(sent, 10)) == 4
    # k=1 has to agree with the classic cut
    assert tokenizer.cut_kbest(sent, 1)[0] == list(tokenizer.cut(sent, HMM=False))

    # ties at the first cut go to the farther right one
    tied = tokenizer.cut_kbest("甲乙", 2)
    assert tied == [["甲乙"], ["甲", "乙"]], tied

    # non-Han text keeps cut's per-character handling
    assert tokenizer.cut_kbest("ab", 1) == [["ab"]]
    assert tokenizer.cut_kbest("", 3) == [[]]
    assert tokenizer.cut_kbest(sent, 0) == []

    # long input: work must grow with k, not with the number of paths
    long_sent = "甲乙丙" * 200
    assert len(tokenizer.cut_kbest(long_sent, 3)) == 3
    print("kbest ok")


if __name__ == "__main__":
    main()
