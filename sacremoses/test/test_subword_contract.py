# -*- coding: utf-8 -*-

"""Tests for the portable BPE rule save/load and unseen-word encoding."""

import json
import os
import tempfile
import unittest

from sacremoses.subwords import EOW, SubwordTokenizer


def make_bpe(*pairs, **kwargs):
    return SubwordTokenizer(merge_rules=list(pairs), **kwargs)


class OrderedApplicationTests(unittest.TestCase):
    def test_acceptance_triplet(self):
        # normal: simple learned merge
        self.assertEqual(make_bpe(("a", "b")).encode("ab"), ["ab"])
        # conflict: earlier a+b must win over the later b+c on "abc"
        self.assertEqual(
            make_bpe(("a", "b"), ("b", "c")).encode("abc"), ["ab", "c"]
        )
        # regression fixture from the deterministic counter-cases
        self.assertEqual(make_bpe(("a", "c")).encode("aca"), ["ac", "a"])

    def test_earlier_rule_beats_longest_later_match(self):
        # A literal-longest-match shortcut would return ["a", "bc"].
        bpe = make_bpe(("a", "b"), ("b", "c"))
        self.assertEqual(bpe.encode("abc"), ["ab", "c"])
        # Repeated application keeps choosing the earliest current pair.
        self.assertEqual(make_bpe(("a", "a")).encode("aaa"), ["aa", "a"])
        self.assertEqual(make_bpe(("a", "a")).encode("aaaa"), ["aa", "aa"])

    def test_learning_order_is_exposed(self):
        bpe = make_bpe(("a", "b"), ("b", "c"))
        self.assertEqual(bpe.get_merge_rules(), [("a", "b"), ("b", "c")])
        self.assertEqual(bpe.merge_ranks[("a", "b")], 0)
        self.assertEqual(bpe.merge_ranks[("b", "c")], 1)

    def test_dict_ranks_are_ordered_by_rank(self):
        bpe = SubwordTokenizer(
            bpe_codes={("b", "c"): 1, ("a", "b"): 0}
        )
        self.assertEqual(bpe.encode("abc"), ["ab", "c"])

    def test_string_rules_are_accepted(self):
        bpe = SubwordTokenizer(codes=["a b", "b c"])
        self.assertEqual(bpe.encode("abc"), ["ab", "c"])


class VocabularyFallbackTests(unittest.TestCase):
    def setUp(self):
        self.bpe = make_bpe(("a", "b"), ("b", "c"))

    def test_accepting_vocab_keeps_merge(self):
        self.assertEqual(
            self.bpe.encode("abc", vocab={"ab", "c"}), ["ab", "c"]
        )

    def test_rejected_merge_expands_only_itself(self):
        # "ab" is not an allowed piece: undo just that merge into a, b.
        self.assertEqual(
            self.bpe.encode("abc", vocab={"a", "b", "c"}), ["a", "b", "c"]
        )

    def test_neighbouring_accepted_merge_is_untouched(self):
        bpe = make_bpe(("a", "b"), ("c", "d"))
        # ab rejected -> a, b ; cd accepted and stays merged.
        self.assertEqual(
            bpe.encode("abcd", vocab={"a", "b", "cd"}), ["a", "b", "cd"]
        )

    def test_rejected_early_merge_is_only_expanded(self):
        # The later b+c rule must not be retrospectively applied after ab is
        # rejected by the vocabulary: undo only the rejected merge itself.
        self.assertEqual(
            self.bpe.encode("abc", vocab={"a", "b", "bc"}),
            ["a", "b", "c"],
        )

    def test_unknown_atomic_symbols_survive_vocab(self):
        # Even with a restrictive vocabulary, unseen characters are kept.
        self.assertEqual(
            self.bpe.encode("abz?", vocab={"ab", "a", "b", "c"}),
            ["ab", "z", "?"],
        )

    def test_vocab_accepts_end_marked_forms(self):
        bpe = make_bpe(("a", "b"))
        # Vocabulary may list the merged piece with the private end marker.
        self.assertEqual(
            bpe.encode("ab", vocab={"ab" + EOW}), ["ab"]
        )
        self.assertEqual(
            bpe.encode("ab", vocab={"ab</w>"}), ["ab"]
        )

    def test_vocab_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vocab.txt")
            with open(path, "w", encoding="utf-8") as fout:
                fout.write("ab 3\nc 9\n")
            self.assertEqual(
                self.bpe.encode("abc", vocab=path), ["ab", "c"]
            )


class EndOfWordRuleTests(unittest.TestCase):
    def setUp(self):
        # Same pair shapes the trainer produces for "abd abd bcd".
        self.bpe = make_bpe(("a", "b"), ("b", "d" + EOW))

    def test_end_rule_applies_at_word_end(self):
        self.assertEqual(self.bpe.encode("cbd"), ["c", "bd"])

    def test_end_rule_does_not_apply_internally(self):
        # d here is internal (e follows), so b+d<EOW> must not merge.
        self.assertEqual(self.bpe.encode("bde"), ["b", "d", "e"])

    def test_internal_and_end_rules_coexist(self):
        # a+b is context-free enough for the internal a,b here; the end
        # rule then merges b with the word-final d.
        self.assertEqual(self.bpe.encode("abd"), ["ab", "d"])

    def test_marked_output_retains_marker(self):
        self.assertEqual(
            self.bpe.encode("abd", marked=True), ("ab", "d" + EOW)
        )

    def test_subword_nmt_end_notation_loads(self):
        bpe = SubwordTokenizer.from_codes([("a", "b"), ("b", "c</w>")])
        self.assertEqual(bpe.encode("abc"), ["ab", "c"])
        # The end-specific b+c rule must not fire on internal bc.
        self.assertEqual(bpe.encode("bcd"), ["b", "c", "d"])


class UnknownCharacterAndLosslessTests(unittest.TestCase):
    def test_unknown_characters_are_dropped_never(self):
        bpe = make_bpe(("a", "b"))
        self.assertEqual(bpe.encode("abx"), ["ab", "x"])
        self.assertEqual(bpe.encode("zzz"), ["z", "z", "z"])
        self.assertEqual(bpe.encode(""), [])

    def test_concatenating_pieces_restores_word(self):
        bpe = make_bpe(("a", "b"), ("b", "c"), ("c", "d"))
        for word in ["abc", "abcd", "xyz", "abracadabra", ""]:
            pieces = bpe.encode(word)
            self.assertEqual("".join(pieces), word)

    def test_batch_and_line_helpers(self):
        bpe = make_bpe(("a", "b"))
        self.assertEqual(bpe.encode(["ab", "xab"]), [["ab"], ["x", "ab"]])
        self.assertEqual(bpe.encode_line("ab xab"), [["ab"], ["x", "ab"]])
        self.assertEqual(bpe.apply_bpe("ab xab"), "ab x@@ ab")


class SaveReloadTests(unittest.TestCase):
    def setUp(self):
        self.bpe = make_bpe(
            ("a", "b"), ("b", "c"), ("b", "d" + EOW), ("c", "d" + EOW)
        )
        self.words = ["ab", "abc", "abd", "cbd", "bcd", "bde", "zzz", ""]

    def test_roundtrip_through_classmethod(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.bpe")
            self.bpe.save_merge_rules(path)
            reloaded = SubwordTokenizer.load_merge_rules(path)
            for word in self.words:
                self.assertEqual(
                    reloaded.encode(word), self.bpe.encode(word), word
                )
            self.assertEqual(reloaded.merge_rules, self.bpe.merge_rules)

    def test_constructor_detects_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.bpe")
            self.bpe.save_merge_rules(path, include_header=True)
            reloaded = SubwordTokenizer(path)
            self.assertEqual(reloaded.encode("cbd"), ["c", "bd"])

    def test_saved_file_is_plain_pair_lines_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.bpe")
            make_bpe(("a", "b"), ("b", "c")).save_merge_rules(path)
            with open(path, encoding="utf-8") as fin:
                self.assertEqual(fin.read().splitlines(), ["a b", "b c"])
            # The plain file is also recognised positionally.
            reloaded = SubwordTokenizer(path)
            self.assertEqual(reloaded.encode("abc"), ["ab", "c"])

    def test_load_from_keyword_and_instance_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.bpe")
            self.bpe.save_merge_rules(path)
            via_kw = SubwordTokenizer(load_from=path)
            self.assertEqual(via_kw.encode("abc"), ["ab", "c"])

            instance = SubwordTokenizer()
            self.assertIs(instance.load_merge_rules(path), instance)
            self.assertEqual(instance.encode("abc"), ["ab", "c"])

    def test_header_less_plain_codes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plain.bpe")
            with open(path, "w", encoding="utf-8") as fout:
                fout.write("a b\nb c\n")
            reloaded = SubwordTokenizer.load_codes(path)
            self.assertEqual(reloaded.encode("abc"), ["ab", "c"])
            # Explicit keyword and positional sniff both accept plain codes.
            via_kw = SubwordTokenizer(load_from=path)
            self.assertEqual(via_kw.encode("abc"), ["ab", "c"])
            positional = SubwordTokenizer(path)
            self.assertEqual(positional.encode("abc"), ["ab", "c"])

    def test_aliases_share_behaviour(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rules.model")
            self.bpe.save_model(path)
            reloaded = SubwordTokenizer.load_model(path)
            self.assertEqual(reloaded.encode("abc"), ["ab", "c"])

    def test_json_rules_text(self):
        data = json.dumps([["a", "b"], ["b", "c"]])
        bpe = SubwordTokenizer(load_from=data)
        self.assertEqual(bpe.encode("abc"), ["ab", "c"])


class TrainingIntegrationTests(unittest.TestCase):
    CORPUS = "abd abd bcd"

    def _train(self, tmp):
        path = os.path.join(tmp, "vocab.txt")
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(self.CORPUS)
        return SubwordTokenizer(path), path

    def test_pair_statistics_are_the_historical_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok, _ = self._train(tmp)
            self.assertEqual(
                dict(tok.stats),
                {
                    ("a", "b"): 2,
                    ("b", "d" + EOW): 2,
                    ("b", "c"): 1,
                    ("c", "d" + EOW): 1,
                },
            )
            self.assertTrue(callable(getattr(tok, "encode", None)))

    def test_learn_records_ordered_rules_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok, corpus_path = self._train(tmp)
            tok.learn(1)
            self.assertEqual(len(tok.merge_rules), 1)
            self.assertIn(
                tok.merge_rules[0],
                [("a", "b"), ("b", "d" + EOW)],
            )

            # A longer learning run records a non-empty ordered rule list.
            tok.learn(6, min_freq=0)
            self.assertTrue(tok.merge_rules)
            rules_path = os.path.join(tmp, "learned.bpe")
            tok.save_merge_rules(rules_path)
            reloaded = SubwordTokenizer.load_merge_rules(rules_path)
            for word in ["abd", "bcd", "abc", "cbd", "unseen", "x"]:
                self.assertEqual(
                    reloaded.encode(word), tok.encode(word), word
                )
                self.assertEqual("".join(reloaded.encode(word)), word)

    def test_top_level_package_export(self):
        import sacremoses

        self.assertIs(sacremoses.SubwordTokenizer, SubwordTokenizer)


if __name__ == "__main__":
    unittest.main()
