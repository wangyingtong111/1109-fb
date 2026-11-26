import unittest

from nltk import RegexpParser, Tree


class TestChunkRule(unittest.TestCase):
    def test_tag_pattern2re_pattern_quantifier(self):
        """Test for bug https://github.com/nltk/nltk/issues/1597

        Ensures that curly bracket quantifiers can be used inside a chunk rule.
        This type of quantifier has been used for the supplementary example
        in https://www.nltk.org/book/ch07.html#exploring-text-corpora.
        """
        sent = [
            ("The", "AT"),
            ("September-October", "NP"),
            ("term", "NN"),
            ("jury", "NN"),
            ("had", "HVD"),
            ("been", "BEN"),
            ("charged", "VBN"),
            ("by", "IN"),
            ("Fulton", "NP-TL"),
            ("Superior", "JJ-TL"),
            ("Court", "NN-TL"),
            ("Judge", "NN-TL"),
            ("Durwood", "NP"),
            ("Pye", "NP"),
            ("to", "TO"),
            ("investigate", "VB"),
            ("reports", "NNS"),
            ("of", "IN"),
            ("possible", "JJ"),
            ("``", "``"),
            ("irregularities", "NNS"),
            ("''", "''"),
            ("in", "IN"),
            ("the", "AT"),
            ("hard-fought", "JJ"),
            ("primary", "NN"),
            ("which", "WDT"),
            ("was", "BEDZ"),
            ("won", "VBN"),
            ("by", "IN"),
            ("Mayor-nominate", "NN-TL"),
            ("Ivan", "NP"),
            ("Allen", "NP"),
            ("Jr.", "NP"),
            (".", "."),
        ]  # source: brown corpus
        cp = RegexpParser("CHUNK: {<N.*>{4,}}")
        tree = cp.parse(sent)
        assert (
            tree.pformat()
            == """(S
  The/AT
  September-October/NP
  term/NN
  jury/NN
  had/HVD
  been/BEN
  charged/VBN
  by/IN
  Fulton/NP-TL
  Superior/JJ-TL
  (CHUNK Court/NN-TL Judge/NN-TL Durwood/NP Pye/NP)
  to/TO
  investigate/VB
  reports/NNS
  of/IN
  possible/JJ
  ``/``
  irregularities/NNS
  ''/''
  in/IN
  the/AT
  hard-fought/JJ
  primary/NN
  which/WDT
  was/BEDZ
  won/VBN
  by/IN
  (CHUNK Mayor-nominate/NN-TL Ivan/NP Allen/NP Jr./NP)
  ./.)"""
        )


class TestRegexpParserColumns(unittest.TestCase):
    def setUp(self):
        self.parser = RegexpParser(
            "NP: {<JJ>?<NN>}\nWrap: {<NP><OLD>}",
            loop=2,
        )
        self.red = ("red", "JJ")
        self.tea = ("tea", "NN")
        self.dark = ("dark", "NN")

    def test_tokens_in_one_column_still_chunk(self):
        result = self.parser.parse_columns([[self.red, self.tea]])

        assert [tree.pformat() for tree in result.trees] == [
            "(S (NP red/JJ tea/NN))"
        ]
        assert result.positions == [(0, 0), (0, 1)]

    def test_chunk_rules_do_not_cross_column_or_empty_column_boundaries(self):
        columns = [
            [self.red],
            [],
            [self.tea, Tree("OLD", [self.dark])],
        ]

        result = self.parser.parse_columns(columns)

        assert [tree.pformat() for tree in result.trees] == [
            "(S red/JJ)",
            "(S )",
            "(S (Wrap (NP tea/NN) (OLD dark/NN)))",
        ]
        assert result.positions == [(0, 0), (2, 1), (2, 2)]
        assert [tree.column_index for tree in result.trees] == [0, 1, 2]
        assert result.trees[2].leaf_positions == [(2, 1), (2, 2)]
        # The position objects remain tuple-compatible.
        assert [tuple(position) for position in result.positions] == [
            (0, 0),
            (2, 1),
            (2, 2),
        ]

    def test_nested_list_input_is_column_aware_but_flat_input_unchanged(self):
        columns = [[self.red], [], [self.tea]]

        column_result = self.parser.parse(columns)
        assert column_result.trees[0].pformat() == "(S red/JJ)"
        assert column_result.trees[2].pformat() == "(S (NP tea/NN))"

        flat_tree = self.parser.parse([self.red, self.tea])
        assert flat_tree.pformat() == "(S (NP red/JJ tea/NN))"

    def test_columns_keyword_entry_point(self):
        result = self.parser.parse(columns=[[self.red, self.tea]])
        assert result.trees[0].pformat() == "(S (NP red/JJ tea/NN))"
        assert result.positions == [(0, 0), (0, 1)]
