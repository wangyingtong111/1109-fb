# -*- coding: utf-8 -*-

"""
Tests for MosesTokenizer
"""

import unittest

from sacremoses.normalize import MosesPunctNormalizer


class TestNormalizer(unittest.TestCase):
    def test_moses_normalize_documents(self):
        moses = MosesPunctNormalizer()
        # Examples from normalizing big.txt
        inputs = [
            "The United States in 1805 (color map)                 _Facing_     193",
            "=Formation of the Constitution.=--(1) The plans before the convention,",
            "directions--(1) The infective element must be eliminated. When the ulcer",
            "College of Surgeons, Edinburgh.)]",
        ]
        expected = [
            "The United States in 1805 (color map) _Facing_ 193",
            "=Formation of the Constitution.=-- (1) The plans before the convention,",
            "directions-- (1) The infective element must be eliminated. When the ulcer",
            "College of Surgeons, Edinburgh.) ]",
        ]
        for text, expect in zip(inputs, expected):
            assert moses.normalize(text) == expect

    def test_moses_normalize_quote_comma(self):
        moses_norm_quote = MosesPunctNormalizer("en", norm_quote_commas=True)
        moses_no_norm_quote = MosesPunctNormalizer("en", norm_quote_commas=False)
        text = 'THIS EBOOK IS OTHERWISE PROVIDED TO YOU "AS-IS".'

        expected_norm_quote = 'THIS EBOOK IS OTHERWISE PROVIDED TO YOU "AS-IS."'
        assert moses_norm_quote.normalize(text) == expected_norm_quote

        expected_no_norm_quote = 'THIS EBOOK IS OTHERWISE PROVIDED TO YOU "AS-IS".'
        assert moses_no_norm_quote.normalize(text) == expected_no_norm_quote

    def test_moses_normalize_numbers(self):
        # See https://stackoverflow.com/a/55233871/610569
        moses_norm_num = MosesPunctNormalizer("en", norm_numbers=True)
        moses_no_norm_num = MosesPunctNormalizer("en", norm_numbers=False)

        text = "12{}123".format("\u00A0")
        expected = "12.123"
        assert moses_norm_num.normalize(text) == expected

        text = expected = "12 123"
        assert moses_no_norm_num.normalize(text) == expected

    def test_moses_noralize_single_apostrophe(self):
        moses_norm_num = MosesPunctNormalizer("en")
        text = "yesterday ’s reception"
        expected = "yesterday 's reception"
        assert moses_norm_num.normalize(text) == expected

    def test_replace_unicode_punct(self):
        moses_norm_unicode = MosesPunctNormalizer()
        text = "０《１２３》 ４５６％ 【７８９】"
        expected = '0"123" 456% [789]'
        assert moses_norm_unicode.replace_unicode_punct(text) == expected

    def test_normalization_pipeline(self):
        moses_norm_unicode = MosesPunctNormalizer(
            pre_replace_unicode_punct=True, post_remove_control_chars=True
        )
        text = "０《１２３》      ４５６％  '' 【７８９】"
        expected = '0"123" 456% " [789]'
        assert moses_norm_unicode.normalize(text) == expected

    def test_moses_normalize_with_perl_parity(self):
        moses_perl_parity = MosesPunctNormalizer(perl_parity=True)
        text = 'from the ‘bad bank’, Northern, wala\u00A0«\u00A0dox ci jawwu Les «\u00A0wagonways\u00A0»\u00A0étaient construits'
        expected = '''from the 'bad bank," Northern, wala "dox ci jawwu Les "wagonways" étaient construits'''
        assert moses_perl_parity.normalize(text) == expected


class TestLocaleNumberGrouping(unittest.TestCase):
    """Optional locale thousands grouping on MosesPunctNormalizer."""

    NBSP = "\u00A0"
    NNBSP = "\u202F"

    def test_french_grouped_number_is_merged_and_located(self):
        moses = MosesPunctNormalizer(lang="fr", locale_number_grouping=True)
        text = "12" + self.NBSP + "345,50"
        assert moses.normalize(text, return_spans=True) == ("12345,50", [(0, 9, text)])

    def test_narrow_nbsp_is_also_a_separator(self):
        moses = MosesPunctNormalizer(lang="fr", locale_number_grouping=True)
        text = "1" + self.NNBSP + "234" + self.NBSP + "567,80"
        assert moses.normalize(text, return_spans=True) == ("1234567,80", [(0, 12, text)])

    def test_invalid_grouping_is_left_as_separate_numbers(self):
        # `1 23 456,78` is not a legal grouping: the middle group is too short.
        moses = MosesPunctNormalizer(lang="fr", locale_number_grouping=True)
        text = "1" + self.NBSP + "23" + self.NBSP + "456,78"
        assert moses.normalize(text, return_spans=True) == ("1 23 456,78", [])

    def test_only_merged_numbers_are_reported(self):
        moses = MosesPunctNormalizer(lang="fr", locale_number_grouping=True)
        group = "1" + self.NNBSP + "234" + self.NBSP + "567,80"
        text = ("\u00ABA\u00BB1" + self.NBSP + "23" + self.NBSP + "456,78\u2014B "
                + group + " et 1234567,80")
        normalized, spans = moses.normalize(text, return_spans=True)
        assert normalized == '"A"1 23 456,78 - B 1234567,80 et 1234567,80'
        # The offset is into the *input*, where the dash is one character; the
        # output is longer because `\u2014` became ` - `.
        anchor = text.index(group)
        assert spans == [(anchor, anchor + len(group), group)]
        assert [text[start:end] for start, end, _ in spans] == [group]

    def test_output_is_stable_under_reprocessing(self):
        moses = MosesPunctNormalizer(lang="fr", locale_number_grouping=True)
        text = "12" + self.NBSP + "345,50"
        once, spans = moses.normalize(text, return_spans=True)
        assert moses.normalize(once, return_spans=True) == (once, [])

    def test_disabled_mode_keeps_nbsp_behaviour(self):
        moses = MosesPunctNormalizer(lang="fr")
        text = "12" + self.NBSP + "345,50"
        assert moses.normalize(text) == "12,345,50"

    def test_default_word_boundaries_are_untouched(self):
        moses = MosesPunctNormalizer(lang="fr", locale_number_grouping=True)
        assert moses.normalize("cat dog", return_spans=True) == ("cat dog", [])

    def test_dot_decimal_language_uses_its_own_mark(self):
        moses = MosesPunctNormalizer(lang="en", locale_number_grouping=True)
        text = "12" + self.NBSP + "345.50"
        assert moses.normalize(text, return_spans=True) == ("12345.50", [(0, 9, text)])
