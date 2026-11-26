#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import regex

from itertools import chain


class MosesPunctNormalizer:
    """
    This is a Python port of the Moses punctuation normalizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl
    """

    EXTRA_WHITESPACE = [  # lines 21 - 30
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]

    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]  # lines 33 - 34

    NORMALIZE_UNICODE = [  # lines 37 - 50
        ("„", r'"'),
        ("“", r'"'),
        ("”", r'"'),
        ("–", r"-"),
        ("—", r" - "),
        (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"),
        ("‚", r"'"),
        ("’", r"'"),
        (r"''", r'"'),
        ("´´", r'"'),
        ("…", r"..."),
    ]

    FRENCH_QUOTES = [  # lines 52 - 57
        ("\u00A0«\u00A0", r'"'),
        ("«\u00A0", r'"'),
        ("«", r'"'),
        ("\u00A0»\u00A0", r'"'),
        ("\u00A0»", r'"'),
        ("»", r'"'),
    ]

    HANDLE_PSEUDO_SPACES = [  # lines 59 - 67
        ("\u00A0%", r"%"),
        ("nº\u00A0", "nº "),
        ("\u00A0:", r":"),
        ("\u00A0ºC", " ºC"),
        ("\u00A0cm", r" cm"),
        ("\u00A0\\?", "?"),
        ("\u00A0\\!", "!"),
        ("\u00A0;", r";"),
        (",\u00A0", r", "),
        (r" +", r" "),
    ]

    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]

    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),  # don't fix period at end of sentence
    ]

    DE_ES_CZ_CS_FR = [
        ("(\\d)\u00A0(\\d)", r"\g<1>,\g<2>"),
    ]

    OTHER = [
        ("(\\d)\u00A0(\\d)", r"\g<1>.\g<2>"),
    ]

    #: Characters that may act as a thousands separator inside a locale number:
    #: a no-break space and a narrow no-break space.
    GROUP_SEPARATORS = "\u00A0\u202F"
    GROUP_SEPARATOR_RE = re.compile("[\u00A0\u202F]")

    #: Languages whose decimal mark is a comma, i.e. ``12 345,50``. Everything
    #: else defaults to the dot, i.e. ``12 345.50``.
    COMMA_DECIMAL_LANGS = frozenset(
        [
            "bg", "ca", "cs", "cz", "da", "de", "el", "es", "et", "eu", "fa",
            "fi", "fr", "gl", "hr", "hu", "id", "is", "it", "lt", "lv", "mk",
            "nb", "nl", "nn", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sq",
            "sr", "sv", "tr", "uk", "vi",
        ]
    )

    # Regex substitutions from replace-unicode-punctuation.perl
    # https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    REPLACE_UNICODE_PUNCTUATION = [
        ("，", ","),
        (r"。\s*", ". "),
        ("、", ","),
        ("”", '"'),
        ("“", '"'),
        ("∶", ":"),
        ("：", ":"),
        ("？", "?"),
        ("《", '"'),
        ("》", '"'),
        ("）", ")"),
        ("！", "!"),
        ("（", "("),
        ("；", ";"),
        ("」", '"'),
        ("「", '"'),
        ("０", "0"),
        ("１", "1"),
        ("２", "2"),
        ("３", "3"),
        ("４", "4"),
        ("５", "5"),
        ("６", "6"),
        ("７", "7"),
        ("８", "8"),
        ("９", "9"),
        (r"．\s*", ". "),
        ("～", "~"),
        ("’", "'"),
        ("…", "..."),
        ("━", "-"),
        ("〈", "<"),
        ("〉", ">"),
        ("【", "["),
        ("】", "]"),
        ("％", "%"),
    ]

    def __init__(
        self,
        lang="en",
        penn=True,
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=False,
        post_remove_control_chars=False,
        perl_parity=False,
        locale_number_grouping=False,
    ):
        """
        :param language: The two-letter language code.
        :type lang: str
        :param penn: Normalize Penn Treebank style quotations.
        :type penn: bool
        :param norm_quote_commas: Normalize quotations and commas
        :type norm_quote_commas: bool
        :param norm_numbers: Normalize numbers
        :type norm_numbers: bool
        :param perl_parity: exact parity with perl script
        :type perl_parity: bool
        :param locale_number_grouping: Merge numbers written with locale
            thousands grouping, e.g. French ``12 345,50`` -> ``12345,50``.
            Off by default, in which case numbers keep the historical NBSP
            handling. With this on, ``normalize(text, return_spans=True)``
            also reports the input range each merged number came from.
        :type locale_number_grouping: bool
        """

        if perl_parity:
            # Copy first: these are class attributes, so assigning into them
            # in place would rewrite the defaults for every other normalizer in
            # the process, including ones built with perl_parity=False.
            self.NORMALIZE_UNICODE = list(self.NORMALIZE_UNICODE)
            self.FRENCH_QUOTES = list(self.FRENCH_QUOTES)
            self.NORMALIZE_UNICODE[11] = ("’", r'"')
            self.FRENCH_QUOTES[0] = ("\u00A0«\u00A0", r' "')
            self.FRENCH_QUOTES[3] = ("\u00A0»\u00A0", r'" ')

        self.substitutions = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]

        if penn:  # Adds the penn substitutions after extra_whitespace regexes.
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)

        if norm_quote_commas:
            if lang == "en":
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)

        # Kept in both modes. Locale grouping resolves and consumes the
        # separators of every candidate number before this rule runs, so all it
        # can still see are NBSPs that no candidate claimed; a merged number is
        # separator-free already and passes through untouched.
        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)

        self.substitutions = list(chain(*self.substitutions))

        self.pre_replace_unicode_punct = pre_replace_unicode_punct
        self.post_remove_control_chars = post_remove_control_chars

        # Locale grouping is not a substitution: it needs to know where each
        # number sat in the caller's string, which the substitution chain
        # destroys as soon as a rule changes a single character's length.
        self.locale_number_grouping = locale_number_grouping
        #: Spans of the numbers merged by the most recent ``normalize()`` call,
        #: in the same shape as ``normalize(..., return_spans=True)`` returns.
        self.number_spans = []
        self.decimal_mark = "," if lang in self.COMMA_DECIMAL_LANGS else "."
        self.number_candidate_re = re.compile(
            r"(?<!\d)\d+(?:[{seps}]\d+)+(?:{mark}\d+)?".format(
                seps=self.GROUP_SEPARATORS, mark=re.escape(self.decimal_mark)
            )
        )

    def group_numbers(self, text):
        """
        Merges numbers written with locale thousands grouping.

        A candidate is a maximal run of digit groups joined by NBSP / narrow
        NBSP (plus the language's decimal mark and digits, if present). It is
        merged only when every group is well formed: one to three digits in
        the first group, exactly three in each of the rest. Otherwise the
        candidate is not a number at all and is kept as separate numbers, its
        separators turned into ordinary spaces so the pieces stay apart.

        :return: ``(text, spans)``. Each span is ``(start, end, source)``, a
            half-open Python character range into ``text`` and the original
            grouped spelling found there.
        """
        spans = []

        def replace(match):
            candidate = match.group(0)
            integer, mark, fraction = candidate.partition(self.decimal_mark)
            groups = self.GROUP_SEPARATOR_RE.split(integer)
            widths = [len(group) for group in groups]
            if 1 <= widths[0] <= 3 and all(width == 3 for width in widths[1:]):
                spans.append((match.start(), match.end(), candidate))
                return "".join(groups) + mark + fraction
            return " ".join(groups) + mark + fraction

        return self.number_candidate_re.sub(replace, text), spans

    def normalize(self, text, return_spans=False):
        """
        Returns a string with normalized punctuation.

        With ``return_spans=True`` the result is ``(text, spans)`` instead.
        Each span is ``(start, end, source)``: a half-open Python character
        range into the input that still spells the number with its grouping
        separators, and that spelling itself. Offsets always address the
        input, never the output, whose length the punctuation rules change.
        """
        text = str(text)

        # Grouping runs on the raw input, before any rule that inserts or
        # removes characters, so the reported offsets stay valid even when
        # ``—`` becomes ``" - "`` or ``«`` becomes ``"``.
        spans = []
        if self.locale_number_grouping:
            text, spans = self.group_numbers(text)
        self.number_spans = spans

        # Optionally, replace unicode puncts BEFORE normalization.
        if self.pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)

        # Actual normalization.
        for regexp, substitution in self.substitutions:
            # print(regexp, substitution)
            text = re.sub(regexp, substitution, text)
            # print(text)

        # Optionally, replace unicode puncts BEFORE normalization.
        if self.post_remove_control_chars:
            text = self.remove_control_chars(text)

        text = text.strip()
        return (text, spans) if return_spans else text

    def replace_unicode_punct(self, text):
        for regexp, substitution in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, str(text))
        return text

    def remove_control_chars(self, text):
        return regex.sub(r"\p{C}", "", text)
