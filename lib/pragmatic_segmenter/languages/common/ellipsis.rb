# -*- encoding : utf-8 -*-
# frozen_string_literal: true

module PragmaticSegmenter
  module Languages
    module Common
      # This class searches for ellipses within a string and
      # replaces the periods.

      # http://www.dailywritingtips.com/in-search-of-a-4-dot-ellipsis/
      # http://www.thepunctuationguide.com/ellipses.html

      module EllipsisRules
        # Rubular: http://rubular.com/r/i60hCK81fz
        ThreeConsecutiveRule = Rule.new(/\.\.\.(?=\s+[A-Z])/, '☏.')

        # Opening quotation marks that can stand between an ellipsis and the
        # first word of whatever follows it. One character, so "'Tis" after a
        # pause is still read as a word starting with an apostrophe, not as a
        # quotation mark hiding a capital.
        OPENING_QUOTATION_REGEX = /[«‹“"„‘‚「『]/

        # Optional rule for interview transcripts. Instead of only accepting
        # an ASCII capital as evidence of a new sentence, it walks over the
        # whitespace (and any opening quotation marks) and looks at the first
        # letter of the next word. Both a precomposed É and a decomposed
        # E + U+0301 are capitalized, so the same input written in either
        # normalization form gets the same boundary. Only the three periods are
        # turned into the ellipsis marker, so the returned text keeps the
        # original glyphs and quotation marks.
        CapitalizedWordRule = Rule.new(
          /(?<!\.)\.\.\.(?=\s+#{OPENING_QUOTATION_REGEX.source}?\s*[\p{Lu}\p{Lt}])/,
          '☏.'
        )

        # Rubular: http://rubular.com/r/Hdqpd90owl
        FourConsecutiveRule = Rule.new(/(?<=\S)\.{3}(?=\.\s[A-Z])/, 'ƪ')

        # Rubular: http://rubular.com/r/YBG1dIHTRu
        ThreeSpaceRule = Rule.new(/(\s\.){3}\s/, '♟')

        # Rubular: http://rubular.com/r/2VvZ8wRbd8
        FourSpaceRule = Rule.new(/(?<=[a-z])(\.\s){3}\.(\z|$|\n)/, '♝')

        OtherThreePeriodRule = Rule.new(/\.\.\./, 'ƪ')

        All = [
          ThreeSpaceRule,
          FourSpaceRule,
          FourConsecutiveRule,
          ThreeConsecutiveRule,
          OtherThreePeriodRule
        ]
      end
    end
  end
end
