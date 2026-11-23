# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/punctuation_replacer'
require 'pragmatic_segmenter/between_punctuation'


require 'pragmatic_segmenter/list'
require 'pragmatic_segmenter/abbreviation_replacer'
require 'pragmatic_segmenter/exclamation_words'

module PragmaticSegmenter
  # This class processing segmenting the text.
  class Processor

    attr_reader :text
    def initialize(language: Languages::Common)
      @language = language
    end

    def process(text:)
      @text = List.new(text: text).add_line_break
      replace_abbreviations
      replace_numbers
      replace_continuous_punctuation
      replace_periods_before_numeric_references
      Rule.apply(@text, @language::Abbreviations::WithMultiplePeriodsAndEmailRule)
      Rule.apply(@text, @language::GeoLocationRule)
      Rule.apply(@text, @language::FileFormatRule)
      split_into_segments
    end

    private

    def split_into_segments
      check_for_parens_between_quotes(@text).split("\r")
         .map! { |segment| Rule.apply(segment, @language::SingleNewLineRule, @language::EllipsisRules::All) }
         .map { |segment| check_for_punctuation(segment) }.flatten
         .map! { |segment| Rule.apply(segment, @language::SubSymbolsRules::All) }
         .map { |segment| post_process_segments(segment) }
         .flatten.compact.delete_if(&:empty?)
         .map! { |segment| Rule.apply(segment, @language::SubSingleQuoteRule) }
    end

    def post_process_segments(txt)
      return txt if txt.length < 2 && txt =~ /\A[a-zA-Z]*\Z/
      return if consecutive_underscore?(txt) || txt.length < 2
      Rule.apply(
        txt,
        @language::ReinsertEllipsisRules::All,
        @language::ExtraWhiteSpaceRule
      )

      if txt =~ @language::QUOTATION_AT_END_OF_SENTENCE_REGEX
        split_at_outermost_quotation(txt)
      else
        txt.tr("\n", '').strip
      end
    end

    # A closing quotation mark only ends a sentence when it closes the
    # outermost quotation. This keeps a nested quotation (for example «...»
    # inside "...") from ending the sentence on its own, while still
    # allowing the sentence to end where the outer quotation settles.
    #
    # Nesting is only tracked for texts that use guillemets («...»/‹...›),
    # the style where two levels of quotation commonly co-occur. Other
    # languages keep the original behaviour untouched.
    def split_at_outermost_quotation(txt)
      positions = []
      txt.to_enum(:scan, @language::SPLIT_SPACE_QUOTATION_AT_END_OF_SENTENCE_REGEX).each do
        positions << Regexp.last_match.begin(0)
      end
      positions.select! { |position| outermost_quotation?(txt, position) }
      return txt.tr("\n", '').strip if positions.empty?

      segments = []
      start = 0
      positions.each do |position|
        segments << txt[start...position]
        start = position + 1
      end
      segments << txt[start..-1]
      segments
    end

    # The split regex matches at the space following a closing quotation mark,
    # so the closing mark itself sits immediately before +position+. A mark
    # only ends the sentence when no quotation is still open at that point,
    # which keeps punctuation inside a nested quotation from splitting.
    def outermost_quotation?(txt, position)
      prefix = txt[0...position]
      return true unless nested_quotation_text?(prefix)
      quote_depth(prefix).eql?(0)
    end

    GUILLEMET_OPENING_MARKS = ['«', '‹'].freeze

    # Pairs used to track nesting inside guillemet quotations. Straight double
    # quotes are included so an outer "..." keeps a nested «...» from ending
    # the sentence. Ambiguous marks used by other quotation styles (such as
    # the German ,, ... “ style) are deliberately left out.
    NESTED_QUOTE_PAIRS = {
      '«' => '»',
      '‹' => '›',
      '“' => '”',
      '"' => '"'
    }.freeze

    def nested_quotation_text?(txt)
      txt.each_char.any? { |char| GUILLEMET_OPENING_MARKS.include?(char) }
    end

    # Counts quotations left open after reading +txt+. A mark closes the
    # innermost matching quotation when possible, otherwise it opens one.
    def quote_depth(txt)
      open_marks = []
      txt.each_char do |char|
        if open_marks.any? && NESTED_QUOTE_PAIRS[open_marks.last].eql?(char)
          open_marks.pop
        elsif NESTED_QUOTE_PAIRS.key?(char)
          open_marks << char
        end
      end
      open_marks.length
    end

    def check_for_parens_between_quotes(txt)
      return txt unless txt =~ @language::PARENS_BETWEEN_DOUBLE_QUOTES_REGEX
      txt.gsub!(@language::PARENS_BETWEEN_DOUBLE_QUOTES_REGEX) do |match|
        match.gsub(/\s(?=\()/, "\r").gsub(/(?<=\))\s/, "\r")
      end
    end

    def replace_continuous_punctuation
      @text.gsub!(@language::CONTINUOUS_PUNCTUATION_REGEX) do |match|
        match.gsub(/!/, '&ᓴ&').gsub(/\?/, '&ᓷ&')
      end
    end

    def replace_periods_before_numeric_references
      @text.gsub!(@language::NUMBERED_REFERENCE_REGEX, "∯\\2\r\\7")
    end

    def consecutive_underscore?(txt)
      # Rubular: http://rubular.com/r/fTF2Ff3WBL
      txt.gsub(/_{3,}/, '').length.eql?(0)
    end

    def check_for_punctuation(txt)
      if @language::Punctuations.any? { |p| txt.include?(p) }
        process_text(txt)
      else
        txt
      end
    end

    def process_text(txt)
      txt << 'ȸ' unless @language::Punctuations.any? { |p| txt[-1].include?(p) }
      ExclamationWords.apply_rules(txt)
      between_punctuation(txt)
      txt = Rule.apply(
        txt,
        @language::DoublePunctuationRules::All,
        @language::QuestionMarkInQuotationRule,
        @language::ExclamationPointRules::All
      )
      txt = List.new(text: txt).replace_parens
      sentence_boundary_punctuation(txt)
    end

    def replace_numbers
      Rule.apply @text, @language::Numbers::All
    end

    def abbreviations_replacer
      if defined? @language::AbbreviationReplacer
        @language::AbbreviationReplacer
      else
        AbbreviationReplacer
      end
    end

    def replace_abbreviations
      @text = abbreviations_replacer.new(text: @text, language: @language).replace
    end

    def between_punctuation_processor
      if defined? @language::BetweenPunctuation
        @language::BetweenPunctuation
      else
        BetweenPunctuation
      end
    end

    def between_punctuation(txt)
      between_punctuation_processor.new(text: txt).replace
    end

    def sentence_boundary_punctuation(txt)
      txt = Rule.apply txt, @language::ReplaceColonBetweenNumbersRule if defined? @language::ReplaceColonBetweenNumbersRule
      txt = Rule.apply txt, @language::ReplaceNonSentenceBoundaryCommaRule if defined? @language::ReplaceNonSentenceBoundaryCommaRule

      txt.scan(@language::SENTENCE_BOUNDARY_REGEX)
    end
  end
end
