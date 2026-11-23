# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require_relative 'cleaner/rules'

module PragmaticSegmenter
  # This is an opinionated class that removes errant newlines,
  # xhtml, inline formatting, etc.
  class Cleaner
    include Rules

    attr_reader :text, :doc_type, :vocabulary
    def initialize(text:, doc_type: nil, language: Languages::Common, vocabulary: nil)
      @text = text.dup
      @doc_type = doc_type
      @language = language
      @vocabulary = vocabulary
    end

    # Clean text of unwanted formatting
    #
    # Example:
    #   >> text = "This is a sentence\ncut off in the middle because pdf."
    #   >> PragmaticSegmenter::Cleaner(text: text).clean
    #   => "This is a sentence cut off in the middle because pdf."
    #
    # Arguments:
    #    text:       (String)  *required
    #    language:   (String)  *optional
    #                (two character ISO 639-1 code e.g. 'en')
    #    doc_type:   (String)  *optional
    #                (e.g. 'pdf')
    #    vocabulary: (Array)   *optional
    #                words used to rejoin end-of-line hyphenated words in
    #                PDF text. A hyphenated vocabulary entry keeps the
    #                hyphen; a plain entry joins the two parts without one.

    def clean
      return unless text
      remove_all_newlines
      replace_double_newlines
      replace_newlines
      replace_escaped_newlines

      Rule.apply(@text, HTML::All)

      replace_punctuation_in_brackets
      Rule.apply(@text, InlineFormattingRule)
      clean_quotations
      clean_table_of_contents
      check_for_no_space_in_between_sentences
      clean_consecutive_characters
    end

    private

    def abbreviations
      @language::Abbreviation::ABBREVIATIONS
    end

    def check_for_no_space_in_between_sentences
      words = @text.split(' ')
      words.each do |word|
        search_for_connected_sentences(word, @text, NO_SPACE_BETWEEN_SENTENCES_REGEX, NoSpaceBetweenSentencesRule)
        search_for_connected_sentences(word, @text, NO_SPACE_BETWEEN_SENTENCES_DIGIT_REGEX, NoSpaceBetweenSentencesDigitRule)
      end
      @text
    end

    def replace_punctuation_in_brackets
      @text.dup.gsub!(/\[(?:[^\]])*\]/) do |match|
        @text.gsub!(/#{Regexp.escape(match)}/, match.dup.gsub!(/\?/, '&ᓷ&')) if match.include?('?')
      end
    end

    def search_for_connected_sentences(word, txt, regex, rule)
      if word =~ regex
        unless URL_EMAIL_KEYWORDS.any? { |web| word =~ /#{web}/ }
          unless abbreviations.any? { |abbr| word =~ /#{abbr}/i }
            new_word = Rule.apply(word.dup, rule)
            txt.gsub!(/#{Regexp.escape(word)}/, new_word)
          end
        end
      end
    end

    def remove_all_newlines
      remove_newline_in_middle_of_sentence
      remove_newline_in_middle_of_word
    end

    def remove_newline_in_middle_of_sentence
      @text.gsub!(/(?:[^\.])*/) do |match|
        match.gsub(NEWLINE_IN_MIDDLE_OF_SENTENCE_REGEX, '')
      end
      @text
    end

    def remove_newline_in_middle_of_word
      Rule.apply @text, NewLineInMiddleOfWordRule
    end

    def replace_escaped_newlines
      Rule.apply @text, EscapedNewLineRule, EscapedCarriageReturnRule,
        TypoEscapedNewLineRule, TypoEscapedCarriageReturnRule
    end

    def replace_double_newlines
      Rule.apply @text, DoubleNewLineWithSpaceRule, DoubleNewLineRule
    end

    def replace_newlines
      if doc_type.eql?('pdf')
        remove_pdf_line_breaks
      else
        Rule.apply @text, NewLineFollowedByPeriodRule,
          ReplaceNewlineWithCarriageReturnRule
      end
    end

    def remove_pdf_line_breaks
      join_hyphenated_line_breaks
      Rule.apply @text, NewLineFollowedByBulletRule,

        PDF::NewLineInMiddleOfSentenceRule,
        PDF::NewLineInMiddleOfSentenceNoSpacesRule
    end

    # Rejoins words split by an end-of-line hyphen when a vocabulary is
    # provided. When both the hyphenated spelling (e.g. 're-sign') and the
    # plain spelling (e.g. 'resign') are in the vocabulary, the hyphen is
    # kept. Without a vocabulary nothing is changed here and the existing
    # PDF line-break rules handle the text as before.
    def join_hyphenated_line_breaks
      return @text unless vocabulary

      index = vocabulary.each_with_object({}) { |word, hsh| hsh[word.downcase] = true }
      @text.gsub!(PDF::HyphenatedLineBreakRegex) do
        head = Regexp.last_match(1)
        tail = Regexp.last_match(2)
        hyphenated = "#{head}-#{tail}".downcase
        joined = "#{head}#{tail}".downcase

        if index[hyphenated]
          "#{head}-#{tail}"
        elsif index[joined]
          head + tail
        else
          Regexp.last_match(0)
        end
      end
      @text
    end

    def clean_quotations
      Rule.apply @text, QuotationsFirstRule, QuotationsSecondRule
    end

    def clean_table_of_contents
      Rule.apply @text, TableOfContentsRule, ConsecutivePeriodsRule,
        ConsecutiveForwardSlashRule
    end

    def clean_consecutive_characters
      Rule.apply @text, ConsecutivePeriodsRule, ConsecutiveForwardSlashRule
    end
  end
end
