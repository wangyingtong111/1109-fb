# frozen_string_literal: true

module PragmaticSegmenter
  module Languages
    module Arabic
      include Languages::Common

      Punctuations = ['?', '!', ':', '.', '؟', '،'].freeze
      SENTENCE_BOUNDARY_REGEX = /.*?[:\.!\?؟،]|.*?\z|.*?$/

      # Explicit punctuation mode for written prose.
      #
      # Written Arabic keeps the in-sentence comma (،) inside the running
      # text instead of treating it as a sentence boundary. The Arabic
      # question mark, the exclamation point and the Western period still
      # end a sentence exactly as they do in the default mode.
      #
      # The mode lives in its own module so that activating it never
      # rewrites the default Arabic tables: both modes stay usable in the
      # same process.
      module Written
        include Arabic

        Punctuations = ['?', '!', ':', '.', '؟'].freeze
        SENTENCE_BOUNDARY_REGEX = /.*?[:\.!\?؟]|.*?\z|.*?$/

        # The default mode rewrites commas that are not sentence boundaries
        # so that the scanner can drop them. Written mode never scans for a
        # comma boundary, so the rule is intentionally inert here.
        ReplaceNonSentenceBoundaryCommaRule = Rule.new(/(?!)/, '')
      end

      PUNCTUATION_MODES = { 'written' => Written }.freeze

      def self.with_punctuation_mode(mode)
        PUNCTUATION_MODES.fetch(mode.to_s.strip.downcase) do
          raise ArgumentError, "unsupported Arabic punctuation mode: #{mode.inspect}"
        end
      end

      module Abbreviation
        ABBREVIATIONS = Set.new(['ا', 'ا. د', 'ا.د', 'ا.ش.ا', 'ا.ش.ا', 'إلخ', 'ت.ب', 'ت.ب', 'ج.ب', 'جم', 'ج.ب', 'ج.م.ع', 'ج.م.ع', 'س.ت', 'س.ت', 'سم', 'ص.ب.', 'ص.ب', 'كج.', 'كلم.', 'م', 'م.ب', 'م.ب', 'ه', 'د‪']).freeze
        PREPOSITIVE_ABBREVIATIONS = [].freeze
        NUMBER_ABBREVIATIONS = [].freeze
      end

      # Rubular: http://rubular.com/r/RX5HpdDIyv
      ReplaceColonBetweenNumbersRule = Rule.new(/(?<=\d):(?=\d)/, '♭')

      # Rubular: http://rubular.com/r/kPRgApNHUg
      ReplaceNonSentenceBoundaryCommaRule = Rule.new(/،(?=\s\S+،)/, '♬')

      class AbbreviationReplacer < AbbreviationReplacer
        SENTENCE_STARTERS = [].freeze
        private

        def scan_for_replacements(txt, am, index, character_array)
          txt.gsub!(/(?<=#{am})\./, '∯')
          txt
        end
      end
    end
  end
end
