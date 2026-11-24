# frozen_string_literal: true

module Philiprehberger
  module FuzzyMatch
    module Soundex
      CODES = {
        'B' => '1', 'F' => '1', 'P' => '1', 'V' => '1',
        'C' => '2', 'G' => '2', 'J' => '2', 'K' => '2', 'Q' => '2', 'S' => '2', 'X' => '2', 'Z' => '2',
        'D' => '3', 'T' => '3',
        'L' => '4',
        'M' => '5', 'N' => '5',
        'R' => '6'
      }.freeze

      def self.code(string, separators = nil, **options)
        return words(string, separators, **options) if WordEncoder.word_mode?(separators, options)

        if normalization_options?(options)
          string = WordEncoder.encoding_source(string, options)
        end

        return '' if string.nil? || string.empty?

        upper = string.upcase.gsub(/[^A-Z]/, '')
        return '' if upper.empty?

        result = upper[0]
        prev_code = CODES[result]

        upper[1..].each_char do |char|
          char_code = CODES[char]
          next if char_code.nil? || char_code == prev_code

          result << char_code
          prev_code = char_code
          break if result.length == 4
        end

        result.ljust(4, '0')
      end

      def self.normalization_options?(options)
        options.key?(:normalize) || options.key?(:normalization) ||
          options.key?(:normalize_encoding) || options.key?(:phonetic_normalization)
      end

      private_class_method :normalization_options?

      # Per-word Soundex codes with original-text spans.
      def self.words(string, separators = nil, **options)
        WordEncoder.encode(string, separators, **options, algorithm: :soundex)
      end

      # `[start, stop, normalized_word]` triples for real word boundaries.
      def self.spans(string, separators = nil, **options)
        WordEncoder.spans(string, separators, **options)
      end

      class << self
        alias word_codes words
        alias with_spans words
        alias word_spans spans
      end
    end
  end
end
