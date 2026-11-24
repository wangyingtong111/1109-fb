# frozen_string_literal: true

module Philiprehberger
  module FuzzyMatch
    # Splits text at real word boundaries while retaining offsets in the
    # original Ruby string. Unicode normalization is only used for normalized
    # token text/phonetic encoding; offsets are never recomputed from a
    # normalized copy of the complete input.
    module WordEncoder
      WHITESPACE = /[[:space:]]/

      HYPHENS = [
        '-', # hyphen-minus
        '‐', # hyphen
        '‑', # non-breaking hyphen
        '‒', # figure dash
        '–', # en dash
        '—', # em dash
        '―', # horizontal bar
        '−'  # minus sign
      ].freeze

      SEPARATOR_KEYS = %i[
        separators separator delimiters delimiter split_on split_chars
        word_separators connector connector_punctuation splittable_punctuation
      ].freeze

      HYPHEN_KEYS = %i[
        hyphen hyphens dash dashes split_hyphen split_hyphens
        split_on_hyphen split_on_hyphens hyphen_separators
      ].freeze

      # Hash with convenient method access. Hash access remains the stable API.
      class Token < Hash
        def initialize(attributes)
          super()
          replace(attributes)
        end

        def code = fetch(:code)
        def algorithm = fetch(:algorithm)
        def word = fetch(:word)
        def normalized = fetch(:normalized)
        def original = fetch(:original)
        def start = fetch(:start)
        def stop = fetch(:stop)
        def finish = fetch(:end)
        def range = fetch(:range)
        def span = fetch(:span)
        def offsets = fetch(:offsets)

        alias end finish
        alias text normalized
      end

      # Array of tokens with aggregate accessors for callers that want columns.
      class Result < Array
        attr_reader :algorithm

        def initialize(tokens, algorithm)
          super(tokens)
          @algorithm = algorithm
        end

        def codes
          map(&:code)
        end

        def words
          map(&:word)
        end

        def originals
          map(&:original)
        end

        def spans
          map(&:span)
        end

        def ranges
          spans
        end

        def triples
          map { |token| [token.start, token.stop, token.word] }
        end

        def to_h
          {
            algorithm: algorithm,
            codes: codes,
            words: words,
            spans: spans,
            tokens: map(&:to_h)
          }
        end
      end

      def self.spans(string, separators = nil, **options)
        separators, options = normalize_call(separators, options)
        tokenize(string, separators, options).map do |attributes|
          [attributes[:start], attributes[:stop], attributes[:word]]
        end
      end

      def self.encode(string, separators = nil, algorithm: :metaphone, **options)
        separators, options = normalize_call(separators, options)
        normalization = normalization_form(options)
        encoder = encoder_for(algorithm)

        tokens = tokenize(string, separators, options).map do |attributes|
          phonetic_source = phonetic_source(attributes[:original], normalization, options)
          code = encoder.code(phonetic_source)
          Token.new(attributes.merge(code: code, algorithm: algorithm))
        end

        Result.new(tokens, algorithm)
      end

      def self.encoding_source(original, options)
        form = normalization_form(options)
        phonetic_source(original, form, options)
      end

      def self.word_mode?(separators, options)
        options ||= {}
        return true if separators.is_a?(Hash)
        return true unless separators.nil?
        return true if options[:word_spans] || options[:with_spans] || options[:words] || options[:by_word]
        return true if options[:punctuation] == true
        return true if SEPARATOR_KEYS.any? { |key| !options[key].nil? }
        return true if HYPHEN_KEYS.any? { |key| options[key] == true }

        options.any? do |key, value|
          value && key.to_s.match?(/word|span|offset|location|range/)
        end
      end

      def self.normalize_call(separators, options)
        options = options.dup

        if separators.is_a?(Hash)
          options = separators.merge(options)
          separators = nil
        end

        if separators.nil?
          key = SEPARATOR_KEYS.find { |candidate| options.key?(candidate) && !options[candidate].nil? }
          separators = options[key] if key
        end

        SEPARATOR_KEYS.each { |key| options.delete(key) }
        [separators, options]
      end

      def self.tokenize(string, separators, options)
        text = string.nil? ? +'' : string.to_s
        return [] if text.empty?

        boundary = boundary_predicate(separators, options)
        form = normalization_form(options)
        downcase = downcase?(options)
        tokens = []
        start = nil

        text.each_char.with_index do |char, index|
          if WHITESPACE.match?(char) || boundary.call(char)
            if start
              tokens << build_token(text, start, index, form, downcase)
              start = nil
            end
          elsif start.nil?
            start = index
          end
        end

        if start
          tokens << build_token(text, start, text.length, form, downcase)
        end

        tokens
      end

      def self.build_token(text, start_index, stop_index, form, downcase)
        original = text[start_index...stop_index]
        normalized = normalize_text(original, form)
        normalized = normalized.downcase if downcase
        range = start_index...stop_index

        {
          word: normalized,
          normalized: normalized,
          original: original,
          start: start_index,
          stop: stop_index,
          end: stop_index,
          length: stop_index - start_index,
          range: range,
          span: range,
          offsets: [start_index, stop_index]
        }
      end

      def self.boundary_predicate(separators, options)
        chars = separator_characters(separators)
        values = Array(separators)
        regexes = values.grep(Regexp)
        callables = values.select { |value| value.respond_to?(:call) }
        chars.concat(HYPHENS) if hyphen_requested?(options)
        regexes << /[[:punct:]]/ if options[:punctuation] == true || separators == true

        lambda do |char|
          chars.include?(char) ||
            regexes.any? { |pattern| pattern.match?(char) } ||
            callables.any? { |callable| callable.call(char) }
        end
      end

      def self.separator_characters(separators)
        values = case separators
                 when nil, false, true then []
                 when String then separators.chars
                 when Symbol then preset_characters(separators)
                 when Regexp, Proc then []
                 when Enumerable then separators.flat_map { |value| separator_characters(value) }
                 else [separators.to_s]
                 end
        Array(values)
      end

      def self.preset_characters(name)
        case name.to_sym
        when :hyphen, :hyphens, :dash, :dashes then HYPHENS
        when :ascii_hyphen then ['-']
        else raise Error, "Unknown separator preset: #{name.inspect}"
        end
      end

      def self.hyphen_requested?(options)
        HYPHEN_KEYS.any? { |key| options[key] == true }
      end

      def self.normalization_form(options)
        value = options.fetch(:normalization, options.fetch(:normalize, :nfc))
        return :nfc if value.nil? || value == true
        return false if value == false

        value.to_sym
      end

      def self.downcase?(options)
        return false if options[:case_sensitive] == true

        value = options.fetch(:downcase, options.fetch(:casefold, true))
        value != false
      end

      def self.normalize_text(text, form)
        return text if text.nil? || !form

        text.unicode_normalize(form)
      rescue Encoding::CompatibilityError, ArgumentError
        text
      end

      def self.phonetic_source(original, form, options)
        enabled = options.fetch(:normalize_encoding, options.fetch(:phonetic_normalization, form != false))
        return original unless enabled

        target = options[:encoding_form] || :nfkd
        normalize_text(original, target)
      end

      def self.encoder_for(algorithm)
        case algorithm.to_sym
        when :metaphone then Metaphone
        when :soundex then Soundex
        else raise Error, "Unknown phonetic encoder: #{algorithm.inspect}"
        end
      end

      private_class_method :normalize_call, :tokenize, :build_token, :boundary_predicate,
                           :separator_characters, :preset_characters, :hyphen_requested?,
                           :normalization_form, :downcase?, :normalize_text, :phonetic_source,
                           :encoder_for
    end
  end
end
