class FuzzyMatch
  # Records are the tokens that are passed around when doing scoring and optimizing.
  class Record #:nodoc: all
    # "Foo's" is one word
    # "North-west" is just one word
    # "Bolivia," is just Bolivia
    WORD_BOUNDARY = %r{\W*(?:\s+|$)}
    EMPTY = [].freeze
    BLANK = ''.freeze

    attr_reader :original
    attr_reader :read
    attr_reader :stop_words

    def initialize(original, options = {})
      @original = original
      @read = options[:read]
      @stop_words = options.fetch(:stop_words, EMPTY)
    end

    def self.read_value(original, read)
      case read
      when ::NilClass
        original
      when ::Numeric, ::String
        original[read]
      when ::Proc
        read.call original
      when ::Symbol
        original.respond_to?(read) ? original.send(read) : original[read]
      else
        raise "Expected nil, a proc, or a symbol, got #{read.inspect}"
      end
    end

    def inspect
      "w(#{clean.inspect})"
    end

    def clean
      @clean ||= clean_text(whole)
    end

    def words
      @words ||= words_from(clean)
    end

    # Force every cached projection used by grouping, identity, and scoring.
    # Building the values before assigning them keeps a failed read from
    # replacing a usable projection with partially reset state.
    def project!
      whole
      clean
      words
      self
    end

    # Re-read #original and atomically replace all cached projections.
    def refresh!
      projected_whole = self.class.read_value(original, read).to_s.strip.freeze
      projected_clean = clean_text(projected_whole)
      projected_words = words_from(projected_clean)
      @whole = projected_whole
      @clean = projected_clean
      @words = projected_words
      self
    end

    def similarity(other)
      Similarity.new self, other
    end

    def whole
      @whole ||= self.class.read_value(original, read).to_s.strip.freeze
    end

    private

    def clean_text(text)
      memo = text.dup
      stop_words.each do |stop_word|
        memo.gsub! stop_word, BLANK
      end
      memo.strip.freeze
    end

    def words_from(cleaned_text)
      cleaned_text.downcase.split(WORD_BOUNDARY).freeze
    end
  end
end
