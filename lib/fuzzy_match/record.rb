class FuzzyMatch
  # Raised in cross-encoding mode when a record or needle cannot be
  # decoded/transcoded to UTF-8 (for example, invalid byte sequences or
  # binary data that doesn't map to any character).
  class UndecodableError < ::EncodingError; end

  # Records are the tokens that are passed around when doing scoring and optimizing.
  class Record #:nodoc: all
    # "Foo's" is one word
    # "North-west" is just one word
    # "Bolivia," is just Bolivia
    WORD_BOUNDARY = %r{\W*(?:\s+|$)}
    EMPTY = [].freeze
    BLANK = ''.freeze
    UTF_8 = ::Encoding::UTF_8

    attr_reader :original
    attr_reader :read
    attr_reader :stop_words
    attr_reader :cross_encoding

    def initialize(original, options = {})
      @original = original
      @read = options[:read]
      @stop_words = options.fetch(:stop_words, EMPTY)
      @cross_encoding = options.fetch(:cross_encoding, false)
    end

    # Transcode a string to UTF-8 so that the same text stored in different
    # encodings (UTF-8, ISO-8859-1/Latin-1, UTF-16LE, etc.) compares by its
    # actual characters. Strings already in UTF-8 are returned untouched.
    #
    # Raises FuzzyMatch::UndecodableError with a clear message when the input
    # cannot be decoded (invalid byte sequence, undefined conversion, etc.)
    def self.transcode_to_utf8(str)
      encoding = str.encoding
      return str if encoding == UTF_8
      unless str.valid_encoding?
        raise UndecodableError, "[fuzzy_match] cross-encoding mode: #{str.inspect} is not valid #{encoding} and cannot be decoded to UTF-8"
      end
      begin
        str.encode UTF_8
      rescue ::EncodingError => err
        raise UndecodableError, "[fuzzy_match] cross-encoding mode: could not transcode #{str.inspect} from #{encoding} to UTF-8 (#{err.class}: #{err.message})"
      end
    end

    def inspect
      "w(#{clean.inspect})"
    end

    def clean
      @clean ||= begin
        memo = whole.dup
        stop_words.each do |stop_word|
          memo.gsub! stop_word, BLANK
        end
        memo.strip.freeze
      end
    end

    def words
      @words ||= clean.downcase.split(WORD_BOUNDARY).freeze
    end

    def similarity(other)
      Similarity.new self, other
    end

    def whole
      @whole ||= begin
        memo = case read
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
        end.to_s
        memo = Record.transcode_to_utf8(memo) if cross_encoding
        memo.strip.freeze
      end
    end
  end
end
