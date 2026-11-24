class FuzzyMatch
  # Records are the tokens that are passed around when doing scoring and optimizing.
  class Record #:nodoc: all
    # "Foo's" is one word
    # "North-west" is just one word
    # "Bolivia," is just Bolivia
    WORD_BOUNDARY = %r{\W*(?:\s+|$)}
    EMPTY = [].freeze
    BLANK = ''.freeze

    # Marker used for field positions that exist in the projection but have
    # nothing on one side of the comparison.  The numeric score for such a
    # position is always +nil+; this symbol only makes the "not compared"
    # state explicit in details / explanations.
    NOT_COMPARED = :not_compared
    COMPARED = :compared
    NOT_COMPARED_TEXT = 'not compared'.freeze
    NOT_COMPARED_TEXT_ZH = '未比较'.freeze

    attr_reader :original
    attr_reader :read
    attr_reader :field_readers
    attr_reader :stop_words

    def initialize(original, options = {})
      @original = original
      @read = options[:read]
      @field_readers = options[:fields] || options[:projections]
      @stop_words = options.fetch(:stop_words, EMPTY)
    end

    def inspect
      if structured?
        "fields(#{fields.inspect})"
      else
        "w(#{clean.inspect})"
      end
    end

    # A record is "structured" when its read projection yields several
    # positional fields.  Only fields at the same position are ever compared
    # against each other.
    def structured?
      return @structured if defined?(@structured)
      @structured = field_readers? || raw_read.is_a?(::Array)
    end

    def field_readers?
      !!@field_readers
    end

    # The raw value returned by the +read+ projection (not forced to a String).
    def raw_read
      @raw_read ||= apply_read(read, original)
    end

    # The explicit list of positional field values, before normalization.
    # Missing values stay +nil+ so that "not provided" can be distinguished
    # from a real but non-matching value.
    def field_values
      @field_values ||= if @field_readers
        @field_readers.map { |reader| apply_read(reader, original) }.freeze
      elsif raw_read.is_a?(::Array)
        raw_read.dup.freeze
      else
        [raw_read].freeze
      end
    end

    # The explicit structured projection (raw positional field values).
    alias projection field_values

    # Cleaned field strings, positionally aligned with #field_values.  A +nil+
    # entry means the field was absent/blank and must not be compared.
    def fields
      @fields ||= field_values.map { |value| clean_field(value) }.freeze
    end

    # Per-field word lists, aligned with #fields (nil for uncomparable fields).
    def field_words
      @field_words ||= fields.map do |field|
        field.nil? ? nil : field.downcase.split(WORD_BOUNDARY).freeze
      end.freeze
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
      @whole ||= if structured?
        # Never use the Array#inspect form here: brackets/quotes would pollute
        # the legacy single-string path and create cross-field bigrams.  Joining
        # with spaces keeps field boundaries word-aware without pretending the
        # fields were scored as one string.
        field_values.map { |value| value.to_s.strip }.reject(&:empty?).join(' ').freeze
      else
        raw_read.to_s.strip.freeze
      end
    end

    private

    def clean_field(value)
      return nil if value.nil?
      memo = value.to_s.dup
      stop_words.each do |stop_word|
        memo.gsub! stop_word, BLANK
      end
      memo.strip!
      memo.empty? ? nil : memo.freeze
    end

    def apply_read(reader, record)
      case reader
      when ::NilClass
        record
      when ::Numeric, ::String
        record[reader]
      when ::Proc
        reader.call record
      when ::Symbol
        record.respond_to?(reader) ? record.send(reader) : record[reader]
      when ::Array
        # An array of selectors acts as an explicit multi-field projection.
        reader.map { |part| apply_read(part, record) }
      else
        raise "Expected nil, a proc, a symbol, or an array of field readers, got #{reader.inspect}"
      end
    end
  end
end
