class FuzzyMatch
  # Tolerant per-field detail hash.  Behaves exactly like a Hash in equality
  # and JSON, but also answers to dot access and string keys so callers can
  # read e.g. +detail[:status]+, <tt>detail["status"]</tt>, or
  # +detail.status+.
  class FieldDetail < ::Hash
    def initialize(values = {})
      super()
      values.each { |key, value| self[key.to_sym] = value }
      self.default_proc = proc do |hash, key|
        key.is_a?(::String) ? hash[key.to_sym] : nil
      end
    end

    def compared?
      self[:compared] == true
    end

    def not_compared?
      self[:compared] == false
    end

    def method_missing(name, *args, &block)
      key = name.to_s.sub(/[?=]\z/, '').to_sym
      if key?(key)
        value = self[key]
        name.to_s.end_with?('?') ? !!value : value
      else
        super
      end
    end

    def respond_to_missing?(name, include_private = false)
      key = name.to_s.sub(/\?\z/, '').to_sym
      key?(key) || super
    end
  end

  class Similarity
    attr_reader :record1
    attr_reader :record2
    # Optional stable position of #record2 in the candidate list, used only to
    # break exact score/weight ties deterministically (mirrors the historical
    # stable sort followed by reverse).
    attr_accessor :tie_index

    def initialize(record1, record2)
      @record1 = record1
      @record2 = record2
    end

    def <=>(other)
      by_score = best_score <=> other.best_score
      if by_score == 0
        by_weight = original_weight <=> other.original_weight
        if by_weight == 0 && !tie_index.nil? && !other.tie_index.nil?
          tie_index <=> other.tie_index
        else
          by_weight
        end
      else
        by_score
      end
    end

    def structured?
      record1.structured? && record2.structured?
    end

    def best_score
      @best_score ||= if structured?
        build_structured_score
      else
        FuzzyMatch.score_class.new(record1.clean, record2.clean)
      end
    end

    # Per-field comparison details (empty for ordinary single-string records).
    def field_details
      if best_score.respond_to?(:details)
        best_score.details
      else
        []
      end
    end

    def satisfy?(needle, threshold)
      best_score.dices_coefficient_similar > (threshold || 0) or
        ((record2.clean.length < 3 or needle.clean.length < 3) and best_score.levenshtein_similar > 0) or
        (threshold.nil? && (needle.words & record2.words).any?)
    end

    def inspect
      %{#{record2.clean.inspect} ~ #{record1.clean.inspect} => #{best_score.inspect}}
    end

    # Weight things towards short original strings
    def original_weight
      @original_weight ||= (1.0 / (record1.clean.length * record2.clean.length))
    end

    private

    # Compare field i of the needle only against field i of the candidate.
    # Cross-position comparisons never happen, so two swapped fields cannot
    # count as two hits.  A position missing on either side is "not compared".
    def build_structured_score
      positions = [record1.fields.length, record2.fields.length].max
      dice = []
      lev = []
      details = []
      positions.times do |i|
        a = record1.fields[i]
        b = record2.fields[i]
        if a.nil? || b.nil?
          dice << nil
          lev << nil
          details << build_detail(i, a, b, nil, nil)
        else
          score = FuzzyMatch.score_class.new(a, b)
          d = score.dices_coefficient_similar
          l = score.levenshtein_similar
          dice << d
          lev << l
          details << build_detail(i, a, b, d, l)
        end
      end
      StructuredScore.new(dice, lev, details)
    end

    def build_detail(index, needle_field, straw_field, dice, lev)
      compared = !dice.nil?
      status = compared ? Record::COMPARED : Record::NOT_COMPARED
      FieldDetail.new(
        :index => index,
        :position => index,
        :field => index,
        :status => status,
        :state => status,
        :compared => compared,
        :comparable => compared,
        :score => dice,
        :dice => dice,
        :dices_coefficient_similar => dice,
        :levenshtein => lev,
        :levenshtein_similar => lev,
        :needle => needle_field,
        :query => needle_field,
        :candidate => straw_field,
        :straw => straw_field,
        :marker => (compared ? nil : Record::NOT_COMPARED),
        :not_compared => (compared ? nil : Record::NOT_COMPARED),
        :label => (compared ? nil : Record::NOT_COMPARED_TEXT),
        :text => (compared ? nil : Record::NOT_COMPARED_TEXT),
        :english => (compared ? nil : Record::NOT_COMPARED_TEXT),
        :zh => (compared ? nil : Record::NOT_COMPARED_TEXT_ZH),
        :chinese => (compared ? nil : Record::NOT_COMPARED_TEXT_ZH)
      )
    end
  end
end
