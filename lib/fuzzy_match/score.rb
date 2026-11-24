require 'fuzzy_match/score/pure_ruby'
require 'fuzzy_match/score/amatch'

class FuzzyMatch
  class Score
    include Comparable

    attr_reader :str1
    attr_reader :str2

    def initialize(str1, str2)
      @str1 = str1.downcase
      @str2 = str2.downcase
    end

    def structured?
      false
    end

    def inspect
      %{(dice=#{"%0.5f" % dices_coefficient_similar},lev=#{"%0.5f" % levenshtein_similar})}
    end

    def <=>(other)
      a = dices_coefficient_similar
      b = other.dices_coefficient_similar
      if a.nan? or b.nan? or (by_dices_coefficient = (a <=> b)) == 0
        levenshtein_similar <=> other.levenshtein_similar
      else
        by_dices_coefficient
      end
    end
  end

  # Score for a positional multi-field projection.  Each position carries its
  # own dice/levenshtein pair; positions that could not be compared are +nil+
  # and are excluded from the aggregate so missing fields are not penalized.
  class StructuredScore
    include Comparable

    attr_reader :field_dices_coefficient_similar
    attr_reader :field_levenshtein_similar
    attr_reader :details

    def initialize(field_dice, field_lev, details = [])
      @field_dices_coefficient_similar = field_dice.freeze
      @field_levenshtein_similar = field_lev.freeze
      @details = details.freeze
    end

    def structured?
      true
    end

    def compared_count
      field_dices_coefficient_similar.compact.length
    end

    def not_compared_count
      field_dices_coefficient_similar.length - compared_count
    end

    # Mean dice across the fields that were actually compared.  Returns 0.0
    # when nothing could be compared.
    def dices_coefficient_similar
      values = field_dices_coefficient_similar.compact
      return 0.0 if values.empty?
      values.inject(0.0, :+) / values.length
    end

    # Mean levenshtein across compared fields.  0.0 when nothing was compared
    # so callers that expect a numeric score never see +nil+ here; the +nil+
    # "not compared" marker lives in the per-field arrays/details.
    def levenshtein_similar
      values = field_levenshtein_similar.compact
      return 0.0 if values.empty?
      values.inject(0.0, :+) / values.length
    end

    def <=>(other)
      by_dice = dices_coefficient_similar <=> other.dices_coefficient_similar
      if by_dice == 0
        levenshtein_similar <=> other.levenshtein_similar
      else
        by_dice
      end
    end

    def inspect
      fields = field_dices_coefficient_similar.map { |v| v.nil? ? 'n/a' : ('%0.3f' % v) }.join(',')
      %{(fields=[#{fields}],dice=#{"%0.5f" % dices_coefficient_similar})}
    end
  end
end
