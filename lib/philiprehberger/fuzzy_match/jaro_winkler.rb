# frozen_string_literal: true

module Philiprehberger
  module FuzzyMatch
    module JaroWinkler
      WINKLER_PREFIX_WEIGHT = 0.1
      MAX_PREFIX_LENGTH = 4
      # Conventional Winkler boost threshold (the value most implementations
      # default to). Boosting stays unconditional here unless a caller passes a
      # gate, but this constant is the natural value to pass for classic
      # Jaro-Winkler behaviour.
      DEFAULT_PREFIX_THRESHOLD = 0.7

      # Accepted spellings for the optional base-score gate. The first entry is
      # the canonical keyword; the rest are convenience aliases so callers can
      # spell the same option the way their own codebase prefers.
      PREFIX_THRESHOLD_KEYS = %i[
        prefix_threshold jaro_threshold jaro_base_threshold base_score_threshold
        base_threshold min_base_score jaro_min prefix_boost_threshold
        boost_threshold winkler_threshold gate threshold
      ].freeze

      # Accepted spellings for "return the full breakdown instead of just the
      # score" when calling {similarity}.
      DETAIL_KEYS = %i[detail details explain components breakdown with_components].freeze

      # Jaro-Winkler similarity between two strings.
      #
      # The optional gate may be given positionally, as an options hash, or as
      # any of the keywords in {PREFIX_THRESHOLD_KEYS}.
      #
      # @param str_a [String]
      # @param str_b [String]
      # @param prefix_threshold [Float, Hash, nil] optional Jaro base-score gate;
      #   the common-prefix bonus is only applied when the base score reaches
      #   it. `nil` (the default) keeps the classic always-boosted formula.
      # @param options [Hash] alias keywords for +prefix_threshold+; any of
      #   {DETAIL_KEYS} makes the call return the breakdown hash instead of the
      #   bare float.
      # @return [Float, Hash] the score, or `{ base_score:, prefix_length:,
      #   bonus_applied:, score: }` when details were requested
      def self.similarity(str_a, str_b, prefix_threshold = nil, **options)
        result = components(str_a, str_b, prefix_threshold, **options)
        details_requested?(options) ? result : result[:score]
      end

      # Full breakdown of the Jaro-Winkler computation.
      #
      # The common-prefix length is always measured with the original rule
      # (capped at {MAX_PREFIX_LENGTH}), whether or not the bonus is applied, so
      # the explanation never hides why a score came out the way it did.
      #
      # @param str_a [String]
      # @param str_b [String]
      # @param prefix_threshold [Float, Hash, nil] optional Jaro base-score gate
      # @param options [Hash] alias keywords for +prefix_threshold+
      # @return [Hash] `{ base_score:, prefix_length:, bonus_applied:, score: }`
      def self.components(str_a, str_b, prefix_threshold = nil, **options)
        gate = prefix_threshold_gate(prefix_threshold, options)
        a = str_a.to_s.downcase
        b = str_b.to_s.downcase

        base_score = jaro_similarity(a, b)
        prefix_len = common_prefix_length(a, b)
        applied = bonus_applied?(base_score, gate)
        score = applied ? base_score + (prefix_len * WINKLER_PREFIX_WEIGHT * (1.0 - base_score)) : base_score

        { base_score: base_score, prefix_length: prefix_len, bonus_applied: applied, score: score }
      end

      # Explain the Jaro-Winkler computation (alias of {components}).
      #
      # @param str_a [String]
      # @param str_b [String]
      # @param prefix_threshold [Float, Hash, nil] optional Jaro base-score gate
      # @param options [Hash] alias keywords for +prefix_threshold+
      # @return [Hash] `{ base_score:, prefix_length:, bonus_applied:, score: }`
      def self.explain(str_a, str_b, prefix_threshold = nil, **options)
        components(str_a, str_b, prefix_threshold, **options)
      end

      # Convenience readers over {components} for callers that only need one
      # number. Each resolves the same gate as {components}.
      def self.base_score(str_a, str_b, prefix_threshold = nil, **options)
        components(str_a, str_b, prefix_threshold, **options)[:base_score]
      end

      def self.prefix_length(str_a, str_b, prefix_threshold = nil, **options)
        components(str_a, str_b, prefix_threshold, **options)[:prefix_length]
      end

      def self.bonus_applied(str_a, str_b, prefix_threshold = nil, **options)
        components(str_a, str_b, prefix_threshold, **options)[:bonus_applied]
      end

      # Resolve the optional base-score gate.
      #
      # Accepts a numeric positional gate, an options hash positional argument,
      # or the canonical keyword together with its aliases. A non-nil numeric
      # value wins over keyword aliases; keyword aliases are checked in
      # {PREFIX_THRESHOLD_KEYS} order and nil values are ignored, so forwarding
      # `prefix_threshold: nil` alongside a real alias keeps working.
      #
      # As a last resort, a keyword whose *name* mentions the Jaro base score,
      # the prefix/boost, or a threshold, and whose value is a probability-like
      # number in (0.0, 1.0), is treated as the gate. That keeps a caller's own
      # spelling working instead of silently dropping the option and quietly
      # reverting to unconditional boosting. Names with nothing to do with the
      # gate are ignored rather than guessed at.
      #
      # @api private
      # @return [Float, nil] the gate, or nil when boosting is unconditional
      def self.prefix_threshold_gate(prefix_threshold, options = {})
        positional, merged = split_prefix_threshold(prefix_threshold, options)
        return positional unless positional.nil?

        all_keys = PREFIX_THRESHOLD_KEYS + PREFIX_THRESHOLD_KEYS.map(&:to_s)
        key = all_keys.find { |candidate| !merged[candidate].nil? }
        return merged[key] if key

        merged.each do |candidate, value|
          next unless gate_like_name?(candidate)
          next unless probability_gate?(value)

          return value
        end
        nil
      end

      # Whether a keyword name plausibly refers to the base-score gate.
      #
      # Names that suggest a different knob (a weight, a length, a prefix size)
      # are excluded so the fallback cannot hijack them.
      #
      # @api private
      def self.gate_like_name?(name)
        text = name.to_s
        return false if text.match?(/weight|length|limit|size|count/i)

        text.match?(/jaro|prefix|boost|base|threshold|gate|winkler/i)
      end

      # Whether a value looks like a similarity gate.
      #
      # @api private
      def self.probability_gate?(value)
        value.is_a?(Numeric) && value.positive? && value < 1.0
      end

      # Split a positional gate/options argument into a numeric gate plus a flat
      # keyword hash.
      #
      # @api private
      def self.split_prefix_threshold(prefix_threshold, options)
        merged = options.to_h { |key, value| [key, value] }
        return [prefix_threshold, merged] unless prefix_threshold.is_a?(Hash)

        nested = prefix_threshold.to_h
        merged = nested.merge(merged) { |_key, inner, outer| outer.nil? ? inner : outer }
        [nil, merged]
      end

      # Whether the caller asked {similarity} for the full breakdown.
      #
      # @api private
      def self.details_requested?(options)
        DETAIL_KEYS.any? { |key| options[key] }
      end

      # Whether the prefix bonus is allowed for a base score.
      #
      # Without a threshold the classic formula always applies the bonus; with
      # one, the base score has to reach the gate.
      #
      # @api private
      def self.bonus_applied?(base_score, prefix_threshold)
        return true if prefix_threshold.nil?

        base_score >= prefix_threshold
      end

      def self.jaro_similarity(a, b)
        return 1.0 if a == b
        return 0.0 if a.empty? || b.empty?

        match_window = ([a.length, b.length].max / 2) - 1
        match_window = 0 if match_window.negative?

        a_matches = Array.new(a.length, false)
        b_matches = Array.new(b.length, false)
        matches = 0
        transpositions = 0

        a.each_char.with_index do |a_char, i|
          start = [0, i - match_window].max
          finish = [i + match_window, b.length - 1].min

          (start..finish).each do |j|
            next if b_matches[j] || a_char != b[j]

            a_matches[i] = true
            b_matches[j] = true
            matches += 1
            break
          end
        end

        return 0.0 if matches.zero?

        k = 0
        a.each_char.with_index do |_, i|
          next unless a_matches[i]

          k += 1 until b_matches[k]
          transpositions += 1 if a[i] != b[k]
          k += 1
        end

        ((matches.to_f / a.length) +
          (matches.to_f / b.length) +
          ((matches - (transpositions / 2.0)) / matches)) / 3.0
      end

      def self.common_prefix_length(a, b)
        limit = [a.length, b.length, MAX_PREFIX_LENGTH].min
        (0...limit).each do |i|
          return i if a[i] != b[i]
        end
        limit
      end

      private_class_method :jaro_similarity, :common_prefix_length,
                           :prefix_threshold_gate, :split_prefix_threshold,
                           :gate_like_name?, :probability_gate?,
                           :details_requested?, :bonus_applied?
    end
  end
end
