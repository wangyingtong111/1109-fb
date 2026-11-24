module Eps
  module Evaluators
    class LinearRegression
      attr_reader :features, :field_settings
      alias mining_fields field_settings

      def initialize(coefficients:, features:, text_features:, field_settings: nil, mining_fields: nil)
        @coefficients = coefficients.to_h { |k, v| [k.is_a?(Array) ? [k[0].to_s, k[1]] : k.to_s, v] }
        @features = features
        @text_features = text_features || {}
        @field_settings = normalize_field_settings(field_settings || mining_fields || {})
      end

      def predict(x, probabilities: false)
        raise "Probabilities not supported" if probabilities

        intercept = @coefficients["_intercept"] || 0.0
        scores = Array.new(x.size, intercept)

        @features.each do |k, type|
          column = x.columns[k]
          raise "Missing data in #{k}" unless column

          settings = @field_settings[k] || {}

          case type
          when "categorical"
            column.each_with_index do |xv, i|
              value = normalize_categorical(xv, k, settings)
              scores[i] += categorical_coefficient(k, xv, value)
            end
          when "text"
            values = column.each_with_index.map do |xv, i|
              normalize_text(xv, k, settings)
            end
            encoder = TextEncoder.new(**@text_features[k])
            counts = encoder.transform(values)
            coef = {}
            @coefficients.each do |k2, v|
              next unless k2.is_a?(Array) && k2.first == k
              coef[k2.last] = v
            end

            counts.each_with_index do |xc, i|
              xc.each do |word, count|
                scores[i] += coef[word] * count if coef[word]
              end
            end
          else
            coef = @coefficients[k].to_f
            column.each_with_index do |xv, i|
              value = normalize_numeric(xv, k, settings)
              scores[i] += coef * value
            end
          end
        end

        scores
      end

      def coefficients
        @coefficients.to_h { |k, v| [Array(k).join.to_sym, v] }
      end

      def missing_replacement_for(name)
        @field_settings.dig(name.to_s, :missing_replacement)
      end

      private

      def normalize_field_settings(settings)
        settings.to_h do |name, values|
          normalized =
            if values.is_a?(Hash)
              values.to_h { |key, value| [key.to_sym, value] }
            else
              values
            end
          [name.to_s, normalized]
        end
      end

      def categorical_coefficient(name, original, normalized)
        return 0.0 if normalized == :as_is

        @coefficients[[name, normalized]] ||
          @coefficients[[name, normalized.to_s]] ||
          @coefficients[[name, original]] ||
          @coefficients[[name, original.to_s]].to_f
      end

      def normalize_categorical(value, name, settings)
        if missing_value?(value, settings)
          missing_value(name, settings)
        elsif invalid_categorical?(value, name, settings)
          replacement = invalid_value(value, name, settings)
          case replacement
          when :as_is
            value
          when :as_missing
            missing_value(name, settings)
          else
            replacement
          end
        else
          value
        end
      end

      def normalize_text(value, name, settings)
        if missing_value?(value, settings)
          replacement = declared_missing_replacement(settings)
          return replacement.to_s if replacement
          raise "Missing data in #{name}"
        end

        value
      end

      def normalize_numeric(value, name, settings)
        if missing_value?(value, settings)
          replacement = missing_value(name, settings)
          return replacement.is_a?(Numeric) ? replacement : Float(replacement)
        end

        number =
          if value.is_a?(Numeric)
            value.to_f
          else
            Float(value)
          end

        if invalid_numeric?(number, settings)
          replacement = invalid_value(number, name, settings)
          if replacement == :as_missing
            replacement = missing_value(name, settings)
          end

          return number if replacement == :as_is
          return replacement if replacement.is_a?(Numeric)

          return Float(replacement)
        end

        apply_outlier_policy(number, name, settings)
      end

      def missing_value?(value, settings)
        return true if value.nil?

        missing_values = settings[:missing_values]
        missing_values&.any? do |missing|
          value == missing || value.to_s == missing.to_s
        end || false
      end

      def declared_missing_replacement(settings)
        return nil unless settings.key?(:missing_replacement)

        replacement = settings[:missing_replacement]
        replacement.nil? ? nil : replacement
      end

      def missing_value(name, settings)
        replacement = declared_missing_replacement(settings)
        return replacement unless replacement.nil?

        if settings[:missing_treatment] == "asValue"
          raise ArgumentError, "Missing replacement for field #{name}"
        end

        raise "Missing data in #{name}"
      end

      def invalid_categorical?(value, name, settings)
        return true if explicit_invalid_value?(value, settings)

        allowed = settings[:valid_values]
        if allowed.nil? || allowed.empty?
          return false unless settings[:invalid_policy_declared]

          allowed = fallback_categorical_values(name, settings)
        end

        return false if allowed.empty?

        allowed.none? { |allowed_value| value == allowed_value || value.to_s == allowed_value.to_s }
      end

      def fallback_categorical_values(name, settings)
        values = @coefficients.keys.filter_map do |key|
          key.last if key.is_a?(Array) && key.first == name
        end
        values << settings[:missing_replacement]
        values << settings[:invalid_replacement]
        values.compact!
        values.map(&:to_s).uniq
      end

      def invalid_numeric?(value, settings)
        return true if explicit_invalid_value?(value, settings)

        allowed = settings[:valid_values] || []
        intervals = settings[:intervals] || []

        return false if allowed.empty? && intervals.empty?

        valid =
          allowed.include?(value) ||
            allowed.any? { |allowed_value| allowed_value.is_a?(Numeric) && allowed_value.to_f == value } ||
            intervals.any? { |interval| within_interval?(value, interval) }

        !valid
      end

      def explicit_invalid_value?(value, settings)
        invalid_values = settings[:invalid_values] || []
        invalid_values.any? do |invalid|
          value == invalid || value.to_s == invalid.to_s
        end
      end

      def within_interval?(value, interval)
        left = interval[:left]
        right = interval[:right]

        left_ok =
          case interval[:closure]
          when "closedClosed" then left.nil? || value >= left
          when "closedOpen" then left.nil? || value >= left
          when "openClosed" then left.nil? || value > left
          when "openOpen" then left.nil? || value > left
          else true
          end

        right_ok =
          case interval[:closure]
          when "closedClosed" then right.nil? || value <= right
          when "closedOpen" then right.nil? || value < right
          when "openClosed" then right.nil? || value <= right
          when "openOpen" then right.nil? || value < right
          else true
          end

        left_ok && right_ok
      end

      def apply_outlier_policy(value, name, settings)
        low = settings[:low_value]
        high = settings[:high_value]
        return value if low.nil? && high.nil?

        outside_low = !low.nil? && value < low
        outside_high = !high.nil? && value > high
        return value unless outside_low || outside_high

        case settings[:outliers]
        when "asExtremeValues"
          low && outside_low ? low : high
        when "asMissingValues"
          replacement = missing_value(name, settings)
          replacement.is_a?(Numeric) ? replacement : Float(replacement)
        else
          value
        end
      end

      def invalid_value(value, name, settings)
        treatment = settings[:invalid_treatment] || "returnInvalid"
        replacement = settings[:invalid_replacement]

        if treatment == "asValue" || !replacement.nil?
          return replacement unless replacement.nil?
          raise ArgumentError, "Invalid replacement for field #{name}"
        end

        case treatment
        when "asIs"
          :as_is
        when "asMissing"
          :as_missing
        else
          raise ArgumentError, "Invalid value #{value.inspect} for field #{name}"
        end
      end
    end
  end
end
