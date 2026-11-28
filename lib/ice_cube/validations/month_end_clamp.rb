module IceCube
  # Explicit month-end backfill for monthly rules.
  #
  # A monthly rule pinned to a day that some months do not have (e.g. the
  # 31st) normally skips those months. With month-end clamping enabled the
  # missing day is rolled back to the last day of the month, so a rule on the
  # 31st fires on the 28th/29th of February instead of dropping February.
  #
  module Validations::MonthEndClamp
    # Enable (or disable) month-end clamping for this rule. Returns the rule
    # so it can be chained. When enabled, every day-of-month validation on the
    # rule clamps nonexistent positive days to the month's final day.
    #
    def clamp_to_month_end(clamp = true)
      replace_validations_for(:month_end_clamp, clamp ? [Validation.new] : nil)
      @clamp_to_month_end = clamp
      Array(validations[:day_of_month]).each do |validation|
        validation.clamp_to_month_end = clamp
      end
      self
    end

    # Whether month-end clamping is switched on for this rule.
    #
    def clamp_to_month_end?
      @clamp_to_month_end == true
    end

    # Marker validation. It never filters a candidate on its own; it carries
    # the setting through serialization (to_hash/from_hash, YAML, iCalendar)
    # and keeps the rule's clamp state discoverable via +validations+.
    #
    class Validation
      def key
        :month_end_clamp
      end

      def type
        :limit
      end

      def dst_adjust?
        false
      end

      def validate(_step_time, _start_time)
        nil
      end

      def build_s(builder)
        builder.piece(:month_end_clamp) << IceCube::I18n.t("ice_cube.clamp_to_month_end")
      end

      def build_hash(builder)
        builder.validations_array(:month_end_clamp) << true
      end

      def build_ical(builder)
        # iCalendar has no equivalent of month-end clamping; emit nothing so
        # round trips stay RFC 5545 compatible.
      end

      StringBuilder.register_formatter(:month_end_clamp) do |segments|
        segments.first
      end
    end
  end
end
