module IceCube
  # This abstract validation class is used by the various "fixed-time" (e.g.
  # day, day_of_month, hour_of_day) Validation and ScheduleLock::Validation
  # modules. It is not a standalone rule validation module like the others.
  #
  # Given the including Validation's defined +type+ field, it will lock to the
  # specified +value+ or else the corresponding time unit from the schedule's
  # start_time
  #
  class Validations::FixedValue
    INTERVALS = {min: 60, sec: 60, hour: 24, month: 12, wday: 7}

    def validate(time, start_time)
      case type
      when :day then validate_day_lock(time, start_time)
      when :hour then validate_hour_lock(time, start_time)
      else validate_interval_lock(time, start_time)
      end
    end

    private

    # Validate if the current time unit matches the same unit from the schedule
    # start time, returning the difference to the interval
    #
    def validate_interval_lock(time, start_time)
      t0 = starting_unit(start_time)
      t1 = time.send(type)
      (t0 >= t1) ? t0 - t1 : INTERVALS[type] - t1 + t0
    end

    # Lock the hour if explicitly set by hour_of_day, but allow for the nearest
    # hour during DST start to keep the correct interval.
    #
    def validate_hour_lock(time, start_time)
      h0 = starting_unit(start_time)
      h1 = time.hour
      if h0 >= h1
        h0 - h1
      elsif (dst_offset = TimeUtil.dst_change(time))
        h0 - h1 + dst_offset
      else
        24 - h1 + h0
      end
    end

    # For monthly rules that have no specified day value, the validation relies
    # on the schedule start time and jumps to include every month even if it
    # has fewer days than the schedule's start day.
    #
    # Negative day values (from month end) also include all months.
    #
    # Positive day values are taken literally so months with fewer days will
    # be skipped, unless the owning rule clamps to the month end (e.g. an
    # explicit day 31 in February rolls back to the 28th/29th instead of
    # dropping the whole month).
    #
    def validate_day_lock(time, start_time)
      days_in_month = TimeUtil.days_in_month(time)
      date = Date.new(time.year, time.month, time.day)

      if value && value < 0
        start = TimeUtil.day_of_month(value, date)
        month_overflow = days_in_month - TimeUtil.days_in_next_month(time)
      elsif value && value > 0
        start = value
        month_overflow = 0
        if clamp_to_month_end? && start > days_in_month
          # Roll the requested day back to the last day that exists in this
          # month instead of skipping the month entirely.
          start = days_in_month
        end
      else
        start = TimeUtil.day_of_month(start_time.day, date)
        month_overflow = 0
      end

      sleeps = start - date.day
      return sleeps if sleeps >= 0

      # The target day already passed this month. With month-end clamping the
      # next month may itself be a short month, so jump straight to its clamped
      # day rather than adding a whole (calendar) month, which would overshoot.
      return days_until_clamped_day(date, value) if clamp_to_month_end? && value && value > 0

      if value && value > 0
        days_in_month + sleeps
      else
        until_next_month = (start < 28) ? days_in_month : TimeUtil.days_to_next_month(date)
        until_next_month + sleeps - month_overflow
      end
    end

    # Whether an explicit positive day should roll back to the last available
    # day of a shorter month. Overridden by validations belonging to a rule
    # with month-end clamping enabled; off by default to preserve the literal
    # "skip missing dates" behavior.
    #
    def clamp_to_month_end?
      false
    end

    # Number of days from +date+ (which is past this month's clamped target)
    # to the clamped occurrence in the following month.
    #
    def days_until_clamped_day(date, value)
      next_month = date >> 1
      next_target = Date.new(
        next_month.year, next_month.month,
        [value, TimeUtil.days_in_month(next_month)].min
      )
      (next_target - date).to_i
    end

    def starting_unit(start_time)
      start = value || start_time.send(type)
      start %= INTERVALS[type] if start < 0
      start
    end
  end
end
