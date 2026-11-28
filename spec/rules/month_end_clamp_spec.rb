require File.dirname(__FILE__) + "/../spec_helper"

module IceCube
  describe MonthlyRule, "month-end clamping" do
    describe "#clamp_to_month_end" do
      it "is off by default and keeps skipping days that a month does not have" do
        rule = Rule.monthly.day_of_month(31)
        expect(rule.clamp_to_month_end?).to eq(false)

        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule rule.count(3)
        expect(schedule.all_occurrences.map(&:day)).to eq([31, 31, 31])
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-03-31", "2024-05-31"])
      end

      it "clamps a missing day to the last day of the month in a leap year" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(3).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-02-29", "2024-03-31"])
      end

      it "clamps February back to the 28th in a non-leap year" do
        schedule = Schedule.new(Time.utc(2023, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(3).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2023-01-31", "2023-02-28", "2023-03-31"])
      end

      it "clamps other overflowing days (e.g. the 30th) in February" do
        rule = Rule.monthly.day_of_month(30).clamp_to_month_end
        start = Time.utc(2024, 1, 30)
        first = rule.next_time(start, start, nil)
        second = rule.next_time(first + 1, start, nil)
        third = rule.next_time(second + 1, start, nil)
        expect([first, second, third].map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-30", "2024-02-29", "2024-03-30"])
      end

      it "leaves months that contain the day untouched" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(6).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-01-31 2024-02-29 2024-03-31 2024-04-30 2024-05-31 2024-06-30])
      end

      it "applies to every day_of_month value on the rule" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(15, 31).count(5).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-01-31 2024-02-15 2024-02-29 2024-03-15 2024-03-31])
      end

      it "does not change negative (from month end) day values" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(-1).count(3).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-02-29", "2024-03-31"])
      end

      it "can be turned back off to restore skipping behavior" do
        rule = Rule.monthly.day_of_month(31).clamp_to_month_end.clamp_to_month_end(false)
        expect(rule.clamp_to_month_end?).to eq(false)

        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule rule.count(3)
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-03-31", "2024-05-31"])
      end

      it "returns the rule so it can be chained" do
        rule = Rule.monthly
        expect(rule.clamp_to_month_end).to equal(rule)
      end

      it "keeps firing every month across a full leap year" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(12).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[
            2024-01-31 2024-02-29 2024-03-31 2024-04-30 2024-05-31 2024-06-30
            2024-07-31 2024-08-31 2024-09-30 2024-10-31 2024-11-30 2024-12-31
          ])
      end

      it "rolls the day back to the month end when walking from the prior month" do
        rule = Rule.monthly.day_of_month(30).clamp_to_month_end
        start = Time.utc(2024, 1, 30)
        first = rule.next_time(start, start, nil)
        second = rule.next_time(first + 1, start, nil)
        third = rule.next_time(second + 1, start, nil)
        expect([first, second, third].map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-30", "2024-02-29", "2024-03-30"])
      end

      it "respects a larger monthly interval and only fills visited months" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly(2).day_of_month(31).count(4).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-01-31 2024-03-31 2024-05-31 2024-07-31])
      end

      it "clamps correctly across a year boundary into a non-leap February" do
        schedule = Schedule.new(Time.utc(2024, 11, 30))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(4).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-11-30 2024-12-31 2025-01-31 2025-02-28])
      end

      it "is visible through occurs_on? and occurs_at?" do
        schedule = Schedule.new(Time.utc(2024, 1, 1))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).clamp_to_month_end
        )
        expect(schedule.occurs_on?(Date.new(2024, 2, 29))).to eq(true)
        expect(schedule.occurs_on?(Date.new(2024, 2, 28))).to eq(false)
        expect(schedule.occurs_on?(Date.new(2025, 2, 28))).to eq(true)
        expect(schedule.occurs_on?(Date.new(2024, 4, 30))).to eq(true)
        expect(schedule.occurs_at?(Time.utc(2024, 2, 29))).to eq(true)
      end

      it "still works when combined with by_set_pos" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).by_set_pos(-1).count(3).clamp_to_month_end
        )
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-02-29", "2024-03-31"])
      end
    end

    describe "with multiple rules in one schedule" do
      it "emits a shared clamped day only once" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).clamp_to_month_end
        )
        schedule.add_recurrence_rule Rule.monthly.day_of_month(29)

        occurrences = schedule.occurrences(Time.utc(2024, 3, 31, 23, 59, 59))
        expect(occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-01-31 2024-02-29 2024-03-29 2024-03-31])
      end

      it "still counts a clamped occurrence that an exception removes" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(3).clamp_to_month_end
        )
        schedule.add_exception_time Time.utc(2024, 2, 29)

        # The excluded Feb 29th still consumes one of the three counts, so the
        # schedule ends after generating (and dropping) it.
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-03-31"])
      end

      it "keeps an unclamped rule skipping missing days alongside a clamped one" do
        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(31).count(3).clamp_to_month_end
        )
        schedule.add_recurrence_rule(
          Rule.monthly.day_of_month(30).count(3)
        )

        # The clamped 31st rule fills Feb 29; the unclamped 30th rule skips
        # February entirely and contributes Mar 30 / Apr 30 only.
        occurrences = schedule.occurrences(Time.utc(2024, 5, 1))
        expect(occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-01-31 2024-02-29 2024-03-30 2024-03-31 2024-04-30])
      end
    end

    describe "serialization" do
      let(:rule) { Rule.monthly.day_of_month(31).clamp_to_month_end }

      it "round trips through a hash and preserves clamping" do
        restored = Rule.from_hash(rule.to_hash)
        expect(restored.clamp_to_month_end?).to eq(true)

        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule restored.count(3)
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-02-29", "2024-03-31"])
      end

      it "round trips through YAML and preserves clamping" do
        restored = Rule.from_yaml(rule.to_yaml)
        expect(restored.clamp_to_month_end?).to eq(true)

        schedule = Schedule.new(Time.utc(2024, 1, 31))
        schedule.add_recurrence_rule restored.count(3)
        expect(schedule.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(["2024-01-31", "2024-02-29", "2024-03-31"])
      end

      it "round trips a whole schedule through a hash and YAML" do
        schedule = Schedule.new(Time.utc(2024, 1, 31)) do |s|
          s.add_recurrence_rule(
            Rule.monthly.day_of_month(31).count(3).clamp_to_month_end
          )
        end

        from_hash = Schedule.from_hash(schedule.to_hash)
        from_yaml = Schedule.from_yaml(schedule.to_yaml)

        [from_hash, from_yaml].each do |restored|
          expect(restored.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
            .to eq(["2024-01-31", "2024-02-29", "2024-03-31"])
          expect(restored).to eq(schedule)
        end
      end

      it "round trips alongside an exception rule" do
        schedule = Schedule.new(Time.utc(2024, 1, 31)) do |s|
          s.add_recurrence_rule(
            Rule.monthly.day_of_month(31).count(4).clamp_to_month_end
          )
          s.add_exception_rule Rule.monthly.day_of_month(15)
        end

        restored = Schedule.from_yaml(schedule.to_yaml)
        expect(restored).to eq(schedule)
        expect(restored.all_occurrences.map { |t| t.strftime("%Y-%m-%d") })
          .to eq(%w[2024-01-31 2024-02-29 2024-03-31 2024-04-30])
      end

      it "serializes the flag inside the rule validations hash" do
        expect(rule.to_hash[:validations]).to include(month_end_clamp: [true])
      end

      it "omits the flag when clamping is disabled" do
        hash = Rule.monthly.day_of_month(31).to_hash
        expect(hash[:validations]).not_to have_key(:month_end_clamp)
      end

      it "treats equal clamped rules as equal" do
        expect(rule).to eq(Rule.monthly.day_of_month(31).clamp_to_month_end)
      end

      it "treats clamped and unclamped rules as different" do
        expect(rule).not_to eq(Rule.monthly.day_of_month(31))
      end

      it "does not emit an iCalendar property for the non-standard setting" do
        expect(rule.to_ical).to eq("FREQ=MONTHLY;BYMONTHDAY=31")
      end

      it "describes the clamping in its string representation" do
        expect(rule.to_s)
          .to eq("Monthly on the 31st day of the month clamping to the last " \
                 "day of the month when the day does not exist")
      end

      it "leaves the string representation untouched when disabled" do
        expect(Rule.monthly.day_of_month(31).to_s)
          .to eq("Monthly on the 31st day of the month")
      end
    end
  end
end
