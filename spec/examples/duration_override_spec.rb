require File.dirname(__FILE__) + "/../spec_helper"

describe IceCube::Schedule, "single-occurrence duration overrides" do
  # A weekly review every Monday and Tuesday at 09:00 UTC, normally 30 minutes
  let(:start_time) { Time.utc(2026, 11, 23, 9, 0, 0) } # a Monday
  let(:schedule) do
    IceCube::Schedule.new(start_time, duration: 1800) do |s|
      s.add_recurrence_rule IceCube::Rule.weekly.day(:monday, :tuesday)
    end
  end
  let(:monday) { Time.utc(2026, 11, 30, 9, 0, 0) }
  let(:tuesday) { Time.utc(2026, 12, 1, 9, 0, 0) }

  it "keeps the default duration for regular occurrences" do
    expect(schedule.occurring_at?(monday)).to be true
    expect(schedule.occurring_at?(monday + 1800 - 1)).to be true
    expect(schedule.occurring_at?(monday + 1800)).to be false
    expect(schedule.occurring_at?(monday + 5400)).to be false # 10:30
  end

  it "keeps the initial duration semantics unchanged" do
    expect(schedule.duration).to eq 1800
    expect(schedule.end_time).to eq start_time + 1800
  end

  it "keeps plain RDATE (recurrence time) semantics unchanged" do
    rdate = Time.utc(2026, 12, 5, 9, 0, 0)
    schedule.add_recurrence_time(rdate)
    expect(schedule.occurring_at?(rdate + 1800 - 1)).to be true
    expect(schedule.occurring_at?(rdate + 5400)).to be false
    expect(schedule.duration_override(rdate)).to be_nil
  end

  context "with a single Tuesday occurrence overridden to two hours" do
    before { schedule.override_duration(tuesday, 2 * 3600) }

    it "extends only that occurrence" do
      expect(schedule.occurring_at?(tuesday + 5400)).to be true # Tue 10:30 still busy
      expect(schedule.occurring_at?(monday + 5400)).to be false # Mon 10:30 free
    end

    it "still respects the default duration inside the overridden occurrence" do
      expect(schedule.occurring_at?(tuesday)).to be true
      expect(schedule.occurring_at?(tuesday + 2 * 3600 - 1)).to be true
      expect(schedule.occurring_at?(tuesday + 2 * 3600)).to be false
    end

    it "reports the overridden end time when enumerating occurrences" do
      occurrence = schedule.occurrences(tuesday + 1).find { |o| o.start_time == tuesday }
      expect(occurrence.end_time).to eq tuesday + 2 * 3600
      expect(occurrence.duration).to eq 2 * 3600
    end

    it "spans the overridden duration in occurring_between?" do
      expect(schedule.occurring_between?(tuesday + 3600, tuesday + 5400)).to be true
      expect(schedule.occurring_between?(monday + 3600, monday + 5400)).to be false
    end

    it "survives a round trip through to_hash/from_hash" do
      loaded = IceCube::Schedule.from_hash(schedule.to_hash)
      expect(loaded.duration_override(tuesday)).to eq 2 * 3600
      expect(loaded.occurring_at?(tuesday + 5400)).to be true
      expect(loaded.occurring_at?(monday + 5400)).to be false
    end

    it "survives a round trip through to_yaml/from_yaml" do
      loaded = IceCube::Schedule.from_yaml(schedule.to_yaml)
      expect(loaded.duration_override(tuesday)).to eq 2 * 3600
      expect(loaded.occurring_at?(tuesday + 5400)).to be true
      expect(loaded.occurring_at?(monday + 5400)).to be false
    end

    it "invalidates the override when the occurrence is cancelled" do
      schedule.add_exception_time(tuesday)
      expect(schedule.duration_override(tuesday)).to be_nil
      expect(schedule.occurring_at?(tuesday)).to be false
      expect(schedule.occurring_at?(tuesday + 5400)).to be false
      # other occurrences are unaffected
      expect(schedule.occurring_at?(monday)).to be true
    end

    it "keeps the override invalidated after cancelling, persisting and reloading" do
      schedule.add_exception_time(tuesday)
      loaded = IceCube::Schedule.from_hash(schedule.to_hash)
      expect(loaded.duration_override(tuesday)).to be_nil
      expect(loaded.occurring_at?(tuesday + 5400)).to be false
    end

    it "can be removed explicitly, restoring the default duration" do
      schedule.remove_duration_override(tuesday)
      expect(schedule.duration_override(tuesday)).to be_nil
      expect(schedule.occurring_at?(tuesday + 1800 - 1)).to be true
      expect(schedule.occurring_at?(tuesday + 5400)).to be false
    end

    it "does not affect schedules without overrides" do
      other = IceCube::Schedule.new(start_time, duration: 1800) do |s|
        s.add_recurrence_rule IceCube::Rule.weekly.day(:monday, :tuesday)
      end
      expect(other).not_to eq schedule
      expect(other.occurring_at?(tuesday + 5400)).to be false
    end
  end

  context "with a zero default duration" do
    let(:schedule) do
      IceCube::Schedule.new(start_time) do |s|
        s.add_recurrence_rule IceCube::Rule.weekly.day(:monday, :tuesday)
      end
    end

    it "still matches zero-length occurrences exactly" do
      expect(schedule.occurring_at?(monday)).to be true
      expect(schedule.occurring_at?(monday + 1)).to be false
    end

    it "honours an override on a single occurrence" do
      schedule.override_duration(tuesday, 2 * 3600)
      expect(schedule.occurring_at?(tuesday + 5400)).to be true
      expect(schedule.occurring_at?(monday + 5400)).to be false
      expect(schedule.occurring_at?(monday)).to be true
    end
  end
end
