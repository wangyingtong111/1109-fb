require "spec_helper"

describe "audit history" do
  let!(:user) { Models::ActiveRecord::User.create!(name: "May") }

  # Re-date the audits so the scenario is May -> June, independent of wall clock.
  def rebalance(user, *times)
    user.audits.reload.to_a.zip(times).each do |audit, time|
      audit.update_columns(created_at: time) if time
    end
    user
  end

  describe "when nothing was compacted" do
    it "reports a time before the first audit as not_registered" do
      history = user.history_at(Time.utc(2025, 4, 1))

      expect(history).to be_not_registered
      expect(history.code).to eq("not_registered")
      expect(history.revision).to be_nil
      expect(history.effective_from).to be_nil
      expect(history.folded_versions).to be_empty
      expect(history.history_started_at).to eq(user.audits.first.created_at)
    end

    it "reports a time covered by audits as found, using the usual recovery" do
      user.update!(name: "June")
      rebalance(user, Time.utc(2025, 5, 1), Time.utc(2025, 6, 1))

      history = user.history_at(Time.utc(2025, 6, 2))
      expect(history).to be_found
      expect(history.revision.name).to eq("June")
      expect(history.revision.attributes).to eq(user.revision_at(Time.utc(2025, 6, 2)).attributes)
    end

    it "keeps revision_at unchanged for dates that still have audits" do
      user.update!(name: "June")
      rebalance(user, Time.utc(2025, 5, 1), Time.utc(2025, 6, 1))

      expect(user.revision_at(Time.utc(2025, 6, 2)).name).to eq("June")
    end
  end

  describe "after a compaction" do
    # Created in May, changed once more in June, then the May audit is folded
    # into the June one. The May snapshot itself is gone.
    let!(:record) do
      item = user
      item.update!(name: "June")
      rebalance(item, Time.utc(2025, 5, 1), Time.utc(2025, 6, 1))
      item.combine_audits(item.audits.reload)
      item
    end

    it "keeps only the surviving audit and records the folded window" do
      expect(record.audits.reload.pluck(:version)).to eq([2])
      survivor = record.audits.reload.first
      expect(survivor).to be_compacted_history
      expect(survivor.history_started_at).to eq(Time.utc(2025, 5, 1))
      expect(survivor.folded_versions).to eq([1])
    end

    it "reports a folded date as compacted with a readable start" do
      history = record.history_at(Time.utc(2025, 5, 15))

      expect(history).to be_compacted
      expect(history.code).to eq("compacted")
      expect(history.revision).to be_nil
      expect(history.history_started_at).to eq(Time.utc(2025, 5, 1))
      expect(history.effective_from).to eq(Time.utc(2025, 6, 1))
      expect(history.effective_version).to eq(2)
      expect(history.folded_versions).to eq([1])
    end

    it "reports a date before the start as not_registered, not compacted" do
      history = record.history_at(Time.utc(2025, 4, 1))

      expect(history).to be_not_registered
      expect(history.revision).to be_nil
      expect(history.effective_from).to be_nil
      expect(history.history_started_at).to eq(Time.utc(2025, 5, 1))
    end

    it "tells the two unrecoverable dates apart" do
      compacted = record.history_at(Time.utc(2025, 5, 15))
      missing = record.history_at(Time.utc(2025, 4, 1))

      expect(compacted.status).not_to eq(missing.status)
      expect(compacted.effective_from).not_to be_nil
      expect(missing.effective_from).to be_nil
    end

    it "returns the two different reasons from history_at" do
      expect(record.history_at(Time.utc(2025, 5, 15))).to be_compacted
      expect(record.history_at(Time.utc(2025, 4, 1))).to be_not_registered
    end

    it "still recovers the surviving audit" do
      expect(record.revisions.map(&:name)).to eq(["June"])
      expect(record.revision(2).name).to eq("June")
      expect(record.revision_at(Time.utc(2025, 6, 2)).name).to eq("June")
    end

    it "leaves revision_at returning nil for an unrecoverable date" do
      expect(record.revision_at(Time.utc(2025, 5, 15))).to be_nil
      expect(record.revision_at(Time.utc(2025, 4, 1))).to be_nil
      expect(record.history_at(Time.utc(2025, 5, 15))).to be_compacted
      expect(record.history_at(Time.utc(2025, 4, 1))).to be_not_registered
    end

    it "does not backfill an intermediate snapshot" do
      history = record.history_at(Time.utc(2025, 5, 15))

      expect(history.revision).to be_nil
      expect(record.revisions.size).to eq(record.audits.reload.size)
    end
  end

  describe "when only the first compaction happened" do
    it "keeps reporting the surviving audits in detailed form" do
      user.update!(name: "May again")
      user.update!(name: "June")
      rebalance(user,
        Time.utc(2025, 5, 1), Time.utc(2025, 5, 10), Time.utc(2025, 6, 1))

      user.combine_audits(user.audits.reload.limit(2))

      expect(user.audits.reload.pluck(:version)).to eq([2, 3])
      expect(user.history_at(Time.utc(2025, 5, 20))).to be_found
      expect(user.history_at(Time.utc(2025, 5, 20)).revision.name).to eq("May again")
      expect(user.history_at(Time.utc(2025, 5, 5))).to be_compacted
      expect(user.history_at(Time.utc(2025, 5, 5)).effective_from).to eq(Time.utc(2025, 5, 10))
    end
  end
end
