require "spec_helper"

describe Audited::AuditHistory do
  let(:time) { Time.utc(2025, 5, 15) }

  it "requires a known status" do
    expect { described_class.new(status: :oops, requested_at: time) }.to raise_error(ArgumentError)
  end

  it "is immutable" do
    expect(described_class.not_registered(requested_at: time)).to be_frozen
  end

  it "reports found with a revision" do
    history = described_class.found(requested_at: time, revision: :rev)

    expect(history).to be_found
    expect(history.code).to eq("found")
    expect(history.revision).to eq(:rev)
  end

  it "reports compacted with an effective start" do
    history = described_class.compacted(
      requested_at: time,
      history_started_at: Time.utc(2025, 5, 1),
      earliest_readable_at: Time.utc(2025, 6, 1),
      earliest_readable_version: 4,
      folded_versions: [1, 2, 3]
    )

    expect(history).to be_compacted
    expect(history.code).to eq("compacted")
    expect(history.revision).to be_nil
    expect(history.history_started_at).to eq(Time.utc(2025, 5, 1))
    expect(history.effective_from).to eq(Time.utc(2025, 6, 1))
    expect(history.effective_version).to eq(4)
    expect(history.folded_versions).to eq([1, 2, 3])
  end

  it "reports not_registered without a revision or an effective start" do
    history = described_class.not_registered(requested_at: time)

    expect(history).to be_not_registered
    expect(history.code).to eq("not_registered")
    expect(history.revision).to be_nil
    expect(history.effective_from).to be_nil
    expect(history.effective_version).to be_nil
    expect(history.folded_versions).to be_empty
  end

  it "serializes to a hash for a history page" do
    history = described_class.compacted(
      requested_at: time,
      history_started_at: Time.utc(2025, 5, 1),
      earliest_readable_at: Time.utc(2025, 6, 1),
      earliest_readable_version: 2,
      folded_versions: [1]
    )

    expect(history.to_h).to include(
      status: :compacted,
      code: "compacted",
      revision: nil,
      effective_version: 2,
      folded_versions: [1]
    )
  end
end
