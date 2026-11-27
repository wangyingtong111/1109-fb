require "spec_helper"

# The business renamed its statuses after the enum migration: code 1 used to
# mean "ready" and now means "archived".
class HistoricalEnumUser < ::ActiveRecord::Base
  self.table_name = :users
  audited enum_mapping_version: "2023-05"

  if ::ActiveRecord.version >= Gem::Version.new("7.2")
    enum :status, {archived: 1, live: 2, draft: 0}
  else
    enum status: {archived: 1, live: 2, draft: 0}
  end
end

# A model that predates enum mapping versions.
class UnversionedEnumUser < ::ActiveRecord::Base
  self.table_name = :users
  audited

  if ::ActiveRecord.version >= Gem::Version.new("7.2")
    enum :status, {archived: 1, live: 2, draft: 0}
  else
    enum status: {archived: 1, live: 2, draft: 0}
  end
end

describe "historical enum mappings" do
  before do
    Audited.reset_enum_mappings!
    Audited.register_enum_mapping("2023-05", "HistoricalEnumUser",
      status: {0 => "new", 1 => "ready", 2 => "done"})
  end

  after do
    Audited.reset_enum_mappings!
    Audited.enum_mapping_version = nil
  end

  let(:user) { HistoricalEnumUser.create!(name: "Case", status: :archived) }

  describe "writing audits" do
    it "records the mapping version in effect" do
      expect(user.audits.first.enum_mapping_version).to eq("2023-05")
    end

    it "keeps storing the database codes as evidence" do
      user.update!(status: 2)
      expect(user.audits.last.reload.audited_changes["status"]).to eq([1, 2])
    end

    it "does not record a version when none is configured" do
      other = UnversionedEnumUser.create!(name: "Case", status: :archived)
      expect(other.audits.first.enum_mapping_version).to be_nil
    end
  end

  describe "reading with the recorded version" do
    before { user.update!(status: 2) }

    let(:audit) { user.audits.last }

    it "shows the names that were valid when the audit was written" do
      expect(audit.audited_changes_with_enum_names["status"]).to eq(%w[ready done])
    end

    it "translates the old and new side of an update" do
      expect(audit.old_attributes_with_enum_names["status"]).to eq("ready")
      expect(audit.new_attributes_with_enum_names["status"]).to eq("done")
    end

    it "leaves the stored evidence untouched" do
      expect {
        audit.audited_changes_with_enum_names
      }.not_to change { audit.reload.audited_changes }
    end

    it "leaves the model's current value on the current enum" do
      expect(user.reload.status).to eq("live")
      expect(HistoricalEnumUser.statuses[user.status]).to eq(2)
    end
  end

  describe "reading with a selected version" do
    it "can translate with an explicitly chosen historical version" do
      audit = user.audits.first
      expect(audit.audited_changes_with_enum_names(version: "2023-05")["status"]).to eq("ready")
    end

    it "can follow the current enum when asked" do
      audit = user.audits.first
      expect(audit.audited_changes_with_enum_names(version: :current)["status"]).to eq("archived")
    end

    it "accepts an unregistered ad-hoc mapping" do
      audit = user.audits.first
      mapping = {status: {1 => "queued"}}
      expect(audit.audited_changes_with_enum_names(enum_mapping: mapping)["status"]).to eq("queued")
    end

    it "translates a single code" do
      expect(user.audits.first.enum_name(:status, 2)).to eq("done")
    end
  end

  describe "exporting" do
    it "keeps the evidence and the interpretation side by side" do
      user.update!(status: 2)
      export = user.audits.last.enum_export

      expect(export[:enum_mapping_version]).to eq("2023-05")
      expect(export[:audited_changes]["status"]).to eq([1, 2])
      expect(export[:display_audited_changes]["status"]).to eq(%w[ready done])
      expect(export[:unknown_enum_codes]).to eq([])
    end

    it "does not change the audit when exporting repeatedly" do
      user.update!(status: 2)
      audit = user.audits.last

      2.times { audit.enum_export(version: "2023-05") }

      expect(audit.reload.audited_changes["status"]).to eq([1, 2])
    end

    it "lists codes that could not be explained" do
      audit = Audited::Audit.new(
        action: "update",
        auditable_type: "HistoricalEnumUser",
        audited_changes: {"status" => [1, 9]},
        enum_mapping_version: "2023-05"
      )

      expect(audit.enum_export[:unknown_enum_codes]).to eq([{attribute: "status", code: 9}])
    end
  end

  describe ".with_enum_mapping_version" do
    it "selects audits written under a version" do
      user.update!(status: 2)
      versions = user.audits.merge(Audited::Audit.with_enum_mapping_version("2023-05")).pluck(:version)

      expect(versions).to eq([1, 2])
      expect(user.audits.merge(Audited::Audit.with_enum_mapping_version("1999-01"))).to be_empty
    end
  end

  describe "codes without a matching version" do
    it "reports a code the version does not define as unknown" do
      audit = Audited::Audit.new(
        action: "update",
        auditable_type: "HistoricalEnumUser",
        audited_changes: {"status" => [1, 9]},
        enum_mapping_version: "2023-05"
      )

      translated = audit.audited_changes_with_enum_names["status"]

      expect(translated.first).to eq("ready")
      expect(translated.last).to be_a(Audited::UnknownEnumCode)
      expect(translated.last).to be_unknown
      expect(translated.last.raw).to eq(9)
    end

    it "reports every enum code as unknown when the version is not registered" do
      audit = Audited::Audit.new(
        action: "update",
        auditable_type: "HistoricalEnumUser",
        audited_changes: {"status" => [1, 2]},
        enum_mapping_version: "1999-01"
      )

      translated = audit.audited_changes_with_enum_names["status"]

      expect(translated).to all(be_a(Audited::UnknownEnumCode))
      expect(audit).to be_unknown_enum_codes
    end

    it "does not translate attributes that are not enums" do
      audit = Audited::Audit.new(
        action: "update",
        auditable_type: "HistoricalEnumUser",
        audited_changes: {"name" => %w[A B]},
        enum_mapping_version: "2023-05"
      )

      expect(audit.audited_changes_with_enum_names["name"]).to eq(%w[A B])
    end

    it "reports enum values as unknown when the audit has no version" do
      audit = Audited::Audit.new(
        action: "update",
        auditable_type: "HistoricalEnumUser",
        audited_changes: {"status" => [1, 2]}
      )

      translated = audit.audited_changes_with_enum_names["status"]

      expect(translated).to all(be_a(Audited::UnknownEnumCode))
      expect(audit.audited_changes["status"]).to eq([1, 2])
    end

    it "returns stored codes unchanged when no historical mappings are registered" do
      Audited.reset_enum_mappings!
      audit = Audited::Audit.new(
        action: "update",
        auditable_type: "HistoricalEnumUser",
        audited_changes: {"status" => [1, 2]}
      )

      expect(audit.audited_changes_with_enum_names["status"]).to eq([1, 2])
    end
  end

  describe ".register_enum_mapping" do
    it "accepts name => code orientation" do
      Audited.register_enum_mapping("2024-01", "HistoricalEnumUser", status: {"ready" => 1})
      audit = user.audits.first
      expect(audit.audited_changes_with_enum_names(version: "2024-01")["status"]).to eq("ready")
    end

    it "replaces a previous registration for the same version" do
      Audited.register_enum_mapping("2023-05", "HistoricalEnumUser", status: {1 => "queued"})
      expect(user.audits.first.audited_changes_with_enum_names["status"]).to eq("queued")
    end
  end
end

