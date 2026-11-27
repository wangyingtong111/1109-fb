# frozen_string_literal: true

require "spec_helper"

describe "Audited history restoration" do
  let(:item_class) { Models::ActiveRecord::RestorableItem }

  # Builds a destroyed item whose audit history contains the title but never
  # the required_ref, mirroring a record that predates auditing of that column.
  def destroyed_item_with_unaudited_ref(title: "A", ref:)
    item_class.new(title: title, required_ref: ref).tap do |item|
      item.save(validate: false)
      item.destroy
    end
  end

  describe "Audited::Audit#restore_preview" do
    it "replays audited values and accepts a human supplied value for the unaudited field" do
      item = destroyed_item_with_unaudited_ref(ref: "R01")
      destroy_audit = item.audits.destroys.to_a.last

      restoration = destroy_audit.restore_preview(required_ref: "R07")

      expect(restoration).to be_valid
      expect(restoration.record).to be_a_new_record
      expect(restoration.record.title).to eq("A")
      expect(restoration.record.required_ref).to eq("R07")

      expect(restoration.audit_sourced_attributes).to eq("title" => "A")
      expect(restoration.provided_attributes).to eq("required_ref" => "R07")
      expect(restoration.attribute_sources).to eq(
        "title" => :audit,
        "required_ref" => :provided
      )
      expect(restoration).to be_audit_sourced("title")
      expect(restoration).to be_provided("required_ref")
      expect(restoration.conflicts).to be_empty
    end

    it "does not create a record or an audit while previewing" do
      item = destroyed_item_with_unaudited_ref(ref: "R02")
      destroy_audit = item.audits.destroys.to_a.last

      expect {
        destroy_audit.restore_preview(required_ref: "R08")
      }.to not_change(item_class, :count).and not_change(Audited::Audit, :count)

      expect(item_class.find_by(id: item.id)).to be_nil
    end

    it "validates the candidate against the model's real current constraints" do
      item = destroyed_item_with_unaudited_ref(ref: "R03")
      destroy_audit = item.audits.destroys.to_a.last

      restoration = destroy_audit.restore_preview

      expect(restoration).to be_invalid
      expect(restoration.errors.details[:required_ref]).to include(hash_including(error: :blank))
      expect(restoration.record).to be_a_new_record
      expect(restoration.provided_attributes).to be_empty
    end

    it "rejects a supplied value that conflicts with audited evidence" do
      item = destroyed_item_with_unaudited_ref(ref: "R04")
      destroy_audit = item.audits.destroys.to_a.last

      restoration = destroy_audit.restore_preview(title: "B", required_ref: "R09")

      expect(restoration).to be_invalid
      expect(restoration.conflicts).to eq("title" => ["A", "B"])
      expect(restoration.errors[:title].join).to include("conflicts with audited history")
      # The historical value is preserved; the supplied one never lands.
      expect(restoration.record.title).to eq("A")
      # The gap-fill value is still tracked as human supplied.
      expect(restoration.provided_attributes).to eq("required_ref" => "R09")
    end

    it "treats a supplied value equal to the audited value as audited evidence" do
      item = destroyed_item_with_unaudited_ref(ref: "R05")
      destroy_audit = item.audits.destroys.to_a.last

      restoration = destroy_audit.restore_preview(title: "A", required_ref: "R10")

      expect(restoration).to be_valid
      expect(restoration.conflicts).to be_empty
      expect(restoration.attribute_sources["title"]).to eq(:audit)
      expect(restoration).not_to be_provided("title")
    end
  end

  describe "Audited::Audit#restore" do
    it "rejects a restore violating the current uniqueness constraint and leaves state untouched" do
      item = destroyed_item_with_unaudited_ref(ref: "R06")
      destroy_audit = item.audits.destroys.to_a.last
      audits_before = Audited::Audit.count

      # Another live object now owns the unique reference R11.
      item_class.create!(title: "Occupant", required_ref: "R11")

      restoration = destroy_audit.restore(required_ref: "R11")

      expect(restoration).to be_invalid
      expect(restoration).not_to be_committed
      expect(restoration.errors.details[:required_ref]).to include(hash_including(error: :taken))
      expect(item_class.find_by(id: item.id)).to be_nil
      expect(Audited::Audit.count).to eq(audits_before)
    end

    it "restores with an available reference and writes exactly one new audit" do
      item = destroyed_item_with_unaudited_ref(ref: "R12")
      destroy_audit = item.audits.destroys.to_a.last
      historical_audit_count = item.audits.count

      restoration = nil
      expect {
        restoration = destroy_audit.restore(required_ref: "R13")
      }.to change(item_class, :count).by(1).and change(Audited::Audit, :count).by(1)

      expect(restoration).to be_valid
      expect(restoration).to be_committed
      restored = item_class.find_by(title: "A", required_ref: "R13")
      expect(restored).to be_present
      expect(restored.id).not_to eq(item.id)

      new_audits = restored.audits.reload.to_a
      expect(new_audits.size).to eq(1)
      expect(new_audits.first.action).to eq("create")
      # The destroyed record's original history remains untouched.
      expect(Audited::Audit.where(auditable_id: item.id).count).to eq(historical_audit_count)
    end

    it "does not write an audit when the restore is invalid" do
      item = destroyed_item_with_unaudited_ref(ref: "R14")
      destroy_audit = item.audits.destroys.to_a.last

      expect {
        destroy_audit.restore(required_ref: nil)
      }.to not_change(item_class, :count).and not_change(Audited::Audit, :count)
    end

    it "refuses to silently rewrite audited history through a supplied field" do
      item = destroyed_item_with_unaudited_ref(ref: "R15")
      destroy_audit = item.audits.destroys.to_a.last

      restoration = destroy_audit.restore(title: "B", required_ref: "R16")

      expect(restoration).to be_invalid
      expect(restoration).not_to be_committed
      expect(restoration.conflicts).to have_key("title")
      expect(item_class.where(title: "B")).to be_empty
      expect(item_class.find_by(id: item.id)).to be_nil
    end
  end

  describe "with a complete audited history" do
    let(:item_class) { Models::ActiveRecord::FullyAuditedRestorableItem }

    it "restores as before without any operator supplied values" do
      item = item_class.create!(title: "A", required_ref: "R20")
      item.destroy
      destroy_audit = item.audits.destroys.to_a.last

      restoration = destroy_audit.restore_preview

      expect(restoration).to be_valid
      expect(restoration.provided_attributes).to be_empty
      expect(restoration.audit_sourced_attributes).to include(
        "title" => "A",
        "required_ref" => "R20"
      )
      expect(restoration.attribute_sources.values).to all(eq(:audit))

      expect {
        destroy_audit.restore
      }.to change(item_class, :count).by(1).and change(Audited::Audit, :count).by(1)

      restored = item_class.find_by(required_ref: "R20")
      expect(restored.title).to eq("A")
    end
  end
end