require "spec_helper"

describe Audited::EnumMapping do
  it "normalizes code => name maps" do
    mapping = described_class.new(version: "v1", auditable_type: "Case", attributes: {status: {1 => "ready", 2 => "done"}})

    expect(mapping.codes_for(:status)).to eq(1 => "ready", 2 => "done")
  end

  it "normalizes name => code maps" do
    mapping = described_class.new(version: "v1", auditable_type: "Case", attributes: {status: {"ready" => 1, "done" => 2}})

    expect(mapping.codes_for("status")).to eq(1 => "ready", 2 => "done")
  end

  it "builds from ActiveRecord defined_enums" do
    mapping = described_class.from_enums(version: "current", auditable_type: "Case", enums: {"status" => {"ready" => 1}})

    expect(mapping.name_for(:status, 1)).to eq("ready")
  end

  it "returns nil for attributes the version does not describe" do
    mapping = described_class.new(version: "v1", auditable_type: "Case", attributes: {status: {1 => "ready"}})

    expect(mapping.name_for(:title, 1)).to be_nil
  end

  it "returns an unknown code for a code the version does not define" do
    mapping = described_class.new(version: "v1", auditable_type: "Case", attributes: {status: {1 => "ready"}})

    name = mapping.name_for(:status, 9)

    expect(name).to be_a(Audited::UnknownEnumCode)
    expect(name.code).to eq(9)
    expect(name).to be_unknown
  end

  it "accepts a name that was stored as the value" do
    mapping = described_class.new(version: "v1", auditable_type: "Case", attributes: {status: {1 => "ready"}})

    expect(mapping.name_for(:status, "ready")).to eq("ready")
  end
end
