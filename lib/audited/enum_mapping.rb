# frozen_string_literal: true

module Audited
  # A value that could not be translated with the selected enum mapping
  # version. It behaves like a String so it can be rendered directly, and it
  # identifies itself as an unknown code so callers can flag the audit.
  class UnknownEnumCode < String
    attr_reader :code

    def initialize(code)
      @code = code
      super("unknown code #{code}")
    end

    def unknown?
      true
    end
    alias_method :unknown_enum_code?, :unknown?

    def raw
      @code
    end

    def inspect
      "#<Audited::UnknownEnumCode #{@code.inspect}>"
    end
  end

  # The enum codes that were in effect for one auditable type under one
  # mapping version.
  #
  #   Audited::EnumMapping.new(
  #     version: "2023-05",
  #     auditable_type: "Case",
  #     attributes: {status: {1 => "ready", 2 => "done"}}
  #   )
  #
  # Attribute maps accept either orientation: <tt>{code => name}</tt> or
  # <tt>{name => code}</tt>.
  class EnumMapping
    attr_reader :version, :auditable_type, :attributes

    def initialize(version:, auditable_type:, attributes:)
      @version = version.to_s
      @auditable_type = auditable_type.to_s
      @attributes = normalize_attributes(attributes || {})
    end

    # Builds a mapping from ActiveRecord's <tt>defined_enums</tt>, which is
    # already oriented as <tt>{name => code}</tt>.
    def self.from_enums(version:, auditable_type:, enums:)
      new(version: version, auditable_type: auditable_type, attributes: enums || {})
    end

    def key?(attribute)
      @attributes.key?(attribute.to_s)
    end
    alias_method :attribute?, :key?

    # Returns the normalized <tt>{code => name}</tt> map for +attribute+, or
    # nil when this version does not describe it.
    def codes_for(attribute)
      @attributes[attribute.to_s]
    end

    # Returns the business name for +code+, nil when the version does not
    # describe the attribute, or an UnknownEnumCode when the version describes
    # the attribute but not that code.
    def name_for(attribute, code)
      codes = codes_for(attribute)
      return nil if codes.nil?
      return code if code.nil?

      name = codes[lookup_key(code)]
      return name.dup unless name.nil?

      # Audits recorded with synthesized enums (or written by hand) may
      # already store the business name instead of the database code.
      return code.to_s if codes.value?(code.to_s)

      UnknownEnumCode.new(code)
    end

    # The normalized <tt>{code => name}</tt> view of this version.
    def to_h
      @attributes.each_with_object({}) { |(attribute, codes), all| all[attribute] = codes.dup }
    end

    private

    def normalize_attributes(attributes)
      attributes.each_with_object({}) do |(attribute, mapping), all|
        all[attribute.to_s] = normalize_mapping(mapping)
      end
    end

    def normalize_mapping(mapping)
      mapping.each_with_object({}) do |(key, value), codes|
        code, name = code_and_name(key, value)
        codes[lookup_key(code)] = name.to_s
      end
    end

    def code_and_name(key, value)
      if numeric?(value) && !numeric?(key)
        [value, key]
      elsif numeric?(key) && !numeric?(value)
        [key, value]
      else
        [key, value]
      end
    end

    def lookup_key(code)
      numeric?(code) ? code.to_i : code.to_s
    end

    def numeric?(value)
      return true if value.is_a?(Integer)
      return false unless value.is_a?(String)
      !value.empty? && value.match?(/\A-?\d+\z/)
    end
  end
end
