# frozen_string_literal: true

module Audited
  # Read-side helpers that explain stored enum codes using the mapping version
  # an audit was written under.
  #
  # The methods here never change +audited_changes+. The stored codes remain
  # the audit evidence; only the returned hash is translated.
  module EnumMappingReader
    extend ActiveSupport::Concern

    included do
      before_create :set_enum_mapping_version
    end

    module ClassMethods
      # True when the audits table stores the enum mapping version an audit was
      # written under. Older installations get the column through the upgrade
      # generator; until then the version is kept in memory only.
      def enum_mapping_attribute?
        columns_hash.key?("enum_mapping_version")
      rescue ::ActiveRecord::ActiveRecordError
        # No table yet (e.g. before the install migration ran).
        false
      end

      # Audits recorded under +version+. Returns an empty relation when the
      # installation has not added the enum mapping version column yet.
      def with_enum_mapping_version(version)
        return none unless enum_mapping_attribute?

        where(enum_mapping_version: version.to_s)
      end
    end

    # The enum mapping version this audit was written under, if one was
    # recorded. Returns nil for audits written before mappings were tracked.
    # When the audits table has no version column the value is kept in memory
    # on the instance, so a freshly created audit can still explain itself.
    def enum_mapping_version
      self.class.enum_mapping_attribute? ? self[:enum_mapping_version] : @enum_mapping_version
    end

    def enum_mapping_version=(version)
      version = version.nil? ? nil : version.to_s
      version = nil if version.empty?
      if self.class.enum_mapping_attribute?
        self[:enum_mapping_version] = version
      else
        @enum_mapping_version = version
      end
    end

    # Returns the audited changes with enum codes translated to the business
    # names that were valid for the selected mapping version.
    #
    #   audit.audited_changes_with_enum_names # uses the recorded version
    #   audit.audited_changes_with_enum_names(version: "2023-05")
    #   audit.audited_changes_with_enum_names(enum_mapping_version: "2023-05")
    #   audit.audited_changes_with_enum_names(version: :current)
    #
    # A code the selected version does not define comes back as an
    # Audited::UnknownEnumCode so it is reported as an unknown code instead of
    # being translated with today's enum definition, which may mean something
    # else now.
    def audited_changes_with_enum_names(version = nil, **options)
      options[:version] ||= version
      selected = options[:enum_mapping_version] || options[:version] || self.enum_mapping_version
      mapping = coerce_enum_mapping(options[:enum_mapping]) || Audited.enum_mapping_for(selected, auditable_type)
      resolve_enum_changes(audited_changes || {}, mapping, selected)
    end

    # The changed attributes with their new values, translated like
    # #audited_changes_with_enum_names.
    def new_attributes_with_enum_names(version = nil, **options)
      changes = audited_changes_with_enum_names(version, **options)
      changes.each_with_object({}.with_indifferent_access) do |(attr, values), attrs|
        attrs[attr] = (action == "update") ? values.last : values
      end
    end

    # The changed attributes with their old values, translated like
    # #audited_changes_with_enum_names.
    def old_attributes_with_enum_names(version = nil, **options)
      changes = audited_changes_with_enum_names(version, **options)
      changes.each_with_object({}.with_indifferent_access) do |(attr, values), attrs|
        attrs[attr] = (action == "update") ? values.first : values
      end
    end

    # A re-exportable view of the audit that keeps the evidence and its
    # interpretation side by side. The record itself is never written to, so
    # exporting the same audit again returns the same stored codes.
    #
    #   audit.enum_export
    #   # => {enum_mapping_version: "2023-05",
    #   #     audited_changes: {"status" => [1, 2]},
    #   #     display_audited_changes: {"status" => ["ready", "done"]},
    #   #     unknown_enum_codes: []}
    def enum_export(version = nil, **options)
      options[:version] ||= version
      translated = audited_changes_with_enum_names(**options)
      {
        enum_mapping_version: (options[:enum_mapping_version] || options[:version] || self.enum_mapping_version),
        audited_changes: deep_dup_changes(audited_changes || {}),
        display_audited_changes: translated,
        unknown_enum_codes: collect_unknown_enum_codes(translated)
      }
    end

    # One enum code translated with the selected mapping version.
    def enum_name(attribute, code, version = nil, **options)
      options[:version] ||= version
      selected = options[:enum_mapping_version] || options[:version] || self.enum_mapping_version
      mapping = coerce_enum_mapping(options[:enum_mapping]) || Audited.enum_mapping_for(selected, auditable_type)
      resolve_enum_code(attribute, code, mapping, selected)
    end

    # The Audited::EnumMapping that applies to this audit, if the selected
    # version has been registered.
    def enum_mapping(version = enum_mapping_version)
      Audited.enum_mapping_for(version, auditable_type)
    end

    # True when at least one enum value cannot be explained by the selected
    # version.
    def unknown_enum_codes?(version = nil, **options)
      options[:version] ||= version
      audited_changes_with_enum_names(**options).any? do |attribute, values|
        next false unless enum_attribute?(attribute)

        Array(values).any? { |value| value.is_a?(UnknownEnumCode) }
      end
    end
    alias_method :unknown_enum_codes, :unknown_enum_codes?

    private

    # Records which enum mapping version the changes belong to. Only audits
    # created while a version is configured carry one; older audits stay
    # versionless and are read with an explicitly selected version.
    def set_enum_mapping_version
      return if auditable_type.blank?

      version = self.class.enum_mapping_attribute? ? self[:enum_mapping_version].presence : @enum_mapping_version
      version ||= ::Audited.enum_mapping_version_for(auditable_type)
      return if version.nil? || version.to_s.empty?

      self.enum_mapping_version = version
    end

    def resolve_enum_changes(changes, mapping, selected_version)
      changes.each_with_object({}.with_indifferent_access) do |(attribute, values), resolved|
        resolved[attribute] = \
          if values.is_a?(Array)
            values.map { |value| resolve_enum_code(attribute, value, mapping, selected_version) }
          else
            resolve_enum_code(attribute, values, mapping, selected_version)
          end
      end
    end

    def resolve_enum_code(attribute, code, mapping, selected_version)
      return code if code.nil?
      return code unless enum_attribute?(attribute)

      if mapping.nil? || !mapping.key?(attribute)
        # Nothing explains this value. When the application uses historical
        # mappings for this model (or an explicit version was selected), the
        # code cannot be translated faithfully, so report it as unknown instead
        # of relabelling it with today's enum, which may mean something else
        # now. Without any historical mappings the feature is not in use, so
        # the stored evidence is returned unchanged.
        return code unless selected_version.present? || historical_mappings?

        return UnknownEnumCode.new(code)
      end

      mapping.name_for(attribute, code)
    end

    # True when the attribute is an enum for this model, either today or under
    # a registered historical mapping.
    def enum_attribute?(attribute)
      return true if Audited.enum_attributes_for(auditable_type).include?(attribute.to_s)
      return false unless auditable_class.respond_to?(:defined_enums)

      auditable_class.defined_enums.key?(attribute.to_s)
    end

    # True when any mapping version was registered for this auditable type.
    def historical_mappings?
      !Audited.enum_attributes_for(auditable_type).empty?
    end

    def auditable_class
      auditable_type.to_s.safe_constantize
    rescue NameError
      nil
    end

    def deep_dup_changes(changes)
      changes.each_with_object({}) do |(attribute, values), copy|
        copy[attribute] = values.is_a?(Array) ? values.map { |value| duplicate_value(value) } : duplicate_value(values)
      end
    end

    def duplicate_value(value)
      value.duplicable? ? value.dup : value
    end

    def collect_unknown_enum_codes(translated)
      translated.each_with_object([]) do |(attribute, values), unknown|
        next unless enum_attribute?(attribute)

        Array(values).each do |value|
          unknown << {attribute: attribute.to_s, code: value.raw} if value.is_a?(UnknownEnumCode)
        end
      end
    end

    # Accepts either an Audited::EnumMapping or a plain attribute hash so
    # callers can pass an ad-hoc mapping without registering it first.
    def coerce_enum_mapping(enum_mapping)
      return nil if enum_mapping.nil?
      return enum_mapping if enum_mapping.is_a?(EnumMapping)

      EnumMapping.new(
        version: enum_mapping_version,
        auditable_type: auditable_type,
        attributes: enum_mapping
      )
    end
  end
end
