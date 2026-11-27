# frozen_string_literal: true

require "active_record"
require "set"

module Audited
  # Wrapper around ActiveSupport::CurrentAttributes
  class RequestStore < ActiveSupport::CurrentAttributes
    attribute :audited_store
  end

  class << self
    attr_accessor \
      :auditing_enabled,
      :current_user_method,
      :enum_mapping_version,
      :enum_mappings,
      :ignored_attributes,
      :ignored_default_callbacks,
      :max_audits,
      :store_synthesized_enums
    attr_writer :audit_class

    def audit_class
      # The audit_class is set as String in the initializer. It can not be constantized during initialization and must
      # be constantized at runtime. See https://github.com/collectiveidea/audited/issues/608
      @audit_class = @audit_class.safe_constantize if @audit_class.is_a?(String)
      @audit_class ||= Audited::Audit
    end

    # remove audit_model in next major version it was only shortly present in 5.1.0
    alias_method :audit_model, :audit_class
    deprecate audit_model: "use Audited.audit_class instead of Audited.audit_model. This method will be removed.",
              deprecator: ActiveSupport::Deprecation.new('6.0.0', 'Audited')

    def store
      RequestStore.audited_store ||= {}
    end

    # Returns the identifier of the enum mapping that new audits for
    # +auditable_type+ should record. A model can override the global
    # +enum_mapping_version+ with the +audited enum_mapping_version:+ option.
    def enum_mapping_version_for(auditable)
      version = audited_enum_mapping_version_option(auditable) || enum_mapping_version
      version = version.call if version.respond_to?(:call)
      return nil if version.nil?
      version = version.to_s
      version.empty? ? nil : version
    end

    # Registers the enum codes that were in effect for a business object under
    # +version+. Pass a hash of attribute name to <tt>{code => name}</tt>.
    #
    #   Audited.register_enum_mapping("2023-05", "Case", status: {1 => "ready", 2 => "done"})
    #
    # Later enum migrations can change what a code means, but audits written
    # under +version+ can still be read with these names.
    def register_enum_mapping(version, auditable_type, attributes = nil, **attribute_map)
      attributes = (attributes || {}).merge(attribute_map)
      key = version.to_s
      mapping = EnumMapping.new(
        version: key,
        auditable_type: auditable_type,
        attributes: attributes
      )
      (enum_mappings[key] ||= {})[auditable_type.to_s] = mapping
    end

    # Finds the registered mapping for +version+ and +auditable_type+.
    # +version+ may be <tt>:current</tt> to build a mapping from the model's
    # current enum definitions.
    def enum_mapping_for(version, auditable_type)
      return nil if version.nil?

      key = version.to_s
      return current_enum_mapping(auditable_type) if key == "current"

      mappings = enum_mappings[key]
      return nil unless mappings

      mappings[auditable_type.to_s]
    end
    # Builds a mapping from the enum definitions currently configured on the
    # auditable model. Useful for reading a value that should follow today's
    # business names rather than the names recorded with the audit.
    def current_enum_mapping(auditable_type)
      model = auditable_type.to_s.safe_constantize
      return nil unless model.respond_to?(:defined_enums)

      EnumMapping.from_enums(
        version: "current",
        auditable_type: auditable_type,
        enums: model.defined_enums
      )
    end

    # The enum attributes that were ever recorded for +auditable_type+. Used to
    # decide whether an untranslatable value should be reported as an unknown
    # code instead of being shown as an ordinary value.
    #
    # Pass +version+ to look at one registered version only.
    def enum_attributes_for(auditable_type, version = :any)
      by_version = if version == :any
        enum_mappings.values
      else
        registered = enum_mappings[version.to_s]
        registered ? [registered] : []
      end

      by_version.each_with_object(Set.new) do |mappings, names|
        mapping = mappings[auditable_type.to_s]
        names.merge(mapping.attributes.keys) if mapping
      end
    end

    def reset_enum_mappings!
      @enum_mappings = {}
      @enum_mapping_version = nil
    end

    def config
      yield(self)
    end

    private

    def audited_enum_mapping_version_option(auditable)
      model = auditable.is_a?(Class) ? auditable : auditable.to_s.safe_constantize
      model.audited_options[:enum_mapping_version] if model.respond_to?(:audited_options)
    end
  end

  @ignored_attributes = %w[lock_version created_at updated_at created_on updated_on]
  @ignored_default_callbacks = []

  @current_user_method = :current_user
  @auditing_enabled = true
  @store_synthesized_enums = false
  @enum_mappings = {}
end

require "audited/enum_mapping"
require "audited/enum_mapping_reader"
require "audited/auditor"

ActiveSupport.on_load :active_record do
  require "audited/audit"
  include Audited::Auditor
end

require "audited/sweeper"
require "audited/railtie" if Audited.const_defined?(:Rails)
