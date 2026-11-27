# frozen_string_literal: true

module Audited
  # Wraps a candidate record rebuilt from audit history together with any
  # values an operator had to supply by hand because the attribute was never
  # audited (or had not been introduced when the history was recorded).
  #
  # A Restoration is used in two modes:
  #
  # * preview mode - the candidate is validated against the *current* model
  #   constraints but nothing is written to the database, and
  # * commit mode  - the same validation is repeated inside a transaction and
  #   the record is only saved when every check passes. A successful commit
  #   produces exactly one new audit.
  #
  # The provenance of every value is exposed so callers can tell audited
  # evidence apart from human supplied data:
  #
  #   restoration.audit_sourced_attributes  # => {"title" => "A"}
  #   restoration.provided_attributes       # => {"required_ref" => "R7"}
  #   restoration.attribute_sources         # => {"title" => :audit, ...}
  class Restoration
    # Attributes that describe the audit itself rather than the auditable
    # record and therefore must never be assigned to the rebuilt record.
    # Both string and symbol forms are filtered because reconstructed audit
    # history uses symbol keys for its bookkeeping values.
    META_ATTRIBUTES = %w[audit_version].freeze
    META_ATTRIBUTE_KEYS = (META_ATTRIBUTES + META_ATTRIBUTES.map(&:to_sym)).freeze

    attr_reader :record, :audit, :audit_sourced_attributes, :provided_attributes, :conflicts

    def initialize(record:, audit:, audit_sourced_attributes:, provided_attributes:, conflicts:)
      @record = record
      @audit = audit
      @audit_sourced_attributes = audit_sourced_attributes
      @provided_attributes = provided_attributes
      @conflicts = conflicts
      @committed = false
    end

    # Re-runs the model's own validations, including the conflict checks,
    # without touching the database. Previews and commits share this code path
    # so both enforce the current, real constraints of the model.
    def valid?
      validate
      record.errors.empty?
    end

    def invalid?
      !valid?
    end

    # Validation errors. Before a commit these are produced by the model's
    # current validations and conflict checks; after a commit attempt they also
    # include database level failures (for example a unique index without a
    # matching model validator) captured during the rolled back transaction.
    def errors
      return record.errors if @committed

      validate
      record.errors
    end

    # Maps every candidate attribute to +:audit+ (replayed from history) or
    # +:provided+ (supplied by the operator because history had no value).
    def attribute_sources
      audit_sourced_attributes.each_with_object({}) do |(attr, _), sources|
        sources[attr] = :audit
      end.merge(provided_attributes.transform_values { :provided })
    end

    def audit_sourced?(attribute)
      audit_sourced_attributes.key?(attribute.to_s)
    end

    def provided?(attribute)
      provided_attributes.key?(attribute.to_s)
    end

    # Persists the candidate. Returns the Restoration so callers can inspect
    # +valid?+, +errors+ or the provenance helpers regardless of the outcome.
    #
    # The whole operation runs in a transaction: a failed validation or a
    # database level constraint violation rolls everything back, leaving the
    # record in its previous (for example destroyed) state and writing no new
    # audit. A successful save produces exactly one audit through the normal
    # auditing callbacks.
    def commit
      @committed = true
      return self unless validate

      record.class.transaction do
        begin
          record.save!
        rescue ActiveRecord::RecordInvalid,
               ActiveRecord::RecordNotSaved,
               ActiveRecord::RecordNotUnique,
               ActiveRecord::InvalidForeignKey => e
          record.errors.add(:base, e.message)
          raise ActiveRecord::Rollback
        end
      end

      self
    end

    # Whether +commit+ actually persisted the candidate.
    def committed?
      @committed && record.persisted? && record.errors.empty?
    end

    def to_h
      {
        audit_sourced_attributes: audit_sourced_attributes,
        provided_attributes: provided_attributes,
        attribute_sources: attribute_sources,
        conflicts: conflicts,
        valid: @committed ? committed? : valid?
      }
    end

    private

    def validate
      record.errors.clear
      record.valid?
      conflicts.each do |attr, (historical, supplied)|
        record.errors.add(
          attr.to_sym,
          "conflicts with audited history: historical value #{historical.inspect} " \
          "cannot be silently replaced with #{supplied.inspect}"
        )
      end
      record.errors.empty?
    end
  end
end