# frozen_string_literal: true

module Audited
  # The result of an Audit#undo operation.
  #
  # When undoing an "update" audit, only the attributes whose current value
  # still matches the value set by that audit are reverted. Attributes that
  # have been changed by later edits are left untouched and reported as
  # conflicts.
  #
  #   result = audit.undo
  #   result.conflict?  # => true if any attribute could not be reverted
  #   result.conflicts  # => {"title" => {expected: "B", current: "C"}}
  #   result.applied    # => {"status" => "open"} attributes actually reverted
  class UndoResult
    # The action of the audit that was undone ("create", "update" or "destroy")
    attr_reader :action

    # The auditable record the undo was applied to (nil if it no longer exists)
    attr_reader :record

    # Hash of attributes that were actually reverted, as {attribute => restored_value}
    attr_reader :applied

    # Hash of attributes that could not be reverted because a later edit
    # changed them, as {attribute => {expected:, current:}}
    attr_reader :conflicts

    def initialize(action:, record: nil, applied: {}, conflicts: {})
      @action = action
      @record = record
      @applied = applied
      @conflicts = conflicts
    end

    # True if any attribute was left untouched due to a conflicting later edit
    def conflict?
      conflicts.any?
    end

    # True if any attribute was actually reverted
    def applied?
      applied.any?
    end
  end
end
