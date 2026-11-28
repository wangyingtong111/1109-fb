# Minimal bootstrap for the linked operation tests. It deliberately avoids
# test/minitest_helper.rb, which pulls in the optional ORM gems.
$LOAD_PATH.unshift(File.expand_path(File.join(File.dirname(__FILE__), '..', 'lib')))
require 'aasm'
require 'minitest/autorun'

# A tiny persistence adapter that behaves like a database row without needing
# one. It follows the same shape as the ActiveRecord adapter: state lives in
# an attribute, and persistence is a separate "save" step.
module LinkedRowPersistence
  def aasm_read_state(name = :default)
    column = self.class.aasm(name).attribute_name
    value = linked_attributes[column]
    return value.to_sym unless value.nil? || value.to_s.empty?

    aasm_new_record? ? aasm(name).determine_state_name(self.class.aasm(name).initial_state) : nil
  end

  def aasm_write_state(state, name = :default)
    column = self.class.aasm(name).attribute_name
    linked_attributes[column] = state.to_s
    linked_rows[column] = state.to_s
    true
  end

  def aasm_write_state_without_persistence(state, name = :default)
    linked_attributes[self.class.aasm(name).attribute_name] = state.to_s
  end

  def linked_attributes
    @linked_attributes ||= {}
  end

  # what a fresh read from the "database" would see
  def linked_rows
    @linked_rows ||= {}
  end

  # commit the current in-memory attribute values to the "row" (what an ORM
  # does when a new record is saved)
  def persist_initial_states!
    AASM::StateMachineStore.fetch(self.class, true).machine_names.each do |name|
      column = self.class.aasm(name).attribute_name
      value = linked_attributes[column]
      linked_rows[column] = value unless value.nil?
    end
    self
  end

  def aasm_new_record?
    true
  end
end

# The acceptance scenario: materials archiving and publication are confirmed
# together. Publication refuses while the delivery date is empty, and in that
# case archiving must not be left completed.
class LinkedApplication
  include AASM
  include LinkedRowPersistence

  attr_accessor :delivery_date, :materials_ok, :notifications

  def initialize
    @materials_ok = true
    @delivery_date = nil
    @notifications = []
  end

  def materials_valid?
    materials_ok
  end

  def delivery_date?
    !delivery_date.to_s.strip.empty?
  end

  aasm(:docs, namespace: true, column: :docs_state) do
    state :collecting, initial: true
    state :archived
    event :finish, success: :docs_finished do
      transitions from: :collecting, to: :archived, guard: :materials_valid?
    end
  end

  aasm(:pub, namespace: true, column: :pub_state, whiny_transitions: false) do
    state :queued, initial: true
    state :published
    event :finish, success: :pub_finished do
      transitions from: :queued, to: :published, guard: :delivery_date?
    end
  end

  aasm_linked_event :confirm do
    part :docs, :finish
    part :pub, :finish, reason: 'delivery date is empty'
    success :application_confirmed
  end

  def docs_finished(*)
    notifications << [:docs_finished, linked_result]
  end

  def pub_finished(*)
    notifications << [:pub_finished, linked_result]
  end

  def application_confirmed(result)
    notifications << [:application_confirmed, result]
  end

  # what the success notifications saw
  def linked_result
    aasm_linked_result(:confirm)
  end
end

# Guards that take the business argument handed to the operation.
class LinkedArgsApplication
  include AASM
  include LinkedRowPersistence

  attr_accessor :seen

  def initialize
    @seen = []
  end

  def docs_token?(token)
    seen << [:docs, token]
    token == 'R7'
  end

  def pub_token?(token)
    seen << [:pub, token]
    token == 'R7'
  end

  aasm(:docs, namespace: true) do
    state :collecting, initial: true
    state :archived
    event :finish do
      transitions from: :collecting, to: :archived, guard: :docs_token?
    end
  end

  aasm(:pub, namespace: true, whiny_transitions: false) do
    state :queued, initial: true
    state :published
    event :finish do
      transitions from: :queued, to: :published, guard: :pub_token?
    end
  end

  aasm_linked_event :confirm do
    part :docs, :finish
    part :pub, :finish, reason: :pub_token?
  end
end
