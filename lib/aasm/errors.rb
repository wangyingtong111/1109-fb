module AASM

  class UnknownStateMachineError < RuntimeError; end

  class InvalidTransition < RuntimeError
    attr_reader :object, :event_name, :originating_state, :failures, :state_machine_name

    def initialize(object, event_name, state_machine_name, failures = [])
      @object, @event_name, @originating_state, @failures = object, event_name, object.aasm(state_machine_name).current_state, failures
      @state_machine_name = state_machine_name
      super("Event '#{event_name}' cannot transition from '#{originating_state}'.#{reasoning}")
    end

    def reasoning
      " Failed callback(s): #{failures}." unless failures.empty?
    end
  end

  class UndefinedState < RuntimeError; end
  class UndefinedEvent < UndefinedState; end
  class NoDirectAssignmentError < RuntimeError; end

  # Raised when an instance is asked to perform a linked operation that has not
  # been declared on its class.
  class UnknownLinkedOperation < RuntimeError; end

  # Internal signal used to abort the commit phase of a linked operation so
  # that any surrounding transaction can roll back the parts that already ran.
  class LinkedRollback < RuntimeError; end
end
