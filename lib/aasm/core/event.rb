# frozen_string_literal: true

module AASM::Core
  class Event
    include AASM::DslHelper

    attr_reader :name, :state_machine, :options, :default_display_name

    def initialize(name, state_machine, options = {}, &block)
      @name = name
      @state_machine = state_machine
      @transitions = []
      @valid_transitions = Hash.new { |h, k| h[k] = {} }
      @guards = Array(options[:guard] || options[:guards] || options[:if])
      @unless = Array(options[:unless]) #TODO: This could use a better name
      @default_display_name = name.to_s.gsub(/_/, ' ').capitalize

      # from aasm4
      @options = options # QUESTION: .dup ?
      add_options_from_dsl(@options, [
        :after,
        :after_commit,
        :after_transaction,
        :before,
        :before_transaction,
        :ensure,
        :error,
        :before_success,
        :success,
      ], &block) if block
    end

    # called internally by Ruby 1.9 after clone()
    def initialize_copy(orig)
      super
      @transitions = @transitions.collect { |transition| transition.clone }
      @guards      = @guards.dup
      @unless      = @unless.dup
      @options     = {}
      orig.options.each_pair { |name, setting| @options[name] = setting.is_a?(Hash) || setting.is_a?(Array) ? setting.dup : setting }
    end

    # a neutered version of fire - it doesn't actually fire the event, it just
    # executes the transition guards to determine if a transition is even
    # an option given current conditions.
    def may_fire?(obj, to_state=::AASM::NO_VALUE, *args)
      _fire(obj, {:test_only => true}, to_state, *args) # true indicates test firing
    end

    def fire(obj, options={}, to_state=::AASM::NO_VALUE, *args)
      _fire(obj, options, to_state, *args) # false indicates this is not a test (fire!)
    end

    # Fire the event on the strength of a transition that was already selected
    # during a preflight pass. The guards are not re-evaluated, which keeps
    # preflight and commit on exactly one decision and avoids guards with side
    # effects running twice.
    def fire_with_preflight(obj, transition, *args)
      clear_failed_callbacks
      return nil if transition.nil?
      return nil unless transition.from == obj.aasm(state_machine.name).current_state || transition.from.nil?

      result = Array(transition.to).first
      Array(transition.to).each { |to| @valid_transitions[obj.object_id][to] = transition }
      transition.execute(obj, *args)
      result
    end

    def transitions_from_state?(state)
      transitions_from_state(state).any?
    end

    def transitions_from_state(state)
      @transitions.select { |t| t.from.nil? or t.from == state }
    end

    def transitions_to_state?(state)
      transitions_to_state(state).any?
    end

    def transitions_to_state(state)
      @transitions.select { |t| t.to == state }
    end

    def fire_global_callbacks(callback_name, record, *args)
      invoke_callbacks(state_machine.global_callbacks[callback_name], record, args)
    end

    def fire_callbacks(callback_name, record, *args)
      # strip out the first element in args if it's a valid to_state
      # #given where we're coming from, this condition implies args not empty
      invoke_callbacks(@options[callback_name], record, args)
    end

    # The transition that was selected for +obj+ during the last fire.
    def valid_transition_for(obj)
      from_state = obj.aasm(state_machine.name).current_state
      @valid_transitions[obj.object_id][from_state]
    end

    def fire_transition_callbacks(obj, *args)
      transition = valid_transition_for(obj)
      transition.invoke_success_callbacks(obj, *args) if transition
      @valid_transitions.delete(obj.object_id)
    end

    # Invoque the success callbacks of an explicitly remembered transition.
    # Linked operations use this when the per-event notifications are
    # postponed until every part has committed, so later parts cannot hide the
    # transition of an earlier one.
    def fire_remembered_transition_callbacks(obj, transition, *args)
      @valid_transitions.delete(obj.object_id)
      transition.invoke_success_callbacks(obj, *args) if transition
    end

    # Forget the transition selected for +obj+. Used by linked operations when
    # a part is rolled back, so no stale transition is left behind.
    def clear_valid_transitions(obj)
      @valid_transitions.delete(obj.object_id)
    end

    def ==(event)
      if event.is_a? Symbol
        name == event
      else
        name == event.name
      end
    end

    ## DSL interface
    def transitions(definitions=nil, &block)
      if definitions # define new transitions
        # Create a separate transition for each from-state to the given state
        Array(definitions[:from]).each do |s|
          @transitions << AASM::Core::Transition.new(self, attach_event_guards(definitions.merge(:from => s.to_sym)), &block)
        end
        # Create a transition if :to is specified without :from (transitions from ANY state)
        if !definitions[:from] && definitions[:to]
          @transitions << AASM::Core::Transition.new(self, attach_event_guards(definitions), &block)
        end
      end
      @transitions
    end

    def failed_callbacks
      transitions.flat_map(&:failures)
    end

    def to_s
      name.to_s
    end

  private

    def attach_event_guards(definitions)
      unless @guards.empty?
        given_guards = Array(definitions.delete(:guard) || definitions.delete(:guards) || definitions.delete(:if))
        definitions[:guards] = @guards + given_guards # from aasm4
      end
      unless @unless.empty?
        given_unless = Array(definitions.delete(:unless))
        definitions[:unless] = given_unless + @unless
      end
      definitions
    end

    def _fire(obj, options={}, to_state=::AASM::NO_VALUE, *args)
      result = options[:test_only] ? false : nil
      clear_failed_callbacks
      transitions = @transitions.select { |t| t.from == obj.aasm(state_machine.name).current_state || t.from == nil}
      return result if transitions.size == 0

      if to_state == ::AASM::NO_VALUE
        to_state = nil
      elsif !(to_state.respond_to?(:to_sym) && transitions.map(&:to).flatten.include?(to_state.to_sym))
        # to_state is an argument
        args.unshift(to_state)
        to_state = nil
      end
      
      # nop, to_state is a valid to-state

      transitions.each do |transition|
        next if to_state and !Array(transition.to).include?(to_state)
        if (options.key?(:may_fire) && transition.eql?(options[:may_fire])) ||
           (!options.key?(:may_fire) && transition.allowed?(obj, *args))

          if options[:test_only]
            result = transition
          else
            result = to_state || Array(transition.to).first
            Array(transition.to).each {|to| @valid_transitions[obj.object_id][to] = transition }
            transition.execute(obj, *args)
          end

          break
        end
      end
      result
    end

    def clear_failed_callbacks
      # https://github.com/aasm/aasm/issues/383, https://github.com/aasm/aasm/issues/599
      transitions.each { |transition| transition.failures.clear }
    end

    def invoke_callbacks(code, record, args)
      Invoker.new(code, record, args)
             .with_default_return_value(false)
             .invoke
    end
  end
end # AASM
