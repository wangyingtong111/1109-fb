module AASM
  # this is used internally as an argument default value to represent no value
  NO_VALUE = :_aasm_no_value

  # provide a state machine for the including class
  # make sure to load class methods as well
  # initialize persistence for the state machine
  def self.included(base) #:nodoc:
    base.extend AASM::ClassMethods

    # do not overwrite existing state machines, which could have been created by
    # inheritance, see class method inherited
    AASM::StateMachineStore.register(base)

    AASM::Persistence.load_persistence(base)
    super
  end

  module ClassMethods
    # make sure inheritance (aka subclassing) works with AASM
    def inherited(base)
      AASM::StateMachineStore.register(base, self)

      # linked operations declared on a class are inherited (and shared) by
      # its subclasses, just like the state machines above
      if instance_variable_defined?(:@aasm_linked_events)
        inherited_linked = base.aasm_linked_events
        @aasm_linked_events.each_pair { |name, operation| inherited_linked[name] = operation }
      end

      super
    end

    # this is the entry point for all state and event definitions
    def aasm(*args, &block)
      if args[0].is_a?(Symbol) || args[0].is_a?(String)
        # using custom name
        state_machine_name = args[0].to_sym
        options = args[1] || {}
      else
        # using the default state_machine_name
        state_machine_name = :default
        options = args[0] || {}
      end

      AASM::StateMachineStore.fetch(self, true).register(state_machine_name, AASM::StateMachine.new(state_machine_name))

      # use a default despite the DSL configuration default.
      # this is because configuration hasn't been setup for the AASM class but we are accessing a DSL option already for the class.
      aasm_klass = options[:with_klass] || AASM::Base

      raise ArgumentError, "The class #{aasm_klass} must inherit from AASM::Base!" unless aasm_klass.ancestors.include?(AASM::Base)

      @aasm ||= Concurrent::Map.new
      if @aasm[state_machine_name]
        # make sure to use provided options
        options.each do |key, value|
          @aasm[state_machine_name].state_machine.config.send("#{key}=", value)
        end
      else
        # create a new base
        @aasm[state_machine_name] = aasm_klass.new(
          self,
          state_machine_name,
          AASM::StateMachineStore.fetch(self, true).machine(state_machine_name),
          options
        )
      end
      @aasm[state_machine_name].instance_eval(&block) if block # new DSL
      @aasm[state_machine_name]
    end

    # Registry of declarable linked operations (see #aasm_linked_event).
    def aasm_linked_events
      @aasm_linked_events ||= Concurrent::Map.new
    end

    # Declare an operation that fires several events (usually on several state
    # machines of the same object) as one unit.
    #
    #   aasm_linked_event :confirm do
    #     part :docs, :finish
    #     part :pub, :finish, reason: 'delivery date is missing'
    #   end
    #
    # The block is evaluated in the context of an
    # AASM::Core::LinkedOperation. Declaring the same name again replaces the
    # declaration so reloading works.
    #
    # Declaring the operation creates three methods on the class (respecting an
    # optional :method / :as option):
    # * <name>     - fires the operation without persistence, returns the result
    # * <name>!    - fires the operation with persistence, returns the result
    # * may_<name>? - preflight only, returns true when every part is permitted
    def aasm_linked_event(name, options = {}, &block)
      raise ArgumentError, 'a linked event needs a name' if name.nil?

      operation = AASM::Core::LinkedOperation.new(name, self, options)
      operation.instance_eval(&block) if block
      aasm_linked_events[name.to_sym] = operation

      message = method_name_for_linked(name)
      aasm_name = name.to_sym

      # non-persisting firing: returns the full result of the operation
      define_linked_method(message) do |*args|
        aasm_linked_run(aasm_name, false, args)
      end

      # bang firing: commits with persistence
      define_linked_method("#{message}!") do |*args|
        aasm_linked_run(aasm_name, true, args)
      end

      define_linked_method("may_#{message}?") do |*args|
        aasm_linked_may_fire?(aasm_name, *args)
      end

      operation
    end
    alias_method :aasm_linked_action, :aasm_linked_event
    alias_method :aasm_linked_transition, :aasm_linked_event

    def aasm_linked_event?(name)
      aasm_linked_events.key?(name.to_sym)
    end
    alias_method :aasm_linked_event_defined?, :aasm_linked_event?

    def aasm_linked_operation(name_or_symbol)
      aasm_linked_events.fetch(name_or_symbol.to_sym) do
        raise AASM::UnknownLinkedOperation, "There is no linked operation named '#{name_or_symbol}' declared in #{name}!"
      end
    end
    alias_method :aasm_linked_event_for, :aasm_linked_operation

    private

    def method_name_for_linked(name)
      declared = aasm_linked_events[name.to_sym]&.options || {}
      (declared[:method] || declared[:as] || name).to_s
    end

    # Define an instance method on the class that declares the linked
    # operation and warn (like AASM does for other declarations) if the name
    # is already taken by something not generated by AASM.
    def define_linked_method(method_name, &definition)
      aasm_warn_if_overriding_linked(method_name)

      send(:define_method, method_name, &definition)
      ruby2_keywords(method_name) if RUBY_VERSION >= '2.7.1'
      aasm_generated_methods[method_name] = true
      method_name
    end

    def aasm_warn_if_overriding_linked(method_name)
      return if AASM::Configuration.hide_warnings
      return unless method_defined?(method_name)
      return if aasm_generated_methods.include?(method_name)

      machine = AASM::StateMachineStore.fetch(self, true)&.machine(:default)
      machine&.config&.logger&.warn("#{name}: overriding method '#{method_name}'!")
    end

    def aasm_generated_methods
      @aasm_generated_methods ||= Concurrent::Map.new
    end
  end # ClassMethods

  # this is the entry point for all instance-level access to AASM
  def aasm(name=:default)
    unless AASM::StateMachineStore.fetch(self.class, true).machine(name)
      raise AASM::UnknownStateMachineError.new("There is no state machine with the name '#{name}' defined in #{self.class.name}!")
    end
    @aasm ||= Concurrent::Map.new
    @aasm[name.to_sym] ||= AASM::InstanceBase.new(self, name.to_sym)
  end

  # Fire a declared linked operation by name. Business arguments are handed to
  # every part that did not declare its own; a trailing Hash carrying :persist
  # or :args is taken as firing options.
  #
  #   record.aasm_linked_fire(:confirm)                    # preflight + commit
  #   record.aasm_linked_fire(:confirm, persist: true)     # persist each part
  #   record.aasm_linked_fire(:confirm, args: [R7])        # explicit part args
  #
  # Returns an AASM::Core::LinkedResult describing every part.
  def aasm_linked_fire(name, *args)
    options = {}
    if args.last.is_a?(Hash) && args.last.keys.any? { |key| %i[persist args].include?(key) }
      options = args.pop
    end

    persist = options.key?(:persist) ? options[:persist] : nil
    args = Array(options[:args]) if options.key?(:args)

    aasm_linked_run(name, persist, args)
  end

  # Was the last firing of +name+ successful?
  def aasm_linked_fired?(name)
    aasm_linked_results[name.to_sym]&.success? || false
  end

  # The full result of the last firing of +name+ (nil when never fired).
  def aasm_linked_result(name)
    aasm_linked_results[name.to_sym]
  end

  def aasm_linked_results
    @aasm_linked_results ||= {}
  end

  def initialize_dup(other)
    @aasm = Concurrent::Map.new
    super
  end

private

  # remember the result of the current firing so callbacks fired by the
  # operation can inspect the complete outcome
  def aasm_record_linked_result(name, result)
    aasm_linked_results[name.to_sym] = result
  end

  # run a linked operation; +persist+ nil means "use the declared default"
  def aasm_linked_run(name, persist, args)
    operation = self.class.aasm_linked_operation(name)
    options = { args: args }
    options[:persist] = persist unless persist.nil?
    operation.fire(self, options)
  end

  # preflight for the generated may_<linked>? method
  def aasm_linked_may_fire?(name, *args)
    self.class.aasm_linked_operation(name).may_fire?(self, args)
  end

  # Takes args and a from state and removes the first
  # element from args if it is a valid to_state for
  # the event given the from_state
  def process_args(event, from_state, *args)
    # If the first arg doesn't respond to to_sym then
    # it isn't a symbol or string so it can't be a state
    # name anyway
    return args unless args.first.respond_to?(:to_sym)
    if event.transitions_from_state(from_state).map(&:to).flatten.include?(args.first)
      return args[1..-1]
    end
    return args
  end

  def aasm_fire_event(state_machine_name, event_name, options, *args, &block)
    event = self.class.aasm(state_machine_name).state_machine.events[event_name]
    begin
      old_state = aasm(state_machine_name).state_object_for_name(aasm(state_machine_name).current_state)

      fire_default_callbacks(event, *process_args(event, aasm(state_machine_name).current_state, *args))

      # a linked operation already ran the guard during its preflight; reusing
      # that decision keeps preflight and commit on exactly one evaluation
      if options[:may_fire]
        may_fire_to = options[:may_fire]
      else
        may_fire_to = event.may_fire?(self, *args)
      end

      if may_fire_to
        fire_exit_callbacks(old_state, *process_args(event, aasm(state_machine_name).current_state, *args))
        if new_state_name = fire_event_transition(event, may_fire_to, options[:may_fire], args)
          aasm_fired(state_machine_name, event, old_state, new_state_name, options, *args, &block)
        else
          aasm_failed(state_machine_name, event_name, old_state, event.failed_callbacks)
        end
      else
        aasm_failed(state_machine_name, event_name, old_state, event.failed_callbacks)
      end
    rescue StandardError => e
      event.fire_callbacks(:error, self, e, *process_args(event, aasm(state_machine_name).current_state, *args)) ||
      event.fire_global_callbacks(:error_on_all_events, self, e, *process_args(event, aasm(state_machine_name).current_state, *args)) ||
      raise(e)
      false
    ensure
      event.fire_callbacks(:ensure, self, *process_args(event, aasm(state_machine_name).current_state, *args))
      event.fire_global_callbacks(:ensure_on_all_events, self, *process_args(event, aasm(state_machine_name).current_state, *args))
    end
  end

  # Execute the transition of an event. When a preflight decision was handed
  # in (linked operations), reuse it instead of re-running the guards.
  def fire_event_transition(event, may_fire_to, preflight, args)
    if preflight
      event.fire_with_preflight(self, preflight, *args)
    else
      event.fire(self, {:may_fire => may_fire_to}, *args)
    end
  end

  def fire_default_callbacks(event, *processed_args)
    event.fire_global_callbacks(
      :before_all_events,
      self,
      *processed_args
    )

    # new event before callback
    event.fire_callbacks(
      :before,
      self,
      *processed_args
    )
  end

  def fire_exit_callbacks(old_state, *processed_args)
    old_state.fire_callbacks(:before_exit, self, *processed_args)
    old_state.fire_callbacks(:exit, self, *processed_args)
  end

  def aasm_fired(state_machine_name, event, old_state, new_state_name, options, *args)
    persist = options[:persist]
    # a linked operation hands in a list; success-side work is postponed until
    # every part of the operation has committed
    deferred = options[:deferred]

    new_state = aasm(state_machine_name).state_object_for_name(new_state_name)
    callback_args = process_args(event, aasm(state_machine_name).current_state, *args)

    new_state.fire_callbacks(:before_enter, self, *callback_args)

    new_state.fire_callbacks(:enter, self, *callback_args) # TODO: remove for AASM 4?

    persist_successful = true
    if persist
      persist_successful = aasm(state_machine_name).set_current_state_with_persistence(new_state_name)
      if persist_successful
        yield if block_given?
      end
    else
      aasm(state_machine_name).current_state = new_state_name
      yield if block_given?
    end

    binding_event = event.options[:binding_event]
    if binding_event
      __send__("#{binding_event}#{'!' if persist}")
    end

    # Capture the transition selected for this event now, so the postponed
    # notifications do not depend on whatever runs after this part.
    selected_transition = event.valid_transition_for(self)

    success_side_effects = lambda do
      if persist
        event.fire_callbacks(:before_success, self, *callback_args)
        event.fire_remembered_transition_callbacks(self, selected_transition,
                                                   *process_args(event, old_state.name, *args))
        event.fire_callbacks(:success, self, *callback_args)
      end

      old_state.fire_callbacks(:after_exit, self, *callback_args)
      new_state.fire_callbacks(:after_enter, self, *callback_args)
      event.fire_callbacks(
        :after,
        self,
        *process_args(event, old_state.name, *args)
      )
      event.fire_global_callbacks(
        :after_all_events,
        self,
        *process_args(event, old_state.name, *args)
      )

      self.aasm_event_fired(event.name, old_state.name, aasm(state_machine_name).current_state) if self.respond_to?(:aasm_event_fired)
    end

    if persist_successful
      if deferred
        deferred << success_side_effects
      else
        success_side_effects.call
      end
    else
      self.aasm_event_failed(event.name, old_state.name) if self.respond_to?(:aasm_event_failed)
    end

    persist_successful
  end

  def aasm_failed(state_machine_name, event_name, old_state, failures = [])
    if self.respond_to?(:aasm_event_failed)
      self.aasm_event_failed(event_name, old_state.name)
    end

    if AASM::StateMachineStore.fetch(self.class, true).machine(state_machine_name).config.whiny_transitions
      raise AASM::InvalidTransition.new(self, event_name, state_machine_name, failures)
    else
      false
    end
  end

end
