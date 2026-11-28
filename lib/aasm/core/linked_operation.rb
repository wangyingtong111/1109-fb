# frozen_string_literal: true

module AASM
  module Core
    # A declarable operation that fires several AASM events (usually on
    # different state machines of the same object) as a single unit.
    #
    #   class Application
    #     include AASM
    #
    #     aasm(:docs) do
    #       state :collecting, initial: true
    #       state :archived
    #       event :finish do
    #         transitions from: :collecting, to: :archived, guard: :materials_valid?
    #       end
    #     end
    #
    #     aasm(:pub) do
    #       state :queued, initial: true
    #       state :published
    #       event :finish do
    #         transitions from: :queued, to: :published, guard: :delivery_date?
    #       end
    #     end
    #
    #     aasm_linked_event :confirm do
    #       part :docs, :finish
    #       part :pub,  :finish, reason: 'delivery date is missing'
    #       success :application_confirmed
    #     end
    #   end
    #
    # Firing a linked operation evaluates the guards of every part first
    # (preflight). Only when every part is permitted are the transitions
    # committed - inside one outer transaction when the object supports
    # transactions. A rejected part therefore leaves every state and every
    # persisted state column untouched.
    #
    # The result of a firing is an AASM::Core::LinkedResult which describes
    # every part (permitted, committed, rejected) and the rejection reason.
    class LinkedOperation
      # Callbacks that can be declared on an operation. +ensure+ is spelled
      # +always+ here because +ensure+ is a Ruby keyword and cannot be used as
      # a bare method name in the DSL block.
      CALLBACKS = %i[before success failure after error always].freeze

      # maps the DSL spelling to the internal callback name
      CALLBACK_ALIASES = { always: :ensure }.freeze

      attr_reader :name, :parts, :options, :callbacks

      def initialize(name, klass, options = {})
        raise ArgumentError, 'a linked operation needs a name' if name.nil?

        @name = name.to_sym
        @klass = klass
        @options = options || {}
        @parts = []
        @callbacks = Hash.new { |hash, key| hash[key] = [] }
      end

      # called internally by Ruby after clone()
      def initialize_copy(orig)
        super
        @options = orig.options.dup
        @parts = orig.parts.map(&:dup)
        @callbacks = Hash.new { |hash, key| hash[key] = [] }
        orig.callbacks.each_pair { |key, value| @callbacks[key] = value.dup }
      end

      # Declare one part of the linked operation.
      #
      #   part :pub, :finish                         # state machine, event
      #   part :pub, :finish, reason: 'no delivery'  # declared rejection reason
      #   part event: :finish, on: :pub              # keyword spelling
      #
      # Accepted options:
      # * +reason+  - String, Symbol or callable (evaluated with the instance)
      #               returned as the rejection reason for this part.
      # * +args+    - explicit positional arguments for this part's event.
      def part(state_machine_name = nil, event_name = nil, options = {})
        if state_machine_name.is_a?(Hash)
          options = state_machine_name
          state_machine_name = nil
        elsif event_name.is_a?(Hash)
          options = event_name
          event_name = nil
        end
        options = (options || {}).dup
        state_machine_name ||= options.delete(:state_machine) || options.delete(:machine) || options.delete(:on)
        event_name ||= options.delete(:event) || options.delete(:action)
        raise ArgumentError, 'a linked operation part needs a state machine' if state_machine_name.nil?

        event_name ||= state_machine_name
        if parts.any? { |existing| existing.state_machine_name == state_machine_name.to_sym }
          raise ArgumentError, "linked operation :#{name} already has a part for state machine :#{state_machine_name}; " \
                               'parts are preflighted from the same starting state, so they must target ' \
                               'different state machines'
        end
        parts << LinkedPart.new(state_machine_name, event_name, options)
        parts.last
      end
      alias_method :step, :part

      # Add linked-operation level callbacks. They are regular AASM callbacks:
      # literals (methods), procs (instance_exec) and classes are supported.
      #
      #   success :application_confirmed
      #   failure ->(result) { ... }
      #   always  ->(result) { ... }   # like an :ensure callback
      CALLBACKS.each do |callback|
        define_method(callback) do |*subjects, &block|
          subjects = subjects.dup
          subjects << block if block
          callbacks[CALLBACK_ALIASES.fetch(callback, callback)].concat(subjects)
          self
        end
      end

      # Preflight only: did every part's guard permit the transition?
      def may_fire?(instance, args = [])
        parts.all? { |part| preflight(instance, part, args).permitted? }
      end

      # Fire the operation. Returns an AASM::Core::LinkedResult.
      #
      # Options:
      # * +persist+ - commit with persistence (default: the value declared on
      #   the operation, or false).
      # * +args+    - business arguments handed to every part that did not
      #   declare its own +args+.
      def fire(instance, options = {})
        persist = options.key?(:persist) ? options[:persist] : @options[:persist]
        persist = false if persist.nil?
        args = Array(options[:args])

        result = nil
        begin
          decisions = preflight_all(instance, args)
          result = LinkedResult.new(self, decisions, persistence: persist)
          record_result(instance, result)

          # A part refused during the preflight: nothing has been committed, so
          # there is nothing to undo and no success notification to send. The
          # :before callback does not run either - it marks the start of the
          # commit phase, which never starts here.
          unless decisions.all?(&:permitted?)
            run_callbacks(instance, :failure, result)
            return result
          end

          deferred = []
          begin
            run_callbacks(instance, :before, result)

            atomically(instance, persist) do
              decisions.each { |decision| commit(instance, decision, persist, deferred) }

              # Only once every part reached its target do the per-event success
              # notifications run - inside the same atomic scope, so a failure
              # here still rolls the operation back. Each notification can
              # inspect the complete result via aasm_linked_result(name).
              deferred.each(&:call)
              deferred.clear
            end
          rescue AASM::LinkedRollback => error
            discard(instance, decisions, deferred)
            roll_back(instance, decisions, persist)
            result.fail_with(error)
            run_callbacks(instance, :failure, result)
            return result
          rescue StandardError => error
            discard(instance, decisions, deferred)
            roll_back(instance, decisions, persist)
            result.fail_with(error)
            handled = run_callbacks(instance, :error, result, error)
            raise unless handled
            return result
          end

          run_callbacks(instance, :success, result)
          result
        ensure
          if result
            run_callbacks(instance, :after, result)
            run_callbacks(instance, :ensure, result)
          end
        end
      end

      def record_result(instance, result)
        return unless instance.respond_to?(:aasm_record_linked_result, true)

        instance.send(:aasm_record_linked_result, name, result)
      end

      private

      def preflight_all(instance, args)
        parts.map do |part|
          begin
            preflight(instance, part, args)
          rescue StandardError => error
            LinkedDecision.new(part, status: :rejected, error: error,
                               reason: "#{error.class}: #{error.message}")
          end
        end
      end

      def reject_unknown_event(part, args)
        LinkedDecision.new(
          part, status: :rejected, args: args,
          reason: "unknown event :#{part.event_name} on state machine :#{part.state_machine_name}"
        )
      end

      def preflight(instance, part, args)
        effective_args = part.arguments(args)
        state_machine = instance.class.aasm(part.state_machine_name).state_machine
        event = state_machine.events[part.event_name]

        return reject_unknown_event(part, effective_args) if event.nil?

        view_model = instance.aasm(part.state_machine_name)
        from_state = view_model.current_state
        transition = event.may_fire?(instance, *effective_args)

        if transition
          LinkedDecision.new(
            part, status: :permitted, args: effective_args,
            from: from_state, to: Array(transition.to).first, transition: transition
          )
        else
          LinkedDecision.new(
            part, status: :rejected, args: effective_args, from: from_state,
            failures: event.failed_callbacks,
            reason: part.rejection_reason(instance,
                                          event.failed_callbacks,
                                          from_state)
          )
        end
      end

      def commit(instance, decision, persist, deferred)
        part = decision.part
        instance.aasm(part.state_machine_name).current_event = persist ? :"#{part.event_name}!" : part.event_name

        success = instance.send(
          :aasm_fire_event,
          part.state_machine_name,
          part.event_name,
          { persist: persist, may_fire: decision.transition, deferred: deferred },
          *decision.args
        )

        unless success
          raise_abort(decision, part, 'persistence failed')
        end

        unless decision.reached?(instance)
          raise_abort(decision, part, "did not reach :#{decision.to}")
        end

        decision.commit!(instance.aasm(part.state_machine_name).current_state)
      end

      def raise_abort(decision, part, message)
        text = "#{part}: #{message}"
        decision.fail_with(text)
        raise AASM::LinkedRollback, text
      end

      # Drop success-side work recorded for the parts that already ran, and
      # forget the transitions selected during the preflight so a later
      # attempt starts from a clean slate.
      def discard(instance, decisions, deferred)
        deferred.clear
        decisions.each do |decision|
          event = @klass.aasm(decision.state_machine).state_machine.events[decision.event]
          event.clear_valid_transitions(instance) if event.respond_to?(:clear_valid_transitions)
        end
      end

      # Undo what the attempted commit did. When the object committed inside a
      # transaction the rows are already rolled back; for adapters without
      # transactions (or when firing without persistence) the old value is
      # written back as well, so both state columns keep their original values.
      def roll_back(instance, decisions, persist)
        compensated = persist && !transactional?(instance)

        decisions.each do |decision|
          next unless decision.from

          if compensated && decision.committed?
            begin
              instance.send(:aasm_write_state, decision.from, decision.state_machine)
            rescue StandardError
              # the adapter may refuse the restore; the in-memory rollback
              # below still keeps the object consistent.
            end
          end

          begin
            instance.send(:aasm_write_state_without_persistence, decision.from, decision.state_machine)
          rescue StandardError
            # a persistence layer that cannot write without persisting cannot
            # be rolled back in memory; the transaction already undid the row.
          end

          decision.rollback! if decision.committed?
        end
      end

      def transactional?(instance)
        return false unless instance.respond_to?(:aasm_supports_transactions?, true)

        instance.send(:aasm_supports_transactions?)
      end

      def atomically(instance, persist, &block)
        if persist && instance.respond_to?(:aasm_atomic, true)
          instance.send(:aasm_atomic, &block)
        else
          yield
        end
      end

      def run_callbacks(instance, callback_name, *args)
        code = callbacks[callback_name]
        return nil if code.nil? || code.empty?

        AASM::Core::Invoker.new(code.dup, instance, args)
                          .with_default_return_value(nil)
                          .invoke
      end
    end

    # One declared part of a linked operation.
    class LinkedPart
      attr_reader :state_machine_name, :event_name, :options

      def initialize(state_machine_name, event_name, options = {})
        @state_machine_name = state_machine_name.to_sym
        @event_name = event_name.to_sym
        @options = options || {}
      end

      def initialize_copy(orig)
        super
        @options = orig.options.dup
      end

      def key
        state_machine_name
      end

      def arguments(default_args)
        options.key?(:args) ? Array(options[:args]) : default_args
      end

      # Reason shown for this part when it rejects the linked operation.
      #
      # The declared :reason may be
      # * a String - used verbatim,
      # * a Symbol - the name of a method to call on the instance (which may
      #   return nil to fall back to the failed guard),
      # * anything responding to #call - invoked with the instance.
      def rejection_reason(instance, failures, from_state)
        declared = declared_reason(instance)
        return declared unless declared.nil? || declared.to_s.strip.empty?

        failure_text = failures.map(&:to_s).reject(&:empty?).join(', ')
        if failure_text.empty?
          "no transition for :#{event_name} from :#{from_state}"
        else
          failure_text
        end
      end

      private

      def declared_reason(instance)
        declared = options[:reason]
        case declared
        when nil then nil
        when Symbol then instance.respond_to?(declared, true) ? instance.send(declared) : declared.to_s
        else declared.respond_to?(:call) ? declared.call(instance) : declared
        end
      end

      public

      def to_s
        "#{state_machine_name}.#{event_name}"
      end
      alias_method :label, :to_s
    end

    # Per-part outcome of a linked operation.
    class LinkedDecision
      attr_reader :part, :status, :from, :to, :transition, :failures, :reason, :args, :error

      def initialize(part, status:, from: nil, to: nil, transition: nil, failures: [],
                     reason: nil, args: [], error: nil)
        @part = part
        @status = status
        @from = from
        @to = to
        @transition = transition
        @failures = failures || []
        @reason = reason
        @args = args || []
        @error = error
      end

      def state_machine
        part.state_machine_name
      end

      def event
        part.event_name
      end

      def key
        part.key
      end

      def permitted?
        %i[permitted committed].include?(status)
      end

      def committed?
        status == :committed
      end

      def rejected?
        status == :rejected
      end

      def failed?
        status == :failed
      end

      def reached?(instance)
        to.nil? || instance.aasm(state_machine).current_state == to
      end

      def commit!(state)
        @status = :committed
        @to = state
        self
      end

      # The part had committed but the operation as a whole rolled back, so
      # what the caller observes is the original state again.
      def rollback!
        @status = :rollback
        @to = @from
        self
      end

      def rolled_back?
        status == :rollback
      end

      def fail_with(message, error = nil)
        @status = :failed
        @reason = message
        @error = error if error
        self
      end

      def to_h
        {
          part: part.to_s,
          state_machine: state_machine,
          event: event,
          status: status,
          from: from,
          to: to,
          reason: reason,
          failures: failures
        }
      end
    end

    # The complete outcome of firing (or preflighting) a linked operation.
    #
    #   result.success?          # all parts committed
    #   result.rejected?         # at least one part refused
    #   result.reason            # e.g. "pub.finish: delivery date is missing"
    #   result.states            # { docs: :archived, pub: :published }
    #   result[:pub].reason      # rejection reason of a single part
    class LinkedResult
      attr_reader :operation_name, :decisions, :persisted, :error

      def initialize(operation, decisions, persistence: false)
        @operation_name = operation.is_a?(AASM::Core::LinkedOperation) ? operation.name : operation.to_sym
        @decisions = decisions
        @persisted = persistence ? true : false
        @error = decisions.map(&:error).compact.first
      end

      def success?
        error.nil? && !decisions.empty? && decisions.all?(&:committed?)
      end

      def failure?
        !success?
      end
      alias_method :rejected?, :failure?

      def committed?
        success?
      end

      # Was the last firing attempted with persistence?
      def persisted?
        @persisted
      end

      # Did the firing complete successfully and persist every part?
      def saved?
        success? && persisted?
      end

      def parts
        decisions
      end

      def [](key)
        case key
        when Array
          decisions.find { |d| d.state_machine == key[0].to_sym && d.event == key[1].to_sym }
        when Symbol, String
          decisions.find { |d| d.state_machine == key.to_sym } ||
            decisions.find { |d| d.part.to_s == key.to_s }
        else
          nil
        end
      end

      # The state of every committed part, as one consistent snapshot.
      def states
        decisions.select(&:committed?).each_with_object({}) do |decision, hash|
          hash[decision.state_machine] = decision.to
        end
      end

      def rejected_parts
        decisions.select(&:rejected?)
      end

      def failures
        decisions.flat_map(&:failures)
      end

      # The first rejection or failure as "part: reason" - the publication
      # reason in the application example.
      def reason
        rejection = rejected_parts.first || decisions.find(&:failed?)
        return error.message if rejection.nil? && error.is_a?(AASM::LinkedRollback)
        return nil if rejection.nil?

        text = rejection.reason.to_s
        return text if text.empty? || text.start_with?("#{rejection.part}:")

        "#{rejection.part}: #{text}"
      end

      def reasons
        (rejected_parts + decisions.select(&:failed?)).map do |rejection|
          text = rejection.reason.to_s
          text.empty? || text.start_with?("#{rejection.part}:") ? text : "#{rejection.part}: #{text}"
        end
      end

      def message
        return "#{operation_name}: committed #{states.map { |key, value| "#{key}=#{value}" }.join(', ')}" if success?
        return "#{operation_name}: #{error.class}: #{error.message}" if error && reason.nil?

        "#{operation_name}: rejected by #{reason}"
      end
      alias_method :to_s, :message

      def fail_with(error)
        @error = error
        self
      end

      def to_h
        {
          operation: operation_name,
          success: success?,
          persisted: persisted?,
          reason: reason,
          states: states,
          failures: failures,
          parts: decisions.map(&:to_h)
        }
      end
    end
  end
end