require 'linked_operation_helper'

class LinkedOperationTest < Minitest::Spec
  describe 'the linked confirm operation' do
    let(:application) do
      app = LinkedApplication.new
      # force the initial states into the "database", like an ORM would
      app.aasm(:docs).enter_initial_state
      app.aasm(:pub).enter_initial_state
      app.persist_initial_states!
    end

    it 'declares the operation and generates the entry points' do
      assert LinkedApplication.aasm_linked_event?(:confirm)
      assert application.respond_to?(:confirm)
      assert application.respond_to?(:confirm!)
      assert application.respond_to?(:may_confirm?)
      assert_equal %i[docs pub], LinkedApplication.aasm_linked_operation(:confirm).parts.map(&:state_machine_name)
    end

    it 'refuses to fire an undeclared operation' do
      assert_raises(AASM::UnknownLinkedOperation) { application.class.aasm_linked_operation(:nope) }
    end

    it 'keeps both states, both persisted columns and skips the success notifications when publication rejects' do
      result = application.confirm!

      refute result.success?
      assert_equal :collecting, application.aasm(:docs).current_state
      assert_equal :queued, application.aasm(:pub).current_state
      assert_equal 'collecting', application.linked_rows[:docs_state]
      assert_equal 'queued', application.linked_rows[:pub_state]
      assert_empty application.notifications

      assert_equal 'pub.finish: delivery date is empty', result.reason
      rejected = result[:pub]
      assert rejected.rejected?
      assert_equal :queued, rejected.from
      assert_equal [:delivery_date?], rejected.failures
      permitted = result[:docs]
      assert permitted.permitted?
      refute permitted.committed?
    end

    it 'rejects the whole operation when the first part refuses' do
      application.materials_ok = false
      application.delivery_date = '2026-12-01'

      result = application.confirm!

      refute result.success?
      assert_equal :collecting, application.aasm(:docs).current_state
      assert_equal :queued, application.aasm(:pub).current_state
      assert_equal 'docs.finish: materials_valid?', result.reason
      assert_empty application.notifications
    end

    it 'commits both states, both persisted columns and reports the whole result to the success notifications' do
      application.delivery_date = '2026-12-01'

      result = application.confirm!

      assert result.success?
      assert result.persisted?
      assert_equal({ docs: :archived, pub: :published }, result.states)
      assert_equal :archived, application.aasm(:docs).current_state
      assert_equal :published, application.aasm(:pub).current_state
      assert_equal 'archived', application.linked_rows[:docs_state]
      assert_equal 'published', application.linked_rows[:pub_state]

      assert_equal 3, application.notifications.size
      application.notifications.each do |kind, seen|
        assert seen.success?, "#{kind} should have seen the complete successful result"
        assert_equal({ docs: :archived, pub: :published }, seen.states)
      end
      assert_equal %i[docs_finished pub_finished application_confirmed],
                   application.notifications.map(&:first)
    end

    it 'fires each part guard once, so a second attempt works off the same decision' do
      application.delivery_date = '2026-12-01'
      calls = 0
      application.define_singleton_method(:materials_valid?) do
        calls += 1
        materials_ok
      end

      assert application.confirm!.success?
      assert_equal 1, calls
    end

    it 'preflights without changing anything' do
      refute application.may_confirm?
      assert_equal :collecting, application.aasm(:docs).current_state
      assert_equal :queued, application.aasm(:pub).current_state
      assert_equal 'collecting', application.linked_rows[:docs_state]

      application.delivery_date = '2026-12-01'
      assert application.may_confirm?
      assert_equal :collecting, application.aasm(:docs).current_state
      assert_equal :queued, application.aasm(:pub).current_state
    end

    it 'retries the linked operation after the delivery date is provided' do
      refute application.confirm!.success?

      application.delivery_date = '2026-12-01'
      result = application.confirm!

      assert result.success?
      assert_equal({ docs: :archived, pub: :published }, result.states)
      assert_equal 'archived', application.linked_rows[:docs_state]
      assert_equal 'published', application.linked_rows[:pub_state]
    end
  end

  describe 'rollback and operation callbacks' do
    let(:failing_application) do
      app = LinkedApplication.new
      app.delivery_date = '2026-12-01'
      app.aasm(:docs).enter_initial_state
      app.aasm(:pub).enter_initial_state
      app.persist_initial_states!
      # publication accepts the guard but the row refuses to be written
      app.define_singleton_method(:aasm_write_state) do |state, name = :default|
        return false if name == :pub

        super(state, name)
      end
      app
    end

    it 'rolls the committed parts back when a later part aborts' do
      result = failing_application.confirm!

      refute result.success?
      assert_equal :collecting, failing_application.aasm(:docs).current_state
      assert_equal :queued, failing_application.aasm(:pub).current_state
      assert_equal 'collecting', failing_application.linked_rows[:docs_state]
      assert_equal :rollback, result[:docs].status
      assert result[:docs].rolled_back?
      refute result[:docs].committed?
      assert_equal :failed, result[:pub].status
      assert_equal 'pub.finish: persistence failed', result.reason
      assert_empty failing_application.notifications
      assert_empty result.states, 'a rolled back part must not appear as a committed state'

      # the operation can be retried once publication can be written
      allow = failing_application
      allow.define_singleton_method(:aasm_write_state) { |state, name = :default| super(state, name) }
      assert allow.confirm!.success?
    end

    it 'runs the failure callback with the full result when a part refuses' do
      application = LinkedApplication.new
      application.aasm(:docs).enter_initial_state
      application.aasm(:pub).enter_initial_state
      application.persist_initial_states!

      seen = []
      application.define_singleton_method(:application_confirmed) do |result|
        seen << [:confirmed, result.success?]
      end

      result = application.confirm!
      refute result.success?
      assert_empty seen, 'the success callback must not run'
    end

    it 'restores both in-memory states when a success callback raises' do
      app = LinkedApplication.new
      app.delivery_date = '2026-12-01'
      app.aasm(:docs).enter_initial_state
      app.aasm(:pub).enter_initial_state
      app.persist_initial_states!
      app.define_singleton_method(:docs_finished) do |*args|
        raise 'notification failed'
      end

      assert_raises(RuntimeError) { app.confirm! }
      assert_equal :collecting, app.aasm(:docs).current_state
      assert_equal :queued, app.aasm(:pub).current_state
      assert_equal 'collecting', app.linked_rows[:docs_state]
      assert_equal 'queued', app.linked_rows[:pub_state]
    end
  end

  describe 'argument forwarding' do
    it 'passes business arguments to every part and honours per-part args' do
      model = LinkedArgsApplication.new
      model.aasm(:docs).enter_initial_state
      model.aasm(:pub).enter_initial_state

      result = model.confirm!('R7')
      assert result.success?
      assert_equal [[:docs, 'R7'], [:pub, 'R7']], model.seen
    end

    it 'reports the guard that refused with the business argument' do
      model = LinkedArgsApplication.new
      model.aasm(:docs).enter_initial_state
      model.aasm(:pub).enter_initial_state

      result = model.aasm_linked_fire(:confirm, 'R8')
      refute result.success?
      assert_equal 'docs.finish: docs_token?', result.reason
      assert_equal :collecting, model.aasm(:docs).current_state
      assert_equal :queued, model.aasm(:pub).current_state
    end
  end

  describe 'inheritance and declaration' do
    it 'inherits a linked operation declared on the superclass' do
      subclass = Class.new(LinkedApplication)

      assert subclass.aasm_linked_event?(:confirm)
      assert subclass.aasm_linked_operation(:confirm).parts.length == 2
    end

    it 'remembers the result of the last firing' do
      application = LinkedApplication.new
      application.aasm(:docs).enter_initial_state
      application.aasm(:pub).enter_initial_state
      application.persist_initial_states!

      refute application.aasm_linked_fired?(:confirm)
      first = application.aasm_linked_fire(:confirm)
      assert application.aasm_linked_result(:confirm).equal?(first)

      application.delivery_date = '2026-12-01'
      application.aasm_linked_fire(:confirm)
      assert application.aasm_linked_fired?(:confirm)
    end

    it 'accepts the keyword spelling of a part declaration' do
      operation = LinkedApplication.aasm_linked_operation(:confirm)
      assert_equal %i[docs pub], operation.parts.map(&:state_machine_name)
      assert_equal %i[finish finish], operation.parts.map(&:event_name)
    end

    it 'rejects two parts on the same state machine at declaration time' do
      error = assert_raises(ArgumentError) do
        Class.new do
          include AASM
          aasm(:one) do
            state :a, initial: true
            state :b
            event(:step) { transitions from: :a, to: :b }
          end
          aasm_linked_event(:twice) do
            part :one, :step
            part :one, :step
          end
        end
      end
      assert_match(/already has a part for state machine/, error.message)
    end

    it 'runs the operation callbacks in order, with the whole result' do
      log = []
      LinkedApplication.class_eval do
        aasm_linked_event :audited_confirm do
          part :docs, :finish
          part :pub, :finish, reason: 'delivery date is empty'
          before { |result| log << [:before, result.decisions.map(&:status)] }
          success { |result| log << [:success, result.persisted?] }
          failure { |result| log << [:failure, result.reason] }
          after { |result| log << [:after, result.success?] }
          always { |result| log << [:always, result.success?] }
        end
      end

      application = LinkedApplication.new
      application.aasm(:docs).enter_initial_state
      application.aasm(:pub).enter_initial_state
      application.persist_initial_states!
      log.clear
      application.audited_confirm!

      assert_equal [[:failure, 'pub.finish: delivery date is empty'],
                    [:after, false], [:always, false]], log

      application.delivery_date = '2026-12-01'
      log.clear
      application.audited_confirm!
      assert_equal [[:before, %i[permitted permitted]], [:success, true],
                    [:after, true], [:always, true]], log
    end
  end

  describe 'legacy single-event behaviour' do
    it 'still fires the old events independently with their own result' do
      application = LinkedApplication.new
      application.aasm(:docs).enter_initial_state
      application.aasm(:pub).enter_initial_state
      application.persist_initial_states!

      assert application.finish_docs
      assert_equal :archived, application.aasm(:docs).current_state
      assert_equal :queued, application.aasm(:pub).current_state
      # the plain event does not persist, exactly like before
      assert_equal 'collecting', application.linked_rows[:docs_state]
      assert_equal 'queued', application.linked_rows[:pub_state]

      refute application.finish_pub
      assert_equal :archived, application.aasm(:docs).current_state
      assert_equal :queued, application.aasm(:pub).current_state
      assert_equal 'queued', application.linked_rows[:pub_state]
    end
  end
end
