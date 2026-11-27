# frozen_string_literal: true

module Discard
  # Handles soft deletes of records.
  #
  # Options:
  #
  # - :discard_column - The columns used to track soft delete, defaults to `:discarded_at`.
  module Model
    extend ActiveSupport::Concern

    included do
      class_attribute :discard_column
      self.discard_column = :discarded_at

      scope :kept, ->{ undiscarded }
      scope :undiscarded, ->{ where(discard_column => nil) }
      scope :discarded, ->{ where.not(discard_column => nil) }
      scope :with_discarded, ->{ unscope(where: discard_column) }

      define_model_callbacks :discard
      define_model_callbacks :undiscard
    end

    # :nodoc:
    module ClassMethods
      # Discards the records by instantiating each
      # record and calling its {#discard} method.
      # Each object's callbacks are executed.
      #
      # By default returns the collection of objects that were enumerated and
      # whose callbacks ran, which is the historical return value. Records that
      # were already discarded are not included.
      #
      # Pass <tt>result: true</tt> to get a {Discard::BatchResult} instead. It
      # reports one entry per row the receiver enumerates, classified by what
      # the database ends up holding: <tt>:discarded</tt> for rows this batch
      # changed, <tt>:rejected</tt> for rows a <tt>before_discard</tt> callback
      # refused, and <tt>:already_discarded</tt> for rows that were discarded
      # before the batch started (their callbacks are not run again). Entries
      # are keyed by primary key, so a page can locate the row it sent in.
      #
      # The set of rows is the receiver's own enumeration, so use
      # <tt>with_discarded</tt> when a default scope would otherwise hide the
      # already discarded rows.
      #
      # A rejection does not stop the batch: the remaining rows are still
      # attempted.
      #
      # Note: Instantiation, callback execution, and update of each
      # record can be time consuming when you're discarding many records at
      # once. It generates at least one SQL +UPDATE+ query per record (or
      # possibly more, to enforce your callbacks). If you want to discard many
      # rows quickly, without concern for their associations or callbacks, use
      # #update_all(discarded_at: Time.current) instead.
      #
      # ==== Examples
      #
      #   Person.where(age: 0..18).discard_all
      #
      #   result = Person.where(age: 0..18).discard_all(result: true)
      #   result.discarded          # => [entry, ...]
      #   result.rejected           # => [entry, ...]
      #   result.already_discarded  # => [entry, ...]
      #   result[person.id].status  # => :discarded, :rejected or :already_discarded
      def discard_all(result: false)
        return kept.each(&:discard) unless result

        _discard_batch_all { |record| record.discard }
      end

      # Discards the records by instantiating each
      # record and calling its {#discard!} method.
      # Each object's callbacks are executed.
      # Returns the collection of objects that were discarded.
      #
      # Note: Instantiation, callback execution, and update of each
      # record can be time consuming when you're discarding many records at
      # once. It generates at least one SQL +UPDATE+ query per record (or
      # possibly more, to enforce your callbacks). If you want to discard many
      # rows quickly, without concern for their associations or callbacks, use
      # #update_all!(discarded_at: Time.current) instead.
      #
      # ==== Examples
      #
      #   Person.where(age: 0..18).discard_all!
      def discard_all!
        kept.each(&:discard!)
      end

      # Undiscards the records by instantiating each
      # record and calling its {#undiscard} method.
      # Each object's callbacks are executed.
      # Returns the collection of objects that were undiscarded.
      #
      # Note: Instantiation, callback execution, and update of each
      # record can be time consuming when you're undiscarding many records at
      # once. It generates at least one SQL +UPDATE+ query per record (or
      # possibly more, to enforce your callbacks). If you want to undiscard many
      # rows quickly, without concern for their associations or callbacks, use
      # #update_all(discarded_at: nil) instead.
      #
      # ==== Examples
      #
      #   Person.where(age: 0..18).undiscard_all
      def undiscard_all
        discarded.each(&:undiscard)
      end

      # Undiscards the records by instantiating each
      # record and calling its {#undiscard!} method.
      # Each object's callbacks are executed.
      # Returns the collection of objects that were undiscarded.
      #
      # Note: Instantiation, callback execution, and update of each
      # record can be time consuming when you're undiscarding many records at
      # once. It generates at least one SQL +UPDATE+ query per record (or
      # possibly more, to enforce your callbacks). If you want to undiscard many
      # rows quickly, without concern for their associations or callbacks, use
      # #update_all!(discarded_at: nil) instead.
      #
      # ==== Examples
      #
      #   Person.where(age: 0..18).undiscard_all!
      def undiscard_all!
        discarded.each(&:undiscard!)
      end

      private

      # Runs a discard attempt for the rows the receiver enumerates and builds
      # a {Discard::BatchResult} whose entries are classified from the state
      # the database actually holds, so an entry can never disagree with what
      # a later read sees.
      #
      # Rows that were already discarded are part of the result as
      # <tt>:already_discarded</tt>, but their callbacks are not run again.
      def _discard_batch_all
        klass = all.klass
        primary_key = klass.primary_key

        records = all.to_a
        return BatchResult.new([]) if records.empty?

        ids = records.map(&:id)
        before = _discard_states(klass, primary_key, ids)

        records.each do |record|
          next if before.fetch(record.id, false)

          yield(record)
        end

        after = _discard_states(klass, primary_key, ids)

        entries = records.map do |record|
          id = record.id
          persisted = after.fetch(id, false)
          status =
            if persisted
              before.fetch(id, false) ? :already_discarded : :discarded
            else
              :rejected
            end

          BatchResult::Entry.new(id: id, record: record, status: status)
        end

        BatchResult.new(entries)
      end

      # Reports the state the database actually holds for the given rows.
      # It ignores the default scope so a row is classified by its own value
      # even if the scope would hide it from an ordinary read.
      def _discard_states(klass, primary_key, ids)
        klass.unscoped.where(primary_key => ids)
          .pluck(primary_key, klass.discard_column)
          .each_with_object({}) { |(id, value), memo| memo[id] = value.present? }
      end
    end

    # @return [Boolean] true if this record has been discarded, otherwise false
    def discarded?
      self[self.class.discard_column].present?
    end

    # @return [Boolean] false if this record has been discarded, otherwise true
    def undiscarded?
      !discarded?
    end
    alias kept? undiscarded?

    # Discard the record in the database
    #
    # @return [Boolean] true if successful, otherwise false
    def discard
      return false if discarded?

      with_transaction_returning_status do
        run_callbacks(:discard) do
          update_attribute(self.class.discard_column, Time.current)
        end
      end
    end

    # Discard the record in the database
    #
    # There's a series of callbacks associated with #discard!. If the
    # <tt>before_discard</tt> callback throws +:abort+ the action is cancelled
    # and #discard! raises {Discard::RecordNotDiscarded}.
    #
    # @return [Boolean] true if successful
    # @raise {Discard::RecordNotDiscarded}
    def discard!
      discard || _raise_record_not_discarded
    end

    # Undiscard the record in the database
    #
    # @return [Boolean] true if successful, otherwise false
    def undiscard
      return false unless discarded?

      with_transaction_returning_status do
        run_callbacks(:undiscard) do
          update_attribute(self.class.discard_column, nil)
        end
      end
    end

    # Undiscard the record in the database
    #
    # There's a series of callbacks associated with #undiscard!. If the
    # <tt>before_undiscard</tt> callback throws +:abort+ the action is cancelled
    # and #undiscard! raises {Discard::RecordNotUndiscarded}.
    #
    # @return [Boolean] true if successful
    # @raise {Discard::RecordNotUndiscarded}
    def undiscard!
      undiscard || _raise_record_not_undiscarded
    end

    private

    def _raise_record_not_discarded
      raise ::Discard::RecordNotDiscarded.new(discarded_fail_message, self)
    end

    def _raise_record_not_undiscarded
      raise ::Discard::RecordNotUndiscarded.new(undiscarded_fail_message, self)
    end

    def discarded_fail_message
      return "A discarded record cannot be discarded" if discarded?

      "Failed to discard the record"
    end

    def undiscarded_fail_message
      return "An undiscarded record cannot be undiscarded" if undiscarded?

      "Failed to undiscard the record"
    end
  end
end
