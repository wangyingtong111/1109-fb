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
      # Returns the collection of objects that were discarded.
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
      def discard_all
        kept.each(&:discard)
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
    # Only a record that is backed by a row in the database can be discarded.
    # Unsaved records - including ones that were given the primary key of an
    # existing row - and records whose row has already been deleted are left
    # untouched: no discard callbacks are run, nothing is written (so no new
    # row is ever inserted and no other row sharing the same key is updated)
    # and +false+ is returned.
    #
    # @return [Boolean] true if successful, otherwise false
    def discard
      return false unless _row_in_database?
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
    # A record that is not backed by a row in the database (an unsaved or
    # destroyed record, or one whose row has since been deleted) is not written
    # and raises {Discard::RecordNotDiscarded} without running any callbacks.
    #
    # @return [Boolean] true if successful
    # @raise {Discard::RecordNotDiscarded}
    def discard!
      discard || _raise_record_not_discarded
    end

    # Undiscard the record in the database
    #
    # As with {#discard}, only a record backed by a row in the database is
    # touched: unsaved and destroyed records run no callbacks, write nothing
    # and return +false+.
    #
    # @return [Boolean] true if successful, otherwise false
    def undiscard
      return false unless _row_in_database?
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

    # Whether this object is still backed by a row in the database.
    #
    # This deliberately looks at the record's own persistence state as well as
    # the current contents of the table. An unsaved record - even one given the
    # primary key of an existing row - is not backed by a row of its own, and a
    # record whose row was deleted (directly or in another process) is gone.
    # Updating either of them would insert a new row or hit an unrelated row
    # that happens to share the same primary key.
    def _row_in_database?
      return false if new_record? || destroyed?

      columns = _database_identity_columns
      return false if columns.empty?

      values = columns.map { |column| attribute_in_database(column) }
      return false if values.any?(&:nil?)

      # Look past the default scope so a discarded row is still recognised.
      self.class.unscoped.where(columns.zip(values).to_h).exists?
    end

    # The columns ActiveRecord would use to address this record's row: the
    # query constraints when the model declares them, otherwise the primary key
    # (which may itself be composite).
    def _database_identity_columns
      if self.class.respond_to?(:query_constraints_list) && self.class.query_constraints_list
        self.class.query_constraints_list
      else
        Array(self.class.primary_key)
      end
    end

    def _raise_record_not_discarded
      raise ::Discard::RecordNotDiscarded.new(discarded_fail_message, self)
    end

    def _raise_record_not_undiscarded
      raise ::Discard::RecordNotUndiscarded.new(undiscarded_fail_message, self)
    end

    def discarded_fail_message
      return "A new record cannot be discarded" if new_record?
      return "A destroyed record cannot be discarded" if destroyed?
      return "A deleted record cannot be discarded" unless _row_in_database?
      return "A discarded record cannot be discarded" if discarded?

      "Failed to discard the record"
    end

    def undiscarded_fail_message
      return "A new record cannot be undiscarded" if new_record?
      return "A destroyed record cannot be undiscarded" if destroyed?
      return "A deleted record cannot be undiscarded" unless _row_in_database?
      return "An undiscarded record cannot be undiscarded" if undiscarded?

      "Failed to undiscard the record"
    end
  end
end
