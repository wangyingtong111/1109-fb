# frozen_string_literal: true

module Discard
  # Per-record outcome of a bulk discard requested with
  # <tt>discard_all(result: true)</tt>.
  #
  # The batch gives exactly one {Entry} to every row the receiver enumerates,
  # including rows that were already discarded. The entry's status always
  # reflects the state the database reports after the batch, so a callback veto
  # is never reported as a change and an already discarded row is never
  # reported as freshly discarded.
  #
  # Entries are stable for a caller that wants to locate a row it sent in:
  # {Entry#id} is the record's primary key, and {#[]} looks an entry up by it.
  #
  # ==== Example
  #
  #   result = Post.where(id: [a.id, b.id, c.id]).discard_all(result: true)
  #   result[a.id].status                 # => :discarded
  #   result[b.id].rejected?              # => true (callback aborted)
  #   result[c.id].already_discarded?     # => true (no callbacks ran)
  class BatchResult
    include Enumerable

    # A single row's outcome.
    #
    # +status+ is one of:
    #
    # - +:discarded+ - the row was kept and this batch discarded it.
    # - +:rejected+ - a <tt>before_discard</tt> callback aborted; the row is
    #   still kept.
    # - +:already_discarded+ - the row was already discarded, so the batch
    #   skipped it without running its callbacks.
    class Entry
      attr_reader :id, :record, :status

      def initialize(id:, record:, status:)
        @id = id
        @record = record
        @status = status
      end

      # @return [Boolean] true when this batch changed the row to discarded
      def discarded?
        status == :discarded
      end
      alias changed? discarded?
      alias success? discarded?

      # @return [Boolean] true when a business callback refused the discard
      def rejected?
        status == :rejected
      end
      alias failed? rejected?

      # @return [Boolean] true when the row was discarded before this batch
      def already_discarded?
        status == :already_discarded
      end
      alias skipped? already_discarded?
    end

    attr_reader :entries

    def initialize(entries)
      @entries = entries.freeze
      @by_id = entries.each_with_object({}) { |entry, memo| memo[entry.id] = entry }.freeze
    end

    # Yields every {Entry}, in batch order.
    def each(&block)
      entries.each(&block)
    end

    # @return [Entry, nil] the entry for the given record id
    def [](id)
      @by_id[id]
    end

    # @return [Entry] the entry for the given record id
    # @raise [KeyError] when the id was not part of the batch
    def fetch(id)
      @by_id.fetch(id)
    end

    # @return [Array<Entry>] rows this batch discarded
    def discarded
      entries.select(&:discarded?)
    end
    alias changed discarded

    # @return [Array<Entry>] rows a callback refused
    def rejected
      entries.select(&:rejected?)
    end
    alias failed rejected

    # @return [Array<Entry>] rows that were already discarded
    def already_discarded
      entries.select(&:already_discarded?)
    end
    alias skipped already_discarded

    # @return [Array] the records touched by the batch
    def records
      entries.map(&:record)
    end

    # @return [Array] ids of the rows touched by the batch
    def ids
      entries.map(&:id)
    end

    # @return [Hash{Object => Symbol}] id to status, for page rendering
    def statuses
      entries.each_with_object({}) { |entry, memo| memo[entry.id] = entry.status }
    end

    def size
      entries.size
    end
    alias length size

    def empty?
      entries.empty?
    end
  end
end
