# frozen_string_literal: true

module Audited
  # Answer to "what did this record look like at that moment in time?".
  #
  # A history page needs to tell apart two reasons why an itemized revision is
  # not available:
  #
  # * <tt>:not_registered</tt> - the record had not been registered in the
  #   audit log yet, so the moment predates its history entirely. Nothing is
  #   known and nothing is claimed.
  # * <tt>:compacted</tt> - the record did exist, but the audits covering that
  #   window were folded into a newer audit. The individual snapshots cannot be
  #   rebuilt; only the window edges survive, plus the earliest point that can
  #   still be read in detail.
  # * <tt>:found</tt> - audits covering that moment are still stored one by one,
  #   so the usual reconstruction applies.
  #
  # When +status+ is <tt>:found</tt>, +revision+ is the reconstructed record.
  # Otherwise +revision+ is +nil+: a snapshot of a moment whose detail was
  # folded away is never invented from the current record.
  class AuditHistory
    STATUSES = [:not_registered, :compacted, :found].freeze

    # Stable names a history page can map onto its own wording.
    CODES = {
      not_registered: "not_registered",
      compacted: "compacted",
      found: "found"
    }.freeze

    class << self
      # Moment covered by audits that are still stored individually.
      def found(requested_at:, revision:, history_started_at: nil)
        new(status: :found, requested_at: requested_at, revision: revision,
          history_started_at: history_started_at)
      end

      # Moment inside a window whose audits were folded into a newer one.
      def compacted(requested_at:, history_started_at:, earliest_readable_at:,
        earliest_readable_version: nil, folded_versions: [])
        new(status: :compacted, requested_at: requested_at,
          history_started_at: history_started_at,
          earliest_readable_at: earliest_readable_at,
          earliest_readable_version: earliest_readable_version,
          folded_versions: folded_versions)
      end

      # Moment before the record was registered in the audit log.
      def not_registered(requested_at:, history_started_at: nil)
        new(status: :not_registered, requested_at: requested_at,
          history_started_at: history_started_at)
      end
    end

    attr_reader :status, :requested_at, :revision, :history_started_at,
      :earliest_readable_at, :earliest_readable_version, :folded_versions

    def initialize(status:, requested_at:, revision: nil, history_started_at: nil,
      earliest_readable_at: nil, earliest_readable_version: nil,
      folded_versions: [])
      unless STATUSES.include?(status)
        raise ArgumentError, "unknown history status #{status.inspect}"
      end

      @status = status
      @requested_at = requested_at
      @revision = revision
      @history_started_at = history_started_at
      @earliest_readable_at = earliest_readable_at
      @earliest_readable_version = earliest_readable_version
      @folded_versions = Array(folded_versions).freeze
      freeze
    end

    # The requested moment is covered by audits that are still stored one by one.
    def found?
      status == :found
    end

    # The requested moment is inside a window whose audits were folded away.
    def compacted?
      status == :compacted
    end

    # The requested moment predates the record's audit history.
    def not_registered?
      status == :not_registered
    end

    # Stable name for the status.
    def code
      CODES.fetch(status)
    end

    # Earliest moment that can still be read one audit at a time. This is the
    # "valid start" handed back for a compacted window; it is never earlier
    # than the requested moment's own detail, because that detail is gone.
    def effective_from
      earliest_readable_at
    end

    # Version of the oldest audit that is still stored, if known.
    def effective_version
      earliest_readable_version
    end

    def to_h
      {
        status: status,
        code: code,
        requested_at: requested_at,
        revision: revision,
        history_started_at: history_started_at,
        effective_from: effective_from,
        effective_version: effective_version,
        folded_versions: folded_versions
      }
    end

    def inspect
      "#<Audited::AuditHistory #{status} requested_at=#{requested_at.inspect} " \
        "history_started_at=#{history_started_at.inspect} " \
        "effective_from=#{effective_from.inspect} " \
        "effective_version=#{effective_version.inspect}>"
    end
  end
end
