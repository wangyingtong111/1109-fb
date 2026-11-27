# frozen_string_literal: true

require "time"

module Audited
  # Position of an audit inside one record's history.
  #
  # Audits of the same record may share +created_at+ down to the microsecond,
  # so a timestamp alone is not a usable page boundary: ordering by time and
  # paging with "created_at > last_created_at" silently skips the audits that
  # share the boundary timestamp. The pair (+created_at+, +version+) is unique
  # per record and is what pagination and point lookups use.
  #
  # A cursor may also carry +through+: the cursor that was the ceiling of the
  # browse session it came from. Handing such a cursor back to a fresh browser
  # keeps the session on its original range, so rows appended while the user is
  # paging do not shift the pages already promised.
  class AuditCursor
    include Comparable

    attr_reader :created_at, :version, :through

    class << self
      # Accepts a cursor, an audit, a Hash (symbol or string keys) or an
      # +[created_at, version]+ pair.
      def from(value, through: nil)
        case value
        when nil
          nil
        when AuditCursor
          through ? value.with_through(through) : value
        when Hash
          new(
            value[:created_at] || value["created_at"],
            value[:version] || value["version"],
            through: through || (value[:through] || value["through"])
          )
        when Array
          new(value[0], value[1], through: through)
        else
          if audit_like?(value)
            new(value.created_at, value.version, through: through)
          else
            raise ArgumentError, "cannot build an audit cursor from #{value.class}"
          end
        end
      end

      def audit_like?(value)
        value.respond_to?(:created_at) && value.respond_to?(:version)
      end
    end

    def initialize(created_at, version, through: nil)
      raise ArgumentError, "created_at is required" if created_at.nil?
      raise ArgumentError, "version is required" if version.nil?

      @created_at = cast_time(created_at)
      @version = Integer(version)
      @through = through.is_a?(AuditCursor) ? through : self.class.from(through)
    end

    # Another cursor for the same position, carrying +ceiling+ as its range.
    def with_through(ceiling)
      self.class.new(created_at, version, through: ceiling)
    end

    def <=>(other)
      other = self.class.from(other)
      return nil unless other

      [created_at, version] <=> [other.created_at, other.version]
    end

    def ==(other)
      other = self.class.from(other)
      !other.nil? && created_at == other.created_at && version == other.version
    end
    alias_method :eql?, :==

    def hash
      [created_at, version].hash
    end

    def to_audit
      self
    end

    def to_h
      {
        created_at: created_at.utc.iso8601(6),
        version: version,
        through: through&.to_h
      }
    end

    def as_json(*)
      to_h
    end

    def to_json(*args)
      to_h.to_json(*args)
    end

    def to_s
      "#{created_at.utc.iso8601(6)}@#{version}"
    end

    def inspect
      "#<#{self.class.name} #{self}>"
    end

    private

    def cast_time(value)
      if value.is_a?(Time)
        value
      elsif value.respond_to?(:to_time) && !value.is_a?(String)
        value.to_time
      else
        Time.parse(value.to_s)
      end
    end
  end
end
