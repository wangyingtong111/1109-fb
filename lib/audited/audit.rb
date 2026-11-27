# frozen_string_literal: true

require "set"
require "time"

module Audited
  # Audit saves the changes to ActiveRecord models.  It has the following attributes:
  #
  # * <tt>auditable</tt>: the ActiveRecord model that was changed
  # * <tt>user</tt>: the user that performed the change; a string or an ActiveRecord model
  # * <tt>action</tt>: one of create, update, or delete
  # * <tt>audited_changes</tt>: a hash of all the changes
  # * <tt>comment</tt>: a comment set with the audit
  # * <tt>version</tt>: the version of the model
  # * <tt>request_uuid</tt>: a uuid based that allows audits from the same controller request
  # * <tt>created_at</tt>: Time that the change was performed
  #

  class YAMLIfTextColumnType
    class << self
      def load(obj)
        if text_column?
          ActiveRecord::Coders::YAMLColumn.new(Object).load(obj)
        else
          obj
        end
      end

      def dump(obj)
        if text_column?
          ActiveRecord::Coders::YAMLColumn.new(Object).dump(obj)
        else
          obj
        end
      end

      def text_column?
        Audited.audit_class.columns_hash["audited_changes"].type.to_s == "text"
      end
    end
  end

  class Audit < ::ActiveRecord::Base
    belongs_to :auditable, polymorphic: true
    belongs_to :user, polymorphic: true
    belongs_to :associated, polymorphic: true

    before_create :set_version_number, :set_audit_user, :set_request_uuid, :set_remote_address

    cattr_accessor :audited_class_names
    self.audited_class_names = Set.new

    if ::ActiveRecord.version >= Gem::Version.new("7.1")
      serialize :audited_changes, coder: YAMLIfTextColumnType
    else
      serialize :audited_changes, YAMLIfTextColumnType
    end

    scope :ascending, -> { reorder(version: :asc) }
    scope :descending, -> { reorder(version: :desc) }
    scope :creates, -> { where(action: "create") }
    scope :updates, -> { where(action: "update") }
    scope :destroys, -> { where(action: "destroy") }

    scope :up_until, ->(date_or_time) { where("created_at <= ?", date_or_time) }
    scope :from_version, ->(version) { where("version >= ?", version) }
    scope :to_version, ->(version) { where("version <= ?", version) }
    scope :auditable_finder, ->(auditable_id, auditable_type) { where(auditable_id: auditable_id, auditable_type: auditable_type) }

    # Wording appended to an audit whose changes absorbed older audits.
    COMBINED_COMMENT = "This audit is the result of multiple audits being combined."

    # Machine readable bounds of the window that was folded into an audit. It
    # travels in the comment so the oldest surviving audit can still answer
    # "when did this record's history start?" and "which versions are gone?".
    # The versions are stored as one interval rather than a list so the marker
    # stays short however many folds happen, and because the snapshots inside
    # the window cannot be recovered - they are never invented back.
    HISTORY_MARKER = /\[audited:history started_at=(\S+) folded=(\d+)-(\d+)( pairs)?\]/

    # @private
    def self.parse_history_time(value)
      return nil if value.nil?

      time = case value
      when ::Time then value
      when ::DateTime then value.to_time
      when ::Date then start_of_day(value)
      when ::String
        value.strip.empty? ? nil : (::Time.zone ? ::Time.zone.parse(value) : ::Time.parse(value))
      else
        value.respond_to?(:to_time) ? value.to_time : nil
      end

      zone = ::Time.zone
      (zone && time) ? time.in_time_zone(zone) : time
    rescue ::ArgumentError, ::TypeError
      nil
    end

    # Midnight at the start of +date+ in the application zone, falling back to
    # UTC when no zone data is available.
    # @private
    def self.start_of_day(date)
      zone = ::Time.zone
      return ::Time.utc(date.year, date.month, date.day) if zone.nil?

      begin
        zone.local(date.year, date.month, date.day)
      rescue ::StandardError
        ::Time.utc(date.year, date.month, date.day)
      end
    end

    # Bounds recorded on an audit's comment, or +nil+ when nothing was folded.
    # @private
    def self.history_marker(comment)
      match = HISTORY_MARKER.match(comment.to_s)
      return nil unless match

      first = match[2].to_i
      last = match[3].to_i
      return nil if first > last

      {
        "started_at" => parse_history_time(match[1]),
        "folded_range" => (first..last),
        "pairs" => !match[4].nil?
      }
    end

    # @private
    def self.format_history_marker(started_at, folded_range, pairs: true)
      return nil if started_at.nil? || folded_range.nil?

      first = folded_range.first
      last = folded_range.last
      return nil if first.nil? || last.nil? || first > last

      "[audited:history started_at=#{started_at.utc.iso8601(6)} " \
        "folded=#{first}-#{last}#{" pairs" if pairs}]"
    end

    # @private
    # Changes of an audit normalized so that merging them keeps the [old, new]
    # shape. Creates and destroys store one value per attribute while updates
    # store the pair, so a fold has to wrap single values or the merged audit
    # would read a bare value as if it were a pair.
    def self.mergeable_changes(audit)
      changes = audit.audited_changes || {}
      return changes if audit.action == "update" || audit.pair_values?

      changes.each_with_object({}) { |(attr, value), out| out[attr] = [value, value] }
    end

    # Comment with any earlier fold marker removed, so a further fold can write
    # the accumulated bounds instead of stacking stale ones.
    # @private
    def self.comment_without_history_marker(comment)
      comment.to_s.sub(HISTORY_MARKER, "").strip
    end

    # Marker describing the window left behind by folding +audits+ into
    # +target+. It keeps the earliest known start of the history and widens the
    # folded interval, so repeated compactions still point back to the true
    # beginning. Returns +nil+ when nothing below the target is being removed.
    # @private
    def self.compaction_marker_for(audits, target)
      audits = Array(audits)
      markers = audits.map(&:history_marker).compact

      starts = markers.map { |marker| marker["started_at"] }.compact
      starts.concat(audits.map(&:created_at).compact)
      start = starts.min

      removed = (audits.map(&:version) - [target.version]).compact
      firsts = markers.map { |marker| marker["folded_range"].first }
      lasts = markers.map { |marker| marker["folded_range"].last }
      first = (firsts + removed).compact.min
      last = (lasts + removed).compact.max
      return nil if first.nil? || last.nil? || first > last

      format_history_marker(start, first..last)
    end

    # Return all audits older than the current one.
    def ancestors
      self.class.ascending.auditable_finder(auditable_id, auditable_type).to_version(version)
    end

    # Return an instance of what the object looked like at this revision. If
    # the object has been destroyed, this will be a new record.
    def revision
      clazz = auditable_type.constantize
      (clazz.find_by_id(auditable_id) || clazz.new).tap do |m|
        self.class.assign_revision_attributes(m, self.class.reconstruct_attributes(ancestors).merge(audit_version: version))
      end
    end

    # True when the attribute of this audit stores an <tt>[old, new]</tt> pair.
    #
    # Updates always store pairs; creates and destroys store a single value.
    # An audit that absorbed older audits keeps the pair shape for every
    # attribute (merging normalizes bare values into pairs) and says so with
    # the +pairs+ flag in its marker, so its decoding does not depend on which
    # action it happens to carry.
    def change_pair?(values)
      return false unless values.is_a?(Array) && values.length == 2

      action == "update" || pair_values?
    end

    # True when every attribute of this audit is stored as an <tt>[old, new]</tt>
    # pair, whatever the action is.
    def pair_values?
      marker = history_marker
      !marker.nil? && marker["pairs"]
    end

    # Bounds of the window folded into this audit, or +nil+ when nothing was.
    def history_marker
      self.class.history_marker(history_comment)
    end

    # The recorded comment, or +nil+ when the audit class has no comment column
    # or the column was not selected.
    # @private
    def history_comment
      return nil unless has_attribute?(:comment)

      comment
    rescue ::ActiveModel::MissingAttributeError
      nil
    end

    # True when older audits were folded into this one.
    def compacted_history?
      !history_marker.nil?
    end

    # When the auditable record first appeared in the audit log. Earlier than
    # this audit's own +created_at+ once a compaction folded the beginning.
    def history_started_at
      marker = history_marker
      (marker && marker["started_at"]) || created_at
    end

    # Versions whose individual audits were folded away.
    def folded_versions
      marker = history_marker
      marker ? marker["folded_range"].to_a : []
    end

    # Returns a hash of the changed attributes with the new values
    def new_attributes
      (audited_changes || {}).each_with_object({}.with_indifferent_access) do |(attr, values), attrs|
        attrs[attr] = change_pair?(values) ? values.last : values
      end
    end

    # Returns a hash of the changed attributes with the old values
    def old_attributes
      (audited_changes || {}).each_with_object({}.with_indifferent_access) do |(attr, values), attrs|
        attrs[attr] = change_pair?(values) ? values.first : values
      end
    end

    # Allows user to undo changes
    def undo
      case action
      when "create"
        # destroys a newly created record
        auditable.destroy!
      when "destroy"
        # creates a new record with the destroyed record attributes
        auditable_type.constantize.create!(new_attributes.to_h)
      when "update"
        # changes back attributes
        auditable.update!(old_attributes.to_h)
      else
        raise StandardError, "invalid action given #{action}"
      end
    end

    # Allows user to be set to either a string or an ActiveRecord object
    # @private
    def user_as_string=(user)
      # reset both either way
      self.user_as_model = self.username = nil
      user.is_a?(::ActiveRecord::Base) ?
        self.user_as_model = user :
        self.username = user
    end
    alias_method :user_as_model=, :user=
    alias_method :user=, :user_as_string=

    # @private
    def user_as_string
      user_as_model || username
    end
    alias_method :user_as_model, :user
    alias_method :user, :user_as_string

    # Returns the list of classes that are being audited
    def self.audited_classes
      audited_class_names.map(&:constantize)
    end

    # All audits made during the block called will be recorded as made
    # by +user+. This method is hopefully threadsafe, making it ideal
    # for background operations that require audit information.
    def self.as_user(user)
      last_audited_user = ::Audited.store[:audited_user]
      ::Audited.store[:audited_user] = user
      yield
    ensure
      ::Audited.store[:audited_user] = last_audited_user
    end

    # @private
    def self.reconstruct_attributes(audits)
      audits.each_with_object({}) do |audit, all|
        all.merge!(audit.new_attributes)
        all[:audit_version] = audit.version
      end
    end

    # @private
    def self.assign_revision_attributes(record, attributes)
      attributes.each do |attr, val|
        record = record.dup if record.frozen?

        if record.respond_to?("#{attr}=")
          record.attributes.key?(attr.to_s) ?
            record[attr] = val :
            record.send("#{attr}=", val)
        end
      end
      record
    end

    # use created_at as timestamp cache key
    def self.collection_cache_key(collection = all, *)
      super(collection, :created_at)
    end

    private

    def set_version_number
      if action == "create"
        self.version = 1
      else
        collection = (ActiveRecord::VERSION::MAJOR >= 6) ? self.class.unscoped : self.class
        max = collection.auditable_finder(auditable_id, auditable_type).maximum(:version) || 0
        self.version = max + 1
      end
    end

    def set_audit_user
      self.user ||= ::Audited.store[:audited_user] # from .as_user
      self.user ||= ::Audited.store[:current_user].try!(:call) # from Sweeper
      nil # prevent stopping callback chains
    end

    def set_request_uuid
      self.request_uuid ||= ::Audited.store[:current_request_uuid]
      self.request_uuid ||= SecureRandom.uuid
    end

    def set_remote_address
      self.remote_address ||= ::Audited.store[:current_remote_address]
    end
  end
end
