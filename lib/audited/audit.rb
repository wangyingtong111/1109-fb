# frozen_string_literal: true

require "set"

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

    # Stable order for browsing: timestamps may be shared by several audits, so
    # the version disambiguates them.
    scope :chronological, -> { reorder(created_at: :asc, version: :asc) }
    scope :reverse_chronological, -> { reorder(created_at: :desc, version: :desc) }

    # Audits that existed at a fixed range boundary: written by +date_or_time+
    # and not newer than +version+. The version cap also keeps a backdated audit
    # logged after the boundary out of the range, which tuple ordering alone
    # would let through.
    scope :through_snapshot, ->(date_or_time, version) {
      where("created_at <= :time AND version <= :version", time: date_or_time, version: version)
    }

    # Audits strictly after the compound position (created_at, version).
    scope :after_position, ->(date_or_time, version) {
      where("created_at > :time OR (created_at = :time AND version > :version)",
        time: date_or_time, version: version)
    }

    # Audits strictly before the compound position (created_at, version).
    scope :before_position, ->(date_or_time, version) {
      where("created_at < :time OR (created_at = :time AND version < :version)",
        time: date_or_time, version: version)
    }

    # Return all audits older than the current one.
    def ancestors
      self.class.ascending.auditable_finder(auditable_id, auditable_type).to_version(version)
    end

    # Return an instance of what the object looked like at this revision. If
    # the object has been destroyed, this will be a new record.
    def revision
      revision_from(ancestors)
    end

    # Return an instance of what the object looked like right after this audit,
    # reconstructed from the audits in +scope+ that are at or before this
    # audit's position. The scope is expected to be bound to this audit's
    # auditable record (a snapshot relation does that), which keeps another
    # object's audits out of the result.
    def revision_within(scope)
      relation = scope.auditable_finder(auditable_id, auditable_type)
        .through_snapshot(created_at, version)
        .chronological
      revision_from(relation)
    end

    # Position of this audit inside its record's history. Because
    # +created_at+ is not unique, the version is part of the boundary.
    def to_cursor
      Audited::AuditCursor.new(created_at, version)
    end
    alias_method :audit_cursor, :to_cursor

    # Returns a hash of the changed attributes with the new values
    def new_attributes
      (audited_changes || {}).each_with_object({}.with_indifferent_access) do |(attr, values), attrs|
        attrs[attr] = (action == "update") ? values.last : values
      end
    end

    # Returns a hash of the changed attributes with the old values
    def old_attributes
      (audited_changes || {}).each_with_object({}.with_indifferent_access) do |(attr, values), attrs|
        attrs[attr] = (action == "update") ? values.first : values
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
        auditable_type.constantize.create!(audited_changes)
      when "update"
        # changes back attributes
        auditable.update!(audited_changes.transform_values(&:first))
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

    def revision_from(relation)
      clazz = auditable_type.constantize
      (clazz.find_by_id(auditable_id) || clazz.new).tap do |m|
        self.class.assign_revision_attributes(
          m,
          self.class.reconstruct_attributes(relation).merge(audit_version: version)
        )
      end
    end

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
