# frozen_string_literal: true

module Audited
  # A row of a browse session: one audit of one record.
  #
  # The revision is reconstructed only from audits that belong to the browsed
  # record and fall inside the session's snapshot, so clicking a row never
  # reads attributes from another object's audits.
  class AuditRow
    attr_reader :browser, :audit

    def initialize(browser, audit)
      @browser = browser
      @audit = audit
    end

    def cursor
      @cursor ||= audit.to_cursor.with_through(browser.snapshot)
    end
    alias_method :to_cursor, :cursor
    alias_method :position, :cursor

    def version
      audit.version
    end
    alias_method :audit_version, :version

    def created_at
      audit.created_at
    end

    def action
      audit.action
    end

    def audited_changes
      audit.audited_changes
    end

    def comment
      audit.comment
    end

    def user
      audit.user
    end

    # The record as it looked right after this audit was written.
    def revision
      @revision ||= audit.revision_within(browser.snapshot_scope)
    end
    alias_method :snapshot, :revision
    alias_method :record, :revision
    alias_method :restore, :revision

    def inspect
      "#<#{self.class.name} version=#{version} created_at=#{created_at.inspect}>"
    end
  end

  # One page of a browse session.
  class AuditPage
    include Enumerable

    attr_reader :browser, :rows, :next_cursor, :previous_cursor

    def initialize(browser, rows, next_cursor: nil, previous_cursor: nil)
      @browser = browser
      @rows = rows.freeze
      @next_cursor = next_cursor
      @previous_cursor = previous_cursor
    end

    def each(&block)
      rows.each(&block)
    end

    def size
      rows.size
    end
    alias_method :length, :size
    alias_method :count, :size

    def empty?
      rows.empty?
    end

    def audits
      rows.map(&:audit)
    end
    alias_method :entries, :audits
    alias_method :items, :audits

    def revisions
      rows.map(&:revision)
    end
    alias_method :records, :revisions

    def cursors
      rows.map(&:cursor)
    end

    def snapshot
      browser.snapshot
    end

    def has_next?
      !next_cursor.nil?
    end
    alias_method :next?, :has_next?

    def has_previous?
      !previous_cursor.nil?
    end
    alias_method :previous?, :has_previous?

    def next_page
      return nil unless next_cursor

      browser.page(cursor: next_cursor)
    end

    def previous_page
      return nil unless previous_cursor

      browser.page(cursor: previous_cursor, direction: :backward)
    end

    def first_page?
      previous_cursor.nil?
    end

    def last_page?
      next_cursor.nil?
    end

    def to_a
      rows
    end

    def inspect
      "#<#{self.class.name} rows=#{size} next=#{next_cursor.inspect}>"
    end
  end

  # Paginated, point-in-time view of the audits of a single record.
  #
  # The browser fixes its range when it is created: only audits at or before
  # the record's current (created_at, version) position are visible. Audits
  # appended while a user is paging, including ones appended with the same
  # +created_at+ as the last row, do not change the pages this session already
  # promised. A new browser sees the new audits.
  #
  # Page boundaries are compound (created_at, version) cursors, so audits that
  # share a timestamp are neither skipped nor duplicated.
  class AuditBrowser
    include Enumerable

    DEFAULT_PER_PAGE = 25

    attr_reader :record, :per_page, :snapshot

    def initialize(record, per_page: DEFAULT_PER_PAGE, limit: nil, snapshot: nil, through: nil)
      @record = record
      @per_page = Integer(per_page || limit || DEFAULT_PER_PAGE)
      raise ArgumentError, "per_page must be positive" unless @per_page.positive?

      ceiling = snapshot || through
      @snapshot = ceiling.nil? ? current_ceiling : AuditCursor.from(ceiling)
      @opened_at = Time.now
    end
    alias_method :auditable, :record

    # Wall clock time at which this session was opened.
    attr_reader :opened_at

    # Cursor of the newest audit visible to this session.
    def ceiling
      snapshot
    end
    alias_method :snapshot_cursor, :ceiling
    alias_method :range, :snapshot

    # The audit that sits at the session ceiling, if it still exists.
    def snapshot_audit
      return nil unless snapshot

      snapshot_scope.where(version: snapshot.version).first
    end

    # Relation of this record's audits inside the session's snapshot.
    def snapshot_scope
      relation = record.audits
      snapshot ? relation.through_snapshot(snapshot.created_at, snapshot.version) : relation.none
    end
    alias_method :scope, :snapshot_scope

    # A page of rows. +cursor+ is exclusive: rows strictly after it are
    # returned (or strictly before it when +direction+ is +:backward+).
    #
    # Passing a cursor that carries a +through+ range re-opens this browser on
    # that range, which is what happens when a session's cursor is handed back
    # to a freshly created browser.
    def page(cursor: nil, after: nil, before: nil, limit: nil, per_page: nil, direction: :forward)
      position = AuditCursor.from(cursor || after || before)
      if position && position.through && snapshot.nil?
        @snapshot = position.through
      elsif position&.through && snapshot && position.through < snapshot
        @snapshot = position.through
      end

      size = Integer(per_page || limit || self.per_page)
      direction = derive_direction(cursor: cursor, after: after, before: before, direction: direction)

      relation = direction == :backward ? backward_relation(position) : forward_relation(position)

      fetched = relation.limit(size + 1).to_a
      more = fetched.size > size
      fetched = fetched.first(size)
      fetched = fetched.reverse if direction == :backward

      rows = fetched.map { |audit| AuditRow.new(self, audit) }
      next_cursor = direction == :backward ? nil : (more ? rows.last&.cursor : nil)
      previous_cursor = direction == :backward ? (more ? rows.first&.cursor : nil) : nil

      if direction == :backward
        next_cursor = rows.last&.cursor if rows.last && has_rows_after?(rows.last)
      elsif rows.first && has_rows_before?(rows.first)
        previous_cursor = rows.first.cursor
      end

      @last_page = AuditPage.new(self, rows, next_cursor: next_cursor, previous_cursor: previous_cursor)
    end
    alias_method :rows, :page
    alias_method :paginate, :page
    alias_method :slice, :page

    def first_page
      page
    end

    def last_page
      page(direction: :backward)
    end

    def next_page(cursor = nil)
      position = cursor || (@last_page || page).next_cursor
      position && page(cursor: position, direction: :forward)
    end

    def previous_page(cursor = nil)
      position = cursor || (@last_page || page).previous_cursor
      position && page(cursor: position, direction: :backward)
    end

    # Yield every page of the snapshot in order.
    def each_page
      return enum_for(:each_page) unless block_given?

      cursor = nil
      loop do
        current = page(cursor: cursor)
        yield current
        break unless current.next_cursor

        cursor = current.next_cursor
      end
    end
    alias_method :pages, :each_page

    # Yield every row of the snapshot in order.
    def each(&block)
      return enum_for(:each) unless block

      each_page { |page| page.each(&block) }
    end
    alias_method :each_row, :each

    def all
      to_a
    end

    # Locate one row of this record inside the snapshot.
    #
    # Accepts a cursor, an audit, a +[created_at, version]+ pair, a Hash, a
    # version number, or a time (the time resolves to the last audit at or
    # before it, matching the classic time-only reading). Returns +nil+ when
    # the position belongs to another record, sits outside the snapshot, or
    # does not exist, so one record's history can never surface another's rows.
    def find(position)
      audit = resolve_audit(position)
      return nil unless audit

      AuditRow.new(self, audit)
    end
    alias_method :row_at, :find
    alias_method :at, :find
    alias_method :fetch, :find
    alias_method :[], :find

    # The restored record for a row, or +nil+ when the row is not visible.
    def revision_at(position)
      row = find(position)
      row&.revision
    end
    alias_method :snapshot_at, :revision_at
    alias_method :restore, :revision_at

    def exists?(position)
      !resolve_audit(position).nil?
    end
    alias_method :include?, :exists?

    def count
      snapshot_scope.count
    end

    def empty?
      count.zero?
    end

    def inspect
      "#<#{self.class.name} record=#{record.class.name}##{record.id} per_page=#{per_page} snapshot=#{snapshot.inspect}>"
    end

    # Build a cursor pointing at +audit+ but bounded by this snapshot.
    def cursor_for(audit)
      audit.to_cursor.with_through(snapshot)
    end

    protected

    def current_ceiling
      newest = record.audits.reverse_chronological.first
      newest&.to_cursor
    end

    def forward_relation(position)
      relation = snapshot_scope.chronological
      position ? relation.after_position(position.created_at, position.version) : relation
    end

    def backward_relation(position)
      relation = snapshot_scope.reverse_chronological
      position ? relation.before_position(position.created_at, position.version) : relation
    end

    def has_rows_after?(row)
      forward_relation(row.cursor).exists?
    end

    def has_rows_before?(row)
      backward_relation(row.cursor).exists?
    end

    def derive_direction(cursor:, after:, before:, direction:)
      return :backward if before
      return :forward if after
      return :backward if direction.to_s == "backward"

      if cursor.is_a?(Hash)
        requested = cursor[:direction] || cursor["direction"]
        return :backward if requested.to_s == "backward"
      end

      :forward
    end

    def resolve_audit(position)
      case position
      when nil
        nil
      when Integer
        resolve_version(position)
      when AuditCursor
        resolve_cursor(position)
      when Hash
        resolve_cursor(AuditCursor.from(position))
      when Array
        resolve_cursor(AuditCursor.from(position))
      else
        if AuditCursor.audit_like?(position)
          resolve_cursor(AuditCursor.from(position))
        else
          resolve_time(position)
        end
      end
    end

    def resolve_version(version)
      snapshot_scope.where(version: version).first
    end

    # Versions are unique within one record, so a cursor resolves by version.
    # The snapshot still bounds the answer: a cursor at or beyond the ceiling
    # (for example a row appended while this session was open) resolves to nil.
    def resolve_cursor(position)
      return nil if position.nil?
      return nil if snapshot && position > snapshot

      resolve_version(position.version)
    end

    def resolve_time(time)
      snapshot_scope.up_until(time).reverse_chronological.first
    end
  end
end
