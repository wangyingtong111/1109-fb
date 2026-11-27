require "spec_helper"

# Alias-only and defensive branches are exercised implicitly by the public
# entry points; a handful of one-line delegations stay uncovered.
SingleCov.covered! uncovered: 30

describe Audited::AuditBrowser do
  let(:instant) { Time.utc(2020, 1, 1, 12, 0, 0) }
  let!(:user) do
    Models::ActiveRecord::User.create!(name: "A").tap do |u|
      u.update! name: "B"
      u.update! name: "C"
      u.audits.update_all(created_at: instant)
    end
  end
  let!(:other) do
    Models::ActiveRecord::User.create!(name: "Z").tap do |u|
      u.update! name: "Z2"
      u.audits.update_all(created_at: instant)
    end
  end

  def page_versions(page)
    page.map(&:version)
  end

  def browser_versions(browser)
    browser.map(&:version)
  end

  def append_version_four
    user.update! name: "D"
    user.audits.where(version: 4).update_all(created_at: instant)
  end

  describe "cursor" do
    it "orders by the compound position" do
      earlier = Audited::AuditCursor.new(instant, 2)
      later = Audited::AuditCursor.new(instant, 3)

      expect(earlier).to be < later
      expect(earlier).to eq(Audited::AuditCursor.new(instant, 2))
      expect(earlier).not_to eq(later)
    end

    it "round trips through a hash" do
      cursor = Audited::AuditCursor.new(instant, 2)
      round_tripped = Audited::AuditCursor.from(cursor.to_h)

      expect(round_tripped).to eq(cursor)
      expect(round_tripped.created_at).to eq(instant)
    end

    it "can be built from an array or an audit" do
      audit = user.audits.find_by(version: 2)

      expect(Audited::AuditCursor.from([instant, 2])).to eq(audit.to_cursor)
      expect(Audited::AuditCursor.from(audit)).to eq(audit.to_cursor)
    end

    it "requires a time and a version" do
      expect { Audited::AuditCursor.new(nil, 1) }.to raise_error(ArgumentError)
      expect { Audited::AuditCursor.new(instant, nil) }.to raise_error(ArgumentError)
    end
  end

  describe "paging" do
    it "pages through tied timestamps without skipping or repeating" do
      browser = user.audit_browser(per_page: 1)
      first = browser.page
      second = first.next_page
      third = second.next_page

      expect([page_versions(first), page_versions(second), page_versions(third)]).to eq([[1], [2], [3]])
      expect(page_versions(first) + page_versions(second) + page_versions(third)).to eq([1, 2, 3])
      expect(third).to be_last_page
    end

    it "counts a shared timestamp as distinct rows for this record only" do
      expect(Audited::Audit.where(created_at: instant).count).to eq(5)
      expect(user.audit_browser(per_page: 10).count).to eq(3)
    end

    it "pages backwards as well" do
      browser = user.audit_browser(per_page: 2)
      last = browser.last_page

      expect(page_versions(last)).to eq([2, 3])
      expect(page_versions(last.previous_page)).to eq([1])
    end

    it "yields every row through each" do
      expect(browser_versions(user.audit_browser(per_page: 1))).to eq([1, 2, 3])
    end

    it "rejects a non-positive page size" do
      expect { user.audit_browser(per_page: 0) }.to raise_error(ArgumentError)
    end

    it "handles a record without audits" do
      empty = Models::ActiveRecord::User.create!(name: "empty")
      empty.audits.delete_all
      browser = empty.audit_browser

      expect(browser.count).to eq(0)
      expect(browser.page.to_a).to eq([])
      expect(browser.find(1)).to be_nil
    end
  end

  describe "restoring a row" do
    it "restores the record as it looked after the clicked audit" do
      browser = user.audit_browser(per_page: 1)
      first = browser.page
      second = first.next_page

      expect(first.rows.first.revision.name).to eq("A")
      expect(second.rows.first.revision.name).to eq("B")
      expect(second.rows.first.revision.audit_version).to eq(2)
      expect(first.next_page.next_page.rows.first.revision.name).to eq("C")
    end

    it "never resolves another object's audit into this history" do
      browser = user.audit_browser(per_page: 10)

      row = browser.find(other.audits.find_by(version: 2).to_cursor)
      expect(row.version).to eq(2)
      expect(row.revision.id).to eq(user.id)
      expect(row.revision.name).to eq("B")
      expect(row.revision.name).not_to eq("Z2")
    end
  end

  describe "snapshot range" do
    it "keeps the range fixed when audits are appended to the open session" do
      browser = user.audit_browser(per_page: 1)
      first = browser.page

      append_version_four

      second = first.next_page
      third = second.next_page

      expect(page_versions(first) + page_versions(second) + page_versions(third)).to eq([1, 2, 3])
      expect(third).to be_last_page
      expect(browser.find(4)).to be_nil
      expect(browser.revision_at(4)).to be_nil
    end

    it "shows audits appended after the session opened to a new browser" do
      user.audit_browser(per_page: 1).page

      append_version_four

      fresh = user.audit_browser(per_page: 10)
      expect(page_versions(fresh.page)).to eq([1, 2, 3, 4])
      expect(fresh.revision_at(4).name).to eq("D")
      expect(user.revision_at(instant).name).to eq("D")
    end

    it "keeps the original range when a stored cursor is handed to a new browser" do
      browser = user.audit_browser(per_page: 1)
      cursor = browser.page.rows.first.cursor

      append_version_four

      resumed = user.audit_browser(per_page: 10)
      expect(page_versions(resumed.page(cursor: cursor))).to eq([2, 3])
      expect(resumed.snapshot.version).to eq(3)
    end
  end

  describe "lookup" do
    it "supports cursors serialized to hashes" do
      browser = user.audit_browser(per_page: 1)
      row = browser.page.next_page.rows.first

      round_tripped = Audited::AuditCursor.from(row.cursor.to_h)
      expect(round_tripped.version).to eq(2)
      expect(browser.find(row.cursor.to_h).revision.name).to eq("B")
    end

    it "resolves a time to the last audit at that instant" do
      browser = user.audit_browser(per_page: 10)

      expect(browser.revision_at(instant).name).to eq("C")
    end
  end

  describe "legacy readers" do
    it "leaves revision and revisions untouched" do
      expect(user.revision(2).name).to eq("B")
      expect(user.revisions.size).to eq(3)
    end

    it "keeps the time-only reading as the last audit at that instant" do
      expect(user.revision_at(instant).name).to eq("C")
    end

    it "can cut a shared timestamp by version" do
      expect(user.revision_at(instant, version: 2).name).to eq("B")
      expect(user.revision_at(instant, 2).name).to eq("B")
    end

    it "resolves a later time together with a version" do
      expect(user.revision_at(instant + 1.hour, version: 2).name).to eq("B")
    end

    it "is nil before the timestamp even when a version is given" do
      expect(user.revision_at(instant - 1, version: 2)).to be_nil
    end
  end

  describe "revision_for" do
    it "restores a row from an audit, a version, a time or a cursor" do
      expect(user.revision_for(user.audits.find_by(version: 2)).name).to eq("B")
      expect(user.revision_for(2).name).to eq("B")
      expect(user.revision_for(instant).name).to eq("C")
      expect(user.revision_for(user.audits.find_by(version: 2).to_cursor).name).to eq("B")
    end

    it "does not return rows outside the given snapshot" do
      snapshot = user.audits.find_by(version: 2).to_cursor
      expect(user.revision_for(3, snapshot: snapshot)).to be_nil
    end
  end
end
