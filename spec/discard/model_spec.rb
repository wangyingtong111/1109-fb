# frozen_string_literal: true

RSpec.describe Discard::Model do
  context "with simple Post model" do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
      end
    end

    context "an undiscarded Post" do
      let!(:post) { Post.create!(title: "My very first post") }

      it "is included in the default scope" do
        expect(Post.all).to eq([post])
      end

      it "is included in kept scope" do
        expect(Post.kept).to eq([post])
      end

      it "is included in undiscarded scope" do
        expect(Post.undiscarded).to eq([post])
      end

      it "is not included in discarded scope" do
        expect(Post.discarded).to eq([])
      end

      it "should not be discarded?" do
        expect(post).not_to be_discarded
      end

      it "should be undiscarded?" do
        expect(post).to be_undiscarded
      end

      it "should be kept?" do
        expect(post).to be_kept
      end

      describe '#discard' do
        it "sets discarded_at" do
          expect {
            post.discard
          }.to change { post.discarded_at }
        end

        it "sets discarded_at in DB" do
          expect {
            post.discard
          }.to change { post.reload.discarded_at }
        end
      end

      describe '#discard!' do
        it "sets discarded_at" do
          expect {
            post.discard!
          }.to change { post.discarded_at }
        end

        it "sets discarded_at in DB" do
          expect {
            post.discard!
          }.to change { post.reload.discarded_at }
        end
      end

      describe '#undiscard' do
        it "doesn't change discarded_at" do
          expect {
            post.undiscard
          }.not_to change { post.discarded_at }
        end

        it "doesn't change discarded_at in DB" do
          expect {
            post.undiscard
          }.not_to change { post.reload.discarded_at }
        end

        it "returns false" do
          expect(post.undiscard).to be false
        end
      end

      describe '#undiscard!' do
        it "raises Discard::RecordNotUndiscarded" do
          expect {
            post.undiscard!
          }.to raise_error(Discard::RecordNotUndiscarded, 'An undiscarded record cannot be undiscarded')
        end
      end
    end

    context "discarded Post" do
      let!(:post) { Post.create!(title: "A discarded post", discarded_at: Time.parse('2017-01-01')) }

      it "is included in the default scope" do
        expect(Post.all).to eq([post])
      end

      it "is not included in kept scope" do
        expect(Post.kept).to eq([])
      end

      it "is not included in undiscarded scope" do
        expect(Post.undiscarded).to eq([])
      end

      it "is included in discarded scope" do
        expect(Post.discarded).to eq([post])
      end

      it "should be discarded?" do
        expect(post).to be_discarded
      end

      it "should not be undiscarded?" do
        expect(post).to_not be_undiscarded
      end

      it "should not be kept?" do
        expect(post).to_not be_kept
      end

      describe '#discard' do
        it "doesn't change discarded_at" do
          expect {
            post.discard
          }.not_to change { post.discarded_at }
        end

        it "doesn't change discarded_at in DB" do
          expect {
            post.discard
          }.not_to change { post.reload.discarded_at }
        end

        it "returns false" do
          expect(post.discard).to be false
        end
      end

      describe '#discard!' do
        it "raises Discard::RecordNotDiscarded" do
          expect {
            post.discard!
          }.to raise_error(Discard::RecordNotDiscarded, "A discarded record cannot be discarded")
        end
      end

      describe '#undiscard' do
        it "clears discarded_at" do
          expect {
            post.undiscard
          }.to change { post.discarded_at }.to(nil)
        end

        it "clears discarded_at in DB" do
          expect {
            post.undiscard
          }.to change { post.reload.discarded_at }.to(nil)
        end
      end

      describe '#undiscard!' do
        it "clears discarded_at" do
          expect {
            post.undiscard!
          }.to change { post.discarded_at }.to(nil)
        end

        it "clears discarded_at in DB" do
          expect {
            post.undiscard!
          }.to change { post.reload.discarded_at }.to(nil)
        end
      end
    end
  end

  context "with default scope" do
    with_model :WithDefaultScope, scope: :all do
      table do |t|
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        default_scope -> { kept }
      end
    end
    let(:klass) { WithDefaultScope }

    context "an undiscarded record" do
      let!(:record) { klass.create! }

      it "is included in the default scope" do
        expect(klass.all).to eq([record])
      end

      it "is included in kept scope" do
        expect(klass.kept).to eq([record])
      end

      it "is included in undiscarded scope" do
        expect(klass.undiscarded).to eq([record])
      end

      it "is included in with_discarded scope" do
        expect(klass.with_discarded).to eq([record])
      end

      it "is not included in discarded scope" do
        expect(klass.discarded).to eq([])
      end
    end

    context "a discarded record" do
      let!(:record) { klass.create!(discarded_at: Time.current) }

      it "is not included in the default scope" do
        expect(klass.all).to eq([])
      end

      it "is not included in kept scope" do
        expect(klass.kept).to eq([])
      end

      it "is not included in undiscarded scope" do
        expect(klass.kept).to eq([])
      end

      it "is not included in discarded scope" do
        # This is not ideal, but I don't want to improve it at the expense of
        # models withot a default scope
        expect(klass.discarded).to eq([])
      end

      it "is included in with_discarded scope" do
        expect(klass.with_discarded).to eq([record])
      end

      it "is included in with_discarded.discarded scope" do
        expect(klass.with_discarded.discarded).to eq([record])
      end
    end
  end

  context "with custom column name" do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :deleted_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        self.discard_column = :deleted_at
      end
    end

    context "an undiscarded Post" do
      let!(:post) { Post.create!(title: "My very first post") }

      it "is included in the default scope" do
        expect(Post.all).to eq([post])
      end

      it "is included in kept scope" do
        expect(Post.kept).to eq([post])
      end

      it "is included in undiscarded scope" do
        expect(Post.undiscarded).to eq([post])
      end

      it "is not included in discarded scope" do
        expect(Post.discarded).to eq([])
      end

      it "should not be discarded?" do
        expect(post).not_to be_discarded
      end

      it "should be undiscarded?" do
        expect(post).to be_undiscarded
      end

      it "should be kept?" do
        expect(post).to be_kept
      end

      describe '#discard' do
        it "sets discarded_at" do
          expect {
            post.discard
          }.to change { post.deleted_at }
        end

        it "sets discarded_at in DB" do
          expect {
            post.discard
          }.to change { post.reload.deleted_at }
        end
      end

      describe '#undiscard' do
        it "doesn't change discarded_at" do
          expect {
            post.undiscard
          }.not_to change { post.deleted_at }
        end

        it "doesn't change discarded_at in DB" do
          expect {
            post.undiscard
          }.not_to change { post.reload.deleted_at }
        end
      end
    end

    context "discarded Post" do
      let!(:post) { Post.create!(title: "A discarded post", deleted_at: Time.parse('2017-01-01')) }

      it "is included in the default scope" do
        expect(Post.all).to eq([post])
      end

      it "is not included in kept scope" do
        expect(Post.kept).to eq([])
      end

      it "is not included in undiscarded scope" do
        expect(Post.undiscarded).to eq([])
      end

      it "is included in discarded scope" do
        expect(Post.discarded).to eq([post])
      end

      it "should be discarded?" do
        expect(post).to be_discarded
      end

      it "should not be undiscarded?" do
        expect(post).to_not be_undiscarded
      end

      it "should not be kept?" do
        expect(post).to_not be_kept
      end

      describe '#discard' do
        it "doesn't change discarded_at" do
          expect {
            post.discard
          }.not_to change { post.deleted_at }
        end

        it "doesn't change discarded_at in DB" do
          expect {
            post.discard
          }.not_to change { post.reload.deleted_at }
        end
      end

      describe '#undiscard' do
        it "clears discarded_at" do
          expect {
            post.undiscard
          }.to change { post.deleted_at }.to(nil)
        end

        it "clears discarded_at in DB" do
          expect {
            post.undiscard
          }.to change { post.reload.deleted_at }.to(nil)
        end
      end
    end
  end

  describe '.discard_all' do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
      end
    end

    let!(:post) { Post.create!(title: "My very first post") }
    let!(:post2) { Post.create!(title: "A second post") }

    it "can discard all posts" do
      expect {
        Post.discard_all
      }.to   change { post.reload.discarded? }.to(true)
        .and change { post2.reload.discarded? }.to(true)
    end

    it "can discard a single post" do
      Post.where(id: post.id).discard_all
      expect(post.reload).to be_discarded
      expect(post2.reload).not_to be_discarded
    end

    it "can discard no records" do
      Post.where(id: []).discard_all
      expect(post.reload).not_to be_discarded
      expect(post2.reload).not_to be_discarded
    end

    context "through a collection" do
      with_model :Comment, scope: :all do
        table do |t|
          t.belongs_to :user
          t.datetime :discarded_at
          t.timestamps null: false
        end

        model do
          include Discard::Model
        end
      end

      with_model :User, scope: :all do
        table do |t|
          t.timestamps null: false
        end

        model do
          include Discard::Model

          has_many :comments
        end
      end

      it "can be discard all related posts" do
        user1 = User.create!
        user2 = User.create!

        2.times { user1.comments.create! }
        2.times { user2.comments.create! }

        user1.comments.discard_all
        user1.comments.each do |comment|
          expect(comment).to be_discarded
          expect(comment).to_not be_undiscarded
          expect(comment).to_not be_kept
        end
        user2.comments.each do |comment|
          expect(comment).to_not be_discarded
          expect(comment).to be_undiscarded
          expect(comment).to be_kept
        end
      end
    end
  end

  describe '.discard_all!' do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
      end
    end

    let!(:post) { Post.create!(title: "My very first post") }
    let!(:post2) { Post.create!(title: "A second post") }

    it "can discard all posts" do
      expect {
        Post.discard_all!
      }.to   change { post.reload.discarded? }.to(true)
        .and change { post2.reload.discarded? }.to(true)
    end
  end

  describe '.undiscard_all' do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
      end
    end

    let!(:post) { Post.create!(title: "My very first post", discarded_at: Time.now) }
    let!(:post2) { Post.create!(title: "A second post", discarded_at: Time.now) }

    it "can undiscard all posts" do
      expect {
        Post.undiscard_all
      }.to   change { post.reload.discarded? }.to(false)
        .and change { post2.reload.discarded? }.to(false)
    end

    it "can undiscard a single post" do
      Post.where(id: post.id).undiscard_all
      expect(post.reload).not_to be_discarded
      expect(post2.reload).to be_discarded
    end

    it "can undiscard no records" do
      Post.where(id: []).undiscard_all
      expect(post.reload).to be_discarded
      expect(post2.reload).to be_discarded
    end
  end

  describe '.undiscard_all!' do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
      end
    end

    let!(:post) { Post.create!(title: "My very first post", discarded_at: Time.now) }
    let!(:post2) { Post.create!(title: "A second post", discarded_at: Time.now) }

    it "can undiscard all posts" do
      expect {
        Post.undiscard_all!
      }.to   change { post.reload.discarded? }.to(false)
        .and change { post2.reload.discarded? }.to(false)
    end
  end

  describe 'discard callbacks' do
    with_model :Post, scope: :all do
      table do |t|
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        before_discard :do_before_discard
        before_save :do_before_save
        after_save :do_after_save
        after_discard :do_after_discard

        def do_before_discard; end
        def do_before_save; end
        def do_after_save; end
        def do_after_discard; end
      end
    end

    def abort_callback
      if ActiveRecord::VERSION::MAJOR < 5
        false
      else
        throw :abort
      end
    end

    let!(:post) { Post.create! }

    it "runs callbacks in correct order" do
      expect(post).to receive(:do_before_discard).ordered
      expect(post).to receive(:do_before_save).ordered
      expect(post).to receive(:do_after_save).ordered
      expect(post).to receive(:do_after_discard).ordered

      expect(post.discard).to be true
      expect(post).to be_discarded
    end

    context 'before_discard' do
      it "can allow discard" do
        expect(post).to receive(:do_before_discard).and_return(true)
        expect(post.discard).to be true
        expect(post).to be_discarded
      end

      it "can prevent discard" do
        expect(post).to receive(:do_before_discard) { abort_callback }
        expect(post.discard).to be false
        expect(post).not_to be_discarded
      end

      describe '#discard!' do
        it "raises Discard::RecordNotDiscarded" do
          expect(post).to receive(:do_before_discard) { abort_callback }
          expect {
            post.discard!
          }.to raise_error(Discard::RecordNotDiscarded, "Failed to discard the record")
        end
      end
    end
  end

  describe 'undiscard callbacks' do
    with_model :Post, scope: :all do
      table do |t|
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        before_undiscard :do_before_undiscard
        before_save :do_before_save
        after_save :do_after_save
        after_undiscard :do_after_undiscard

        def do_before_undiscard; end
        def do_before_save; end
        def do_after_save; end
        def do_after_undiscard; end
      end
    end

    def abort_callback
      if ActiveRecord::VERSION::MAJOR < 5
        false
      else
        throw :abort
      end
    end

    let!(:post) { Post.create! discarded_at: Time.now }

    it "runs callbacks in correct order" do
      expect(post).to receive(:do_before_undiscard).ordered
      expect(post).to receive(:do_before_save).ordered
      expect(post).to receive(:do_after_save).ordered
      expect(post).to receive(:do_after_undiscard).ordered

      expect(post.undiscard).to be true
      expect(post).not_to be_discarded
    end

    context 'before_undiscard' do
      it "can allow undiscard" do
        expect(post).to receive(:do_before_undiscard).and_return(true)
        expect(post.undiscard).to be true
        expect(post).not_to be_discarded
      end

      it "can prevent undiscard" do
        expect(post).to receive(:do_before_undiscard) { abort_callback }
        expect(post.undiscard).to be false
        expect(post).to be_discarded
      end

      describe '#undiscard!' do
        it "raises Discard::RecordNotUndiscarded" do
          expect(post).to receive(:do_before_undiscard) { abort_callback }
          expect {
            post.undiscard!
          }.to raise_error(Discard::RecordNotUndiscarded, "Failed to undiscard the record")
        end
      end
    end
  end

  describe 'discarded? value within validations and callbacks' do
    with_model :Post, scope: :all do
      table do |t|
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        validate :guarded_validation, if: :discarded?
        before_discard :guarded_before_discard, if: :discarded?
        after_discard :guarded_after_discard, if: :discarded?
        before_undiscard :guarded_before_undiscard, if: :discarded?
        after_undiscard :guarded_after_undiscard, if: :discarded?

        def guarded_validation; end
        def guarded_before_discard; end
        def guarded_after_discard; end
        def guarded_before_undiscard; end
        def guarded_after_undiscard; end
      end
    end

    context 'on #discard' do
      let!(:post) { Post.create! }

      it "skips validations gated on if: :discarded?" do
        expect(post).not_to receive(:guarded_validation)
        expect(post.discard).to be true
      end

      it "does not run before_discard gated on if: :discarded?" do
        expect(post).not_to receive(:guarded_before_discard)
        expect(post.discard).to be true
      end

      it "runs after_discard gated on if: :discarded?" do
        expect(post).to receive(:guarded_after_discard)
        expect(post.discard).to be true
      end
    end

    context 'on #undiscard' do
      let!(:post) { Post.create! discarded_at: Time.now }

      it "runs before_undiscard gated on if: :discarded?" do
        expect(post).to receive(:guarded_before_undiscard)
        expect(post.undiscard).to be true
      end

      it "does not run after_undiscard gated on if: :discarded?" do
        expect(post).not_to receive(:guarded_after_undiscard)
        expect(post.undiscard).to be true
      end
    end
  end

  describe 'when a callback raises during discard' do
    with_model :Post, scope: :all do
      table do |t|
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        after_discard :boom

        def boom
          raise "boom"
        end
      end
    end

    let!(:post) { Post.create! }

    it "propagates the exception" do
      expect { post.discard }.to raise_error("boom")
    end

    it "rolls back the DB write" do
      expect { post.discard rescue nil }
        .not_to change { Post.find(post.id).discarded? }
    end

    it "clears dirty tracking on the rolled-back attribute" do
      post.discard rescue nil
      expect(post.discarded_at_changed?).to be false
    end

    describe '#discard!' do
      it "propagates the callback exception" do
        expect { post.discard! }.to raise_error("boom")
      end
    end
  end

  describe 'when a callback raises during undiscard' do
    with_model :Post, scope: :all do
      table do |t|
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model
        after_undiscard :boom

        def boom
          raise "boom"
        end
      end
    end

    let!(:post) { Post.create!(discarded_at: Time.current) }

    it "propagates the exception" do
      expect { post.undiscard }.to raise_error("boom")
    end

    it "rolls back the DB write" do
      expect { post.undiscard rescue nil }
        .not_to change { Post.find(post.id).discarded? }
    end

    it "clears dirty tracking on the rolled-back attribute" do
      post.undiscard rescue nil
      expect(post.discarded_at_changed?).to be false
    end
  end

  describe 'discarding a record that is not backed by a row' do
    with_model :Post, scope: :all do
      table do |t|
        t.string :title
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model

        class_attribute :reject_discard, default: false
        class_attribute :discard_notifications, default: 0

        before_discard :guard_business_rules
        after_discard :notify_reviewers
        before_undiscard :guard_undiscard_rules
        after_undiscard :notify_reviewers

        def guard_business_rules
          throw :abort if self.class.reject_discard
        end

        def guard_undiscard_rules
          throw :abort if self.class.reject_discard
        end

        def notify_reviewers
          self.class.discard_notifications += 1
        end
      end
    end

    let!(:existing_post) { Post.create!(title: "the row that must survive") }

    def table_rows
      Post.unscoped.order(:id).map { |row| [row.id, row.title, row.discarded_at] }
    end

    context "an unsaved record manually given an existing row's primary key" do
      let(:draft) { Post.new(title: "unsaved draft", id: existing_post.id) }

      it "still looks like a new record" do
        expect(draft).to be_new_record
        expect(draft.id).to eq(existing_post.id)
      end

      describe '#discard' do
        it "returns false" do
          expect(draft.discard).to be false
        end

        it "runs no discard callbacks" do
          expect(draft).not_to receive(:guard_business_rules)
          expect(draft).not_to receive(:notify_reviewers)
          draft.discard
        end

        it "does not insert a row" do
          expect { draft.discard }.not_to change { Post.unscoped.count }
        end

        it "does not modify the existing row sharing its primary key" do
          expect { draft.discard }.not_to change { table_rows }
          expect(Post.find(existing_post.id)).to be_kept
        end

        it "leaves the object itself untouched" do
          draft.discard
          expect(draft).to be_new_record
          expect(draft.discarded_at).to be_nil
          expect(draft).not_to be_discarded
        end
      end

      describe '#discard!' do
        it "raises Discard::RecordNotDiscarded" do
          expect { draft.discard! }
            .to raise_error(Discard::RecordNotDiscarded, "A new record cannot be discarded")
        end

        it "reports the record that could not be discarded" do
          draft.discard!
        rescue Discard::RecordNotDiscarded => error
          expect(error.record).to be(draft)
        end

        it "runs no discard callbacks" do
          expect(draft).not_to receive(:guard_business_rules)
          expect(draft).not_to receive(:notify_reviewers)
          expect { draft.discard! }.to raise_error(Discard::RecordNotDiscarded)
        end

        it "neither inserts a row nor touches the row sharing its primary key" do
          expect {
            expect { draft.discard! }.to raise_error(Discard::RecordNotDiscarded)
          }.not_to change { [Post.unscoped.count, table_rows, Post.find(existing_post.id).discarded?] }
        end
      end
    end

    context "an instance whose row was hard-deleted underneath it" do
      let(:stale_post) { Post.create!(title: "about to vanish") }

      before do
        Post.unscoped.where(id: stale_post.id).delete_all
        expect(Post.unscoped.where(id: stale_post.id)).to be_empty
      end

      it "does not consider itself a new or destroyed record" do
        expect(stale_post).not_to be_new_record
        expect(stale_post).not_to be_destroyed
      end

      describe '#discard' do
        it "returns false instead of raising" do
          expect(stale_post.discard).to be false
        end

        it "runs no discard callbacks" do
          expect(stale_post).not_to receive(:guard_business_rules)
          expect(stale_post).not_to receive(:notify_reviewers)
          stale_post.discard
        end

        it "does not recreate the deleted row" do
          expect { stale_post.discard }.not_to change { Post.unscoped.where(id: stale_post.id).count }
          expect(Post.unscoped.where(id: stale_post.id)).to be_empty
        end

        it "does not insert any other row" do
          expect { stale_post.discard }.not_to change { Post.unscoped.count }
        end
      end

      describe '#discard!' do
        it "raises Discard::RecordNotDiscarded" do
          expect { stale_post.discard! }
            .to raise_error(Discard::RecordNotDiscarded, "A deleted record cannot be discarded")
        end

        it "runs no discard callbacks" do
          expect(stale_post).not_to receive(:guard_business_rules)
          expect(stale_post).not_to receive(:notify_reviewers)
          expect { stale_post.discard! }.to raise_error(Discard::RecordNotDiscarded)
        end

        it "does not recreate the deleted row" do
          expect {
            expect { stale_post.discard! }.to raise_error(Discard::RecordNotDiscarded)
          }.not_to change { Post.unscoped.count }
          expect(Post.unscoped.where(id: stale_post.id)).to be_empty
        end
      end
    end

    context "an instance hard-deleted through #delete" do
      let(:deleted_post) { Post.create!(title: "hard deleted").tap(&:delete) }

      it "is destroyed" do
        expect(deleted_post).to be_destroyed
      end

      describe '#discard' do
        it "returns false" do
          expect(deleted_post.discard).to be false
        end

        it "runs no discard callbacks and writes nothing" do
          expect(deleted_post).not_to receive(:guard_business_rules)
          expect(deleted_post).not_to receive(:notify_reviewers)
          expect { deleted_post.discard }.not_to change { Post.unscoped.count }
        end
      end

      describe '#discard!' do
        it "raises Discard::RecordNotDiscarded without writing" do
          expect { deleted_post.discard! }
            .to raise_error(Discard::RecordNotDiscarded, "A destroyed record cannot be discarded")
          expect(Post.unscoped.where(id: deleted_post.id)).to be_empty
        end
      end
    end

    context "a record loaded from the database" do
      let(:post) { Post.find(existing_post.id) }

      before do
        Post.reject_discard = false
        Post.discard_notifications = 0
      end

      context "when a business callback refuses the discard" do
        before { Post.reject_discard = true }

        it "returns false and keeps the row" do
          expect(post.discard).to be false
          expect(post.reload).to be_kept
        end

        it "runs no after_discard callback and sends no notification" do
          expect(post).not_to receive(:notify_reviewers)
          expect { post.discard }.not_to change { Post.discard_notifications }.from(0)
        end

        describe '#discard!' do
          it "preserves the real callback failure reason" do
            expect { post.discard! }
              .to raise_error(Discard::RecordNotDiscarded, "Failed to discard the record")
          end

          it "does not write, notify or run the success callback" do
            expect(post).not_to receive(:notify_reviewers)
            expect {
              expect { post.discard! }.to raise_error(Discard::RecordNotDiscarded)
            }.not_to change { [Post.unscoped.where(id: post.id).pick(:discarded_at), Post.discard_notifications] }
            expect(post.reload).to be_kept
          end
        end
      end

      context "once the business callback allows the discard" do
        it "changes state and notifies exactly once" do
          expect { expect(post.discard).to be true }
            .to change { Post.discard_notifications }.from(0).to(1)
          expect(post).to be_discarded
          expect(post.reload.discarded_at).to be_present
        end

        it "runs each discard callback exactly once" do
          expect(post).to receive(:guard_business_rules).once.and_call_original
          expect(post).to receive(:notify_reviewers).once.and_call_original
          expect(post.discard).to be true
        end

        it "is a no-op when discarded a second time" do
          expect(post.discard).to be true
          discarded_at = post.reload.discarded_at

          expect(post.discard).to be false
          expect {
            expect { post.discard! }
              .to raise_error(Discard::RecordNotDiscarded, "A discarded record cannot be discarded")
          }.not_to change { [post.reload.discarded_at, Post.discard_notifications] }
          expect(post.reload.discarded_at).to eq(discarded_at)
          expect(Post.discard_notifications).to eq 1
        end
      end
    end

    context "an unsaved record when undiscarding" do
      let(:draft) { Post.new(title: "unsaved", id: existing_post.id, discarded_at: Time.current) }

      it "returns false and runs no callbacks" do
        expect(draft).not_to receive(:guard_undiscard_rules)
        expect(draft.undiscard).to be false
      end

      it "raises Discard::RecordNotUndiscarded from #undiscard!" do
        expect { draft.undiscard! }
          .to raise_error(Discard::RecordNotUndiscarded, "A new record cannot be undiscarded")
        expect(Post.find(existing_post.id)).to be_kept
      end
    end
  end
end
