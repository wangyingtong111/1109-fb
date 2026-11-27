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

  describe '.discard_all with result: true' do
    with_model :Product, scope: :all do
      table do |t|
        t.string :name
        t.boolean :held, default: false
        t.datetime :discarded_at
        t.timestamps null: false
      end

      model do
        include Discard::Model

        before_discard :reject_when_held
        after_discard :record_discard_call

        def self.discard_calls
          @discard_calls ||= []
        end

        def reject_when_held
          throw :abort if held
        end

        def record_discard_call
          self.class.discard_calls << name
        end
      end
    end

    let!(:a) { Product.create!(name: "A") }
    let!(:b) { Product.create!(name: "B", held: true) }
    let!(:c) { Product.create!(name: "C", discarded_at: Time.utc(2024, 1, 1)) }

    before { Product.discard_calls.clear }

    let(:result) { Product.order(:id).discard_all(result: true) }

    it "classifies each row by what the database holds afterwards" do
      result = Product.order(:id).discard_all(result: true)

      expect(result.discarded.map(&:id)).to eq([a.id])
      expect(result.rejected.map(&:id)).to eq([b.id])
      expect(result.already_discarded.map(&:id)).to eq([c.id])

      expect(Product.find(a.id)).to be_discarded
      expect(Product.find(b.id)).not_to be_discarded
      expect(Product.find(c.id)).to be_discarded
    end

    it "exposes one entry per enumerated row" do
      result = Product.order(:id).discard_all(result: true)

      expect(result.ids).to eq([a.id, b.id, c.id])
      expect(result.size).to eq(3)
      expect(result.map(&:status)).to eq([:discarded, :rejected, :already_discarded])
    end

    it "locates entries by record id" do
      result = Product.order(:id).discard_all(result: true)

      expect(result[a.id].status).to eq(:discarded)
      expect(result[b.id].status).to eq(:rejected)
      expect(result[c.id].status).to eq(:already_discarded)
      expect(result[b.id].id).to eq(b.id)
      expect(result[123456]).to be_nil
    end

    it "records the id of every entry" do
      result = Product.order(:id).discard_all(result: true)

      expect(result.entries.map(&:id)).to eq([a.id, b.id, c.id])
      expect(result.statuses).to eq(
        a.id => :discarded,
        b.id => :rejected,
        c.id => :already_discarded
      )
    end

    it "does not run callbacks for rows that were already discarded" do
      Product.order(:id).discard_all(result: true)

      expect(Product.discard_calls).to eq(["A"])
    end

    it "keeps the rejected row saleable and continues the batch" do
      result = Product.order(:id).discard_all(result: true)

      expect(b.reload).not_to be_discarded
      expect(b).to be_kept
      expect(result.rejected.first.record.reload).to be_kept
    end

    it "does not re-stamp the discard time of an already discarded row" do
      original = c.discarded_at

      Product.order(:id).discard_all(result: true)

      expect(Product.find(c.id).discarded_at).to eq(original)
    end

    it "runs the discard callbacks for the rows it changes" do
      result = Product.order(:id).discard_all(result: true)

      expect(Product.discard_calls).to eq(["A"])
      expect(result.discarded.first.record).to be_discarded
    end

    it "returns an empty result when no rows match" do
      result = Product.where(id: []).discard_all(result: true)

      expect(result).to be_empty
      expect(result.ids).to eq([])
      expect(result.statuses).to eq({})
    end

    it "leaves the legacy return value unchanged" do
      expect(Product.order(:id).discard_all.map(&:name)).to eq(["A", "B"])
    end

    it "classifies rejected rows as rejected rather than discarded when a save fails" do
      Product.where(id: b.id).update_all(held: false)
      allow_any_instance_of(Product).to receive(:update_attribute).and_return(false)

      result = Product.order(:id).discard_all(result: true)

      expect(result.map(&:status)).to eq([:rejected, :rejected, :already_discarded])
      expect(Product.find(a.id)).not_to be_discarded
    end

    context "with rows hidden by a default scope" do
      with_model :Archived, scope: :all do
        table do |t|
          t.string :name
          t.datetime :discarded_at
          t.timestamps null: false
        end

        model do
          include Discard::Model
          default_scope -> { kept }
        end
      end

      it "reports rows hidden by a default scope when the caller opts in" do
        hidden = Archived.create!(name: "archived", discarded_at: Time.utc(2024, 1, 1))
        kept = Archived.create!(name: "kept")

        result = Archived.with_discarded.discard_all(result: true)

        expect(result.ids).to eq([hidden.id, kept.id])
        expect(result[hidden.id].status).to eq(:already_discarded)
        expect(result[kept.id].status).to eq(:discarded)
      end

      it "uses the receiver's own enumeration as the row set" do
        hidden = Archived.create!(name: "archived", discarded_at: Time.utc(2024, 1, 1))

        result = Archived.discard_all(result: true)

        expect(result.ids).to eq([])
        expect(Archived.with_discarded.find(hidden.id)).to be_discarded
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
end
