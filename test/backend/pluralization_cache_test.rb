require 'test_helper'

class I18nBackendPluralizationCacheTest < I18n::TestCase
  class Backend < I18n::Backend::Simple
    include I18n::Backend::Pluralization
  end

  # A backend whose online update can be paused (and aborted) while it runs.
  # The pause happens at the exact midpoint of an update: after the previously
  # cached plural rule has been dropped but before the new package (rule lambda
  # plus entries) becomes visible to readers.
  class OnlineBackend < Backend
    attr_accessor :pause_gate, :fail_update

    def store_translations(locale, data, options = I18n::EMPTY_HASH)
      raise I18n::InvalidLocaleData.new(nil, 'online update failed') if fail_update
      super(locale, data, options) do
        pause_gate.call if pause_gate
      end
    end
  end

  def setup
    super
    I18n.enforce_available_locales = false
    I18n.backend = backend_a

    # Two genuinely independent backends, each with its own :en plural rule.
    # A always uses the English-style rule, B always resolves to :other.
    backend_a.store_translations(:en,
      :i18n => { :plural => { :rule => english_rule } },
      :dogs => { :one => 'A old one', :other => 'A old other' })
    backend_b.store_translations(:en,
      :i18n => { :plural => { :rule => always_other_rule } },
      :dogs => { :one => 'B old one', :other => 'B old other' })
  end

  test "independent backends keep separate plural rules for the same locale" do
    assert_equal 'A old one',   backend_a.translate(:en, :dogs, :count => 1)
    assert_equal 'A old other', backend_a.translate(:en, :dogs, :count => 2)

    # B's rule always returns :other, even for count == 1.
    assert_equal 'B old other', backend_b.translate(:en, :dogs, :count => 1)
    assert_equal 'B old other', backend_b.translate(:en, :dogs, :count => 2)
  end

  test "updating rule and entries together never exposes new words with an old rule" do
    # Warm the cached rule first.
    assert_equal 'A old other', backend_a.translate(:en, :dogs, :count => 2)

    paused  = Queue.new
    release = Queue.new
    backend_a.pause_gate = -> do
      paused << true
      release.pop # block the updater until the readers have observed the state
    end

    updater = Thread.new do
      backend_a.store_translations(:en,
        :i18n => { :plural => { :rule => always_one_rule } },
        :dogs => { :one => 'A new one', :other => 'A new other' })
    end

    paused.pop # update is now blocked at its midpoint

    # Readers racing the in-flight update must see the complete old version.
    during_pause = Array.new(3) { backend_a.translate(:en, :dogs, :count => 2) }
    assert_equal ['A old other'] * 3, during_pause

    release << true
    updater.join

    # Once released the complete new version (new rule + new words) is visible.
    assert_equal 'A new one', backend_a.translate(:en, :dogs, :count => 2)
    assert_equal 'A new one', backend_a.translate(:en, :dogs, :count => 1)
  end

  test "a failed install keeps the previously installed rule instead of falling back to defaults" do
    assert_equal 'A old one',   backend_a.translate(:en, :dogs, :count => 1)
    assert_equal 'A old other', backend_a.translate(:en, :dogs, :count => 2)

    backend_a.fail_update = true
    assert_raises(I18n::InvalidLocaleData) do
      backend_a.store_translations(:en,
        :i18n => { :plural => { :rule => always_one_rule } },
        :dogs => { :one => 'A new one', :other => 'A new other' })
    end
    backend_a.fail_update = false

    # Still the old, explicitly installed rule - not the built-in default and
    # not the rule from the aborted package.
    assert_equal 'A old one',   backend_a.translate(:en, :dogs, :count => 1)
    assert_equal 'A old other', backend_a.translate(:en, :dogs, :count => 2)
  end

  test "a failed install on one backend does not change the rule of another backend" do
    backend_b.translate(:en, :dogs, :count => 1) # warm B's cache

    backend_a.fail_update = true
    assert_raises(I18n::InvalidLocaleData) do
      backend_a.store_translations(:en,
        :i18n => { :plural => { :rule => always_one_rule } },
        :dogs => { :one => 'A new one', :other => 'A new other' })
    end
    backend_a.fail_update = false

    # B is completely untouched and keeps its always-:other rule.
    assert_equal 'B old other', backend_b.translate(:en, :dogs, :count => 1)
    assert_equal 'B old other', backend_b.translate(:en, :dogs, :count => 2)
  end

  test "a text-only update takes effect while the cached plural rule is retained" do
    backend_b.translate(:en, :dogs, :count => 1) # warm B's cache

    backend_b.store_translations(:en,
      :dogs => { :one => 'B fresh one', :other => 'B fresh other' })

    # New words are visible ...
    assert_equal 'B fresh other', backend_b.translate(:en, :dogs, :count => 2)
    # ... while B's own always-:other rule is still in charge for count == 1.
    assert_equal 'B fresh other', backend_b.translate(:en, :dogs, :count => 1)
  end

  private

  def backend_a
    @backend_a ||= OnlineBackend.new
  end

  def backend_b
    @backend_b ||= Backend.new
  end

  def english_rule
    ->(count) { count == 1 ? :one : :other }
  end

  def always_other_rule
    ->(_count) { :other }
  end

  def always_one_rule
    ->(_count) { :one }
  end
end
