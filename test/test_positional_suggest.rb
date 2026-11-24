# frozen_string_literal: true

require "test_helper"

# Coverage for the positional completion mode of `auto_suggest`
# (`positional: true`). A positional suggestion keeps the query's word order
# and repeated slots: each query term becomes a slot filled by exactly one
# indexed term, every completed phrase is reported once per contributing
# document, and each suggestion names the documents it was derived from. The
# default suggestion mode is exercised in test_autosuggest.rb and must stay
# unchanged.
class TestPositionalSuggest < Minitest::Test
  def build(documents, options = {})
    ms = MiniFTS.new({ fields: ["text"], store_fields: ["text"] }.merge(options))
    ms.add_all(documents)
    ms
  end

  def by_suggestion(results)
    results.each_with_object({}) { |result, map| map[result[:suggestion]] = result }
  end

  def test_completes_each_query_position_keeping_word_order_and_repeated_slots
    ms = build([
                 { "id" => 1, "text" => "red apple applet" },
                 { "id" => 2, "text" => "red apple" }
               ])

    results = ms.auto_suggest("red red app", positional: true)
    suggestions = by_suggestion(results)

    assert_equal ["red red apple", "red red applet"], suggestions.keys.sort
    assert_equal 2, results.length

    apple = suggestions["red red apple"]
    assert_equal %w[red red apple], apple[:terms]
    assert_equal [1, 2], apple[:documents]

    applet = suggestions["red red applet"]
    assert_equal %w[red red applet], applet[:terms]
    assert_equal [1], applet[:documents]

    results.each do |result|
      assert_kind_of Float, result[:score]
      assert result[:score].finite?
    end
    scores = results.map { |result| result[:score] }
    assert_equal scores.sort.reverse, scores
  end

  def test_default_and_positional_modes_both_suggest_a_simple_prefix_completion
    ms = build([{ "id" => 1, "text" => "red apple" }])

    positional = ms.auto_suggest("red app", positional: true)
    default = ms.auto_suggest("red app")

    assert_equal ["red apple"], positional.map { |result| result[:suggestion] }
    assert_equal %w[red apple], positional.first[:terms]
    assert_equal [1], positional.first[:documents]

    assert_equal ["red apple"], default.map { |result| result[:suggestion] }
    refute default.first.key?(:documents)
  end

  def test_each_document_contributes_a_given_phrase_at_most_once
    ms = build([{ "id" => 1, "text" => "apple apple applet" }])

    suggestions = by_suggestion(ms.auto_suggest("app", positional: true))

    assert_equal [1], suggestions["apple"][:documents]
    assert_equal [1], suggestions["applet"][:documents]
  end

  def test_a_document_missing_a_completion_for_any_slot_is_excluded
    ms = build([
                 { "id" => 1, "text" => "red apple" },
                 { "id" => 2, "text" => "green apple" }
               ])

    results = ms.auto_suggest("red app", positional: true)

    assert_equal [["red apple", [1]]], results.map { |result| [result[:suggestion], result[:documents]] }
  end

  def test_returns_no_suggestions_for_empty_or_unmatched_queries
    ms = build([{ "id" => 1, "text" => "red apple" }])

    assert_equal [], ms.auto_suggest("", positional: true)
    assert_equal [], ms.auto_suggest("banana", positional: true)
    assert_equal [], ms.auto_suggest("red banana", positional: true)
  end

  def test_can_be_enabled_by_default_in_the_constructor
    ms = build(
      [
        { "id" => 1, "text" => "red apple applet" },
        { "id" => 2, "text" => "red apple" }
      ],
      auto_suggest_options: { positional: true }
    )

    results = ms.auto_suggest("red red app")

    assert(results.all? { |result| result.key?(:documents) })
    suggestions = by_suggestion(results)
    assert_equal [1, 2], suggestions["red red apple"][:documents]
    assert_equal [1], suggestions["red red applet"][:documents]
  end

  def test_respects_custom_filters
    ms = build(
      [
        { "id" => 1, "text" => "red apple applet", "keep" => true },
        { "id" => 2, "text" => "red apple", "keep" => false }
      ],
      store_fields: %w[text keep]
    )

    suggestions = by_suggestion(
      ms.auto_suggest("red red app", positional: true, filter: ->(result) { result["keep"] })
    )

    assert_equal [1], suggestions["red red apple"][:documents]
    assert_equal [1], suggestions["red red applet"][:documents]
  end

  def test_supports_fuzzy_completion_per_slot
    ms = build([{ "id" => 1, "text" => "red apple" }])

    # 0.7 * "apl".length rounds to an edit distance of 2, reaching "apple".
    results = ms.auto_suggest("red apl", positional: true, fuzzy: 0.7)

    assert_equal ["red apple"], results.map { |result| result[:suggestion] }
    assert_equal [1], results.first[:documents]
  end
end
