# frozen_string_literal: true

require "test_helper"

# Per-field query analysis: the index applies each field's own `process_term`
# rule, so a query term looked up under a field must be analyzed by *that*
# field's rule and may only match postings indexed in that same field. Coverage
# is still measured per original query term: alternative index terms derived
# from one source term all count as one coverage slot.
class TestFieldAnalysis < Minitest::Test
  DOCUMENTS = [
    { "id" => 1, "sku" => "C-7", "title" => "kit" },
    { "id" => 2, "sku" => "kit-alias", "title" => "kit" },
    { "id" => 3, "title" => "kit-alias kit" },
    { "id" => 4, "title" => "c-7" }
  ].freeze

  # Mirrors the rules the acceptance scenarios describe: the SKU rule folds
  # `C-7` to `c7`, the title rule expands `C-7` into `c-7` and `kit-alias`.
  def process_term
    lambda do |term, field = nil|
      lower = term.downcase
      case [field, lower]
      when ["sku", "c-7"], ["sku", "c7"]
        "c7"
      when ["title", "c-7"]
        %w[c-7 kit-alias]
      else
        lower
      end
    end
  end

  def build
    ms = MiniFTS.new(
      fields: %w[sku title],
      store_fields: %w[sku title],
      tokenize: ->(value, *_args) { value.split },
      process_term: process_term,
      auto_vacuum: false
    )
    ms.add_all(DOCUMENTS)
    ms
  end

  def ids_with_coverage(results)
    results.map { |r| [r[:id], r[:query_terms].size] }.sort
  end

  # --- normal: selecting a field selects its rule -------------------------

  def test_selecting_a_field_analyzes_the_query_with_that_fields_rule
    results = build.search("C-7", fields: ["sku"])
    assert_equal [[1, 1]], ids_with_coverage(results)
  end

  def test_does_not_match_a_derived_term_indexed_in_another_field
    # Record 2 holds the *title* rule's alternative (`kit-alias`) in its SKU
    # field; the SKU rule derives `c7`, so that posting must not count.
    results = build.search("C-7", fields: ["sku"])
    refute_includes results.map { |r| r[:id] }, 2
  end

  def test_an_explicit_per_field_flag_enables_it_without_naming_fields
    results = build.search("C-7", fields: ["sku"], per_field_analysis: true)
    assert_equal [[1, 1]], ids_with_coverage(results)
  end

  # --- conflict: AND across fields, one coverage slot per original term ----

  def test_and_across_fields_allows_each_original_term_to_match_a_different_field
    results = build.search("C-7 kit", fields: %w[sku title], combine_with: "AND")
    assert_equal [[1, 2], [3, 2]], ids_with_coverage(results)
  end

  def test_alternative_index_terms_of_one_original_term_count_as_one_coverage_slot
    results = build.search("C-7 kit", fields: %w[sku title], combine_with: "AND")

    record3 = results.find { |r| r[:id] == 3 }
    assert_equal ["C-7", "kit"], record3[:query_terms].sort
    # Record 3 matched only the title rule's alternative `kit-alias` for the
    # source term C-7, yet that still covers C-7 exactly once.
    assert_equal({ "kit-alias" => ["title"], "kit" => ["title"] }, record3[:match])

    record1 = results.find { |r| r[:id] == 1 }
    assert_equal({ "c7" => ["sku"], "kit" => ["title"] }, record1[:match])
    assert_equal ["C-7", "kit"], record1[:query_terms].sort
  end

  def test_alternative_index_terms_never_cross_into_another_field
    results = build.search("C-7 kit", fields: %w[sku title], combine_with: "AND")
    results.each do |result|
      result[:match].each do |term, fields|
        assert_equal 1, fields.length, "#{term} should only match one field"
        assert_includes(%w[sku title], fields.first)
      end
    end
    refute_includes results.map { |r| r[:id] }, 2
    refute_includes results.map { |r| r[:id] }, 4
  end

  def test_a_subquery_limited_to_one_field_uses_only_that_field_rule
    title_only = build.search("C-7 kit", fields: ["title"], combine_with: "AND")
    assert_equal [[3, 2]], ids_with_coverage(title_only)
  end

  def test_a_subquery_field_override_cascades_through_a_query_tree
    results = build.search(
      combine_with: "OR",
      queries: [{ fields: ["title"], combine_with: "AND", queries: ["C-7 kit"] }]
    )
    assert_equal [[3, 2]], ids_with_coverage(results)
  end

  def test_the_outer_filter_still_applies_to_a_field_limited_subquery
    tree = { combine_with: "OR", queries: [{ fields: ["title"], combine_with: "AND", queries: ["C-7 kit"] }] }

    unfiltered = build.search(tree)
    assert_equal [3], unfiltered.map { |r| r[:id] }

    filtered = build.search(tree, filter: ->(result) { result[:id] != 3 })
    assert_equal [], filtered
  end

  def test_a_term_analyzed_away_by_every_searched_field_drops_its_coverage_slot
    process = lambda do |term, _field = nil|
      term.downcase == "the" ? nil : term.downcase
    end
    ms = MiniFTS.new(fields: %w[sku title], tokenize: ->(value, *_args) { value.split }, process_term: process)
    ms.add_all(DOCUMENTS)
    results = ms.search("the kit", fields: %w[sku title], combine_with: "AND")
    assert_equal [[1, 1], [2, 1], [3, 1]], ids_with_coverage(results)
  end

  def test_accepts_a_one_argument_search_time_process_term
    ms = build
    results = ms.search("C-7", fields: ["sku"], process_term: ->(term) { term.downcase })
    # The one-argument rule cannot see the field, so it derives `c-7` from the
    # query instead of the SKU rule's `c7`; only the title posting matches.
    assert_equal [], results.map { |r| r[:id] }
    assert_equal [4], ms.search("C-7", fields: ["title"], process_term: ->(term) { term.downcase }).map { |r| r[:id] }.sort
  end

  # --- regression: default search path is untouched -----------------------

  def test_without_field_selection_the_current_single_analysis_is_used
    ms = build
    assert_equal [1, 2, 3], ms.search("kit").map { |r| r[:id] }.sort
    assert_equal [[1, 1], [2, 1], [3, 1]], ids_with_coverage(ms.search("kit"))
  end

  def test_explicitly_disabling_per_field_analysis_restores_the_single_analysis
    ms = build
    assert_equal [[1, 1], [2, 1], [3, 1]], ids_with_coverage(ms.search("kit", fields: ["title"], per_field_analysis: false))
  end

  def test_scores_are_unchanged_when_the_analysis_agrees_across_fields
    ms = build
    plain = ms.search("kit")
    per_field = ms.search("kit", fields: %w[sku title])
    assert_equal plain.map { |r| r[:id] }, per_field.map { |r| r[:id] }
    plain.zip(per_field) { |a, b| assert_in_delta a[:score], b[:score], 1e-12 }
  end

  def test_per_field_analysis_can_be_a_constructor_search_default
    ms = MiniFTS.new(
      fields: %w[sku title],
      tokenize: ->(value, *_args) { value.split },
      process_term: process_term,
      search_options: { per_field_analysis: true }
    )
    ms.add_all(DOCUMENTS)
    assert_equal [[1, 1]], ids_with_coverage(ms.search("C-7", fields: ["sku"]))
  end

  def test_a_constructor_default_can_be_switched_off_per_search
    ms = MiniFTS.new(
      fields: %w[sku title],
      tokenize: ->(value, *_args) { value.split },
      process_term: process_term,
      search_options: { per_field_analysis: true }
    )
    ms.add_all(DOCUMENTS)
    assert_equal [[1, 1], [2, 1], [3, 1]], ids_with_coverage(ms.search("kit", fields: ["title"], per_field_analysis: false))
  end
end
