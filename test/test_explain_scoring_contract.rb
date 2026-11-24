# frozen_string_literal: true

require "test_helper"

# Contract coverage for on-demand score explanation (`explain: true`).
#
# The canonical fixture mirrors the task scenario: the document's title "oak"
# matches exactly while its text reaches "oaks" through the prefix route, and
# the outer AND additionally requires "maple". The explanation must keep the
# two oak sources distinct, enumerate three field/route contributions, expose
# the coverage multiplier, and reproduce result[:score] by recomputation.
class TestExplainScoringContract < Minitest::Test
  def build_index
    ms = MiniFTS.new(fields: %w[title text], store_fields: %w[title])
    ms.add_all([
                 { "id" => 1, "title" => "oak", "text" => "oaks maple" },
                 { "id" => 2, "title" => "Stone", "text" => "a stony path" },
                 { "id" => 3, "title" => "Tea", "text" => "green tea" }
               ])
    ms
  end

  def canonical_query
    { combine_with: "AND", queries: [{ queries: ["oak"], prefix: true }, "maple"] }
  end

  def canonical_explanation
    build_index.search(canonical_query, explain: true).first[:explain]
  end

  def test_three_contributions_distinguish_the_two_oak_sources
    contributions = canonical_explanation[:contributions]

    assert_equal 3, contributions.length

    oak = contributions.select { |c| c[:query_term] == "oak" }
    assert_equal 2, oak.length

    exact_title = oak.find { |c| c[:route] == "exact" }
    prefix_text = oak.find { |c| c[:route] == "prefix" }

    refute_nil exact_title
    assert_equal "oak", exact_title[:term]
    assert_equal "title", exact_title[:field]

    refute_nil prefix_text
    assert_equal "oaks", prefix_text[:term]
    assert_equal "text", prefix_text[:field]

    maple = contributions.find { |c| c[:query_term] == "maple" }
    assert_equal "exact", maple[:route]
    assert_equal "maple", maple[:term]
    assert_equal "text", maple[:field]
  end

  def test_coverage_multiplier_is_two
    explanation = canonical_explanation

    assert_equal 2, explanation[:coverage]
    assert_equal 2, explanation[:coverage_multiplier]
  end

  def test_score_recomputes_from_contributions_and_coverage
    result = build_index.search(canonical_query, explain: true).first
    explanation = result[:explain]

    naive_sum = explanation[:contributions].inject(0.0) { |sum, c| sum + c[:score] }
    recomputed = naive_sum * explanation[:coverage]

    assert_in_delta result[:score], recomputed, 1e-12
    assert_in_delta result[:score], explanation[:score], 1e-12
    assert_in_delta explanation[:raw_score], naive_sum, 1e-12
  end

  def test_explain_off_keeps_original_shape
    # The exact task/probe fixture: a single document.
    ms = MiniFTS.new(fields: %w[title text])
    ms.add("id" => 1, "title" => "oak", "text" => "oaks maple")
    query = { combine_with: "AND", queries: [{ queries: ["oak"], prefix: true }, "maple"] }

    plain = ms.search(query).first
    explicit_off = ms.search(query, explain: false).first

    refute plain.key?(:explain)
    refute plain.key?(:explanation)
    assert_equal plain.keys, explicit_off.keys
    assert_equal 2.027155068671851, plain[:score]
  end

  def test_explanation_alias_and_convenience_method
    ms = build_index
    result = ms.search("tea", explain: true).first

    assert_same result[:explain], result[:explanation]
    assert(ms.explain("tea").all? { |r| r.key?(:explain) })
  end

  def test_field_boost_is_recorded
    contribution = build_index.search("tea", boost: { "text" => 3 }, explain: true)
                              .first[:explain][:contributions]
                              .find { |c| c[:field] == "text" }

    assert_equal 3, contribution[:field_boost]
    assert_in_delta contribution[:score],
                    contribution[:bm25_score] * contribution[:weight] *
                    contribution[:term_boost] * contribution[:field_boost] *
                    contribution[:document_boost],
                    1e-12
  end

  def test_filtered_out_hits_get_no_explanation_but_kept_hit_does
    results = build_index.search("tea", explain: true, filter: ->(r) { r[:id] == 3 })

    assert_equal([3], results.map { |r| r[:id] })
    assert results.first.key?(:explain)
  end

  def test_constructor_default_can_be_disabled_per_call
    ms = MiniFTS.new(fields: %w[title text], search_options: { explain: true })
    ms.add("id" => 9, "title" => "oak", "text" => "pine")

    assert ms.search("oak").first.key?(:explain)
    refute ms.search("oak", explain: false).first.key?(:explain)
  end

  def test_wildcard_explanation_recomputes
    result = build_index.search(MiniFTS::WILDCARD, explain: true).find { |r| r[:id] == 1 }
    explanation = result[:explain]

    assert_equal 1, explanation[:coverage]
    assert_equal 1, explanation[:contributions].length
    assert_in_delta result[:score],
                    explanation[:contributions].first[:score] * explanation[:coverage],
                    1e-12
  end

  def test_grouped_query_nodes_cover_every_contribution
    explanation = build_index.search("oak maple", combine_with: "AND", explain: true).first[:explain]

    assert_equal(%w[oak maple], explanation[:query_nodes].map { |n| n[:query_term] })
    flat_count = explanation[:query_nodes].inject(0) { |count, n| count + n[:contributions].length }
    assert_equal explanation[:contributions].length, flat_count
  end

  def test_common_aliases_are_available_on_hit_and_explanation
    result = build_index.search("tea", explain: true).first

    assert_same result[:explain], result[:explanation]
    assert_same result[:explain], result[:score_explanation]
    assert_equal result[:explain][:coverage], result[:coverage]
    assert_same result[:explain][:contributions], result[:contributions]
    assert_equal "exact", result[:explain][:contributions].first[:match]
    assert_equal :exact, result[:explain][:contributions].first[:route_sym]
  end

  def test_block_receives_returned_hit_and_explanation_once
    seen = []
    results = build_index.search("tea", explain: true) do |hit, explanation|
      seen << [hit[:id], explanation[:coverage]]
    end

    assert_equal results.map { |hit| [hit[:id], hit[:explain][:coverage]] }, seen
    assert(seen.all? { |_id, coverage| coverage == 1 })
  end

  def test_explain_score_returns_first_explanation
    explanation = build_index.explain_score("tea")

    assert_equal 1, explanation[:coverage]
    assert_equal %w[text title], explanation[:contributions].map { |c| c[:field] }.uniq.sort
    assert(explanation[:contributions].all? { |c| c[:route] == "exact" })
  end
end
