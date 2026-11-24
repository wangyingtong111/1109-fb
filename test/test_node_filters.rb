# frozen_string_literal: true

require "test_helper"

class TestNodeFilters < Minitest::Test
  DOCUMENTS = [
    { "id" => 1, "text" => "oak pine", "dept" => "old" },
    { "id" => 2, "text" => "oak pine", "dept" => "new" }
  ].freeze

  def index
    ms = MiniFTS.new(fields: ["text"], store_fields: ["dept"])
    ms.add_all(DOCUMENTS)
    ms
  end

  def old_pine_filter
    ->(result) { result["dept"] == "old" }
  end

  def test_filters_a_composite_query_node_before_and_not
    query = {
      combine_with: "AND_NOT",
      queries: [
        "oak",
        { queries: ["pine"], filter: old_pine_filter }
      ]
    }

    results = index.search(query)

    assert_equal([2], results.map { |result| result[:id] })
    refute_includes results.first[:match].keys, "pine"
    assert_equal ["oak"], results.first[:terms]
  end

  def test_filters_a_leaf_query_node
    %i[query term].each do |leaf_key|
      leaf = { leaf_key => "pine", filter: old_pine_filter }
      query = { combine_with: "AND_NOT", queries: ["oak", leaf] }

      results = index.search(query)

      assert_equal([2], results.map { |result| result[:id] })
      refute_includes results.first[:match].keys, "pine"
    end
  end

  def test_filters_a_combination_node_after_its_children_merge
    seen = []
    node_filter = lambda do |result|
      seen << [result[:id], result[:match].keys.sort]
      result["dept"] == "old"
    end
    query = {
      combine_with: "AND",
      queries: %w[oak pine],
      filter: node_filter
    }

    results = index.search(query)

    assert_equal([1], results.map { |result| result[:id] })
    assert_equal [[1, %w[oak pine]], [2, %w[oak pine]]], seen
  end

  def test_outer_filter_still_receives_final_score_and_merged_evidence
    outer_calls = []
    query = {
      combine_with: "AND_NOT",
      queries: ["oak", { query: "pine", filter: old_pine_filter }]
    }

    results = index.search(query, filter: lambda do |result|
      outer_calls << result
      true
    end)

    assert_equal results.length, outer_calls.length
    outer = outer_calls.first
    assert_equal results.first[:score], outer[:score]
    assert_equal results.first[:match], outer[:match]
    assert_equal ["oak"], outer[:match].keys
    assert_equal "new", outer["dept"]
  end
end
