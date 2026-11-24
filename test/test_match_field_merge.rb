# frozen_string_literal: true

require "test_helper"

class TestMatchFieldMerge < Minitest::Test
  def build_index
    docs = [
      { "id" => 1, "title" => "oak", "text" => "oak" },
      { "id" => 2, "title" => "oak", "text" => "oak reject" }
    ]
    ms = MiniFTS.new(fields: %w[title text])
    ms.add_all(docs)
    ms
  end

  def combined_query
    {
      combine_with: "AND_NOT",
      queries: [
        {
          combine_with: "AND",
          queries: [
            {
              combine_with: "OR",
              queries: [
                { fields: ["text"], prefix: true, queries: ["oa"] },
                { fields: ["title"], queries: ["oak"] }
              ]
            },
            { fields: ["title"], queries: ["oak"] }
          ]
        },
        "reject"
      ]
    }
  end

  def test_lists_each_merged_field_once_in_index_order
    result = build_index.search(combined_query).first

    assert_equal({ "oak" => %w[title text] }, result[:match])
  end

  def test_keeps_terms_coverage_and_score_on_the_previous_path
    result = build_index.search(combined_query).first

    assert_equal ["oak"], result[:terms]
    assert_equal %w[oa oak], result[:query_terms]
    assert_in_delta 1.298523133188762, result[:score], 1e-9
  end

  def test_leaves_single_field_match_arrays_unchanged
    result = build_index.search("oak", fields: ["title"]).first

    assert_equal({ "oak" => ["title"] }, result[:match])
  end
end
