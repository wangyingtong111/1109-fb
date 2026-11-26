# frozen_string_literal: true

require "test_helper"

# Parity port of the reference `describe('search')` option cases not covered by
# the golden/fuzz fixtures: field selection vs boosting, per-combinator empty
# queries, fuzzy/prefix as functions, boost_document skipping, search-time
# tokenizer/term-processor overrides, constructor-level defaults, and the
# wildcard corner cases.
class TestSearchOptions < Minitest::Test
  DOCUMENTS = [
    { "id" => 1, "title" => "Divina Commedia", "text" => "Nel mezzo del cammin di nostra vita" },
    { "id" => 2, "title" => "I Promessi Sposi", "text" => "Quel ramo del lago di Como", "lang" => "it",
      "category" => "fiction" },
    { "id" => 3, "title" => "Vita Nova", "text" => "In quella parte del libro della mia memoria",
      "category" => "poetry" }
  ].freeze

  def build
    ms = MiniFTS.new(fields: %w[title text], store_fields: %w[lang category])
    ms.add_all(DOCUMENTS)
    ms
  end

  def test_searches_only_selected_fields_even_if_other_fields_are_boosted
    results = build.search("vita", fields: ["title"], boost: { "text" => 2 })
    assert_equal 1, results.length
    assert_equal 3, results.first[:id]
  end

  def test_returns_empty_results_for_empty_search_across_all_combinators
    ms = build
    assert_equal [], ms.search("")
    assert_equal [], ms.search("", combine_with: "OR")
    assert_equal [], ms.search("", combine_with: "AND")
    assert_equal [], ms.search("", combine_with: "AND_NOT")
  end

  def test_assigns_weight_lower_than_exact_to_a_prefix_and_fuzzy_match
    ms = MiniFTS.new(fields: ["text"])
    ms.add_all([
                 { "id" => 1, "text" => "Poi che la gente poverella crebbe" },
                 { "id" => 2, "text" => "Deus, venerunt gentes" }
               ])
    exact = ms.search("gente")
    combined = ms.search("gente", fuzzy: 0.2, prefix: true)
    assert_equal([1, 2], combined.map { |r| r[:id] })
    assert_in_delta exact[0][:score], combined[0][:score], 1e-9
    assert_equal ["text"], combined[1][:match]["gentes"]
  end

  def test_accepts_a_function_to_compute_fuzzy_and_prefix_options_from_term
    fuzzy_calls = []
    prefix_calls = []
    fuzzy = lambda do |term, i, terms|
      fuzzy_calls << [term, i, terms]
      term.length > 4 ? 2 : false
    end
    prefix = lambda do |term, i, terms|
      prefix_calls << [term, i, terms]
      term.length > 4
    end
    results = build.search("quel comedia", fuzzy: fuzzy, prefix: prefix)

    assert_equal [["quel", 0, %w[quel comedia]], ["comedia", 1, %w[quel comedia]]], fuzzy_calls
    assert_equal [["quel", 0, %w[quel comedia]], ["comedia", 1, %w[quel comedia]]], prefix_calls
    assert_equal([2, 1], results.map { |r| r[:id] })
  end

  def test_skips_document_if_boost_document_returns_a_falsy_value
    ms = build
    without_boost = ms.search("vita").map { |r| r[:id] }
    assert_includes without_boost, 3

    results = ms.search("vita", boost_document: ->(id, _term, _stored) { id == 3 ? nil : 1 })
    refute_includes results.map { |r| r[:id] }, 3
  end

  def test_uses_a_specific_search_time_tokenizer_if_specified
    tokenize = ->(string) { string.split("X") }
    results = build.search("divinaXcommedia", tokenize: tokenize)
    assert_operator results.length, :>, 0
    assert_equal [1], results.map { |r| r[:id] }.sort
  end

  def test_uses_a_specific_search_time_term_processing_function_if_specified
    process = ->(string) { string.gsub("1", "i").gsub("4", "a").downcase }
    results = build.search("d1v1n4", process_term: process)
    assert_operator results.length, :>, 0
    assert_equal [1], results.map { |r| r[:id] }.sort
  end

  def test_rejects_falsy_terms_at_search_time
    process = ->(term) { term == "quel" ? nil : term }
    results = build.search("quel commedia", process_term: process)
    assert_equal [1], results.map { |r| r[:id] }.sort
  end

  def test_allows_process_term_to_expand_a_single_term_at_search_time
    process = ->(string) { string == "divinacommedia" ? %w[divina commedia] : string }
    results = build.search("divinacommedia", process_term: process)
    assert_equal [1], results.map { |r| r[:id] }.sort
  end

  def test_allows_a_default_filter_upon_instantiation
    ms = MiniFTS.new(
      fields: %w[title text],
      store_fields: ["category"],
      search_options: { filter: ->(r) { r["category"] == "poetry" } }
    )
    ms.add_all(DOCUMENTS)
    results = ms.search("del")
    assert_equal 1, results.length
    assert(results.all? { |r| r["category"] == "poetry" })
  end

  def test_bm25_defaults_are_taken_from_constructor_search_options
    docs = [
      { "id" => 1, "text" => "something very very very cool" },
      { "id" => 2, "text" => "something cool" }
    ]
    with_default = MiniFTS.new(fields: ["text"], search_options: { bm25: { k: 1, b: 0.7, d: 0.5 } })
    with_default.add_all(docs)
    per_call = MiniFTS.new(fields: ["text"])
    per_call.add_all(docs)

    assert_in_delta per_call.search("very", bm25: { k: 1, b: 0.7, d: 0.5 }).first[:score],
                    with_default.search("very").first[:score], 1e-9
  end

  def test_wildcard_matches_all_including_null_field_and_star_is_a_normal_term
    ms = MiniFTS.new(fields: ["text"], store_fields: ["cool"])
    ms.add_all([
                 { "id" => 1, "text" => "something cool", "cool" => true },
                 { "id" => 2, "text" => "something else", "cool" => false },
                 { "id" => 3, "text" => nil, "cool" => true }
               ])

    assert_equal [], ms.search("*")       # "*" is just a normal term
    assert_equal [], ms.search("")        # empty string is a normal query
    assert_equal([1, 2, 3], ms.search(MiniFTS::WILDCARD).map { |r| r[:id] })

    results = ms.search(MiniFTS::WILDCARD, filter: ->(x) { x["cool"] }, boost_document: ->(id, *_) { id })
    assert_equal([3, 1], results.map { |r| r[:id] })
  end

  # Four documents sharing one term; the wildcard callback's per-document
  # returns are 1, 0, nil, 2 in id order. A boost_document only ever sees the
  # empty-string term for a wildcard query, so this callable returns 1 for the
  # ordinary "oak" term and the scripted wildcard value otherwise.
  OAK_WILDCARD_RETURNS = [1, 0, nil, 2].freeze

  def oak_index
    ms = MiniFTS.new(fields: ["text"], store_fields: ["text"])
    ms.add_all((1..4).map { |id| { "id" => id, "text" => "oak" } })
    ms
  end

  def oak_wildcard_boost(returns = OAK_WILDCARD_RETURNS)
    ->(id, term, stored) { term == "" ? returns[id - 1] : 1 }
  end

  def oak_and_not_tree
    { combine_with: "AND_NOT", queries: ["oak", MiniFTS::WILDCARD] }
  end

  def test_gate_wildcard_makes_boost_document_decide_wildcard_candidates_in_and_not
    ms = oak_index
    plain = ms.search("oak")

    results = ms.search(oak_and_not_tree, boost_document: oak_wildcard_boost, gate_wildcard: true)
    # Truthy wildcard returns keep the document in the subtraction set (1, 4);
    # falsy ones (0, nil) spare it, so only documents 2 and 3 survive.
    assert_equal([2, 3], results.map { |r| r[:id] })

    results.each do |result|
      reference = plain.find { |r| r[:id] == result[:id] }
      assert_in_delta reference[:score], result[:score], 1e-9
      assert_equal ["oak"], result[:terms]
    end
  end

  def test_gate_wildcard_still_lets_the_top_level_filter_act_on_the_final_results
    ms = oak_index
    plain = ms.search("oak")

    results = ms.search(
      oak_and_not_tree,
      boost_document: oak_wildcard_boost,
      gate_wildcard: true,
      filter: ->(result) { result[:id] < 3 }
    )

    assert_equal([2], results.map { |r| r[:id] })
    assert_in_delta plain.find { |r| r[:id] == 2 }[:score], results.first[:score], 1e-9
  end

  def test_without_gate_wildcard_the_wildcard_return_values_only_set_scores
    ms = oak_index
    calls = []
    boost = lambda do |id, term, stored|
      calls << [id, term, stored]
      oak_wildcard_boost.call(id, term, stored)
    end

    # Every document is in the wildcard subtraction set regardless of the 0/nil
    # returns, so AND_NOT removes all of them — the pre-existing behaviour.
    assert_equal [], ms.search(oak_and_not_tree, boost_document: boost)

    # The wildcard callback keeps its empty-string term and stored-fields args.
    (1..4).each do |id|
      assert_includes calls, [id, "", { "text" => "oak" }]
    end

    # A standalone wildcard still matches every document, ordered by the
    # returned boost; a falsy 0 does not exclude a document without the gate.
    # (nil is the one pre-existing exception: it crashes later scoring, so the
    # no-gate path is exercised with 0 here instead.)
    zero_returns = [1, 0, 0, 2]
    zero_boost = ->(id, term, _stored) { term == "" ? zero_returns[id - 1] : 1 }
    results = ms.search(MiniFTS::WILDCARD, boost_document: zero_boost)
    assert_equal([1, 2, 3, 4], results.map { |r| r[:id] }.sort)
    assert_equal [4, 1], results.first(2).map { |r| r[:id] }
  end

  def test_gate_wildcard_also_filters_standalone_wildcard_queries
    ms = oak_index
    # The gate excludes the 0/nil entries (documents 2 and 3); the survivors are
    # ordered by their returned boosts, document 4 (2) ahead of document 1 (1).
    results = ms.search(MiniFTS::WILDCARD, boost_document: oak_wildcard_boost, gate_wildcard: true)

    assert_equal([4, 1], results.map { |r| r[:id] })
    assert_equal([2.0, 1.0], results.map { |r| r[:score] })
  end

  # Reference "computes a meaningful score when fields are named liked default
  # properties of object" — trivially safe in Ruby (plain Hash), ported for
  # regression parity.
  def test_meaningful_score_for_fields_named_like_object_defaults
    ms = MiniFTS.new(fields: ["constructor"])
    ms.add("id" => 1, "constructor" => "something")
    ms.add("id" => 2, "constructor" => "something else")
    results = ms.search("something")
    assert_equal 2, results.length
    results.each do |r|
      assert_kind_of Float, r[:score]
      assert r[:score].finite?
    end
  end
end
