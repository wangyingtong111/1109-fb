# frozen_string_literal: true

require "test_helper"
require "json"

# Highlight ranges: an opt-in record of the original-text character spans that
# matched a query. Ranges are collected at indexing time, kept in a side
# structure parallel to the inverted index, survive serialization, and are only
# attached to results when a search asks for them — the default result shape,
# score, and serialized index stay byte-identical to the upstream library.
class TestRanges < Minitest::Test
  # Ranges are half-open [start, stop) offsets into the stringified field text,
  # counted in characters from zero, attributed per field — without storing the
  # field values.
  def test_reports_per_field_half_open_character_ranges
    ms = MiniFTS.new(fields: %w[title text], ranges: true)
    ms.add("id" => 1, "title" => "oak", "text" => "茶 oak")

    result = ms.search("oak", ranges: true).first

    assert_equal({ "text" => [[2, 5]], "title" => [[0, 3]] }, result[:ranges])
  end

  def test_does_not_require_the_text_fields_to_be_stored
    ms = MiniFTS.new(fields: %w[title text], ranges: true)
    ms.add("id" => 1, "title" => "oak", "text" => "茶 oak")

    result = ms.search("oak", ranges: true).first

    refute result.key?("title")
    refute result.key?("text")
    assert_equal({ "text" => [[2, 5]], "title" => [[0, 3]] }, result[:ranges])
  end

  def test_default_search_keeps_its_shape_and_score
    without_ranges = MiniFTS.new(fields: %w[title text])
    with_ranges = MiniFTS.new(fields: %w[title text], ranges: true)
    [without_ranges, with_ranges].each do |ms|
      ms.add("id" => 1, "title" => "oak", "text" => "茶 oak")
    end

    plain = without_ranges.search("oak").first
    indexed = with_ranges.search("oak").first

    assert_equal(%i[id score terms query_terms match], plain.keys)
    assert_equal(plain.keys, indexed.keys)
    assert_equal(plain[:score], indexed[:score])
    assert_equal(plain, indexed)
    refute indexed.key?(:ranges)
  end

  # One original token expanded by process_term into several lookup terms hits
  # the same span; the span is returned only once.
  def test_multiple_derived_terms_from_one_token_share_a_single_range
    process = lambda do |term, _field = nil|
      term.downcase == "nyc" ? %w[new york] : term.downcase
    end
    ms = MiniFTS.new(fields: %w[title text], process_term: process, ranges: true)
    ms.add("id" => 1, "title" => "new", "text" => "茶 NYC NYC")

    result = ms.search("new york", ranges: true).first

    assert_equal({ "text" => [[2, 5], [6, 9]], "title" => [[0, 3]] }, result[:ranges])
  end

  def test_ranges_survive_save_and_reload
    process = lambda do |term, _field = nil|
      term.downcase == "nyc" ? %w[new york] : term.downcase
    end
    options = { fields: %w[title text], process_term: process, ranges: true }
    ms = MiniFTS.new(options)
    ms.add("id" => 1, "title" => "new", "text" => "茶 NYC NYC")

    reloaded = MiniFTS.load_json(ms.to_json, options)

    assert reloaded.ranges?
    result = reloaded.search("new york", ranges: true).first
    assert_equal({ "text" => [[2, 5], [6, 9]], "title" => [[0, 3]] }, result[:ranges])
  end

  def test_serialization_carries_range_data_only_when_enabled
    plain = MiniFTS.new(fields: %w[title text])
    plain.add("id" => 1, "title" => "oak", "text" => "tea oak")
    refute JSON.parse(plain.to_json).key?("termRanges")

    with_ranges = MiniFTS.new(fields: %w[title text], ranges: true)
    with_ranges.add("id" => 1, "title" => "oak", "text" => "tea oak")
    assert JSON.parse(with_ranges.to_json).key?("termRanges")
    assert with_ranges.ranges?
    refute plain.ranges?
  end

  def test_old_index_without_range_data_has_a_distinguishable_status
    # An index serialized before this feature existed simply has no termRanges
    # key; a v1-shaped blob stands in for any pre-ranges archive.
    options = { fields: %w[title text], store_fields: ["category"] }
    legacy_blob = MiniFTS.new(options).tap do |ms|
      ms.add("id" => 1, "title" => "Vita Nova", "text" => "della mia memoria", "category" => "poetry")
    end.to_json
    refute JSON.parse(legacy_blob).key?("termRanges")
    legacy = MiniFTS.load_json(legacy_blob, options)

    refute legacy.ranges?
    error = assert_raises(MiniFTS::RangesUnavailableError) { legacy.search("vita", ranges: true) }
    assert_kind_of MiniFTS::Error, error
    # Searches without ranges are unaffected.
    assert_equal [1], legacy.search("vita").map { |r| r[:id] }
  end

  def test_removing_a_document_drops_its_ranges
    ms = MiniFTS.new(fields: ["text"], ranges: true)
    doc = { "id" => 1, "text" => "tea oak and oak" }
    ms.add(doc)

    ms.remove(doc)

    assert_equal [], ms.search("oak", ranges: true)
  end

  def test_discard_and_vacuum_reclaim_range_entries
    ms = MiniFTS.new(fields: ["text"], ranges: true, auto_vacuum: false)
    ms.add("id" => 1, "text" => "alpha oak")
    ms.add("id" => 2, "text" => "beta oak")
    ms.discard(1)
    ms.vacuum

    result = ms.search("oak", ranges: true).first
    assert_equal 2, result[:id]
    assert_equal({ "text" => [[5, 8]] }, result[:ranges])
  end

  def test_and_queries_merge_ranges
    ms = MiniFTS.new(fields: ["text"], ranges: true)
    ms.add("id" => 1, "text" => "alpha beta gamma")

    result = ms.search("alpha gamma", ranges: true, combine_with: "AND").first

    assert_equal({ "text" => [[0, 5], [11, 16]] }, result[:ranges])
  end

  def test_wildcard_query_returns_empty_ranges
    ms = MiniFTS.new(fields: ["text"], ranges: true)
    ms.add("id" => 1, "text" => "alpha beta")

    result = ms.search(MiniFTS::WILDCARD, ranges: true).first

    assert_equal({}, result[:ranges])
  end

  def test_repeated_token_spans_are_sorted_and_deduplicated
    ms = MiniFTS.new(fields: ["text"], ranges: true)
    ms.add("id" => 1, "text" => "oak tea oak")

    result = ms.search("oak", ranges: true).first

    assert_equal({ "text" => [[0, 3], [8, 11]] }, result[:ranges])
  end

  def test_prefix_search_reports_the_source_token_span
    ms = MiniFTS.new(fields: ["text"], ranges: true)
    ms.add("id" => 1, "text" => "oakland tea")

    result = ms.search("oak", ranges: true, prefix: true).first

    assert_equal({ "text" => [[0, 7]] }, result[:ranges])
  end

  def test_enabling_ranges_does_not_change_scores
    documents = [
      { "id" => 1, "title" => "Divina Commedia", "text" => "Nel mezzo del cammin di nostra vita" },
      { "id" => 2, "title" => "Vita Nova", "text" => "In quella parte del libro della mia memoria" }
    ]
    plain = MiniFTS.new(fields: %w[title text])
    with_ranges = MiniFTS.new(fields: %w[title text], ranges: true)
    [plain, with_ranges].each { |ms| ms.add_all(documents) }

    plain_results = plain.search("vita memoria")
    range_results = with_ranges.search("vita memoria", ranges: true)

    assert_equal(plain_results.map { |r| r[:score] }, range_results.map { |r| r[:score] })
    assert_equal(plain_results.map { |r| r[:id] }, range_results.map { |r| r[:id] })
  end
end
