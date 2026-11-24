# frozen_string_literal: true

require "test_helper"
require "json"

class TestFieldStatistics < Minitest::Test
  DOCUMENTS = [
    { "id" => 1, "summary" => "oak pine" },
    { "id" => 2 },
    { "id" => 3, "summary" => "" }
  ].freeze

  def options(extra = {})
    { fields: ["summary"], auto_vacuum: false }.merge(extra)
  end

  def build_present
    MiniFTS.new(options(field_statistics: :present_fields)).tap { |ms| ms.add_all(DOCUMENTS) }
  end

  def test_present_population_counts_explicit_empty_but_not_missing_field
    ms = build_present

    assert_equal 1.0, ms.average_field_length("summary")
    assert_equal 2, ms.field_document_count("summary")

    plain = JSON.parse(JSON.generate(ms.as_plain_object))
    assert_equal [1.0], plain["averageFieldLength"]
    assert_equal [2], plain["fieldCount"]
    assert plain["fieldLength"].key?("0")
    refute plain["fieldLength"].key?("1")
    assert_equal [0], plain["fieldLength"]["2"]
  end

  def test_present_population_survives_json_reload_and_discard
    ms = build_present
    json = JSON.generate(ms.as_plain_object)
    loaded = MiniFTS.load_json(json, options(field_statistics: :present_fields))

    assert_equal 1.0, loaded.average_field_length("summary")

    loaded.discard(3)

    assert_equal 2.0, loaded.average_field_length("summary")
    assert_equal 1, loaded.field_document_count("summary")

    reloaded = MiniFTS.load(JSON.parse(JSON.generate(loaded.as_plain_object)),
                            options(field_statistics: :present_fields))
    assert_equal 2.0, reloaded.average_field_length("summary")
  end

  def test_present_population_remove_path_returns_to_single_sample_average
    ms = build_present
    ms.remove("id" => 3, "summary" => "")

    assert_equal 2.0, ms.average_field_length("summary")
    assert_equal 1, ms.field_document_count("summary")
  end

  def test_serialized_population_marker_selects_itself_on_load
    ms = build_present
    loaded = MiniFTS.load(JSON.parse(JSON.generate(ms.as_plain_object)), options)

    assert_equal MiniFTS::FIELD_STATISTICS_PRESENT, loaded.field_statistics_mode
    assert_equal 1.0, loaded.average_field_length("summary")
  end

  def test_default_statistics_and_term_frequencies_are_unchanged
    default = MiniFTS.new(options)
    default.add_all(DOCUMENTS)

    plain = JSON.parse(JSON.generate(default.as_plain_object))
    assert_in_delta 5.0 / 3.0, plain["averageFieldLength"].first
    refute plain.key?("fieldCount")
    refute plain.key?("fieldLengthPopulation")

    present = build_present
    assert_equal default.as_plain_object["index"], present.as_plain_object["index"]

    default_score = default.search("oak").first[:score]
    assert_in_delta default_score, present.search("oak", field_statistics: :legacy).first[:score]
    assert_in_delta default_score, MiniFTS.load_json(default.to_json, options).search("oak").first[:score]
  end

  def test_legacy_index_with_field_length_can_rebuild_present_population
    legacy = MiniFTS.new(options)
    legacy.add_all(DOCUMENTS)
    json = JSON.parse(JSON.generate(legacy.as_plain_object))

    loaded = MiniFTS.load(json, options)
    assert loaded.field_statistics_reconstructable?
    assert loaded.field_statistics_available?
    refute loaded.recorded_field_statistics?

    loaded.enable_field_statistics!(:present_fields)
    assert_equal 1.0, loaded.average_field_length("summary")

    loaded.discard(3)
    assert_equal 2.0, loaded.average_field_length("summary")
  end

  def test_index_without_old_field_counts_reports_that_new_population_is_unavailable
    legacy = MiniFTS.new(options)
    legacy.add_all(DOCUMENTS)
    json = JSON.parse(JSON.generate(legacy.as_plain_object))
    json.delete("fieldLength")
    json.delete("legacyFieldLength")

    loaded = MiniFTS.load(json, options)
    refute loaded.field_statistics_reconstructable?
    refute loaded.field_statistics_available?

    error = assert_raises(MiniFTS::Error) { loaded.enable_field_statistics! }
    assert_match(/fieldLength/, error.message)

    assert_raises(MiniFTS::Error) do
      MiniFTS.load(json, options(field_statistics: :present_fields))
    end
  end

  def test_descriptive_option_aliases_select_present_population
    ms = MiniFTS.new(fields: ["summary"], field_length_population: :exclude_missing)
    assert_equal MiniFTS::FIELD_STATISTICS_PRESENT, ms.field_statistics_mode

    ms = MiniFTS.new(fields: ["summary"], count_empty_fields: true)
    assert_equal MiniFTS::FIELD_STATISTICS_PRESENT, ms.field_statistics_mode
  end
end
