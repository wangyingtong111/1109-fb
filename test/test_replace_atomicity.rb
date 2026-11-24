# frozen_string_literal: true

require "test_helper"

class TestReplaceAtomicity < Minitest::Test
  def setup
    @calls = []
    process_term = lambda do |term, field_name = nil|
      @calls << [term, field_name]
      raise "bad term" if term == "bad" && field_name == "text"

      term.downcase
    end

    @options = {
      fields: %w[title text],
      store_fields: %w[title text],
      auto_vacuum: false,
      process_term: process_term
    }
    @index = MiniFTS.new(@options)
    @original = { "id" => 7, "title" => "old", "text" => "old body" }
    @index.add(@original)
  end

  def test_failed_replace_restores_the_complete_previous_state
    saved = JSON.parse(JSON.generate(@index.as_plain_object))
    bad = { "id" => 7, "title" => "new", "text" => "bad" }

    error = assert_raises(RuntimeError) { @index.replace(bad) }
    assert_equal "bad term", error.message

    assert_equal([7], @index.search("old").map { |result| result[:id] })
    assert_empty(@index.search("new").map { |result| result[:id] })
    assert @index.has?(7)
    assert_equal 0, @index.dirt_count
    assert_equal({ "title" => "old", "text" => "old body" }, @index.get_stored_fields(7))
    assert_equal saved, JSON.parse(JSON.generate(@index.as_plain_object))
  end

  def test_successful_retry_processes_each_field_once_and_matches_discard_add
    bad = { "id" => 7, "title" => "new", "text" => "bad" }
    assert_raises(RuntimeError) { @index.replace(bad) }

    @calls.clear
    good = { "id" => 7, "title" => "new", "text" => "good" }
    @index.replace(good)

    assert_equal 1, @calls.count(%w[new title])
    assert_equal 1, @calls.count(%w[good text])

    control = MiniFTS.new(fields: %w[title text], store_fields: %w[title text], auto_vacuum: false)
    control.add(@original)
    control.discard(7)
    control.add(good)

    %w[new good old].each do |query|
      assert_equal control.search(query), @index.search(query)
    end
    assert_equal control.as_plain_object, @index.as_plain_object
  end

  def test_field_extractor_failure_is_also_atomic
    options = @options.merge(
      extract_field: lambda do |document, field|
        raise "cannot extract #{field}" if field == "text" && document["text"] == "bad"

        document[field]
      end
    )
    index = MiniFTS.new(options)
    index.add(@original)
    saved = JSON.parse(JSON.generate(index.as_plain_object))

    assert_raises(RuntimeError) do
      index.replace("id" => 7, "title" => "new", "text" => "bad")
    end

    assert_equal(saved, JSON.parse(JSON.generate(index.as_plain_object)))
    assert_equal([7], index.search("old").map { |result| result[:id] })
    assert_empty(index.search("new").map { |result| result[:id] })
  end
end
