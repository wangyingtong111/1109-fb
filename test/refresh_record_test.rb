# encoding: utf-8
require 'minitest/autorun'
require 'fuzzy_match'

class RefreshRecordTest < Minitest::Test
  def setup
    @reads = []
    @reader = lambda do |row|
      @reads << row[:id]
      raise ArgumentError, 'invalid record label' if row[:label].nil?
      row[:label]
    end
    @rows = [{ id: 1, label: 'oak' }, { id: 2, label: 'oak' }]
    @matcher = FuzzyMatch.new(@rows, primary_key: :id, read: @reader)
    find_ids('oak') # populate the original whole/clean/words projections
    @reads.clear
  end

  def test_replaces_only_selected_projection_by_id
    replacement = { id: 2, label: 'maple' }

    assert_equal true, @matcher.refresh_record(2, replacement)
    assert_equal [1], find_ids('oak')
    assert_equal [2], find_ids('maple')
    assert_equal [2], @reads.uniq.sort
  end

  def test_rereads_mutated_entered_record
    @rows[1][:label] = 'maple'

    assert_equal true, @matcher.refresh(2)
    assert_equal [1], find_ids('oak')
    assert_equal [2], find_ids('maple')
    assert_equal [2], @reads.uniq.sort
  end

  def test_finds_replacement_assigned_in_source_array
    @rows[1] = { id: 2, label: 'maple' }

    assert_equal true, @matcher.refresh(2)
    assert_equal [2], find_ids('maple')
    assert_equal [1], find_ids('oak')
    assert_equal [2], @reads.uniq.sort
  end

  def test_failed_refresh_returns_error_and_preserves_old_projection
    bad_replacement = { id: 2, label: nil }

    error = @matcher.refresh_record(2, bad_replacement)

    assert_instance_of ArgumentError, error
    assert_equal 'invalid record label', error.message
    assert_equal [1, 2], find_ids('oak')
    assert_empty find_ids('maple')
    assert_equal [2], @reads.uniq.sort
  end

  def test_bang_refresh_raises_and_preserves_old_projection
    bad_replacement = { id: 2, label: nil }

    assert_raises(ArgumentError) { @matcher.refresh_record!(2, bad_replacement) }
    assert_equal [1, 2], find_ids('oak')
    assert_empty find_ids('maple')
    assert_equal [2], @reads.uniq.sort
  end

  def test_without_refresh_uses_cached_projections_and_stable_order
    assert_equal [1, 2], find_ids('oak')
    assert_empty find_ids('maple')
    assert_empty @reads
  end

  private

  def find_ids(query)
    @matcher.find_all(query, must_match_at_least_one_word: true).map { |row| row[:id] }.sort
  end
end
