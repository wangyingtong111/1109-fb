# frozen_string_literal: true

module BackendCommonTests
  def test_initial_values
    assert @backend.total_words.zero?
    assert @backend.total_trainings.zero?
    assert @backend.category_keys.empty?
  end

  def test_total_words
    @backend.update_total_words(10)
    assert_equal 10, @backend.total_words
    @backend.update_total_words(-7)
    assert_equal 3, @backend.total_words
  end

  def test_total_trainings
    @backend.update_total_trainings(10)
    assert_equal 10, @backend.total_trainings
    @backend.update_total_trainings(-7)
    assert_equal 3, @backend.total_trainings
  end

  def test_category_training
    refute @backend.category_has_trainings?(:Interesting)
    @backend.update_category_training_count(:Interesting, 10)
    assert @backend.category_has_trainings?(:Interesting)
    assert_equal 10, @backend.category_training_count(:Interesting)
    @backend.update_category_training_count(:Interesting, -7)
    assert_equal 3, @backend.category_training_count(:Interesting)
  end

  def test_category_word
    @backend.update_category_word_count(:Interesting, 10)
    assert_equal 10, @backend.category_word_count(:Interesting)
    @backend.update_category_word_count(:Interesting, -7)
    assert_equal 3, @backend.category_word_count(:Interesting)
  end

  def test_category
    @backend.add_category(:Interesting)
    @backend.add_category(:"Not so interesting")
    assert_equal [:Interesting, :"Not so interesting"].sort, @backend.category_keys.sort
  end

  def test_category_word_frequency
    @backend.add_category(:Interesting)
    refute @backend.word_in_category?(:Interesting, 'foo')
    assert_equal 0, @backend.category_word_frequency(:Interesting, 'foo')
    @backend.update_category_word_frequency(:Interesting, 'foo', 10)
    assert @backend.word_in_category?(:Interesting, 'foo')
    assert_equal 10, @backend.category_word_frequency(:Interesting, 'foo')
    @backend.update_category_word_frequency(:Interesting, 'foo', -7)
    assert_equal 3, @backend.category_word_frequency(:Interesting, 'foo')
    @backend.delete_category_word(:Interesting, 'foo')
    refute @backend.word_in_category?(:Interesting, 'foo')
  end

  def test_reset
    @backend.update_total_words(10)
    @backend.update_total_trainings(10)
    @backend.add_category(:Interesting)
    @backend.update_category_training_count(:Interesting, 10)
    @backend.update_category_word_count(:Interesting, 10)
    @backend.reset
    assert @backend.total_words.zero?
    assert @backend.total_trainings.zero?
    assert @backend.category_keys.empty?
  end

  def test_category_word_frequencies
    @backend.add_category(:Interesting)
    assert_empty @backend.category_word_frequencies(:Interesting)
    @backend.update_category_word_frequency(:Interesting, 'foo', 3)
    @backend.update_category_word_frequency(:Interesting, 'bar', 2)
    frequencies = @backend.category_word_frequencies(:Interesting).transform_keys(&:to_s)
    assert_equal({ 'foo' => 3, 'bar' => 2 }, frequencies)
  end

  def test_absorb_adds_the_statistics_of_the_other_backend
    other = new_empty_backend
    other.add_category(:Interesting)
    other.update_category_training_count(:Interesting, 2)
    other.update_category_word_count(:Interesting, 5)
    other.update_category_word_frequency(:Interesting, 'foo', 3)
    other.update_category_word_frequency(:Interesting, 'bar', 2)
    other.update_total_trainings(2)
    other.update_total_words(5)

    @backend.add_category(:Interesting)
    @backend.update_category_training_count(:Interesting, 1)
    @backend.update_category_word_count(:Interesting, 4)
    @backend.update_category_word_frequency(:Interesting, 'foo', 4)
    @backend.update_total_trainings(1)
    @backend.update_total_words(4)

    assert_same @backend, @backend.absorb(other)

    assert_equal 3, @backend.total_trainings
    assert_equal 9, @backend.total_words
    assert_equal 3, @backend.category_training_count(:Interesting)
    assert_equal 9, @backend.category_word_count(:Interesting)
    assert_equal 7, @backend.category_word_frequency(:Interesting, 'foo')
    assert_equal 2, @backend.category_word_frequency(:Interesting, 'bar')
  end

  def test_absorb_does_not_modify_the_other_backend
    other = new_empty_backend
    other.add_category(:Interesting)
    other.update_category_training_count(:Interesting, 2)
    other.update_category_word_count(:Interesting, 5)
    other.update_category_word_frequency(:Interesting, 'foo', 5)
    other.update_total_trainings(2)
    other.update_total_words(5)

    @backend.absorb(other)
    @backend.update_category_word_frequency(:Interesting, 'foo', 100)
    @backend.update_total_trainings(100)

    assert_equal 2, other.total_trainings
    assert_equal 5, other.total_words
    assert_equal 2, other.category_training_count(:Interesting)
    assert_equal 5, other.category_word_frequency(:Interesting, 'foo')
  end

  def test_absorb_refuses_to_absorb_itself
    assert_raises(ArgumentError) { @backend.absorb(@backend) }
  end
end
