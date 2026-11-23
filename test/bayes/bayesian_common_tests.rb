# frozen_string_literal: true

module BayesianCommonTests
  def test_good_training
    assert_equal ['love'], @classifier.train_interesting('love')
  end

  def test_training_with_utf8
    assert_equal ['Água'], @classifier.train_interesting('Água')
  end

  def test_stemming_enabled_by_default
    assert @classifier.stemmer_enabled?
  end

  def test_bad_training
    assert_raises(StandardError) { @classifier.train_no_category 'words' }
  end

  def test_bad_method
    assert_raises(NoMethodError) { @classifier.forget_everything_you_know '' }
  end

  def test_categories
    assert_equal %w[Interesting Uninteresting].sort, @classifier.categories.sort
  end

  def test_categories_from_array
    assert_equal another_classifier.categories.sort, @classifier.categories.sort
  end

  def test_add_category
    @classifier.add_category 'Test'
    assert_equal %w[Test Interesting Uninteresting].sort, @classifier.categories.sort
  end

  def test_dynamic_category_succeeds_with_auto_categorize
    classifier = auto_categorize_classifier
    classifier.train('Ruby', 'A really sweet language')
    assert classifier.categories.include?('Ruby')
  end

  def test_dynamic_category_succeeds_with_empty_categories
    classifier = empty_classifier
    assert classifier.categories.empty?
    classifier.train('Ruby', 'A really sweet language')
    assert classifier.categories.include?('Ruby')
    assert_equal 1, classifier.categories.size
  end

  def test_dynamic_category_fails_without_auto_categorize
    assert_raises(ClassifierReborn::Bayes::CategoryNotFoundError) do
      @classifier.train('Ruby', 'A really sweet language')
    end
    refute @classifier.categories.include?('Ruby')
  end

  def test_dynamic_category_fails_with_useless_classifier
    classifier = useless_classifier
    assert classifier.categories.empty?
    assert_raises(ClassifierReborn::Bayes::CategoryNotFoundError) do
      classifier.train('Ruby', 'A really sweet language')
    end
    refute classifier.categories.include?('Ruby')
  end

  def test_classification
    @classifier.train_interesting 'here are some good words. I hope you love them'
    @classifier.train_uninteresting 'here are some bad words, I hate you'
    assert_equal 'Uninteresting', @classifier.classify('I hate bad words and you')
  end

  def test_classification_with_threshold
    b = threshold_classifier('Number')
    assert_equal 1, b.categories.size

    refute b.threshold_enabled?
    b.enable_threshold
    assert b.threshold_enabled?
    assert_equal 0.0, b.threshold # default

    b.threshold = -4.0

    %w[one two three four five].each do |a_number|
      b.train_number(a_number)
      b.train_number(a_number)
    end

    %w[one two three four five].each do |a_number|
      assert_equal 'Number', b.classify(a_number)
    end

    refute b.classify('xyzzy')
  end

  def test_classification_with_threshold_again
    b = threshold_classifier('Normal')
    assert_equal 1, b.categories.size

    refute b.threshold_enabled?
    b.enable_threshold
    assert b.threshold_enabled?
    assert_equal 0.0, b.threshold # default

    %w[
      http://example.com/about
      http://example.com/contact
      http://example.com/download
      http://example.com/login
      http://example.com/logout
      http://example.com/blog/2015-04-01
    ].each do |url|
      b.train_normal(url)
    end

    assert 'Normal', b.classify('http://example.com')
    refute b.classify("http://example.com/login/?user='select * from users;'")
  end

  def test_classification_with_score
    @classifier.train_interesting 'here are some good words. I hope you love them'
    @classifier.train_uninteresting 'here are some bad words, I hate you'
    assert_in_delta(-4.85, @classifier.classify_with_score('I hate bad words and you')[1], 0.1)
  end

  def test_untrain
    @classifier.train_interesting 'here are some good words. I hope you love them'
    @classifier.train_uninteresting 'here are some bad words, I hate you'
    @classifier.add_category 'colors'
    @classifier.train_colors 'red orange green blue seven'
    refute_equal Float::INFINITY, @classifier.classify_with_score('seven')[1]

    @classifier.untrain_colors 'red orange green blue seven'

    assert_equal Float::INFINITY, @classifier.classify_with_score('seven')[1]
    assert_equal Float::INFINITY, @classifier.classifications('seven')['Colors']
  end

  # "Maps" was trained twice with the same sample, "Weather" once.
  # Revoking an unknown sample must leave every number and score untouched.
  def test_untrain_of_an_unknown_sample_changes_nothing
    classifier, backend = recoverable_classifier('Maps', 'Weather')
    classifier.train_maps 'bridge routes'
    classifier.train_maps 'bridge routes'
    classifier.train_weather 'rain fog'
    before = model_snapshot(classifier, backend)

    assert_equal false, classifier.untrain('Maps', 'bridge unknown')

    assert_equal before, model_snapshot(classifier, backend)
    assert_equal 2, backend.category_training_count(:Maps)
    assert_equal 1, backend.category_training_count(:Weather)
    assert_equal 4, backend.category_word_count(:Maps)
    assert_equal 6, backend.total_words
    assert_equal 3, backend.total_trainings
  end

  # The real sample still has one contribution left over after revoking one.
  def test_untrain_revokes_a_single_recorded_contribution
    classifier, backend = recoverable_classifier('Maps', 'Weather')
    classifier.train_maps 'bridge routes'
    classifier.train_maps 'bridge routes'
    classifier.train_weather 'rain fog'

    assert classifier.untrain('Maps', 'bridge routes')

    assert_equal 1, backend.category_training_count(:Maps)
    assert_equal 2, backend.category_word_count(:Maps)
    assert_equal 1, backend.category_word_frequency(:Maps, :bridg)
    assert_equal 1, backend.category_word_frequency(:Maps, :rout)
    assert backend.word_in_category?(:Maps, :bridg)
    assert backend.word_in_category?(:Maps, :rout)
    assert_equal 1, backend.category_training_count(:Weather)
    assert_equal 4, backend.total_words
    assert_equal 2, backend.total_trainings
    scores = classifier.classifications('bridge routes')

    # The remaining Maps contribution is proportional to a single training, so
    # the scores match a classifier that only ever saw the sample once.
    reference, = recoverable_classifier('Maps', 'Weather')
    reference.train_maps 'bridge routes'
    reference.train_weather 'rain fog'
    assert_equal reference.classifications('bridge routes'), scores
  end

  # Partial word overlap is not the sample that was trained.
  def test_untrain_ignores_partial_overlap_with_a_trained_sample
    classifier, backend = recoverable_classifier('Maps', 'Weather')
    classifier.train_maps 'bridge routes'
    classifier.train_weather 'rain fog'
    before = model_snapshot(classifier, backend)

    assert_equal false, classifier.untrain('Maps', 'bridge')
    assert_equal false, classifier.untrain('Weather', 'bridge routes')
    assert_equal false, classifier.untrain('Maps', 'bridge routes and tunnels')

    assert_equal before, model_snapshot(classifier, backend)
  end

  def test_untrain_does_not_revoke_a_sample_of_another_category
    classifier, backend = recoverable_classifier('Maps', 'Weather')
    classifier.train_maps 'bridge routes'
    before = model_snapshot(classifier, backend)

    assert_equal false, classifier.untrain('Weather', 'bridge routes')

    assert_equal before, model_snapshot(classifier, backend)
    assert_equal 1, backend.category_training_count(:Maps)
    assert_equal 0, backend.category_training_count(:Weather)
  end

  # Training the same sample twice records two contributions, so revoking works
  # twice as well - and not a third time.
  def test_untrain_twice_revokes_one_contribution_per_call
    classifier, backend = recoverable_classifier('Maps')
    classifier.train_maps 'bridge routes'
    classifier.train_maps 'bridge routes'

    assert classifier.untrain('Maps', 'bridge routes')
    assert_equal 1, backend.category_training_count(:Maps)
    assert_equal 1, backend.total_trainings
    assert_equal 2, backend.category_word_count(:Maps)

    assert classifier.untrain('Maps', 'bridge routes')
    assert_equal 0, backend.category_training_count(:Maps)
    assert_equal 0, backend.total_trainings
    assert_equal 0, backend.category_word_count(:Maps)
    assert backend.total_words.zero?
    refute backend.word_in_category?(:Maps, :bridg)
    refute backend.word_in_category?(:Maps, :rout)

    # No contributions are left, so the same text cannot be revoked again.
    assert_equal false, classifier.untrain('Maps', 'bridge routes')
    assert_equal 0, backend.category_training_count(:Maps)
    assert backend.total_words.zero?
  end

  def test_untrain_uses_the_same_preprocessing_as_training
    classifier, backend = recoverable_classifier('Maps')
    classifier.train_maps 'Bridge Routes'

    assert classifier.untrain('Maps', 'bridge routes')

    assert_equal 0, backend.category_training_count(:Maps)
    assert_equal 0, backend.category_word_count(:Maps)
    assert backend.total_words.zero?
    assert backend.total_trainings.zero?
  end

  def test_skip_empty_training_and_classification
    classifier = empty_classifier
    classifier.train('Ruby', '')
    assert classifier.categories.empty?
    classifier.train('Ruby', 'To be or not to be')
    assert classifier.categories.empty?
    classifier.train('Ruby', 'A really sweet language')
    refute classifier.categories.empty?
    assert_equal Float::INFINITY, classifier.classify_with_score('To be or not to be')[1]
  end

  def test_empty_string_stopwords
    classifier = empty_string_stopwords_classifier
    classifier.train('Stopwords', 'To be or not to be')
    refute classifier.categories.empty?
    refute_equal Float::INFINITY, classifier.classify_with_score('To be or not to be')[1]
  end

  def test_empty_array_stopwords
    classifier = empty_array_stopwords_classifier
    classifier.train('Stopwords', 'To be or not to be')
    refute classifier.categories.empty?
    refute_equal Float::INFINITY, classifier.classify_with_score('To be or not to be')[1]
  end

  def test_custom_array_stopwords
    classifier = array_stopwords_classifier
    classifier.train('Stopwords', 'Custom stopwords')
    assert classifier.categories.empty?
    classifier.train('Stopwords', 'To be or not to be')
    refute classifier.categories.empty?
    assert_equal Float::INFINITY, classifier.classify_with_score('These stopwords')[1]
    refute_equal Float::INFINITY, classifier.classify_with_score('To be or not to be')[1]
  end

  def test_custom_file_stopwords
    classifier = file_stopwords_classifier
    classifier.train('Stopwords', 'Custom stopwords')
    assert classifier.categories.empty?
    classifier.train('Stopwords', 'To be or not to be')
    refute classifier.categories.empty?
    assert_equal Float::INFINITY, classifier.classify_with_score('These stopwords')[1]
    refute_equal Float::INFINITY, classifier.classify_with_score('To be or not to be')[1]
  end

  def test_reset
    @classifier.add_category 'Test'
    assert_equal %w[Test Interesting Uninteresting].sort, @classifier.categories.sort
    @classifier.reset
    assert_equal %w[Interesting Uninteresting].sort, @classifier.categories.sort
    classifier = empty_classifier
    classifier.train('Ruby', 'A really sweet language')
    assert classifier.categories.include?('Ruby')
    classifier.reset
    assert classifier.categories.empty?
  end

  private

  # Returns a classifier and the backend it stores its model in, so that the
  # untraining tests can inspect the raw counters as well as the scores. Each
  # call gets its own, isolated backend.
  def recoverable_classifier(*categories)
    backend = fresh_backend
    [ClassifierReborn::Bayes.new(*categories, backend: backend), backend]
  end

  # Everything untraining is allowed to change, so that "nothing changed" can
  # be asserted on the word frequencies, the document priors and the scores.
  def model_snapshot(classifier, backend)
    words = %i[bridg rout rain fog unknown]
    categories = backend.category_keys.sort
    {
      total_words: backend.total_words,
      total_trainings: backend.total_trainings,
      categories: categories,
      training_counts: categories.map { |c| backend.category_training_count(c) },
      word_counts: categories.map { |c| backend.category_word_count(c) },
      frequencies: categories.map do |category|
        words.map do |word|
          [backend.word_in_category?(category, word), backend.category_word_frequency(category, word)]
        end
      end,
      scores: classifier.classifications('bridge routes rain fog')
    }
  end

  def another_classifier
    ClassifierReborn::Bayes.new %w[Interesting Uninteresting], backend: @alternate_backend
  end

  def auto_categorize_classifier
    ClassifierReborn::Bayes.new 'Interesting', 'Uninteresting', auto_categorize: true, backend: @alternate_backend
  end

  def threshold_classifier(category)
    ClassifierReborn::Bayes.new category, backend: @alternate_backend
  end

  def empty_classifier
    ClassifierReborn::Bayes.new backend: @alternate_backend
  end

  def useless_classifier
    ClassifierReborn::Bayes.new auto_categorize: false, backend: @alternate_backend
  end

  def empty_string_stopwords_classifier
    ClassifierReborn::Bayes.new stopwords: '', backend: @alternate_backend
  end

  def empty_array_stopwords_classifier
    ClassifierReborn::Bayes.new stopwords: [], backend: @alternate_backend
  end

  def array_stopwords_classifier
    ClassifierReborn::Bayes.new stopwords: %w[these are custom stopwords], backend: @alternate_backend
  end

  def file_stopwords_classifier
    ClassifierReborn::Bayes.new stopwords: File.dirname(__FILE__) + '/../data/stopwords/en', backend: @alternate_backend
  end
end
