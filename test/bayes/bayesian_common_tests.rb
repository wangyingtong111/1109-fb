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

  def test_word_frequency_disabled_by_default
    assert @classifier.word_frequency_disabled?
    refute @classifier.word_frequency_enabled?
  end

  def test_word_frequency_mode_scores_repeated_words_multiple_times
    classifier = word_frequency_classifier
    assert classifier.word_frequency_enabled?
    classifier.train('Maps', 'harbor harbor rain')
    classifier.train('Weather', 'rain rain harbor')

    assert_equal 'Maps', classifier.classify('harbor harbor harbor rain')
    assert_equal 'Weather', classifier.classify('harbor rain rain rain')
  end

  def test_default_mode_ignores_repeated_word_counts
    classifier = ClassifierReborn::Bayes.new 'Maps', 'Weather', backend: @alternate_backend
    classifier.train('Maps', 'harbor harbor rain')
    classifier.train('Weather', 'rain rain harbor')

    query_a = 'harbor harbor harbor rain'
    query_b = 'harbor rain rain rain'
    assert_equal classifier.classifications(query_a), classifier.classifications(query_b)
    assert_equal classifier.classify(query_a), classifier.classify(query_b)
  end

  def test_word_frequency_mode_counts_tokens_after_tokenizer_and_filters
    classifier = ClassifierReborn::Bayes.new 'Harbor', enable_word_frequency: true, backend: @alternate_backend
    classifier.train('Harbor', 'harbor harbor rain')

    # Stopwords ("the", "an", "a") and short tokens ("xy") are filtered out,
    # so only the surviving tokens contribute, with their occurrence counts.
    filtered = 'harbor harbor harbor rain'
    with_stopwords = 'harbor harbor harbor rain the an a xy'
    assert_equal classifier.classifications(filtered), classifier.classifications(with_stopwords)
  end

  def test_word_frequency_mode_adds_category_prior_once_per_text
    classifier = word_frequency_classifier
    classifier.train('Maps', 'harbor harbor rain')
    classifier.train('Weather', 'rain rain harbor')

    backend = classifier.instance_variable_get(:@backend)
    once = classifier.classifications('harbor')
    thrice = classifier.classifications('harbor harbor harbor')
    once.each do |category, score|
      prior = Math.log(backend.category_training_count(category.intern) / backend.total_trainings.to_f)
      single_word_contribution = score - prior
      assert_in_delta(score + 2 * single_word_contribution, thrice[category], 1e-12)
    end
  end

  def test_word_frequency_mode_can_be_toggled_dynamically
    classifier = ClassifierReborn::Bayes.new 'Maps', 'Weather', backend: @alternate_backend
    classifier.train('Maps', 'harbor harbor rain')
    classifier.train('Weather', 'rain rain harbor')

    default_scores = classifier.classifications('harbor harbor harbor rain')
    classifier.enable_word_frequency
    assert classifier.word_frequency_enabled?
    frequency_scores = classifier.classifications('harbor harbor harbor rain')
    refute_equal default_scores, frequency_scores
    assert_equal 'Maps', classifier.classify('harbor harbor harbor rain')

    classifier.disable_word_frequency
    assert classifier.word_frequency_disabled?
    assert_equal default_scores, classifier.classifications('harbor harbor harbor rain')
  end

  def test_untrain
    @classifier.train_interesting 'here are some good words. I hope you love them'
    @classifier.train_uninteresting 'here are some bad words, I hate you'
    @classifier.add_category 'colors'
    @classifier.train_colors 'red orange green blue seven'
    classification_of_bad_data = @classifier.classify 'seven'
    @classifier.untrain_colors 'seven'
    classification_after_untrain = @classifier.classify 'seven'
    refute_equal classification_of_bad_data, classification_after_untrain
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

  def another_classifier
    ClassifierReborn::Bayes.new %w[Interesting Uninteresting], backend: @alternate_backend
  end

  def auto_categorize_classifier
    ClassifierReborn::Bayes.new 'Interesting', 'Uninteresting', auto_categorize: true, backend: @alternate_backend
  end

  def threshold_classifier(category)
    ClassifierReborn::Bayes.new category, backend: @alternate_backend
  end

  def word_frequency_classifier
    ClassifierReborn::Bayes.new 'Maps', 'Weather', enable_word_frequency: true, backend: @alternate_backend
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
