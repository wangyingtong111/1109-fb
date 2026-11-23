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

  def test_stopword_lists_are_isolated_between_classifiers
    with_isolated_backends do |maps_backend, gravel_backend|
      maps_classifier   = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[bridge], backend: maps_backend
      gravel_classifier = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[gravel], backend: gravel_backend

      maps_classifier.train('Maps', 'bridge gravel')
      gravel_classifier.train('Maps', 'bridge gravel')

      # Stemming is enabled by default, so "bridge" is stored as "bridg".
      assert_equal %w[gravel], learned_words(maps_classifier, 'Maps')
      assert_equal %w[bridg], learned_words(gravel_classifier, 'Maps')
    end
  end

  def test_stopword_isolation_is_independent_of_creation_order
    with_isolated_backends do |maps_backend, gravel_backend|
      gravel_classifier = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[gravel], backend: gravel_backend
      maps_classifier   = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[bridge], backend: maps_backend

      maps_classifier.train('Maps', 'bridge gravel')
      gravel_classifier.train('Maps', 'bridge gravel')

      assert_equal %w[gravel], learned_words(maps_classifier, 'Maps')
      assert_equal %w[bridg], learned_words(gravel_classifier, 'Maps')
    end
  end

  def test_classification_uses_each_classifiers_own_stopwords
    with_isolated_backends do |maps_backend, gravel_backend|
      maps_classifier   = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[bridge], backend: maps_backend
      gravel_classifier = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[gravel], backend: gravel_backend

      maps_classifier.train('Maps', 'bridge gravel')
      maps_classifier.train('Other', 'river valley')
      gravel_classifier.train('Maps', 'bridge gravel')
      gravel_classifier.train('Other', 'river valley')

      assert_equal 'Maps', maps_classifier.classify('gravel')
      assert_equal 'Maps', gravel_classifier.classify('bridge')

      # A document consisting only of the model's own stopword has no features
      assert_equal Float::INFINITY, maps_classifier.classify_with_score('bridge')[1]
      assert_equal Float::INFINITY, gravel_classifier.classify_with_score('gravel')[1]
    end
  end

  def test_default_language_stopwords_coexist_with_custom_stopwords
    with_isolated_backends do |default_backend, custom_backend|
      default_classifier = ClassifierReborn::Bayes.new 'Maps', 'Other', backend: default_backend
      custom_classifier  = ClassifierReborn::Bayes.new 'Maps', 'Other', stopwords: %w[bridge], backend: custom_backend

      default_classifier.train('Maps', 'bridge gravel')
      custom_classifier.train('Maps', 'bridge gravel')

      assert_includes learned_words(default_classifier, 'Maps'), 'bridg'
      refute_includes learned_words(custom_classifier, 'Maps'), 'bridg'
      # The shared bundled vocabulary is untouched and still filters "the"
      default_classifier.train('Maps', 'the quick brown fox')
      refute_includes learned_words(default_classifier, 'Maps'), 'the'
    end
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

  # Builds two fresh, independent backends so classifiers under test never
  # share categories or word counts. Concrete test classes provide the class
  # via +backend_class+ and reset the store through +teardown+ as usual.
  def with_isolated_backends
    first_backend = backend_class.new(*isolated_backend_options(2))
    second_backend = backend_class.new(*isolated_backend_options(3))
    yield first_backend, second_backend
  ensure
    first_backend&.reset
    second_backend&.reset
  end

  def isolated_backend_options(_db_index)
    []
  end

  def learned_words(classifier, category)
    category = ClassifierReborn::CategoryNamer.prepare_name(category)
    backend = classifier.instance_variable_get(:@backend)
    if backend.instance_variable_defined?(:@redis)
      backend.instance_variable_get(:@redis).hkeys(category).map(&:to_s)
    else
      backend.instance_variable_get(:@categories)[category].keys.map(&:to_s)
    end
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
