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

  # --- Minimum expected cost decisions -------------------------------

  def test_without_cost_table_keeps_log_score_classification
    classifier = review_classifier
    assert_equal 'Direct', classifier.classify('anything')
    assert_nil classifier.cost_table
  end

  def test_cost_table_switches_to_lowest_risk_action
    legacy = review_classifier
    risky = review_classifier(cost_table: review_cost_table)
    assert_equal 'Direct', legacy.classify('anything') # default max log score
    assert_equal 'Review', risky.classify('anything')
    assert_equal 'Review', risky.minimum_risk_action('anything')
  end

  def test_expected_costs_sum_over_every_true_category
    classifier = review_classifier(cost_table: review_cost_table)
    assert_equal({ 'Review' => 0.8, 'Direct' => 2.0 }, classifier.expected_costs('anything'))
  end

  def test_classify_with_risk_returns_action_and_costs
    classifier = review_classifier(cost_table: review_cost_table)
    action, risks = classifier.classify_with_risk('anything')
    assert_equal 'Review', action
    assert_equal 0.8, risks['Review']
    assert_equal 2.0, risks['Direct']
  end

  def test_three_categories_with_reject_action
    classifier = reject_classifier(reject_costs: 1.0)
    action, risks = classifier.classify_with_risk('anything')
    assert_equal 'Human', action
    assert_equal({'Direct' => 3.8, 'Review' => 1.6, 'Later' => 2.8, 'Human' => 1.0}.sort.to_h, risks.sort.to_h)
  end

  def test_common_log_offset_does_not_change_risks
    baseline = reject_classifier(reject_costs: 1.0).expected_costs(scores_without_offset)
    shifted = reject_classifier(reject_costs: 1.0).expected_costs(shifted_scores)
    assert_equal baseline, shifted
  end

  def test_cost_table_order_does_not_change_the_decision
    ordered = reject_classifier(reject_costs: 1.0).classify_with_risk('anything')
    shuffled = ClassifierReborn::Bayes.new(*three_categories, enable_stemmer: false,
                                           cost_table: three_category_table.to_a.reverse.to_h,
                                           reject_action: 'Human', reject_costs: 1.0)
    assert_equal ordered, stub_classifications(shuffled, shifted_scores).classify_with_risk('anything')
  end

  def test_reject_action_returns_its_own_identity_when_cheapest
    classifier = reject_classifier(reject_costs: 1.0)
    assert_equal 'Human', classifier.classify('anything')
    refute_includes three_categories, classifier.classify('anything')
  end

  def test_reject_action_loses_when_it_costs_more
    classifier = reject_classifier(reject_costs: 5.0)
    action, risks = classifier.classify_with_risk('anything')
    assert_equal 'Review', action
    assert_equal 5.0, risks['Human']
  end

  def test_category_keyed_cost_table
    classifier = ClassifierReborn::Bayes.new(*three_categories, enable_stemmer: false,
                                             cost_table: category_keyed_table,
                                             cost_table_orientation: :category,
                                             reject_action: 'Human',
                                             reject_costs: {'Review' => 1, 'Direct' => 1, 'Later' => 1})
    stub_classifications(classifier, shifted_scores)
    assert_equal reject_classifier(reject_costs: 1.0).classify_with_risk('anything'),
                 classifier.classify_with_risk('anything')
  end

  def test_per_call_cost_table
    classifier = review_classifier
    assert_equal 'Direct', classifier.classify('anything')
    assert_equal 'Review', classifier.classify('anything', review_cost_table)
    assert_nil classifier.cost_table
  end

  def test_missing_cost_raises
    classifier = review_classifier(cost_table: { 'Direct' => { 'Review' => 10 } })
    assert_raises(ArgumentError) { classifier.expected_costs('anything') }
  end

  def test_threshold_still_applies_only_without_cost_table
    classifier = review_classifier(enable_threshold: true, threshold: -0.1)
    assert_nil classifier.classify('anything')
    risky = review_classifier(cost_table: review_cost_table, enable_threshold: true, threshold: -0.1)
    assert_equal 'Review', risky.classify('anything')
  end

  def test_positional_rows_follow_category_order
    classifier = ClassifierReborn::Bayes.new(*three_categories, enable_stemmer: false)
    positional = ClassifierReborn::Bayes.new(*three_categories, enable_stemmer: false,
                                             cost_table: three_category_table)
    assert_equal positional.expected_costs(shifted_scores),
                 classifier.expected_costs(shifted_scores, three_category_table)
  end

  def test_expected_costs_accept_a_score_hash
    classifier = review_classifier(cost_table: review_cost_table)
    assert_equal classifier.expected_costs('anything'),
                 classifier.expected_costs(classifier.classifications('anything'))
  end

  private

  def review_scores
    { 'Review' => Math.log(0.2), 'Direct' => Math.log(0.8) }
  end

  def review_cost_table
    # Row order matches #classifications keys: Review, Direct.
    { 'Review' => { 'Review' => 0, 'Direct' => 1 },
      'Direct' => { 'Review' => 10, 'Direct' => 0 } }
  end

  def review_classifier(**options)
    classifier = ClassifierReborn::Bayes.new('Review', 'Direct', { enable_stemmer: false }.merge(options))
    stub_classifications(classifier, review_scores)
  end

  def three_categories
    %w[Review Direct Later]
  end

  def three_category_table
    { 'Direct' => [10, 0, 9], 'Review' => [0, 1, 5], 'Later' => [5, 3, 0] }
  end

  def category_keyed_table
    { 'Review' => { 'Direct' => 10, 'Review' => 0, 'Later' => 5 },
      'Direct' => { 'Direct' => 0, 'Review' => 1, 'Later' => 3 },
      'Later' => { 'Direct' => 9, 'Review' => 5, 'Later' => 0 } }
  end

  def scores_without_offset
    { 'Review' => Math.log(0.2), 'Direct' => Math.log(0.6), 'Later' => Math.log(0.2) }
  end

  def shifted_scores
    scores_without_offset.transform_values { |score| score - 1000.0 }
  end

  def reject_classifier(**options)
    classifier = ClassifierReborn::Bayes.new(*three_categories,
                                             { enable_stemmer: false,
                                               cost_table: three_category_table,
                                               reject_action: 'Human' }.merge(options))
    stub_classifications(classifier, shifted_scores)
  end

  def stub_classifications(classifier, scores)
    classifier.define_singleton_method(:classifications) { |*| scores.dup }
    classifier
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
