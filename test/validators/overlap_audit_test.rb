# frozen_string_literal: true

require_relative '../test_helper'

# Covers the train/holdout overlap audit built on the effective feature
# extraction of the classifier under validation.
class OverlapAuditTest < Minitest::Test
  CV = ClassifierValidator

  TWO_LABELS = [
    ['Green Tea Leaves', 'Green Tea Leaves'],
    ['green tea leaves', 'green tea leaves'],
    ['Honey', 'orchid blossom honey']
  ].freeze

  HOLDOUT = [
    ['Green Tea Leaves', 'GREEN TEA LEAVES'],
    ['Honey', 'the and of'],
    ['Honey', 'volcanic obsidian shard']
  ].freeze

  def teardown
    # A classifier constructed with :stopwords replaces the process-wide
    # English stopword set; clearing the cache reloads it from data/stopwords.
    TokenFilter::Stopword::STOPWORDS.clear
    TokenFilter::Stopword.language = 'en'
  end

  def audit(training = TWO_LABELS, holdout = HOLDOUT, classifier = 'Bayes', **options)
    klass = classifier.is_a?(String) ? ClassifierReborn.const_get(classifier).new(options) : classifier
    CV.train_classifier(klass, training)
    CV.audit_overlap(klass, training, holdout)
  end

  def entry(audit, index)
    audit[:holdout].find { |e| e[:index] == index }
  end

  def test_overlap_uses_the_effective_feature_extraction
    result = audit
    assert_equal :classifier, result[:feature_extraction]
    assert_equal [:green, :leav, :tea], CV.audit_features(ClassifierReborn::Bayes.new, 'GREEN TEA LEAVES').keys.sort
    assert_equal %w[green leav tea], entry(result, 0)[:features]
  end

  def test_overlap_keeps_every_training_source_and_label
    conflicted = entry(audit, 0)
    assert_equal :overlap, conflicted[:status]
    assert_equal 2, conflicted[:sources].length
    assert_equal [0, 1], conflicted[:sources].map { |s| s[:index] }
    assert_equal %w[Green\ Tea\ Leaves green\ tea\ leaves], conflicted[:labels]
    assert conflicted[:conflicted]
  end

  def test_last_label_does_not_win_over_the_conflict
    conflicted = entry(audit, 0)
    refute_equal ['green tea leaves'], conflicted[:labels]
    assert_equal 2, conflicted[:labels].uniq.length
    assert_equal 1, audit[:counts][:holdout_conflicted]
  end

  def test_empty_features_are_reported_separately_and_never_matched
    result = audit(TWO_LABELS, [['Honey', 'the and of'], ['Honey', 'a is to'], ['Honey', 'of it in']])
    assert_equal %i[empty_features empty_features empty_features], result[:holdout].map { |e| e[:status] }
    assert_equal 0, result[:counts][:holdout_overlapping]
    assert_equal 3, result[:counts][:holdout_empty_features]
    result[:holdout].each do |e|
      assert_empty e[:sources]
      assert_empty e[:labels]
      refute e[:conflicted]
    end
  end

  def test_a_new_document_stays_in_the_effective_holdout
    fresh = entry(audit, 2)
    assert_equal :unique, fresh[:status]
    assert_empty fresh[:sources]
    assert_equal %w[obsidian shard volcan], fresh[:features]
  end

  def test_both_denominators_are_kept
    counts = audit[:counts]
    assert_equal 3, counts[:training_total]
    assert_equal 3, counts[:holdout_total]
    assert_equal 1, counts[:holdout_empty_features]
    assert_equal 2, counts[:comparable_holdout]
    assert_equal 1, counts[:holdout_overlapping]
    assert_equal 1, counts[:holdout_unique]
    assert_equal 1, counts[:audited_holdout]
    assert_equal counts[:holdout_total],
                 counts[:holdout_unique] + counts[:holdout_overlapping] + counts[:holdout_empty_features]
    assert_equal counts[:comparable_holdout], counts[:holdout_total] - counts[:holdout_empty_features]
  end

  def test_empty_training_records_are_not_indexed_as_sources
    training = TWO_LABELS + [['Green Tea Leaves', 'the and of']]
    result = audit(training)
    assert_equal 1, result[:counts][:training_empty_features]
    assert_equal 2, entry(result, 0)[:sources].length
  end

  def test_audit_follows_the_stemmer_setting
    with_stemmer = audit([['Spam', 'winning prizes']], [['Spam', 'win prize']])
    plain = ClassifierReborn::Bayes.new('Spam', enable_stemmer: false)
    without_stemmer = audit([['Spam', 'winning prizes']], [['Spam', 'win prize']], plain)
    assert_equal :overlap, with_stemmer[:holdout].first[:status]
    assert_equal :unique, without_stemmer[:holdout].first[:status]
  end

  def test_audit_follows_configured_stopwords
    training = [['Honey', 'orchid blossom honey']]
    custom = ClassifierReborn::Bayes.new('Honey', 'Green Tea Leaves', 'green tea leaves',
                                         stopwords: %w[orchid blossom])
    result = audit(training, [['Honey', 'orchid blossom']], custom)
    assert_equal :empty_features, result[:holdout].first[:status]
  end

  def test_falls_back_to_the_default_rule_for_a_foreign_classifier
    result = CV.audit_overlap(Object.new, [['A', 'green tea']], [['A', 'GREEN TEA']])
    assert_equal :default, result[:feature_extraction]
    assert_equal :overlap, result[:holdout].first[:status]
  end

  def test_validate_with_audit_returns_confusion_matrix_and_overlap
    result = CV.validate_with_audit('Bayes', TWO_LABELS, HOLDOUT)
    assert_equal 3, result[:conf_mat].values.sum { |cols| cols.values.sum }
    assert_equal 3, result[:overlap][:counts][:holdout_total]
  end

  def test_cross_validate_with_audit_keeps_the_fold_of_each_source
    words = %w[alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima
               mike november oscar papa quebec romeo sierra tango uniform victor]
    data = words.each_with_index.map { |word, i| ['Ham', "#{word} #{words[(i + 1) % words.length]}"] } +
           words.each_with_index.map { |word, i| ['Spam', "#{word} #{words[(i + 2) % words.length]}"] }
    result = CV.cross_validate_with_audit('Bayes', data, 4)
    assert_equal 4, result[:conf_mats].length
    assert_equal [1, 2, 3, 4], result[:overlap][:training].map { |e| e[:fold] }.uniq.sort
    assert_equal data.length, result[:overlap][:counts][:holdout_total]
  end

  def test_lsi_audit_uses_the_rule_the_indexer_stores
    lsi = ClassifierReborn::LSI.new
    items = ['Green Tea Leaves', 'orchid blossom honey']
    items.each { |item| lsi.add_item(item, :sample) }
    items.each do |item|
      stored = lsi.instance_variable_get(:@items)[item].word_hash
      assert_equal stored, lsi.word_hash(item)
    end

    training = items.map { |item| ['Sample', item] }
    result = CV.audit_overlap(lsi, training, [['Sample', 'GREEN TEA LEAVES'], ['Sample', 'the and of']])
    assert_equal :classifier, result[:feature_extraction]
    assert_equal :overlap, entry(result, 0)[:status]
    assert_equal :empty_features, entry(result, 1)[:status]
  end

  def test_asking_for_a_word_hash_does_not_change_the_rule
    classifier = ClassifierReborn::Bayes.new('Ham', 'Spam', enable_stemmer: false)
    before = classifier.instance_variable_get(:@token_filters).dup
    5.times { classifier.word_hash('running dogs') }
    assert_equal before, classifier.instance_variable_get(:@token_filters)
  end

  def test_print_overlap_audit_lists_both_denominators
    out, = capture_io { CV.print_overlap_audit(audit) }
    assert_includes out, 'No comparable evidence (empty feature vectors)'
    assert_includes out, 'CONFLICT'
    assert_includes out, 'Full holdout sample'
    assert_includes out, 'Audited holdout denominator'
    assert_includes out, 'training source'
  end
end
