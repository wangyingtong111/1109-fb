# frozen_string_literal: true

require File.dirname(__FILE__) + '/../test_helper'
require_relative './bayesian_common_tests'

class BayesianMemoryTest < Minitest::Test
  include BayesianCommonTests

  def setup
    @alternate_backend = ClassifierReborn::BayesMemoryBackend.new
    @classifier = ClassifierReborn::Bayes.new 'Interesting', 'Uninteresting'
    @old_stopwords = TokenFilter::Stopword::STOPWORDS['en']
  end

  def teardown
    TokenFilter::Stopword::STOPWORDS['en'] = @old_stopwords
  end

  # A Marshal round-trip keeps the training records, so untraining still works
  # on a classifier that was restored from disk.
  def test_untrain_after_marshal_round_trip
    classifier = ClassifierReborn::Bayes.new 'Maps', 'Weather'
    classifier.train_maps 'bridge routes'
    restored = Marshal.load(Marshal.dump(classifier))

    assert_equal true, restored.untrain('Maps', 'bridge routes')
    assert_equal Float::INFINITY, restored.classifications('bridge routes')['Maps']
  end

  private

  def fresh_backend
    ClassifierReborn::BayesMemoryBackend.new
  end
end
