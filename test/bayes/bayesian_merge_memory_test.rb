# frozen_string_literal: true

require File.dirname(__FILE__) + '/../test_helper'
require_relative './bayesian_merge_common_tests'

class BayesianMergeMemoryTest < Minitest::Test
  include BayesianMergeCommonTests

  def setup
    @old_stopwords = TokenFilter::Stopword::STOPWORDS['en']
  end

  def teardown
    TokenFilter::Stopword::STOPWORDS['en'] = @old_stopwords
  end

  def test_merged_model_can_be_marshaled
    shard_a = new_classifier('Maps', 'Weather', enable_stemmer: false)
    shard_a.train(:Maps, 'harbor harbor')
    shard_b = new_classifier('Maps', 'Weather', enable_stemmer: false)
    shard_b.train(:Weather, 'rain fog')
    merged = shard_a.merge(shard_b)

    restored = Marshal.load(Marshal.dump(merged))

    assert_equal merged.classifications('harbor rain'), restored.classifications('harbor rain')
    assert_equal 2, restored.backend.total_trainings
    assert_equal 2, restored.backend.category_word_frequency(:Maps, :harbor)
  end

  private

  def build_backend
    ClassifierReborn::BayesMemoryBackend.new
  end

  def new_classifier(*categories, **options)
    ClassifierReborn::Bayes.new(*categories, **options, backend: build_backend)
  end
end