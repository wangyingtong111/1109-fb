# frozen_string_literal: true

require File.dirname(__FILE__) + '/../test_helper'
require_relative './bayesian_merge_common_tests'

class BayesianMergeRedisTest < Minitest::Test
  include BayesianMergeCommonTests

  # Databases 0 and 1 belong to the other Redis backed tests.
  FIRST_DATABASE = 2

  def setup
    @old_stopwords = TokenFilter::Stopword::STOPWORDS['en']
    @backends = []
  rescue Redis::CannotConnectError => e
    skip(e)
  end

  def teardown
    TokenFilter::Stopword::STOPWORDS['en'] = @old_stopwords
    (@backends || []).each(&:reset)
  end

  private

  def build_backend
    backend = ClassifierReborn::BayesRedisBackend.new(db: FIRST_DATABASE + @backends.size)
    @backends << backend
    backend
  rescue NoRedisError, Redis::CannotConnectError => e
    skip(e)
  end

  def new_classifier(*categories, **options)
    ClassifierReborn::Bayes.new(*categories, **options, backend: build_backend)
  end
end