# frozen_string_literal: true

require File.dirname(__FILE__) + '/../test_helper'
require_relative './bayesian_common_tests'

class BayesianRedisTest < Minitest::Test
  include BayesianCommonTests

  def setup
    @old_stopwords = TokenFilter::Stopword::STOPWORDS['en']
    @backend = ClassifierReborn::BayesRedisBackend.new
    @backend.instance_variable_get(:@redis).config(:set, 'save', '')
    @alternate_backend = ClassifierReborn::BayesRedisBackend.new(db: 1)
    @classifier = ClassifierReborn::Bayes.new 'Interesting', 'Uninteresting', backend: @backend
  rescue Redis::CannotConnectError => e
    skip(e)
  end

  def teardown
    TokenFilter::Stopword::STOPWORDS['en'] = @old_stopwords
    if defined? @backend
      @backend.reset
      @alternate_backend.reset
      Array(@scratch_backends).each(&:reset)
    end
  end

  private

  # Every classifier in these tests needs its own keyspace, so each one gets a
  # backend on its own Redis database, all of which are flushed in teardown.
  def fresh_backend
    backend = ClassifierReborn::BayesRedisBackend.new(db: next_redis_db)
    (@scratch_backends ||= []) << backend
    backend
  end

  def next_redis_db
    @redis_db = (@redis_db || 1) + 1
  end
end
