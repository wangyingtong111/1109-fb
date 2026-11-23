# frozen_string_literal: true

# Shared examples for the shard merging API of Bayes. The including test class
# has to provide two helpers:
#
#   build_backend      -> an empty backend of the same kind as @backend
#   new_classifier(*categories, **options) -> a classifier with a private backend
#
# so that the same scenarios run against the memory and the Redis backend.
module BayesianMergeCommonTests
  # Shard A: Maps "harbor harbor", Weather "rain fog".
  # Shard B: Maps "harbor route", Weather "fog route", Weather "rain route".
  def test_merge_matches_training_every_document_in_one_model
    shard_a = two_category_shard
    shard_a.train(:Maps, 'harbor harbor')
    shard_a.train(:Weather, 'rain fog')

    shard_b = two_category_shard
    shard_b.train(:Maps, 'harbor route')
    shard_b.train(:Weather, 'fog route')
    shard_b.train(:Weather, 'rain route')

    merged = merge_classifiers(shard_a, shard_b)

    # Sufficient statistics add up across shards.
    assert_equal 2, merged.backend.category_training_count(:Maps)
    assert_equal 3, merged.backend.category_training_count(:Weather)
    assert_equal 5, merged.backend.total_trainings
    assert_equal 10, merged.backend.total_words
    assert_equal 4, merged.backend.category_word_count(:Maps)
    assert_equal 6, merged.backend.category_word_count(:Weather)
    assert_equal 3, merged.backend.category_word_frequency(:Maps, :harbor)
    assert_equal 1, merged.backend.category_word_frequency(:Maps, :route)

    # The ranking and the score are those of a single model that learned all
    # five documents.
    whole = two_category_shard
    whole.train(:Maps, 'harbor harbor')
    whole.train(:Weather, 'rain fog')
    whole.train(:Maps, 'harbor route')
    whole.train(:Weather, 'fog route')
    whole.train(:Weather, 'rain route')

    assert_equal whole.classify_with_score('harbor rain'), merged.classify_with_score('harbor rain')
    assert_equal whole.classifications('harbor rain'), merged.classifications('harbor rain')
    assert_equal 'Maps', merged.classify('harbor rain')
    assert_equal whole.classifications('harbor harbor route'),
                 merged.classifications('harbor harbor route')
  end

  def test_merge_keeps_the_shards_untouched_and_usable
    shard_a = two_category_shard
    shard_a.train(:Maps, 'harbor harbor')
    shard_a.train(:Weather, 'rain fog')

    shard_b = two_category_shard
    shard_b.train(:Maps, 'harbor route')
    shard_b.train(:Weather, 'fog route')
    shard_b.train(:Weather, 'rain route')

    predictions_a = shard_a.classifications('harbor fog')
    predictions_b = shard_b.classifications('rain route')
    training_counts_a = [shard_a.backend.total_trainings, shard_a.backend.total_words]
    training_counts_b = [shard_b.backend.total_trainings, shard_b.backend.total_words]

    merged = merge_classifiers(shard_a, shard_b)

    assert_equal predictions_a, shard_a.classifications('harbor fog')
    assert_equal predictions_b, shard_b.classifications('rain route')
    assert_equal training_counts_a, [shard_a.backend.total_trainings, shard_a.backend.total_words]
    assert_equal training_counts_b, [shard_b.backend.total_trainings, shard_b.backend.total_words]
    assert_equal [2, 4], training_counts_a
    assert_equal [3, 6], training_counts_b
    assert_equal 1, shard_a.backend.category_training_count(:Maps)
    assert_equal 2, shard_a.backend.category_word_frequency(:Maps, :harbor)
    assert_equal 1, shard_b.backend.category_training_count(:Maps)

    # The shards keep working for their other callers: training one of them
    # after the merge must not leak into the merged model.
    shard_a.train(:Maps, 'harbor harbor harbor')
    assert_equal 2, merged.backend.category_training_count(:Maps)
    assert_equal 3, merged.backend.category_word_frequency(:Maps, :harbor)
  end

  def test_merge_result_owns_its_storage
    shard_a = two_category_shard
    shard_a.train(:Maps, 'harbor harbor')
    shard_b = two_category_shard
    shard_b.train(:Maps, 'harbor route')

    merged = merge_classifiers(shard_a, shard_b)

    refute_same shard_a.backend, merged.backend
    refute_same shard_b.backend, merged.backend
    refute_same shard_a.backend, shard_b.backend

    merged.train(:Maps, 'harbor harbor')
    assert_equal 5, merged.backend.category_word_frequency(:Maps, :harbor)
    assert_equal 2, shard_a.backend.category_word_frequency(:Maps, :harbor)
    assert_equal 1, shard_b.backend.category_word_frequency(:Maps, :harbor)
  end

  def test_merge_accepts_categories_only_one_shard_knows
    shard_a = new_classifier('Maps', 'Weather', enable_stemmer: false)
    shard_a.train(:Maps, 'harbor harbor')
    shard_b = new_classifier('Weather', 'Traffic', enable_stemmer: false)
    shard_b.train(:Weather, 'rain fog')
    shard_b.train(:Traffic, 'harbor jam')
    shard_b.train(:Traffic, 'route jam')

    merged = merge_classifiers(shard_a, shard_b)

    assert_equal %w[Maps Traffic Weather], merged.categories.sort
    assert_equal 1, merged.backend.category_training_count(:Maps)
    assert_equal 1, merged.backend.category_training_count(:Weather)
    assert_equal 2, merged.backend.category_training_count(:Traffic)
    # A word seen by both shards in different categories keeps both counts.
    assert_equal 2, merged.backend.category_word_frequency(:Maps, :harbor)
    assert_equal 1, merged.backend.category_word_frequency(:Traffic, :harbor)
  end

  def test_merge_of_more_than_two_shards
    shards = Array.new(3) { two_category_shard }
    shards[0].train(:Maps, 'harbor')
    shards[1].train(:Maps, 'harbor route')
    shards[2].train(:Weather, 'rain fog')

    whole = two_category_shard
    whole.train(:Maps, 'harbor')
    whole.train(:Maps, 'harbor route')
    whole.train(:Weather, 'rain fog')

    merged = merge_classifiers(*shards)

    assert_equal 2, merged.backend.category_training_count(:Maps)
    assert_equal 1, merged.backend.category_training_count(:Weather)
    assert_equal 2, merged.backend.category_word_frequency(:Maps, :harbor)
    assert_equal whole.classifications('harbor rain'), merged.classifications('harbor rain')
  end

  def test_merge_ignores_empty_shards
    shard_a = two_category_shard
    shard_a.train(:Maps, 'harbor harbor')
    empty_shard = two_category_shard

    merged = merge_classifiers(shard_a, empty_shard)

    assert_equal 1, merged.backend.total_trainings
    assert_equal 2, merged.backend.total_words
  end

  def test_merge_does_not_depend_on_the_order_documents_were_trained_in
    documents = [[:Maps, 'harbor harbor'], [:Weather, 'rain fog']]
    forward = two_category_shard
    documents.each { |category, text| forward.train(category, text) }
    reverse = two_category_shard
    documents.reverse_each { |category, text| reverse.train(category, text) }

    merged = merge_classifiers(forward, reverse)

    twice = two_category_shard
    documents.each { |category, text| twice.train(category, text) }
    documents.each { |category, text| twice.train(category, text) }

    assert_equal 2, merged.backend.category_training_count(:Maps)
    assert_equal twice.classifications('harbor rain'), merged.classifications('harbor rain')
  end

  def test_merge_requires_another_classifier
    classifier = two_category_shard
    classifier.train(:Maps, 'harbor')

    assert_raises(ArgumentError) { classifier.merge }
    assert_raises(ArgumentError) { classifier.merge(nil) }
  end

  def test_merge_requires_matching_features
    classifier = two_category_shard
    classifier.train(:Maps, 'harbor')

    assert_raises(StandardError) { classifier.merge('not a classifier', into: build_backend) }
  end

  def test_merge_rejects_stemmer_mismatch
    stemmed = new_classifier('Maps', enable_stemmer: true)
    stemmed.train(:Maps, 'harbors')
    plain = new_classifier('Maps', enable_stemmer: false)
    plain.train(:Maps, 'harbor')

    assert_raises(ClassifierReborn::Bayes::IncompatibleClassifierError) do
      merge_classifiers(stemmed, plain)
    end
  end

  def test_merge_rejects_tokenizer_mismatch
    bigrams = lambda do |str|
      str.each_char.each_cons(2).map { |chars| ClassifierReborn::Tokenizer::Token.new(chars.join) }
    end
    word_shard = new_classifier('Maps', enable_stemmer: false)
    word_shard.train(:Maps, 'harbor')
    bigram_shard = new_classifier('Maps', enable_stemmer: false, tokenizer: bigrams)
    bigram_shard.train(:Maps, 'harbor')

    assert_raises(ClassifierReborn::Bayes::IncompatibleClassifierError) do
      merge_classifiers(word_shard, bigram_shard)
    end
  end

  def test_merge_rejects_token_filter_mismatch
    upper = lambda { |tokens| tokens.reject { |token| token =~ /harbor/ } }
    plain = new_classifier('Maps', enable_stemmer: false)
    plain.train(:Maps, 'harbor')
    filtered = new_classifier('Maps', enable_stemmer: false,
                                      token_filters: [ClassifierReborn::TokenFilter::Stopword, upper])
    filtered.train(:Maps, 'route')

    assert_raises(ClassifierReborn::Bayes::IncompatibleClassifierError) do
      merge_classifiers(plain, filtered)
    end
  end

  def test_merge_accepts_shards_sharing_the_same_custom_stopwords
    first = new_classifier('Maps', enable_stemmer: false, stopwords: %w[harbor])
    first.train(:Maps, 'harbor route')
    second = new_classifier('Maps', enable_stemmer: false, stopwords: %w[harbor])
    second.train(:Maps, 'harbor harbor')

    merged = merge_classifiers(first, second)

    # "harbor" was filtered out by both shards, so it never became a feature.
    refute merged.backend.word_in_category?(:Maps, :harbor)
    assert_equal 1, merged.backend.category_word_frequency(:Maps, :route)
    assert_equal 1, merged.backend.total_words
  ensure
    TokenFilter::Stopword.language = 'en'
  end

  def test_merge_rejects_a_shard_trained_with_stopwords_that_were_later_replaced
    shard = new_classifier('Maps', enable_stemmer: false)
    shard.train(:Maps, 'harbor route')
    other = new_classifier('Maps', enable_stemmer: false)
    other.train(:Maps, 'rain fog')
    # Any later classifier for the same language rewrites the process wide list.
    new_classifier('Maps', enable_stemmer: false, stopwords: %w[harbor])

    assert_raises(ClassifierReborn::Bayes::IncompatibleClassifierError) do
      merge_classifiers(shard, other)
    end
  ensure
    TokenFilter::Stopword.language = 'en'
  end

  def test_merge_rejects_language_mismatch
    english = new_classifier('Maps', enable_stemmer: false)
    english.train(:Maps, 'harbor')
    french = new_classifier('Maps', enable_stemmer: false, language: 'fr')
    french.train(:Maps, 'port')

    assert_raises(ClassifierReborn::Bayes::IncompatibleClassifierError) do
      merge_classifiers(english, french)
    end
  ensure
    TokenFilter::Stopword.language = 'en'
  end

  def test_merge_rejects_shards_with_legacy_string_category_keys
    classifier = two_category_shard
    classifier.train(:Maps, 'harbor harbor')
    # Simulate a shard persisted before categories were normalised to symbols.
    legacy = new_classifier('Maps')
    legacy.backend.add_category('Maps')
    legacy.backend.update_category_training_count('Maps', 1)
    legacy.backend.update_category_word_count('Maps', 1)
    legacy.backend.update_category_word_frequency('Maps', :harbor, 1)
    legacy.backend.update_total_words(1)
    legacy.backend.update_total_trainings(1)

    assert_raises(ClassifierReborn::Bayes::IncompatibleClassifierError) do
      merge_classifiers(classifier, legacy)
    end
  end

  def test_merge_destination_must_be_empty
    classifier = two_category_shard
    classifier.train(:Maps, 'harbor')
    dirty = build_backend
    dirty.update_total_trainings(1)

    assert_raises(ArgumentError) { classifier.merge(two_category_shard, into: dirty) }
  end

  def test_merge_destination_must_not_be_a_shard_backend
    classifier = two_category_shard
    classifier.train(:Maps, 'harbor')

    assert_raises(ArgumentError) { classifier.merge(two_category_shard, into: classifier.backend) }
  end

  def test_class_level_merge
    shard_a = two_category_shard
    shard_a.train(:Maps, 'harbor harbor')
    shard_b = two_category_shard
    shard_b.train(:Weather, 'rain fog')

    merged = ClassifierReborn::Bayes.merge(shard_a, shard_b, into: build_backend)

    assert_equal 2, merged.backend.total_trainings
    assert_equal 4, merged.backend.total_words
    assert_raises(ArgumentError) { ClassifierReborn::Bayes.merge(shard_a) }
  end

  def test_combine_is_an_alias_of_merge
    shard_a = two_category_shard
    shard_a.train(:Maps, 'harbor')
    shard_b = two_category_shard
    shard_b.train(:Weather, 'rain')

    merged = shard_a.combine(shard_b, into: build_backend)

    assert_equal 2, merged.backend.total_trainings
  end

  def test_merge_keeps_threshold_settings
    shard_a = new_classifier('Maps', enable_stemmer: false, enable_threshold: true, threshold: -9.0)
    shard_a.train(:Maps, 'harbor')
    shard_b = new_classifier('Maps', enable_stemmer: false)
    shard_b.train(:Maps, 'route')

    merged = merge_classifiers(shard_a, shard_b)

    assert merged.threshold_enabled?
    assert_equal(-9.0, merged.threshold)
  end

  def test_merged_model_can_keep_training_like_its_shards
    shard_a = new_classifier('Maps', 'Weather', enable_stemmer: false, auto_categorize: true)
    shard_a.train(:Maps, 'harbor harbor')
    shard_b = new_classifier('Maps', 'Weather', enable_stemmer: false, auto_categorize: true)
    shard_b.train(:Weather, 'rain fog')

    merged = merge_classifiers(shard_a, shard_b)
    merged.train(:Traffic, 'harbor jam')

    assert_includes merged.categories, 'Traffic'
    assert_equal 1, merged.backend.category_training_count(:Traffic)
    assert_equal 1, merged.backend.category_word_frequency(:Traffic, :harbor)
    refute_includes shard_a.categories, 'Traffic'
    refute_includes shard_b.categories, 'Traffic'
  end

  def test_merge_resets_to_the_union_of_the_initial_categories
    shard_a = new_classifier('Maps', enable_stemmer: false)
    shard_a.train(:Maps, 'harbor')
    shard_b = new_classifier('Weather', enable_stemmer: false)
    shard_b.train(:Weather, 'rain')
    merged = merge_classifiers(shard_a, shard_b)

    merged.reset

    assert_equal %w[Maps Weather], merged.categories.sort
    assert_equal 0, merged.backend.total_trainings
  end

  def test_merge_rejects_the_same_shard_twice
    shard = two_category_shard
    shard.train(:Maps, 'harbor')

    assert_raises(ArgumentError) { merge_classifiers(shard, shard) }
  end

  private

  def two_category_shard
    new_classifier('Maps', 'Weather', enable_stemmer: false)
  end

  def merge_classifiers(*classifiers)
    classifiers.first.merge(*classifiers[1..-1], into: build_backend)
  end
end