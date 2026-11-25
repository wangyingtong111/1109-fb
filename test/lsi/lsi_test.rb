# frozen_string_literal: true

require File.dirname(__FILE__) + '/../test_helper'

class LSITest < Minitest::Test
  def setup
    # we repeat principle words to help weight them.
    # This test is rather delicate, since this system is mostly noise.
    @str1 = 'This text deals with dogs. Dogs.'
    @str2 = 'This text involves dogs too. Dogs! '
    @str3 = 'This text revolves around cats. Cats.'
    @str4 = 'This text also involves cats. Cats!'
    @str5 = 'This text involves birds. Birds.'
  end

  def test_basic_indexing
    lsi = ClassifierReborn::LSI.new
    [@str1, @str2, @str3, @str4, @str5].each { |x| lsi << x }
    assert !lsi.needs_rebuild?

    # note that the closest match to str1 is str2, even though it is not
    # the closest text match.
    assert_equal [@str2, @str5, @str3], lsi.find_related(@str1, 3)
  end

  def test_not_auto_rebuild
    lsi = ClassifierReborn::LSI.new auto_rebuild: false
    lsi.add_item @str1, 'Dog'
    lsi.add_item @str2, 'Dog'
    assert lsi.needs_rebuild?
    lsi.build_index
    assert !lsi.needs_rebuild?
  end

  def test_zero_vector_normalization
    lsi = ClassifierReborn::LSI.new auto_rebuild: false
    lsi.add_item @str1[0...8], 'Dog'
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.build_index(0.75)
  end

  def test_basic_categorizing
    lsi = ClassifierReborn::LSI.new
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'

    assert_equal 'Dog', lsi.classify(@str1)
    assert_equal 'Cat', lsi.classify(@str3)
    assert_equal 'Bird', lsi.classify(@str5)
  end

  def test_basic_categorizing_with_score
    lsi = ClassifierReborn::LSI.new
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'

    assert_in_delta 2.49, lsi.classify_with_score(@str1)[1], 0.1
    assert_in_delta 1.41, lsi.classify_with_score(@str3)[1], 0.1
    assert_in_delta 1.99, lsi.classify_with_score(@str5)[1], 0.1
  end

  def test_scored_categories
    lsi = ClassifierReborn::LSI.new
    lsi.add_item @str1, 'Dog'
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'

    scored_categories = lsi.scored_categories('dog bird cat')
    assert_equal 2, scored_categories.size
    assert_equal %w[Bird Dog], scored_categories.map(&:first)
  end

  def test_external_classifying
    lsi = ClassifierReborn::LSI.new
    bayes = ClassifierReborn::Bayes.new 'Dog', 'Cat', 'Bird'
    lsi.add_item @str1, 'Dog'
    bayes.train_dog @str1
    lsi.add_item @str2, 'Dog'
    bayes.train_dog @str2
    lsi.add_item @str3, 'Cat'
    bayes.train_cat @str3
    lsi.add_item @str4, 'Cat'
    bayes.train_cat @str4
    lsi.add_item @str5, 'Bird'
    bayes.train_bird @str5

    # We're talking about dogs. Even though the text matches the corpus on
    # cats better.  Dogs have more semantic weight than cats. So bayes
    # will fail here, but the LSI recognizes content.
    tricky_case = 'This text revolves around dogs.'
    assert_equal 'Dog', lsi.classify(tricky_case)
    refute_equal 'Dog', bayes.classify(tricky_case)
  end

  def test_recategorize_interface
    lsi = ClassifierReborn::LSI.new
    lsi.add_item @str1, 'Dog'
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'

    tricky_case = 'This text revolves around dogs.'
    assert_equal 'Dog', lsi.classify(tricky_case)

    # Recategorize as needed.
    lsi.categories_for(@str1).clear.push 'Cow'
    lsi.categories_for(@str2).clear.push 'Cow'

    assert !lsi.needs_rebuild?
    assert_equal 'Cow', lsi.classify(tricky_case)
  end

  def test_search
    lsi = ClassifierReborn::LSI.new
    [@str1, @str2, @str3, @str4, @str5].each { |x| lsi << x }

    # Searching by content and text, note that @str2 comes up first, because
    # both "dog" and "involve" are present. But, the next match is @str1 instead
    # of @str4, because "dog" carries more weight than involves.
    assert_equal([@str2, @str1, @str4, @str5, @str3],
                 lsi.search('dog involves', 100))

    # Keyword search shows how the space is mapped out in relation to
    # dog when magnitude is remove. Note the relations. We move from dog
    # through involve and then finally to other words.
    assert_equal([@str1, @str2, @str4, @str5, @str3],
                 lsi.search('dog', 5))
  end

  def test_serialize_safe
    lsi = ClassifierReborn::LSI.new
    [@str1, @str2, @str3, @str4, @str5].each { |x| lsi << x }

    lsi_md = Marshal.dump lsi
    lsi_m = Marshal.load lsi_md

    assert_equal lsi_m.search('cat', 3), lsi.search('cat', 3)
    assert_equal lsi_m.find_related(@str1, 3), lsi.find_related(@str1, 3)
  end

  def test_uncached_content_node_option
    lsi = ClassifierReborn::LSI.new
    [@str1, @str2, @str3, @str4, @str5].each { |x| lsi << x }
    lsi.instance_variable_get(:@items).values.each do |node|
      assert node.instance_of?(ContentNode)
    end
  end

  def test_cached_content_node_option
    lsi = ClassifierReborn::LSI.new(cache_node_vectors: true)
    [@str1, @str2, @str3, @str4, @str5].each { |x| lsi << x }
    lsi.instance_variable_get(:@items).values.each do |node|
      assert node.instance_of?(CachedContentNode)
    end
  end

  def test_clears_cached_content_node_cache
    skip "transposed_search_vector is only used by GSL implementation" if $SVD == :ruby

    lsi = ClassifierReborn::LSI.new(cache_node_vectors: true)
    lsi.add_item @str1, 'Dog'
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'

    assert_equal 'Dog', lsi.classify('something about dogs, but not an exact dog string')

    first_content_node = lsi.instance_variable_get(:@items).values.first
    refute_nil first_content_node.instance_variable_get(:@transposed_search_vector)
    lsi.clear_cache!
    assert_nil first_content_node.instance_variable_get(:@transposed_search_vector)
  end

  def test_keyword_search
    lsi = ClassifierReborn::LSI.new
    lsi.add_item @str1, 'Dog'
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'

    assert_equal %i[dog text deal], lsi.highest_ranked_stems(@str1)
  end

  def test_invalid_searching_with_linalg_lib
    skip "Only GSL currently raises invalid search error" if $SVD == :ruby

    lsi = ClassifierReborn::LSI.new
    lsi.add_item @str1, 'Dog'
    lsi.add_item @str2, 'Dog'
    lsi.add_item @str3, 'Cat'
    lsi.add_item @str4, 'Cat'
    lsi.add_item @str5, 'Bird'
    assert_output(/There are no documents that are similar to penguin/) { lsi.search('penguin') }
  end

  def test_warn_when_adding_bad_document
    lsi = ClassifierReborn::LSI.new
    assert_output(/Input: 'i can' is entirely stopwords or words with 2 or fewer characters. Classifier-Reborn cannot handle this document properly./) { lsi.add_item('i can') }
  end

  def test_summary
    assert_equal 'This text involves dogs too [...] This text also involves cats', Summarizer.summary([@str1, @str2, @str3, @str4, @str5].join, 2)
  end

  def test_reset
    lsi = ClassifierReborn::LSI.new
    assert lsi.items.empty?
    lsi.add_item @str1, 'Dog'
    refute lsi.items.empty?
    lsi.reset
    assert lsi.items.empty?
    lsi.add_item @str3, 'Cat'
    refute lsi.items.empty?
  end

  # Build an index where the proximity array is fully controlled by the test,
  # so the voting math can be asserted with exact similarity scores. Since the
  # proximity array is stubbed, no semantic index needs to be built.
  def build_stubbed_lsi(labels)
    lsi = ClassifierReborn::LSI.new(auto_rebuild: false)
    labels.each_with_index do |node_labels, i|
      lsi.add_item "document number #{i} with unique wording #{i}", *node_labels
    end
    lsi
  end

  def stub_proximities(lsi, proximities)
    lsi.define_singleton_method(:proximity_array_for_content) do |_doc, &_block|
      proximities
    end
    lsi
  end

  def test_quota_voting_splits_a_single_vote_across_unique_categories
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = build_stubbed_lsi(labels)
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    # Two neighbors are selected by the cutoff: 0.8 with [A,B,B] and 0.5 with
    # [B]. In quota mode the first neighbor splits its vote between A and B
    # once each, so the duplicated B does not buy a larger share.
    scores = lsi.scored_categories('query', 0.4, true).to_h
    assert_in_delta 0.4, scores['A'], 1e-9
    assert_in_delta 0.9, scores['B'], 1e-9
  end

  def test_quota_voting_does_not_transfer_hidden_shares_between_categories
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = build_stubbed_lsi(labels)
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    # Asking only about B must not reallocate the hidden A share, and B must
    # still gather 0.4 (from the split neighbor) + 0.5 = 0.9.
    b_score = lsi.scored_categories('query', 0.4, true).to_h['B']
    assert_in_delta 0.9, b_score, 1e-9
  end

  def test_quota_voting_ignores_unlabeled_neighbors_without_backfilling
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = build_stubbed_lsi(labels)
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    # The unlabeled neighbor (0.4) still occupies a cutoff slot without voting,
    # and the B neighbor outside the cutoff (0.3) is not pulled in to replace
    # it: B remains 0.9 rather than gaining the backfilled 0.3 share.
    scores = lsi.scored_categories('query', 0.4, true).to_h
    assert_equal %w[A B], scores.keys.sort
    assert_in_delta 0.9, scores['B'], 1e-9
  end

  def test_full_voting_remains_the_default_and_counts_duplicate_categories
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = build_stubbed_lsi(labels)
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    refute lsi.quota_voting
    scores = lsi.scored_categories('query', 0.4).to_h
    # Legacy whole-vote behavior: B appears twice on the 0.8 neighbor and once
    # on the 0.5 neighbor, so it collects 0.8 + 0.8 + 0.5 = 2.1.
    assert_in_delta 0.8, scores['A'], 1e-9
    assert_in_delta 2.1, scores['B'], 1e-9
  end

  def test_quota_voting_can_be_enabled_on_construction
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = ClassifierReborn::LSI.new(auto_rebuild: false, quota_voting: true)
    labels.each_with_index { |node_labels, i| lsi.add_item "document #{i}", *node_labels }
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    assert lsi.quota_voting
    scores = lsi.scored_categories('query', 0.4).to_h
    assert_in_delta 0.4, scores['A'], 1e-9
    assert_in_delta 0.9, scores['B'], 1e-9
  end

  def test_quota_voting_can_be_overridden_per_call
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = ClassifierReborn::LSI.new(auto_rebuild: false, quota_voting: true)
    labels.each_with_index { |node_labels, i| lsi.add_item "document #{i}", *node_labels }
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    # An instance configured for quota voting can still run a legacy full vote.
    scores = lsi.scored_categories('query', 0.4, false).to_h
    assert_in_delta 2.1, scores['B'], 1e-9
    assert_in_delta 0.8, scores['A'], 1e-9
  end

  def test_quota_voting_accepts_positional_argument_and_quota_alias
    labels = [['A', 'B', 'B'], ['B'], [], ['B'], ['A']]
    lsi = build_stubbed_lsi(labels)
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    stub_proximities(lsi, proximities)

    positional = lsi.scored_categories('query', 0.4, true).to_h
    assert_in_delta 0.9, positional['B'], 1e-9
    aliased = lsi.scored_categories('query', 0.4, quota: true).to_h
    assert_in_delta 0.9, aliased['B'], 1e-9
    assert_equal 'B', lsi.classify('query', 0.4, true)
    assert_equal ['B', 0.9], lsi.classify_with_score('query', 0.4, true)
  end

  def test_scored_categories_keeps_passing_the_content_block
    lsi = build_stubbed_lsi([['A', 'B', 'B'], ['B'], [], ['B'], ['A']])
    proximities = lsi.items.each_with_index.map { |item, i| [item, [0.8, 0.5, 0.4, 0.3, 0.2][i]] }
    received = nil
    lsi.define_singleton_method(:proximity_array_for_content) do |doc, &block|
      received = block.call(doc)
      proximities
    end

    lsi.scored_categories('external item', 0.4, true) { |item| "resolved #{item}" }
    assert_equal 'resolved external item', received
  end

  def test_quota_voting_survives_reset
    lsi = ClassifierReborn::LSI.new(quota_voting: true)
    lsi.add_item @str1, 'Dog'
    lsi.reset
    assert lsi.quota_voting
  end
end

