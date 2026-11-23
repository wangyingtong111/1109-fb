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

  # --- Highest-weighted term explanations ---------------------------------

  # Three documents with overlapping vocabularies but different word orders.
  # :doc_a is the document being explained; its three content words appear
  # 3, 3 and 1 times respectively.
  def build_explanation_lsi(cached)
    lsi = ClassifierReborn::LSI.new(cache_node_vectors: cached)
    lsi.add_item(:doc_b) { 'cat dog bird fish cat bird' }
    lsi.add_item(:doc_c) { 'bird horse dog cat bird horse' }
    lsi.add_item(:doc_a) { 'dog dog dog cat cat cat bird' }
    lsi
  end

  def item_node(lsi, item)
    lsi.instance_variable_get(:@items)[item]
  end

  def replace_lsi_vector(node, coords)
    node.lsi_vector =
      case $SVD
      when :numo then Numo::DFloat[*coords]
      when :gsl then GSL::Vector[*coords]
      else Vector[*coords]
      end
  end

  def test_highest_ranked_stems_with_scores_handles_ties
    [false, true].each do |cached|
      lsi = build_explanation_lsi(cached)
      node = item_node(lsi, :doc_a)

      counts = node.word_hash
      assert_equal [3, 3, 1], counts.values
      stem3a, stem3b, stem1 = counts.keys

      # Give the document a vector whose three explanation weights are 3, 3, 1.
      coords = Array.new(lsi.word_list.size, 0.0)
      coords[lsi.word_list[stem3a]] = 3.0
      coords[lsi.word_list[stem3b]] = 3.0
      coords[lsi.word_list[stem1]] = 1.0
      replace_lsi_vector(node, coords)

      tied = [stem3a, stem3b].sort_by { |word| lsi.word_list[word] }

      # Taking two results yields two different words with the same score,
      # instead of one tied word shadowing the other.
      top2 = lsi.highest_ranked_stems_with_scores(:doc_a, 2)
      assert_equal tied, top2.map(&:first)
      assert_equal [3.0, 3.0], top2.map(&:last)

      # The full explanation carries the 3, 3, 1 weights in stable order.
      top3 = lsi.highest_ranked_stems_with_scores(:doc_a, 3)
      assert_equal [[tied[0], 3.0], [tied[1], 3.0], [stem1, 1.0]], top3

      # The original interface still returns just the word list.
      assert_equal [tied[0], tied[1], stem1], lsi.highest_ranked_stems(:doc_a, 3)

      # Every weight sits at its word's position in the node's vector.
      top3.each do |word, weight|
        assert_equal weight, node.search_vector.to_a[lsi.word_list[word]]
      end
    end
  end

  def test_explanations_survive_vocabulary_regeneration
    explanations = {}
    [false, true].each do |cached|
      lsi = build_explanation_lsi(cached)
      old_dog_position = lsi.word_list[:dog]

      # Remove another document and introduce a brand new word that lands
      # at the front of the vocabulary, then rebuild the index.
      lsi.remove_item(:doc_c)
      lsi.add_item(:doc_b) { 'aardvark cat dog bird' }
      assert !lsi.needs_rebuild?

      new_stem = item_node(lsi, :doc_b).word_hash.keys.first
      assert_equal 0, lsi.word_list[new_stem], 'new word should land at the front of the vocabulary'
      refute_equal old_dog_position, lsi.word_list[:dog], 'word positions should have shifted'

      # Re-query the old document: the first explanation after the rebuild
      # must already use the new generation of word positions.
      explained = lsi.highest_ranked_stems_with_scores(:doc_a, lsi.word_list.size)
      refute_empty explained
      assert_equal explained, lsi.highest_ranked_stems_with_scores(:doc_a, lsi.word_list.size),
                   'explanations should be stable across repeated calls'

      node = item_node(lsi, :doc_a)
      coords = node.search_vector.to_a
      explained.each do |word, weight|
        position = lsi.word_list[word]
        refute_nil position
        assert_in_delta coords[position], weight, 1e-9,
                        "weight of #{word} must belong to its current word position"
      end

      if cached
        # The word list generation survives a Marshal round trip as well.
        restored = Marshal.load(Marshal.dump(lsi))
        assert_equal explained, restored.highest_ranked_stems_with_scores(:doc_a, restored.word_list.size)
      end

      explanations[cached] = explained
    end

    # Plain and cached content nodes return the same verifiable explanation.
    assert_equal explanations[false].map(&:first), explanations[true].map(&:first)
    explanations[false].zip(explanations[true]).each do |(_, plain_weight), (_, cached_weight)|
      assert_in_delta plain_weight, cached_weight, 1e-9
    end
  end

  def test_explanation_uses_word_list_generation_that_built_the_vector
    [ClassifierReborn::ContentNode, ClassifierReborn::CachedContentNode].each do |node_class|
      gen1 = ClassifierReborn::WordList.new
      %i[dog cat bird].each { |word| gen1.add_word(word) }
      gen2 = ClassifierReborn::WordList.new
      %i[aardvark dog cat bird].each { |word| gen2.add_word(word) }

      node = node_class.new(dog: 3, cat: 3, bird: 1)
      node.raw_vector_with(gen1)
      assert_same gen1, node.vector_word_list

      # Even when a newer-generation word list is offered as fallback, the
      # cached vector must be explained with the positions of the word list
      # generation that produced it, never with another generation's.
      explained = node.highest_weighted_terms(3, gen2)
      coords = node.search_vector.to_a
      assert_equal %i[dog cat bird], explained.map(&:first)
      explained.each_with_index do |(_, weight), i|
        assert_equal coords[i], weight
      end

      # A node whose vector was never built has a deterministic empty explanation.
      assert_equal [], node_class.new(dog: 1).highest_weighted_terms(3, gen1)
    end
  end

  def test_highest_ranked_stems_edge_cases
    [false, true].each do |cached|
      lsi = build_explanation_lsi(cached)

      # A miss raises, on both interfaces.
      error = assert_raises(RuntimeError) { lsi.highest_ranked_stems(:no_such_doc) }
      assert_match(/non-indexed/, error.message)
      assert_raises(RuntimeError) { lsi.highest_ranked_stems_with_scores(:no_such_doc) }

      # Zero or fewer requested stems deterministically yields nothing.
      assert_equal [], lsi.highest_ranked_stems(:doc_a, 0)
      assert_equal [], lsi.highest_ranked_stems_with_scores(:doc_a, 0)
      assert_equal [], lsi.highest_ranked_stems_with_scores(:doc_a, -2)

      # Requesting more stems than words yields one pair per word position.
      vocab_size = lsi.word_list.size
      all = lsi.highest_ranked_stems_with_scores(:doc_a, vocab_size + 10)
      assert_equal vocab_size, all.size
      assert_equal all, lsi.highest_ranked_stems_with_scores(:doc_a, vocab_size)
      assert_equal vocab_size, all.map(&:first).uniq.size
      assert_equal all, lsi.highest_ranked_stems_with_scores(:doc_a, vocab_size + 10)
    end
  end
end
