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

  # The projection mode is tested on a deliberately truncated index: a single
  # singular direction (the maple term space) survives the SVD, so the orange
  # direction is discarded entirely. With six documents a cutoff of 1/6 keeps
  # exactly one singular value.
  def build_rank_one_lsi(**options)
    lsi = ClassifierReborn::LSI.new(auto_rebuild: false, **options)
    lsi.add_item 'maple syrup', 'Tree'
    lsi.add_item 'maple leaf', 'Tree'
    lsi.add_item 'maple tree', 'Tree'
    lsi.add_item 'maple wood', 'Tree'
    lsi.add_item 'maple bark', 'Tree'
    lsi.add_item 'orange citrus', 'Fruit'
    lsi.build_index(1.0 / 6.0)
    refute lsi.needs_rebuild?
    lsi
  end

  def projection_scores(lsi, query)
    lsi.send(:proximity_projection_for_content, query)
  end

  def test_projection_search_finds_dominant_direction
    lsi = build_rank_one_lsi

    results = lsi.search('maple', 10, projection: true)

    # Every maple document lies on the retained direction, so they are all
    # recovered ahead of the orange document.
    maple_documents = ['maple syrup', 'maple leaf', 'maple tree', 'maple wood', 'maple bark']
    assert_equal maple_documents.sort, results.first(5).sort
    assert_equal 'orange citrus', results.last
  end

  def test_projection_ignores_discarded_direction
    lsi = build_rank_one_lsi

    maple_scores = projection_scores(lsi, 'maple')
    mixed_scores = projection_scores(lsi, 'maple orange')

    # "orange" lives in a direction the truncation discarded, so adding it to
    # the query changes neither the projected vector nor the main-direction
    # scores.
    assert_equal maple_scores.map(&:last), mixed_scores.map(&:last)
    assert_equal maple_scores.map(&:first), mixed_scores.map(&:first)

    # A query made solely of discarded (or unknown) terms has no component in
    # the retained space, hence no valid projection at all.
    assert_empty lsi.search('orange', 10, projection: true)
    assert_empty projection_scores(lsi, 'orange')
    assert_empty lsi.search('penguin', 10, projection: true)
  end

  def test_projection_search_does_not_mutate_the_index
    lsi = build_rank_one_lsi

    word_list_before = lsi.word_list.instance_variable_get(:@location_table).dup
    version_before = lsi.instance_variable_get(:@version)
    vectors_before = lsi.instance_variable_get(:@items).values.map { |node| node.lsi_vector.to_a }

    projection_queries = ['maple', 'maple orange', 'orange', 'penguin']
    lexical_queries = ['maple', 'maple orange', 'orange']
    2.times do
      projection_queries.each { |query| lsi.search(query, 10, projection: true) }
      lexical_queries.each { |query| lsi.search(query, 10) }
    end

    assert_equal word_list_before, lsi.word_list.instance_variable_get(:@location_table)
    assert_equal version_before, lsi.instance_variable_get(:@version)
    assert_equal vectors_before,
                 lsi.instance_variable_get(:@items).values.map { |node| node.lsi_vector.to_a }
    refute lsi.needs_rebuild?
  end

  def test_projection_and_lexical_search_can_differ
    lsi = build_rank_one_lsi

    # The default query keeps the historical lexical-first semantics, while the
    # projection query is measured only in the retained truncated space. The
    # discarded "orange" term still contributes lexically, but vanishes under
    # projection.
    lexical_scores = lsi.send(:proximity_norms_for_content, 'maple orange')
    projection_scores = projection_scores(lsi, 'maple orange')
    refute_equal lexical_scores.map(&:last), projection_scores.map(&:last)

    assert_empty lsi.search('orange', 10, projection: true)
    refute_empty lsi.search('orange', 10)
  end

  def test_projection_search_is_serializable
    lsi = build_rank_one_lsi

    restored = Marshal.load(Marshal.dump(lsi))

    assert_equal lsi.search('maple', 10, projection: true),
                 restored.search('maple', 10, projection: true)
    assert_empty restored.search('orange', 10, projection: true)
  end

  def test_projection_search_with_cached_content_nodes
    lsi = build_rank_one_lsi(cache_node_vectors: true)
    assert lsi.instance_variable_get(:@items).values.all? { |node| node.is_a?(CachedContentNode) }

    refute_empty lsi.search('maple', 10, projection: true)
    assert_empty lsi.search('orange', 10, projection: true)
    assert_equal projection_scores(lsi, 'maple').map(&:last),
                 projection_scores(lsi, 'maple orange').map(&:last)
  end

  def test_projection_search_supports_a_content_block
    lsi = ClassifierReborn::LSI.new
    %w[dogs cats birds].each { |word| lsi.add_item("#{word} #{word}") }

    # The query key is not indexed text itself; the block supplies the text.
    results = lsi.search('/tmp/query.txt', 1, projection: true) { 'dogs' }

    assert_equal ['dogs dogs'], results
  end
end
