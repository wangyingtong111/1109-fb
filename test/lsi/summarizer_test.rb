# frozen_string_literal: true

require File.dirname(__FILE__) + '/../test_helper'

class SummarizerTest < Minitest::Test
  LEAF_ONE = 'The green leaves on the oak tree rustled in the wind all afternoon.'
  LEAF_TWO = 'Green foliage of the oak tree swayed in the breeze throughout the afternoon.'
  DRIVING  = 'Drive straight for two miles then turn left at the stone bridge.'

  def test_sentence_summary_keeps_its_original_result
    str1 = 'This text deals with dogs. Dogs.'
    str2 = 'This text involves dogs too. Dogs! '
    str3 = 'This text revolves around cats. Cats.'
    str4 = 'This text also involves cats. Cats!'
    str5 = 'This text involves birds. Birds.'

    result = Summarizer.summary([str1, str2, str3, str4, str5].join, 2)
    assert_equal 'This text involves dogs too [...] This text also involves cats', result
    assert_instance_of String, result
  end

  def test_paragraph_summary_defaults_to_centrality_and_returns_a_string
    document = [LEAF_ONE, LEAF_TWO, DRIVING].join("\n\n")
    result = Summarizer.paragraph_summary(document, 2)

    assert_instance_of String, result
    # Without a redundancy weight the two most central (leaf themed)
    # paragraphs win, even though they repeat the same theme.
    assert_includes result, 'green leaves'
    assert_includes result, 'Green foliage'
    refute_includes result, 'Drive straight'
  end

  def test_two_dissimilar_paragraphs_are_both_kept_in_document_order
    document = [LEAF_ONE, DRIVING].join("\n\n")
    result = Summarizer.paragraph_summary(document, 2, ' | ')

    assert_equal "#{LEAF_ONE} | #{DRIVING}", result
  end

  def test_redundancy_weight_prefers_an_independent_paragraph
    document = [LEAF_ONE, LEAF_TWO, DRIVING].join("\n\n")
    result = Summarizer.paragraph_summary(document, 2, ' | ', redundancy_weight: 1.0)

    # One leaf themed paragraph plus the unrelated driving paragraph, in the
    # order they appear in the source text.
    assert_equal "#{LEAF_TWO} | #{DRIVING}", result
  end

  def test_diversity_weight_is_an_alias_for_redundancy_weight
    document = [LEAF_ONE, LEAF_TWO, DRIVING].join("\n\n")
    result = Summarizer.paragraph_summary(document, 2, ' | ', diversity_weight: 1.0)

    assert_equal "#{LEAF_TWO} | #{DRIVING}", result
  end

  def test_identical_paragraphs_keep_their_own_positions
    document = [LEAF_ONE, LEAF_ONE, DRIVING].join("\n\n")

    # One slot only returns a single occurrence.
    single = Summarizer.paragraph_summary(document, 1, ' | ', redundancy_weight: 1.0)
    assert_equal DRIVING, single

    # When both identical paragraphs are needed, each position is returned and
    # the driving paragraph still appears, all in document order.
    full = Summarizer.paragraph_summary(document, 3, ' | ', redundancy_weight: 1.0)
    assert_equal "#{LEAF_ONE} | #{LEAF_ONE} | #{DRIVING}", full
  end

  def test_only_one_distinct_paragraph_repeats_without_errors
    document = [LEAF_ONE, LEAF_ONE, LEAF_ONE].join("\n\n")
    result = Summarizer.paragraph_summary(document, 2, ' | ', redundancy_weight: 1.0)

    assert_equal "#{LEAF_ONE} | #{LEAF_ONE}", result
  end

  def test_empty_and_single_word_paragraphs_are_ignored
    assert_equal '', Summarizer.paragraph_summary("\n\n  \n\n", 2)

    document = "hello\n\n#{LEAF_ONE}\n\n#{DRIVING}"
    result = Summarizer.paragraph_summary(document, 2, ' | ')

    refute_includes result, 'hello'
    assert_includes result, LEAF_ONE
    assert_includes result, DRIVING
  end
end
