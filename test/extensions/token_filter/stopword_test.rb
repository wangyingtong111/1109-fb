# frozen_string_literal: true

require_relative '../../test_helper'
require 'tempfile'

class TokenFilterStopwordTest < Minitest::Test
  def setup
    @original_stopwords_path = TokenFilter::Stopword::STOPWORDS_PATH.dup
  end

  def test_default_stopwords
    refute_empty TokenFilter::Stopword::STOPWORDS['en']
    refute_empty TokenFilter::Stopword::STOPWORDS['fr']
    assert_empty TokenFilter::Stopword::STOPWORDS['gibberish']
  end

  def test_loads_custom_stopwords
    default_english_stopwords = TokenFilter::Stopword::STOPWORDS['en']

    # Remove the english stopwords
    TokenFilter::Stopword::STOPWORDS.delete('en')

    # Add a custom stopwords path
    TokenFilter::Stopword::STOPWORDS_PATH.unshift File.expand_path(File.dirname(__FILE__) + '/../../data/stopwords')

    custom_english_stopwords = TokenFilter::Stopword::STOPWORDS['en']

    refute_equal default_english_stopwords, custom_english_stopwords
  end

  def test_add_custom_stopword_path
    # Create stopword tempfile in current directory
    temp_stopwords = Tempfile.new('xy', (File.dirname(__FILE__) + '/').to_s)

    # Add some stopwords to tempfile
    temp_stopwords << 'this words fun'
    temp_stopwords.close

    # Get path of tempfile
    temp_stopwords_path = File.dirname(temp_stopwords)

    # Get tempfile name.
    temp_stopwords_name = File.basename(temp_stopwords.path)

    TokenFilter::Stopword.add_custom_stopword_path(temp_stopwords_path)
    words = Tokenizer::Whitespace.call('this is a list of cool words!')
    TokenFilter::Stopword.language = temp_stopwords_name
    words = TokenFilter::Stopword.call(words)
    assert_equal %w[list cool !], words
  end

  def test_per_instance_custom_stopwords_are_isolated
    bridge_filter = TokenFilter::StopwordFilter.new
    gravel_filter = TokenFilter::StopwordFilter.new
    bridge_filter.custom_stopwords = %w[bridge]
    gravel_filter.custom_stopwords = %w[gravel]

    tokens = Tokenizer::Whitespace.call('bridge gravel')

    assert_equal %w[gravel], bridge_filter.call(tokens.dup).map(&:to_s)
    assert_equal %w[bridge], gravel_filter.call(tokens.dup).map(&:to_s)
  end

  def test_custom_stopwords_on_one_instance_never_touch_another
    first = TokenFilter::StopwordFilter.new
    second = TokenFilter::StopwordFilter.new
    first.custom_stopwords = %w[bridge]

    refute_includes first.stopwords, 'gravel'
    assert_includes first.stopwords, 'bridge'
    # The second instance must still use the bundled language vocabulary
    assert_same TokenFilter::StopwordFilter::STOPWORDS['en'], second.stopwords
  end

  def test_per_instance_language_selection_is_isolated
    english = TokenFilter::StopwordFilter.new(language: 'en')
    french = TokenFilter::StopwordFilter.new(language: 'fr')

    assert_equal 'en', english.language
    assert_equal 'fr', french.language
    assert_same TokenFilter::StopwordFilter::STOPWORDS['en'], english.stopwords
    assert_same TokenFilter::StopwordFilter::STOPWORDS['fr'], french.stopwords
  end

  def test_empty_custom_stopwords_disables_filtering_for_that_instance
    disabled = TokenFilter::StopwordFilter.new
    disabled.custom_stopwords = []
    tokens = Tokenizer::Whitespace.call('the and for')

    assert_equal %w[the and for], disabled.call(tokens).map(&:to_s)
  end

  def test_bundled_stopwords_remain_available_without_custom_list
    filter = TokenFilter::StopwordFilter.new
    refute_empty filter.stopwords
    assert_same TokenFilter::StopwordFilter::STOPWORDS['en'], filter.stopwords
  end

  def teardown
    TokenFilter::Stopword::STOPWORDS.clear
    TokenFilter::Stopword::STOPWORDS_PATH.clear.concat @original_stopwords_path
    TokenFilter::Stopword.language = 'en'
  end
end
