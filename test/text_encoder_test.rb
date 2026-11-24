require_relative "test_helper"

class TextEncoderTest < Minitest::Test
  def test_default_term_frequency
    encoder = Eps::TextEncoder.new(max_features: 1)
    encoder.fit(["apple apple apple apple", "pear", "pear pear"])
    assert_equal ["apple"], encoder.vocabulary
  end

  def test_document_frequency_without_documents
    encoder = Eps::TextEncoder.new(max_features: 1, document_frequency: true)
    encoder.fit(["apple apple apple apple", "pear", "pear pear"])
    assert_equal ["pear"], encoder.vocabulary
  end

  def test_document_frequency_with_documents
    encoder = Eps::TextEncoder.new(max_features: 1, document_frequency: true)
    encoder.fit(["a", "a a b", "a", "b b"], documents: ["A", "A", "A", "B"])
    assert_equal ["b"], encoder.vocabulary

    counts = encoder.transform(["a", "a a b", "a", "b b"])
    assert_equal [0, 1, 0, 2], counts.map { |c| c["b"] }
  end

  def test_document_frequency_order_invariant
    expected = nil
    [
      [["a", "a a b", "a", "b b"], ["A", "A", "A", "B"]],
      [["b b", "a", "a a b", "a"], ["B", "A", "A", "A"]],
      [["a", "b b", "a a b", "a"], ["A", "B", "A", "A"]]
    ].each do |lines, documents|
      encoder = Eps::TextEncoder.new(max_features: 1, document_frequency: true)
      encoder.fit(lines, documents: documents)
      expected ||= encoder.vocabulary
      assert_equal expected, encoder.vocabulary
    end
    assert_equal ["b"], expected
  end

  def test_document_frequency_documents_size_mismatch
    encoder = Eps::TextEncoder.new(document_frequency: true)
    error = assert_raises(ArgumentError) do
      encoder.fit(["a", "b"], documents: ["A"])
    end
    assert_equal "Number of documents differs from number of lines", error.message
  end

  def test_transform_does_not_expand_vocabulary
    encoder = Eps::TextEncoder.new(max_features: 1, document_frequency: true)
    encoder.fit(["a", "a a b", "a", "b b"], documents: ["A", "A", "A", "B"])
    assert_equal ["b"], encoder.vocabulary

    # validation set with same document identifiers
    encoder.transform(["a a a", "c c c"])
    assert_equal ["b"], encoder.vocabulary
  end

  def test_document_frequency_min_max_occurrences
    encoder = Eps::TextEncoder.new(document_frequency: true, min_occurrences: 2)
    encoder.fit(["a", "a a b", "a", "b b"], documents: ["A", "A", "A", "B"])
    assert_equal ["b"], encoder.vocabulary

    encoder = Eps::TextEncoder.new(document_frequency: true, max_occurrences: 1)
    encoder.fit(["a", "a a b", "a", "b b"], documents: ["A", "A", "A", "B"])
    assert_equal ["a"], encoder.vocabulary
  end
end
