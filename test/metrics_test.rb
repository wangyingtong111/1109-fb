require_relative "test_helper"

class MetricsTest < Minitest::Test
  def test_regression_metrics
    actual = [1, 2, 3]
    estimated = [1, 2, 9]
    metrics = Eps.metrics(actual, estimated)

    assert_in_delta 3.464, metrics[:rmse]
    assert_in_delta 2, metrics[:mae]
    assert_in_delta (-2), metrics[:me]
  end

  def test_regression_metrics_weight
    actual = [1, 1, 1]
    estimated = [0, 1, 4]
    weight = [1, 2, 3]
    metrics = Eps.metrics(actual, estimated, weight: weight)

    assert_in_delta 2.160246899469287, metrics[:rmse]
    assert_in_delta 1.6667, metrics[:mae]
    assert_in_delta (-1.3333), metrics[:me]
  end

  def test_classification_metrics
    actual = ["up", "up", "down"]
    estimated = ["down", "up", "down"]
    metrics = Eps.metrics(actual, estimated)

    assert_in_delta 0.667, metrics[:accuracy]
  end

  def test_classification_metrics_weight
    actual = ["down", "up", "down"]
    estimated = ["down", "up", "up"]
    weight = [1, 2, 3]
    metrics = Eps.metrics(actual, estimated, weight: weight)

    assert_in_delta 0.5, metrics[:accuracy]
  end

  def test_log_loss_weight
    log_loss = Eps::Metrics.log_loss([0, 1, 0], [0, 1, 1], weight: [1, 2, 3])
    assert_in_delta 17.26978799617044, log_loss
  end

  def test_log_loss_multiclass
    log_loss = Eps::Metrics.log_loss(["a", "c"], [{"a" => 0.2}, {"c" => 0.5}], weight: [1, 3])
    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4

    assert_in_delta expected, log_loss
  end

  def test_log_loss_multiclass_uses_true_label
    y_true = ["a", "c"]
    predictions = [
      {"a" => 0.2, "b" => 0.3, "c" => 0.5},
      {"a" => 0.1, "b" => 0.4, "c" => 0.5}
    ]
    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4

    assert_in_delta expected, Eps::Metrics.log_loss(y_true, predictions, weight: [1, 3])
  end

  def test_log_loss_multiclass_hash_order
    y_true = ["a", "c"]
    first = [{"a" => 0.2, "c" => 0.5}, {"c" => 0.5, "a" => 0.4}]
    second = [{"c" => 0.5, "a" => 0.2}, {"a" => 0.4, "c" => 0.5}]

    assert_in_delta Eps::Metrics.log_loss(y_true, first, weight: [1, 3]),
      Eps::Metrics.log_loss(y_true, second, weight: [1, 3])
  end

  def test_log_loss_multiclass_key_types
    # A symbol label finds a string key and vice versa
    log_loss = Eps::Metrics.log_loss([:a, :c], [{"a" => 0.2}, {"c" => 0.5}], weight: [1, 3])
    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4

    assert_in_delta expected, log_loss
  end

  def test_log_loss_missing_label
    error = assert_raises(ArgumentError) do
      Eps::Metrics.log_loss(["a"], [{"b" => 0.5, "c" => 0.5}])
    end

    assert_match(/Missing probability for label/, error.message)
  end

  def test_log_loss_invalid_probability
    assert_raises(ArgumentError) { Eps::Metrics.log_loss(["a"], [{"a" => 1.2}]) }
    assert_raises(ArgumentError) { Eps::Metrics.log_loss(["a"], [{"a" => -0.2}]) }
    assert_raises(ArgumentError) { Eps::Metrics.log_loss(["a"], [{"a" => Float::NAN}]) }
    assert_raises(ArgumentError) { Eps::Metrics.log_loss(["a"], [{"a" => Float::INFINITY}]) }
  end

  def test_log_loss_zero_probability_is_finite
    log_loss = Eps::Metrics.log_loss(["a"], [{"a" => 0.0}])

    assert log_loss.finite?
    assert_in_delta(-Math.log(Eps::Metrics::DEFAULT_EPS), log_loss)
  end

  def test_log_loss_old_binary
    # True label 0 selects the complement of the positive-class probability
    assert_in_delta(-Math.log(0.9), Eps::Metrics.log_loss([1, 0], [0.9, 0.1]))
    assert_in_delta(-Math.log(0.9), Eps::Metrics.log_loss([0], [0.1]))
    # A wrong probability of 0.9 for a true label of 0 costs -log(1 - 0.9)
    assert_in_delta(-Math.log(0.1), Eps::Metrics.log_loss([0], [0.9]))
  end

  def test_log_loss_legacy_exact
    # The pre-multiclass behavior is preserved bit-for-bit
    assert_equal 17.26978799617044, Eps::Metrics.log_loss([0, 1, 0], [0, 1, 1], weight: [1, 2, 3])
  end

  def test_log_loss_legacy_correct_probabilities
    # Two correct probabilities of 0.9 with 0/1 truth is still -log(0.9)
    log_loss = Eps::Metrics.log_loss([0, 1], [0.1, 0.9])

    assert_equal(-Math.log(0.9), log_loss)
  end

  def test_log_loss_one_shot_enumerators
    labels = %w[a c].each
    predictions = [{"a" => 0.2}, {"c" => 0.5}].each
    weights = [1, 3].each

    assert_in_delta((-Math.log(0.2) - 3 * Math.log(0.5)) / 4,
      Eps::Metrics.log_loss(labels, predictions, weight: weights))
  end

  def test_log_loss_segments
    chunks = [
      [["a"], [{"a" => 0.2}], [1]],
      [["c"], [{"c" => 0.5}], [3]]
    ]
    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4

    assert_in_delta expected, Eps::Metrics.log_loss_segments(chunks)
  end

  def test_log_loss_segments_with_zero_weight_chunk
    chunks = [
      [["a"], [{"a" => 0.2}], [1]],
      [["a"], [{"a" => 0.1}], [0]],
      [["c"], [{"c" => 0.5}], [3]]
    ]
    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4

    assert_in_delta expected, Eps::Metrics.log_loss_segments(chunks)
  end

  def test_log_loss_segments_order_independent
    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4
    chunks = [
      [["a"], [{"a" => 0.2}], [1]],
      [["a"], [{"a" => 0.1}], [0]],
      [["c"], [{"c" => 0.5}], [3]]
    ]

    chunks.permutation.each do |order|
      assert_in_delta expected, Eps::Metrics.log_loss_segments(order)
    end
  end

  def test_log_loss_segments_matches_whole_batch
    labels = ["a", "c", "a"]
    predictions = [{"a" => 0.2}, {"c" => 0.5}, {"a" => 0.1}]
    weights = [1, 3, 0]
    whole = Eps::Metrics.log_loss(labels, predictions, weight: weights)

    accumulator = Eps::Metrics.log_loss_accumulator
    labels.each_index do |i|
      accumulator.add([labels[i]], [predictions[i]], weight: [weights[i]])
    end

    assert_in_delta whole, accumulator.value
  end

  def test_log_loss_accumulator_not_averaged
    # Averaging the per-chunk metrics would give a different value
    accumulator = Eps::Metrics.log_loss_accumulator
    accumulator.add(["a"], [{"a" => 0.2}], weight: [1])
    accumulator.add(["c"], [{"c" => 0.5}], weight: [3])

    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4
    shortcut = (Math.log(1 / 0.2) + Math.log(1 / 0.5)) / 2

    assert_in_delta expected, accumulator.value
    refute_in_delta shortcut, accumulator.value
  end

  def test_log_loss_accumulator_segment_hashes
    accumulator = Eps::Metrics.log_loss_accumulator
    accumulator.add(y_true: ["a"], y_pred: [{"a" => 0.2}], weight: [1])
    accumulator.add(y_true: ["c"], y_pred: [{"c" => 0.5}], weight: [3])

    assert_in_delta((-Math.log(0.2) - 3 * Math.log(0.5)) / 4, accumulator.value)
  end

  def test_log_loss_accumulator_merge
    build = lambda do |label, prediction, weight|
      accumulator = Eps::Metrics.log_loss_accumulator
      accumulator.add([label], [prediction], weight: [weight])
      accumulator
    end

    left = build.call("a", {"a" => 0.2}, 1)
    middle = build.call("a", {"a" => 0.1}, 0)
    right = build.call("c", {"c" => 0.5}, 3)

    expected = (-Math.log(0.2) - 3 * Math.log(0.5)) / 4

    assert_in_delta expected, left.merge(middle).merge(right).value
    assert_in_delta expected,
      Eps::Metrics::LogLossAccumulator.combine([
        build.call("c", {"c" => 0.5}, 3),
        build.call("a", {"a" => 0.2}, 1),
        build.call("a", {"a" => 0.1}, 0)
      ]).value
  end

  def test_log_loss_accumulator_merge_does_not_mutate_inputs
    left = Eps::Metrics.log_loss_accumulator
    left.add(["a"], [{"a" => 0.2}], weight: [1])

    right = Eps::Metrics.log_loss_accumulator
    right.add(["c"], [{"c" => 0.5}], weight: [3])

    Eps::Metrics::LogLossAccumulator.combine([left, right])

    assert_in_delta(-Math.log(0.2), left.value)
    assert_in_delta(-Math.log(0.5), right.value)
  end

  def test_log_loss_accumulator_zero_weight_only_segment
    accumulator = Eps::Metrics.log_loss_accumulator
    accumulator.add(["a"], [{"a" => 0.2}], weight: [1])
    accumulator.add(["c"], [{"c" => 0.5}], weight: [0])

    # A zero-weight segment adds no loss and no denominator weight
    assert_in_delta(-Math.log(0.2), accumulator.value)
    assert_in_delta 1.0, accumulator.total_weight
    assert_equal 1, accumulator.effective_size
    assert_equal 2, accumulator.size
  end

  def test_log_loss_accumulator_all_zero_weight
    accumulator = Eps::Metrics.log_loss_accumulator
    accumulator.add(["a", "c"], [{"a" => 0.2}, {"c" => 0.5}], weight: [0, 0])

    assert accumulator.value.nan?
  end

  def test_log_loss_accumulator_retry_after_invalid_segment
    accumulator = Eps::Metrics.log_loss_accumulator
    accumulator.add(["a"], [{"a" => 0.2}], weight: [1])

    assert_raises(ArgumentError) do
      accumulator.add(["c"], [{"c" => 1.5}], weight: [3])
    end

    # Previous rows were not counted twice or lost
    assert_in_delta(-Math.log(0.2), accumulator.value)
    assert_equal 1, accumulator.size
    assert_in_delta 1.0, accumulator.total_weight

    # The corrected segment can be retried
    accumulator.add(["c"], [{"c" => 0.5}], weight: [3])
    assert_in_delta((-Math.log(0.2) - 3 * Math.log(0.5)) / 4, accumulator.value)
  end

  def test_log_loss_accumulator_invalid_weight_leaves_state
    accumulator = Eps::Metrics.log_loss_accumulator
    accumulator.add(["a"], [{"a" => 0.2}], weight: [1])

    assert_raises(ArgumentError) do
      accumulator.add(["c"], [{"c" => 0.5}], weight: [-1])
    end

    assert_in_delta(-Math.log(0.2), accumulator.value)
    assert_equal 1, accumulator.size
  end

  def test_metrics_multiclass_probabilities
    metrics = Eps.metrics(["a", "c"], [{"a" => 0.9, "c" => 0.1}, {"a" => 0.2, "c" => 0.8}])

    assert_in_delta((-Math.log(0.9) - Math.log(0.8)) / 2, metrics[:log_loss])
    assert_in_delta 1.0, metrics[:accuracy]
  end

  def test_metrics_multiclass_accuracy_tie_breaks_on_label
    metrics = Eps.metrics(["a"], [{"b" => 0.5, "a" => 0.5}])
    reversed = Eps.metrics(["a"], [{"a" => 0.5, "b" => 0.5}])

    assert_in_delta metrics[:accuracy], reversed[:accuracy]
    assert_in_delta 1.0, metrics[:accuracy]
  end

  def test_metrics_multiclass_enumerator_weight
    metrics = Eps.metrics(["a", "c"], [{"a" => 0.2}, {"c" => 0.5}], weight: [1, 3].each)

    assert_in_delta((-Math.log(0.2) - 3 * Math.log(0.5)) / 4, metrics[:log_loss])
  end

  def test_metrics_legacy_point_predictions
    assert_in_delta 0.6667, Eps.metrics(%w[up up down], %w[down up down])[:accuracy]
    assert_in_delta 2, Eps.metrics([1, 2, 3], [1, 2, 9])[:mae]
  end
end
