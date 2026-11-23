# frozen_string_literal: true

require_relative '../test_helper'
require_relative '../../lib/classifier-reborn/validators/classifier_validator'

# A deterministic stand-in for a threshold enabled classifier. It replays a
# scripted prediction (including +nil+ for threshold rejections) for each
# validation text in order and reports only the categories it was trained on.
class ScriptedClassifier
  attr_reader :categories

  def initialize(categories, predictions)
    @categories = categories
    @predictions = predictions.dup
  end

  def reset
    # Nothing to reset; predictions are scripted per validation run.
  end

  def train(category, _text)
    @categories << category unless @categories.include?(category)
  end

  def classify(_text)
    @predictions.shift
  end
end

class ClassifierValidatorTest < Minitest::Test
  include ClassifierReborn::ClassifierValidator

  # Fold 1: three validation texts -> one correct, one incorrect, one refusal.
  FOLD_ONE_TEST = [
    ['cat', 'correctly a cat'],
    ['cat', 'actually a cat but called dog'],
    ['dog', 'below the decision threshold']
  ].freeze

  # The model answering fold 1 was trained on Cat and Dog only.
  FOLD_ONE_PREDICTIONS = ['Cat', 'Dog', nil].freeze

  # Fold 2: one validation text whose real category is literally "拒判".
  # The other fold never trained this category, so it is unseen at training time.
  FOLD_TWO_TEST = [
    ['拒判', 'a genuinely abstention named class']
  ].freeze

  FOLD_TWO_PREDICTIONS = ['拒判'].freeze

  def setup
    @fold_one = evaluate(
      ScriptedClassifier.new(['Cat', 'Dog'], FOLD_ONE_PREDICTIONS),
      FOLD_ONE_TEST
    )
    @fold_two = evaluate(
      ScriptedClassifier.new(['Cat', 'Dog'], FOLD_TWO_PREDICTIONS),
      FOLD_TWO_TEST
    )
  end

  def test_abstained_is_a_distinct_identity_that_never_matches_a_real_label
    assert_same ClassifierReborn::ClassifierValidator::ABSTAINED,
                ClassifierReborn::ClassifierValidator::ABSTAINED
    refute_equal '拒判', ClassifierReborn::ClassifierValidator::ABSTAINED
    refute_equal 'Abstained', ClassifierReborn::ClassifierValidator::ABSTAINED
    refute_equal :Abstained, ClassifierReborn::ClassifierValidator::ABSTAINED
    assert_equal 'Abstained', ClassifierReborn::ClassifierValidator::ABSTAINED.to_s
  end

  def test_fold_one_records_correct_incorrect_and_abstained_separately
    abstained = ClassifierReborn::ClassifierValidator::ABSTAINED
    assert_equal(
      {
        'Cat' => { 'Cat' => 1, 'Dog' => 1, abstained => 0 },
        'Dog' => { 'Cat' => 0, 'Dog' => 0, abstained => 1 }
      },
      @fold_one
    )

    stats = build_run_report(@fold_one)
    assert_equal 3, stats[:total]
    assert_equal 1, stats[:correct]
    assert_equal 1, stats[:incorrect]
    assert_equal 1, stats[:abstained]
    assert_equal 2, stats[:answered]
    assert_in_delta 1.0 / 3, stats[:accuracy]
    assert_in_delta 0.5, stats[:answered_accuracy]
    assert_in_delta 2.0 / 3, stats[:coverage]
  end

  def test_fold_two_keeps_unseen_true_class_as_a_row
    stats = build_run_report(@fold_two)

    # The unseen true class survives as its own confusion row even though the
    # trained model only knew Cat and Dog.
    assert_includes @fold_two.keys, '拒判'
    assert_equal 1, @fold_two['拒判']['拒判']
    # Model categories are still reported as rows (with zero counts).
    assert_equal %w[Cat Dog], (@fold_two.keys - ['拒判'])
    # No refusal happened in this fold.
    refute @fold_two['拒判'][ClassifierReborn::ClassifierValidator::ABSTAINED].to_i.positive?

    assert_equal 1, stats[:total]
    assert_equal 1, stats[:correct]
    assert_equal 0, stats[:abstained]
    assert_in_delta 1.0, stats[:accuracy]
    assert_in_delta 1.0, stats[:coverage]
  end

  def test_merge_accumulates_raw_counts_and_merges_in_any_order
    forward = merge_conf_mats([@fold_one, @fold_two])
    reverse = merge_conf_mats([@fold_two, @fold_one])
    assert_equal forward, reverse

    abstained = ClassifierReborn::ClassifierValidator::ABSTAINED
    expected = {
      'Cat' => { 'Cat' => 1, 'Dog' => 1, '拒判' => 0, abstained => 0 },
      'Dog' => { 'Cat' => 0, 'Dog' => 0, '拒判' => 0, abstained => 1 },
      '拒判' => { 'Cat' => 0, 'Dog' => 0, '拒判' => 1, abstained => 0 }
    }
    assert_equal expected, forward
    # Deterministic key order, with ABSTAINED always rendered last.
    assert_equal ['Cat', 'Dog', '拒判'], forward.keys
    assert_equal ['Cat', 'Dog', '拒判', abstained], forward.values.first.keys
  end

  def test_merged_metrics_match_the_required_denominators
    merged = merge_conf_mats([@fold_one, @fold_two])
    stats = build_run_report(merged)

    assert_equal 4, stats[:total]
    assert_equal 2, stats[:correct]
    assert_equal 1, stats[:incorrect]
    assert_equal 1, stats[:abstained]
    assert_equal 3, stats[:answered]

    # Full sample accuracy 1/2, answered accuracy 2/3, coverage 3/4.
    assert_in_delta 1.0 / 2, stats[:accuracy]
    assert_in_delta 2.0 / 3, stats[:answered_accuracy]
    assert_in_delta 3.0 / 4, stats[:coverage]
  end

  def test_report_is_identical_regardless_of_fold_merge_order
    forward = capture_report { generate_report([@fold_one, @fold_two]) }
    reverse = capture_report { generate_report([@fold_two, @fold_one]) }

    # The per-fold lines are numbered by merge order, so compare everything
    # from the accumulated "All" line onward: matrix and denominators must be
    # identical no matter which fold is merged first.
    forward_summary = summary_from(forward)
    reverse_summary = summary_from(reverse)
    assert_equal forward_summary, reverse_summary

    # Both the real "拒判" class and the separate abstention column appear.
    assert_includes forward_summary, '拒判'
    assert_includes forward_summary, 'Abstained'
  end

  def test_legacy_two_level_hash_interface_still_serves_category_confusion
    # An older style matrix without the ABSTAINED column keeps working through
    # the existing reporting/statistics interface.
    legacy = { 'Ham' => { 'Ham' => 8, 'Spam' => 2 }, 'Spam' => { 'Ham' => 1, 'Spam' => 9 } }
    stats = build_run_report(legacy)
    assert_equal 20, stats[:total]
    assert_equal 17, stats[:correct]
    assert_equal 3, stats[:incorrect]
    assert_equal 0, stats[:abstained]

    tab = conf_mat_to_tab(legacy)
    assert_equal 8, tab['Ham'][:p][:t]
    assert_equal 9, tab['Spam'][:p][:t]
    assert_equal 9, tab['Ham'][:n][:t]

    printed = capture_report { print_conf_mat(legacy) }
    assert_includes printed, 'Confusion Matrix'
    assert_includes printed, 'Ham'
    assert_includes printed, 'Spam'
  end

  private

  def summary_from(report)
    report.split("\n").drop_while { |line| !line.start_with?('All') }.join("\n")
  end

  def capture_report
    original_stdout = $stdout
    $stdout = StringIO.new
    yield
    $stdout.string
  ensure
    $stdout = original_stdout
  end
end
