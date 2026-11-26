# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # AdjustedRandScore is a class that calculates the adjusted rand index.
    #
    # @example
    #   require 'rumale/evaluation_measure/adjusted_rand_score'
    #
    #   evaluator = Rumale::EvaluationMeasure::AdjustedRandScore.new
    #   puts evaluator.score(ground_truth, predicted)
    #
    # @example Incremental scoring
    #   evaluator = Rumale::EvaluationMeasure::AdjustedRandScore.new
    #   evaluator.add_batch(ground_truth_a, predicted_a)
    #   evaluator.add_batch(ground_truth_b, predicted_b)
    #   puts evaluator.score
    #
    # *Reference*
    # - Vinh, N X., Epps, J., and Bailey, J., "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance", J. Machine Learnig Research, Vol. 11, pp.2837--2854, 2010.
    class AdjustedRandScore
      include ::Rumale::Base::Evaluator

      # The number of samples appended so far.
      # @return [Integer]
      attr_reader :n_samples

      # The number of samples observed for each ground truth label.
      # @return [Hash{Object => Integer}]
      attr_reader :true_counts

      # The number of samples observed for each predicted cluster label.
      # @return [Hash{Object => Integer}]
      attr_reader :pred_counts

      # The number of samples observed for each [true_label, pred_label] pair.
      # @return [Hash{Array => Integer}]
      attr_reader :pair_counts

      alias class_counts true_counts
      alias cluster_counts pred_counts
      alias contingency pair_counts

      # Create a new evaluator, optionally appending an initial batch.
      #
      # @param y_true [Numo::Int32, Array, nil] Ground truth labels or label pairs.
      # @param y_pred [Numo::Int32, Array, nil] Predicted cluster labels.
      def initialize(y_true = nil, y_pred = nil)
        reset!
        add_batch(y_true, y_pred) unless y_true.nil? && y_pred.nil?
      end

      # Calculate adjusted rand index.
      #
      # When both labels are omitted, returns the index calculated from all
      # samples appended so far. Otherwise, calculates the index for the given
      # labels without changing the accumulated statistics.
      #
      # @param y_true [Numo::Int32, Array, nil] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32, Array, nil] (shape: [n_samples]) Predicted cluster labels.
      # @return [Float] Adjusted rand index.
      def score(y_true = nil, y_pred = nil)
        return current_score if y_true.nil? && y_pred.nil?

        self.class.new.add_batch(y_true, y_pred).current_score
      end

      # Append a batch of labels to the accumulated statistics.
      #
      # The batch can be given as two label arrays or as an enumerable of
      # [true_label, pred_label] pairs when only one argument is passed.
      #
      # @param y_true [Numo::Int32, Array] Ground truth labels or label pairs.
      # @param y_pred [Numo::Int32, Array, nil] Predicted cluster labels.
      # @return [AdjustedRandScore] This evaluator.
      def add_batch(y_true, y_pred = nil)
        true_labels, pred_labels = normalize_batch(y_true, y_pred)
        raise ArgumentError, 'y_true and y_pred must have the same size.' unless true_labels.size == pred_labels.size

        true_labels.size.times { |i| add_label_pair(true_labels[i], pred_labels[i]) }
        self
      end
      alias partial_fit add_batch
      alias append_batch add_batch
      alias update add_batch

      # Append a single label pair.
      #
      # @param true_label [Object] Ground truth label.
      # @param pred_label [Object] Predicted cluster label.
      # @return [AdjustedRandScore] This evaluator.
      def add_pair(true_label, pred_label)
        add_label_pair(true_label, pred_label)
        self
      end
      alias observe add_pair

      # Append a batch and return the adjusted rand index of all appended samples.
      #
      # @return [Float] Cumulative adjusted rand index.
      def partial_score(y_true, y_pred = nil)
        add_batch(y_true, y_pred)
        current_score
      end
      alias score_partial partial_score
      alias incremental_score partial_score

      # Return the adjusted rand index calculated from all appended samples.
      #
      # @return [Float] Cumulative adjusted rand index.
      def current_score
        n_classes = @true_counts.size
        n_clusters = @pred_counts.size

        return 1.0 if special_cases?(@n_samples, n_classes, n_clusters)

        prod_comb = (@sum_comb_true * @sum_comb_pred).fdiv(comb_two(@n_samples))
        mean_comb = (@sum_comb_true + @sum_comb_pred).fdiv(2)
        (@sum_comb_pair - prod_comb).fdiv(mean_comb - prod_comb)
      end
      alias ari current_score
      alias current_ari current_score
      alias adjusted_rand_index current_score

      # Remove all accumulated statistics.
      #
      # @return [AdjustedRandScore] This evaluator.
      def reset!
        @n_samples = 0
        @true_counts = Hash.new(0)
        @pred_counts = Hash.new(0)
        @pair_counts = Hash.new(0)
        @sum_comb_true = 0
        @sum_comb_pred = 0
        @sum_comb_pair = 0
        self
      end
      alias clear reset!

      private

      def add_label_pair(true_label, pred_label)
        @n_samples += 1

        true_count = @true_counts[true_label]
        @sum_comb_true += true_count
        @true_counts[true_label] = true_count + 1

        pred_count = @pred_counts[pred_label]
        @sum_comb_pred += pred_count
        @pred_counts[pred_label] = pred_count + 1

        pair = [true_label, pred_label]
        pair_count = @pair_counts[pair]
        @sum_comb_pair += pair_count
        @pair_counts[pair] = pair_count + 1
      end

      def normalize_batch(y_true, y_pred)
        return [to_label_array(y_true), to_label_array(y_pred)] unless y_pred.nil?

        rows = to_label_array(y_true)
        return [[], []] if rows.empty?
        raise ArgumentError, 'y_pred is required unless label pairs are given.' unless rows.all? { |row| pair_like?(row) }

        true_labels = []
        pred_labels = []
        rows.each do |row|
          values = pair_values(row)
          raise ArgumentError, 'Each label pair must contain exactly two labels.' unless values.size == 2

          true_labels << values[0]
          pred_labels << values[1]
        end
        [true_labels, pred_labels]
      end

      def to_label_array(labels)
        return labels if labels.is_a?(Array)
        return labels.to_a if labels.respond_to?(:to_a)

        [labels]
      end

      def pair_like?(value)
        value.respond_to?(:to_a)
      end

      def pair_values(row)
        row.is_a?(Array) ? row : row.to_a
      end

      def special_cases?(n_samples, n_classes, n_clusters)
        (n_classes.zero? && n_clusters.zero?) ||
          (n_classes == 1 && n_clusters == 1) ||
          (n_classes == n_samples && n_clusters == n_samples)
      end

      def comb_two(k)
        k * (k - 1) / 2
      end
    end
  end
end
