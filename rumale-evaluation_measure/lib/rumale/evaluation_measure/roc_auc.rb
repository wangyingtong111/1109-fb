# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # ROCAUC is a class that calculate area under the receiver operation characteristic curve from predicted scores.
    #
    # @example
    #   require 'rumale/preprocessing'
    #   require 'rumale/linear_model'
    #   require 'rumale/evaluation_measure/roc_auc'
    #
    #   # Encode labels to integer array.
    #   labels = %w[A B B C A A C C C A]
    #   label_encoder = Rumale::Preprocessing::LabelEncoder.new
    #   y = label_encoder.fit_transform(labels)
    #   # Fit classifier.
    #   classifier = Rumale::LinearModel::LogisticRegression.new
    #   classifier.fit(x, y)
    #   # Predict class probabilities.
    #   y_score = classifier.predict_proba(x)
    #   # Encode labels to one-hot vectors.
    #   one_hot_encoder = Rumale::Preprocessing::OneHotEncoder.new
    #   y_onehot = one_hot_encoder.fit_transform(y)
    #   # Calculate ROC AUC.
    #   evaluator = Rumale::EvaluationMeasure::ROCAUC.new
    #   puts evaluator.score(y_onehot, y_score)
    #
    #   # Calculate ROC AUC from a ground truth label vector and the class labels of the score columns.
    #   # The column order of the predicted scores is fixed by the model, and a validation fold
    #   # may not contain some of the classes.
    #   y_true = Numo::Int32[10, 20, 10]
    #   y_score = Numo::DFloat[[0.05, 0.88, 0.07], [0.03, 0.12, 0.85], [0.08, 0.80, 0.12]]
    #   column_labels = [30, 10, 20]
    #   p evaluator.score_each_class(y_true, y_score, column_labels)
    #   # => {30=>nil, 10=>1.0, 20=>1.0}
    #   puts evaluator.score(y_true, y_score, column_labels, skip_undefined: true)
    #   # => 1.0
    #
    #   # The column labels and the skipping rule can also be given to the constructor.
    #   evaluator = Rumale::EvaluationMeasure::ROCAUC.new(labels: column_labels, skip_undefined: true)
    #   puts evaluator.score(y_true, y_score)
    #   # => 1.0
    class ROCAUC
      include ::Rumale::Base::Evaluator

      # Return the labels of the predicted score columns.
      # @return [Array/Nil] The column labels.
      attr_reader :labels

      # Return the average type for calculation of ROC AUC.
      # @return [String] ('macro', 'none')
      attr_reader :average

      # Return the flag indicating whether the classes without positive or negative examples
      # are ignored in the macro average.
      # @return [Boolean]
      attr_reader :skip_undefined

      # Create a new evaluation measure calculator for ROC AUC.
      #
      # @param average [String] The average type ('macro' or 'none') of the ROC AUC for the classes.
      # @param skip_undefined [Boolean] Whether to ignore the classes that have no positive or
      #   negative examples in y_true when calculating the macro average.
      # @param labels [Array/Nil] Class labels corresponding to the columns of the predicted scores.
      #   These are used as the default column labels of the score method.
      def initialize(average: 'macro', skip_undefined: false, labels: nil)
        @average = average
        @skip_undefined = skip_undefined
        @labels = labels
      end

      # Calculate area under the receiver operation characteristic curve (ROC AUC).
      #
      # When y_true is a one-dimensional label vector and y_score is a two-dimensional score matrix,
      # each column of y_score is regarded as the score of the class given by the column labels.
      # In that case, the method returns the macro average of the classes that have both positive
      # and negative examples, and a class that has neither of them is either skipped with
      # skip_undefined or reported with ArgumentError. If the average is 'none', the method
      # returns the ROC AUC of each class as a hash instead of the macro average.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples] or [n_samples, n_classes])
      #   Ground truth binary labels, ground truth class labels, or one-hot encoded multi-labels.
      # @param y_score [Numo::DFloat] (shape: [n_samples] or [n_samples, n_classes])
      #   Predicted class probabilities or confidence scores.
      # @param column_labels [Array] (shape: [n_classes]) Class labels corresponding to the columns
      #   of y_score. This argument is used when y_true is a one-dimensional label vector. It can
      #   also be given with the labels keyword argument. If nil is given, the sorted unique labels
      #   of y_true are used.
      # @param average [String] The average type ('macro' or 'none') of the ROC AUC for the classes.
      # @param skip_undefined [Boolean] Whether to ignore the classes that have no positive or
      #   negative examples in y_true when calculating the macro average. If false is given,
      #   the method raises ArgumentError when such a class exists.
      # @return [Float/Hash] (macro-averaged) ROC AUC, or ROC AUC for each class when the average
      #   is 'none'.
      def score(y_true, y_score, column_labels = nil, average: nil, labels: nil, skip_undefined: nil)
        labels ||= column_labels || @labels
        average ||= @average
        skip_undefined = @skip_undefined if skip_undefined.nil?
        y_true, y_score = cast_arrays(y_true, y_score)
        n_classes = y_score.shape[1]
        if n_classes.nil?
          fpr, tpr, = roc_curve(y_true, y_score)
          return auc(fpr, tpr)
        end

        return matrix_score(y_true, y_score) if y_true.shape[1]

        scores = score_each_class(y_true, y_score, labels)
        return scores if average.to_s == 'none'

        undefined_labels = scores.select { |_label, value| value.nil? }.keys
        unless undefined_labels.empty? || skip_undefined
          raise ArgumentError, "y_true has no positive or negative examples for #{undefined_labels.join(', ')}."
        end

        defined_scores = scores.values.compact
        raise ArgumentError, 'y_true must have positive and negative examples for at least one class.' if defined_scores.empty?

        defined_scores.sum.fdiv(defined_scores.size)
      end

      # Calculate ROC AUC for each class of the given ground truth label vector.
      #
      # The class of each column in y_score is determined by the given column labels, so this
      # method keeps working when the column order differs from the sorted or the appearance order
      # of the ground truth labels. A class that has no positive or negative examples in y_true
      # is reported with nil.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth class labels.
      # @param y_score [Numo::DFloat] (shape: [n_samples, n_classes])
      #   Predicted class probabilities or confidence scores.
      # @param column_labels [Array] (shape: [n_classes]) Class labels corresponding to the columns
      #   of y_score. It can also be given with the labels keyword argument. If nil is given, the
      #   sorted unique labels of y_true are used.
      # @param labels [Array] (shape: [n_classes]) Alias of column_labels.
      # @return [Hash] ROC AUC for each class. The value of a class that has no positive or
      #   negative examples in y_true is nil.
      def score_each_class(y_true, y_score, column_labels = nil, labels: nil)
        labels ||= column_labels || @labels
        y_true, y_score = cast_arrays(y_true, y_score)
        n_classes = y_score.shape[1]
        raise ArgumentError, 'y_score must be a two-dimensional array of predicted scores.' if n_classes.nil?
        raise ArgumentError, 'y_true must be a one-dimensional array of class labels.' unless y_true.shape[1].nil?

        labels = (labels || y_true.to_a.uniq.sort).to_a
        if labels.size != n_classes
          raise ArgumentError, "labels must have #{n_classes} elements corresponding to the columns " \
                               "of y_score, but #{labels.size} elements were given. " \
                               'The column labels are required when y_true does not contain every class of y_score.'
        end
        raise ArgumentError, 'labels must not contain duplicated class labels.' unless labels.uniq.size == labels.size

        unknown_labels = y_true.to_a.uniq - labels
        unless unknown_labels.empty?
          raise ArgumentError, "y_true has class labels #{unknown_labels.join(', ')} that are not included in labels."
        end

        labels.each_with_index.to_h { |label, col| [label, each_class_auc(y_true, y_score, label, col)] }
      end

      # Calculate receiver operation characteristic curve.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth binary labels.
      # @param y_score [Numo::DFloat] (shape: [n_samples]) Predicted class probabilities or confidence scores.
      # @param pos_label [Integer] Label to be a positive label when binarizing the given labels.
      #   If nil is given, the method considers the maximum value of the label as a positive label.
      # @return [Array] fpr (Numo::DFloat): false positive rates. tpr (Numo::DFloat): true positive rates.
      #   thresholds (Numo::DFloat): thresholds on the decision function used to calculate fpr and tpr.
      def roc_curve(y_true, y_score, pos_label = nil)
        labels = y_true.to_a.uniq
        if pos_label.nil?
          unless labels.size == 2
            raise ArgumentError,
                  'y_true must be binary labels or pos_label must be specified if y_true is multi-label'
          end
        else
          unless y_true.to_a.uniq.include?(pos_label)
            raise ArgumentError,
                  'y_true must have elements whose values are pos_label.'
          end
        end

        false_pos, true_pos, thresholds = binary_roc_curve(y_true, y_score, pos_label)

        if true_pos.empty? || false_pos[0] != 0 || true_pos[0] != 0
          # NOTE: Numo::NArray#insert is not a destructive method.
          # rubocop:disable Style/RedundantSelfAssignment
          true_pos = true_pos.insert(0, 0)
          false_pos = false_pos.insert(0, 0)
          thresholds = thresholds.insert(0, thresholds[0] + 1)
          # rubocop:enable Style/RedundantSelfAssignment
        end

        tpr = true_pos / true_pos[-1].to_f
        fpr = false_pos / false_pos[-1].to_f

        [fpr, tpr, thresholds]
      end

      # Calculate area under the curve using the trapezoidal rule.
      #
      # @param x [Numo::Int32/Numo::DFloat] (shape: [n_elements])
      #   x coordinates. These are expected to monotonously increase or decrease.
      # @param y [Numo::Int32/Numo::DFloat] (shape: [n_elements]) y coordinates.
      # @return [Float] area under the curve.
      def auc(x, y)
        n_samples = [x.shape[0], y.shape[0]].min
        raise ArgumentError, 'At least two points are required to calculate area under curve.' if n_samples < 2

        (0...n_samples).to_a.each_cons(2).sum { |i, j| 0.5 * (x[i] - x[j]).abs * (y[i] + y[j]) }
      end

      private

      def cast_arrays(y_true, y_score)
        [y_true.is_a?(Numo::NArray) ? y_true : Numo::NArray.asarray(y_true),
         y_score.is_a?(Numo::NArray) ? y_score : Numo::DFloat.asarray(y_score)]
      end

      def matrix_score(y_true, y_score)
        n_classes = y_score.shape[1]
        scores = Array.new(n_classes) do |c|
          fpr, tpr, = roc_curve(y_true[true, c], y_score[true, c])
          auc(fpr, tpr)
        end

        scores.sum.fdiv(n_classes)
      end

      def each_class_auc(y_true, y_score, label, col)
        bin_y_true = y_true.eq(label)
        n_positives = bin_y_true.count
        return nil if n_positives.zero? || n_positives == y_true.size

        fpr, tpr, = roc_curve(bin_y_true, y_score[true, col])
        auc(fpr, tpr)
      end

      def binary_roc_curve(y_true, y_score, pos_label = nil)
        pos_label = y_true.to_a.uniq.max if pos_label.nil?

        bin_y_true = y_true.eq(pos_label)
        desc_pred_ids = y_score.sort_index.reverse

        desc_y_true = Numo::Int32.cast(bin_y_true[desc_pred_ids])
        desc_y_score = y_score[desc_pred_ids]

        threshold_ids = Numo::Int32.cast(desc_y_score.diff.ne(0).where.to_a.append(desc_y_true.size - 1))

        true_pos = desc_y_true.cumsum[threshold_ids]
        false_pos = 1 + threshold_ids - true_pos

        [false_pos, true_pos, desc_y_score[threshold_ids]]
      end
    end
  end
end
