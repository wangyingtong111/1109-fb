# frozen_string_literal: true

require 'rumale/base/splitter'
require 'rumale/preprocessing/label_encoder'

module Rumale
  module ModelSelection
    # StratifiedGroupKFold is a class that generates the set of data indices for K-fold cross-validation.
    # The folds are made by preserving the proportion of the number of samples for each class
    # while keeping the data points belonging to the same group in the same fold.
    # The class distribution of each fold is evaluated with the true label counts of each group.
    # The group-to-fold assignment is deterministic: it depends only on the class-count vector
    # and the label of each group, so reordering groups or samples within a group never changes
    # the assignment (only the resulting sample indices move).
    # If a class appears in fewer groups than the number of splits, the class cannot be covered
    # by all folds because groups are never split. In that case, a warning is issued.
    #
    # @example
    #   require 'rumale/model_selection/stratified_group_k_fold'
    #
    #   cv = Rumale::ModelSelection::StratifiedGroupKFold.new(n_splits: 2)
    #   x = Numo::DFloat.new(16, 2).rand
    #   y = Numo::Int32[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    #   groups = Numo::Int32[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    #   cv.split(x, y, groups).each do |train_ids, test_ids|
    #     puts '---'
    #     pp train_ids
    #     pp test_ids
    #   end
    #
    #   # ---
    #   # [4, 5, 6, 7, 8, 9, 10, 11]
    #   # [0, 1, 2, 3, 12, 13, 14, 15]
    #   # ---
    #   # [0, 1, 2, 3, 12, 13, 14, 15]
    #   # [4, 5, 6, 7, 8, 9, 10, 11]
    #
    class StratifiedGroupKFold
      include ::Rumale::Base::Splitter

      # Return the number of folds.
      # @return [Integer]
      attr_reader :n_splits

      # Create a new data splitter for stratified grouped K-fold cross validation.
      #
      # @param n_splits [Integer] The number of folds.
      def initialize(n_splits: 5)
        @n_splits = n_splits
      end

      # Generate data indices for stratified grouped K-fold cross validation.
      #
      # @overload split(x, y, groups) -> Array
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #     The dataset to be used to generate data indices for stratified grouped K-fold cross validation.
      #     This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      #   @param y [Numo::Int32] (shape: [n_samples])
      #     The labels to be used to generate data indices for stratified grouped K-fold cross validation.
      #   @param groups [Numo::Int32] (shape: [n_samples])
      #     The group labels to be used to generate data indices for stratified grouped K-fold cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(_x, y, groups)
        group_encoder = ::Rumale::Preprocessing::LabelEncoder.new
        group_ids = group_encoder.fit_transform(groups).to_a
        n_groups = group_encoder.classes.size

        unless @n_splits.between?(2, n_groups)
          raise ArgumentError, 'The value of n_splits must be not less than 2 and not more than the number of groups.'
        end

        label_encoder = ::Rumale::Preprocessing::LabelEncoder.new
        labels = label_encoder.fit_transform(y).to_a
        n_classes = label_encoder.classes.size

        # Count the number of samples of each class in each group with the true labels.
        class_counts = Array.new(n_groups) { Array.new(n_classes, 0) }
        labels.zip(group_ids).each { |label, gid| class_counts[gid][label] += 1 }

        warn_uncoverable_classes(class_counts, label_encoder.classes)

        group_to_fold = assign_groups_to_folds(class_counts)

        n_samples = labels.size
        sample_ids = Array(0...n_samples)
        fold_ids = group_ids.map { |gid| group_to_fold[gid] }
        Array.new(@n_splits) do |fid|
          test_ids = sample_ids.select { |sid| fold_ids[sid] == fid }
          [sample_ids - test_ids, test_ids]
        end
      end

      private

      # Issue a warning for each class that appears in fewer groups than the number of splits.
      # Such a class cannot be covered by all folds, since groups are never split across folds.
      def warn_uncoverable_classes(class_counts, classes)
        classes.size.times do |cid|
          n_covering_groups = class_counts.count { |counts| counts[cid].positive? }
          next if n_covering_groups >= @n_splits

          warn "Class #{classes[cid].inspect} appears in only #{n_covering_groups} group(s), " \
               "which is less than n_splits=#{@n_splits}. " \
               'The class cannot be covered by all folds because groups are never split.'
        end
      end

      # Assign each group to a fold so that the class distribution of each fold becomes
      # as close to the whole dataset as possible. The assignment is deterministic and
      # independent of the order of samples and groups: groups are processed in a canonical
      # order defined by their class-count vectors and encoded labels, and ties between
      # folds are broken by the fold index.
      def assign_groups_to_folds(class_counts)
        n_classes = class_counts.first.size
        targets = Array.new(n_classes) { |cid| class_counts.sum { |counts| counts[cid] }.fdiv(@n_splits) }
        fold_counts = Array.new(@n_splits) { Array.new(n_classes, 0) }
        group_to_fold = Array.new(class_counts.size)

        canonical_group_order(class_counts).each do |gid|
          fid = (0...@n_splits).min_by { |f| [distribution_cost(fold_counts[f], class_counts[gid], targets), f] }
          n_classes.times { |cid| fold_counts[fid][cid] += class_counts[gid][cid] }
          group_to_fold[gid] = fid
        end

        group_to_fold
      end

      def canonical_group_order(class_counts)
        (0...class_counts.size).sort_by { |gid| [class_counts[gid].map(&:-@), gid] }
      end

      def distribution_cost(fold_counts, group_counts, targets)
        fold_counts.zip(group_counts, targets).sum { |fc, gc, t| (fc + gc - t).abs }
      end
    end
  end
end
