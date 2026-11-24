# frozen_string_literal: true

require 'rumale/base/splitter'

module Rumale
  module ModelSelection
    # TimeSeriesSplit is a class that generates the set of data indices for time series cross-validation.
    # It is assumed that the dataset given are already ordered by time information.
    #
    # The sample-wise split generates the validation indices only from the rows that actually exist
    # in the dataset. When timestamps (or time group labels) are given, the splitter makes the splits
    # in units of time groups so that the samples sharing a timestamp are never split across the
    # training, isolated, and validation parts of a fold.
    #
    # @example
    #   require 'rumale/model_selection/time_series_split'
    #
    #   cv = Rumale::ModelSelection::TimeSeriesSplit.new(n_splits: 5)
    #   x = Numo::DFloat.new(6, 2).rand
    #   cv.split(x, nil).each do |train_ids, test_ids|
    #     puts '---'
    #     pp train_ids
    #     pp test_ids
    #   end
    #
    #   # ---
    #   # [0]
    #   # [1]
    #   # ---
    #   # [0, 1]
    #   # [2]
    #   # ---
    #   # [0, 1, 2]
    #   # [3]
    #   # ---
    #   # [0, 1, 2, 3]
    #   # [4]
    #   # ---
    #   # [0, 1, 2, 3, 4]
    #   # [5]
    #
    # @example Split in units of time groups
    #   require 'rumale/model_selection/time_series_split'
    #
    #   cv = Rumale::ModelSelection::TimeSeriesSplit.new(n_splits: 3, max_train_size: 3, gap: 1)
    #   x = Numo::DFloat.new(11, 2).rand
    #   time = Numo::Int32[0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 6]
    #   cv.split(x, nil, time).each do |train_ids, test_ids|
    #     puts '---'
    #     pp train_ids
    #     pp test_ids
    #   end
    #
    #   # ---
    #   # [0]
    #   # [3, 4, 5, 6]
    #   # ---
    #   # [1, 2, 3]
    #   # [7, 8, 9]
    #   # ---
    #   # [7]
    #   # [10]
    #
    class TimeSeriesSplit
      include ::Rumale::Base::Splitter

      # Return the number of splits.
      # @return [Integer]
      attr_reader :n_splits

      # Return the maximum number of training samples in a split.
      # @return [Integer/Nil]
      attr_reader :max_train_size

      # Return the number of time groups isolated between the training and validation data.
      # It is counted in samples for the sample-wise split and in time groups for the time group split.
      # @return [Integer]
      attr_reader :gap

      # Create a new data splitter for time series cross-validation.
      #
      # @param n_splits [Integer] The number of splits.
      # @param max_train_size [Integer/Nil] The maximum number of training samples in a split.
      # @param gap [Integer] The number of time groups to isolate between the training and validation data.
      #   It is counted in samples for the sample-wise split and in time groups for the time group split.
      def initialize(n_splits: 5, max_train_size: nil, gap: 0)
        @n_splits = n_splits
        @max_train_size = max_train_size
        @gap = gap
      end

      # Generate data indices for time series cross-validation.
      #
      # When the timestamps are not given, the dataset is split sample-wise and the validation
      # blocks are taken from the tail of the dataset so that they consist only of the rows that
      # actually exist. When the timestamps are given, the dataset is split in units of time groups,
      # where the samples sharing the same timestamp form a single time group.
      #
      # @overload split(x, y) -> Array
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #     The dataset to be used to generate data indices for time series cross-validation.
      #     It is expected that the data will be ordered by time information.
      #   @param y [Numo::Int32] (shape: [n_samples])
      #     This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      #   @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      # @overload split(x, y, timestamps) -> Array
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #     The dataset to be used to generate data indices for time series cross-validation.
      #     It is expected that the data will be ordered by time information.
      #   @param y [Numo::Int32] (shape: [n_samples])
      #     This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      #   @param timestamps [Numo::Int32/Array] (shape: [n_samples])
      #     The timestamps (or the time group labels) of the samples.
      #     The samples sharing the same timestamp are treated as a single time group.
      #   @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, y, timestamps = nil)
        return split_by_row(x, y) if timestamps.nil?

        split_by_time_group(x, y, timestamps)
      end

      # Generate data indices for time series cross-validation sample-wise.
      #
      # The validation blocks are placed at the tail of the dataset so that they contain only the
      # rows that actually exist. The last sample appears in the validation part exactly once.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for time series cross-validation.
      # @param y [Numo::Int32] (shape: [n_samples])
      #   This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split_by_row(x, _y)
        n_samples = x.shape[0]
        unless (@n_splits + 1).between?(2, n_samples)
          raise ArgumentError,
                'The number of folds (n_splits + 1) must be not less than 2 and not more than the number of samples.'
        end
        raise ArgumentError, 'The gap must be a non-negative integer.' if @gap.negative?

        test_size = n_samples / (@n_splits + 1)
        first_test_start = n_samples - @n_splits * test_size

        Array.new(@n_splits) do |n|
          test_start = first_test_start + test_size * n
          test_ids = Array(test_start...(test_start + test_size))
          train_ids = sample_wise_train_ids(test_start - @gap)
          [train_ids, test_ids]
        end
      end

      # Generate data indices for time series cross-validation in units of time groups.
      #
      # The number of the time groups for validation is derived from the number of splits, and the
      # validation segments are taken from the left of the last consecutive time groups. The +gap+
      # time groups immediately before each validation segment are isolated from both the training
      # and validation data. The training window is built backwards from the most recent available
      # historical group, adding whole consecutive time groups as long as the number of training
      # rows does not exceed +max_train_size+.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for time series cross-validation.
      # @param y [Numo::Int32] (shape: [n_samples])
      #   This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      # @param timestamps [Numo::Int32/Array] (shape: [n_samples])
      #   The timestamps (or the time group labels) of the samples. The samples sharing the same
      #   timestamp are treated as a single time group, and a group is never split between the
      #   training, isolated, and validation parts of a fold.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split_by_time_group(x, _y, timestamps)
        raise ArgumentError, 'The gap must be a non-negative integer.' if @gap.negative?

        groups = extract_time_groups(timestamps, x.shape[0])
        n_groups = groups.size
        n_valid_groups = n_groups - @gap - 1
        if @n_splits < 1 || n_valid_groups < @n_splits
          raise ArgumentError,
                'The number of time groups must be greater than the number of splits plus the length of the gap.'
        end

        n_groups_per_split, n_rest = n_valid_groups.divmod(@n_splits)
        start_group = n_groups - n_valid_groups

        Array.new(@n_splits) do |n|
          n_split_groups = n_groups_per_split + (n < n_rest ? 1 : 0)
          test_ids = Array(start_group...(start_group + n_split_groups)).flat_map { |gid| groups[gid] }
          train_ids = time_group_wise_train_ids(groups, start_group - @gap - 1)
          start_group += n_split_groups
          [train_ids, test_ids]
        end
      end

      private

      # Take the rows that actually exist right before the given end, keeping at most max_train_size of them.
      def sample_wise_train_ids(train_end)
        train_ids = Array(0...train_end)
        train_ids = train_ids.last(@max_train_size) unless @max_train_size.nil? || train_ids.size <= @max_train_size
        if train_ids.empty?
          raise ArgumentError, 'The training samples are insufficient to generate the splits.'
        end

        train_ids
      end

      # Split the ordered timestamps into consecutive time groups.
      def extract_time_groups(timestamps, n_samples)
        labels = timestamps.respond_to?(:to_a) ? timestamps.to_a : Array(timestamps)
        if labels.size != n_samples
          raise ArgumentError, 'The number of timestamps must be equal to the number of samples.'
        end

        group_ids = {}
        groups = []
        labels.each_with_index do |label, row_id|
          gid = group_ids[label]
          if gid.nil?
            gid = groups.size
            group_ids[label] = gid
            groups.push([])
          end
          groups[gid].push(row_id)
        end
        groups
      end

      # Take consecutive whole time groups backwards from the most recent available group,
      # stopping before the first group that would exceed max_train_size rows.
      def time_group_wise_train_ids(groups, last_group)
        train_ids = []
        gid = last_group
        while gid >= 0
          rows = groups[gid]
          break unless @max_train_size.nil? || train_ids.size + rows.size <= @max_train_size

          train_ids = rows + train_ids
          gid -= 1
        end
        if train_ids.empty?
          raise ArgumentError, 'The training samples are insufficient to generate the splits.'
        end

        train_ids
      end
    end
  end
end
