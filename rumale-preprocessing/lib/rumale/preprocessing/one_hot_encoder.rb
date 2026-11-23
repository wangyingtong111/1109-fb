# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Encode categorical integer features to one-hot-vectors.
    #
    # @example
    #   require 'rumale/preprocessing/one_hot_encoder'
    #
    #   encoder = Rumale::Preprocessing::OneHotEncoder.new
    #   labels = Numo::Int32[0, 0, 2, 3, 2, 1]
    #   one_hot_vectors = encoder.fit_transform(labels)
    #   # > pp one_hot_vectors
    #   # Numo::DFloat#shape[6, 4]
    #   # [[1, 0, 0, 0],
    #   #  [1, 0, 0, 0],
    #   #  [0, 0, 1, 0],
    #   #  [0, 0, 0, 1],
    #   #  [0, 0, 1, 0],
    #   #  [0, 1, 0, 0]]
    #
    # @example In seen-categories dictionary mode, only the values that actually occur in the training data get a slot.
    #   encoder = Rumale::Preprocessing::OneHotEncoder.new(handle_unknown: 'ignore')
    #   training = Numo::Int32[[0, 4], [1000000, 1], [0, 4]]
    #   testing = Numo::Int32[[5, 4], [1000000, 9], [0, 1]]
    #   pp encoder.fit_transform(training)
    #   # Numo::DFloat#shape[3,4]
    #   # [[1, 0, 0, 1],
    #   #  [0, 1, 1, 0],
    #   #  [1, 0, 0, 1]]
    #   pp encoder.transform(testing)
    #   # Numo::DFloat#shape[3,4]
    #   # [[0, 0, 0, 1],
    #   #  [0, 1, 0, 0],
    #   #  [1, 0, 1, 0]]
    class OneHotEncoder < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the maximum values for each feature.
      # @return [Numo::Int32] (shape: [n_features])
      attr_reader :n_values

      # Return the indices for feature values that actually occur in the training set.
      # @return [Nimo::Int32]
      attr_reader :active_features

      # Return the indices to feature ranges.
      # @return [Numo::Int32] (shape: [n_features + 1])
      attr_reader :feature_indices

      # Return the seen values for each feature in seen-categories dictionary mode.
      # Each feature has a sorted array that only contains values actually occurring in the training data.
      # @return [Array] (size: [n_features]) Return nil unless the encoder works in seen-categories dictionary mode.
      attr_reader :categories

      # Create a new encoder for encoding categorical integer features to one-hot-vectors
      #
      # @param handle_unknown [String/Nil] The strategy for values that do not occur in the training data.
      #   If nil is given, the legacy behavior that allocates slots up to the maximum value is used.
      #   If 'error' is given, the encoder works in seen-categories dictionary mode
      #   and raises ArgumentError when an unseen value is found on transformation.
      #   If 'ignore' is given, the encoder works in seen-categories dictionary mode
      #   and only clears the feature segment corresponding to an unseen value.
      def initialize(handle_unknown: nil)
        super()
        @params = { handle_unknown: handle_unknown }
        unless handle_unknown.nil? || %w[error ignore].include?(handle_unknown)
          raise ArgumentError, "handle_unknown must be nil, 'error' or 'ignore'."
        end

        @categories = nil
      end

      # Fit one-hot-encoder to samples.
      #
      # @overload fit(x) -> OneHotEncoder
      #   @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to fit one-hot-encoder.
      # @return [OneHotEncoder]
      def fit(x, _y = nil)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?

        if dictionary_mode?
          fit_dictionary(x)
        else
          @n_values = x.max(0) + 1
          @feature_indices = Numo::Int32.hstack([[0], @n_values]).cumsum
          @active_features = encode_legacy(x, @feature_indices).sum(axis: 0).ne(0).where
        end
        self
      end

      # Fit one-hot-encoder to samples, then encode samples into one-hot-vectors
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to encode into one-hot-vectors.
      # @return [Numo::DFloat] The one-hot-vectors.
      def fit_transform(x, _y = nil)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?

        fit(x).transform(x)
      end

      # Encode samples into one-hot-vectors.
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to encode into one-hot-vectors.
      # @return [Numo::DFloat] The one-hot-vectors.
      def transform(x)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?

        if dictionary_mode?
          transform_dictionary(x)
        else
          codes = encode_legacy(x, @feature_indices)
          codes[true, @active_features].dup
        end
      end

      private

      def dictionary_mode?
        !@params[:handle_unknown].nil?
      end

      def fit_dictionary(x)
        x = to_two_dimensional(x)
        n_samples, n_features = x.shape
        @categories = Array.new(n_features) { |n| x[true, n].to_a.uniq.sort }
        @n_values = Numo::Int32.asarray(@categories.map(&:size))
        @feature_indices = Numo::Int32.hstack([[0], @n_values]).cumsum
        codes = Numo::DFloat.zeros(n_samples, @feature_indices[-1])
        n_features.times do |n|
          value_to_slot = {}
          @categories[n].each_with_index { |v, slot| value_to_slot[v] = slot }
          offset = @feature_indices[n]
          x[true, n].each_with_index { |v, row| codes[row, offset + value_to_slot[v]] = 1.0 }
        end
        @active_features = codes.sum(axis: 0).ne(0).where
      end

      def transform_dictionary(x)
        x = to_two_dimensional(x)
        n_samples, n_features = x.shape
        codes = Numo::DFloat.zeros(n_samples, @feature_indices[-1])
        n_features.times do |n|
          value_to_slot = {}
          @categories[n].each_with_index { |v, slot| value_to_slot[v] = slot }
          offset = @feature_indices[n]
          x[true, n].each_with_index do |v, row|
            slot = value_to_slot[v]
            if slot.nil?
              next if @params[:handle_unknown] == 'ignore'

              raise ArgumentError, "Found unseen value #{v.inspect} at row #{row} and column #{n}."
            end
            codes[row, offset + slot] = 1.0
          end
        end
        codes
      end

      def to_two_dimensional(x)
        x.shape[1].nil? ? x.reshape(x.size, 1) : x
      end

      def encode_legacy(x, indices)
        n_samples, n_features = x.shape
        n_features = 1 if n_features.nil?
        col_indices = (x + indices[0...-1]).flatten.to_a
        row_indices = Numo::Int32.new(n_samples).seq.repeat(n_features).to_a
        codes = Numo::DFloat.zeros(n_samples, indices[-1])
        row_indices.zip(col_indices).each { |r, c| codes[r, c] = 1.0 }
        codes
      end
    end
  end
end
