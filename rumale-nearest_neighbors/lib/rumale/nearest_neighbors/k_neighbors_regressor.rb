# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module NearestNeighbors
    # KNeighborsRegressor is a class that implements the regressor with the k-nearest neighbors rule.
    # The current implementation uses the Euclidean distance for finding the neighbors.
    #
    # @example
    #   require 'rumale/nearest_neighbors/k_neighbors_regressor'
    #
    #   estimator =
    #     Rumale::NearestNeighbors::KNeighborsRegressor.new(n_neighbors: 5, weights: 'distance')
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    #
    class KNeighborsRegressor < ::Rumale::Base::Estimator
      include ::Rumale::Base::Regressor

      # Return the prototypes for the nearest neighbor regressor.
      # If the metric is 'precomputed', that returns nil.
      # If the algorithm is 'vptree', that returns Rumale::NearestNeighbors::VPTree.
      # @return [Numo::DFloat] (shape: [n_training_samples, n_features])
      attr_reader :prototypes

      # Return the values of the prototypes
      # @return [Numo::DFloat] (shape: [n_training_samples, n_outputs])
      attr_reader :values

      # Create a new regressor with the nearest neighbor rule.
      #
      # @param n_neighbors [Integer] The number of neighbors.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and predict methods expect to be given a distance matrix.
      # @param weights [String] The weight function used in prediction.
      #   If weights is 'uniform', all neighbors are weighted equally and the mean of their target values is predicted.
      #   If weights is 'distance', the neighbors are weighted by the inverse of their distances.
      #   When some of the selected neighbors have zero distance, only the zero-distance neighbors are averaged.
      def initialize(n_neighbors: 5, metric: 'euclidean', weights: 'uniform')
        raise ArgumentError, "Expect weights to be 'uniform' or 'distance'" unless %w[uniform distance].include?(weights)

        super()
        @params = {
          n_neighbors: n_neighbors,
          metric: (metric == 'precomputed' ? 'precomputed' : 'euclidean'),
          weights: weights
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_training_samples, n_features]) The training data to be used for fitting the model.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_training_samples, n_training_samples]).
      # @param y [Numo::DFloat] (shape: [n_training_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [KNeighborsRegressor] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
          raise ArgumentError, 'Expect the input distance matrix to be square.'
        end

        @prototypes = x.dup if @params[:metric] == 'euclidean'
        @values = y.dup
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_features]) The samples to predict the values.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_testing_samples, n_training_samples]).
      # @return [Numo::DFloat] (shape: [n_testing_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        if @params[:metric] == 'precomputed' && x.shape[1] != @values.shape[0]
          raise ArgumentError, 'Expect the size input matrix to be n_testing_samples-by-n_training_samples.'
        end

        # Initialize some variables.
        n_samples = x.shape[0]
        n_prototypes, n_outputs = @values.shape
        n_neighbors = [@params[:n_neighbors], n_prototypes].min
        # Predict values for the given samples.
        distance_matrix = @params[:metric] == 'precomputed' ? x : ::Rumale::PairwiseMetric.euclidean_distance(x, @prototypes)
        predicted_values = Array.new(n_samples) do |n|
          sorted_neighbors = distance_matrix[n, true].to_a.each_with_index.sort
          neighbor_ids = sorted_neighbors.map(&:last)[0...n_neighbors]
          neighbor_distances = sorted_neighbors.map(&:first)[0...n_neighbors]
          neighbor_values = n_outputs.nil? ? @values[neighbor_ids] : @values[neighbor_ids, true]
          if @params[:weights] == 'distance'
            weighted_neighbor_values(neighbor_values, neighbor_distances, n_outputs)
          elsif n_outputs.nil?
            neighbor_values.mean
          else
            neighbor_values.mean(0).to_a
          end
        end
        Numo::DFloat[*predicted_values]
      end

      private

      # Compute the inverse-distance-weighted average of the neighbor target values.
      # The weights are normalized per query by the minimum distance to avoid division by zero or overflow,
      # while keeping the true ratio of the inverse distances among positive distances.
      # When zero-distance neighbors exist, only their target values are averaged.
      def weighted_neighbor_values(neighbor_values, neighbor_distances, n_outputs)
        zero_ids = neighbor_distances.each_index.select { |i| neighbor_distances[i].zero? }
        unless zero_ids.empty?
          zero_values = n_outputs.nil? ? neighbor_values[zero_ids] : neighbor_values[zero_ids, true]
          return n_outputs.nil? ? zero_values.mean : zero_values.mean(0).to_a
        end

        min_distance = neighbor_distances.min
        weights = Numo::DFloat.asarray(neighbor_distances.map { |d| min_distance / d })
        if n_outputs.nil?
          (neighbor_values * weights).sum / weights.sum
        else
          (neighbor_values.transpose.dot(weights) / weights.sum).to_a
        end
      end
    end
  end
end
