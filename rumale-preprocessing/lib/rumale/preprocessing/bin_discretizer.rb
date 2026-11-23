# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module Preprocessing
    # Discretizes features with a given number of bins.
    # In some cases, discretizing features may accelerate decision tree training.
    #
    # The bin edges are determined only by the training samples. When transforming,
    # values below the lowest edge are assigned to the first bin, values above the
    # highest edge are assigned to the last bin, and missing values (NaN) are
    # assigned to -1 so that they can be distinguished from the bins.
    #
    # Two strategies are available for determining the bin edges.
    # The 'uniform' strategy divides the range between the minimum and the maximum
    # value of each feature into bins of equal width.
    # The 'quantile' strategy divides the observed values of each feature into bins
    # holding the same number of samples, where the quantiles are calculated with
    # linear interpolation and duplicated edges are merged.
    # The 'quantile' strategy is useful for features whose values are highly
    # duplicated because it avoids producing many empty bins.
    #
    # @example
    #   require 'rumale/preprocessing/bin_discretizer'
    #
    #   discretizer = Rumale::Preprocessing::BinDiscretizer.new(n_bins: 4)
    #   samples = Numo::DFloat.new(5, 2).rand - 0.5
    #   transformed = discretizer.fit_transform(samples)
    #   # > pp samples
    #   # Numo::DFloat#shape=[5,2]
    #   # [[-0.438246, -0.126933],
    #   #  [ 0.294815, -0.298958],
    #   #  [-0.383959, -0.155968],
    #   #  [ 0.039948,  0.237815],
    #   #  [-0.334911, -0.449117]]
    #   # > pp transformed
    #   # Numo::DFloat#shape=[5,2]
    #   # [[0, 1],
    #   #  [3, 0],
    #   #  [0, 1],
    #   #  [2, 3],
    #   #  [0, 0]]
    #
    # @example Quantile strategy
    #   require 'rumale/preprocessing/bin_discretizer'
    #
    #   discretizer = Rumale::Preprocessing::BinDiscretizer.new(n_bins: 4, strategy: 'quantile')
    #   samples = Numo::DFloat[[0], [0], [0], [0], [10], [20]]
    #   discretizer.fit(samples)
    #   # > pp discretizer.feature_steps[0]
    #   # Numo::DFloat#shape=[2]
    #   # [0, 7.5]
    #   discretizer.transform(Numo::DFloat[[-1], [0], [7.5], [20], [99], [Float::NAN]])
    #   # > pp transformed
    #   # Numo::DFloat#shape=[6,1]
    #   # [[0],
    #   #  [0],
    #   #  [1],
    #   #  [1],
    #   #  [1],
    #   #  [-1]]
    class BinDiscretizer < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the feature steps to be used discretizing.
      # Each element is the vector consisting of the lower edge of each bin for a feature.
      # The size of the vector is the number of bins of the feature, which can be
      # smaller than the value of the n_bins parameter when the edges are duplicated.
      # @return [Array<Numo::DFloat>] (shape: [n_features, n_bins])
      attr_reader :feature_steps

      # Create a new discretizer for features with given number of bins.
      #
      # @param n_bins [Integer] The number of bins to be used disretizing feature values.
      # @param strategy [String] The strategy to determine bin edges ('uniform' or 'quantile').
      def initialize(n_bins: 32, strategy: 'uniform')
        super()

        raise ArgumentError, 'Expect the value of n_bins parameter greater than or equal to 1.' if n_bins < 1
        unless %w[uniform quantile].include?(strategy)
          raise ArgumentError, "Expect the strategy parameter to be given 'uniform' or 'quantile'."
        end

        @params = { n_bins: n_bins, strategy: strategy }
      end

      # Fit feature ranges to be discretized.
      #
      # @overload fit(x) -> BinDiscretizer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the feature ranges.
      # @return [BinDiscretizer]
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_features = x.shape[1]
        @feature_steps = Array.new(n_features) do |n|
          values = observed_values(x[true, n])
          @params[:strategy] == 'quantile' ? quantile_steps(values) : uniform_steps(values)
        end
        self
      end

      # Fit feature ranges to be discretized, then return discretized samples.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be discretized.
      # @return [Numo::DFloat] The discretized samples.
      def fit_transform(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x).transform(x)
      end

      # Peform discretizing the given samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be discretized.
      # @return [Numo::DFloat] The discretized samples. Missing values are assigned to -1.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples, n_features = x.shape
        transformed = Numo::DFloat.zeros(n_samples, n_features)
        n_features.times do |n|
          steps = @feature_steps[n]
          col = x[true, n]
          steps.size.times do |bin|
            mask = col.ge(steps[bin]).where
            transformed[mask, n] = bin
          end
          transformed[col.isnan.where, n] = -1
        end
        transformed
      end

      private

      # Extract the observed values of the given feature, ignoring missing values.
      #
      # @param col [Numo::DFloat] (shape: [n_samples]) The feature values.
      # @return [Numo::DFloat] (shape: [n_observed]) The sorted observed values.
      def observed_values(col)
        values = col[col.isnan.eq(0)]
        values.empty? ? values : values.sort
      end

      # Determine the lower edges of the equally spaced bins.
      #
      # @param values [Numo::DFloat] (shape: [n_observed]) The sorted observed values.
      # @return [Numo::DFloat] (shape: [n_bins]) The lower edges of the bins.
      def uniform_steps(values)
        min_val, max_val = values.empty? ? [0.0, 0.0] : [values[0], values[-1]]
        Numo::DFloat.linspace(min_val, max_val, @params[:n_bins] + 1)[0...@params[:n_bins]]
      end

      # Determine the lower edges of the bins holding the same number of samples.
      # Duplicated edges are merged because they generate empty bins.
      #
      # @param values [Numo::DFloat] (shape: [n_observed]) The sorted observed values.
      # @return [Numo::DFloat] (shape: [n_bins]) The lower edges of the bins.
      def quantile_steps(values)
        return Numo::DFloat.zeros(0) if values.empty?

        steps = Array.new(@params[:n_bins]) { |bin| quantile(values, bin.fdiv(@params[:n_bins])) }
        Numo::DFloat[*steps.uniq]
      end

      # Calculate the quantile of the given sorted values with linear interpolation.
      #
      # @param values [Numo::DFloat] (shape: [n_observed]) The sorted observed values.
      # @param q [Float] The quantile in the range [0, 1].
      # @return [Float] The quantile value.
      def quantile(values, q)
        pos = q * (values.size - 1)
        lower = pos.floor
        upper = pos.ceil
        return values[lower] if lower == upper

        values[lower] + (pos - lower) * (values[upper] - values[lower])
      end
    end
  end
end
