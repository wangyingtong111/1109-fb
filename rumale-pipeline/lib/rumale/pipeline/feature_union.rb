# frozen_string_literal: true

require 'rumale/base/estimator'

module Rumale
  module Pipeline
    # FeatureUnion is a class that implements the function concatenating the multi-transformer results.
    #
    # @example
    #   require 'rumale/kernel_approximation/rbf'
    #   require 'rumale/decomposition/pca'
    #   require 'rumale/pipeline/feature_union'
    #
    #   fu = Rumale::Pipeline::FeatureUnion.new(
    #     transformers: {
    #       'rbf': Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: 96, random_seed: 1),
    #       'pca': Rumale::Decomposition::PCA.new(n_components: 32)
    #     }
    #   )
    #   fu.fit(training_samples, traininig_labels)
    #   results = fu.predict(testing_samples)
    #
    #   # > p results.shape[1]
    #   # > 128
    #
    class FeatureUnion < ::Rumale::Base::Estimator
      # Return the transformers.
      # @return [Hash]
      attr_reader :transformers

      # Return the half-open column interval of each branch within the concatenated feature matrix.
      # The layout is determined by the first successful concatenation after fitting,
      # and is rebuilt whenever the feature union is fitted again.
      # @return [Hash/Nil] The mapping from transformer name to the half-open column interval [start, stop).
      attr_reader :column_ranges

      # Create a new feature union.
      #
      # @param transformers [Hash] List of transformers.  The order of transforms follows the insertion order of hash keys.
      def initialize(transformers:)
        super()
        @params = {}
        @transformers = transformers
        @column_ranges = nil
      end

      # Fit the model with given training data.
      #
      # The column layout memorized by previous transformations is reset, and a new layout
      # is established by the first successful concatenation performed after this fit.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the transformers.
      # @param y [Numo::NArray/Nil] (shape: [n_samples, n_outputs]) The target values or labels to be used for fitting the transformers.
      # @return [FeatureUnion] The learned feature union itself.
      def fit(x, y = nil)
        @transformers.each_value { |t| t.fit(x, y) }
        @column_ranges = nil
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # The first successful concatenation establishes the column width and the half-open
      # column interval of every branch, and the same layout is reused by subsequent transformations.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the transformers.
      # @param y [Numo::NArray/Nil] (shape: [n_samples, n_outputs]) The target values or labels to be used for fitting the transformers.
      # @return [Numo::DFloat] (shape: [n_samples, sum_n_components]) The transformed and concatenated data.
      def fit_transform(x, y = nil)
        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # A transformer is allowed to return a one-dimensional vector, in which case the vector
      # is treated as a single column.  A transformer that returns a matrix with zero columns
      # contributes nothing to the concatenated result.
      #
      # On the first successful call after fitting, the column width and half-open column
      # interval of every branch are memorized.  On subsequent calls, each branch must keep
      # the same number of columns and all branches must return the same number of rows as
      # the input data; otherwise an ArgumentError that identifies the offending branch is raised.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned transformers.
      # @return [Numo::DFloat] (shape: [n_samples, sum_n_components]) The transformed and concatenated data.
      def transform(x)
        outputs = @transformers.map { |name, transformer| [name, normalize_output(name, transformer.transform(x))] }
        check_row_contracts(x, outputs)
        establish_layout(outputs) if @column_ranges.nil?
        check_column_contracts(outputs)
        concatenate(x, outputs)
      end

      # Return the half-open column interval [start, stop) of the given branch within the
      # concatenated feature matrix.
      #
      # @param name [Object] The transformer name given as a key of the transformers hash.
      # @return [Array/Nil] The half-open column interval [start, stop), or nil when the feature union has not yet been transformed.
      # @raise [KeyError] If the given name is not one of the transformer names.
      def column_range(name)
        @column_ranges.fetch(name) unless @column_ranges.nil?
      end

      private

      def normalize_output(name, output)
        raise ArgumentError, "transformer '#{name}' is expected to respond to transform with a Numo::NArray" unless output.is_a?(Numo::NArray)
        raise ArgumentError, "transformer '#{name}' is expected to return a 1-D or 2-D array" unless output.ndim <= 2

        output.ndim == 1 ? output.reshape(output.size, 1) : output
      end

      def check_row_contracts(x, outputs)
        n_samples = x.shape[0]
        outputs.each do |name, output|
          next if output.shape[0] == n_samples

          raise ArgumentError,
                "transformer '#{name}' returned #{output.shape[0]} rows but the input data has #{n_samples} rows"
        end
      end

      def establish_layout(outputs)
        @column_ranges = {}
        start = 0
        outputs.each do |name, output|
          width = output.shape[1]
          @column_ranges[name] = [start, start + width]
          start += width
        end
      end

      def check_column_contracts(outputs)
        outputs.each do |name, output|
          expected_start, expected_stop = @column_ranges.fetch(name)
          expected_width = expected_stop - expected_start
          actual_width = output.shape[1]
          next if actual_width == expected_width

          raise ArgumentError,
                "transformer '#{name}' changed its number of columns from #{expected_width} to #{actual_width}, " \
                'which violates the feature layout established after fitting'
        end
      end

      def concatenate(x, outputs)
        matrices = outputs.map(&:last)
        return Numo::DFloat.zeros(x.shape[0], 0) if matrices.empty?

        Numo::NArray.hstack(matrices)
      end
    end
  end
end
