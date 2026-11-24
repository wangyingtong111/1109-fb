# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Preprocessing
    # KernelCalculator is a class that calculates the kernel matrix with training data.
    #
    # @example
    #   require 'rumale/preprocessing/kernel_calculator'
    #   require 'rumale/kernel_machine/kernel_ridge'
    #   require 'rumale/pipeline/pipeline'
    #
    #   transformer = Rumale::Preprocessing::KernelCalculator.new(kernel: 'rbf', gamma: 0.5)
    #   regressor = Rumale::KernelMachine::KernelRidge.new
    #   pipeline = Rumale::Pipeline::Pipeline.new(
    #     steps: { trs: transfomer, est: regressor }
    #   )
    #   pipeline.fit(x_train, y_train)
    #   results = pipeline.predict(x_test)
    class KernelCalculator < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the training data for calculating kernel matrix.
      # @return [Numo::DFloat] (shape: n_components, n_features)
      attr_reader :components

      # Returns the column means of the kernel matrix on the training data.
      # These values are used for centering the kernel matrix with the training data as a reference.
      # @return [Numo::DFloat] (shape: n_components)
      attr_reader :col_means

      # Returns the mean of all elements of the kernel matrix on the training data.
      # @return [Float]
      attr_reader :total_mean

      # Create a new transformer that transforms feature vectors into a kernel matrix.
      #
      # @param kernel [String] The type of kernel function ('rbf', 'linear', 'poly', and 'sigmoid').
      # @param gamma [Float] The gamma parameter in rbf/poly/sigmoid kernel function.
      # @param degree [Integer] The degree parameter in polynomial kernel function.
      # @param coef [Float] The coefficient in poly/sigmoid kernel function.
      # @param center [Boolean] The flag indicating whether to center the kernel matrix
      #   with the kernel matrix on the training data as a reference.
      def initialize(kernel: 'rbf', gamma: 1, degree: 3, coef: 1, center: false)
        super()
        @params = {
          kernel: kernel,
          gamma: gamma,
          degree: degree,
          coef: coef,
          center: center
        }
        @col_means = nil
        @total_mean = nil
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> KernelCalculator
      #   @param x [Numo::NArray] (shape: [n_samples, n_features]) The training data to be used for calculating kernel matrix.
      # @return [KernelCalculator] The learned transformer itself.
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        @components = x.dup
        calc_kernel_mat_means if @params[:center]
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for calculating kernel matrix.
      # @return [Numo::DFloat] (shape: [n_samples, n_samples]) The calculated kernel matrix.
      def fit_transform(x, y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be used for calculating kernel matrix with the training data.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The calculated kernel matrix.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        kernel_mat = kernel_mat(x, @components)
        return kernel_mat unless @params[:center]

        row_means = kernel_mat.mean(1).expand_dims(1)
        kernel_mat - @col_means - row_means + @total_mean
      end

      private

      def calc_kernel_mat_means
        kernel_mat = kernel_mat(@components, @components)
        @col_means = kernel_mat.mean(0)
        @total_mean = @col_means.mean
      end

      def kernel_mat(x, y)
        case @params[:kernel]
        when 'rbf'
          ::Rumale::PairwiseMetric.rbf_kernel(x, y, @params[:gamma])
        when 'poly'
          ::Rumale::PairwiseMetric.polynomial_kernel(x, y, @params[:degree], @params[:gamma], @params[:coef])
        when 'sigmoid'
          ::Rumale::PairwiseMetric.sigmoid_kernel(x, y, @params[:gamma], @params[:coef])
        when 'linear'
          ::Rumale::PairwiseMetric.linear_kernel(x, y)
        else
          raise ArgumentError, "Expect kernel parameter to be given 'rbf', 'linear', 'poly', or 'sigmoid'."
        end
      end
    end
  end
end
