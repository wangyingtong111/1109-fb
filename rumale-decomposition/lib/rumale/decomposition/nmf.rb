# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/utils'
require 'rumale/validation'

module Rumale
  module Decomposition
    # @!visibility private
    #
    # Validation helpers for the partially observed NMF fit.
    module NMFValidator
      module_function

      # Convert the mask into a binary DFloat matrix, or return nil when no mask is given.
      #
      # @param mask [Numo::NArray/Nil]
      # @param shape [Array<Integer>] The expected shape of the mask.
      # @param require_observation [Boolean] Whether at least one observed entry is mandatory.
      # @return [Numo::DFloat/Nil]
      def check_mask(mask, shape, require_observation: true)
        return nil if mask.nil?

        mask = Numo::DFloat.cast(mask) unless mask.is_a?(Numo::DFloat)
        raise ArgumentError, 'the mask is expected to be a 2-D array' unless mask.ndim == 2
        raise ArgumentError, 'the mask is expected to have the same shape as the sample array' unless mask.shape == shape
        raise ArgumentError, 'the mask is expected to contain only binary values (0 or 1)' unless binary_mask?(mask)
        raise ArgumentError, 'at least one entry must be observed when fitting NMF' if require_observation && mask.sum.zero?

        mask
      end

      # Convert the frozen row indices into a 1-D Int32 array, or return an empty array for nil.
      #
      # @param frozen_rows [Array/Numo::NArray/Nil]
      # @param n_components [Integer] The number of basis rows, used for the range check.
      # @return [Numo::Int32]
      def check_frozen_rows(frozen_rows, n_components)
        return Numo::Int32[] if frozen_rows.nil?

        rows = Numo::Int32.cast(frozen_rows)
        raise ArgumentError, 'the frozen row indices are expected to be a 1-D array' unless rows.ndim == 1
        if rows.lt(0).count_true.positive? || rows.ge(n_components).count_true.positive?
          raise ArgumentError, 'the frozen row indices are out of the components row range'
        end

        rows
      end

      # Raise an error when the given array contains a negative value.
      #
      # @param array [Numo::DFloat]
      # @param message [String]
      def check_non_negative(array, message)
        raise ArgumentError, message if array.lt(0).count_true.positive?
      end

      def binary_mask?(mask)
        mask.ge(0).count_true == mask.size && (mask.eq(0).count_true + mask.eq(1).count_true) == mask.size
      end
    end

    # @!visibility private
    #
    # Multiplicative-update solver for the masked non-negative factorization.
    # It operates on a host object that exposes +components+, +params+, and the helper methods below.
    module NMFOptimizer
      module_function

      # Run alternating multiplicative updates until convergence or the iteration limit.
      #
      # @param host [Object] The NMF estimator holding +components+ and hyper-parameters.
      # @param x [Numo::DFloat] The (possibly partially observed) data matrix.
      # @param observed_x [Numo::DFloat] The data matrix with unobserved entries replaced by zero.
      # @param mask [Numo::DFloat/Nil] The binary observation mask.
      # @param coefficients [Numo::DFloat] The coefficient matrix to optimize in place.
      # @param frozen_rows [Numo::Int32] The basis rows that must remain fixed.
      # @param update_comps [Boolean] Whether the basis matrix is allowed to change.
      # @return [Numo::DFloat] The optimized coefficient matrix.
      def optimize(host, x, observed_x, mask, coefficients, frozen_rows, update_comps)
        host.params[:max_iter].times do
          update_components(host, observed_x, mask, coefficients, frozen_rows, update_comps)
          update_coefficients(host, observed_x, mask, coefficients)
          normalize_components(host, coefficients, frozen_rows, update_comps)
          break if reconstruction_error(host, x, observed_x, mask, coefficients) < host.params[:tol]
        end
        coefficients
      end

      def update_components(host, observed_x, mask, coefficients, frozen_rows, update_comps)
        return unless update_comps

        components = host.components
        basis = coefficients.dot(components)
        basis[] = basis * mask unless mask.nil?
        factor = coefficients.transpose.dot(observed_x) / (coefficients.transpose.dot(basis) + host.params[:eps])
        if frozen_rows.empty?
          components[] = components * factor
        else
          free_rows = host.send(:free_component_rows, frozen_rows)
          components[free_rows, true] = components[free_rows, true] * factor[free_rows, true] unless free_rows.empty?
        end
      end

      def update_coefficients(host, observed_x, mask, coefficients)
        components = host.components
        reconstruction = coefficients.dot(components)
        reconstruction[] = reconstruction * mask unless mask.nil?
        numerator = observed_x.dot(components.transpose)
        denominator = reconstruction.dot(components.transpose) + host.params[:eps]
        # Use element-wise assignment so the caller's coefficient matrix is mutated in place.
        coefficients[] = coefficients * (numerator / denominator)
        # Multiplicative updates never create mass for rows without observations; pin them to zero.
        host.send(:zero_coefficients!, coefficients, mask)
      end

      def normalize_components(host, coefficients, frozen_rows, update_comps)
        return unless update_comps

        components = host.components
        norm = Numo::NMath.sqrt((components**2).sum(axis: 1)) + host.params[:eps]
        if frozen_rows.empty?
          components[] = components / norm.expand_dims(1)
          coefficients[] = coefficients * norm
        else
          free_rows = host.send(:free_component_rows, frozen_rows)
          normalize_free_rows(components, coefficients, norm, free_rows)
        end
      end

      def normalize_free_rows(components, coefficients, norm, free_rows)
        return if free_rows.empty?

        # Rescale only the free basis rows; frozen rows keep their original magnitude.
        components[free_rows, true] = components[free_rows, true] / norm[free_rows].expand_dims(1)
        coefficients[true, free_rows] = coefficients[true, free_rows] * norm[free_rows]
      end

      def reconstruction_error(host, x, observed_x, mask, coefficients)
        components = host.components
        return ((x - coefficients.dot(components))**2).sum(axis: 1).mean if mask.nil?

        residual = observed_x - (coefficients.dot(components) * mask)
        n_observed = mask.sum
        n_observed.zero? ? 0.0 : (residual**2).sum / n_observed
      end
    end

    # NMF is a class that implements Non-negative Matrix Factorization.
    #
    # @example
    #   require 'rumale/decomposition/nmf'
    #
    #   decomposer = Rumale::Decomposition::NMF.new(n_components: 2)
    #   representaion = decomposer.fit_transform(samples)
    #
    # *Reference*
    # - Xu, W., Liu, X., and Gong, Y., "Document Clustering Based On Non-negative Matrix Factorization," Proc. SIGIR' 03 , pp. 267--273, 2003.
    class NMF < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the factorization matrix.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with NMF.
      #
      # @param n_components [Integer] The number of components.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param eps [Float] A small value close to zero to avoid zero division error.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: 2, max_iter: 500, tol: 1.0e-4, eps: 1.0e-16, random_seed: nil)
        super()
        @params = {
          n_components: n_components,
          max_iter: max_iter,
          tol: tol,
          eps: eps,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> NMF
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [NMF] The learned transformer itself.
      # @overload fit(x, mask: nil, components: nil, frozen_rows: nil) -> NMF
      #   Fit the model with partially observed training data, optionally starting from a fixed basis.
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The non-negative training data.
      #     Unobserved entries may hold any value because they are ignored when a mask is given.
      #   @param mask [Numo::NArray/Nil] (shape: [n_samples, n_features]) The binary observation mask,
      #     where a nonzero value means the corresponding entry in x is observed and zero means it is missing.
      #     If nil is given, every entry is treated as observed.
      #   @param components [Numo::DFloat/Nil] (shape: [n_components, n_features]) The initial basis matrix.
      #     When a DFloat array is given it is reused in place without being copied.
      #   @param frozen_rows [Array/Numo::NArray/Nil] The indices of basis rows that must stay fixed.
      #     Frozen rows are neither updated nor rescaled during fitting.
      #   @return [NMF] The learned transformer itself.
      # @raise [ArgumentError] If x or components contains negative values, the mask has a wrong shape or
      #   non-binary values, a frozen row index is out of range, or fitting is attempted without any observed entry.
      def fit(x, _y = nil, mask: nil, components: nil, frozen_rows: nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        partial_fit(x, mask: mask, components: components, frozen_rows: frozen_rows)
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      # @overload fit_transform(x, mask: nil, components: nil, frozen_rows: nil) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The non-negative training data.
      #   @param mask [Numo::NArray/Nil] (shape: [n_samples, n_features]) The binary observation mask,
      #     where a nonzero value means the corresponding entry in x is observed and zero means it is missing.
      #   @param components [Numo::DFloat/Nil] (shape: [n_components, n_features]) The initial basis matrix,
      #     reused in place without being copied when a DFloat array is given.
      #   @param frozen_rows [Array/Numo::NArray/Nil] The indices of basis rows that must stay fixed.
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      #     Rows without any observed entry are returned as all-zero coefficient vectors.
      # @raise [ArgumentError] If the given data, mask, components, or frozen row indices are invalid.
      def fit_transform(x, _y = nil, mask: nil, components: nil, frozen_rows: nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        partial_fit(x, mask: mask, components: components, frozen_rows: frozen_rows)
      end

      # Transform the given data with the learned model.
      #
      # Only the coefficients of the given rows are optimized; the learned basis matrix is never modified.
      #
      # @overload transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      # @overload transform(x, mask: nil) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The non-negative data to be transformed.
      #   @param mask [Numo::NArray/Nil] (shape: [n_samples, n_features]) The binary observation mask,
      #     where a nonzero value means the corresponding entry in x is observed and zero means it is missing.
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      #     Rows without any observed entry are returned as all-zero coefficient vectors.
      # @raise [ArgumentError] If the given data or mask is invalid, or the model has not been fitted yet.
      def transform(x, mask: nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        check_fitted
        partial_fit(x, mask: mask, update_comps: false)
      end

      # Inverse transform the given transformed data with the learned model.
      #
      # @param z [Numo::DFloat] (shape: [n_samples, n_components]) The data to be restored into original space with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_featuress]) The restored data.
      def inverse_transform(z)
        z = ::Rumale::Validation.check_convert_sample_array(z)

        z.dot(@components)
      end

      private

      def partial_fit(x, mask: nil, components: nil, frozen_rows: nil, update_comps: true)
        n_samples, n_features = x.shape
        mask = NMFValidator.check_mask(mask, x.shape, require_observation: update_comps)
        NMFValidator.check_non_negative(x, 'the sample array is expected to contain non-negative values')
        n_components = @params[:n_components]
        # Missing entries are masked out of every update and of the reconstruction error.
        observed_x = mask.nil? ? x : x * mask
        scale = initialize_scale(observed_x, mask, n_components)
        sub_rng = @rng.dup

        if update_comps
          @components = initialize_components(components, n_components, n_features, scale, sub_rng)
          frozen_rows = NMFValidator.check_frozen_rows(frozen_rows, n_components)
        else
          frozen_rows = Numo::Int32[]
        end
        coefficients = ::Rumale::Utils.rand_uniform([n_samples, n_components], sub_rng) * scale
        # A row without any observed entry cannot determine any coefficient, so keep it exactly zero.
        zero_coefficients!(coefficients, mask)
        NMFOptimizer.optimize(self, x, observed_x, mask, coefficients, frozen_rows, update_comps)
      end

      def initialize_components(components, n_components, n_features, scale, sub_rng)
        if components.nil?
          # Keep the original dense initialization: random basis scaled by the data magnitude.
          return ::Rumale::Utils.rand_uniform([n_components, n_features], sub_rng) * scale
        end

        components = ::Rumale::Validation.check_convert_sample_array(components)
        unless components.shape == [n_components, n_features]
          raise ArgumentError, 'the components array is expected to have shape [n_components, n_features]'
        end

        NMFValidator.check_non_negative(components, 'the components array is expected to contain non-negative values')
        # Reuse the given basis in place so callers can lock a fixed basis without copying their data.
        components
      end

      def initialize_scale(observed_x, mask, n_components)
        if mask.nil?
          Math.sqrt(observed_x.mean / n_components)
        else
          n_observed = mask.sum
          n_observed.zero? ? 1.0 : Math.sqrt(observed_x.sum / (n_observed * n_components))
        end
      end

      def zero_coefficients!(coefficients, mask)
        return if mask.nil?

        missing_rows = mask.sum(axis: 1).eq(0).where
        return if missing_rows.empty?

        coefficients[missing_rows, true] = Numo::DFloat.zeros(missing_rows.size, coefficients.shape[1])
      end

      def free_component_rows(frozen_rows)
        Numo::Int32.cast((0...@components.shape[0]).to_a - frozen_rows.to_a)
      end

      def check_fitted
        raise ArgumentError, 'the NMF model has not been fitted yet' if @components.nil?
      end
    end
  end
end
