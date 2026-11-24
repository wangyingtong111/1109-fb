# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module Preprocessing
    # Generating polynomial features from the given samples.
    #
    # @example
    #   require 'rumale/preprocessing/polynomial_features'
    #
    #   transformer = Rumale::Preprocessing::PolynomialFeatures.new(degree: 2)
    #   x = Numo::DFloat[[0, 1], [2, 3], [4, 5]]
    #   z = transformer.fit_transform(x)
    #   p z
    #
    #   # Numo::DFloat#shape=[3,6]
    #   # [[1, 0, 1, 0, 0, 1],
    #   #  [1, 2, 3, 4, 6, 9],
    #   #  [1, 4, 5, 16, 20, 25]]
    #
    #   # If you want to perform polynomial regression, combine it with LinearRegression as follows:
    #   require 'rumale/preprocessing/polynomial_features'
    #   require 'rumale/linear_model/linear_regression'
    #   require 'rumale/pipeline/pipeline'
    #
    #   ply = Rumale::Preprocessing::PolynomialFeatures.new(degree: 2)
    #   reg = Rumale::LinearModel::LinearRegression.new(fit_bias: false, random_seed: 1)
    #   pipeline = Rumale::Pipeline::Pipeline.new(steps: { trs: ply, est: reg })
    #   pipeline.fit(training_samples, training_values)
    #   results = pipeline.predict(testing_samples)
    #
    # @example Grouped interaction mode (e.g. one-hot columns from the same source never interact):
    #   transformer = Rumale::Preprocessing::PolynomialFeatures.new(degree: 3, groups: [[0, 1], [2], [3]])
    #   x = Numo::DFloat[[2, 3, 5, 7]]
    #   z = transformer.fit_transform(x)
    #   p z
    #
    #   # Numo::DFloat#shape=[1,12]
    #   # [[1, 2, 3, 5, 7, 10, 14, 15, 21, 35, 70, 105]]
    #
    class PolynomialFeatures < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the number of polynomial features.
      # @return [Integer]
      attr_reader :n_output_features

      # Create a transformer for generating polynomial features.
      #
      # @param degree [Integer] The degree of polynomial features.
      # @param groups [Array<Array<Integer>>, Nil] The groups of input column indices used for grouped interaction mode.
      #   Two columns belonging to the same group never appear in the same product term, and a single variable
      #   never appears twice in a term (no powers of a variable). Columns not covered by any group are treated
      #   as singleton groups. The set of generated terms is fixed at fit time, so zero values in the data
      #   passed later do not remove output columns. If nil, ordinary polynomial features are generated.
      def initialize(degree: 2, groups: nil)
        raise ArgumentError, 'Expect the value of degree parameter greater than or eqaul to 1.' if degree < 1
        unless groups.nil? || (groups.is_a?(Array) && groups.all?(Array))
          raise ArgumentError, 'Expect the groups parameter to be an array of arrays of column indices or nil.'
        end

        super()
        @params = { degree: degree, groups: groups }
      end

      # Calculate the number of output polynomial fetures.
      #
      # @overload fit(x) -> PolynomialFeatures
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the number of output polynomial fetures.
      # @return [PolynomialFeatures]
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_features = x.shape[1]
        if @params[:groups].nil?
          @combinations = nil
          @n_output_features = 1
          @params[:degree].times do |t|
            @n_output_features += Array.new(n_features) { |n| n }.repeated_combination(t + 1).size
          end
        else
          groups = normalize_groups(@params[:groups], n_features)
          # Generate only the legal terms directly: pick distinct groups and take the
          # cartesian product of their member columns. Terms are sorted lexicographically
          # by column index, so the output column order is fixed after fitting.
          @combinations = {}
          (2..@params[:degree]).each do |d|
            combos = []
            groups.combination(d) do |gs|
              gs.first.product(*gs[1..]).each { |combo| combos.push(combo.sort) }
            end
            @combinations[d] = combos.sort.uniq
          end
          @n_output_features = 1 + n_features + @combinations.values.sum(&:size)
        end
        self
      end

      # Calculate the number of polynomial features, and then transform samples to polynomial features.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the number of polynomial features
      #     and be transformed.
      # @return [Numo::DFloat] (shape: [n_samples, n_output_features]) The transformed samples.
      def fit_transform(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x).transform(x)
      end

      # Transform the given samples to polynomial features.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed.
      # @return [Numo::DFloat] (shape: [n_samples, n_output_features]) The transformed samples.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        # initialize transformed features
        n_samples, n_features = x.shape
        z = Numo::DFloat.zeros(n_samples, n_output_features)
        # bias
        z[true, 0] = 1
        curr_col = 1
        # itself
        z[true, 1..n_features] = x
        curr_col += n_features
        # high degree features
        if @combinations.nil?
          curr_feat_ids = Array.new(n_features + 1) { |n| n + 1 }
          (1...@params[:degree]).each do
            next_feat_ids = []
            n_features.times do |d|
              f_range = curr_feat_ids[d]...curr_feat_ids.last
              next_col = curr_col + f_range.size
              z[true, curr_col...next_col] = z[true, f_range] * x[true, d..d]
              next_feat_ids.push(curr_col)
              curr_col = next_col
            end
            next_feat_ids.push(curr_col)
            curr_feat_ids = next_feat_ids
          end
        else
          # Compute only the legal terms fixed at fit time. Lower-degree terms are
          # reused for building higher-degree ones, so the intermediate feature scale
          # grows with the number of legal terms, not with the full expansion.
          term_values = {}
          (2..@params[:degree]).each do |d|
            @combinations[d].each do |combo|
              prefix = combo[0...-1]
              base = prefix.size == 1 ? x[true, prefix[0]] : term_values[prefix]
              term = base * x[true, combo[-1]]
              z[true, curr_col] = term
              term_values[combo] = term
              curr_col += 1
            end
          end
        end
        z
      end

      private

      # Normalize the given groups into a partition of all column indices.
      # Columns not covered by any group become singleton groups.
      def normalize_groups(groups, n_features)
        assigned = {}
        normalized = groups.map do |group|
          group.map do |col|
            unless col.is_a?(Integer) && col >= 0 && col < n_features
              raise ArgumentError, "Invalid column index in groups: #{col.inspect}."
            end
            raise ArgumentError, "Column index #{col} appears in multiple groups." if assigned[col]

            assigned[col] = true
            col
          end.sort
        end
        normalized.reject!(&:empty?)
        normalized.concat((0...n_features).reject { |col| assigned[col] }.zip)
        normalized
      end
    end
  end
end
