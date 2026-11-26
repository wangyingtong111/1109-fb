# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/preprocessing/label_encoder'

module Rumale
  module Ensemble
    # VotingClassifier is a class that implements classifier with voting ensemble method.
    #
    # @example
    #   require 'rumale/ensemble/voting_classifier'
    #
    #   estimators = {
    #     lgr: Rumale::LinearModel::LogisticRegression.new(reg_param: 1e-2),
    #     mlp: Rumale::NeuralNetwork::MLPClassifier.new(hidden_units: [256], random_seed: 1),
    #     rnd: Rumale::Ensemble::RandomForestClassifier.new(random_seed: 1)
    #   }
    #   weights = { lgr: 0.2, mlp: 0.3, rnd: 0.5 }
    #
    #   classifier = Rumale::Ensemble::VotingClassifier.new(estimators: estimators, weights: weights, voting: 'soft')
    #   classifier.fit(x_train, y_train)
    #   results = classifier.predict(x_test)
    #
    # *Reference*
    # - Zhou, Z-H., "Ensemble Methods - Foundations and Algorithms," CRC Press Taylor and Francis Group, Chapman and Hall/CRC, 2012.
    class VotingClassifier < ::Rumale::Base::Estimator
      include ::Rumale::Base::Classifier

      # Return the sub-classifiers that voted.
      # @return [Hash<Symbol,Classifier>]
      attr_reader :estimators

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Create a new ensembled classifier with voting rule.
      #
      # @param estimators [Hash<Symbol,Classifier>] The sub-classifiers to vote.
      # @param weights [Hash<Symbol,Float>] The weight value for each classifier.
      # @param voting [String] The voting rule for the predicted results of each classifier.
      #   If 'hard' is given, the ensembled classifier predicts the class label by majority vote.
      #   If 'soft' is given, the ensembled classifier uses the weighted average of predicted probabilities for the prediction.
      # @param probability_classes [Hash<Symbol,Array<Integer>>,Array<Integer>,Nil]
      #   The internal class IDs corresponding to probability columns. When a Hash is given,
      #   each entry is used for the estimator with the same name. An Array is applied to all
      #   sub-estimators. Estimator-level declarations such as #numbers take precedence over
      #   this option.
      def initialize(estimators:, weights: nil, voting: 'hard', probability_classes: nil, **options)
        super()
        @estimators = estimators
        @params = {
          weights: weights || estimators.each_key.with_object({}) { |name, w| w[name] = 1.0 },
          voting: voting,
          probability_classes: probability_classes || extract_probability_classes_option(options)
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [VotingClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        @encoder = ::Rumale::Preprocessing::LabelEncoder.new
        y_encoded = @encoder.fit_transform(y)
        @classes = Numo::NArray[*@encoder.classes]
        @estimators.each_key { |name| @estimators[name].fit(x, y_encoded) }

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) The confidence score per sample.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        return predict_proba(x) if soft_voting?

        n_samples = x.shape[0]
        n_classes = @classes.size
        z = Numo::DFloat.zeros(n_samples, n_classes)
        @estimators.each do |name, estimator|
          estimator.predict(x).to_a.each_with_index { |c, i| z[i, c] += @params[:weights][name] }
        end
        z
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) The predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        n_classes = @classes.size
        z = decision_function(x)
        predicted = z.max_index(axis: 1) - Numo::Int32.new(n_samples).seq * n_classes
        Numo::Int32.cast(@encoder.inverse_transform(predicted))
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probabilities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        n_classes = @classes.size
        z = Numo::DFloat.zeros(n_samples, n_classes)
        sum_weight = @params[:weights].each_value.sum
        @estimators.each do |name, estimator|
          probs = estimator.predict_proba(x)
          probs = align_probabilities(probs, declared_probability_classes(name, estimator))
          z += @params[:weights][name] * probs
        end
        z /= sum_weight
      end

      private

      PROBABILITY_CLASS_METHODS = %i[
        numbers
        class_numbers
        probability_class_numbers
        probability_column_ids
        probability_column_classes
        proba_column_ids
        proba_classes
        probability_classes
        probability_encoding
        class_encoding
        column_encoding
        encoded_classes
      ].freeze

      private_constant :PROBABILITY_CLASS_METHODS

      def soft_voting?
        @params[:voting] == 'soft'
      end

      def extract_probability_classes_option(options)
        options.values_at(
          :probability_class_numbers, :class_numbers, :numbers,
          :probability_encodings, :class_encodings, :column_encodings
        ).compact.first
      end

      def declared_probability_classes(name, estimator)
        declaration = declaration_from_estimator(estimator)
        if declaration.nil?
          option = @params[:probability_classes]
          declaration = option.is_a?(Hash) ? option[name] : option
        end
        declaration&.to_a
      end

      def declaration_from_estimator(estimator)
        PROBABILITY_CLASS_METHODS.each do |method_name|
          next unless estimator.respond_to?(method_name)

          declaration = estimator.public_send(method_name)
          return declaration unless declaration.nil?
        end

        estimator.classes if estimator.respond_to?(:classes)
      end

      def align_probabilities(probs, declared)
        return probs if declared.nil?

        class_ids = normalize_class_ids(declared)
        return probs unless valid_class_ids?(class_ids, probs.shape[1])

        aligned = Numo::DFloat.zeros(probs.shape[0], @classes.size)
        class_ids.each_with_index do |class_id, column|
          aligned[true, class_id] = probs[true, column]
        end
        aligned
      end

      def normalize_class_ids(declared)
        declared.to_a.map do |class_id|
          case class_id
          when Integer
            class_id
          when Numeric
            integer_id = class_id.to_i
            integer_id if integer_id == class_id
          else
            Integer(class_id)
          end
        end
      rescue ArgumentError, TypeError
        nil
      end

      def valid_class_ids?(class_ids, n_columns)
        return false if class_ids.nil?
        return false unless class_ids.size == n_columns
        return false if class_ids.any?(&:nil?)

        class_ids.all? { |class_id| class_id.is_a?(Integer) && class_id >= 0 && class_id < @classes.size }
      end
    end
  end
end
