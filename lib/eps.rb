# dependencies
require "lightgbm"
require "matrix"
require "nokogiri"

# stdlib
require "json"

# modules
require_relative "eps/base"
require_relative "eps/base_estimator"
require_relative "eps/data_frame"
require_relative "eps/label_encoder"
require_relative "eps/lightgbm"
require_relative "eps/linear_regression"
require_relative "eps/metrics"
require_relative "eps/model"
require_relative "eps/naive_bayes"
require_relative "eps/statistics"
require_relative "eps/text_encoder"
require_relative "eps/utils"
require_relative "eps/version"

# pmml
require_relative "eps/pmml"
require_relative "eps/pmml/generator"
require_relative "eps/pmml/loader"

# evaluators
require_relative "eps/evaluators/linear_regression"
require_relative "eps/evaluators/lightgbm"
require_relative "eps/evaluators/naive_bayes"
require_relative "eps/evaluators/node"

module Eps
  class Error < StandardError; end
  class UnstableSolution < Error; end

  # Computes metrics for a batch of predictions.
  #
  # Predictions may be the usual point predictions (numeric for regression,
  # labels for classification) or the label-to-probability Hashes returned by
  # predict_probability. With probability Hashes the result includes log loss
  # as well as accuracy taken from each row's most likely label.
  def self.metrics(y_true, y_pred, weight: nil)
    if probabilities?(y_pred)
      y_true = y_true.to_a
      y_pred = y_pred.to_a
      weight = weight.to_a if weight && !weight.is_a?(Array) && weight.respond_to?(:to_a)
      {
        log_loss: Metrics.log_loss(y_true, y_pred, weight: weight),
        accuracy: Metrics.accuracy(y_true, most_likely_labels(y_pred), weight: weight)
      }
    elsif Utils.column_type(y_true, "actual") == "numeric"
      {
        rmse: Metrics.rmse(y_true, y_pred, weight: weight),
        mae: Metrics.mae(y_true, y_pred, weight: weight),
        me: Metrics.me(y_true, y_pred, weight: weight)
      }
    else
      {
        accuracy: Metrics.accuracy(y_true, y_pred, weight: weight)
      }
    end
  end

  # True when predictions are label-to-probability Hashes. An empty batch
  # keeps the usual point-prediction behavior.
  def self.probabilities?(y_pred)
    return false unless y_pred.respond_to?(:to_a)

    y_pred = y_pred.to_a
    y_pred.any? && y_pred.all? { |v| v.is_a?(Hash) }
  end
  private_class_method :probabilities?

  # The label with the highest probability for each row.
  #
  # Ties break on the label text so the choice does not depend on Hash key
  # order.
  def self.most_likely_labels(y_pred)
    y_pred.map { |row| row.min_by { |label, probability| [-probability, label.to_s] }&.first }
  end
  private_class_method :most_likely_labels

  # backwards compatibility
  Regressor = Model
end
