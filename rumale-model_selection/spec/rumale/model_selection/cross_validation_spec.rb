# frozen_string_literal: true

require 'spec_helper'

# A sub model that accumulates the row ids given to fit.
class RowIdAccumulator
  attr_reader :seen_ids

  def initialize
    @seen_ids = []
  end

  def fit(ids)
    @seen_ids.concat(ids)
  end
end

# An estimator that carries state before training, embeds a sub model accumulating row ids,
# and holds a non-marshalable scorer (a fixed closure). It implements the explicit
# configuration-rebuild contract (#rebuild_estimator) to be cloned for each fold.
class StatefulEstimator
  SCORER = ->(history, x) { history.sum * 10 + x[true, 0].to_a.sum }

  attr_reader :history, :sub_model

  class << self
    def fold_estimators
      @fold_estimators ||= []
    end

    def reset_fold_estimators
      @fold_estimators = []
    end
  end

  def initialize
    @history = []
    @sub_model = RowIdAccumulator.new
    @scorer = SCORER
  end

  def rebuild_estimator
    estimator = self.class.new
    self.class.fold_estimators.push(estimator)
    estimator
  end

  def fit(x, _y)
    ids = x[true, 0].to_a
    @history.concat(ids)
    @sub_model.fit(ids)
    self
  end

  def score(x, _y)
    @scorer.call(@history, x)
  end
end

# An estimator of which fit method raises an error on the fold containing the row id 4.
class FailingFoldEstimator < StatefulEstimator
  # An error raised when fitting the fold containing the row id 4.
  class FitError < StandardError; end

  def fit(x, y)
    raise FitError, 'fit failed on fold' if x[true, 0].to_a.include?(4.0)

    super
  end
end

# A splitter that returns the given folds as-is.
class FixedSplitter
  def initialize(folds)
    @folds = folds
  end

  def split(_x, _y)
    @folds
  end
end

RSpec.describe Rumale::ModelSelection::CrossValidation do
  let(:xor) { xor_dataset }
  let(:samples) { xor[0] }
  let(:labels) { xor[1] }
  let(:values) { samples.dot(Numo::DFloat[1.0, 2.0]) }
  let(:kernel_mat) { Rumale::PairwiseMetric.rbf_kernel(samples, nil, 1.0) }
  let(:kernel_svc) { Rumale::KernelMachine::KernelSVC.new(reg_param: 1.0, max_iter: 1000, random_seed: 1) }
  let(:linear_svc) { Rumale::LinearModel::SVC.new }
  let(:linear_svr) { Rumale::LinearModel::SVR.new }
  let(:logit_reg) { Rumale::LinearModel::LogisticRegression.new }
  let(:f_score) { Rumale::EvaluationMeasure::FScore.new }
  let(:log_loss) { Rumale::EvaluationMeasure::LogLoss.new }
  let(:n_splits) { 5 }
  let(:kfold) { Rumale::ModelSelection::KFold.new(n_splits: n_splits, shuffle: true, random_seed: 1) }
  let(:skfold) { Rumale::ModelSelection::StratifiedKFold.new(n_splits: n_splits, shuffle: true, random_seed: 1) }

  it 'performs k-fold cross validation with linear svc', :aggregate_failures do
    cv = described_class.new(estimator: linear_svc, splitter: kfold)
    report = cv.perform(samples, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs k-fold cross validation with kernel svc', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: kfold)
    report = cv.perform(kernel_mat, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs stratified k-fold cross validation with kernel svc', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: skfold)
    report = cv.perform(kernel_mat, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs k-fold cross validation with linear svr', :aggregate_failures do
    cv = described_class.new(estimator: linear_svr, splitter: kfold)
    report = cv.perform(samples, values)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'also calculates scores of training dataset', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: skfold, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].sum / n_splits
    mean_train_score = report[:train_score].sum / n_splits
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.1).of(0.9)
    expect(mean_train_score).to eq(1.0)
  end

  it 'performs k-fold cross validation with kernel svc to evaluate the results using F1-score', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: kfold, evaluator: f_score, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].sum / n_splits
    mean_train_score = report[:train_score].sum / n_splits
    expect(cv.evaluator).to be_a(Rumale::EvaluationMeasure::FScore)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.1).of(0.9)
    expect(mean_train_score).to eq(1.0)
  end

  it 'performs k-fold cross validation with logistic regression to evaluate the results using Log-loss',
     :aggregate_failures do
    cv = described_class.new(estimator: logit_reg, splitter: kfold, evaluator: log_loss, return_train_score: true)
    report = cv.perform(samples, labels)
    mean_test_score = report[:test_score].sum / n_splits
    mean_train_score = report[:train_score].sum / n_splits
    expect(cv.evaluator).to be_a(Rumale::EvaluationMeasure::LogLoss)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.01).of(0.7)
    expect(mean_train_score).to be_within(0.01).of(0.7)
  end

  describe 'private method' do
    let(:kernel_svc_cv) { described_class.new(estimator: kernel_svc, splitter: kfold) }
    let(:linear_svc_cv) { described_class.new(estimator: linear_svc, splitter: kfold) }

    it 'detects type of classifier', :aggregate_failures do
      expect(kernel_svc_cv.send(:kernel_machine?)).to be_truthy
      expect(linear_svc_cv.send(:kernel_machine?)).to be_falsey
    end
  end

  describe 'estimator cloning' do
    let(:fold_samples) { Numo::DFloat[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]] }
    let(:fold_labels) { Numo::Int32[0, 1, 0, 1, 0, 1] }
    let(:folds) do
      ids_a = Numo::Int32[0, 1, 2]
      ids_b = Numo::Int32[3, 4, 5]
      [[ids_a, ids_b], [ids_b, ids_a]]
    end
    let(:stateful_estimator) do
      estimator = StatefulEstimator.new
      estimator.history.push(:sentinel)
      estimator.sub_model.fit([-1.0])
      estimator
    end

    before { StatefulEstimator.reset_fold_estimators }

    it 'gives each fold an unfitted estimator rebuilt from the configuration', :aggregate_failures do
      cv = described_class.new(estimator: stateful_estimator, splitter: FixedSplitter.new(folds))
      report = cv.perform(fold_samples, fold_labels)
      built = StatefulEstimator.fold_estimators
      expect(built.size).to eq(2)
      expect(built[0].history).to eq([0.0, 1.0, 2.0])
      expect(built[1].history).to eq([3.0, 4.0, 5.0])
      expect(built[0].sub_model.seen_ids).to eq([0.0, 1.0, 2.0])
      expect(built[1].sub_model.seen_ids).to eq([3.0, 4.0, 5.0])
      expect(report[:test_score]).to eq([42.0, 123.0])
      expect(stateful_estimator.history).to eq([:sentinel])
      expect(stateful_estimator.sub_model.seen_ids).to eq([-1.0])
    end

    it 'produces results independent of fold order', :aggregate_failures do
      cv = described_class.new(estimator: stateful_estimator, splitter: FixedSplitter.new(folds))
      report = cv.perform(fold_samples, fold_labels)
      StatefulEstimator.reset_fold_estimators
      swapped_cv = described_class.new(estimator: stateful_estimator, splitter: FixedSplitter.new(folds.reverse))
      swapped_report = swapped_cv.perform(fold_samples, fold_labels)
      expect(report[:test_score]).to eq([42.0, 123.0])
      expect(swapped_report[:test_score]).to eq([123.0, 42.0])
      expect(swapped_report[:test_score].sort).to eq(report[:test_score].sort)
      built = StatefulEstimator.fold_estimators
      expect(built[0].history).to eq([3.0, 4.0, 5.0])
      expect(built[1].history).to eq([0.0, 1.0, 2.0])
      expect(stateful_estimator.history).to eq([:sentinel])
      expect(stateful_estimator.sub_model.seen_ids).to eq([-1.0])
    end

    it 'reports a fit failure as-is' do
      estimator = FailingFoldEstimator.new
      cv = described_class.new(estimator: estimator, splitter: FixedSplitter.new(folds))
      expect { cv.perform(fold_samples, fold_labels) }.to raise_error(FailingFoldEstimator::FitError, 'fit failed on fold')
      expect(estimator.history).to be_empty
    end

    it 'does not share trained state with the given simple estimator' do
      cv = described_class.new(estimator: linear_svc, splitter: kfold)
      cv.perform(samples, labels)
      expect(linear_svc.instance_variable_get(:@weight_vec)).to be_nil
    end
  end
end
