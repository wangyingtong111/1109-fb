# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::ROCAUC do
  let(:roc_auc) { described_class.new }
  let(:y_bin) { Numo::Int32[1, 1, 2, 2] }
  let(:y_bin_score) { Numo::DFloat[0.1, 0.4, 0.35, 0.8] }
  let(:y_mult) do
    Numo::Int32[[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0]]
  end
  let(:y_mult_score) do
    Numo::DFloat[[0.6, 0.2, 0.1, 0.1],
                 [0.1, 0.6, 0.2, 0.1],
                 [0.1, 0.6, 0.2, 0.1],
                 [0.1, 0.2, 0.1, 0.6],
                 [0.1, 0.2, 0.1, 0.6],
                 [0.1, 0.1, 0.6, 0.2],
                 [0.1, 0.1, 0.6, 0.2],
                 [0.6, 0.2, 0.1, 0.1]]
  end
  let(:y_class) { Numo::Int32[10, 20, 10] }
  let(:y_class_score) do
    Numo::DFloat[[0.05, 0.88, 0.07],
                 [0.03, 0.12, 0.85],
                 [0.08, 0.80, 0.12]]
  end
  let(:column_labels) { [30, 10, 20] }

  it 'caclulates ROC-AUC', :aggregate_failures do
    expect(roc_auc.score(y_bin, y_bin_score)).to be_within(1e-4).of(0.75)
    expect(roc_auc.score(y_mult, y_mult_score)).to be_within(1e-4).of(1.0)
  end

  it 'calculates ROC-AUC for each class from a label vector and the column labels' do
    scores = roc_auc.score_each_class(y_class, y_class_score, column_labels)
    expect(scores).to eq(30 => nil, 10 => 1.0, 20 => 1.0)
  end

  it 'returns the ROC-AUC of each class from the score method when the average is none' do
    scores = roc_auc.score(y_class, y_class_score, column_labels, average: 'none')
    expect(scores).to eq(30 => nil, 10 => 1.0, 20 => 1.0)
  end

  it 'accepts the average type in the constructor' do
    evaluator = described_class.new(average: 'none', labels: column_labels)
    expect(evaluator.average).to eq('none')
    expect(evaluator.score(y_class, y_class_score)).to eq(30 => nil, 10 => 1.0, 20 => 1.0)
  end

  it 'accepts the column labels with the labels keyword argument' do
    scores = roc_auc.score_each_class(y_class, y_class_score, labels: column_labels)
    expect(scores).to eq(30 => nil, 10 => 1.0, 20 => 1.0)
  end

  it 'binds the class of each predicted column to the column labels', :aggregate_failures do
    scores = roc_auc.score_each_class(Numo::Int32[20, 10, 10],
                                      Numo::DFloat[[0.03, 0.12, 0.85],
                                                   [0.05, 0.88, 0.07],
                                                   [0.08, 0.80, 0.12]],
                                      column_labels)
    expect(scores.keys).to eq(column_labels)
    expect(scores).to eq(30 => nil, 10 => 1.0, 20 => 1.0)
  end

  it 'skips the classes without positive examples in the macro average' do
    expect(roc_auc.score(y_class, y_class_score, column_labels, skip_undefined: true)).to be_within(1e-4).of(1.0)
  end

  it 'skips the class missing from the validation fold when the column order differs', :aggregate_failures do
    fold_score = Numo::DFloat[[0.88, 0.07, 0.05],
                              [0.12, 0.85, 0.03],
                              [0.80, 0.12, 0.08]]
    scores = roc_auc.score_each_class(y_class, fold_score, [10, 20, 30])
    expect(scores).to eq(10 => 1.0, 20 => 1.0, 30 => nil)
    expect(roc_auc.score(y_class, fold_score, [10, 20, 30], skip_undefined: true)).to be_within(1e-4).of(1.0)
  end

  it 'accepts the column labels and the skipping rule in the constructor' do
    evaluator = described_class.new(labels: column_labels, skip_undefined: true)
    expect(evaluator.labels).to eq(column_labels)
    expect(evaluator.skip_undefined).to be(true)
    expect(evaluator.score(y_class, y_class_score)).to be_within(1e-4).of(1.0)
  end

  it 'raises ArgumentError given a class without positive examples when skipping is disabled' do
    expect { roc_auc.score(y_class, y_class_score, column_labels) }.to raise_error(ArgumentError)
  end

  it 'uses the sorted unique labels of y_true when the column labels are not given' do
    scores = roc_auc.score_each_class(y_class,
                                      Numo::DFloat[[0.88, 0.07],
                                                   [0.12, 0.85],
                                                   [0.80, 0.12]])
    expect(scores).to eq(10 => 1.0, 20 => 1.0)
    expect(roc_auc.score(y_class, Numo::DFloat[[0.88, 0.07], [0.12, 0.85], [0.80, 0.12]])).to be_within(1e-4).of(1.0)
  end

  it 'calculates the macro average from a label vector and the column labels' do
    expect(roc_auc.score(Numo::Int32[10, 20], Numo::DFloat[[0.9, 0.1], [0.2, 0.8]], [10, 20])).to be_within(1e-4).of(1.0)
  end

  it 'calculates chance level ROC-AUC given equal scores' do
    expect(roc_auc.score(Numo::Int32[10, 20], Numo::DFloat[[0.5, 0.5], [0.5, 0.5]], [10, 20])).to be_within(1e-4).of(0.5)
  end

  it 'raises ArgumentError given column labels whose size differs from the columns of y_score' do
    expect { roc_auc.score_each_class(y_class, y_class_score, [30, 10]) }.to raise_error(ArgumentError)
  end

  it 'raises ArgumentError given a class of y_true that is not included in the column labels' do
    expect { roc_auc.score_each_class(y_class, y_class_score, [30, 10, 40]) }.to raise_error(ArgumentError)
  end

  it 'calculates receiver operating characteristic curve', :aggregate_failures do
    fpr, tpr, thd = roc_auc.roc_curve(y_bin, y_bin_score, 2)
    expect(fpr).to eq(Numo::DFloat[0, 0, 0.5, 0.5, 1])
    expect(tpr).to eq(Numo::DFloat[0, 0.5, 0.5, 1, 1])
    expect(thd).to eq(Numo::DFloat[1.8, 0.8, 0.4, 0.35, 0.1])
  end

  it 'calculates area under the ROC' do
    fpr, tpr, = roc_auc.roc_curve(y_bin, y_bin_score, 2)
    expect(roc_auc.auc(fpr, tpr)).to be_within(1e-4).of(0.75)
  end

  it 'keeps the threshold rule for tied binary scores', :aggregate_failures do
    fpr, tpr, thd = roc_auc.roc_curve(Numo::Int32[1, 1, 2, 2], Numo::DFloat[0.2, 0.6, 0.6, 0.8], 2)
    expect(fpr).to eq(Numo::DFloat[0, 0, 0.5, 1])
    expect(tpr).to eq(Numo::DFloat[0, 0.5, 1, 1])
    expect(thd).to eq(Numo::DFloat[1.8, 0.8, 0.6, 0.2])
    expect(roc_auc.score(Numo::Int32[1, 1, 2, 2], Numo::DFloat[0.2, 0.6, 0.6, 0.8])).to be_within(1e-4).of(0.875)
  end

  it 'raises ArgumentError given non-existent label to the roc_curve method' do
    expect { roc_auc.roc_curve(Numo::Int32[0, 1, 2], Numo::DFloat[0.1, 0.2, 0.8], 3) }.to raise_error(ArgumentError)
  end

  it 'raises ArgumentError given ground truth multi-label to the roc_curve method' do
    expect { roc_auc.roc_curve(Numo::Int32[0, 1, 2], Numo::DFloat[0.1, 0.2, 0.8]) }.to raise_error(ArgumentError)
  end

  it 'raises ArgumentError given arrays with only one element to the auc method', :aggregate_failures do
    expect { roc_auc.auc(Numo::Int32[0, 1, 2], Numo::Int32[1]) }.to raise_error(ArgumentError)
    expect { roc_auc.auc(Numo::Int32[0], Numo::Int32[1, 2, 3]) }.to raise_error(ArgumentError)
    expect { roc_auc.auc(Numo::Int32[0], Numo::Int32[1]) }.to raise_error(ArgumentError)
  end
end
