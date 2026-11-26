# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::AdjustedRandScore do
  let(:evaluator) { described_class.new }
  let(:ground_truth) { Numo::Int32[0, 0, 0, 1, 1, 1] }
  let(:predicted_a) { Numo::Int32[0, 0, 1, 1, 2, 2] }
  let(:predicted_b) { Numo::Int32[1, 1, 0, 0, 2, 2] }

  it 'calculates adjuested rand score of clustering result', :aggregate_failures do
    expect(evaluator.score(ground_truth, predicted_a)).to be_a(Float)
    expect(evaluator.score(ground_truth, predicted_a)).to be_within(1e-4).of(0.2424)
    expect(evaluator.score(ground_truth, predicted_b)).to be_within(1e-4).of(0.2424)
  end

  it 'returns one on special cases', :aggregate_failures do
    expect(evaluator.score(Numo::Int32[0, 1, 2], Numo::Int32[3, 4, 5])).to eq(1.0)
    expect(evaluator.score(Numo::Int32[0, 0, 0], Numo::Int32[1, 1, 1])).to eq(1.0)
    expect(evaluator.score(Numo::Int32[], Numo::Int32[])).to eq(1.0)
  end

  it 'calculates one third for the non-trivial sample regardless of label ids', :aggregate_failures do
    y_true = Numo::Int32[10, 10, 20, 30]
    y_pred = Numo::Int32[7, 7, 7, 8]
    expect(evaluator.score(y_true, y_pred)).to be_within(1e-12).of(1.0 / 3)
    expect(evaluator.score(Numo::Int32[101, 101, 202, 303], Numo::Int32[-7, -7, -7, 80])).to be_within(1e-12).of(1.0 / 3)
  end

  it 'accumulates label-pair and marginal counts across batches', :aggregate_failures do
    scorer = described_class.new
    batches = [
      [Numo::Int32[10, 20], Numo::Int32[7, 7]],
      [Numo::Int32[10, 30], Numo::Int32[7, 8]],
      [Numo::Int32[40, 30], Numo::Int32[9, 8]]
    ]

    expect(scorer.score).to eq(1.0)

    scores = batches.map { |y_true, y_pred| scorer.partial_score(y_true, y_pred) }
    expected = batches.each_index.map do |i|
      all_true = batches[0..i].flat_map { |y_true, _| y_true.to_a }
      all_pred = batches[0..i].flat_map { |_, y_pred| y_pred.to_a }
      described_class.new.score(all_true, all_pred)
    end

    expect(scores).to all(be_a(Float))
    expect(scores).to eq(expected)
    expect(scores[0]).to be_within(1e-12).of(0.0)
    expect(scores[1]).to be_within(1e-12).of(1.0 / 3)
    expect(scores[2]).to be_within(1e-12).of(22.0 / 37)
    expect(scorer.score).to be_within(1e-12).of(22.0 / 37)
    expect(scorer.current_ari).to be_within(1e-12).of(22.0 / 37)
  end

  it 'accepts label pairs and keeps one-shot scoring independent', :aggregate_failures do
    scorer = described_class.new
    expect(scorer.add_pair(10, 7).partial_score([[20, 7]])).to eq(0.0)
    expect(scorer.add_batch([[10, 7], [30, 8]])).to be_a(described_class)
    expect(scorer.current_score).to be_within(1e-12).of(1.0 / 3)

    full_score = described_class.new.score(Numo::Int32[10, 10, 20, 30], Numo::Int32[7, 7, 7, 8])
    expect(full_score).to be_within(1e-12).of(1.0 / 3)
    expect(scorer.score).to be_within(1e-12).of(1.0 / 3)
  end

  it 'stores only the labels and pairs that actually appear', :aggregate_failures do
    scorer = described_class.new
    scorer.add_batch(Numo::Int32[10, 20, 20], Numo::Int32[7, 7, 9])

    expect(scorer.n_samples).to eq(3)
    expect(scorer.true_counts).to eq(10 => 1, 20 => 2)
    expect(scorer.pred_counts).to eq(7 => 2, 9 => 1)
    expect(scorer.pair_counts).to eq([10, 7] => 1, [20, 7] => 1, [20, 9] => 1)
    expect(scorer.score).to be_within(1e-12).of(described_class.new.score([10, 20, 20], [7, 7, 9]))

    scorer.clear
    expect(scorer.n_samples).to eq(0)
    expect(scorer.pair_counts).to be_empty
    expect(scorer.score).to eq(1.0)
  end
end
