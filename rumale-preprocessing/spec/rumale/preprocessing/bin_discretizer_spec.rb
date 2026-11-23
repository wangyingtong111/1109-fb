# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::BinDiscretizer do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand - 0.5 }
  let(:n_bins) { 8 }
  let(:discretizer) { described_class.new(n_bins: n_bins) }
  let(:transformed) { discretizer.fit_transform(samples) }

  it 'discretizes with k-bins', :aggregate_failures do
    expect(transformed.shape[0]).to eq(n_samples)
    expect(transformed.shape[1]).to eq(n_features)
    expect(transformed[true, 0].to_a.uniq.size).to be <= n_bins
    expect(discretizer.feature_steps).to be_a(Array)
    expect(discretizer.feature_steps[0]).to be_a(Numo::DFloat)
    expect(discretizer.feature_steps.size).to eq(n_features)
    expect(discretizer.feature_steps[0].size).to eq(n_bins)
  end

  context 'when strategy is uniform' do
    let(:samples) { Numo::DFloat[[0], [0], [0], [0], [10], [20]] }
    let(:n_bins) { 4 }

    it 'divides the feature range into equal-width bins' do
      discretizer.fit(samples)
      expect(discretizer.feature_steps[0].to_a).to eq([0.0, 5.0, 10.0, 15.0])
    end

    it 'assigns values below the lowest edge to the first bin and above the highest edge to the last bin' do
      queried = Numo::DFloat[[-1], [0], [7.5], [20], [99]]
      expect(discretizer.fit(samples).transform(queried).to_a.flatten).to eq([0.0, 0.0, 1.0, 3.0, 3.0])
    end
  end

  context 'when strategy is quantile' do
    let(:n_bins) { 4 }
    let(:discretizer) { described_class.new(n_bins: n_bins, strategy: 'quantile') }
    let(:samples) { Numo::DFloat[[0], [0], [0], [0], [10], [20]] }

    it 'determines the edges with linear interpolation and merges duplicated edges', :aggregate_failures do
      discretizer.fit(samples)
      expect(discretizer.feature_steps[0].to_a).to eq([0.0, 7.5])
      expect(discretizer.feature_steps[0].size).to be < n_bins
    end

    it 'discretizes values with bins including the left edge and the last bin including the upper edge', :aggregate_failures do
      queried = Numo::DFloat[[-1], [0], [7.5], [20], [99], [Float::NAN]]
      expect(discretizer.fit_transform(samples).to_a.flatten).to eq([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
      expect(discretizer.transform(queried).to_a.flatten).to eq([0.0, 0.0, 1.0, 1.0, 1.0, -1.0])
    end

    it 'uses only the edges determined at fit time' do
      discretizer.fit(samples)
      steps = discretizer.feature_steps[0].dup
      discretizer.transform(Numo::DFloat[[-100], [100]])
      expect(discretizer.feature_steps[0]).to eq(steps)
    end

    it 'assigns missing values to -1 instead of the first bin', :aggregate_failures do
      samples_with_missing = Numo::DFloat[[0], [0], [Float::NAN], [10], [10]]
      discretized = described_class.new(n_bins: 2, strategy: 'quantile').fit_transform(samples_with_missing)
      expect(discretized.to_a.flatten).to eq([0.0, 0.0, -1.0, 1.0, 1.0])
      expect(discretized[2, 0]).to eq(-1.0)
    end

    it 'ignores missing values when determining the edges' do
      discretizer.fit(Numo::DFloat[[0], [0], [0], [0], [10], [20], [Float::NAN]])
      expect(discretizer.feature_steps[0].to_a).to eq([0.0, 7.5])
    end

    context 'when all values of a feature are missing' do
      let(:samples) { Numo::DFloat[[Float::NAN], [Float::NAN]] }

      it 'returns no edges and assigns all values to -1', :aggregate_failures do
        expect(discretizer.fit_transform(samples).to_a.flatten).to eq([-1.0, -1.0])
        expect(discretizer.feature_steps[0].size).to be_zero
      end
    end

    context 'when all values of a feature are identical' do
      let(:samples) { Numo::DFloat[[3], [3], [3]] }

      it 'returns a single edge' do
        discretizer.fit(samples)
        expect(discretizer.feature_steps[0].to_a).to eq([3.0])
      end
    end
  end

  context 'when given invalid parameters' do
    it 'raises an error for a non-positive number of bins' do
      expect { described_class.new(n_bins: 0) }.to raise_error(ArgumentError)
    end

    it 'raises an error for an unknown strategy' do
      expect { described_class.new(strategy: 'kmeans') }.to raise_error(ArgumentError)
    end
  end
end
