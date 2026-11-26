# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsRegressor do
  let(:dataset) { two_clusters_dataset }
  let(:single_target) { dataset[0][true, 0] + dataset[0][true, 1]**2 }
  let(:multi_target) { dataset[0].dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_outputs) { y.shape[1] }
  let(:estimator) { described_class.new(n_neighbors: 5, metric: metric).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { dataset[0] }
    let(:n_features) { x.shape[1] }

    context 'when single target problem' do
      let(:y) { single_target }

      it 'learns the model for single regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_a(Numo::DFloat)
        expect(estimator.prototypes).to be_contiguous
        expect(estimator.prototypes.ndim).to eq(2)
        expect(estimator.prototypes.shape[0]).to eq(n_samples)
        expect(estimator.prototypes.shape[1]).to eq(n_features)
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(1)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.01).of(1.0)
      end
    end

    context 'when multi-target problem' do
      let(:y) { multi_target }

      it 'learns the model for multiple regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_a(Numo::DFloat)
        expect(estimator.prototypes).to be_contiguous
        expect(estimator.prototypes.ndim).to eq(2)
        expect(estimator.prototypes.shape[0]).to eq(n_samples)
        expect(estimator.prototypes.shape[1]).to eq(n_features)
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(2)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(estimator.values.shape[1]).to eq(n_outputs)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end

  context 'when metric is "precomputed"' do
    let(:metric) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.manhattan_distance(dataset[0]) }

    context 'when single target problem' do
      let(:y) { single_target }

      it 'learns the model for single regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_nil
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(1)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.01).of(1.0)
      end
    end

    context 'when multi-target problem' do
      let(:y) { multi_target }

      it 'learns the model for multiple regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_nil
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(2)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(estimator.values.shape[1]).to eq(n_outputs)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end

    context 'when wrong size matrix is given' do
      let(:estimator) { described_class.new(n_neighbors: 5, metric: 'precomputed') }
      let(:y) { single_target }

      it 'raises ArgumentError', :aggregate_failures do
        expect { estimator.fit(Numo::DFloat.new(n_samples, 2).rand, y) }.to raise_error(ArgumentError)
        expect { estimator.fit(x, y).predict(Numo::DFloat.new(2, n_samples / 2).rand) }.to raise_error(ArgumentError)
      end
    end
  end

  context 'when weights is "distance"' do
    let(:training_samples) { Numo::DFloat[[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]] }
    let(:targets) { Numo::DFloat[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]] }
    let(:zero_distances) { Numo::DFloat[[0.0, 0.0, 2.0]] }
    let(:tiny_distances) { Numo::DFloat[[1e-310, 2e-310, 4e-310]] }
    let(:query) { Numo::DFloat[[0.0, 0.0]] }

    context 'when metric is "euclidean"' do
      let(:estimator) { described_class.new(n_neighbors: 3, weights: 'distance') }

      it 'returns a two-dimensional prediction array', :aggregate_failures do
        predicted = estimator.fit(training_samples, targets).predict(query)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted.ndim).to eq(2)
      end

      it 'averages only the zero-distance neighbors for the first target' do
        predicted = estimator.fit(training_samples, targets).predict(query)
        expect(predicted[0, 0]).to be_within(1e-10).of(3.0)
      end

      it 'averages only the zero-distance neighbors for the second target' do
        predicted = estimator.fit(training_samples, targets).predict(query)
        expect(predicted[0, 1]).to be_within(1e-10).of(6.0)
      end

      it 'keeps the 4:2:1 inverse-distance ratio of positive distances for the first target' do
        # The prototypes lie at distances 1, 2, and 4 from the query, so their weights have ratio 4:2:1.
        collinear_samples = Numo::DFloat[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
        predicted = estimator.fit(collinear_samples, targets).predict(query)
        expect(predicted[0, 0]).to be_within(1e-10).of(116.0 / 7)
      end

      it 'keeps the 4:2:1 inverse-distance ratio of positive distances for the second target' do
        collinear_samples = Numo::DFloat[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]
        predicted = estimator.fit(collinear_samples, targets).predict(query)
        expect(predicted[0, 1]).to be_within(1e-10).of(132.0 / 7)
      end
    end

    context 'when metric is "precomputed"' do
      let(:estimator) { described_class.new(n_neighbors: 3, metric: 'precomputed', weights: 'distance') }

      before { estimator.fit(Numo::DFloat.eye(3), targets) }

      it 'predicts the zero-distance query for the first target by averaging the zero-distance neighbors' do
        expect(estimator.predict(zero_distances)[0, 0]).to be_within(1e-10).of(3.0)
      end

      it 'predicts the zero-distance query for the second target by averaging the zero-distance neighbors' do
        expect(estimator.predict(zero_distances)[0, 1]).to be_within(1e-10).of(6.0)
      end

      it 'gives the same result as the euclidean path for the zero-distance query' do
        euclidean_estimator = described_class.new(n_neighbors: 3, weights: 'distance').fit(training_samples, targets)
        expect(estimator.predict(zero_distances).to_a).to eq(euclidean_estimator.predict(query).to_a)
      end

      it 'keeps the finite prediction for tiny positive distances' do
        expect(estimator.predict(tiny_distances).to_a.flatten).to all(be_finite)
      end

      it 'keeps the 4:2:1 weight ratio for tiny positive distances of the first target' do
        expect(estimator.predict(tiny_distances)[0, 0]).to be_within(1e-10).of(116.0 / 7)
      end

      it 'keeps the 4:2:1 weight ratio for tiny positive distances of the second target' do
        expect(estimator.predict(tiny_distances)[0, 1]).to be_within(1e-10).of(132.0 / 7)
      end

      it 'keeps finite predictions for a batch containing tiny positive and zero distances' do
        batch = Numo::DFloat.vstack([tiny_distances, zero_distances])
        expect(estimator.predict(batch).to_a.flatten).to all(be_finite)
      end

      it 'keeps the zero-distance behavior for the second row of a mixed batch' do
        batch = Numo::DFloat.vstack([tiny_distances, zero_distances])
        expect(estimator.predict(batch)[1, 0]).to be_within(1e-10).of(3.0)
      end

      it 'supports a single target', :aggregate_failures do
        single_targets = Numo::DFloat[2.0, 4.0, 100.0]
        single_estimator = described_class.new(n_neighbors: 3, metric: 'precomputed', weights: 'distance')
                                          .fit(Numo::DFloat.eye(3), single_targets)
        predicted = single_estimator.predict(zero_distances)
        expect(predicted.ndim).to eq(1)
        expect(predicted[0]).to be_within(1e-10).of(3.0)
      end

      it 'weights a single target with positive distances by the inverse distance' do
        single_targets = Numo::DFloat[2.0, 4.0, 100.0]
        single_estimator = described_class.new(n_neighbors: 3, metric: 'precomputed', weights: 'distance')
                                          .fit(Numo::DFloat.eye(3), single_targets)
        expect(single_estimator.predict(tiny_distances)[0]).to be_within(1e-10).of(116.0 / 7)
      end
    end

    context 'when n_neighbors exceeds the number of training samples' do
      let(:estimator) { described_class.new(n_neighbors: 100, weights: 'distance') }

      it 'clips the number of neighbors along the existing boundary for the first target' do
        predicted = estimator.fit(training_samples, targets).predict(query)
        expect(predicted[0, 0]).to be_within(1e-10).of(3.0)
      end

      it 'clips the number of neighbors along the existing boundary for the second target' do
        predicted = estimator.fit(training_samples, targets).predict(query)
        expect(predicted[0, 1]).to be_within(1e-10).of(6.0)
      end
    end
  end

  context 'when weights is "uniform"' do
    let(:training_samples) { Numo::DFloat[[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]] }
    let(:targets) { Numo::DFloat[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]] }
    let(:query) { Numo::DFloat[[0.0, 0.0]] }

    it 'predicts the plain mean of the neighbor target values for the first target' do
      predicted = described_class.new(n_neighbors: 3).fit(training_samples, targets).predict(query)
      expect(predicted[0, 0]).to be_within(1e-10).of(106.0 / 3)
    end

    it 'predicts the plain mean of the neighbor target values for the second target' do
      predicted = described_class.new(n_neighbors: 3).fit(training_samples, targets).predict(query)
      expect(predicted[0, 1]).to be_within(1e-10).of(112.0 / 3)
    end

    it 'raises ArgumentError when an unsupported weight is given' do
      expect { described_class.new(weights: 'foo') }.to raise_error(ArgumentError, /uniform.*distance/)
    end
  end
end
