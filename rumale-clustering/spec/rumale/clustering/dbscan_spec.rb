# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::DBSCAN do
  let(:three_clusters) { three_clusters_dataset }
  let(:x_mlt) { three_clusters[0] }
  let(:y_mlt) { three_clusters[1] }
  let(:n_samples) { x_mlt.shape[0] }
  let(:analyzer) { described_class.new(eps: 1.0) }
  let(:cluster_labels) { analyzer.fit_predict(x) }
  let(:x_mlt_with_outlier) do
    max_vec = x_mlt.max(0) * 2.0
    min_vec = x_mlt.min(0) * 2.0
    x = Numo::NArray.vstack([x_mlt, max_vec])
    Numo::NArray.vstack([x, min_vec])
  end
  # Samples are placed at 0, 1, and 2 on a line. With eps 1.1 and min_samples 3,
  # every sample is density-reachable from the others, but only the middle sample
  # has three samples in its eps-neighborhood.
  let(:line_samples) { Numo::DFloat[[0.0], [1.0], [2.0]] }
  # Samples 0 to 3 and 5 to 8 form dense groups. Sample 4 is at distance 1.0 from
  # both groups, so with eps 1.01 and min_samples 4 it is a border sample shared by them.
  let(:shared_border_samples) do
    Numo::DFloat[
      [-0.5, 0.5], [-0.5, -0.5], [-0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
      [2.0, 0.0], [2.5, 0.5], [2.5, -0.5], [2.5, 0.0]
    ]
  end

  shared_examples 'cluster analysis' do
    it 'finds clusters', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.size).to eq(n_samples)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(cluster_labels.eq(2).count).to eq(100)
      expect(analyzer.labels).to be_a(Numo::Int32)
      expect(analyzer.labels).to be_contiguous
      expect(analyzer.labels.shape[0]).to eq(n_samples)
      expect(analyzer.labels.shape[1]).to be_nil
      expect(analyzer.core_sample_ids).to be_a(Numo::Int32)
      expect(analyzer.core_sample_ids).to be_contiguous
      expect(analyzer.core_sample_ids.shape[0]).not_to be_nil
      expect(analyzer.core_sample_ids.shape[1]).to be_nil
      expect(analyzer.score(x, y_mlt)).to eq(1)
    end
  end

  shared_examples 'outlier detection' do
    it 'detects outlier points', :aggregate_failures do
      expect(cluster_labels.eq(-1).count).to eq(2)
      expect(analyzer.labels.eq(-1).count).to eq(2)
    end
  end

  shared_examples 'core sample detection' do
    it 'reports only the samples satisfying the core point condition', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels.to_a).to eq([0, 0, 0])
      expect(analyzer.core_sample_ids).to be_a(Numo::Int32)
      expect(analyzer.core_sample_ids.to_a).to eq([1])
    end
  end

  shared_examples 'shared border sample assignment' do
    it 'assigns a border sample shared by two clusters to the first cluster reaching it', :aggregate_failures do
      expect(cluster_labels.to_a).to eq([0, 0, 0, 0, 0, 1, 1, 1, 1])
      expect(analyzer.core_sample_ids.to_a).to eq([0, 1, 2, 3, 5, 6, 7, 8])
      # The shared border sample is not reported as a core sample.
      expect(analyzer.core_sample_ids.to_a).not_to include(4)
      # Border samples are distinguished from noise samples by their cluster label.
      expect(cluster_labels[4]).to eq(0)
    end
  end

  context "when metric is 'euclidean'" do
    context 'when dataset does not contain outlier' do
      let(:x) { x_mlt }

      it_behaves_like 'cluster analysis'
    end

    context 'when dataset contains outlier' do
      let(:x) { x_mlt_with_outlier }

      it_behaves_like 'outlier detection'
    end

    context 'when only some samples in a cluster are core samples' do
      let(:x) { line_samples }
      let(:analyzer) { described_class.new(eps: 1.1, min_samples: 3) }

      it_behaves_like 'core sample detection'
    end

    context 'when a border sample is shared by two clusters' do
      let(:x) { shared_border_samples }
      let(:analyzer) { described_class.new(eps: 1.01, min_samples: 4) }

      it_behaves_like 'shared border sample assignment'
    end
  end

  context "when metric is 'precomputed'" do
    let(:analyzer) { described_class.new(eps: 1.0, metric: 'precomputed') }

    context 'when dataset does not contain outlier' do
      let(:x) { Rumale::PairwiseMetric.manhattan_distance(x_mlt) }

      it_behaves_like 'cluster analysis'
    end

    context 'when dataset contains outlier' do
      let(:x) { Rumale::PairwiseMetric.manhattan_distance(x_mlt_with_outlier) }

      it_behaves_like 'outlier detection'
    end

    context 'when only some samples in a cluster are core samples' do
      let(:x) { Rumale::PairwiseMetric.euclidean_distance(line_samples) }
      let(:analyzer) { described_class.new(eps: 1.1, min_samples: 3, metric: 'precomputed') }

      it_behaves_like 'core sample detection'
    end

    context 'when a border sample is shared by two clusters' do
      let(:x) { Rumale::PairwiseMetric.euclidean_distance(shared_border_samples) }
      let(:analyzer) { described_class.new(eps: 1.01, min_samples: 4, metric: 'precomputed') }

      it_behaves_like 'shared border sample assignment'
    end

    it 'raises ArgumentError when given a non-square matrix', :aggregate_failures do
      expect { analyzer.fit(Numo::DFloat.new(3, 2).rand) }.to raise_error(ArgumentError)
      expect { analyzer.fit_predict(Numo::DFloat.new(2, 3).rand) }.to raise_error(ArgumentError)
    end
  end
end
