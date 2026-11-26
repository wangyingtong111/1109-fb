# frozen_string_literal: true

require 'spec_helper'

require 'rumale/kernel_approximation/rbf'
require 'rumale/decomposition/pca'
require 'rumale/decomposition/nmf'

# A configurable transformer used for exercising one-dimensional, zero-column and
# variable-width outputs in the feature union scenarios.
class StubTransformer
  attr_accessor :n_columns, :one_dimensional, :n_rows, :value

  def initialize(n_columns: 1, one_dimensional: false, n_rows: nil, value: 1.0)
    @n_columns = n_columns
    @one_dimensional = one_dimensional
    @n_rows = n_rows
    @value = value
    @fitted = false
  end

  def fit(x, _y = nil)
    @n_rows ||= x.shape[0]
    @fitted = true
    self
  end

  def fitted?
    @fitted
  end

  def transform(_x = nil)
    rows = @n_rows
    if @one_dimensional
      Numo::DFloat.new(rows).fill(@value)
    elsif @n_columns.zero?
      Numo::DFloat.zeros(rows, 0)
    else
      Numo::DFloat.new(rows, @n_columns).fill(@value)
    end
  end
end

RSpec.describe Rumale::Pipeline::FeatureUnion do
  let(:n_samples) { 32 }
  let(:n_features) { 16 }
  let(:x) { Numo::DFloat.new(n_samples, n_features).rand }
  let(:n_rbf_comps) { 64 }
  let(:n_nmf_comps) { 8 }
  let(:n_pca_comps) { 4 }
  let(:sum_n_comps) { n_rbf_comps + n_pca_comps + n_nmf_comps }
  let(:rbf) { Rumale::KernelApproximation::RBF.new(gamma: 0.1, n_components: n_rbf_comps, random_seed: 1) }
  let(:pca) { Rumale::Decomposition::PCA.new(n_components: n_pca_comps, tol: 1.0e-8, random_seed: 1) }
  let(:nmf) { Rumale::Decomposition::NMF.new(n_components: n_nmf_comps, random_seed: 1) }
  let(:fu) { described_class.new(transformers: { rbf: rbf, pca: pca, nmf: nmf }) }
  let(:z) { fu.fit_transform(x) }

  it 'concatenates each transformed data', :aggregate_failures do
    expect(z.class).to eq(Numo::DFloat)
    expect(z.ndim).to eq(2)
    expect(z.shape[0]).to eq(n_samples)
    expect(z.shape[1]).to eq(sum_n_comps)
  end

  context 'with one-dimensional and zero-column branches' do
    let(:dense) { StubTransformer.new(n_columns: 2, value: 7.0) }
    let(:flag) { StubTransformer.new(n_columns: 1, one_dimensional: true, value: 3.0) }
    let(:optional) { StubTransformer.new(n_columns: 0) }
    let(:union) { described_class.new(transformers: { dense: dense, flag: flag, optional: optional }) }
    let(:three_samples) { Numo::DFloat.new(3, 2).rand }

    it 'treats a one-dimensional vector as one column and records half-open column intervals', :aggregate_failures do
      result = union.fit_transform(three_samples)
      expect(result.ndim).to eq(2)
      expect(result.shape).to eq([3, 3])
      expect(union.column_ranges).to eq({ dense: [0, 2], flag: [2, 3], optional: [3, 3] })
      expect(union.column_range(:dense)).to eq([0, 2])
      expect(union.column_range(:flag)).to eq([2, 3])
      expect(union.column_range(:optional)).to eq([3, 3])
      expect(result[true, 0...2].to_a).to eq(Numo::DFloat.new(3, 2).fill(7.0).to_a)
      expect(result[true, 2].to_a).to eq([3.0, 3.0, 3.0])
    end

    it 'reuses the established layout on subsequent transforms and reports contract conflicts by branch', :aggregate_failures do
      union.fit_transform(three_samples)

      # Keep the same number of samples but change the branch widths while preserving total width.
      dense.n_columns = 1
      flag.one_dimensional = false
      flag.n_columns = 2
      expect { union.transform(three_samples) }.to raise_error(ArgumentError, /dense/)

      # The original layout remains queryable after the failed transformation.
      expect(union.column_ranges).to eq({ dense: [0, 2], flag: [2, 3], optional: [3, 3] })

      # A branch growing wider is also a contract conflict.
      dense.n_columns = 2
      flag.one_dimensional = true
      dense.n_columns = 3
      expect { union.transform(three_samples) }.to raise_error(ArgumentError, /dense/)

      # Restore normal shapes and confirm the original column sources and ordering are preserved.
      dense.n_columns = 2
      restored = union.transform(three_samples)
      expect(restored.shape).to eq([3, 3])
      expect(restored[true, 0...2].to_a).to eq(Numo::DFloat.new(3, 2).fill(7.0).to_a)
      expect(restored[true, 2].to_a).to eq([3.0, 3.0, 3.0])
      expect(union.column_ranges).to eq({ dense: [0, 2], flag: [2, 3], optional: [3, 3] })
    end

    it 'identifies the branch whose number of rows does not match', :aggregate_failures do
      union.fit_transform(three_samples)
      dense.n_rows = 2
      expect { union.transform(three_samples) }.to raise_error(ArgumentError, /dense/)
      expect(union.column_ranges).to eq({ dense: [0, 2], flag: [2, 3], optional: [3, 3] })
    end

    it 'builds a fresh layout when fitted again', :aggregate_failures do
      union.fit_transform(three_samples)

      dense.n_columns = 1
      flag.one_dimensional = false
      flag.n_columns = 2
      union.fit(three_samples)
      expect(union.column_ranges).to be_nil

      result = union.transform(three_samples)
      expect(result.shape).to eq([3, 3])
      expect(union.column_ranges).to eq({ dense: [0, 1], flag: [1, 3], optional: [3, 3] })
      expect(union.column_range(:flag)).to eq([1, 3])
    end

    it 'does not expose column intervals before the first successful transformation', :aggregate_failures do
      union.fit(three_samples)
      expect(union.column_ranges).to be_nil
      expect(union.column_range(:dense)).to be_nil
    end
  end

  context 'with branches that always return stable two-dimensional matrices' do
    let(:stable_a) { StubTransformer.new(n_columns: 2, value: 1.0) }
    let(:stable_b) { StubTransformer.new(n_columns: 3, value: 2.0) }
    let(:union) { described_class.new(transformers: { a: stable_a, b: stable_b }) }

    it 'concatenates them in insertion order under a stable layout', :aggregate_failures do
      result = union.fit_transform(x)
      expect(result.ndim).to eq(2)
      expect(result.shape).to eq([n_samples, 5])
      expect(union.column_ranges).to eq({ a: [0, 2], b: [2, 5] })
      expect(result[true, 0...2]).to eq(Numo::DFloat.new(n_samples, 2).fill(1.0))
      expect(result[true, 2...5]).to eq(Numo::DFloat.new(n_samples, 3).fill(2.0))

      again = union.transform(x)
      expect(again.to_a).to eq(result.to_a)
      expect(union.column_ranges).to eq({ a: [0, 2], b: [2, 5] })
    end
  end
end
