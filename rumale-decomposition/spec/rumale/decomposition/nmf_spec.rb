# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::NMF do
  let(:n_samples) { 200 }
  let(:n_features) { 6 }
  let(:n_components) { 3 }
  let(:decomposer) { described_class.new(n_components: n_components, max_iter: 10_000, tol: 1.0e-4, random_seed: 1) }
  let(:x) do
    rng = Random.new(1)
    a = Rumale::Utils.rand_uniform([n_samples, n_components], rng)
    b = Rumale::Utils.rand_uniform([n_components, n_features], rng)
    a.dot(b)
  end

  it 'decomposes non-negative matrix', :aggregate_failures do
    x_coef = decomposer.fit_transform(x)
    expect(x_coef).to be_a(Numo::DFloat)
    expect(x_coef).to be_contiguous
    expect(x_coef.ndim).to eq(2)
    expect(x_coef.shape[0]).to eq(n_samples)
    expect(x_coef.shape[1]).to eq(n_components)
    expect(decomposer.components).to be_a(Numo::DFloat)
    expect(decomposer.components).to be_contiguous
    expect(decomposer.components.ndim).to eq(2)
    expect(decomposer.components.shape[0]).to eq(n_components)
    expect(decomposer.components.shape[1]).to eq(n_features)
    x_rec = decomposer.inverse_transform(x_coef)
    mse = ((x - x_rec)**2).sum(axis: 1).mean
    expect(mse).to be <= 1.0e-4
  end

  it 'transform non-negative coefficients', :aggregate_failures do
    decomposer.fit(x)
    comp = decomposer.components.dup
    y = x[0...10, true] + Numo::DFloat.new(10, n_features).rand * 0.01
    y_coef = decomposer.transform(y)
    expect(y_coef).to be_a(Numo::DFloat)
    expect(y_coef).to be_contiguous
    expect(y_coef.ndim).to eq(2)
    expect(y_coef.shape[0]).to eq(10)
    expect(y_coef.shape[1]).to eq(n_components)
    expect(decomposer.components).to eq(comp)
    expect(decomposer.components).to be_a(Numo::DFloat)
    expect(decomposer.components).to be_contiguous
    expect(decomposer.components.ndim).to eq(2)
    expect(decomposer.components.shape[0]).to eq(n_components)
    expect(decomposer.components.shape[1]).to eq(n_features)
    y_rec = decomposer.inverse_transform(y_coef)
    mse = ((y - y_rec)**2).sum(axis: 1).mean
    expect(mse).to be <= 1.0e-4
  end

  it 'decomposes with one-component factor', :aggregate_failures do
    liner = described_class.new(n_components: 1, random_seed: 1)
    sub_x = liner.fit_transform(x)
    rec_x = liner.inverse_transform(sub_x)
    expect(rec_x.shape[0]).to eq(x.shape[0])
    expect(rec_x.shape[1]).to eq(x.shape[1])
    mse = Numo::NMath.sqrt(((x - rec_x)**2).sum(axis: 1)).mean
    expect(mse).to be <= 0.5
  end

  context 'with an explicit observation mask' do
    let(:fixed_basis) { Numo::DFloat[[1.0, 2.0]] }
    let(:projected_row) { Numo::DFloat[[5.0, 0.0]] }
    let(:projector) { described_class.new(n_components: 1, max_iter: 10_000, tol: 1.0e-12, random_seed: 7) }

    it 'ignores the unobserved second column and recovers a coefficient close to five', :aggregate_failures do
      mask = Numo::DFloat[[1, 0]]
      coef = projector.fit_transform(projected_row, mask: mask, components: fixed_basis.dup, frozen_rows: [0])
      expect(coef).to be_a(Numo::DFloat)
      expect(coef.shape).to eq([1, 1])
      expect(coef[0, 0]).to be_within(1.0e-4).of(5.0)
      expect(projector.components).to eq(fixed_basis)
    end

    it 'treats the masked zero as an observed zero and recovers a coefficient close to one' do
      mask = Numo::DFloat[[1, 1]]
      coef = projector.fit_transform(projected_row, mask: mask, components: fixed_basis.dup, frozen_rows: [0])
      expect(coef[0, 0]).to be_within(1.0e-4).of(1.0)
    end

    it 'accepts a Numo::Bit mask and index-array frozen rows' do
      mask = Numo::Bit[[1, 0]]
      coef = projector.fit_transform(projected_row,
                                     mask: mask,
                                     components: fixed_basis.dup,
                                     frozen_rows: Numo::Int32[0])
      expect(coef[0, 0]).to be_within(1.0e-4).of(5.0)
    end

    it 'projects masked rows with transform while keeping the learned basis fixed', :aggregate_failures do
      projector.fit(Numo::DFloat[[1.0, 2.0], [2.0, 4.0]])
      basis = projector.components.dup
      mask = Numo::DFloat[[1, 0]]
      coef = projector.transform(projected_row, mask: mask)
      expected = projected_row.dot(basis.transpose)[0, 0] / mask.dot(basis.transpose**2)[0, 0]
      expect(coef[0, 0]).to be_within(1.0e-3).of(expected)
      expect(projector.components).to eq(basis)
    end

    it 'fits only the observed entries and reconstructs them accurately', :aggregate_failures do
      rng = Random.new(3)
      data = Rumale::Utils.rand_uniform([120, 3], rng).dot(Rumale::Utils.rand_uniform([3, 5], rng))
      mask = Numo::DFloat.new(120, 5).rand(2).floor
      model = described_class.new(n_components: 3, max_iter: 5000, tol: 1.0e-8, random_seed: 2)
      coef = model.fit_transform(data, mask: mask)
      observed_mse = (((data - coef.dot(model.components)) * mask)**2).sum / mask.sum
      expect(observed_mse).to be <= 1.0e-3
      expect(model.components.lt(0).count_true).to be_zero
      expect(coef.lt(0).count_true).to be_zero
    end
  end

  context 'with frozen basis rows' do
    let(:basis) { Numo::DFloat[[1.0, 2.0], [0.5, 0.5]] }
    let(:data) { Numo::DFloat[[3.0, 5.0], [6.0, 10.0]] }
    let(:model) { described_class.new(n_components: 2, max_iter: 1000, tol: 1.0e-10, random_seed: 11) }

    it 'reuses the given components in place without copying' do
      model.fit(data, components: basis, frozen_rows: [0])
      expect(model.components).to equal(basis)
    end

    it 'keeps the frozen row fixed and updates the remaining basis rows', :aggregate_failures do
      model.fit_transform(data, components: basis, frozen_rows: [0])
      expect(model.components[0, true].to_a).to eq([1.0, 2.0])
      expect(model.components[1, true].to_a).not_to eq([0.5, 0.5])
    end

    it 'supports freezing every basis row for pure projection', :aggregate_failures do
      projector = described_class.new(n_components: 1, max_iter: 1000, tol: 1.0e-12, random_seed: 4)
      coef = projector.fit_transform(Numo::DFloat[[5.0, 0.0]],
                                     mask: Numo::DFloat[[1, 0]],
                                     components: Numo::DFloat[[1.0, 2.0]],
                                     frozen_rows: [0])
      expect(projector.components.to_a).to eq([[1.0, 2.0]])
      expect(coef[0, 0]).to be_within(1.0e-4).of(5.0)
    end
  end

  context 'with rows that have no observed entries' do
    let(:data) { Numo::DFloat[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 0.0, 2.0]] }
    let(:mask) { Numo::DFloat[[1, 1, 1], [0, 0, 0], [1, 0, 1]] }

    it 'assigns an all-zero coefficient vector to the fully missing row on fit', :aggregate_failures do
      model = described_class.new(n_components: 2, max_iter: 300, random_seed: 5)
      coef = model.fit_transform(data, mask: mask)
      expect(coef[1, true].eq(0).count_true).to eq(2)
      expect(coef.lt(0).count_true).to be_zero
    end

    it 'assigns an all-zero coefficient vector to a fully missing row on transform' do
      model = described_class.new(n_components: 2, max_iter: 300, random_seed: 5)
      model.fit(Numo::DFloat.new(6, 3).rand + 0.1)
      coef = model.transform(Numo::DFloat[[0.0, 0.0, 0.0]], mask: Numo::DFloat[[0, 0, 0]])
      expect(coef.eq(0).count_true).to eq(2)
    end
  end

  context 'when validating arguments' do
    let(:model) { described_class.new(n_components: 2) }

    it 'raises an error when the samples contain negative values' do
      expect { model.fit(Numo::DFloat[[-1.0, 2.0]]) }.to raise_error(ArgumentError)
    end

    it 'raises an error when the mask has a different shape from the samples' do
      expect { model.fit(Numo::DFloat[[1.0, 2.0]], mask: Numo::DFloat[[1]]) }.to raise_error(ArgumentError)
    end

    it 'raises an error when the mask has non-binary values' do
      expect { model.fit(Numo::DFloat[[1.0, 2.0]], mask: Numo::DFloat[[1, 2]]) }.to raise_error(ArgumentError)
    end

    it 'raises an error when fitting with a fully unobserved mask' do
      expect { model.fit(Numo::DFloat[[1.0, 2.0]], mask: Numo::DFloat[[0, 0]]) }.to raise_error(ArgumentError)
    end

    it 'raises an error when a frozen row index is out of range' do
      comps = Numo::DFloat[[1.0, 2.0], [3.0, 4.0]]
      expect { model.fit(Numo::DFloat[[1.0, 2.0]], components: comps, frozen_rows: [5]) }.to raise_error(ArgumentError)
    end

    it 'raises an error when the given components contain negative values' do
      comps = Numo::DFloat[[-1.0, 2.0], [3.0, 4.0]]
      expect { model.fit(Numo::DFloat[[1.0, 2.0]], components: comps) }.to raise_error(ArgumentError)
    end

    it 'raises an error when the given components have a wrong shape' do
      comps = Numo::DFloat[[1.0, 2.0]]
      expect { model.fit(Numo::DFloat[[1.0, 2.0]], components: comps) }.to raise_error(ArgumentError)
    end

    it 'raises an error when transform is called before fitting' do
      expect { model.transform(Numo::DFloat[[1.0, 2.0]]) }.to raise_error(ArgumentError)
    end
  end
end
