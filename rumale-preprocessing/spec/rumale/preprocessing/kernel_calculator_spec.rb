# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::KernelCalculator do
  describe 'kernel type parameter' do
    let(:gamma) { 1 }
    let(:degree) { 3 }
    let(:coef) { 1 }
    let(:calculator) { described_class.new(kernel: kernel, gamma: gamma, degree: degree, coef: coef) }
    let(:x_train) { Numo::DFloat[[1, 2], [3, 2], [4, 3]] }
    let(:x_test) { Numo::DFloat[[2, 4], [3, 2]] }

    shared_examples 'kernel matrix' do
      let(:kernel_mat_train) { calculator.fit_transform(x_train) }
      let(:kernel_mat_test) { calculator.fit(x_train).transform(x_test) }

      it 'calculates kernel matrix', :aggregate_failures do
        expect(kernel_mat_train).to be_a(Numo::DFloat)
        expect(kernel_mat_train.ndim).to eq(2)
        expect(kernel_mat_train.shape[0]).to eq(3)
        expect(kernel_mat_train.shape[1]).to eq(3)
        expect(kernel_mat_test).to be_a(Numo::DFloat)
        expect(kernel_mat_test.ndim).to eq(2)
        expect(kernel_mat_test.shape[0]).to eq(2)
        expect(kernel_mat_test.shape[1]).to eq(3)
      end
    end

    context "when kernel is 'rbf'" do
      let(:kernel) { 'rbf' }

      it_behaves_like 'kernel matrix'
    end

    context "when kernel is 'linear'" do
      let(:kernel) { 'linear' }

      it_behaves_like 'kernel matrix'
    end

    context "when kernel is 'poly'" do
      let(:kernel) { 'poly' }

      it_behaves_like 'kernel matrix'
    end

    context "when kernel is 'sigmoid'" do
      let(:kernel) { 'sigmoid' }
      let(:gamma) { 0.01 }
      let(:coef) { 0 }

      it_behaves_like 'kernel matrix'
    end

    context 'when wrong kernel type is given' do
      let(:kernel) { 'foo' }

      it 'raises ArgumentError', :aggregate_failures do
        expect { calculator.fit_transform(x_train) }.to raise_error(ArgumentError)
        expect { calculator.fit(x_train).transform(x_test) }.to raise_error(ArgumentError)
      end
    end
  end

  describe 'kernel centering' do
    let(:calculator) { described_class.new(kernel: 'linear', center: true) }
    let(:x_train) { Numo::DFloat[[1], [3]] }
    let(:x_single) { Numo::DFloat[[4]] }
    let(:x_batch) { Numo::DFloat[[4], [1]] }

    it 'centers kernel matrix with the training data as a reference', :aggregate_failures do
      calculator.fit(x_train)
      expect(calculator.transform(x_single)).to eq(Numo::DFloat[[-2, 2]])
      expect(calculator.transform(x_batch)[0, true]).to eq(Numo::DFloat[-2, 2])
    end

    it 'centers kernel matrix on the training data', :aggregate_failures do
      kernel_mat = calculator.fit_transform(x_train)
      expect(kernel_mat).to eq(Numo::DFloat[[1, -1], [-1, 1]])
      expect(kernel_mat.mean(0)).to eq(Numo::DFloat[0, 0])
      expect(kernel_mat.mean(1)).to eq(Numo::DFloat[0, 0])
    end

    it 'stores column means of the kernel matrix on the training data', :aggregate_failures do
      calculator.fit(x_train)
      expect(calculator.col_means).to eq(Numo::DFloat[2, 6])
      expect(calculator.total_mean).to eq(4)
    end

    it 'restores column means with the dumped model', :aggregate_failures do
      calculator.fit(x_train)
      copied = Marshal.load(Marshal.dump(calculator))
      expect(copied.col_means).to eq(Numo::DFloat[2, 6])
      expect(copied.total_mean).to eq(4)
      expect(copied.transform(x_single)).to eq(Numo::DFloat[[-2, 2]])
    end

    context 'when centering is not enabled' do
      let(:calculator) { described_class.new(kernel: 'linear') }

      it 'returns the kernel matrix without centering', :aggregate_failures do
        calculator.fit(x_train)
        expect(calculator.col_means).to be_nil
        expect(calculator.total_mean).to be_nil
        expect(calculator.transform(x_single)).to eq(Numo::DFloat[[4, 12]])
        expect(calculator.fit_transform(x_train)).to eq(Numo::DFloat[[1, 3], [3, 9]])
      end
    end
  end
end
