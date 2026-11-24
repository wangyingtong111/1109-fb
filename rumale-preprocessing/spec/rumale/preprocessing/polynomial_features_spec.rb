# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::PolynomialFeatures do
  let(:x) { Numo::DFloat[[0, 1], [2, 3], [4, 5]] }
  let(:transformer) { described_class.new(degree: degree) }
  let(:z) { transformer.fit_transform(x) }

  context 'when degree is 0' do
    let(:degree) { 0 }

    it 'raises ArgumentError' do
      expect { transformer }.to raise_error(ArgumentError)
    end
  end

  context 'when degree is 1' do
    let(:degree) { 1 }

    it 'obtains polynomial expanded features', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[
        [1, 0, 1],
        [1, 2, 3],
        [1, 4, 5]
      ])
      expect(transformer.n_output_features).to eq(3)
    end
  end

  context 'when degree is 2' do
    let(:degree) { 2 }

    it 'obtains polynomial expanded features', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[
        [1, 0, 1, 0, 0, 1],
        [1, 2, 3, 4, 6, 9],
        [1, 4, 5, 16, 20, 25]
      ])
      expect(transformer.n_output_features).to eq(6)
    end
  end

  context 'when degree is 3' do
    let(:degree) { 3 }

    it 'obtains polynomial expanded features', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 2, 3, 4, 6, 9, 8, 12, 18, 27],
        [1, 4, 5, 16, 20, 25, 64, 80, 100, 125]
      ])
      expect(transformer.n_output_features).to eq(10)
    end
  end

  context 'when groups are given' do
    let(:x) { Numo::DFloat[[2, 3, 5, 7]] }
    let(:degree) { 3 }
    let(:transformer) { described_class.new(degree: degree, groups: [[0, 1], [2], [3]]) }

    it 'generates only interactions across different groups with no repeated variable', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[[1, 2, 3, 5, 7, 10, 14, 15, 21, 35, 70, 105]])
      expect(transformer.n_output_features).to eq(12)
    end

    it 'keeps the number and meaning of columns when a column becomes zero after fitting', :aggregate_failures do
      transformer.fit(x)
      z_zero = transformer.transform(Numo::DFloat[[0, 3, 5, 7]])
      expect(z_zero).to eq(Numo::DFloat[[1, 0, 3, 5, 7, 0, 0, 15, 21, 35, 0, 105]])
      expect(z_zero.shape[1]).to eq(transformer.n_output_features)
    end

    context 'with a large group and singleton groups' do
      let(:x) { Numo::DFloat[Array.new(42) { |n| n + 1 }] }
      let(:transformer) { described_class.new(degree: degree, groups: [(0...40).to_a, [40], [41]]) }

      it 'scales with the number of legal terms', :aggregate_failures do
        transformer.fit(x)
        # 1 (bias) + 42 (linear) + (40 + 40 + 1) (degree 2) + 40 (degree 3)
        expect(transformer.n_output_features).to eq(164)
        z_large = transformer.transform(x)
        expect(z_large.shape).to eq([1, 164])
        # degree-2 terms (lexicographic): [0,40], [0,41], [1,40], [1,41], ..., [39,40], [39,41], [40,41]
        expect(z_large[0, 43..47].to_a).to eq([41, 42, 82, 84, 123])
        expect(z_large[0, 121..123].to_a).to eq([1640, 1680, 1722])
        # degree-3 terms: 1*41*42, 2*41*42, ..., 40*41*42
        expect(z_large[0, 124..126].to_a).to eq([1722, 3444, 5166])
        expect(z_large[0, 163]).to eq(40 * 41 * 42)
      end
    end

    context 'when a column is not covered by any group' do
      let(:transformer) { described_class.new(degree: 2, groups: [[0, 1]]) }

      it 'treats it as a singleton group', :aggregate_failures do
        expect(z).to eq(Numo::DFloat[[1, 2, 3, 5, 7, 10, 14, 15, 21, 35]])
        expect(transformer.n_output_features).to eq(10)
      end
    end

    context 'when groups contain invalid column indices' do
      it 'raises ArgumentError for out-of-range index' do
        expect { described_class.new(degree: 2, groups: [[0, 4]]).fit(x) }.to raise_error(ArgumentError)
      end

      it 'raises ArgumentError for duplicated index' do
        expect { described_class.new(degree: 2, groups: [[0, 1], [1, 2]]).fit(x) }.to raise_error(ArgumentError)
      end

      it 'raises ArgumentError for malformed groups parameter' do
        expect { described_class.new(degree: 2, groups: 'a') }.to raise_error(ArgumentError)
      end
    end
  end
end
