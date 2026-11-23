# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::OneHotEncoder do
  let(:encoder) { described_class.new }
  let(:labels) { Numo::Int32[0, 0, 2, 3, 2, 1] }
  let(:codes) { Numo::DFloat[[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]] }

  it 'encodes multi-label vector into one-hot-vectors' do
    expect(encoder.fit_transform(labels)).to eq(codes)
  end

  it 'encodes samples into one-hot-vectors categorically', :aggregate_failures do
    x = Numo::Int32[[0, 0, 10], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
    y = Numo::Int32[[0, 1, 1]]
    encoder.fit(x)
    expect(encoder.n_values).to eq(Numo::Int32[2, 3, 11])
    expect(encoder.active_features).to eq(Numo::Int32[0, 1, 2, 3, 4, 5, 6, 7, 15])
    expect(encoder.feature_indices).to eq(Numo::Int32[0, 2, 5, 16])
    expect(encoder.transform(y)).to eq(Numo::DFloat[[1, 0, 0, 1, 0, 0, 1, 0, 0]])
  end

  it 'raises ArgumentError when given the sample contains negative value', :aggregate_failures do
    x = Numo::Int32[[-1, 0, 1], [0, 1, 2]]
    expect do
      encoder.fit(x)
    end.to raise_error(ArgumentError, 'Expected the input samples only consists of non-negative integer values.')
    expect do
      encoder.transform(x)
    end.to raise_error(ArgumentError,
                       'Expected the input samples only consists of non-negative integer values.')
    expect do
      encoder.fit_transform(x)
    end.to raise_error(ArgumentError,
                       'Expected the input samples only consists of non-negative integer values.')
  end

  context 'when handle_unknown option is invalid' do
    it 'raises ArgumentError' do
      expect do
        described_class.new(handle_unknown: 'invalid')
      end.to raise_error(ArgumentError, "handle_unknown must be nil, 'error' or 'ignore'.")
    end
  end

  context 'with seen-categories dictionary mode' do
    let(:training) { Numo::Int32[[0, 4], [1_000_000, 1], [0, 4]] }
    let(:testing) { Numo::Int32[[5, 4], [1_000_000, 9], [0, 1]] }
    let(:training_codes) { Numo::DFloat[[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]] }
    let(:testing_codes) { Numo::DFloat[[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0]] }

    context 'with handle_unknown: "ignore"' do
      let(:encoder) { described_class.new(handle_unknown: 'ignore') }

      it 'allocates a slot only for values occurring in the training data', :aggregate_failures do
        codes = encoder.fit_transform(training)
        expect(codes).to eq(training_codes)
        expect(codes.shape).to eq([3, 4])
        expect(encoder.categories).to eq([[0, 1_000_000], [1, 4]])
        expect(encoder.n_values).to eq(Numo::Int32[2, 2])
        expect(encoder.feature_indices).to eq(Numo::Int32[0, 2, 4])
        expect(encoder.active_features).to eq(Numo::Int32[0, 1, 2, 3])
      end

      it 'clears only the segment of the feature that has an unseen value' do
        encoder.fit(training)
        expect(encoder.transform(testing)).to eq(testing_codes)
      end

      it 'keeps the total width fixed regardless of unseen values' do
        encoder.fit(training)
        expect(encoder.transform(testing).shape[1]).to eq(4)
      end

      it 'builds the same mapping regardless of the training row order', :aggregate_failures do
        shuffled = Numo::Int32[[1_000_000, 1], [0, 4], [0, 4]]
        another = described_class.new(handle_unknown: 'ignore').fit(shuffled)
        encoder.fit(training)
        expect(another.categories).to eq(encoder.categories)
        expect(another.n_values).to eq(encoder.n_values)
        expect(another.feature_indices).to eq(encoder.feature_indices)
        expect(another.transform(testing)).to eq(encoder.transform(testing))
      end

      it 'keeps the mapping and unknown handling after marshaling and loading', :aggregate_failures do
        encoder.fit(training)
        copied = Marshal.load(Marshal.dump(encoder))
        expect(copied.categories).to eq(encoder.categories)
        expect(copied.params).to eq(encoder.params)
        expect(copied.transform(testing)).to eq(testing_codes)
        expect(copied.n_values).to eq(encoder.n_values)
        expect(copied.feature_indices).to eq(encoder.feature_indices)
        expect(copied.active_features).to eq(encoder.active_features)
      end

      it 'encodes a one-dimensional vector', :aggregate_failures do
        vector = Numo::Int32[5, 5, 9]
        expect(encoder.fit_transform(vector)).to eq(Numo::DFloat[[1, 0], [1, 0], [0, 1]])
        expect(encoder.categories).to eq([[5, 9]])
      end

      it 'rejects negative values on fitting and transformation', :aggregate_failures do
        expect { encoder.fit(Numo::Int32[[-1, 0]]) }.to raise_error(ArgumentError)
        encoder.fit(training)
        expect { encoder.transform(Numo::Int32[[-1, 0]]) }.to raise_error(ArgumentError)
      end
    end

    context 'with handle_unknown: "error"' do
      let(:encoder) { described_class.new(handle_unknown: 'error') }

      it 'encodes seen values in the same way as the ignore strategy' do
        expect(encoder.fit_transform(training)).to eq(training_codes)
      end

      it 'raises ArgumentError that reports the row, column and value of the first unseen value' do
        encoder.fit(training)
        expect { encoder.transform(testing) }.to raise_error(
          ArgumentError, 'Found unseen value 5 at row 0 and column 0.'
        )
      end

      it 'reports the position of the first unseen value while scanning the samples' do
        encoder.fit(training)
        expect do
          encoder.transform(Numo::Int32[[0, 4], [1_000_000, 9]])
        end.to raise_error(ArgumentError, 'Found unseen value 9 at row 1 and column 1.')
      end

      it 'keeps raising after marshaling and loading' do
        encoder.fit(training)
        copied = Marshal.load(Marshal.dump(encoder))
        expect { copied.transform(testing) }.to raise_error(
          ArgumentError, 'Found unseen value 5 at row 0 and column 0.'
        )
      end
    end
  end
end
