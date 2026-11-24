# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::TimeSeriesSplit do
  let(:x) { Numo::DFloat.new(8, 2).rand }
  let(:y) { nil }
  let(:n_splits) { 3 }
  let(:splitter) { described_class.new(n_splits: n_splits, max_train_size: max_train_size) }
  let(:validation_ids) { splitter.split(x, y) }

  context 'when max_train_size is not given' do
    let(:max_train_size) { nil }

    it 'splits the dataset by time series style', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(splitter.max_train_size).to be_nil
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0]).to contain_exactly([0, 1], [2, 3])
      expect(validation_ids[1]).to contain_exactly([0, 1, 2, 3], [4, 5])
      expect(validation_ids[2]).to contain_exactly([0, 1, 2, 3, 4, 5], [6, 7])
    end
  end

  context 'when max_train_size is given' do
    let(:max_train_size) { 1 }

    it 'splits the dataset by time series style with the specified training data size', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(splitter.max_train_size).to eq(max_train_size)
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0][0].size).to eq(max_train_size)
      expect(validation_ids[1][0].size).to eq(max_train_size)
      expect(validation_ids[2][0].size).to eq(max_train_size)
    end
  end

  context 'when given a wrong split number' do
    it 'raises ArgumentError', :aggregate_failures do
      expect { described_class.new(n_splits: 0).split(x, y) }.to raise_error(ArgumentError)
      expect { described_class.new(n_splits: x.shape[0]).split(x, y) }.to raise_error(ArgumentError)
    end
  end

  context 'when the number of samples is not divisible by the number of folds plus one' do
    let(:x) { Numo::DFloat.new(11, 2).rand }
    let(:n_splits) { 3 }
    let(:max_train_size) { nil }

    it 'generates the validation blocks only from the rows that actually exist', :aggregate_failures do
      all_test_ids = validation_ids.flat_map { |_train_ids, test_ids| test_ids }
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids.map { |_train_ids, test_ids| test_ids }).to eq([[5, 6], [7, 8], [9, 10]])
      expect(all_test_ids.count(x.shape[0] - 1)).to eq(1)
      all_test_ids.each { |id| expect(id).to be < x.shape[0] }
    end

    context 'when given independent timestamps' do
      let(:time) { Numo::Int32.new(x.shape[0]).seq }

      it 'keeps the same guarantees through the time group entry', :aggregate_failures do
        group_ids = splitter.split(x, y, time)
        all_test_ids = group_ids.flat_map { |_train_ids, test_ids| test_ids }
        expect(group_ids.size).to eq(n_splits)
        expect(all_test_ids.count(x.shape[0] - 1)).to eq(1)
        (all_test_ids + group_ids.flat_map { |train_ids, _test_ids| train_ids }).each do |id|
          expect(id).to be < x.shape[0]
        end
      end
    end
  end

  context 'when time groups are given' do
    # The time group sizes are [1, 2, 1, 3, 1, 2, 1].
    let(:x) { Numo::DFloat.new(11, 2).rand }
    let(:y) { nil }
    let(:time) { Numo::Int32[0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 6] }
    let(:n_splits) { 3 }
    let(:max_train_size) { 3 }
    let(:splitter) { described_class.new(n_splits: n_splits, max_train_size: max_train_size, gap: 1) }
    let(:validation_ids) { splitter.split(x, y, time) }

    it 'splits the dataset in units of time groups with the gap of one time group', :aggregate_failures do
      expect(splitter.gap).to eq(1)
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0][0]).to eq([0])
      expect(validation_ids[1][0]).to eq([1, 2, 3])
      expect(validation_ids[2][0]).to eq([7])
      expect(validation_ids[0][1]).to eq([3, 4, 5, 6])
      expect(validation_ids[1][1]).to eq([7, 8, 9])
      expect(validation_ids[2][1]).to eq([10])
    end

    it 'keeps the samples with the same timestamp in a single group', :aggregate_failures do
      validation_ids.each do |train_ids, test_ids|
        test_groups = time[test_ids].to_a.uniq
        expect(train_ids & test_ids).to be_empty
        expect(time[train_ids].to_a.uniq & test_groups).to be_empty
      end
    end

    it 'accepts the timestamps through the explicit time group method' do
      expect(splitter.split_by_time_group(x, y, time)).to eq(validation_ids)
    end

    context 'when the history is insufficient' do
      let(:n_splits) { 6 }

      it 'raises ArgumentError' do
        expect { splitter.split(x, y, time) }.to raise_error(ArgumentError)
      end
    end
  end
end
