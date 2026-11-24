# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::StratifiedGroupKFold do
  # Four groups with (positive, negative) sample counts:
  # group 0 (A) = (4, 0), group 1 (B) = (3, 1), group 2 (C) = (1, 3), group 3 (D) = (0, 4).
  let(:x) { Numo::DFloat.new(16, 2).rand }
  let(:y) { Numo::Int32[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0] }
  let(:groups) { Numo::Int32[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] }
  let(:n_splits) { 2 }
  let(:splitter) { described_class.new(n_splits: n_splits) }
  let(:validation_ids) { splitter.split(x, y, groups) }

  def group_fold_map(ids, groups_)
    groups_.to_a.each_with_index.with_object({}) do |(gid, sid), map|
      ids.each_with_index { |(_, test_ids), fid| map[gid] = fid if test_ids.include?(sid) }
    end
  end

  it 'generates folds that cover both classes with balanced class counts', :aggregate_failures do
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids).to be_a(Array)
    expect(validation_ids.size).to eq(n_splits)
    validation_ids.each do |train_ids, test_ids|
      expect(train_ids.size + test_ids.size).to eq(16)
      test_labels = y.to_a.values_at(*test_ids)
      expect(test_labels.count(1)).to eq(4)
      expect(test_labels.count(0)).to eq(4)
    end
  end

  it 'assigns groups A and D to one fold and groups B and C to the other', :aggregate_failures do
    expect(validation_ids[0][1]).to contain_exactly(0, 1, 2, 3, 12, 13, 14, 15)
    expect(validation_ids[1][1]).to contain_exactly(4, 5, 6, 7, 8, 9, 10, 11)
  end

  it 'does not split samples belonging to the same group into different folds' do
    validation_ids.each do |train_ids, test_ids|
      train_groups = groups.to_a.values_at(*train_ids).uniq
      test_groups = groups.to_a.values_at(*test_ids).uniq
      expect(train_groups & test_groups).to be_empty
    end
  end

  context 'when the order of groups and samples within groups is permuted' do
    let(:permutation) { [15, 12, 14, 13, 2, 0, 3, 1, 9, 11, 8, 10, 6, 4, 7, 5] }
    let(:permuted_x) { Numo::DFloat[*permutation.map { |i| x[i, true].to_a }] }
    let(:permuted_y) { Numo::Int32[*y.to_a.values_at(*permutation)] }
    let(:permuted_groups) { Numo::Int32[*groups.to_a.values_at(*permutation)] }
    let(:permuted_ids) { splitter.split(permuted_x, permuted_y, permuted_groups) }

    it 'keeps the same group-to-fold assignment' do
      expect(group_fold_map(permuted_ids, permuted_groups)).to eq(group_fold_map(validation_ids, groups))
    end

    it 'moves only the sample indices' do
      validation_ids.each_with_index do |(_, test_ids), fid|
        original_groups = groups.to_a.values_at(*test_ids).uniq.sort
        permuted_groups_in_fold = permuted_groups.to_a.values_at(*permuted_ids[fid][1]).uniq.sort
        expect(permuted_groups_in_fold).to eq(original_groups)
      end
    end
  end

  context 'when a class appears in only one group' do
    # Replace group D with a third class that appears only in that group.
    let(:y) { Numo::Int32[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 2, 2, 2, 2] }

    it 'warns that the class cannot be covered by all folds' do
      expect { splitter.split(x, y, groups) }.to output(/cannot be covered by all folds/).to_stderr
    end

    it 'keeps complete folds and intact groups', :aggregate_failures do
      expect(validation_ids.size).to eq(n_splits)
      validation_ids.each do |train_ids, test_ids|
        expect(train_ids.size + test_ids.size).to eq(16)
        expect(train_ids & test_ids).to be_empty
        train_groups = groups.to_a.values_at(*train_ids).uniq
        test_groups = groups.to_a.values_at(*test_ids).uniq
        expect(train_groups & test_groups).to be_empty
      end
    end

    it 'covers the third class in only one fold' do
      n_folds_with_third_class = validation_ids.count do |_, test_ids|
        test_ids.any? { |sid| y[sid] == 2 }
      end
      expect(n_folds_with_third_class).to eq(1)
    end
  end

  context 'when the number of splits is greater than the number of groups' do
    let(:n_splits) { 5 }

    it 'raises ArgumentError' do
      expect { splitter.split(x, y, groups) }.to raise_error(ArgumentError)
    end
  end

  context 'when the number of splits is less than two' do
    let(:n_splits) { 1 }

    it 'raises ArgumentError' do
      expect { splitter.split(x, y, groups) }.to raise_error(ArgumentError)
    end
  end
end
