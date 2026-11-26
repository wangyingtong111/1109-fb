# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Philiprehberger::FuzzyMatch::DamerauLevenshtein do
  describe '.distance' do
    it 'returns 0 for identical strings' do
      expect(described_class.distance('hello', 'hello')).to eq(0)
    end

    it 'returns 0 for empty/empty' do
      expect(described_class.distance('', '')).to eq(0)
    end

    it 'returns length for empty vs non-empty' do
      expect(described_class.distance('abc', '')).to eq(3)
      expect(described_class.distance('', 'abc')).to eq(3)
    end

    it 'counts transposition as 1 edit' do
      expect(described_class.distance('teh', 'the')).to eq(1)
    end

    it 'counts transposition as less than standard Levenshtein' do
      dl_dist = described_class.distance('teh', 'the')
      lev_dist = Philiprehberger::FuzzyMatch::Levenshtein.distance('teh', 'the')
      expect(dl_dist).to be < lev_dist
    end

    it 'returns known distance for ca/abc' do
      expect(described_class.distance('ca', 'abc')).to eq(3)
    end

    context 'with full Damerau-Levenshtein mode' do
      it 'counts AB/BA as one transposition in both modes' do
        expect(described_class.distance('AB', 'BA')).to eq(1)
        expect(described_class.distance('AB', 'BA', mode: :full)).to eq(1)
        expect(described_class.distance('BA', 'AB', mode: :full)).to eq(1)
      end

      it 'allows insertion after a transposition position' do
        expect(described_class.distance('CA', 'ABC')).to eq(3)
        expect(described_class.distance('CA', 'ABC', mode: :full)).to eq(2)
        expect(described_class.distance('ABC', 'CA', mode: :full)).to eq(2)
      end

      it 'counts repeated inserted B characters independently' do
        expect(described_class.distance('CA', 'ABBC')).to eq(4)
        expect(described_class.distance('CA', 'ABBC', mode: :full)).to eq(3)
        expect(described_class.distance('ABBC', 'CA', mode: :full)).to eq(3)
      end

      it 'covers ABBA/BCA in both directions' do
        expect(described_class.distance('ABBA', 'BCA')).to eq(2)
        expect(described_class.distance('ABBA', 'BCA', mode: :full)).to eq(2)
        expect(described_class.distance('BCA', 'ABBA', mode: :full)).to eq(2)
      end

      it 'accepts the explicit mode through common option forms' do
        expect(described_class.distance('CA', 'ABC', :full)).to eq(2)
        expect(described_class.distance('CA', 'ABC', full: true)).to eq(2)
        expect(described_class.distance('CA', 'ABC', variant: :full)).to eq(2)
        expect(described_class.full_distance('CA', 'ABC')).to eq(2)
      end

      it 'keeps empty-string distances unchanged' do
        expect(described_class.distance('', '', mode: :full)).to eq(0)
        expect(described_class.distance('ABC', '', mode: :full)).to eq(3)
        expect(described_class.distance('', 'ABC', mode: :full)).to eq(3)
      end

      it 'is symmetric across a small A/B/C replay' do
        strings = (0..4).flat_map do |length|
          %w[A B C].repeated_permutation(length).map(&:join)
        end

        strings.each do |a|
          strings.each do |b|
            forward = described_class.distance(a, b, mode: :full)
            backward = described_class.distance(b, a, mode: :full)
            expect(forward).to eq(backward), -> { "expected #{a.inspect}<->#{b.inspect} symmetric" }
            expect(forward).to be <= described_class.distance(a, b)
          end
        end
      end
    end

    it 'handles substitution' do
      expect(described_class.distance('cat', 'car')).to eq(1)
    end

    it 'handles insertion' do
      expect(described_class.distance('cat', 'cats')).to eq(1)
    end

    it 'handles deletion' do
      expect(described_class.distance('cats', 'cat')).to eq(1)
    end

    it 'is case insensitive' do
      expect(described_class.distance('ABC', 'abc')).to eq(0)
    end
  end
end
