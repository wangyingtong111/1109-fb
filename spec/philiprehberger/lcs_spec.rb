# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Philiprehberger::FuzzyMatch::Lcs do
  describe '.length' do
    it 'returns 0 for empty strings' do
      expect(described_class.length('', '')).to eq(0)
    end

    it 'returns 0 for empty vs non-empty' do
      expect(described_class.length('abc', '')).to eq(0)
      expect(described_class.length('', 'abc')).to eq(0)
    end

    it 'returns full length for identical strings' do
      expect(described_class.length('abc', 'abc')).to eq(3)
    end

    it 'returns known LCS length for kitten/sitting' do
      # LCS of kitten/sitting is "ittn" = 4
      expect(described_class.length('kitten', 'sitting')).to eq(4)
    end

    it 'returns known LCS length for ABCBDAB/BDCAB' do
      # Classic example: LCS is "BCAB" = 4
      expect(described_class.length('ABCBDAB', 'BDCAB')).to eq(4)
    end

    it 'is case insensitive' do
      expect(described_class.length('ABC', 'abc')).to eq(3)
    end

    it 'handles single character strings' do
      expect(described_class.length('a', 'a')).to eq(1)
      expect(described_class.length('a', 'b')).to eq(0)
    end
  end

  describe '.ratio' do
    it 'returns 1.0 for identical strings' do
      expect(described_class.ratio('hello', 'hello')).to eq(1.0)
    end

    it 'returns 1.0 for empty/empty' do
      expect(described_class.ratio('', '')).to eq(1.0)
    end

    it 'returns 0.0 for completely different strings' do
      expect(described_class.ratio('abc', 'xyz')).to eq(0.0)
    end

    it 'returns a value between 0.0 and 1.0' do
      ratio = described_class.ratio('kitten', 'sitting')
      expect(ratio).to be_between(0.0, 1.0)
    end

    it 'is case insensitive' do
      expect(described_class.ratio('Hello', 'hello')).to eq(1.0)
    end
  end

  describe '.alignment' do
    def position_pairs(str_a, str_b)
      described_class.alignment(str_a, str_b).map { |pair| [pair[:a_range], pair[:b_range]] }
    end

    it 'returns four ordered position pairs for Anne/Anene' do
      pairs = position_pairs('Anne', 'Anene')

      expect(pairs).to eq([[0..0, 0..0], [1..1, 1..1], [2..2, 3..3], [3..3, 4..4]])
    end

    it 'reports the matched text for every pair' do
      alignment = described_class.alignment('Anne', 'Anene')

      expect(alignment.map { |pair| pair[:a_text] }).to eq(%w[A n n e])
      expect(alignment.map { |pair| pair[:b_text] }).to eq(%w[A n n e])
    end

    it 'matches a decomposed cluster against its precomposed form' do
      pairs = position_pairs('Åaa', 'Åaaa')

      expect(pairs).to eq([[0..1, 0..0], [2..2, 1..1], [3..3, 2..2]])
    end

    it 'keeps repeated characters after the paired glyph on both sides' do
      alignment = described_class.alignment('Åaa', 'Åaaa')

      expect(alignment.map { |pair| [pair[:a_text], pair[:b_text]] })
        .to eq([%w[Å Å], %w[a a], %w[a a]])
    end

    it 'reports a range that slices whole glyphs out of the original strings' do
      str_a = 'Åaa'
      str_b = 'Åaaa'

      described_class.alignment(str_a, str_b).each do |pair|
        expect(str_a[pair[:a_range]]).to eq(pair[:a_text])
        expect(str_b[pair[:b_range]]).to eq(pair[:b_text])
      end
      expect(str_a[described_class.alignment(str_a, str_b).first[:a_range]]).to eq('Å')
    end

    it 'does not let case folding leak into the reported ranges' do
      pairs = position_pairs('ABC', 'abc')

      expect(pairs).to eq([[0..0, 0..0], [1..1, 1..1], [2..2, 2..2]])
      expect(described_class.alignment('ABC', 'abc').map { |pair| pair[:a_text] }).to eq(%w[A B C])
    end

    it 'returns the same length as .length' do
      expect(described_class.alignment('kitten', 'sitting').length).to eq(described_class.length('kitten', 'sitting'))
      expect(described_class.alignment('Åaa', 'Åaaa').length).to eq(described_class.length('Åaa', 'Åaaa'))
    end

    it 'keeps both sides strictly increasing' do
      alignment = described_class.alignment('mississippi', 'misisipi')

      a_ranges = alignment.map { |pair| pair[:a_range] }
      b_ranges = alignment.map { |pair| pair[:b_range] }

      expect(a_ranges).to eq(a_ranges.sort_by(&:first))
      expect(b_ranges).to eq(b_ranges.sort_by(&:first))
      expect(a_ranges.map(&:first).uniq.length).to eq(a_ranges.length)
      expect(b_ranges.map(&:first).uniq.length).to eq(b_ranges.length)
      expect(a_ranges.each_cons(2).all? { |left, right| left.last < right.first }).to be(true)
      expect(b_ranges.each_cons(2).all? { |left, right| left.last < right.first }).to be(true)
    end

    it 'is case-insensitive when comparing' do
      expect(position_pairs('Anne', 'ANENE')).to eq(position_pairs('Anne', 'Anene'))
    end

    it 'returns an empty array when either side has no comparable text' do
      expect(described_class.alignment('', '')).to eq([])
      expect(described_class.alignment('abc', '')).to eq([])
      expect(described_class.alignment('', 'abc')).to eq([])
      expect(described_class.alignment('abc', 'xyz')).to eq([])
    end

    it 'picks the same positions for repeated symbols on every run' do
      runs = Array.new(20) { position_pairs('aa', 'aaa') }

      expect(runs.uniq).to eq([[[0..0, 0..0], [1..1, 1..1]]])
    end

    it 'picks the same positions for equal-length strings with several valid paths' do
      runs = Array.new(20) { position_pairs('aba', 'aab') }

      expect(runs.uniq.length).to eq(1)
      expect(runs.first.length).to eq(described_class.length('aba', 'aab'))
    end

    it 'mirrors the pairs when the arguments are swapped' do
      forward = position_pairs('Anene', 'Anne')
      backward = position_pairs('Anne', 'Anene').map { |a_range, b_range| [b_range, a_range] }

      expect(forward).to eq(backward)
    end

    it 'does not depend on a stored score matrix' do
      units_a = %w[a b a]
      units_b = %w[b a b a a]

      left = described_class.send(:prefix_scores, units_a, 0, 1, units_b, 0, units_b.length)
      right = described_class.send(:suffix_scores, units_a, 1, 3, units_b, 0, units_b.length)

      # Both are single flat rows of O(width) integers, not a width * height grid.
      expect(left).to all(be_a(Integer))
      expect(right).to all(be_a(Integer))
      expect(left.length).to eq(units_b.length + 1)
      expect(right.length).to eq(units_b.length + 1)
      expect(left).to eq([0, 0, 1, 1, 1, 1])
      expect(right).to eq([2, 2, 2, 1, 1, 0])
    end

    it 'aligns long strings without building a quadratic table' do
      str_a = 'ab' * 1000
      str_b = 'ba' * 1000

      alignment = described_class.alignment(str_a, str_b)

      expect(alignment.length).to eq(described_class.length(str_a, str_b))
      expect(alignment.length).to be > 1_500
    end

    it 'honours every invariant over all short inputs' do
      strings = ['']
      (1..5).each { |len| %w[a b].repeated_permutation(len) { |perm| strings << perm.join } }

      strings.each do |str_a|
        strings.each do |str_b|
          alignment = described_class.alignment(str_a, str_b)

          expect(alignment.length).to eq(described_class.length(str_a, str_b))
          alignment.each_cons(2) do |left, right|
            expect(left[:a_range].last).to be < right[:a_range].first
            expect(left[:b_range].last).to be < right[:b_range].first
          end
          alignment.each do |pair|
            expect(str_a[pair[:a_range]]).to eq(pair[:a_text])
            expect(str_b[pair[:b_range]]).to eq(pair[:b_text])
            expect(pair[:a_text].downcase(:fold)).to eq(pair[:b_text].downcase(:fold))
          end
          mirrored = described_class.alignment(str_b, str_a).map { |pair| [pair[:b_range], pair[:a_range]] }
          expect(mirrored).to eq(alignment.map { |pair| [pair[:a_range], pair[:b_range]] })
        end
      end
    end
  end
end
