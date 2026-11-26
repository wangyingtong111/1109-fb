# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Philiprehberger::FuzzyMatch::JaroWinkler do
  describe '.similarity' do
    it 'returns 1.0 for identical strings' do
      expect(described_class.similarity('abc', 'abc')).to eq(1.0)
    end

    it 'returns 0.0 for empty string vs non-empty' do
      expect(described_class.similarity('', 'abc')).to eq(0.0)
      expect(described_class.similarity('abc', '')).to eq(0.0)
    end

    it 'returns known similarity for martha/marhta' do
      score = described_class.similarity('martha', 'marhta')
      expect(score).to be_within(0.01).of(0.96)
    end

    it 'returns known similarity for dwayne/duane' do
      score = described_class.similarity('dwayne', 'duane')
      expect(score).to be_within(0.02).of(0.84)
    end

    it 'is case insensitive' do
      expect(described_class.similarity('Hello', 'hello')).to eq(1.0)
    end

    it 'returns value between 0 and 1' do
      score = described_class.similarity('abc', 'xyz')
      expect(score).to be_between(0.0, 1.0)
    end

    it 'keeps the classic always-boosted formula when no threshold is given' do
      # base Jaro is exactly 2/3, prefix 4 => 2/3 + 4*0.1*(1/3) == 0.8
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy')).to eq(0.8)
    end
  end

  describe '.similarity with prefix_threshold' do
    it 'applies the prefix bonus when the base score reaches the gate' do
      # base Jaro 7/8 = 0.875 >= 0.7, prefix 4 => 0.95
      expect(described_class.similarity('abcdxxxx', 'abcdxxxy', prefix_threshold: 0.7)).to eq(0.95)
    end

    it 'skips the prefix bonus when the base score is below the gate' do
      # base Jaro is 2/3 < 0.7, so the 0.8-boosted score is withheld
      score = described_class.similarity('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)
      expect(score).to eq(2.0 / 3)
      expect(score).to be < 0.8
    end

    it 'returns the base score unchanged when the gate excludes the bonus' do
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7))
        .to eq(described_class.components('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)[:base_score])
    end

    it 'treats the gate as inclusive at exactly the base score' do
      base = described_class.components('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)[:base_score]
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy', prefix_threshold: base)).to be_within(1e-12).of(0.8)
    end

    it 'supports the jaro_threshold alias' do
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy', jaro_threshold: 0.7)).to eq(2.0 / 3)
    end
  end

  describe '.components' do
    it 'reports the base score, prefix length, bonus flag, and final score' do
      components = described_class.components('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)
      expect(components.keys).to contain_exactly(:base_score, :prefix_length, :bonus_applied, :score)
      expect(components[:base_score]).to eq(2.0 / 3)
      expect(components[:prefix_length]).to eq(4)
      expect(components[:bonus_applied]).to be(false)
      expect(components[:score]).to eq(2.0 / 3)
    end

    it 'measures the common prefix with the original capped rule even when unboosted' do
      # A 6-character shared prefix still reports 4, the historical cap.
      components = described_class.components('abcdefzz', 'abcdefyy', prefix_threshold: 0.99)
      expect(components[:prefix_length]).to eq(described_class::MAX_PREFIX_LENGTH)
      expect(components[:bonus_applied]).to be(false)
    end

    it 'flags the bonus as applied when the gate is met' do
      components = described_class.components('abcdxxxx', 'abcdxxxy', prefix_threshold: 0.7)
      expect(components[:base_score]).to be_within(1e-12).of(11.0 / 12)
      expect(components[:prefix_length]).to eq(4)
      expect(components[:bonus_applied]).to be(true)
      expect(components[:score]).to eq(0.95)
    end

    it 'always applies the bonus without a threshold' do
      components = described_class.components('abcdxxxx', 'abcdyyyy')
      expect(components[:base_score]).to eq(2.0 / 3)
      expect(components[:bonus_applied]).to be(true)
      expect(components[:score]).to eq(0.8)
    end

    it 'is case insensitive' do
      lower = described_class.components('abcdxxxx', 'abcdxxxy', prefix_threshold: 0.7)
      upper = described_class.components('ABCDXXXX', 'ABCDXXXY', prefix_threshold: 0.7)
      expect(upper).to eq(lower)
    end

    it 'agrees with .similarity' do
      %w[abcdxxxx martha kitten].product(%w[abcdyyyy marhta sitting]).each do |a, b|
        [nil, 0.7, 0.9].each do |gate|
          expect(described_class.components(a, b, prefix_threshold: gate)[:score])
            .to eq(described_class.similarity(a, b, prefix_threshold: gate))
        end
      end
    end

    it 'reports 1.0 with no prefix for identical strings' do
      components = described_class.components('abc', 'abc', prefix_threshold: 0.7)
      expect(components[:base_score]).to eq(1.0)
      expect(components[:score]).to eq(1.0)
      expect(components[:prefix_length]).to eq(3)
    end

    it 'reports a zero base score for empty input' do
      components = described_class.components('', 'abc', prefix_threshold: 0.7)
      expect(components[:base_score]).to eq(0.0)
      expect(components[:prefix_length]).to eq(0)
      expect(components[:score]).to eq(0.0)
    end
  end

  describe '.explain' do
    it 'mirrors .components' do
      expect(described_class.explain('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7))
        .to eq(described_class.components('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7))
    end
  end

  describe 'component readers' do
    it 'returns the base score on its own' do
      expect(described_class.base_score('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)).to eq(2.0 / 3)
    end

    it 'returns the original-rule prefix length on its own' do
      expect(described_class.prefix_length('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)).to eq(4)
    end

    it 'returns the bonus flag on its own' do
      expect(described_class.bonus_applied('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)).to be(false)
      expect(described_class.bonus_applied('abcdxxxx', 'abcdxxxy', prefix_threshold: 0.7)).to be(true)
    end

    it 'always reports a bonus when no threshold is configured' do
      expect(described_class.bonus_applied('abcdxxxx', 'abcdyyyy')).to be(true)
    end
  end

  describe 'option forms' do
    let(:expected) { described_class.components('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7) }

    it 'accepts a positional gate' do
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy', 0.7)).to eq(expected[:score])
    end

    it 'accepts a positional options hash' do
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy', { prefix_threshold: 0.7 })).to eq(expected[:score])
    end

    it 'accepts alias keywords' do
      %i[jaro_threshold jaro_base_threshold base_score_threshold min_base_score].each do |key|
        expect(described_class.similarity('abcdxxxx', 'abcdyyyy', key => 0.7)).to eq(expected[:score])
      end
    end

    it 'ignores nil aliases when the canonical keyword is given' do
      score = described_class.similarity('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7, jaro_threshold: nil)
      expect(score).to eq(expected[:score])
    end

    it 'returns the breakdown from .similarity when details are requested' do
      expect(described_class.similarity('abcdxxxx', 'abcdyyyy', 0.7, explain: true)).to eq(expected)
    end
  end
end
