# -*- encoding: utf-8 -*-
require 'spec_helper'

describe PragmaticTokenizer::Tokenizer do
  let(:legal_amount) { "−1.234,50" }

  describe 'German numeral mode' do
    it 'keeps sign, grouping dots and decimal comma inside the number token' do
      pt = described_class.new(language: 'de', numbers: :only)
      expect(pt.tokenize("−1.234,50 oak")).to eq(["−1.234,50"])
    end

    it 'leaves the text token to :none' do
      pt = described_class.new(language: 'de', numbers: :none)
      expect(pt.tokenize("−1.234,50 oak")).to eq(%w[oak])
    end

    it 'only classifies the legal amount as a number in :only' do
      text = "−1.234,50 v1.2 12.34,50 ,"
      pt = described_class.new(language: 'de', numbers: :only)
      expect(pt.tokenize(text)).to eq(["−1.234,50"])
    end

    it 'removes only the legal amount in :none, keeping versions, bad grouping and punctuation' do
      text = "−1.234,50 v1.2 12.34,50 ,"
      pt = described_class.new(language: 'de', numbers: :none)
      expect(pt.tokenize(text)).to eq(['v1.2', '12.34,50', ','])
    end

    it 'keeps sentence ending punctuation for :none' do
      text = "−1.234,50 v1.2 12.34,50."
      pt = described_class.new(language: 'de', numbers: :none)
      expect(pt.tokenize(text)).to eq(['v1.2', '12.34,50', '.'])
    end

    it 'makes :only and :none exact complements' do
      text = "−1.234,50 v1.2 12.34,50 and 42 ."
      only = described_class.new(language: 'de', numbers: :only).tokenize(text)
      none = described_class.new(language: 'de', numbers: :none).tokenize(text)
      all  = described_class.new(language: 'de', numbers: :all).tokenize(text)
      expect(only).to eq(["−1.234,50", '42'])
      expect(none).to eq(['v1.2', '12.34,50', 'and', '.'])
      expect(only + none).to match_array(all)
    end

    it 'accepts an explicit german_numbers option' do
      pt = described_class.new(german_numbers: true, numbers: :only)
      expect(pt.tokenize("-1.234,50 v1.2 12.34,50.")).to eq(['-1.234,50'])
    end

    it 'accepts a number_locale option' do
      pt = described_class.new(number_locale: :de, numbers: :none)
      expect(pt.tokenize("-1.234,50 v1.2 12.34,50.")).to eq(['v1.2', '12.34,50', '.'])
    end

    it 'infers the mode from an unambiguous U+2212 signal without an option' do
      only = described_class.new(numbers: :only).tokenize("−1.234,50 oak")
      none = described_class.new(numbers: :none).tokenize("−1.234,50 v1.2 12.34,50 ,")
      expect(only).to eq(["−1.234,50"])
      expect(none).to eq(['v1.2', '12.34,50', ','])
    end

    it 'does not merge the U+2212 sign away from default tokenization' do
      all = described_class.new.tokenize("−1.234,50 oak")
      expect(all.first).to eq("−1.234,50")
    end
  end

  describe 'German numeral pattern' do
    def german?(token)
      !!(token =~ PragmaticTokenizer::Regex::ONLY_GERMAN_NUMBER)
    end

    it 'accepts legal forms' do
      expect(german?("−1.234,50")).to be true
      expect(german?('-1.234,50')).to be true
      expect(german?('1.234,50')).to be true
      expect(german?('1.234.567,89')).to be true
      expect(german?('42')).to be true
    end

    it 'rejects versions and illegal grouping' do
      expect(german?('v1.2')).to be false
      expect(german?('12.34,50')).to be false
      expect(german?('1.2')).to be false
      expect(german?('abc')).to be false
    end
  end

  describe 'legacy numbers behaviour' do
    it 'still keeps ordinary ASCII digits with :only' do
      expect(described_class.new(numbers: :only).tokenize('42 oak')).to eq(['42'])
    end

    it 'still keeps alphanumeric number-like tokens with :only when German mode is off' do
      text = '2pac U2 50cent blink-182 zero7 M83 B-52s 500'
      expected = %w[2pac u2 50cent blink-182 zero7 m83 b-52s 500]
      expect(described_class.new(numbers: :only).tokenize(text)).to eq(expected)
    end

    it 'can explicitly disable German inference' do
      pt = described_class.new(german_numbers: false, numbers: :only)
      expect(pt.tokenize("−1.234,50 v1.2")).to eq(["−1.234,50", 'v1.2'])
    end
  end
end
