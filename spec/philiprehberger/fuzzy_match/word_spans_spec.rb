# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Philiprehberger::FuzzyMatch do
  let(:hyphen) { '‐' }
  let(:decomposed_cafe) { 'Café' }
  let(:unicode_store) { "#{decomposed_cafe}#{hyphen}knight" }

  describe '.word_spans / .phonetic_spans' do
    it 'returns exclusive original-text offsets for space-separated words' do
      expected = [[0, 4, 'shop'], [5, 11, 'knight']]

      expect(described_class.word_spans('shop knight', hyphen)).to eq(expected)
      expect(described_class.phonetic_spans('shop knight', hyphen)).to eq(expected)
    end

    it 'keeps combining marks inside the original span and returns NFC text' do
      spans = described_class.word_spans(unicode_store, separators: hyphen)

      expect(spans).to eq([[0, 5, 'café'], [6, 12, 'knight']])
      expect(unicode_store[spans[0][0]...spans[0][1]]).to eq(decomposed_cafe)
      expect(unicode_store[spans[1][0]...spans[1][1]]).to eq('knight')
    end

    it 'uses shorter precomposed offsets without confusing them with decomposed input' do
      nfc_store = unicode_store.unicode_normalize(:nfc)
      spans = described_class.word_spans(nfc_store, separators: hyphen)
      nfc_cafe = decomposed_cafe.unicode_normalize(:nfc).downcase

      expect(spans).to eq([[0, 4, nfc_cafe], [5, 11, 'knight']])
      expect(nfc_store[spans[0][0]...spans[0][1]]).to eq(decomposed_cafe.unicode_normalize(:nfc))
    end

    it 'does not split concatenated words from prefix knowledge' do
      expect(described_class.word_spans('shopknight', '')).to eq([[0, 10, 'shopknight']])
    end

    it 'always treats whitespace as a boundary when connector splitting is off' do
      expected = [[0, 4, 'shop'], [5, 11, 'knight']]

      expect(described_class.word_spans('shop knight')).to eq(expected)
      expect(described_class.word_spans('shop knight', false)).to eq(expected)
      expect(described_class.word_spans('shop knight', separators: '')).to eq(expected)
    end

    it 'keeps an undeclared connector and both sides in one span' do
      expect(described_class.word_spans(unicode_store, separators: '')).to eq(
        [[0, 12, "café#{hyphen}knight"]]
      )
      expect(described_class.word_spans(unicode_store)).to eq(
        [[0, 12, "café#{hyphen}knight"]]
      )
    end

    it 'accepts arrays and common option aliases for declared separators' do
      expected = [[0, 4, 'shop'], [5, 11, 'knight']]

      expect(described_class.word_spans('shop-knight', separators: ['-'])).to eq(expected)
      expect(described_class.word_spans('shop-knight', delimiters: '-')).to eq(expected)
      expect(described_class.word_spans('shop-knight', split_hyphens: true)).to eq(expected)
    end

    it 'returns an empty array for nil and empty input' do
      expect(described_class.word_spans(nil)).to eq([])
      expect(described_class.word_spans('   ')).to eq([])
    end
  end

  describe '.metaphone_words' do
    it 'applies word-initial Metaphone rules independently' do
      result = described_class.metaphone_words('shop knight')

      expect(result.codes).to eq(%w[XP NT])
      expect(result[1][:code]).to eq(described_class.metaphone('knight'))
      expect(result.triples).to eq([[0, 4, 'shop'], [5, 11, 'knight']])
    end

    it 'does not give concatenated shopknight artificial word boundaries' do
      result = described_class.metaphone_words('shopknight')

      expect(result.length).to eq(1)
      expect(result[0][:code]).to eq('XPKNT')
      expect(result[0][:span]).to eq(0...10)
    end

    it 'encodes declared connector parts separately while preserving original offsets' do
      result = described_class.metaphone_words(unicode_store, separators: hyphen)

      expect(result.codes).to eq(%w[KF NT])
      expect(result[0][:start]).to eq(0)
      expect(result[0][:stop]).to eq(5)
      expect(result[0][:end]).to eq(5)
      expect(result[0][:original]).to eq(decomposed_cafe)
      expect(result[0][:word]).to eq('café')
      expect(result[1][:span]).to eq(6...12)
    end

    it 'keeps an undeclared connector in the whole word' do
      result = described_class.metaphone_words(unicode_store)

      expect(result.length).to eq(1)
      expect(result[0][:start]).to eq(0)
      expect(result[0][:stop]).to eq(12)
      expect(result[0][:original]).to eq(unicode_store)
    end

    it 'returns hash and method accessors for each token' do
      token = described_class.metaphone_words('knight').first

      expect(token).to be_a(Hash)
      expect(token[:code]).to eq('NT')
      expect(token.code).to eq('NT')
      expect(token[:span]).to eq(0...6)
      expect(token.range).to eq(0...6)
    end

    it 'has discoverable compatibility aliases' do
      expected = %w[XP NT]

      expect(described_class.metaphone_with_spans('shop knight').codes).to eq(expected)
      expect(described_class.word_metaphone('shop knight').codes).to eq(expected)
      expect(described_class::Metaphone.words('shop knight').codes).to eq(expected)
    end
  end

  describe '.metaphone compatibility' do
    it 'continues to return the ordinary whole-string code' do
      expect(described_class.metaphone('shop knight')).to eq('XPKNT')
      expect(described_class.metaphone('shopknight')).to eq('XPKNT')
    end

    it 'opts into word encoding only when span options are supplied' do
      expect(described_class.metaphone('shop knight', with_spans: true)).to be_a(Array)
      expect(described_class.metaphone('shop knight', separators: hyphen)).to be_a(Array)
    end
  end
end
