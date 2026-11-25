# encoding: utf-8
require 'spec_helper'

describe PragmaticTokenizer do
  context 'canonical stop word matching' do
    it 'removes only the exact stop word while preserving glyph and casing (fr, downcase: false)' do
      pt = PragmaticTokenizer::Tokenizer.new(
          language: :fr,
          downcase: false,
          remove_stop_words: true,
          stop_words: ['café'],
          canonical: true
      )
      expect(pt.tokenize('café cafe Érable')).to eq(['cafe', 'Érable'])
    end

    it 'removes a composed stop word and its decomposed glyph with the canonical key' do
      composed   = 'café'
      decomposed = 'café'.unicode_normalize(:nfd)
      pt = PragmaticTokenizer::Tokenizer.new(
          language: :fr,
          downcase: false,
          remove_stop_words: true,
          stop_words: [composed],
          canonical: true
      )
      expect(pt.tokenize(decomposed)).to eq([])
    end

    it 'applies the Turkish case folding rules for the i family (tr, downcase: false)' do
      pt = PragmaticTokenizer::Tokenizer.new(
          language: :tr,
          downcase: false,
          remove_stop_words: true,
          stop_words: ['i'],
          canonical: true
      )
      expect(pt.tokenize('İ I ı i')).to eq(['I', 'ı'])
    end

    it 'does not carry the Turkish lookup key over to the non-Turkish rules' do
      turkish = PragmaticTokenizer::Tokenizer.new(
          language: :tr,
          downcase: false,
          remove_stop_words: true,
          stop_words: ['i'],
          canonical: true
      )
      plain = PragmaticTokenizer::Tokenizer.new(
          language: :en,
          downcase: false,
          remove_stop_words: true,
          stop_words: ['i'],
          canonical: true
      )
      expect(turkish.tokenize('İ I ı i')).to eq(['I', 'ı'])
      expect(plain.tokenize('İ I ı i')).to eq(['İ', 'ı'])
    end

    it 'keeps literal matching, including composed vs decomposed glyphs, when canonical is off' do
      composed   = 'café'
      decomposed = 'café'.unicode_normalize(:nfd)
      pt = PragmaticTokenizer::Tokenizer.new(
          language: :fr,
          downcase: false,
          remove_stop_words: true,
          stop_words: [composed],
          canonical: false
      )
      expect(pt.tokenize("#{composed} #{decomposed} cafe")).to eq([decomposed, 'cafe'])
    end

    it 'accepts the canonical_comparison option name as an alias' do
      pt = PragmaticTokenizer::Tokenizer.new(
          language: :fr,
          downcase: false,
          remove_stop_words: true,
          stop_words: ['café'],
          canonical_comparison: true
      )
      expect(pt.tokenize('café cafe Érable')).to eq(['cafe', 'Érable'])
    end
  end
end
