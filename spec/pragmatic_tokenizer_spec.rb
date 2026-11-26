require 'spec_helper'

describe PragmaticTokenizer do
  it 'has a version number' do
    expect(PragmaticTokenizer::VERSION).not_to be nil
  end

  describe '#initialize' do
    it 'raises an error if the text argument is nil' do
      -> { expect(PragmaticTokenizer::Tokenizer.new(nil, language: 'en').tokenize).to raise_error }
    end

    it 'raises an error if the text argument is empty' do
      -> { expect(PragmaticTokenizer::Tokenizer.new('', language: 'en').tokenize).to raise_error }
    end

    it 'raises an error if minimum_length is not an Integer' do
      -> { expect(PragmaticTokenizer::Tokenizer.new("heelo", minimum_length: "strawberry").tokenize).to raise_error }
    end

    it 'raises an error if long_word_split is not an Integer' do
      -> { expect(PragmaticTokenizer::Tokenizer.new("heeloo", long_word_split: "yes!").tokenize).to raise_error }
    end

    it 'raises an error if the text is not a String' do
      -> { expect(PragmaticTokenizer::Tokenizer.new(5).tokenize).to raise_error }
    end

    it "raises an error if the punctuation argument is not nil, 'all', 'semi', or 'none'" do
      -> { expect(PragmaticTokenizer::Tokenizer.new('', language: 'en', punctuation: 'world').tokenize).to raise_error }
    end

    it "raises an error if the numbers argument is not nil, 'all', 'semi', or 'none'" do
      -> { expect(PragmaticTokenizer::Tokenizer.new('', language: 'en', numbers: 'world').tokenize).to raise_error }
    end

    it "raises an error if the mentions argument is not nil, 'all', 'semi', or 'none'" do
      -> { expect(PragmaticTokenizer::Tokenizer.new('', language: 'en', mentions: 'world').tokenize).to raise_error }
    end
  end

  describe '#tokenize (long input scanning)' do
    let(:tokenizer) { PragmaticTokenizer::Tokenizer.new }

    it 'keeps a single character input as a token' do
      expect(tokenizer.tokenize('b')).to eq(['b'])
    end

    it 'keeps both words of a short input, including the final single character' do
      expect(tokenizer.tokenize('one b')).to eq(['one', 'b'])
    end

    it 'keeps an unbreakable word longer than the chunk budget whole without dropping characters' do
      tokens = tokenizer.tokenize('a' * 12_000)
      expect(tokens).to eq(['a' * 12_000])
    end

    it 'scans long input in order: long word, a single word appearing once and a URL tokenized as if standalone' do
      url    = 'https://a.test/a?x=1&amp;y=2'
      tokens = tokenizer.tokenize(('a' * 12_000) + ' b ' + url)

      expect(tokens.first).to eq('a' * 12_000)
      expect(tokens.count('b')).to eq(1)
      expect(tokens.last(tokenizer.tokenize(url).length)).to eq(tokenizer.tokenize(url))
      expect(tokens).to eq(['a' * 12_000, 'b', 'https://a.test/a?x=1', '&', 'y=2'])
    end

    it 'does not drop any non-whitespace characters when scanning many chunks' do
      text = ('word ' * 3_000).strip
      expect(tokenizer.tokenize(text).join).to eq(text.gsub(/\s+/, ''))
    end

    it 'preserves the first and last single characters across chunk boundaries' do
      text = 'z ' + ('m' * 15_000) + ' q'
      tokens = tokenizer.tokenize(text)
      expect(tokens.first).to eq('z')
      expect(tokens.last).to eq('q')
      expect(tokens).to eq(['z', 'm' * 15_000, 'q'])
    end

    it 'applies the same HTML unescaping, punctuation and URL rules to regular short text' do
      expect(tokenizer.tokenize('read &amp; https://a.test'))
        .to eq(['read', '&', 'https://a.test'])
    end
  end
end
