require 'spec_helper'

describe PragmaticTokenizer do
  context 'language annotated fragments of one subtitle' do
    let(:tokenizer) { PragmaticTokenizer::Tokenizer.new(language: :en, filter_languages: [:fr]) }

    it 'applies the apostrophe rules of the language declared at each word start (normal case)' do
      fragments = [[:fr, "l'amour "], [:en, "can't"]]
      expect(tokenizer.tokenize(fragments)).to eq(["l'", 'amour', "can't"])
    end

    it 'never treats a fragment boundary as a tokenization boundary (conflict case)' do
      fragments = [[:fr, 'l'], [nil, "'amour "], [:en, "'can't' https:"], [nil, '//a.test']]
      expect(tokenizer.tokenize(fragments)).to eq(["l'", 'amour', "'", "can't", "'", 'https://a.test'])
    end

    it 'keeps an undeclared fragment on the main language while original spaces are preserved' do
      spaced   = tokenizer.tokenize([[:fr, 'l'], [nil, "'amour "], [:en, "'can't' https:"], [nil, '//a.test']])
      unspaced = tokenizer.tokenize([[:fr, "l'amour"], [:en, " 'can't' https://a.test"]])
      expect(spaced).to   eq(["l'", 'amour', "'", "can't", "'", 'https://a.test'])
      expect(unspaced).to eq(["l'", 'amour', "'", "can't", "'", 'https://a.test'])
    end

    it 'accepts string language codes as well as symbols' do
      fragments = [['fr', "l'amour"], ['en', " can't"]]
      expect(tokenizer.tokenize(fragments)).to eq(["l'", 'amour', "can't"])
    end

    it 'reassembles a URL split across fragments into a single token' do
      fragments = [[:en, "'can't' https:"], [nil, '//a.test']]
      expect(tokenizer.tokenize(fragments)).to eq(["'", "can't", "'", 'https://a.test'])
    end

    it 'matches the plain string when every fragment uses the same language' do
      text      = "Hello world, l'amour and can't plus https://a.test now."
      cuts      = [0, 3, 7, 11, 14, 20, 27, 35, text.length]
      fragments = cuts.each_cons(2).map { |start, finish| [:en, text[start...finish]] }
      expect(tokenizer.tokenize(fragments)).to eq(tokenizer.tokenize(text))
    end

    it 'keeps word-start alignment even when punctuation makes preprocessing grow the text' do
      french = PragmaticTokenizer::Tokenizer.new(language: :fr)
      text   = "Hello, world! Visit https://a.test now; don't wait."
      fragments = []
      index = 0
      while index < text.length
        width = [1 + (index % 5), text.length - index].min
        fragments << [:fr, text[index, width]]
        index += width
      end
      result = french.tokenize(fragments)
      expect(result).to eq(french.tokenize(text))
      expect(result & ["n'", 't']).to eq(["n'", 't'])
    end

    it 'applies one language consistently inside a word split across same-language fragments' do
      french = PragmaticTokenizer::Tokenizer.new(language: :fr)
      expect(french.tokenize([[:fr, 'ab don'], [:fr, "'t cd"]]))
        .to eq(french.tokenize("ab don't cd"))
    end

    it 'matches the plain string when no fragment declares a language' do
      text      = "l'amour can't"
      fragments = text.each_char.each_slice(4).map { |chars| [nil, chars.join] }
      expect(tokenizer.tokenize(fragments)).to eq(tokenizer.tokenize(text))
    end

    it 'keeps language alignment across HTML entities that shrink when unescaped' do
      expect(tokenizer.tokenize([[:en, 'area &lt;0.8 cm2']])).to eq(tokenizer.tokenize('area &lt;0.8 cm2'))
      expect(tokenizer.tokenize([[:en, 'x &lt; '], [:fr, "l'amour"]])).to eq(['x', '<', "l'", 'amour'])
    end

    it 'keeps language alignment across the long-input chunk boundary' do
      french = "l'amour " * 1500
      english = "can't " * 1500
      two_fragments = tokenizer.tokenize([[:fr, french], [:en, english]])
      small_fragments = []
      buffer = french.dup
      small_fragments << [:fr, buffer.slice!(0, 500)] until buffer.empty?
      buffer = english.dup
      small_fragments << [:en, buffer.slice!(0, 500)] until buffer.empty?
      expect(tokenizer.tokenize(small_fragments)).to eq(two_fragments)
      expect(two_fragments.count("l'")).to eq(1500)
      expect(two_fragments.count('amour')).to eq(1500)
      expect(two_fragments.count("can't")).to eq(1500)
    end

    it 'returns an empty array for nil and for empty fragment text' do
      expect(tokenizer.tokenize(nil)).to eq([])
      expect(tokenizer.tokenize([[:fr, '']])).to eq([])
    end

    it 'raises for malformed fragments' do
      expect { tokenizer.tokenize([[:en]]) }.to raise_error(RuntimeError)
      expect { tokenizer.tokenize([[:en, 123]]) }.to raise_error(RuntimeError)
    end
  end

  context 'regression: no fragment annotations' do
    let(:tokenizer) { PragmaticTokenizer::Tokenizer.new(language: :en, filter_languages: [:fr]) }

    it 'keeps the main language and filter_languages semantics with the original String data structure' do
      expect(tokenizer.tokenize("l'amour can't")).to eq(["l'amour", "can't"])
    end

    it 'still tokenizes ordinary English sentences unchanged' do
      expect(PragmaticTokenizer::Tokenizer.new.tokenize('Fresh fruit and green tea.'))
        .to eq(['fresh', 'fruit', 'and', 'green', 'tea', '.'])
    end
  end
end
