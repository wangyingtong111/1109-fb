# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PragmaticSegmenter::Segmenter, 'code protection mode' do
  describe '#segment with protect_code: true' do
    it 'does not split on punctuation inside inline code' do
      text = 'Use `obj.Run?No`. Next step.'

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(['Use `obj.Run?No`.', 'Next step.'])
    end

    it 'keeps the backticks and symbols of the first sentence intact' do
      text = 'Use `obj.Run?No`. Next step.'
      segments = PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment

      expect(segments.first).to eq('Use `obj.Run?No`.')
      expect(segments.first).to include('`', '?')
    end

    it 'handles double backticks containing a literal single backtick' do
      text = 'Use ``obj.Run?No` tail`` and `self.run`. Next step.'

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(['Use ``obj.Run?No` tail`` and `self.run`.', 'Next step.'])
    end

    it 'returns fenced code blocks verbatim as independent content' do
      text = "First sentence.\n```ruby\nx = 1\nputs x\n```\nSecond sentence."

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(['First sentence.', "\n```ruby\nx = 1\nputs x\n```", 'Second sentence.'])
    end

    it 'supports tilde fenced code blocks' do
      text = "Intro.\n~~~\na.b? c!\n~~~\nEnd."

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(['Intro.', "\n~~~\na.b? c!\n~~~", 'End.'])
    end

    it 'does not let an unclosed backtick swallow the following sentence' do
      text = 'Unclosed backtick ` never closes. Next step.'

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(['Unclosed backtick \' never closes.', 'Next step.'])
    end

    it 'segments plain text exactly like the default entry point' do
      text = 'Hello world. Hello.'

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(PragmaticSegmenter::Segmenter.new(text: text).segment)
    end

    it 'still applies the ordinary rules to natural sentences around code' do
      text = 'Hello world. Call `foo!` now. Then stop.'

      expect(PragmaticSegmenter::Segmenter.new(text: text, protect_code: true).segment)
        .to eq(['Hello world.', 'Call `foo!` now.', 'Then stop.'])
    end

    it 'works together with clean: false' do
      text = "It was a cold \nnight. Next."

      expect(PragmaticSegmenter::Segmenter.new(text: text, clean: false, protect_code: true).segment)
        .to eq(['It was a cold', 'night.', 'Next.'])
    end
  end

  describe '#segment with protect_code disabled (default)' do
    it 'keeps the existing plain text behavior' do
      text = 'Hello world. Hello.'

      expect(PragmaticSegmenter::Segmenter.new(text: text).segment)
        .to eq(['Hello world.', 'Hello.'])
    end
  end
end
