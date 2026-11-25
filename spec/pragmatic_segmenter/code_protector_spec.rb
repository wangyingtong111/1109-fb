# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PragmaticSegmenter::CodeProtector do
  subject(:protector) { described_class.new }

  describe '#protect / #restore' do
    it 'replaces inline code with a marker and restores it verbatim' do
      text = 'Use `obj.Run?No` here.'
      protected_text = protector.protect(text)

      expect(protected_text).not_to include('`')
      expect(protected_text).not_to include('?')

      expect(protector.restore([protected_text])).to eq([text])
    end

    it 'treats the single backtick inside double backticks as a literal' do
      text = 'Use ``obj.Run?No` tail`` and `self.run`.'
      protected_text = protector.protect(text)

      expect(protector.restore([protected_text])).to eq([text])
    end

    it 'isolates fenced code blocks with carriage returns and restores them verbatim' do
      text = "First.\n```ruby\nx = 1\n```\nSecond."
      protected_text = protector.protect(text)

      expect(protected_text).to include("\r")
      expect(protected_text).not_to include('```')

      restored = protector.restore([protected_text]).first
      expect(restored).to include("```ruby\nx = 1\n```")
      expect(restored).to include('First.')
      expect(restored).to include('Second.')
    end

    it 'leaves an unclosed backtick untouched so it cannot swallow following text' do
      text = 'Unclosed ` never closes. Next step.'

      expect(protector.protect(text)).to eq(text)
    end
  end
end
