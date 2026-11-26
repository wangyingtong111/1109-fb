# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PragmaticSegmenter::Segmenter do
  # Raw transport forms: backslashes are literal characters in these strings.
  let(:legacy_literal) { "Review \\n next." }
  let(:escaped_first) { "First.\\nNext." }
  # One.\r\n Two.\\(two backslashes)n Three. \r Four.
  let(:escaped_conflict) { "One.\\r\\nTwo.\\\\nThree.\\rFour." }

  describe 'default (undeclared) source' do
    it 'keeps the historical decoding of literal escaped newlines' do
      ps = described_class.new(text: legacy_literal)
      expect(ps.source).to eq(:legacy)
      expect(ps.text).to eq("Review \n next.")
      expect(ps.segment).to eq(['Review', 'next.'])
    end

    it 'treats source: nil / :legacy / :default the same way' do
      [nil, :legacy, 'legacy', :default].each do |source|
        expect(described_class.new(text: legacy_literal, source: source).segment)
          .to eq(['Review', 'next.'])
      end
    end
  end

  describe 'ordinary prose source (:plain / :literal)' do
    %i[plain literal prose text body].each do |name|
      it "preserves literal \\n and \\r for source: #{name}" do
        # Only one, terminal, period: any split would have to come from the
        # literal backslash sequences being decoded, which must not happen.
        text = "Review \\n next and \\r more words."
        ps = described_class.new(text: text, source: name)
        expect(ps.text).to eq(text)
        expect(ps.segment).to eq([text])
      end
    end

    it 'keeps a literal \\r as text beside a normal sentence boundary' do
      text = "Review \\n next.\\rAnd more."
      ps = described_class.new(text: text, source: :plain)
      expect(ps.text).to eq(text)
      expect(ps.segment).to eq(["Review \\n next.", "\\rAnd more."])
    end

    it 'preserves literals even with clean: false' do
      ps = described_class.new(text: legacy_literal, source: :plain, clean: false)
      expect(ps.text).to eq(legacy_literal)
      expect(ps.segment).to eq([legacy_literal])
    end
  end

  describe 'escaped transport field source (:escaped)' do
    it 'decodes First.\\nNext. like a real newline during normal cleaning' do
      ps = described_class.new(text: escaped_first, source: :escaped)
      expect(ps.text).to eq("First.\rNext.")
      expect(ps.segment).to eq(['First.', 'Next.'])

      real_newline = described_class.new(text: "First.\nNext.").segment
      expect(ps.segment).to eq(real_newline)
    end

    it 'decodes CRLF, paired backslashes and standalone CR in one pass' do
      ps = described_class.new(text: escaped_conflict, source: :escaped, clean: false)

      # Exactly two line boundaries (CRLF + CR) and one literal backslash.
      expect(ps.text).to eq("One.\r\nTwo.\\nThree.\rFour.")
      expect(ps.segment).to eq(['One.', 'Two.', "\\nThree.", 'Four.'])
    end

    it 'does not decode the resulting literal backslash a second time when cleaning' do
      ps = described_class.new(text: escaped_conflict, source: :escaped, clean: true)
      expect(ps.segment).to eq(['One.', 'Two.', "\\nThree.", 'Four.'])
    end

    it 'accepts the common source keyword/value aliases' do
      expect(described_class.new(text: escaped_conflict, clean: false, input_source: :escaped).segment)
        .to eq(['One.', 'Two.', "\\nThree.", 'Four.'])
      expect(described_class.new(text: escaped_conflict, clean: false, mode: :escaped).segment)
        .to eq(['One.', 'Two.', "\\nThree.", 'Four.'])
      expect(described_class.new(text: escaped_conflict, clean: false, escaped: true).segment)
        .to eq(['One.', 'Two.', "\\nThree.", 'Four.'])
      expect(described_class.new(text: escaped_conflict, clean: false, source: 'transport').segment)
        .to eq(['One.', 'Two.', "\\nThree.", 'Four.'])
    end
  end

  it 'still handles nil text' do
    ps = described_class.new(text: nil, source: :escaped)
    expect(ps.segment).to eq([])
    expect(ps.text).to be_nil
  end
end