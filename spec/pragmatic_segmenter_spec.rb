require 'spec_helper'

RSpec.describe PragmaticSegmenter::Segmenter do

  describe '#segment' do
    it 'handles nil' do
      ps = PragmaticSegmenter::Segmenter.new(text: nil)
      expect(ps.segment).to eq([])
    end

    it 'handles no language' do
      ps = PragmaticSegmenter::Segmenter.new(text: 'Hello world. Hello.')
      expect(ps.segment).to eq(["Hello world.", "Hello."])
    end

    it 'handles empty strings' do
      ps = PragmaticSegmenter::Segmenter.new(text: "\n")
      expect(ps.segment).to eq([])
    end

    it 'handles empty strings' do
      ps = PragmaticSegmenter::Segmenter.new(text: "<b></b>")
      expect(ps.segment).to eq([])
    end

    it 'handles empty strings' do
      ps = PragmaticSegmenter::Segmenter.new(text: '')
      expect(ps.segment).to eq([])
    end

    it 'has an option to not use the cleaner' do
      ps = PragmaticSegmenter::Segmenter.new(text: "It was a cold \nnight in the city.", language: "en", clean: false)
      expect(ps.segment).to eq(["It was a cold", "night in the city."])
    end

    it 'does not mutate the input string' do
      text = "It was a cold \nnight in the city."
      PragmaticSegmenter::Segmenter.new(text: text, language: "en").segment
      expect(text).to eq("It was a cold \nnight in the city.")
    end

    describe 'the optional ellipsis rule' do
      def segment(text, **options)
        PragmaticSegmenter::Segmenter.new(text: text, language: 'fr', **options).segment
      end

      it 'starts a new sentence at a capitalized accented word' do
        expect(segment('Après... Émile revient.', ellipsis: true))
          .to eq(['Après...', 'Émile revient.'])
      end

      it 'keeps a lowercase word in the same sentence' do
        expect(segment('après... encore un mot.', ellipsis: true))
          .to eq(['après... encore un mot.'])
      end

      it 'keeps the original precomposed glyph' do
        text = 'Après... Émile revient.'
        expect(segment(text, ellipsis: true).join(' ')).to eq(text)
      end

      it 'looks at the first word inside an opening quotation mark' do
        expect(segment('Après... «Émile revient.»', ellipsis: true))
          .to eq(['Après...', '«Émile revient.»'])
      end

      it 'keeps a lowercase word inside an opening quotation mark in the same sentence' do
        expect(segment('après... «encore un mot.»', ellipsis: true))
          .to eq(['après... «encore un mot.»'])
      end

      it 'treats a decomposed capital the same as a precomposed one' do
        # "E" + U+0301 COMBINING ACUTE ACCENT, i.e. the decomposed spelling of É.
        text = "Après... «Émile revient.»"
        segments = segment(text, ellipsis: true)
        expect(segments).to eq(["Après...", "«Émile revient.»"])
        expect(segments.last.codepoints.first(3)).to eq([0x00AB, 0x0045, 0x0301])
        expect(segments.last.length).to eq(text.length - 'Après... '.length)
      end

      it 'is off by default and keeps the original rule' do
        expect(segment('Après... Marie revient.')).to eq(['Après...', 'Marie revient.'])
        expect(segment('Après... Émile revient.')).to eq(['Après... Émile revient.'])
        expect(segment('après... «Émile revient.»')).to eq(['après... «Émile revient.»'])
      end
    end

    describe '#clean' do
      it 'cleans the text #001' do
        ps = PragmaticSegmenter::Cleaner.new(text: "It was a cold \nnight in the city.", language: "en")
        expect(ps.clean).to eq("It was a cold night in the city.")
      end

      it 'cleans the text #002' do
        text = 'injections made by the Shareholder through the years. 7 (max.) 3. Specifications/4.Design and function The operating instructions are part of the product and must be kept in the immediate vicinity of the instrument and readily accessible to skilled "'
        ps = PragmaticSegmenter::Cleaner.new(text: text)
        expect(ps.clean).to eq("injections made by the Shareholder through the years. 7 (max.) 3. Specifications/4.Design and function The operating instructions are part of the product and must be kept in the immediate vicinity of the instrument and readily accessible to skilled \"")
      end

      it 'does not mutate the input string (cleaner)' do
        text = "It was a cold \nnight in the city."
        PragmaticSegmenter::Cleaner.new(text: text, language: "en").clean
        expect(text).to eq("It was a cold \nnight in the city.")
      end
    end
  end
end
