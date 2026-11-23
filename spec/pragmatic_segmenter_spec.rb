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

      describe 'PDF end-of-line hyphens with a vocabulary' do
        let(:text) do
          "This is a classi-\nfier that is noise-\nrobust and they re-\nsign the contract. Next sentence alone."
        end
        let(:vocabulary) { ['classifier', 'noise-robust', 're-sign', 'resign'] }

        it 'rejoins ordinary broken words and keeps hyphens of vocabulary compound words' do
          ps = PragmaticSegmenter::Cleaner.new(text: text, doc_type: 'pdf', vocabulary: vocabulary)
          expect(ps.clean).to eq(
            "This is a classifier that is noise-robust and they re-sign the contract. Next sentence alone."
          )
        end

        it 'keeps the hyphen when both spellings are in the vocabulary' do
          cleaned = PragmaticSegmenter::Cleaner.new(
            text: "they re-\nsign today",
            doc_type: 'pdf',
            vocabulary: vocabulary
          ).clean
          expect(cleaned).to eq("they re-sign today")
        end

        it 'removes the hyphen when only the joined spelling is in the vocabulary' do
          cleaned = PragmaticSegmenter::Cleaner.new(
            text: "a classi-\nfier",
            doc_type: 'pdf',
            vocabulary: vocabulary
          ).clean
          expect(cleaned).to eq("a classifier")
        end

        it 'segments rejoined text and returns the sentence after the period separately' do
          ps = PragmaticSegmenter::Segmenter.new(text: text, doc_type: 'pdf', vocabulary: vocabulary)
          expect(ps.segment).to eq(
            [
              "This is a classifier that is noise-robust and they re-sign the contract.",
              "Next sentence alone."
            ]
          )
        end

        it 'looks up vocabulary entries case-insensitively while keeping the original casing' do
          cleaned = PragmaticSegmenter::Cleaner.new(
            text: "Classi-\nfier model.",
            doc_type: 'pdf',
            vocabulary: vocabulary
          ).clean
          expect(cleaned).to eq("Classifier model.")
        end

        it 'leaves unknown hyphenated line breaks to the default PDF rules' do
          cleaned = PragmaticSegmenter::Cleaner.new(
            text: "A flibber-\ngibbet appears.",
            doc_type: 'pdf',
            vocabulary: vocabulary
          ).clean
          expect(cleaned).to eq("A flibber- gibbet appears.")
        end

        it 'keeps the legacy behavior when no vocabulary is provided (cleaner)' do
          cleaned = PragmaticSegmenter::Cleaner.new(text: text, doc_type: 'pdf').clean
          expect(cleaned).to eq(
            "This is a classi- fier that is noise- robust and they re- sign the contract. Next sentence alone."
          )
        end

        it 'keeps the legacy behavior when no vocabulary is provided (segmenter)' do
          ps = PragmaticSegmenter::Segmenter.new(text: text, doc_type: 'pdf')
          expect(ps.segment).to eq(
            [
              "This is a classi- fier that is noise- robust and they re- sign the contract.",
              "Next sentence alone."
            ]
          )
        end

        it 'does not use the vocabulary outside of PDF documents' do
          cleaned = PragmaticSegmenter::Cleaner.new(
            text: text,
            vocabulary: vocabulary
          ).clean
          expect(cleaned).not_to include('classifier')
        end

        it 'exposes the vocabulary' do
          cleaner = PragmaticSegmenter::Cleaner.new(text: text, vocabulary: vocabulary)
          expect(cleaner.vocabulary).to eq(vocabulary)
        end
      end
    end
  end
end
