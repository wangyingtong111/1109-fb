# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PragmaticSegmenter::Segmenter do
  describe '#segment_with_indices' do
    it 'handles nil' do
      ps = PragmaticSegmenter::Segmenter.new(text: nil)
      expect(ps.segment_with_indices).to eq([])
    end

    it 'handles empty strings' do
      ps = PragmaticSegmenter::Segmenter.new(text: '')
      expect(ps.segment_with_indices).to eq([])
    end

    it 'keeps the existing string based segment method unchanged' do
      text = "Élan.\n\nDr. Li\narrived. Dr. Li\narrived."
      ps = PragmaticSegmenter::Segmenter.new(text: text)
      expect(ps.segment).to eq(['Élan.', 'Dr. Li', 'arrived.', 'Dr. Li', 'arrived.'])
    end

    it 'returns the segment together with its half open original text interval' do
      text = 'Hello world. Hello.'
      ps = PragmaticSegmenter::Segmenter.new(text: text)
      expect(ps.segment_with_indices).to eq(
        [
          { segment: 'Hello world.', start_index: 0, end_index: 12 },
          { segment: 'Hello.', start_index: 13, end_index: 19 }
        ]
      )
    end

    it 'keeps an intra sentence line break inside the sentence interval' do
      text = 'It was a cold ' + "\n" + 'night in the city.'
      ps = PragmaticSegmenter::Segmenter.new(text: text)
      expect(ps.segment).to eq(['It was a cold night in the city.'])
      result = ps.segment_with_indices
      expect(result.length).to eq(1)
      expect(result.first[:start_index]).to eq(0)
      expect(result.first[:end_index]).to eq(text.length)
      expect(text[result.first[:start_index]...result.first[:end_index]]).to eq(text)
    end

    it 'never counts the separating whitespace between sentences' do
      text = "Élan.\n\nDr. Li\narrived. Dr. Li\narrived."
      results = PragmaticSegmenter::Segmenter.new(text: text).segment_with_indices

      covered = results.flat_map { |h| (h[:start_index]...h[:end_index]).to_a }
      expect(covered).not_to include(5) # first "\n" of the blank line
      expect(covered).not_to include(6) # second "\n" of the blank line
      expect(covered).not_to include(22) # space between the two repeated clauses
    end

    context 'with the described two Dr. Li clauses' do
      let(:text) { "Élan.\n\nDr. Li\narrived. Dr. Li\narrived." }
      let(:results) { PragmaticSegmenter::Segmenter.new(text: text).segment_with_indices }

      it 'locates every segment in the original text' do
        expect(results).to eq(
          [
            { segment: 'Élan.', start_index: 0, end_index: 5 },
            { segment: 'Dr. Li', start_index: 7, end_index: 14 },
            { segment: 'arrived.', start_index: 14, end_index: 22 },
            { segment: 'Dr. Li', start_index: 23, end_index: 30 },
            { segment: 'arrived.', start_index: 30, end_index: 38 }
          ]
        )
      end

      it 'maps identical cleaned segments to different half open intervals' do
        expect(results[1][:segment]).to eq(results[3][:segment])
        expect(results[1][:start_index]).not_to eq(results[3][:start_index])
        expect(results[1][:end_index]).not_to eq(results[3][:end_index])
      end

      it 'still contains the original line break when slicing the original text' do
        expect(text[results[1][:start_index]...results[1][:end_index]]).to eq("Dr. Li\n")
        expect(text[results[3][:start_index]...results[3][:end_index]]).to eq("Dr. Li\n")
      end

      it 'excludes the blank line after the first sentence' do
        expect(text[results.first[:start_index]...results.first[:end_index]]).to eq('Élan.')
      end
    end

    it 're-aligns when the cleaner removes non whitespace characters' do
      text = '<b>Hello</b> world. Bye.'
      ps = PragmaticSegmenter::Segmenter.new(text: text)
      expect(ps.segment).to eq(['Hello world.', 'Bye.'])
      results = ps.segment_with_indices
      results.each do |h|
        slice = text[h[:start_index]...h[:end_index]]
        expect(slice.gsub(/<[^>]*>/, '').gsub(/\s+/, ' ')).to eq(h[:segment])
      end
    end

    it 'works when cleaning is disabled' do
      text = "It was a cold \nnight in the city."
      ps = PragmaticSegmenter::Segmenter.new(text: text, clean: false)
      expect(ps.segment).to eq(['It was a cold', 'night in the city.'])
      results = ps.segment_with_indices
      expect(results.map { |h| h[:segment] }).to eq(ps.segment)
      results.each do |h|
        expect(h[:start_index]).to be >= 0
        expect(h[:end_index]).to be <= text.length
        expect(h[:start_index]).to be < h[:end_index]
      end
    end

    it 'does not mutate the input string' do
      text = "It was a cold \nnight in the city."
      PragmaticSegmenter::Segmenter.new(text: text).segment_with_indices
      expect(text).to eq("It was a cold \nnight in the city.")
    end
  end

  describe '#segment_with_ranges' do
    it 'returns Range objects that slice the sentences out of the original text' do
      text = "Élan.\n\nDr. Li\narrived. Dr. Li\narrived."
      results = PragmaticSegmenter::Segmenter.new(text: text).segment_with_ranges
      expect(results.map { |h| h[:range] }).to eq(
        [0...5, 7...14, 14...22, 23...30, 30...38]
      )
      expect(results.map { |h| text[h[:range]] }).to eq(
        ['Élan.', "Dr. Li\n", 'arrived.', "Dr. Li\n", 'arrived.']
      )
    end

    it 'handles nil' do
      expect(PragmaticSegmenter::Segmenter.new(text: nil).segment_with_ranges).to eq([])
    end
  end
end
