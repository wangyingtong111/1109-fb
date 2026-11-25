require 'spec_helper'

RSpec.describe PragmaticSegmenter::LayoutText do
  let(:paragraph) { "Alpha fragment Beta continues" }
  let(:mixed) { "inter national travel Next fragment Beta" }
  let(:first_position) { mixed.index(" ") }
  let(:second_position) { mixed.index(" ", first_position + 1) }

  describe 'opt-in segmentation' do
    it 'uses U+2029 as a separator without returning it' do
      result = PragmaticSegmenter::Segmenter.new(text: paragraph, layout_text: true).segment
      expect(result).to eq(['Alpha fragment', 'Beta continues'])
      expect(result.join).not_to include(" ")
    end

    it 'interprets individual U+2028 positions as intra-word or hard breaks' do
      breaks = {
        first_position => :intra_word,
        second_position => :hard_line
      }

      result = PragmaticSegmenter::Segmenter.new(
        text: mixed,
        layout_text: true,
        line_breaks: breaks
      ).segment

      expect(result).to eq(['international travel', 'Next fragment', 'Beta'])
    end

    it 'accepts U+2028 roles in source order' do
      result = PragmaticSegmenter::Segmenter.new(
        text: mixed,
        layout_text: [:intra_word, :hard_line]
      ).segment

      expect(result).to eq(['international travel', 'Next fragment', 'Beta'])
    end

    it 'accepts character positions through the layout: shortcut' do
      breaks = {
        first_position => :intra_word,
        second_position => :hard_line
      }

      result = PragmaticSegmenter::Segmenter.new(text: mixed, layout: breaks).segment
      expect(result).to eq(['international travel', 'Next fragment', 'Beta'])
    end

    it 'accepts byte positions from the original UTF-8 input' do
      first_byte = mixed.bytes.index(0xe2)
      second_byte = mixed.bytes.each_with_index
                               .find { |byte, index| index > first_byte && byte == 0xe2 }
                               .last

      result = PragmaticSegmenter::Segmenter.new(
        text: mixed,
        layout: { first_byte => :intra_word, second_byte => :hard_line }
      ).segment

      expect(result).to eq(['international travel', 'Next fragment', 'Beta'])
    end

    it 'converts inter-word wraps to ordinary spaces' do
      text = "international travel by train"
      result = PragmaticSegmenter::Segmenter.new(
        text: text,
        layout_text: [:inter_word]
      ).segment

      expect(result).to eq(['international travel by train'])
    end

    it 'processes hard layout blocks independently' do
      text = "Dr. Smith arrived."
      result = PragmaticSegmenter::Segmenter.new(
        text: text,
        layout_text: [:hard_line]
      ).segment

      expect(result).to eq(['Dr.', 'Smith arrived.'])
    end

    it 'accepts a block classified by original position' do
      result = PragmaticSegmenter::Segmenter.new(text: mixed, layout_text: true) do |position, index|
        index.zero? ? :intra_word : :hard_line
      end.segment

      expect(result).to eq(['international travel', 'Next fragment', 'Beta'])
    end
  end

  describe 'default segmentation' do
    it 'keeps ordinary visible spaces on the original path' do
      expect(PragmaticSegmenter::Segmenter.new(text: 'Alpha fragment').segment)
        .to eq(['Alpha fragment'])
    end

    it 'does not enable Unicode layout behavior without an option' do
      expect(PragmaticSegmenter::Segmenter.new(text: paragraph).segment)
        .to eq([paragraph])
    end

    it 'does not mutate the input when layout mode is enabled' do
      text = mixed.dup
      PragmaticSegmenter::Segmenter.new(
        text: text,
        layout_text: { first_position => :intra_word, second_position => :hard_line }
      ).segment

      expect(text).to eq(mixed)
    end
  end
end
