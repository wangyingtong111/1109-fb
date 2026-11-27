require 'spec_helper'

RSpec.describe PragmaticSegmenter::TurnSegmenter do
  def words(*entries)
    entries.map { |text, start, finish| [text, start, finish] }
  end

  describe 'turns as hard sentence boundaries' do
    it 'returns one full sentence for A and two for B, keeping 10:30 together' do
      turns = [
        ['A', words(['at', 0, 1], ['10:30', 1, 2], ['we', 2, 3], ['met', 3, 4])],
        ['B', words(['yes.', 10, 11], ['Then', 11, 12], ['left.', 12, 13])]
      ]

      results = PragmaticSegmenter::TurnSegmenter.new(turns).segment

      expect(results.map(&:sentence)).to eq(['at 10:30 we met', 'yes.', 'Then left.'])
      expect(results.map(&:turn)).to eq([0, 1, 1])
      expect(results.map(&:speaker)).to eq(%w[A B B])

      expect(results[0].word_range).to eq(0...4)
      expect(results[0].time_range).to eq([0, 4])

      expect(results[1].word_range).to eq(0...1)
      expect(results[1].time_range).to eq([10, 11])

      expect(results[2].word_range).to eq(1...3)
      expect(results[2].time_range).to eq([11, 13])
    end
  end

  describe 'formatting markers removed by the cleaner' do
    it 'does not let a vanished marker take the word slot of the next sentence' do
      turns = [[
        'A',
        [
          ['<em>', 0, 1], ['yes.', 1, 2], ['</em>', 2, 3],
          ['<i>', 3, 4], ['yes.', 4, 5], ['</i>', 5, 6],
          ['Then', 6, 7], ['left.', 7, 8]
        ]
      ]]

      results = PragmaticSegmenter::TurnSegmenter.new(turns).segment

      expect(results.map(&:sentence)).to eq(['yes.', 'yes.', 'Then left.'])
      expect(results[0].word_indices).to eq([1, 2])
      expect(results[1].word_indices).to eq([4, 5])
      expect(results[2].word_indices).to eq([6, 8])
      expect(results[0].time_range).to eq([1, 2])
      expect(results[1].time_range).to eq([4, 5])
      expect(results[2].time_range).to eq([6, 8])
    end

    it 'keeps a tag that ASR split over two words from claiming word slots' do
      turns = [['A', [['<em', 0, 1], ['>', 1, 2], ['yes.', 2, 3], ['Again.', 3, 4]]]]

      results = PragmaticSegmenter::TurnSegmenter.new(turns).segment

      expect(results.map(&:sentence)).to eq(['yes.', 'Again.'])
      expect(results[0].word_indices).to eq([2, 3])
      expect(results[1].word_indices).to eq([3, 4])
    end
  end

  describe 'consecutive turns by the same speaker' do
    it 'keeps the new turn number and only cites the words of that turn' do
      turns = [
        ['A', [['<em>', 0, 1], ['yes.', 1, 2], ['</em>', 2, 3],
               ['<i>', 3, 4], ['yes.', 4, 5], ['</i>', 5, 6],
               ['Then', 6, 7], ['left.', 7, 8]]],
        ['A', [['yes.', 10, 11], ['Again.', 11, 12]]]
      ]

      results = PragmaticSegmenter::TurnSegmenter.new(turns).segment

      expect(results.map(&:turn)).to eq([0, 0, 0, 1, 1])
      expect(results.map(&:speaker)).to eq(%w[A A A A A])
      expect(results.map(&:sentence)).to eq(['yes.', 'yes.', 'Then left.', 'yes.', 'Again.'])

      expect(results[3].word_indices).to eq([0, 1])
      expect(results[3].time_range).to eq([10, 11])
      expect(results[4].word_indices).to eq([1, 2])
      expect(results[4].time_range).to eq([11, 12])
    end
  end

  describe 'result fields' do
    it 'exposes turn, speaker, sentence, word range and time as fields only' do
      turns = [['A', [['yes.', 0, 1]]]]
      result = PragmaticSegmenter::TurnSegmenter.new(turns).segment.first

      expect(result.to_a).to eq([0, 'A', 'yes.', 0...1, [0, 1]])
      expect(result[:speaker]).to eq('A')
      expect(result[:sentence]).to eq('yes.')
      expect(result[:word_range]).to eq(0...1)
      expect(result[:start_time]).to eq(0)
      expect(result[:end_time]).to eq(1)
      expect(result.to_h[:first_word_time]).to eq([0, 1])
      expect(result.to_h[:last_word_time]).to eq([0, 1])
    end

    it 'accepts hashes and objects as turns and words' do
      word = Struct.new(:text, :start, :end)
      turns = [{ speaker: 'A', words: [word.new('yes.', 0, 1), word.new('Again.', 1, 2)] }]

      results = PragmaticSegmenter::TurnSegmenter.new(turns).segment
      expect(results.map(&:sentence)).to eq(['yes.', 'Again.'])
    end
  end

  describe 'the plain string entry point' do
    it 'still segments a flat string without dialogue metadata' do
      text = 'Dr. Smith arrived. Next came Dr. Jones.'
      expect(PragmaticSegmenter::Segmenter.new(text: text).segment)
        .to eq(['Dr. Smith arrived.', 'Next came Dr. Jones.'])
    end

    it 'returns exactly the sentence strings, with nothing added' do
      results = PragmaticSegmenter::Segmenter.new(text: 'yes. Then left.').segment
      expect(results).to eq(['yes.', 'Then left.'])
      expect(results).to all(be_a(String))
    end
  end
end
