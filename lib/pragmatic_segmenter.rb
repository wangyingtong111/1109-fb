require "set"
require "pragmatic_segmenter/version"
require "pragmatic_segmenter/segmenter"
require "pragmatic_segmenter/turn_segmenter"

module PragmaticSegmenter
  # Segments a turn-delimited dialogue transcript into sentences.
  #
  # Each entry of +turns+ is a [speaker, words] pair (a hash or an object with
  # the same names is accepted too), and every word carries its text and its
  # start/end time. The plain string entry point is unaffected: it still
  # returns just the sentence strings.
  def self.segment_turns(turns, language: 'en', doc_type: nil, clean: true)
    TurnSegmenter.new(
      turns,
      language: language,
      doc_type: doc_type,
      clean: clean
    ).segment
  end
end
