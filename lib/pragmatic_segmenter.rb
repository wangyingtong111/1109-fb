require "set"
require "pragmatic_segmenter/version"
require "pragmatic_segmenter/segmenter"

module PragmaticSegmenter
  def self.stream(chunks = nil, **options, &block)
    Segmenter.stream(chunks, **options, &block)
  end

  def self.stream_events(events, **options)
    Segmenter.stream_events(events, **options)
  end
end