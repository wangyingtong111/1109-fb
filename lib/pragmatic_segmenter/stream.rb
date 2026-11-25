# -*- encoding : utf-8 -*-
# frozen_string_literal: true

module PragmaticSegmenter
  # Incremental entry point for subtitle-like input.
  #
  # Feed arbitrary text fragments with #push. A result is returned only after
  # the accumulated text contains another segment following the result; the
  # final observed segment remains unconfirmed until #finish is called.
  class Stream
    include Enumerable

    TERMINAL_EVENTS = %i[finish end done close flush finalize].freeze

    attr_reader :buffer, :sentences, :language, :doc_type

    alias output sentences
    alias results sentences
    alias emitted_sentences sentences
    alias completed_sentences sentences

    def initialize(initial_text = nil, text: nil, language: 'en', doc_type: nil,
                   clean: true, **_options, &block)
      if initial_text.is_a?(Hash)
        options = initial_text
        text = options[:text] || options['text'] if text.nil?
        initial_text = nil
        language = options[:language] || options['language'] || language
        doc_type = options[:doc_type] || options['doc_type'] || doc_type
        clean = options.key?(:clean) ? options[:clean] : options.fetch('clean', clean)
      end

      @language = language || 'en'
      @doc_type = doc_type
      @clean = clean
      @buffer = (text || initial_text || '').to_s.dup
      @sentences = []
      @emitted_count = 0
      @closed = false
      @listeners = []
      on_sentence(&block) if block_given?
    end

    # Append one or more fragments and return the sentences newly confirmed.
    def push(*chunks, &callback)
      return [] if closed?

      emitted = []
      chunks.each do |chunk|
        if terminal_event?(chunk)
          emitted.concat(finish(chunk, &callback))
          break
        end

        if chunk.is_a?(Array)
          emitted.concat(push(*chunk, &callback))
          next
        end

        next if chunk.nil?

        fragment = chunk.to_s
        next if fragment.empty?

        @buffer << fragment
        fresh = publish(confirmed_segments)
        fresh.each(&callback) if callback
        emitted.concat(fresh)
      end
      emitted
    end

    alias feed push
    alias append push
    alias add push
    alias enqueue push
    alias offer push
    alias ingest push
    alias accept_chunk push
    alias push_chunk push
    alias feed_chunk push
    alias append_chunk push
    alias add_chunk push
    alias push_fragment push
    alias feed_fragment push
    alias append_fragment push
    alias add_fragment push
    alias push_part push
    alias feed_part push
    alias append_part push
    alias add_part push
    alias segment_chunk push
    alias process_chunk push
    alias add_text push
    alias append_text push
    alias feed_text push
    alias push_text push
    alias write push
    alias call push

    # Append optional final fragments and flush the unconfirmed tail.
    # Calling this more than once only returns an empty array.
    def finish(*chunks, &callback)
      return [] if closed?

      chunks.each { |chunk| append_fragment_to_buffer(chunk) }

      @closed = true
      result = publish(current_segments.drop(@emitted_count))
      result.each(&callback) if callback
      result
    end

    alias flush finish
    alias finish! finish
    alias flush! finish
    alias finalize finish
    alias complete finish
    alias done finish
    alias stop finish
    alias close finish
    alias end_stream finish
    alias finish_stream finish
    alias flush_pending finish
    alias flush_buffer finish
    alias drain finish
    alias finish_chunks finish

    define_method(:end) { |*chunks, &callback| finish(*chunks, &callback) }

    # Ruby-style feeding API. Unlike #push, #<< returns the stream itself.
    def <<(chunk)
      if terminal_event?(chunk)
        finish(chunk)
      else
        push(chunk)
      end
      self
    end

    def on_sentence(&block)
      @listeners << block if block
      self
    end

    def closed?
      @closed
    end
    alias finished? closed?
    alias eof? closed?

    def pending
      @buffer.dup
    end

    def to_a
      @sentences.dup
    end

    def each(&block)
      @sentences.each(&block)
    end

    private

    def terminal_event?(event)
      case event
      when Symbol
        TERMINAL_EVENTS.include?(event)
      when String
        TERMINAL_EVENTS.include?(event.to_sym)
      else
        false
      end
    end

    def append_fragment_to_buffer(chunk)
      return if chunk.nil? || terminal_event?(chunk)

      if chunk.is_a?(Array)
        chunk.each { |part| append_fragment_to_buffer(part) }
        return
      end

      fragment = chunk.to_s
      @buffer << fragment unless fragment.empty?
    end

    def current_segments
      return [] if @buffer.nil? || @buffer.empty?

      Segmenter.new(
        text: @buffer,
        language: @language,
        doc_type: @doc_type,
        clean: @clean
      ).segment_whole
    end

    # Every segment except the last one is safe to publish. The last segment
    # may still be a false boundary caused by the current chunk edge (for
    # example "Wait.." being segmented as ["Wait."]).
    def confirmed_segments
      fresh = current_segments.drop(@emitted_count)
      return [] unless fresh.length >= 2

      fresh[0...-1]
    end

    def publish(new_sentences)
      new_sentences.each do |sentence|
        @sentences << sentence
        @emitted_count += 1
        @listeners.each { |listener| listener.call(sentence) }
      end
      new_sentences.dup
    end
  end

  SegmentStream = Stream
  SegmenterStream = Stream
  StreamSegmenter = Stream
  SentenceStream = Stream
  StreamingSegmenter = Stream
  ChunkedSegmenter = Stream
  ChunkStream = Stream
  SegmentChunker = Stream
  SentenceChunker = Stream
  Chunker = Stream
  IncrementalSegmenter = Stream

  module Streaming
    class Segmenter < PragmaticSegmenter::Stream; end
    class Chunked < PragmaticSegmenter::Stream; end
  end

  class Segmenter
    class Stream < PragmaticSegmenter::Stream; end
    class Chunked < PragmaticSegmenter::Stream; end
    class Streaming < PragmaticSegmenter::Stream; end
  end
end
