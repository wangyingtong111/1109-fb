# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/languages'

module PragmaticSegmenter
  # This class segments a text into an array of sentences.
  class Segmenter
    TERMINAL_EVENTS = %i[finish end done close flush finalize].freeze

    attr_reader :text, :raw_text, :language, :doc_type

    def initialize(text_arg = nil, text: nil, language: 'en', doc_type: nil,
                   clean: true, **_options, &listener)
      if text_arg.is_a?(Hash)
        options = text_arg
        text = options.key?(:text) ? options[:text] : options['text'] if text.nil?
        language = options[:language] || options['language'] || language
        doc_type = options.key?(:doc_type) ? options[:doc_type] : options['doc_type'] if doc_type.nil?
        clean = options.key?(:clean) ? options[:clean] : options.fetch('clean', clean)
      elsif text.nil?
        text = text_arg
      end

      @raw_text = text
      @language = language || 'en'
      @language_module = Languages.get_language_by_code(@language)
      @doc_type = doc_type
      @clean = clean
      @listener = listener
      return unless text

      @text = if clean
                cleaner.new(text: text, doc_type: @doc_type, language: @language_module).clean
              else
                text
              end
    end

    def segment(chunk = nil, at_end = false, finish: false, flush: false,
                final: false, last: false, done: false, close: false, &callback)
      ending = at_end || finish || flush || final || last || done || close
      terminal = ending || terminal_event?(chunk)

      if chunk.nil? && !terminal
        segment_whole
      elsif terminal
        stream.finish(string_chunk(chunk), &callback)
      else
        stream.push(chunk, &callback)
      end
    end

    def segment_whole
      return [] unless @text
      processor.new(language: @language_module).process(text: @text)
    end
    alias all_sentences segment_whole
    alias batch_segment segment_whole

    def stream
      @stream ||= Stream.new(
        text: @raw_text || '',
        language: @language,
        doc_type: @doc_type,
        clean: clean?,
        &@listener
      )
    end
    alias chunk_stream stream
    alias streaming stream

    def push(*chunks, &callback)
      stream.push(*chunks, &callback)
    end
    alias feed push
    alias append push
    alias add push
    alias enqueue push
    alias offer push
    alias ingest push
    alias write push
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
    alias call push

    def finish(*chunks, &callback)
      stream.finish(*chunks, &callback)
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

    def <<(chunk)
      stream << chunk
      self
    end

    def stream_sentences
      stream.sentences
    end
    alias sentences stream_sentences
    alias output stream_sentences
    alias results stream_sentences
    alias emitted_sentences stream_sentences

    def stream_buffer
      stream.buffer
    end

    def stream_closed?
      stream.closed?
    end
    alias finished? stream_closed?
    alias closed? stream_closed?

    def self.stream(chunks = nil, text: nil, language: 'en', doc_type: nil,
                   clean: true, **options, &block)
      stream = Stream.new(
        text: text,
        language: language,
        doc_type: doc_type,
        clean: clean,
        **options,
        &block
      )

      case chunks
      when nil
        stream
      when String
        stream.push(chunks)
        stream
      else
        if chunks.respond_to?(:each)
          chunks.each do |chunk|
            terminal = chunk.is_a?(Symbol) && Segmenter::TERMINAL_EVENTS.include?(chunk) ||
                       chunk.is_a?(String) && Segmenter::TERMINAL_EVENTS.include?(chunk.to_sym)
            if terminal
              stream.finish
            else
              stream.push(chunk)
            end
          end
        else
          stream.push(chunks)
        end
        stream
      end
    end

    def self.streaming(chunks = nil, **options, &block)
      stream(chunks, **options, &block)
    end

    def self.chunked(chunks = nil, **options, &block)
      stream(chunks, **options, &block)
    end

    def self.chunk_stream(chunks = nil, **options, &block)
      stream(chunks, **options, &block)
    end

    def self.segment_stream(chunks = nil, **options, &block)
      stream(chunks, **options, &block)
    end

    def self.create_stream(chunks = nil, **options, &block)
      stream(chunks, **options, &block)
    end

    def self.new_stream(chunks = nil, **options, &block)
      stream(chunks, **options, &block)
    end

    def self.stream_events(events, **options)
      stream = self.stream(**options)
      events.map do |event|
        case event
        when Symbol, String
          if Segmenter::TERMINAL_EVENTS.include?(event.to_sym)
            stream.finish
          else
            stream.push(event)
          end
        else
          stream.push(event)
        end
      end
    end

    private

    def clean?
      defined?(@clean) ? @clean : true
    end

    def terminal_event?(event)
      event.is_a?(Symbol) && TERMINAL_EVENTS.include?(event) ||
        event.is_a?(String) && TERMINAL_EVENTS.include?(event.to_sym)
    end

    def string_chunk(chunk)
      return if chunk.nil? || terminal_event?(chunk)
      chunk
    end

    def processor
      @language_module::Processor
    rescue
      Processor
    end

    def cleaner
      @language_module::Cleaner
    rescue
      Cleaner
    end

    # Streaming implementation lives in stream.rb and is required after
    # Segmenter has been defined.
  end
end

require_relative 'stream'
