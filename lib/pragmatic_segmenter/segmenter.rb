# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/languages'
require 'pragmatic_segmenter/layout_text'

module PragmaticSegmenter
  # This class segments a text into an array of sentences.
  class Segmenter
    attr_reader :text, :language, :doc_type, :layout_text
    alias layout layout_text

    def self.layout_segment(text, **options, &block)
      new(text: text, layout_text: true, **options, &block).segment
    end

    def initialize(text:, language: 'en', doc_type: nil, clean: true,
                   layout_text: nil, layout: nil, line_breaks: nil, **options, &block)
      return unless text
      @raw_text = text.dup
      @language = language
      @language_module = Languages.get_language_by_code(language)
      @doc_type = doc_type
      @clean = clean
      @layout_text = LayoutText.resolve(
        layout_text: layout_text,
        layout: layout,
        line_breaks: line_breaks,
        options: options,
        &block
      )

      if @layout_text
        # Positions supplied by callers refer to the original input. Keep that
        # input available and clean each logical layout block independently.
        @text = @raw_text
      elsif clean
        @text = cleaner.new(text: text, doc_type: @doc_type, language: @language_module).clean
      else
        @text = text
      end
    end

    def segment
      return [] unless @text
      return segment_layout_text if layout_text

      processor.new(language: @language_module).process(text: @text)
    end

    private

    def segment_layout_text
      layout_text.hard_blocks(@raw_text).flat_map do |block|
        self.class.new(
          text: block,
          language: language,
          doc_type: doc_type,
          clean: @clean,
          layout_text: false
        ).segment
      end
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
  end
end
