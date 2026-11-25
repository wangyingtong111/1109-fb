# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/languages'

module PragmaticSegmenter
  # This class segments a text into an array of sentences.
  class Segmenter
    attr_reader :text, :language, :doc_type, :ellipsis

    def initialize(text:, language: 'en', doc_type: nil, clean: true, ellipsis: false)
      return unless text
      @language = language
      @language_module = Languages.get_language_by_code(language)
      @doc_type = doc_type
      @ellipsis = ellipsis

      if clean
        @text = cleaner.new(text: text, doc_type: @doc_type, language: @language_module).clean
      else
        @text = text
      end
    end

    def segment
      return [] unless @text
      processor.new(language: @language_module, ellipsis: ellipsis).process(text: @text)
    end

    private

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
