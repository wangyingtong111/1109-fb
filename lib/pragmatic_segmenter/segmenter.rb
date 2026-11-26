# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/languages'

module PragmaticSegmenter
  # This class segments a text into an array of sentences.
  class Segmenter
    attr_reader :text, :language, :doc_type, :punctuation_mode

    def initialize(text:, language: 'en', doc_type: nil, clean: true, punctuation_mode: nil)
      return unless text
      @language = language
      @language_module = Languages.get_language_by_code(language)
      @punctuation_mode = punctuation_mode
      @language_module = punctuation_mode_module(punctuation_mode) if punctuation_mode
      @doc_type = doc_type

      if clean
        @text = cleaner.new(text: text, doc_type: @doc_type, language: @language_module).clean
      else
        @text = text.dup
      end
    end

    def segment
      return [] unless @text
      process(@text.dup)
    end

    private

    # Every run works on its own copy of the cleaned text. The punctuation
    # substitution markers used while segmenting are therefore never
    # written back into the instance or the caller's string, so repeated
    # calls cannot accumulate markers or mutate the input.
    def process(txt)
      processor.new(language: @language_module).process(text: txt)
    end

    def punctuation_mode_module(mode)
      unless @language_module.respond_to?(:with_punctuation_mode)
        raise ArgumentError, "#{@language_module} does not support a punctuation mode"
      end
      @language_module.with_punctuation_mode(mode)
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
