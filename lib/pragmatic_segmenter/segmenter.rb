# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/languages'
require 'pragmatic_segmenter/index_mapper'

module PragmaticSegmenter
  # This class segments a text into an array of sentences.
  class Segmenter
    attr_reader :text, :language, :doc_type, :original_text

    def initialize(text:, language: 'en', doc_type: nil, clean: true)
      return unless text
      @language = language
      @language_module = Languages.get_language_by_code(language)
      @doc_type = doc_type
      @original_text = text

      if clean
        @text = cleaner.new(text: text, doc_type: @doc_type, language: @language_module).clean
      else
        @text = text
      end
    end

    def segment
      return [] unless @text
      processor.new(language: @language_module).process(text: @text)
    end

    # Segments the text and additionally locates every sentence inside the
    # original, unmodified input text.
    #
    # Example:
    #   >> ps = PragmaticSegmenter::Segmenter.new(text: "Dr. Li\narrived.")
    #   >> ps.segment_with_indices
    #   => [{:segment=>"Dr. Li", :start_index=>0, :end_index=>7},
    #       {:segment=>"arrived.", :start_index=>7, :end_index=>15}]
    #
    # The indices are half-open character offsets into the text originally
    # passed to the segmenter, so +original_text[start_index...end_index]+
    # returns the sentence with any line breaks removed while cleaning still
    # included. Whitespace separating two sentences belongs to no interval.
    #
    # @return [Array<Hash>] one hash per segment with +:segment+,
    #   +:start_index+ and +:end_index+ keys
    def segment_with_indices
      return [] unless @text
      IndexMapper.new(original_text: @original_text, segments: segment).map
    end

    # Same as {#segment_with_indices} but returns Ruby Range objects
    # (+start_index...end_index+) instead of index integers.
    #
    # @return [Array<Hash>] one hash per segment with +:segment+ and
    #   +:range+ keys
    def segment_with_ranges
      segment_with_indices.map do |result|
        { segment: result[:segment], range: result[:start_index]...result[:end_index] }
      end
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
