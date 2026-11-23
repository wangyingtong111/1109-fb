# -*- encoding : utf-8 -*-
# frozen_string_literal: true

module PragmaticSegmenter
  # Maps the sentences produced by the segmenter back onto the original,
  # unmodified input text as half-open character intervals [start, end).
  #
  # The cleaning pipeline can delete characters (e.g. line breaks inside a
  # sentence, xhtml tags) so the cleaned segments are re-aligned with the
  # original text sequentially. Whitespace in either string is skipped while
  # matching the non-whitespace content, and when the cleaner has deleted a
  # non-whitespace character the alignment jumps forward to the next matching
  # position.
  #
  # A line break that used to sit inside a sentence but was removed while
  # cleaning is still covered by that sentence's interval. Whitespace that
  # separates two sentences (including blank lines made of two or more line
  # breaks) is never part of any interval.
  class IndexMapper
    # Maximum number of original characters scanned while recovering from a
    # character that the cleaner removed (e.g. an inline html tag).
    RECOVERY_WINDOW = 200

    # @param original_text [String] the text handed to the segmenter
    # @param segments [Array<String>] the sentences returned by {Segmenter#segment}
    def initialize(original_text:, segments:)
      @original_text = original_text
      @segments = segments
    end

    # @return [Array<Hash>] one entry per segment, each containing the
    #   segment text and its half-open +[start_index, end_index)+ interval
    #   in the original text
    def map
      return [] if @segments.nil? || @segments.empty?

      cursor = 0
      @segments.map do |segment|
        start_index = nil
        end_index = nil

        segment.each_char do |char|
          # Spaces inside the cleaned segment do not occupy an interval on
          # their own; the surrounding real characters do.
          next if char =~ /\s/

          cursor = skip_whitespace(cursor)
          cursor = recover_position(cursor, char, segment)
          start_index = cursor if start_index.nil?
          end_index = cursor + 1
          cursor += 1
        end

        raise AlignmentError, "unable to map segment #{segment.inspect} onto the original text" if start_index.nil?

        end_index = extend_across_internal_line_break(end_index)
        { segment: segment, start_index: start_index, end_index: end_index }
      end
    end

    class AlignmentError < StandardError; end

    private

    def skip_whitespace(index)
      index += 1 while index < @original_text.length && @original_text[index] =~ /\s/
      index
    end

    # The cleaner occasionally removes non-whitespace characters (xhtml,
    # table of contents markers, ...). Skip past them to the next position
    # holding the expected character instead of losing the alignment.
    def recover_position(index, char, segment)
      return index if index < @original_text.length && @original_text[index] == char

      limit = [index + RECOVERY_WINDOW, @original_text.length].min
      candidate = index
      candidate += 1 while candidate < limit && @original_text[candidate] != char
      if candidate < limit
        candidate
      else
        raise AlignmentError,
          "unable to map segment #{segment.inspect} onto the original text"
      end
    end

    # When a single line break directly follows the sentence in the original
    # text it was an intra-sentence line break removed by the cleaner, so the
    # interval keeps covering it. Two or more line breaks separate sentences
    # and stay outside every interval.
    def extend_across_internal_line_break(end_index)
      return end_index unless line_break_at?(end_index)

      if @original_text[end_index] == "\r" &&
         end_index + 1 < @original_text.length &&
         @original_text[end_index + 1] == "\n"
        line_break_end = end_index + 2
      else
        line_break_end = end_index + 1
      end

      line_break_at?(line_break_end) ? end_index : line_break_end
    end

    def line_break_at?(index)
      index < @original_text.length && @original_text[index] =~ /\r|\n/
    end
  end
end
