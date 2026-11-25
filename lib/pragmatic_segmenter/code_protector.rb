# -*- encoding: utf-8 -*-
# frozen_string_literal: true

module PragmaticSegmenter
  # Protects Markdown style code spans from the sentence segmentation rules.
  #
  # When code protection is enabled, fenced code blocks and inline code spans
  # are swapped for inert placeholders before the text is cleaned and
  # segmented. The placeholders contain none of the punctuation that the
  # segmentation rules look for, so the content inside a code span can never
  # introduce a natural sentence boundary. The original code is restored
  # verbatim once segmentation is finished.
  #
  # * Fenced code blocks (``` ... ``` or ~~~ ... ~~~) are returned as their
  #   own independent, untouched segments.
  # * Inline code stays attached to the surrounding natural sentence.
  # * Only closed spans are matched, so an unclosed backtick is treated as a
  #   literal character and can never swallow the following sentence.
  class CodeProtector
    # Characters from the Unicode Private Use Area. They are extremely
    # unlikely to appear in real text and contain no sentence punctuation.
    MARKER_PREFIX = ''
    MARKER_SUFFIX = ''
    MARKER_REGEX = /#{MARKER_PREFIX}(\d+)#{MARKER_SUFFIX}/

    # A fenced code block, or an inline code span delimited by one or more
    # backticks. Alternatives are ordered so the longest fence is tried first
    # and so double backticks win over a single backtick.
    CODE_SPAN_REGEX =
      /(?:^|\n)(?<fence>`{3,}|~{3,})[^\n]*\n.*?\n\k<fence>[^\n]*(?=\n|\z)|
       (?<double>`{2})(?:(?!\k<double>).)*\k<double>|
       `[^`\n]*`/mx.freeze

    def initialize
      @code_blocks = []
    end

    # Replace every code span with an inert placeholder. Fenced blocks are
    # surrounded with carriage returns so that they become standalone
    # segments once the text is segmented.
    def protect(text)
      return text unless text

      text.gsub(CODE_SPAN_REGEX) do |match|
        marker = store(match)
        if fenced?(match)
          "\r#{marker}\r"
        else
          marker
        end
      end
    end

    # Restore the original code inside the already segmented sentences.
    def restore(segments)
      segments.map do |segment|
        segment.gsub(MARKER_REGEX) { @code_blocks[Regexp.last_match(1).to_i] }
      end
    end

    private

    def store(code)
      index = @code_blocks.length
      @code_blocks << code
      "#{MARKER_PREFIX}#{index}#{MARKER_SUFFIX}"
    end

    def fenced?(code)
      code =~ /\A\n?(?:`{3,}|~{3,})/
    end
  end
end
