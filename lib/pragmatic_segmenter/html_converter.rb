# frozen_string_literal: true

module PragmaticSegmenter
  # Converts HTML into plain text suitable for sentence segmentation.
  #
  # Block-level elements (e.g. <p>, <div>, <li>) become sentence
  # boundaries (newlines), inline elements (e.g. <em>, <b>, <span>)
  # are removed without introducing a break, and HTML entities
  # (e.g. &nbsp;, &lt;) are decoded into their literal characters.
  #
  # This only runs when the caller explicitly passes doc_type: 'html';
  # the default plain text path is left untouched.
  module HTMLConverter
    # Rubular: http://rubular.com/r/9d0OVOEJWj
    ANY_TAG_REGEX = /<[^>]*>/

    SCRIPT_OR_STYLE_REGEX = /<(script|style)[^>]*>.*?<\/\1>/im

    HTML_COMMENT_REGEX = /<!--.*?-->/m

    BLOCK_ELEMENTS = %w(
      address article aside blockquote br dd details dialog div dl dt
      fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 header
      hgroup hr li main nav ol p pre section table tbody td tfoot th
      thead tr ul
    ).freeze

    BLOCK_TAG_REGEX = /<\/?(?:#{BLOCK_ELEMENTS.join('|')})(?:\s[^>]*)?\/?>/i

    NAMED_ENTITIES = {
      'nbsp'   => ' ',
      'lt'     => '<',
      'gt'     => '>',
      'amp'    => '&',
      'quot'   => '"',
      'apos'   => "'",
      'mdash'  => '—',
      'ndash'  => '–',
      'hellip' => '…',
      'lsquo'  => '‘',
      'rsquo'  => '’',
      'ldquo'  => '“',
      'rdquo'  => '”',
      'laquo'  => '«',
      'raquo'  => '»',
      'copy'   => '©',
      'reg'    => '®',
      'trade'  => '™'
    }.freeze

    ENTITY_REGEX = /&(#x?[0-9a-fA-F]+|\w+);/

    class << self
      # The order of operations matters: script/style content and
      # comments are dropped first, block-level tags are turned into
      # newlines, remaining (inline) tags are removed, and entities are
      # decoded last so that escaped markup such as &lt;ok&gt; survives
      # as literal text instead of being treated as a tag.
      def to_text(html)
        text = html.dup
        text.gsub!(SCRIPT_OR_STYLE_REGEX, ' ')
        text.gsub!(HTML_COMMENT_REGEX, ' ')
        text.gsub!(BLOCK_TAG_REGEX, "\n")
        text.gsub!(ANY_TAG_REGEX, '')
        decode_entities(text)
      end

      private

      def decode_entities(text)
        text.gsub(ENTITY_REGEX) do
          entity = Regexp.last_match(1)
          if entity.start_with?('#x', '#X')
            codepoint_to_char(entity[2..-1].to_i(16)) || "&#{entity};"
          elsif entity.start_with?('#')
            codepoint_to_char(entity[1..-1].to_i) || "&#{entity};"
          else
            NAMED_ENTITIES.fetch(entity.downcase, "&#{entity};")
          end
        end
      end

      def codepoint_to_char(codepoint)
        codepoint.chr(Encoding::UTF_8)
      rescue RangeError
        nil
      end
    end
  end
end
