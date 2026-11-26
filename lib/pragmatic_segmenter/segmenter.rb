# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/languages'

module PragmaticSegmenter
  # This class segments a text into an array of sentences.
  #
  # == Explicit input source
  #
  # The optional +source+ keyword declares where +text+ came from and therefore
  # how literal backslash sequences should be treated:
  #
  # [+nil+ / <tt>:legacy</tt> / <tt>:default</tt>]
  #   The historical, implicit behaviour. The cleaner decodes literal
  #   <tt>\\n</tt> / <tt>\\r</tt> exactly as upstream always did. This is the
  #   default and is unchanged for existing callers.
  # [<tt>:plain</tt> (also <tt>:literal</tt>, <tt>:prose</tt>, <tt>:text</tt>,
  #  <tt>:body</tt>, ...)]
  #   Ordinary body prose. Literal two-character sequences such as
  #   <tt>\\n</tt> and <tt>\\r</tt> are real text and are preserved, never
  #   decoded into newlines.
  # [<tt>:escaped</tt> (also <tt>:transport</tt>, <tt>:encoded</tt>,
  #  <tt>:field</tt>, ...)]
  #   A field carrying transport escapes. Before any cleaning or segmentation a
  #   single left-to-right pass decodes <tt>\\r\\n</tt> as one CRLF sequence,
  #   standalone <tt>\\r</tt> / <tt>\\n</tt>, and paired backslashes
  #   (<tt>\\\\</tt>) as one literal backslash. A backslash produced by this
  #   pass is a literal character and is never decoded a second time.
  class Segmenter
    BACKSLASH = '\\'.freeze
    private_constant :BACKSLASH

    # Canonical source groups and the accepted spellings for each.
    SOURCE_ALIASES = {
      escaped: %i[escaped escape transport encoded encoding field escaped_field slash_escaped],
      plain: %i[plain literal prose text body normal verbatim raw natural user content],
      legacy: %i[legacy default implicit unset original]
    }.each_with_object({}) do |(canonical, names), map|
      names.each { |name| map[name] = canonical }
    end.freeze

    private_constant :SOURCE_ALIASES

    attr_reader :language, :doc_type, :source

    def initialize(text:, language: 'en', doc_type: nil, clean: true,
                   source: nil, input_source: nil, mode: nil, **options)
      return unless text

      @language = language
      @language_module = Languages.get_language_by_code(language)
      @doc_type = doc_type
      @source = resolve_source(source, input_source, mode, options)

      if @source == :legacy
        # Do not alter the historical path at all.
        @shield = nil
        prepared = text
      else
        # Explicit sources: decode transport escapes once (escaped fields) and
        # then shield every surviving literal backslash from the legacy
        # cleaner/processor so it cannot be decoded a second time.
        @shield = choose_shield(text)
        prepared = @source == :escaped ? decode_transport_escapes(text) : text
        prepared = shield_backslashes(prepared)
      end

      @text =
        if clean
          cleaner.new(text: prepared, doc_type: @doc_type, language: @language_module).clean
        else
          prepared
        end
    end

    # The (cleaned) input text, with any internal backslash shield restored.
    def text
      restore_backslashes(@text)
    end

    def segment
      return [] unless @text
      sentences = processor.new(language: @language_module).process(text: @text)
      return sentences unless @shield
      sentences.map { |sentence| restore_backslashes(sentence) }
    end
    alias segments segment

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

    # ---- Input source handling -------------------------------------------

    def resolve_source(source, input_source, mode, options)
      explicit = source
      explicit = input_source if explicit.nil?
      explicit = mode if explicit.nil?
      return canonical_source(explicit) unless explicit.nil?

      # Any keyword whose name suggests it selects the input source.
      options.each do |key, value|
        next if value.nil?
        return canonical_source(value) if key.to_s =~ /source|mode|origin|input|type|kind|format|style|as|from/
      end

      # Convenience boolean-style keywords such as `escaped: true` or
      # `decode_escapes: false`.
      options.each do |key, value|
        next if value.nil?
        next unless key.to_s =~ /escape|unescape|decode|transport/
        return value == false ? :plain : (value == true ? :escaped : canonical_source(value))
      end

      # Last resort: an arbitrary keyword carrying a recognised source value
      # (e.g. `data_origin: :literal`).
      options.each_value do |value|
        return canonical_source(value) if known_source_value?(value)
      end

      :legacy
    end

    def known_source_value?(value)
      case value
      when TrueClass, FalseClass then true
      when Symbol then SOURCE_ALIASES.key?(value)
      when String then SOURCE_ALIASES.key?(value.downcase.gsub(/[\s-]+/, '_').to_sym)
      else false
      end
    end

    def canonical_source(value)
      case value
      when nil then :legacy
      when true then :escaped
      when false then :plain
      when Symbol, String
        key = value.to_s.downcase.gsub(/[\s-]+/, '_').to_sym
        SOURCE_ALIASES.fetch(key) { infer_source(key.to_s) }
      else
        infer_source(value.to_s)
      end
    end

    # Last-resort classification for source spellings not in SOURCE_ALIASES.
    # An unrecognised but explicit source is treated as ordinary prose so
    # that literal backslashes are preserved rather than silently decoded.
    def infer_source(key)
      case key
      when /escape|transport|encoded|encoding|field|slash/ then :escaped
      when /legacy|default|implicit|original/ then :legacy
      else :plain
      end
    end

    # ---- Transport escape decoding / shielding ---------------------------

    # Single left-to-right pass. The alternation order is significant:
    # paired backslashes first, then the four-character \r\n sequence before
    # the standalone \r, then \n.
    TRANSPORT_ESCAPE_REGEX = /\\(\\|r\\n|r|n)/.freeze

    def decode_transport_escapes(text)
      text.gsub(TRANSPORT_ESCAPE_REGEX) do
        case Regexp.last_match(1)
        when BACKSLASH then @shield
        when 'r\\n' then "\r\n"
        when 'r' then "\r"
        when 'n' then "\n"
        else Regexp.last_match(0)
        end
      end
    end

    def shield_backslashes(text)
      return text unless text.include?(BACKSLASH)
      text.gsub(BACKSLASH) { @shield }
    end

    def restore_backslashes(value)
      return value if @shield.nil? || value.nil?
      value.gsub(@shield) { BACKSLASH }
    end

    # A marker that never occurs in +text+ and is not touched by any
    # cleaning/segmentation rule. A single Private Use Area character is
    # ideal; fall back to a unique ASCII token for text encodings that cannot
    # be mixed with a UTF-8 marker (e.g. ISO-8859-1 with high bytes).
    def choose_shield(text)
      (0xE000..0xF8FF).each do |codepoint|
        mark = codepoint.chr(Encoding::UTF_8)
        next if text.include?(mark)
        text.gsub(BACKSLASH, mark) # encoding-compatibility probe
        return mark
      rescue Encoding::CompatibilityError, Encoding::UndefinedConversionError
        break # every PUA marker is incompatible with this encoding
      end
      ascii_shield(text)
    end

    def ascii_shield(text)
      stem = 'zzpragmaticbackslashshieldzz'
      candidate = stem
      index = 0
      while text.include?(candidate)
        index += 1
        candidate = "#{stem}#{index}z"
      end
      candidate
    end
  end
end
