module PragmaticTokenizer
  module PreProcessor

    NOTHING = ''.freeze
    SPACE   = ' '.freeze

    def pre_process(language: Languages::Common, language_at: nil)
      if language_at.nil?
        legacy_pre_process(language)
      else
        annotated_pre_process(language_at, language)
      end
    end

    private

      # The original, whole-string pipeline. Untouched for plain String input.
      def legacy_pre_process(language)
        remove_non_breaking_space!
        shift_various_characters!
        replace_colon_in_url!
        shift_remaining_colons!
        shift_hashtag!
        convert_double_quotes!
        convert_single_quotes!(language)
        convert_acute_accent_s!
        shift_hyphens!
        squeeze(SPACE)
      end

      # Annotated pipeline. `language_at` holds one entry per character of this
      # segment: the language module declared at that character's original position,
      # or nil when the fragment was not annotated. Every rewrite below keeps the
      # array in lockstep with the text, so a word that crosses a transport fragment
      # boundary still knows the language of its first character. Fragment boundaries
      # are never treated as word boundaries.
      def annotated_pre_process(language_at, main_language)
        @language_at = language_at.dup
        apply_with_languages!(Regex::NO_BREAK_SPACE, SPACE)
        apply_with_languages!(Regex::PRE_PROCESS, ' \1 \2 \3 \4 \5 \6 \7 \8 \9 ')
        apply_with_languages!(Regex::COLON_IN_URL, replacement_for_key(':'.freeze))
        apply_with_languages!(':'.freeze, ' :'.freeze) if self !~ Regex::TIME_WITH_COLON
        apply_with_languages!('#'.freeze, ' #'.freeze)
        apply_with_languages!(Regex::QUOTE, replacements_for_quotes)
        convert_single_quotes_with_languages!(main_language)
        convert_acute_accent_s!
        shift_hyphens!
        squeeze(SPACE)
      end

      # Mirror a gsub/replace step while rebuilding the per-character language array.
      # Original characters keep their language, inserted characters (punctuation the
      # rewrite produces) inherit the language of the match's first character, and
      # inserted spaces get nil (a space can never begin a word). The replacement may
      # be a plain String (with \1..\9 backreferences) or a Hash like #convert_double_quotes!
      # uses; a matched key missing from the Hash deletes the match, exactly like gsub.
      def apply_with_languages!(pattern, replacement)
        text       = self[0, 0].dup
        languages  = []
        cursor     = 0

        to_enum(:scan, pattern).each do
          match = Regexp.last_match
          text << self[cursor...match.begin(0)]
          languages.concat(@language_at[cursor...match.begin(0)])

          language  = @language_at[match.begin(0)]
          rendered  = render_replacement(replacement, match)
          rendered.each_char do |character|
            text << character
            languages << (character == SPACE ? nil : language)
          end
          cursor = match.end(0)
        end

        text << self[cursor..-1].to_s
        languages.concat(@language_at[cursor..-1] || [])

        replace(text)
        @language_at = languages
      end

      def render_replacement(replacement, match)
        return expand_backreferences(replacement, match) unless replacement.is_a?(Hash)
        value = replacement.key?(match[0]) ? replacement[match[0]] : nil
        value.to_s
      end

      def expand_backreferences(replacement, match)
        expanded = self[0, 0].dup
        index    = 0
        length   = replacement.length
        while index < length
          character = replacement[index]
          if character == '\\' && index + 1 < length &&
             replacement[index + 1] >= '0' && replacement[index + 1] <= '9'
            expanded << (match[replacement[index + 1].to_i] || NOTHING)
            index += 2
          else
            expanded << character
            index += 1
          end
        end
        expanded
      end

      # Apostrophe handling is the only language-dependent rewrite. Apply it word by
      # word (whitespace delimited), always with the language declared at the word's
      # first character; an undeclared word start falls back to the main language.
      def convert_single_quotes_with_languages!(main_language)
        processed = self[0, 0].dup
        scan(/\S+|\s+/) do
          match = Regexp.last_match
          piece = match[0]
          if piece =~ /\S/
            language = @language_at[match.begin(0)] || main_language
            processed << class_for_single_quotes(language).new.handle_single_quotes(piece)
          else
            processed << piece
          end
        end
        replace(processed)
      end

      def remove_non_breaking_space!
        gsub!(Regex::NO_BREAK_SPACE, ' '.freeze)
      end

      def shift_various_characters!
        gsub!(Regex::PRE_PROCESS, ' \1 \2 \3 \4 \5 \6 \7 \8 \9 ')
      end

      def replace_colon_in_url!
        gsub!(Regex::COLON_IN_URL, replacement_for_key(':'.freeze))
      end

      def shift_remaining_colons!
        gsub!(':'.freeze, ' :'.freeze) if self !~ Regex::TIME_WITH_COLON
      end

      def shift_hashtag!
        gsub!('#'.freeze, ' #'.freeze)
      end

      def convert_double_quotes!
        gsub!(Regex::QUOTE, replacements_for_quotes)
      end

      def replacements_for_quotes
        @replacements_for_quotes ||= {
            "''" => ' ' << replacement_for_key('"'.freeze) << ' ',
            '"'  => ' ' << replacement_for_key('"'.freeze) << ' ',
            '“'  => ' ' << replacement_for_key('“'.freeze) << ' '
        }.freeze
      end

      def convert_single_quotes!(language)
        replace(class_for_single_quotes(language).new.handle_single_quotes(self))
      end

      def class_for_single_quotes(language)
        defined?(language::SingleQuotes) ? language::SingleQuotes : PragmaticTokenizer::Languages::Common::SingleQuotes
      end

      def convert_acute_accent_s!
        gsub!(Regex::ACUTE_ACCENT_S, replacement_for_key('`'.freeze))
      end

      # can these two regular expressions be merged somehow?
      def shift_hyphens!
        gsub!(Regex::HYPHEN_AFTER_NON_WORD,  ' - '.freeze)
        gsub!(Regex::HYPHEN_BEFORE_NON_WORD, ' - '.freeze)
      end

      def replacement_for_key(replacement_key)
        PragmaticTokenizer::Languages::Common::PUNCTUATION_MAP[replacement_key]
      end

  end
end
