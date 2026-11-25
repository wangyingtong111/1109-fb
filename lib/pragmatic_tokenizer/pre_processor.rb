module PragmaticTokenizer
  module PreProcessor

    # Punctuation that ends a sentence rather than an address
    URL_TRAILING_PUNCTUATION = %w[. , ; : ! ? # & | ^ - ” ’ ， … « » „ \\].freeze
    BRACKET_PAIRS            = { '(' => ')', '[' => ']', '{' => '}', '<' => '>' }.freeze

    def pre_process(language: Languages::Common)
      remove_non_breaking_space!
      shift_various_characters!
      replace_colon_in_url!
      shift_remaining_colons!
      shift_hashtag!
      convert_double_quotes!
      convert_single_quotes!(language)
      convert_acute_accent_s!
      shift_hyphens!
      squeeze(' '.freeze)
    end

    # Trailing punctuation is not part of an address: it belongs to the
    # surrounding sentence (e.g. the ")." of ".../Foo_(bar)#intro)."). Splits a
    # URL candidate into the address itself and the punctuation that follows it.
    def self.split_trailing_punctuation(candidate)
      trailing = ''
      loop do
        break if candidate.length <= 1
        character = candidate[-1]
        if URL_TRAILING_PUNCTUATION.include?(character) || stray_bracket?(candidate, character)
          trailing = character + trailing
          candidate = candidate[0...-1]
        else
          break
        end
      end
      [candidate, trailing]
    end

    # A closing bracket belongs to the address when it closes a bracket that was
    # opened inside it (e.g. ".../wiki/Foo_(bar)"); otherwise it belongs to the
    # surrounding text (e.g. the ")" of "(.../docs#intro)"). An opening bracket
    # at the very end is never part of an address.
    def self.stray_bracket?(candidate, character)
      if BRACKET_PAIRS.key?(character)
        candidate.count(character) > candidate.count(BRACKET_PAIRS[character])
      elsif BRACKET_PAIRS.value?(character)
        candidate.count(character) > candidate.count(BRACKET_PAIRS.key(character))
      else
        false
      end
    end

    private

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
