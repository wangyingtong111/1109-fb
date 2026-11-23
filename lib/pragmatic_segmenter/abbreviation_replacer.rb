# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'unicode'

module PragmaticSegmenter
  # This class searches for periods within an abbreviation and
  # replaces the periods.
  class AbbreviationReplacer

    attr_reader :text
    def initialize(text:, language:, abbreviation: nil)
      @text = text.dup
      @language = language
      @abbreviation_config = abbreviation || {}
    end

    def replace
      Rule.apply(@text,
        @language::PossessiveAbbreviationRule,
        @language::KommanditgesellschaftRule,
        @language::SingleLetterAbbreviationRules::All)

      @text = search_for_abbreviations_in_string(@text)
      @text = replace_multi_period_abbreviations(@text)
      Rule.apply(@text, @language::AmPmRules::All)
      replace_abbreviation_as_sentence_boundary(@text)
    end

    private

    # Abbreviations supplied by the document side. They are layered on top of
    # the language's built-in abbreviation table without mutating it, so the
    # default behavior of every language stays untouched when no custom
    # configuration is provided.
    def custom_abbreviations
      @custom_abbreviations ||= to_set(@abbreviation_config[:abbreviations])
    end

    # Ordinary honorifics/titles (e.g. "Mr.", "Dr.") whose period never starts
    # a new sentence, regardless of the capitalization of the following word.
    def custom_prepositive_abbreviations
      @custom_prepositive_abbreviations ||= to_set(@abbreviation_config[:prepositive_abbreviations])
    end

    # Units of measure (e.g. "ms.") whose period only closes a sentence when
    # the following word is an explicit sentence starter. A period followed by
    # a lower-case word, a digit or punctuation is still treated as part of the
    # same sentence.
    def custom_unit_abbreviations
      @custom_unit_abbreviations ||= to_set(@abbreviation_config[:unit_abbreviations])
    end

    # Figure/number prefixes (e.g. "Fig.") whose period is glued to the number
    # that follows it and is only part of a reference, never a sentence end.
    def custom_number_abbreviations
      @custom_number_abbreviations ||= to_set(@abbreviation_config[:number_abbreviations])
    end

    def to_set(value)
      case value
      when ::Set then value.dup
      when nil then ::Set.new
      else ::Set.new(value)
      end
    end

    def abbreviations
      @abbreviations ||= to_set(@language::Abbreviation::ABBREVIATIONS)
        .merge(custom_abbreviations)
        .merge(custom_prepositive_abbreviations)
        .merge(custom_unit_abbreviations)
        .merge(custom_number_abbreviations)
    end

    def prepositive_abbreviations
      # A unit configured on the document side overrides a built-in honorific
      # role (e.g. "Ms.") so its period is allowed to close a sentence.
      @prepositive_abbreviations ||= to_set(@language::Abbreviation::PREPOSITIVE_ABBREVIATIONS)
        .merge(custom_prepositive_abbreviations)
        .subtract(custom_unit_abbreviations)
    end

    def number_abbreviations
      @number_abbreviations ||= to_set(@language::Abbreviation::NUMBER_ABBREVIATIONS)
        .merge(custom_number_abbreviations)
    end

    def search_for_abbreviations_in_string(txt)
      original = txt.dup
      downcased = Unicode::downcase(txt)
      abbreviations.each do |abbreviation|
        stripped = abbreviation.strip
        next unless downcased.include?(stripped)
        abbrev_match = original.scan(/(?:^|\s|\r|\n)#{Regexp.escape(stripped)}/i)
        next if abbrev_match.empty?
        # Grab the first non-space character following the abbreviation's own
        # period (which may already have been substituted with ∯). Looking past
        # the period ensures we inspect the word that actually starts the next
        # token instead of the period itself.
        next_word_start = /#{Regexp.escape(stripped)}[.∯]\s*(\S)/i
        character_array = txt.scan(next_word_start).flatten
        abbrev_match.each_with_index do |am, index|
          txt = scan_for_replacements(txt, am, index, character_array)
        end
      end
      txt
    end

    def scan_for_replacements(txt, am, index, character_array)
      character = character_array[index]
      prepositive = prepositive_abbreviations
      number_abbr = number_abbreviations
      key = Unicode::downcase(am.strip)
      upper = /[[:upper:]]/.match(character.to_s)
      if upper.nil? || prepositive.include?(key)
        if prepositive.include?(key)
          txt = replace_prepositive_abbr(txt, am)
        elsif number_abbr.include?(key)
          txt = replace_pre_number_abbr(txt, am)
        else
          txt = replace_period_of_abbr(txt, am)
        end
      end
      txt
    end

    def replace_abbreviation_as_sentence_boundary(txt)
      # As we are being conservative and keeping ambiguous
      # sentence boundaries as one sentence instead of
      # splitting into two, we can split at words that
      # we know for certain never follow these abbreviations.
      # Some might say that the set of words that follow an
      # abbreviation such as U.S. (i.e. U.S. Government) is smaller than
      # the set of words that could start a sentence and
      # never follow U.S. However, we are being conservative
      # and not splitting by default, so we need to look for places
      # where we definitely can split. Obviously SENTENCE_STARTERS
      # will never cover all cases, but as the gem is named
      # 'Pragmatic Segmenter' we need to be pragmatic
      # and try to cover the words that most often start a
      # sentence but could never follow one of the abbreviations below.

      # Rubular: http://rubular.com/r/PkBQ3PVBS8
      @language::AbbreviationReplacer::SENTENCE_STARTERS.each do |word|
        escaped = Regexp.escape(word)
        regex   = /(U∯S|U\.S|U∯K|E∯U|E\.U|U∯S∯A|U\.S\.A|I|i.v|I.V)∯(?=\s#{escaped}\s)/
        txt.gsub!(regex, '\1.')
      end
      txt
    end

    def replace_multi_period_abbreviations(txt)
      mpa = txt.scan(@language::MULTI_PERIOD_ABBREVIATION_REGEX)
      return txt if mpa.empty?
      mpa.each do |r|
        txt.gsub!(/#{Regexp.escape(r)}/, "#{r.gsub!('.', '∯')}")
      end
      txt
    end

    def replace_pre_number_abbr(txt, abbr)
      txt.gsub!(/(?<=\s#{abbr.strip})\.(?=\s\d)|(?<=^#{abbr.strip})\.(?=\s\d)/i, '∯')
      txt.gsub!(/(?<=\s#{abbr.strip})\.(?=\s+\()|(?<=^#{abbr.strip})\.(?=\s+\()/i, '∯')
      txt
    end

    def replace_prepositive_abbr(txt, abbr)
      txt.gsub!(/(?<=\s#{abbr.strip})\.(?=\s)|(?<=^#{abbr.strip})\.(?=\s)/, '∯')
      txt.gsub!(/(?<=\s#{abbr.strip})\.(?=:\d+)|(?<=^#{abbr.strip})\.(?=:\d+)/, '∯')
      txt
    end

    def replace_period_of_abbr(txt, abbr)
      txt.gsub!(/(?<=\s#{abbr.strip})\.(?=((\.|\:|-|\?)|(\s([a-z]|I\s|I'm|I'll|\d|\())))|(?<=^#{abbr.strip})\.(?=((\.|\:|\?)|(\s([a-z]|I\s|I'm|I'll|\d))))/, '∯')
      txt.gsub!(/(?<=\s#{abbr.strip})\.(?=,)|(?<=^#{abbr.strip})\.(?=,)/, '∯')
      txt
    end

    def replace_possessive_abbreviations(txt)
      txt.gsub!(@language::POSSESSIVE_ABBREVIATION_REGEX, '∯')
      txt
    end
  end
end
