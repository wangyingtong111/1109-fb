# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'pragmatic_segmenter/segmenter'
require 'pragmatic_segmenter/languages'

module PragmaticSegmenter
  # Half-open interval over the original word indices of one turn.
  #
  # It is a real Range (so it splices, compares and prints as [first, last + 1)
  # would) but it also compares equal to the equivalent two element array,
  # because the boundary is usually reported as "[1,2)" in transcripts.
  class WordRange < Range
    def self.build(first, exclusive_end)
      new(first, exclusive_end, true)
    end

    # [first, last + 1)
    def to_indices
      [first, last]
    end

    # Lets a caller read the boundary as the [first, last + 1] pair it reports.
    def to_ary
      to_indices
    end

    def to_s
      "[#{first},#{last})"
    end

    def ==(other)
      if other.is_a?(Array)
        other.length == 2 && other[0] == first && other[1] == last
      else
        super
      end
    end
    alias eql? ==

    def hash
      [first, last].hash
    end
  end

  # One sentence carved out of a single dialogue turn.
  #
  # The word range and the times always point back at the original word
  # sequence of that turn, never at the words that happened to survive the
  # cleaner. A pure formatting marker that the cleaner erases keeps its slot
  # in the original numbering but is never reported as a word of the sentence,
  # so it can not take the word position of a later sentence.
  class TurnSentence
    attr_reader :turn, :speaker, :sentence,
                :first_word_index, :last_word_index,
                :first_word_time, :last_word_time

    def initialize(turn:, speaker:, sentence:, first_word_index:, last_word_index:,
                   first_word_time: nil, last_word_time: nil)
      @turn = turn
      @speaker = speaker
      @sentence = sentence
      @first_word_index = first_word_index
      @last_word_index = last_word_index
      @first_word_time = first_word_time
      @last_word_time = last_word_time
    end

    alias turn_index turn

    # Turn counted from 1, for callers that prefer that convention.
    def turn_number
      @turn + 1
    end

    # Half-open interval over the original words of the turn: [first, last + 1).
    def word_range
      WordRange.build(@first_word_index, @last_word_index + 1)
    end

    # The same half-open interval written as a two element array.
    def word_indices
      [@first_word_index, @last_word_index + 1]
    end
    alias words word_indices

    # Start time of the first word that is actually voiced.
    def start_time
      @first_word_time && @first_word_time[0]
    end

    # End time of the last word that is actually voiced.
    def end_time
      @last_word_time && @last_word_time[1]
    end

    # [start of the first voiced word, end of the last voiced word]
    def time_range
      [start_time, end_time]
    end
    alias times time_range

    def first_time
      @first_word_time
    end

    def last_time
      @last_word_time
    end

    def to_a
      [@turn, @speaker, @sentence, word_range, time_range]
    end

    def to_h
      {
        turn: @turn,
        speaker: @speaker,
        sentence: @sentence,
        word_range: word_range,
        word_indices: word_indices,
        first_word_index: @first_word_index,
        last_word_index: @last_word_index,
        first_word_time: @first_word_time,
        last_word_time: @last_word_time,
        start_time: start_time,
        end_time: end_time,
        time_range: time_range
      }
    end

    def deconstruct
      to_a
    end

    def deconstruct_keys(_keys)
      to_h
    end

    def [](key)
      case key
      when Integer then to_a[key]
      when Symbol then to_h[key]
      when String then to_h[key.to_sym]
      else to_h[key]
      end
    end

    def ==(other)
      case other
      when TurnSentence then to_h == other.to_h
      when Array then to_a == other
      when Hash then to_h == other
      else false
      end
    end
    alias eql? ==

    def hash
      to_h.hash
    end

    def inspect
      "#<#{self.class.name} turn=#{@turn.inspect} speaker=#{@speaker.inspect} " \
        "sentence=#{@sentence.inspect} words=#{word_indices.inspect} " \
        "time=#{time_range.inspect}>"
    end
  end

  # Segments an already turn-delimited dialogue transcript into sentences.
  #
  # Every turn is a hard boundary: turns are segmented one by one in input
  # order, and each turn keeps its own word numbering even when two
  # consecutive turns belong to the same speaker.
  class TurnSegmenter
    # A tag may be spread over a couple of ASR words (e.g. "<em" + ">").
    MAX_TAG_FRAGMENTS = 4

    attr_reader :turns, :language, :doc_type, :clean

    def self.segment(*args, **options)
      new(*args, **options).segment
    end

    def initialize(*args, turns: nil, language: 'en', doc_type: nil, clean: true)
      @turns = turns || args.first || []
      @language = language
      @doc_type = doc_type
      @clean = clean
      @language_module = Languages.get_language_by_code(language)
    end

    def segment(turns = nil)
      @turns = turns if turns
      return [] unless @turns

      @turns.each_with_index.flat_map do |turn, turn_index|
        speaker, words = normalize_turn(turn)
        segment_turn(turn_index, speaker, words)
      end
    end

    private

    def segment_turn(turn_index, speaker, words)
      tokens = voice(words)
      return [] if tokens.empty?

      text = tokens.map { |token| token[:text] }.join(' ')
      sentence_texts = segment_text(text)
      located = locate(text, tokens, sentence_texts)

      located.map do |sentence, first, last|
        TurnSentence.new(
          turn: turn_index,
          speaker: speaker,
          sentence: sentence,
          first_word_index: first[:index],
          last_word_index: last[:index],
          first_word_time: first[:first_time],
          last_word_time: last[:last_time]
        )
      end
    end

    # Sentence boundaries are found by the existing plain string entry point,
    # so abbreviations, numbers, colons inside times and ellipses keep exactly
    # the behaviour they already have. The text handed to it was already
    # cleaned word by word, so it is segmented without cleaning again and
    # every character stays traceable to an original word.
    def segment_text(text)
      Segmenter.new(
        text: text,
        language: @language,
        doc_type: @doc_type,
        clean: false
      ).segment
    end

    # Maps the sentence strings returned by the segmenter back onto original
    # word indices through their character span in the joined text.
    #
    # Each sentence is searched starting after the end of the previous one, so
    # two identical sentences (e.g. "yes." twice) still get different word
    # ranges instead of being matched by a count of visible words. The start
    # index never moves backwards, so sentences can not be mixed up; two
    # sentences that really sit inside the same original word simply report
    # that same word.
    def locate(text, tokens, sentence_texts)
      offsets = []
      offset = 0
      tokens.each do |token|
        offsets << offset
        offset += token[:text].length + 1
      end

      cursor = 0
      previous_start = 0
      sentence_texts.each_with_object([]) do |sentence, results|
        next if sentence.nil? || sentence.empty?

        found = find_span(text, sentence, cursor)
        next unless found

        matched, from_char, to_char = found
        first = token_at(tokens, offsets, from_char)
        last = token_at(tokens, offsets, to_char - 1)
        next if first.nil? || last.nil? || last < first
        next if first < previous_start

        results << [matched, tokens[first], tokens[last]]
        cursor = to_char
        previous_start = first
      end
    end

    def find_span(text, sentence, cursor)
      start = text.index(sentence, cursor) || text.index(sentence)
      return [sentence, start, start + sentence.length] if start

      words = sentence.split(/\s+/).reject(&:empty?)
      return nil if words.empty?

      pattern = Regexp.new(words.map { |word| Regexp.escape(word) }.join('\s+'))
      match = text.match(pattern, cursor) || text.match(pattern)
      return nil unless match

      [match[0], match.begin(0), match.end(0)]
    end

    # Index of the token whose character span covers char_position.
    def token_at(tokens, offsets, char_position)
      return nil if char_position.nil? || char_position.negative?

      first_after = offsets.bsearch_index { |start| start > char_position }
      index = first_after.nil? ? tokens.length - 1 : first_after - 1
      return nil if index.negative?

      index if char_position < offsets[index] + tokens[index][:text].length
    end

    # Turns the original words into the words that are actually spoken.
    #
    # A word (or a whole tag spread over several words) that the cleaner
    # erases produces no token at all: it keeps its original index but never
    # takes a slot in the visible stream, so the next sentence is not shifted
    # onto it.
    def voice(words)
      return [] if words.nil? || words.empty?

      normalized = words.map.with_index do |word, index|
        text, start_time, end_time = normalize_word(word)
        { index: index, text: text.to_s, start: start_time, end: end_time }
      end

      groups = @clean ? group_words(normalized) : normalized.map { |word| [word] }

      groups.filter_map do |group|
        text = group.map { |word| word[:text] }.join(' ')
        text = clean_word(text) if @clean
        text = text.strip
        next nil if text.empty?

        {
          index: group.first[:index],
          text: text,
          first_time: [group.first[:start], group.first[:end]],
          last_time: [group.last[:start], group.last[:end]]
        }
      end
    end

    # Joins a tag that ASR happened to split over several words. A lone
    # fragment without a closing bracket in reach stays visible as ordinary
    # text instead of swallowing the rest of the turn.
    def group_words(words)
      groups = []
      index = 0
      while index < words.length
        word = words[index]
        if incomplete_tag?(word[:text])
          last = [index + MAX_TAG_FRAGMENTS, words.length - 1].min
          cursor = index + 1
          cursor += 1 while cursor <= last && !words[cursor][:text].include?('>')
          if cursor <= last
            groups << words[index..cursor]
            index = cursor + 1
            next
          end
        end
        groups << [word]
        index += 1
      end
      groups
    end

    def incomplete_tag?(text)
      text.start_with?('<') && !text.include?('>')
    end

    def clean_word(text)
      cleaner.new(
        text: text,
        doc_type: @doc_type,
        language: @language_module
      ).clean.to_s
    end

    def cleaner
      @cleaner ||= begin
        @language_module.const_get(:Cleaner)
      rescue NameError
        PragmaticSegmenter::Cleaner
      end
    end

    def normalize_turn(turn)
      return [turn[0], turn[1]] if turn.is_a?(Array)
      return [hash_value(turn, :speaker, :speaker_id, :who),
              hash_value(turn, :words, :tokens)] if turn.is_a?(Hash)
      return [turn.speaker, turn.words] if turn.respond_to?(:speaker) && turn.respond_to?(:words)

      raise ArgumentError, "unsupported turn: #{turn.inspect}"
    end

    def normalize_word(word)
      return [word, nil, nil] if word.is_a?(String)

      if word.is_a?(Array)
        text = word[0]
        if word.length >= 2 && word[1].is_a?(Array)
          return [text, word[1][0], word[1][1]]
        end
        return [text, word[1], word[2]]
      end

      if word.is_a?(Hash)
        return [
          hash_value(word, :text, :word, :token),
          hash_value(word, :start, :start_time, :begin),
          hash_value(word, :end, :end_time, :stop)
        ]
      end

      if word.respond_to?(:text) || word.respond_to?(:word)
        text = word.respond_to?(:text) ? word.text : word.word
        return [
          text,
          time_attribute(word, :start, :start_time, :begin),
          time_attribute(word, :end, :end_time, :stop)
        ]
      end

      raise ArgumentError, "unsupported word: #{word.inspect}"
    end

    def time_attribute(object, *names)
      names.each do |name|
        return object.public_send(name) if object.respond_to?(name)
      end
      nil
    end

    def hash_value(hash, *keys)
      keys.each do |key|
        return hash[key] if hash.key?(key)
        return hash[key.to_s] if hash.key?(key.to_s)
      end
      nil
    end
  end
end
