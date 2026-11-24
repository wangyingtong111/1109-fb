# frozen_string_literal: true

require_relative 'fuzzy_match/version'
require_relative 'fuzzy_match/levenshtein'
require_relative 'fuzzy_match/jaro_winkler'
require_relative 'fuzzy_match/dice'
require_relative 'fuzzy_match/soundex'
require_relative 'fuzzy_match/metaphone'
require_relative 'fuzzy_match/lcs'
require_relative 'fuzzy_match/damerau_levenshtein'
require_relative 'fuzzy_match/hamming'

module Philiprehberger
  module FuzzyMatch
    class Error < StandardError
    end

    def self.levenshtein(str_a, str_b)
      Levenshtein.distance(str_a, str_b)
    end

    def self.jaro_winkler(str_a, str_b)
      JaroWinkler.similarity(str_a, str_b)
    end

    def self.dice_coefficient(str_a, str_b)
      Dice.coefficient(str_a, str_b)
    end

    # Length of the longest common subsequence
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Integer]
    def self.lcs(str_a, str_b)
      Lcs.length(str_a, str_b)
    end

    # Normalized LCS similarity (0.0 to 1.0)
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Float]
    def self.lcs_ratio(str_a, str_b)
      Lcs.ratio(str_a, str_b)
    end

    # Damerau-Levenshtein edit distance (counts transpositions as 1 edit)
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Integer]
    def self.damerau_levenshtein(str_a, str_b)
      DamerauLevenshtein.distance(str_a, str_b)
    end

    # Normalized Damerau-Levenshtein similarity (0.0 to 1.0)
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Float]
    def self.damerau_ratio(str_a, str_b)
      a = str_a.to_s.downcase
      b = str_b.to_s.downcase
      max_len = [a.length, b.length].max
      return 1.0 if max_len.zero?

      distance = DamerauLevenshtein.distance(a, b)
      1.0 - (distance.to_f / max_len)
    end

    def self.ratio(str_a, str_b)
      a = str_a.to_s.downcase
      b = str_b.to_s.downcase
      max_len = [a.length, b.length].max
      return 1.0 if max_len.zero?

      distance = Levenshtein.distance(a, b)
      1.0 - (distance.to_f / max_len)
    end

    def self.best(query, candidates, threshold: 0.0)
      return nil if candidates.empty?

      best_result = nil
      best_score = -1.0

      candidates.each do |candidate|
        score = ratio(query, candidate.to_s)
        if score > best_score
          best_result = candidate
          best_score = score
        end
      end

      return nil if best_score < threshold

      { match: best_result, score: best_score.round(4) }
    end

    def self.search(query, candidates, threshold: 0.3)
      scored = candidates.map do |candidate|
        score = ratio(query, candidate.to_s)
        { match: candidate, score: score.round(4) }
      end

      results = scored.select { |r| r[:score] >= threshold }
      results.sort_by { |r| -r[:score] }
    end

    def self.suggest(query, candidates, threshold: 0.6, max: 5)
      results = search(query, candidates, threshold: threshold)
      results.first(max).map { |r| r[:match] }
    end

    # Rank all candidates by similarity to the query (descending)
    #
    # @param query [String]
    # @param candidates [Array<String>]
    # @param algorithm [Symbol] :jaro_winkler (default), :dice, or :levenshtein
    # @return [Array<Hash>] all candidates as `{ value:, score: }` sorted by score desc,
    #   ties broken by original input order (stable)
    def self.rank(query, candidates, algorithm: :jaro_winkler)
      scored = candidates.each_with_index.map do |candidate, index|
        [{ value: candidate, score: score_for(query, candidate.to_s, algorithm: algorithm) }, index]
      end
      scored.sort_by { |pair, index| [-pair[:score], index] }.map(&:first)
    end

    # Return the top N matches sorted by score descending
    #
    # @param query [String]
    # @param candidates [Array<String>]
    # @param n [Integer] number of results to return (required)
    # @param algorithm [Symbol] :jaro_winkler (default), :dice, or :levenshtein
    # @return [Array<Hash>] up to n candidates as `{ match:, score: }` sorted by score desc
    def self.closest_n(query, candidates, n:, algorithm: :jaro_winkler)
      ranked = rank(query, candidates, algorithm: algorithm)
      ranked.first(n).map { |entry| { match: entry[:value], score: entry[:score] } }
    end

    # Return the top N best matches above an optional similarity floor
    #
    # @param query [String]
    # @param candidates [Array<String>]
    # @param n [Integer] number of results to return (required)
    # @param algorithm [Symbol] :jaro_winkler (default), :dice, or :levenshtein
    # @param min_similarity [Float] minimum similarity threshold; entries below are filtered out (default 0.0)
    # @return [Array<Hash>] up to n candidates as `{ value:, similarity: }` sorted by similarity desc
    def self.top_n(query, candidates, n:, algorithm: :jaro_winkler, min_similarity: 0.0)
      ranked = rank(query, candidates, algorithm: algorithm)
      filtered = ranked.select { |entry| entry[:score] >= min_similarity }
      filtered.first(n).map { |entry| { value: entry[:value], similarity: entry[:score] } }
    end

    # Generate a Soundex code for a string
    #
    # @param string [String]
    # @return [String] 4-character Soundex code
    def self.soundex(string)
      Soundex.code(string)
    end

    # Generate a Metaphone code for a string
    #
    # @param string [String]
    # @return [String] Metaphone phonetic code
    def self.metaphone(string)
      Metaphone.code(string)
    end

    # Check if two strings match phonetically (same Soundex code)
    #
    # @param a [String]
    # @param b [String]
    # @return [Boolean]
    def self.phonetic_match?(a, b)
      sa = Soundex.code(a)
      sb = Soundex.code(b)
      !sa.empty? && sa == sb
    end

    # Hamming distance for equal-length strings
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Integer]
    # @raise [Error] if strings have different lengths
    def self.hamming(str_a, str_b)
      Hamming.distance(str_a, str_b)
    end

    # Substring-style similarity (FuzzyWuzzy parity)
    #
    # Slides the shorter string across every same-length window of the longer
    # string and returns the **maximum** Levenshtein-based similarity ratio.
    # This is the canonical answer to "does string A appear approximately
    # inside string B?" and complements `.ratio`, which compares the whole
    # strings.
    #
    # Empty inputs:
    # - both empty → 1.0
    # - one empty  → 0.0
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Float] similarity between 0.0 and 1.0
    def self.partial_ratio(str_a, str_b)
      a = str_a.to_s.downcase
      b = str_b.to_s.downcase
      return 1.0 if a.empty? && b.empty?
      return 0.0 if a.empty? || b.empty?

      shorter, longer = a.length <= b.length ? [a, b] : [b, a]
      window = shorter.length

      best = 0.0
      (0..(longer.length - window)).each do |i|
        slice = longer[i, window]
        distance = Levenshtein.distance(shorter, slice)
        score = 1.0 - (distance.to_f / window)
        best = score if score > best
        break if best >= 1.0
      end
      best
    end

    # Token-sort ratio: sort tokens alphabetically, then compute Jaro-Winkler similarity
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Float] similarity between 0.0 and 1.0
    def self.token_sort_ratio(str_a, str_b)
      a = str_a.to_s.downcase.split.sort.join(' ')
      b = str_b.to_s.downcase.split.sort.join(' ')
      JaroWinkler.similarity(a, b)
    end

    # Token-set ratio: compare token sets using Jaccard-like string similarity
    #
    # @param str_a [String]
    # @param str_b [String]
    # @return [Float] similarity between 0.0 and 1.0
    def self.token_set_ratio(str_a, str_b)
      tokens_a = str_a.to_s.downcase.split.uniq.sort
      tokens_b = str_b.to_s.downcase.split.uniq.sort

      return 1.0 if tokens_a == tokens_b
      return 0.0 if tokens_a.empty? && tokens_b.empty?

      intersection = tokens_a & tokens_b
      combined = (tokens_a | tokens_b).sort

      common = intersection.join(' ')
      rest_a = (tokens_a - intersection).sort.join(' ')
      rest_b = (tokens_b - intersection).sort.join(' ')

      joined_a = [common, rest_a].reject(&:empty?).join(' ')
      joined_b = [common, rest_b].reject(&:empty?).join(' ')
      joined_full = combined.join(' ')

      scores = [
        JaroWinkler.similarity(joined_a, joined_b),
        JaroWinkler.similarity(common, joined_a),
        JaroWinkler.similarity(common, joined_b),
        JaroWinkler.similarity(common, joined_full)
      ]

      scores.compact.max || 0.0
    end

    # Weighted score combining multiple algorithms
    #
    # @param str_a [String]
    # @param str_b [String]
    # @param weights [Hash] algorithm => weight pairs (must sum to 1.0)
    # @return [Float] weighted similarity score
    # @raise [Error] if weights do not sum to 1.0
    def self.weighted_score(str_a, str_b, weights:)
      sum = weights.values.reduce(0.0, :+)
      raise Error, "Weights must sum to 1.0, got #{sum}" unless (sum - 1.0).abs < 1e-9

      algorithm_map = {
        jaro_winkler: ->(a, b) { jaro_winkler(a, b) },
        dice: ->(a, b) { dice_coefficient(a, b) },
        levenshtein_ratio: ->(a, b) { ratio(a, b) },
        lcs_ratio: ->(a, b) { lcs_ratio(a, b) },
        damerau_ratio: ->(a, b) { damerau_ratio(a, b) }
      }

      weights.reduce(0.0) do |total, (algo, weight)|
        fn = algorithm_map[algo]
        raise Error, "Unknown algorithm: #{algo}" unless fn

        total + (fn.call(str_a, str_b) * weight)
      end
    end

    # Hybrid text + phonetic similarity for every candidate.
    #
    # Each candidate is scored from two components:
    # - +:text+     — a string similarity score (default Levenshtein +#ratio+).
    #   Unavailable when either side is empty.
    # - +:phonetic+ — 1.0 when both sides produce a usable phonetic code and the
    #   codes match, 0.0 when both codes exist but differ. A missing/empty code
    #   means the phonetic component is *unavailable*, not zero.
    #
    # Only available components participate. Their configured weights are
    # renormalized over the components that actually have values, so a candidate
    # without a usable phonetic code is scored on text alone at the text weight's
    # full effective strength (e.g. text 0.8 with weights 0.6/0.4 stays 0.8).
    # A candidate with a real phonetic mismatch keeps its phonetic 0.0 in the
    # mix (e.g. text 0.8, phonetic 0.0 → 0.8*0.6 + 0.0*0.4 = 0.48).
    #
    # Candidates with no usable component at all (e.g. an empty/nil value that
    # has neither text nor a phonetic code) are returned separately in
    # +:unscored+ (status +:unscored+) and never take part in the ranking.
    # Scored entries are sorted by score descending, ties broken by original
    # input order (stable), matching the existing same-score convention.
    #
    # @param query [String]
    # @param candidates [Array<String>]
    # @param text_weight [Float] weight of the text component (default 0.6)
    # @param phonetic_weight [Float] weight of the phonetic component (default 0.4)
    # @param text_algorithm [Symbol] :jaro_winkler, :dice, or :levenshtein (default)
    # @param phonetic_algorithm [Symbol] :soundex (default) or :metaphone
    # @return [Hash] +{ results:, unscored:, weights: }+
    # @raise [Error] when a weight is negative or an algorithm is unknown
    def self.hybrid_match(query, candidates,
                          text_weight: 0.6, phonetic_weight: 0.4,
                          text_algorithm: :levenshtein, phonetic_algorithm: :soundex)
      weights = { text: text_weight.to_f, phonetic: phonetic_weight.to_f }
      validate_hybrid_weights(weights)

      results = []
      unscored = []

      candidates.each_with_index do |candidate, index|
        entry = hybrid_entry(query, candidate.to_s, index, weights, text_algorithm, phonetic_algorithm)
        if entry[:status] == :unscored
          unscored << entry
        else
          results << entry
        end
      end

      {
        results: results.sort_by { |entry| [-entry[:score], entry[:index]] },
        unscored: unscored,
        weights: weights
      }
    end

    # Group and deduplicate similar strings
    #
    # @param array [Array<String>] strings to deduplicate
    # @param threshold [Float] similarity threshold (0.0 to 1.0)
    # @param algorithm [Symbol] :jaro_winkler (default), :dice, or :levenshtein
    # @return [Array<String>] unique representatives
    def self.deduplicate(array, threshold: 0.8, algorithm: :jaro_winkler)
      representatives = []
      array.each do |item|
        duplicate = representatives.any? do |rep|
          score_for(rep, item, algorithm: algorithm) >= threshold
        end
        representatives << item unless duplicate
      end
      representatives
    end

    def self.similarity_matrix(strings, algorithm: :jaro_winkler, threshold: nil)
      matrix = {}
      strings.each do |a|
        matrix[a] = {}
        strings.each do |b|
          score = score_for(a, b, algorithm: algorithm).round(4)
          matrix[a][b] = score if threshold.nil? || score >= threshold
        end
      end
      matrix
    end

    # Dispatch a similarity score for the given algorithm symbol.
    #
    # @param str_a [String]
    # @param str_b [String]
    # @param algorithm [Symbol] :jaro_winkler, :dice, or :levenshtein
    # @return [Float] similarity between 0.0 and 1.0
    # @raise [Error] when the algorithm is not supported
    def self.score_for(str_a, str_b, algorithm:)
      case algorithm
      when :jaro_winkler then jaro_winkler(str_a, str_b)
      when :dice then dice_coefficient(str_a, str_b)
      when :levenshtein then ratio(str_a, str_b)
      else raise Error, "Unknown algorithm: #{algorithm}"
      end
    end

    # Build one hybrid-match entry, deciding which components actually
    # participate and renormalizing their weights over the available ones.
    #
    # @return [Hash] a scored entry, or an +:unscored+ entry when no component
    #   has a usable value
    def self.hybrid_entry(query, candidate, index, weights, text_algorithm, phonetic_algorithm)
      base = {
        value: candidate,
        index: index,
        components: [],
        component_scores: {},
        applied_weights: {}
      }

      if text_available?(query, candidate)
        add_component(base, :text, score_for(query, candidate, algorithm: text_algorithm), weights[:text])
      end

      phonetic = phonetic_component(query, candidate, phonetic_algorithm)
      add_component(base, :phonetic, phonetic[:score], weights[:phonetic]) if phonetic[:available]

      return base.merge(status: :unscored, score: nil) if base[:components].empty?

      total_weight = base[:components].sum { |name| base[:applied_weights][name] }
      score = base[:components].sum do |name|
        base[:component_scores][name] * base[:applied_weights][name]
      end / total_weight

      base.merge(status: :scored, score: score.round(4))
    end

    # Record a participating component plus its configured weight.
    def self.add_component(entry, name, score, weight)
      entry[:components] << name
      entry[:component_scores][name] = score
      entry[:applied_weights][name] = weight
    end

    # The text component needs non-empty content on both sides.
    def self.text_available?(query, candidate)
      !query.to_s.empty? && !candidate.to_s.empty?
    end

    # Phonetic component: unavailable when either side lacks a usable code,
    # otherwise 1.0 for equal codes and 0.0 for differing codes.
    #
    # @return [Hash] +{ available:, score: }+
    def self.phonetic_component(query, candidate, algorithm)
      coder = case algorithm
              when :soundex then ->(string) { Soundex.code(string) }
              when :metaphone then ->(string) { Metaphone.code(string) }
              else raise Error, "Unknown phonetic algorithm: #{algorithm}"
              end

      query_code = coder.call(query)
      candidate_code = coder.call(candidate)
      return { available: false, score: nil } if query_code.empty? || candidate_code.empty?

      { available: true, score: query_code == candidate_code ? 1.0 : 0.0 }
    end

    # Reject negative hybrid weights; an all-zero configuration cannot be
    # renormalized over any participating component.
    def self.validate_hybrid_weights(weights)
      weights.each do |name, weight|
        raise Error, "Weight for #{name} cannot be negative, got #{weight}" if weight.negative?
      end

      return unless weights.values.all?(&:zero?)

      raise Error, 'At least one hybrid weight must be positive'
    end

    private_class_method :score_for, :hybrid_entry, :add_component,
                         :text_available?, :phonetic_component, :validate_hybrid_weights
  end
end
