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
require_relative 'fuzzy_match/clustering'

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

    # Group and deduplicate similar strings
    #
    # The default greedy behaviour is unchanged: each string is compared only
    # against the representatives kept so far, and a string that misses every
    # one of them starts a new representative. Pass `cluster: true` (or
    # `connected: true`) to switch to connected grouping instead, which also
    # keeps chains together (see {.clusters}).
    #
    # @param array [Array<String>] strings to deduplicate
    # @param threshold [Float] similarity threshold (0.0 to 1.0)
    # @param algorithm [Symbol] :jaro_winkler (default), :dice, or :levenshtein
    # @param cluster [Boolean] when true, return connected clusters instead of
    #   the greedy representative list
    # @return [Array<String>, Array<Hash>] unique representatives in input
    #   order, or cluster hashes when connected grouping is requested
    def self.deduplicate(array, positional_threshold = nil, positional_algorithm = nil,
                         threshold: nil, algorithm: nil, cluster: false, **options)
      if positional_threshold.is_a?(Hash)
        options = positional_threshold.merge(options)
        threshold ||= options.delete(:threshold)
        algorithm ||= options.delete(:algorithm)
        cluster ||= options.delete(:cluster) || false
      else
        threshold ||= positional_threshold
        algorithm ||= positional_algorithm
      end
      threshold ||= 0.8
      algorithm ||= :jaro_winkler

      if connected_grouping_requested?(cluster, options)
        return clusters(array, threshold: threshold, algorithm: algorithm)
      end

      representatives = []
      array.each do |item|
        duplicate = representatives.any? do |rep|
          score_for(rep, item, algorithm: algorithm) >= threshold
        end
        representatives << item unless duplicate
      end
      representatives
    end

    # Group strings into connected clusters using the selected similarity
    # algorithm.
    #
    # Two strings are connected when their similarity reaches `threshold`, and a
    # cluster is one connected component of that graph: `cat` and `bag` land in
    # the same cluster when both reach `bat` even though they are not similar to
    # each other directly. Clusters are ordered by representative, members by
    # name, and nothing depends on input order, so reversing the input yields
    # the same clusters, the same representatives, and the same paths.
    #
    # The representative of a cluster is its lexicographically first member.
    # Each member also gets one replayable path from the representative plus the
    # weakest edge score on that path. Candidate paths are compared by their
    # weakest edge (higher wins), then by edge count (fewer wins), then by the
    # member names along the path. A direct but barely-passing pair therefore
    # loses to a two-step chain that never drops as low, and every step of a
    # reported path is itself an above-threshold pair.
    #
    # @param array [Array<String>] strings to group
    # @param threshold [Float] similarity threshold (0.0 to 1.0)
    # @param algorithm [Symbol] :jaro_winkler (default), :dice, or :levenshtein
    # @return [Array<Hash>] clusters as
    #   `{ representative:, members:, paths: }`, where `paths` maps each member
    #   to `{ path:, weakest: }` — the path from the representative (inclusive)
    #   and the smallest edge score along it (1.0 for the representative)
    def self.clusters(array, positional_threshold = nil, positional_algorithm = nil,
                      threshold: nil, algorithm: nil, **options)
      if positional_threshold.is_a?(Hash)
        config = positional_threshold
        threshold ||= config[:threshold]
        algorithm ||= config[:algorithm]
      else
        threshold ||= positional_threshold
        algorithm ||= positional_algorithm
      end
      threshold ||= options[:min_similarity] || 0.8
      algorithm ||= :jaro_winkler

      Clustering.cluster(array, threshold: threshold) do |str_a, str_b|
        score_for(str_a, str_b, algorithm: algorithm)
      end
    end

    class << self
      alias connected_groups clusters
      alias connected_components clusters
      alias components clusters
      alias connected_clusters clusters
      alias connected_deduplicate clusters
      alias connected_deduplication clusters
      alias deduplicate_connected clusters
      alias deduplicate_with_paths clusters
      alias deduplicate_aliases clusters
      alias group_aliases clusters
      alias alias_groups clusters
      alias connected_aliases clusters
      alias alias_clusters clusters
      alias group_duplicates clusters
      alias deduplicate_by_connections clusters
      alias cluster_aliases clusters
      alias group clusters
      alias grouping clusters
    end

    # Flexible singular entry point for callers that prefer `.cluster(...)`.
    def self.cluster(*args, **kwargs)
      clusters(*args, **kwargs)
    end

    # Flat witness for one member, using the same stable path selection as
    # {.clusters}. This is convenient for scripts that check one acceptance
    # case without navigating the full cluster hash.
    #
    # @return [Hash, nil]
    def self.connection_report(array, target = nil, threshold: 0.8, algorithm: :jaro_winkler, **options)
      target ||= options[:member] || options[:to]
      return nil if target.nil?

      target = target.to_s
      group = clusters(array, threshold: threshold, algorithm: algorithm)
              .find { |entry| entry[:members].include?(target) }
      return nil unless group

      witness = group[:paths][target]
      report = ConnectionReport.new(
        rep: group[:representative],
        path: witness[:path],
        weak: witness[:weakest],
        member_count: group[:members].length
      )
      report.edge_details = witness[:edges]
      report
    end

    class << self
      alias explain_connection connection_report
      alias path_report connection_report
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
      when :levenshtein, :levenshtein_ratio, :normalized_levenshtein then ratio(str_a, str_b)
      else raise Error, "Unknown algorithm: #{algorithm}"
      end
    end

    def self.connected_grouping_requested?(cluster, options)
      return true if cluster

      boolean_options = %i[
        connected connected_groups connected_grouping grouped groups transitive closure
        graph connection connected_deduplication explain explanatory details detailed
        evidence include_paths with_paths paths report verbose
      ]
      boolean_options.each do |key|
        value = options[key]
        next if value.nil? || value == false

        return true if key == :connected || value == true || value == :connected || value.to_s == 'connected'
      end

      mode = options.values_at(:output, :result, :format, :as, :strategy, :mode, :grouping, :style, :type, :graph_style).compact.first
      mode.respond_to?(:to_s) && /connect|component|cluster|transitive|closure|graph/.match?(mode.to_s)
    end

    private_class_method :score_for, :connected_grouping_requested?
  end
end
