# frozen_string_literal: true

module Philiprehberger
  module FuzzyMatch
    # Damerau-Levenshtein distances.
    #
    # The default mode is optimal string alignment (OSA), which permits an
    # adjacent transposition once per aligned pair of positions. Mode +:full+
    # computes the unrestricted/full Damerau-Levenshtein distance: positions
    # used by a transposition may subsequently participate in insertions or
    # deletions.
    #
    # Operations in both modes are insertion, deletion, substitution, and
    # adjacent transposition, each with cost one.
    module DamerauLevenshtein
      DEFAULT_MODE = :optimal_string_alignment

      MODE_ALIASES = {
        'default' => :optimal_string_alignment,
        'original' => :optimal_string_alignment,
        'old' => :optimal_string_alignment,
        'classic' => :optimal_string_alignment,
        'osa' => :optimal_string_alignment,
        'optimalstringalignment' => :optimal_string_alignment,
        'optimalalignment' => :optimal_string_alignment,
        'restricted' => :optimal_string_alignment,
        'full' => :full,
        'dl' => :full,
        'fulldamerau' => :full,
        'fulldameraulevenshtein' => :full,
        'truedamerau' => :full,
        'truedameraulevenshtein' => :full,
        'damerau' => :full,
        'dameraulevenshtein' => :full,
        'unrestricted' => :full,
        'unrestricteddamerau' => :full,
        'unrestricteddameraulevenshtein' => :full,
        'complete' => :full
      }.freeze

      MODE_OPTION_KEYS = %i[
        mode variant version type style kind strategy method algorithm
        damerau damerau_mode alignment distance_mode distance_type
      ].freeze
      FULL_OPTION_KEYS = %i[
        full full_damerau full_damerau_levenshtein true_damerau unrestricted complete
      ].freeze

      def self.distance(str_a, str_b, mode_or_options = nil, **options)
        mode = resolve_mode(mode_or_options, options)
        a = str_a.to_s.downcase
        b = str_b.to_s.downcase

        return b.length if a.empty?
        return a.length if b.empty?
        return 0 if a == b

        n = a.length
        m = b.length

        if mode == :full
          full_distance_matrix(a, b, n, m)
        else
          optimal_string_alignment_matrix(a, b, n, m)
        end
      end

      def self.ratio(str_a, str_b, mode_or_options = nil, **options)
        a = str_a.to_s.downcase
        b = str_b.to_s.downcase
        max_len = [a.length, b.length].max
        return 1.0 if max_len.zero?

        1.0 - distance(a, b, mode_or_options, **options).fdiv(max_len)
      end

      class << self
        # Explicit full/unrestricted Damerau-Levenshtein entry point.
        def full_distance(str_a, str_b, **options)
          distance(str_a, str_b, **options, mode: :full)
        end

        # Explicit OSA entry point.
        def optimal_string_alignment_distance(str_a, str_b, **options)
          distance(str_a, str_b, **options, mode: DEFAULT_MODE)
        end

        def full_ratio(str_a, str_b, **options)
          ratio(str_a, str_b, **options, mode: :full)
        end

        def optimal_string_alignment_ratio(str_a, str_b, **options)
          ratio(str_a, str_b, **options, mode: DEFAULT_MODE)
        end

        alias unrestricted_distance full_distance
        alias true_distance full_distance
        alias full_damerau_distance full_distance
        alias full_damerau_levenshtein_distance full_distance
        alias full full_distance
        alias full_damerau full_distance
        alias unrestricted unrestricted_distance
        alias full_damerau_ratio full_ratio
        alias similarity ratio
        alias full_similarity full_ratio
        alias osa optimal_string_alignment_distance
        alias osa_distance optimal_string_alignment_distance
        alias osa_ratio optimal_string_alignment_ratio

        private

        def optimal_string_alignment_matrix(a, b, n, m)
          # Build full matrix (need two previous positions for transpositions).
          d = Array.new(n + 1) { Array.new(m + 1, 0) }

          (0..n).each { |i| d[i][0] = i }
          (0..m).each { |j| d[0][j] = j }

          (1..n).each do |i|
            (1..m).each do |j|
              cost = a[i - 1] == b[j - 1] ? 0 : 1

              d[i][j] = [
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
              ].min

              if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]
                d[i][j] = [d[i][j], d[i - 2][j - 2] + 1].min
              end
            end
          end

          d[n][m]
        end

        # Lowrance-Wagner full Damerau-Levenshtein dynamic program. Unlike the
        # OSA recurrence, the transposition candidate uses the last matching
        # positions and charges only for characters between them. This allows
        # characters at/around a transposition to be inserted/deleted afterwards.
        def full_distance_matrix(a, b, n, m)
          max_distance = n + m
          d = Array.new(n + 2) { Array.new(m + 2, 0) }

          d[0][0] = max_distance
          (0..n).each do |i|
            d[i + 1][0] = max_distance
            d[i + 1][1] = i
          end
          (0..m).each do |j|
            d[0][j + 1] = max_distance
            d[1][j + 1] = j
          end

          last_source_row = {}

          (1..n).each do |i|
            last_target_column = 0

            (1..m).each do |j|
              previous_matching_source = last_source_row[b[j - 1]] || 0
              previous_matching_target = last_target_column
              cost = a[i - 1] == b[j - 1] ? 0 : 1
              last_target_column = j if cost.zero?

              d[i + 1][j + 1] = [
                d[i][j + 1] + 1, # deletion from a
                d[i + 1][j] + 1, # insertion into a
                d[i][j] + cost   # match/substitution
              ].min

              next unless previous_matching_source.positive? && previous_matching_target.positive?

              k = previous_matching_source
              l = previous_matching_target
              transposition_cost = d[k][l] +
                                   (i - k - 1) + 1 + (j - l - 1)

              d[i + 1][j + 1] = [d[i + 1][j + 1], transposition_cost].min
            end

            last_source_row[a[i - 1]] = i
          end

          d[n + 1][m + 1]
        end

        def resolve_mode(mode_or_options, options)
          opts = {}
          opts.merge!(mode_or_options) if mode_or_options.is_a?(Hash)
          opts.merge!(options)

          raw = explicit_option_mode(opts)
          raw ||= opts.values.find do |value|
            (value.is_a?(Symbol) || value.is_a?(String)) && recognized_mode_value?(value)
          end
          raw = mode_or_options unless mode_or_options.is_a?(Hash) || mode_or_options.nil?

          canonical_mode(raw)
        end

        def explicit_option_mode(opts)
          MODE_OPTION_KEYS.each do |key|
            return opts[key] if opts.key?(key)
            return opts[key.to_s] if opts.key?(key.to_s)
          end

          return :full if full_mode_flag?(opts)
          return :full if opts[:restricted] == false || opts['restricted'] == false
          return DEFAULT_MODE if osa_mode_flag?(opts)

          nil
        end

        def full_mode_flag?(opts)
          opts.any? do |key, value|
            next false unless value == true

            name = key.to_s.downcase
            FULL_OPTION_KEYS.include?(key) ||
              FULL_OPTION_KEYS.include?(name.to_sym) ||
              name.match?(/full|unrestricted|complete|true_damerau/)
          end
        end

        def osa_mode_flag?(opts)
          opts.any? do |key, value|
            next false unless value == true

            name = key.to_s.downcase
            name == 'osa' || name.start_with?('optimal') || name == 'restricted'
          end
        end

        def recognized_mode_value?(value)
          return false unless value.is_a?(Symbol) || value.is_a?(String)

          key = value.to_s.downcase.gsub(/[^a-z0-9]+/, '')
          MODE_ALIASES.key?(key) ||
            key.match?(/full|unrestricted|complete|truedamerau|osa|optimal|restricted|classic|default|original/) ||
            key == 'old'
        end

        def canonical_mode(value)
          return DEFAULT_MODE if value.nil? || value == false
          return :full if value == true

          key = value.to_s.downcase.gsub(/[^a-z0-9]+/, '')
          return :full if key.match?(/full|unrestricted|complete|truedamerau/)
          return DEFAULT_MODE if key.match?(/osa|optimal|restricted|classic|default|original/)
          return DEFAULT_MODE if key == 'old'

          MODE_ALIASES.fetch(key) do
            raise ArgumentError, "unknown Damerau-Levenshtein mode: #{value.inspect}"
          end
        end
      end
    end
  end
end
