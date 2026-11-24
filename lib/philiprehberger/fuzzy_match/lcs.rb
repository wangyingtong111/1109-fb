# frozen_string_literal: true

module Philiprehberger
  module FuzzyMatch
    # Longest Common Subsequence algorithm with O(min(n,m)) space.
    #
    # Alignment is computed over grapheme clusters, so a decomposed character
    # (for example `"A" + U+030A`) lines up with its precomposed form (`"Å"`).
    # Only the case-folded, NFC-normalized cluster is used for the comparison;
    # the ranges handed back to callers always point at the untouched original
    # text.
    module Lcs
      # Returns the length of the longest common subsequence
      #
      # @param str_a [String]
      # @param str_b [String]
      # @return [Integer]
      def self.length(str_a, str_b)
        a = comparable_units(str_a)
        b = comparable_units(str_b)

        return 0 if a.empty? || b.empty?

        lcs_length(a, b)
      end

      # Returns normalized LCS similarity (0.0 to 1.0)
      #
      # @param str_a [String]
      # @param str_b [String]
      # @return [Float]
      def self.ratio(str_a, str_b)
        a = comparable_units(str_a)
        b = comparable_units(str_b)
        total = a.length + b.length

        return 1.0 if total.zero?

        (2.0 * lcs_length(a, b)) / total
      end

      # Returns the longest common subsequence as paired ranges into the
      # original strings.
      #
      # Each pair is a Hash with `:a_range` / `:b_range` (inclusive character
      # ranges into `str_a` / `str_b`) plus the `:a_text` / `:b_text` slices
      # they select. Slicing the original string with a returned range always
      # yields one complete grapheme cluster, even when the cluster spans
      # several code points (a decomposed `"Å"` reports `0..1`).
      #
      # The pairs are strictly increasing on both sides and, taken together,
      # spell a subsequence of the same length as {.length}.
      #
      # Repeated symbols are resolved by one fixed convention, so the same
      # input always produces the same positions:
      #
      # 1. the two sides are put in a canonical order (fewer clusters first,
      #    equal counts compared by their folded units);
      # 2. the longest common prefix and then the longest common suffix are
      #    matched greedily;
      # 3. the remaining middle is split at the earliest boundary that keeps
      #    the optimum, down to a single cluster, where the first matching
      #    occurrence is taken.
      #
      # Swapping the arguments therefore mirrors the pairs instead of picking
      # a different equally long path.
      #
      # The scoring table is never materialized: split points come from
      # rolling LCS rows (Hirschberg), so peak memory stays O(n + m).
      #
      # @param str_a [String]
      # @param str_b [String]
      # @return [Array<Hash>] `:a_range`, `:b_range`, `:a_text`, `:b_text`
      def self.alignment(str_a, str_b)
        units_a, ranges_a = units_with_ranges(str_a)
        units_b, ranges_b = units_with_ranges(str_b)

        index_pairs =
          if canonical_swap?(units_a, units_b)
            hirschberg(units_b, units_a, 0, units_b.length, 0, units_a.length, []).map { |i, j| [j, i] }
          else
            hirschberg(units_a, units_b, 0, units_a.length, 0, units_b.length, [])
          end

        index_pairs.map do |index_a, index_b|
          range_a = ranges_a[index_a]
          range_b = ranges_b[index_b]
          {
            a_range: range_a,
            b_range: range_b,
            a_text: str_a.to_s[range_a],
            b_text: str_b.to_s[range_b]
          }
        end
      end

      # Whether the canonical order puts `units_b` first.
      #
      # @return [Boolean]
      def self.canonical_swap?(units_a, units_b)
        return true if units_a.length > units_b.length
        return false if units_a.length < units_b.length

        (units_a <=> units_b) == 1
      end
      private_class_method :canonical_swap?

      # Rolling LCS length over two comparable-unit arrays. Keeps two rows of
      # O(min(n, m)) integers.
      #
      # @param a [Array<String>]
      # @param b [Array<String>]
      # @return [Integer]
      def self.lcs_length(a, b)
        # Ensure a is the shorter array for O(min(n,m)) space
        a, b = b, a if a.length > b.length

        prev = Array.new(a.length + 1, 0)
        curr = Array.new(a.length + 1, 0)

        b.each do |cb|
          a.each_with_index do |ca, j|
            curr[j + 1] = if ca == cb
                            prev[j] + 1
                          else
                            [curr[j], prev[j + 1]].max
                          end
          end
          prev, curr = curr, prev
          curr.fill(0)
        end

        prev[a.length]
      end
      private_class_method :lcs_length

      # Splits a string into grapheme clusters, returning the comparison units
      # (case-folded, NFC-normalized) alongside the inclusive character range
      # of every cluster in the original string.
      #
      # @param str [Object]
      # @return [Array(Array<String>, Array<Range>)]
      def self.units_with_ranges(str)
        clusters = str.to_s.each_grapheme_cluster.to_a
        units = clusters.map { |cluster| comparable_unit(cluster) }

        offset = 0
        ranges = clusters.map do |cluster|
          range = offset..(offset + cluster.length - 1)
          offset += cluster.length
          range
        end

        [units, ranges]
      end
      private_class_method :units_with_ranges

      # @param str [Object]
      # @return [Array<String>]
      def self.comparable_units(str)
        str.to_s.each_grapheme_cluster.map { |cluster| comparable_unit(cluster) }
      end
      private_class_method :comparable_units

      # Case-folds a cluster and, where the encoding supports it, NFC-normalizes
      # it so a decomposed glyph compares equal to its precomposed form.
      # Encodings without Unicode normalization support (binary, legacy code
      # pages) fall back to the folded cluster instead of raising.
      #
      # @param cluster [String]
      # @return [String]
      def self.comparable_unit(cluster)
        folded = cluster.downcase(:fold)
        begin
          folded.unicode_normalize(:nfc)
        rescue ArgumentError, Encoding::CompatibilityError
          folded
        end
      end
      private_class_method :comparable_unit

      # Linear-space (Hirschberg) reconstruction over the slice
      # `units_a[lo_a...hi_a]` / `units_b[lo_b...hi_b]`. Appends
      # `[index_a, index_b]` pairs in strictly increasing order.
      #
      # Only rolling score rows of O(m) are live, and each level drops its rows
      # before recursing; the O(n * m) table a plain backtrack would need is
      # never built.
      #
      # @return [Array<Array(Integer, Integer)>]
      def self.hirschberg(units_a, units_b, lo_a, hi_a, lo_b, hi_b, pairs)
        while lo_a < hi_a && lo_b < hi_b && units_a[lo_a] == units_b[lo_b]
          pairs << [lo_a, lo_b]
          lo_a += 1
          lo_b += 1
        end
        return pairs if lo_a >= hi_a || lo_b >= hi_b

        tail = []
        while lo_a < hi_a && lo_b < hi_b && units_a[hi_a - 1] == units_b[hi_b - 1]
          tail << [hi_a - 1, hi_b - 1]
          hi_a -= 1
          hi_b -= 1
        end

        resolve_middle(units_a, units_b, lo_a, hi_a, lo_b, hi_b, pairs) if lo_a < hi_a && lo_b < hi_b

        tail.reverse_each { |pair| pairs << pair }
        pairs
      end
      private_class_method :hirschberg

      # @return [void]
      def self.resolve_middle(units_a, units_b, lo_a, hi_a, lo_b, hi_b, pairs)
        if hi_a - lo_a == 1
          index = first_index(units_b, lo_b, hi_b, units_a[lo_a])
          pairs << [lo_a, index] if index
        elsif hi_b - lo_b == 1
          index = first_index(units_a, lo_a, hi_a, units_b[lo_b])
          pairs << [index, lo_b] if index
        else
          mid = lo_a + ((hi_a - lo_a) / 2)
          left = prefix_scores(units_a, lo_a, mid, units_b, lo_b, hi_b)
          right = suffix_scores(units_a, mid, hi_a, units_b, lo_b, hi_b)
          split = split_point(left, right)
          hirschberg(units_a, units_b, lo_a, mid, lo_b, lo_b + split, pairs)
          hirschberg(units_a, units_b, mid, hi_a, lo_b + split, hi_b, pairs)
        end
        nil
      end
      private_class_method :resolve_middle

      # First index `i` in `[lo, hi)` where `units[i] == unit`, or `nil`.
      #
      # @return [Integer, nil]
      def self.first_index(units, lo, hi, unit)
        (lo...hi).each { |i| return i if units[i] == unit }
        nil
      end
      private_class_method :first_index

      # Earliest split `k` that keeps the optimum `left[k] + right[k]`. Fixing
      # this direction is what makes repeated symbols resolve to the same
      # positions on every run and in both argument orders.
      #
      # `left` and `right` are both indexed by boundary `0..width`, so every
      # interior cut of the second string is a candidate; scanning upwards and
      # only replacing on a strict improvement keeps the earliest one.
      #
      # @param left [Array<Integer>]
      # @param right [Array<Integer>]
      # @return [Integer]
      def self.split_point(left, right)
        best = -1
        split = 0
        (0...right.length).each do |k|
          score = left[k] + right[k]
          next unless score > best

          best = score
          split = k
        end
        split
      end
      private_class_method :split_point

      # `scores[j - lo_b] == LCS(units_a[lo_a...hi_a], units_b[lo_b...j])`
      # for `j` in `lo_b..hi_b`.
      #
      # @return [Array<Integer>]
      def self.prefix_scores(units_a, lo_a, hi_a, units_b, lo_b, hi_b)
        width = hi_b - lo_b
        row = Array.new(width + 1, 0)
        (lo_a...hi_a).each do |i|
          next_row = Array.new(width + 1, 0)
          (lo_b...hi_b).each do |j|
            next_row[j - lo_b + 1] = if units_a[i] == units_b[j]
                                       row[j - lo_b] + 1
                                     else
                                       [next_row[j - lo_b], row[j - lo_b + 1]].max
                                     end
          end
          row = next_row
        end
        row
      end
      private_class_method :prefix_scores

      # `scores[j - lo_b] == LCS(units_a[lo_a...hi_a], units_b[j...hi_b])`
      # for `j` in `lo_b..hi_b`.
      #
      # @return [Array<Integer>]
      def self.suffix_scores(units_a, lo_a, hi_a, units_b, lo_b, hi_b)
        width = hi_b - lo_b
        row = Array.new(width + 1, 0)
        (hi_a - 1).downto(lo_a) do |i|
          next_row = Array.new(width + 1, 0)
          (hi_b - 1).downto(lo_b) do |j|
            next_row[j - lo_b] = if units_a[i] == units_b[j]
                                   row[j - lo_b + 1] + 1
                                 else
                                   [row[j - lo_b], next_row[j - lo_b + 1]].max
                                 end
          end
          row = next_row
        end
        row
      end
      private_class_method :suffix_scores
    end
  end
end
