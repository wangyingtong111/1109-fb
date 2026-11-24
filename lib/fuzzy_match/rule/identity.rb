class FuzzyMatch
  class Rule
    # Identities take effect when needle and haystack both match a regexp
    # Then the captured part of the regexp has to match exactly
    class Identity < Rule
      attr_reader :proc

      def initialize(regexp_or_proc)
        case regexp_or_proc
        when Regexp
          @regexp = regexp_or_proc
        when Proc
          @proc = regexp_or_proc
        else
          raise ArgumentError, "[FuzzyMatch] Identity must be set with either Regexp objects or Procs, but got #{regexp_or_proc.inspect} (#{regexp_or_proc.class.name})"
        end
      end

      def ==(other)
        other.class == self.class and (regexp ? regexp == other.regexp : proc == other.proc)
      end

      # Two strings are "identical" if they both match this identity and any
      # one complete combination of captures is the same in both strings.
      #
      # A title can contain several article numbers at once (for example a
      # current SKU and a previous SKU), so a string can match the regexp
      # more than once. Every match keeps its captures together as one
      # combination: fields captured by different matches are never split
      # apart and recombined. A group that captures an empty string ("") is
      # kept distinct from a group that did not participate in the match
      # (nil).
      #
      # Returns true if a complete combination is shared, false when both
      # strings match the regexp but none of their combinations are shared,
      # and nil if either string does not match the regexp at all.
      def identical?(record1, record2)
        if regexp
          captures1 = capture_combinations(record1.whole)
          captures2 = capture_combinations(record2.whole)
          if captures1.nil? or captures2.nil?
            nil
          else
            captures1.any? { |combination| captures2.include? combination }
          end
        else
          proc.call record1.original, record2.original
        end
      end

      private

      # Every complete capture-group combination found in a string, one per
      # regexp match, with each capture downcased.
      #
      # Returns nil when the string does not match the regexp at all, so
      # callers can tell "no match" apart from "matched without captures".
      def capture_combinations(str)
        memo = str.scan(regexp).map do |captures|
          # String#scan yields one array of captures per match when the
          # regexp has capture groups, and the matched string itself when
          # it has none, in which case the combination is simply empty.
          captures = [] unless captures.is_a?(::Array)
          captures.map { |capture| capture && capture.downcase }
        end
        memo.empty? ? nil : memo
      end
    end
  end
end
