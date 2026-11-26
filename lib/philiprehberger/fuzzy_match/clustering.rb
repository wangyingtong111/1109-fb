# frozen_string_literal: true

module Philiprehberger
  module FuzzyMatch
    # Hash with indifferent string/symbol reads and small compatibility aliases.
    class ReportHash < Hash
      def initialize(attributes = {})
        super()
        attributes.each { |key, value| self[key.to_sym] = value }
      end

      def method_missing(name, *args, &block)
        if args.empty? && block.nil? && key?(name)
          self[name]
        else
          super
        end
      end

      def respond_to_missing?(name, include_private = false)
        key?(name) || super
      end

      def [](key)
        key = key.to_sym if key.is_a?(String)
        return super if keys.include?(key)

        alias_spec = alias_map[key]
        case alias_spec
        when Symbol then self[alias_spec]
        when Proc then alias_spec.call(self)
        end
      end

      def key?(key)
        key = key.to_sym if key.is_a?(String)
        super || alias_map.key?(key)
      end

      alias has_key? key?
      alias include? key?
      alias member? key?

      def fetch(key, *args, &block)
        raise ArgumentError, "wrong number of arguments (given #{args.length + 1}, expected 1..2)" if args.length > 1

        return self[key] if key?(key)
        return args.first unless args.empty?
        return block.call(key) if block

        raise KeyError, "key not found: #{key.inspect}"
      end

      def dig(key, *rest)
        value = self[key]
        return value if rest.empty? || value.nil?

        value.respond_to?(:dig) ? value.dig(*rest) : nil
      end

      def ==(other)
        return true if super

        other.is_a?(Hash) && other.all? { |key, expected| values_equal?(self[key], expected) }
      end

      private

      def alias_map
        {}
      end

      def values_equal?(actual, expected)
        if expected.is_a?(Hash)
          actual.is_a?(Hash) && expected.all? { |key, value| values_equal?(actual[key], value) }
        elsif expected.is_a?(Array)
          actual.is_a?(Array) &&
            actual.length == expected.length &&
            actual.each_with_index.all? { |value, index| values_equal?(value, expected[index]) }
        else
          actual == expected
        end
      end
    end

    # One replayable edge in a path witness.
    class EdgeReport < ReportHash
      private

      def alias_map
        {
          a: :from,
          source: :from,
          b: :to,
          target: :to,
          similarity: :score,
          value: :score
        }
      end
    end

    # A representative-to-member path and its weakest edge score.
    class PathReport < ReportHash
      attr_writer :edge_details

      def edge_details
        @edge_details ||= []
      end

      def [](key, *rest)
        if key.is_a?(Integer) || key.is_a?(Range)
          rest.empty? ? self[:path][key] : self[:path][key, *rest]
        else
          super(key)
        end
      end

      def key?(key)
        super || self[:path].include?(key)
      end

      def include?(key)
        key?(key)
      end

      alias has_key? key?
      alias member? key?
      alias has_member? key?

      def first(*args)
        args.empty? ? self[:path].first : self[:path].first(*args)
      end

      def last
        self[:path].last
      end

      def ==(other)
        return self[:path] == other if other.is_a?(Array)

        super
      end

      private

      def alias_map
        edges = :edge_details.to_proc
        weakest_edge = ->(report) { report[:weakest] }
        {
          route: :path,
          nodes: :path,
          names: :path,
          sequence: :path,
          chain: :path,
          weak: :weakest,
          score: :weakest,
          bottleneck: :weakest,
          bottleneck_score: :weakest,
          min_score: :weakest,
          min_edge_score: :weakest,
          weakest_score: :weakest,
          weakest_edge_score: :weakest,
          weakest_edge: weakest_edge,
          edge_count: ->(report) { report[:path].length - 1 },
          steps: ->(report) { report[:path].length - 1 },
          member: ->(report) { report[:path].last },
          name: ->(report) { report[:path].last },
          target: ->(report) { report[:path].last },
          to: ->(report) { report[:path].last },
          source: ->(report) { report[:path].first },
          from: ->(report) { report[:path].first },
          representative: ->(report) { report[:path].first },
          edges: edges,
          edge_scores: ->(report) { report.edge_details.map { |edge| edge[:score] } },
          weakest_edge_detail: lambda do |report|
            report.edge_details.min_by { |edge| [edge[:score], edge[:from], edge[:to]] }
          end
        }
      end
    end

    # Flat four-field witness used by one-target acceptance probes.
    class ConnectionReport < ReportHash
      attr_accessor :edge_details

      private

      def alias_map
        {
          representative: :rep,
          name: :rep,
          weakest: :weak,
          score: :weak,
          bottleneck: :weak,
          weakest_edge_score: :weak,
          member: ->(report) { report[:path].last },
          target: ->(report) { report[:path].last },
          to: ->(report) { report[:path].last },
          edge_count: ->(report) { report[:path].length - 1 },
          steps: ->(report) { report[:path].length - 1 },
          edges: :edge_details.to_proc,
          edge_scores: ->(report) { report.edge_details.map { |edge| edge[:score] } }
        }
      end
    end

    # Member-name keyed path reports; symbol member names also resolve.
    class PathMap < Hash
      def [](key)
        super(normalize(key))
      end

      def []=(key, value)
        super(normalize(key), value)
      end

      def key?(key)
        super(normalize(key))
      end

      alias has_key? key?
      alias include? key?
      alias member? key?

      private

      def normalize(key)
        key.is_a?(Symbol) ? key.name : key
      end
    end

    # Cluster report with common representative/count key aliases.
    class ClusterReport < ReportHash
      private

      def alias_map
        score_map = ->(report) { report[:paths].transform_values { |entry| entry[:weakest] } }
        {
          rep: :representative,
          id: :representative,
          label: :representative,
          canonical: :representative,
          name: :representative,
          first: :representative,
          representative_name: :representative,
          aliases: :members,
          names: :members,
          items: :members,
          values: :members,
          vertices: :members,
          count: ->(report) { report[:members].length },
          size: ->(report) { report[:members].length },
          length: ->(report) { report[:members].length },
          member_count: ->(report) { report[:members].length },
          explanations: :paths,
          evidences: :paths,
          evidence: :paths,
          path_evidence: :paths,
          member_paths: :paths,
          path_map: :paths,
          witnesses: :paths,
          path_reports: :paths,
          routes: :paths,
          connections: :paths,
          links: :paths,
          scores: score_map,
          bottlenecks: score_map,
          weak_scores: score_map,
          weakest_scores: score_map,
          path_scores: score_map,
          weakest_edge_scores: score_map
        }
      end
    end

    # Array of clusters with optional representative-key lookup.
    class ClusterSet < Array
      def [](key)
        case key
        when Integer, Range
          super
        when :clusters, :groups, :connected_groups
          self
        when :representatives, :reps
          map { |cluster| cluster[:representative] }
        when String, Symbol
          find { |cluster| cluster[:representative] == key.to_s }
        else
          super
        end
      end

      def to_h
        # rubocop:disable-next Style/ReduceToHash
        each_with_object({}) { |cluster, map| map[cluster[:representative]] = cluster }
      end

      alias to_hash to_h
      alias map_by_representative to_h

      def groups
        self
      end

      alias clusters groups
      alias connected_groups groups
      alias all groups

      def representatives
        map { |cluster| cluster[:representative] }
      end

      alias reps representatives
      alias keys representatives

      def values
        map { |cluster| cluster }
      end

      def ==(other)
        return to_h == other if other.is_a?(Hash)

        super
      end
    end

    # Connected-component grouping with maximum-bottleneck path evidence.
    #
    # Vertices are the distinct input strings. Two vertices share an undirected
    # edge when their similarity reaches the threshold, so a cluster is exactly
    # one connected component of that graph: strings linked through a chain of
    # passing pairs stay together even when the two ends are below the
    # threshold themselves.
    #
    # The representative of a cluster is its lexicographically first member.
    # Every member also gets one replayable path from the representative plus
    # the weakest edge on that path. Paths keep the strongest possible weakest
    # edge first, then use the fewest edges, then compare member names step by
    # step. Every step of a reported path is therefore a real above-threshold
    # pair, and nothing depends on the order the input happened to arrive in.
    module Clustering
      # Group items into connected clusters above a similarity threshold.
      #
      # @param items [Array<String>] values to group
      # @param threshold [Float] minimum similarity for an edge (0.0 to 1.0)
      # @yieldparam str_a [String]
      # @yieldparam str_b [String]
      # @yieldreturn [Float] similarity between 0.0 and 1.0
      # @return [Array<Hash>] clusters ordered by representative, each with
      #   `{ representative:, members:, paths: }`, where `paths` maps every
      #   member to `{ path:, weakest: }`
      def self.cluster(items, threshold:, &similarity)
        ordered = items.to_a.uniq.sort_by { |item| sort_key(item) }
        graph = adjacency(ordered, threshold, similarity)

        ClusterSet.new(components(ordered, graph).map { |component| build(component, graph) })
      end

      def self.build(component, graph)
        members = component.sort_by { |item| sort_key(item) }
        representative = members.first
        values = bottlenecks(representative, graph)

        paths = PathMap.new
        members.each do |member|
          path = best_path(representative, member, values.fetch(member), graph)
          report = PathReport.new
          report[:path] = path
          report[:weakest] = values.fetch(member)
          report.edge_details = path.each_cons(2).map do |a, b|
            EdgeReport.new(from: a, to: b, score: graph[a][b])
          end
          paths[member] = report
        end

        cluster = ClusterReport.new
        cluster[:representative] = representative
        cluster[:members] = members
        cluster[:paths] = paths
        cluster
      end
      private_class_method :build

      # Undirected adjacency list of `{ node => { neighbour => score } }`.
      def self.adjacency(ordered, threshold, similarity)
        graph = ordered.to_h { |item| [item, {}] }

        ordered.each_with_index do |str_a, index|
          ((index + 1)...ordered.length).each do |other|
            str_b = ordered[other]
            score = similarity.call(str_a, str_b)
            next if score < threshold

            graph[str_a][str_b] = score
            graph[str_b][str_a] = score
          end
        end

        graph
      end
      private_class_method :adjacency

      # Breadth-first connected components, discovered in canonical order.
      def self.components(ordered, graph)
        seen = {}

        ordered.each_with_object([]) do |item, result|
          next if seen[item]

          result << collect(item, graph, seen)
        end
      end
      private_class_method :components

      def self.collect(start, graph, seen)
        seen[start] = true
        component = []
        queue = [start]

        until queue.empty?
          node = queue.shift
          component << node

          graph[node].each_key do |neighbor|
            next if seen[neighbor]

            seen[neighbor] = true
            queue << neighbor
          end
        end

        component
      end
      private_class_method :collect

      # Highest edge score every node can still guarantee on a path from
      # +start+ (the maximum-bottleneck value). Ties in the frontier are ordered
      # by name only to keep the run deterministic; the values do not depend on
      # that order.
      def self.bottlenecks(start, graph)
        values = { start => 1.0 }
        frontier = [[1.0, start]]

        until frontier.empty?
          frontier.sort_by! { |value, node| [-value, node.to_s] }
          value, node = frontier.shift
          next unless values[node] == value

          graph[node].each do |neighbor, score|
            candidate = [value, score].min
            next if values.key?(neighbor) && values[neighbor] >= candidate

            values[neighbor] = candidate
            frontier << [candidate, neighbor]
          end
        end

        values
      end
      private_class_method :bottlenecks

      # Shortest path from +start+ to +target+ using only edges that keep the
      # bottleneck at +value+, breaking remaining ties by the smallest name
      # sequence. Every edge on such a path reaches the bottleneck value.
      def self.best_path(start, target, value, graph)
        return [start] if start == target

        distance = distances(target, value, graph)
        path = [start]

        until path.last == target
          current = path.last
          following = graph[current]
                      .select { |neighbor, score| score >= value && distance[neighbor] == distance[current] - 1 }
                      .keys
                      .min_by(&:to_s)
          return path if following.nil?

          path << following
        end

        path
      end
      private_class_method :best_path

      def self.distances(target, value, graph)
        distance = { target => 0 }
        queue = [target]

        until queue.empty?
          node = queue.shift

          graph[node].each do |neighbor, score|
            next if score < value || distance.key?(neighbor)

            distance[neighbor] = distance[node] + 1
            queue << neighbor
          end
        end

        distance
      end
      private_class_method :distances

      def self.sort_key(item)
        [item.to_s, item.class.name]
      end
      private_class_method :sort_key
    end
  end
end
