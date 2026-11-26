# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Philiprehberger::FuzzyMatch do
  describe '.clusters' do
    context 'with a chain of passing pairs (Levenshtein, 0.5)' do
      let(:input) { %w[cat bat bag] }
      let(:clusters) { described_class.clusters(input, threshold: 0.5, algorithm: :levenshtein) }

      it 'keeps all three in one connected cluster' do
        expect(clusters.length).to eq(1)
        expect(clusters.first[:members]).to eq(%w[bag bat cat])
      end

      it 'picks the lexicographically first member as the representative' do
        expect(clusters.first[:representative]).to eq('bag')
      end

      it 'routes bag -> cat through bat even though cat and bag are below threshold' do
        expect(clusters.first[:paths]['cat'][:path]).to eq(%w[bag bat cat])
        expect(described_class.ratio('cat', 'bag')).to be < 0.5
      end

      it 'reports the weakest edge on every path' do
        paths = clusters.first[:paths]
        expect(paths['bag']).to eq(path: %w[bag], weakest: 1.0)
        expect(paths['bat'][:weakest]).to be_within(1e-9).of(2.0 / 3)
        expect(paths['cat'][:weakest]).to be_within(1e-9).of(2.0 / 3)
      end

      it 'only uses edges that actually reach the threshold' do
        clusters.first[:paths].each_value do |entry|
          entry[:path].each_cons(2) do |a, b|
            expect(described_class.ratio(a, b)).to be >= 0.5
          end
        end
      end

      it 'is unchanged when the input is reversed' do
        reversed = described_class.clusters(input.reverse, threshold: 0.5, algorithm: :levenshtein)
        expect(reversed).to eq(clusters)
      end
    end

    context 'with a short weak edge and two stronger detours (Levenshtein, 0.5)' do
      let(:input) { %w[bbaa baaa aaaa abaa] }
      let(:clusters) { described_class.clusters(input, threshold: 0.5, algorithm: :levenshtein) }

      it 'puts all four members in one cluster represented by aaaa' do
        expect(clusters.length).to eq(1)
        expect(clusters.first[:representative]).to eq('aaaa')
        expect(clusters.first[:members]).to eq(%w[aaaa abaa baaa bbaa])
      end

      it 'prefers the stronger two-step path over the barely-passing direct edge' do
        entry = clusters.first[:paths]['bbaa']
        expect(described_class.ratio('aaaa', 'bbaa')).to eq(0.5)
        expect(entry[:path]).to eq(%w[aaaa abaa bbaa])
        expect(entry[:weakest]).to eq(0.75)
      end

      it 'breaks the equal-strength tie by comparing the names on the path' do
        # aaaa -> baaa -> bbaa is equally strong and equally long, but abaa sorts first
        expect(described_class.ratio('aaaa', 'baaa')).to eq(0.75)
        expect(described_class.ratio('baaa', 'bbaa')).to eq(0.75)
        expect(clusters.first[:paths]['bbaa'][:path]).not_to include('baaa')
      end

      it 'reports the same cluster, representative, and path for reversed input' do
        reversed = described_class.clusters(input.reverse, threshold: 0.5, algorithm: :levenshtein)
        expect(reversed).to eq(clusters)
      end

      it 'reports the same result for every permutation of the input' do
        input.permutation.each do |permutation|
          expect(described_class.clusters(permutation, threshold: 0.5, algorithm: :levenshtein)).to eq(clusters)
        end
      end
    end

    context 'when a direct edge ties a longer detour' do
      it 'prefers the path with fewer edges' do
        clusters = described_class.clusters(%w[abc abd abe], threshold: 0.5, algorithm: :levenshtein)
        entry = clusters.first[:paths]['abe']
        expect(entry[:path]).to eq(%w[abc abe])
        expect(entry[:weakest]).to be_within(1e-9).of(2.0 / 3)
      end
    end

    context 'with several disconnected clusters' do
      let(:input) { %w[world wrld hello helo] }
      let(:clusters) { described_class.clusters(input, threshold: 0.8, algorithm: :levenshtein) }

      it 'returns one cluster per connected component, ordered by representative' do
        expect(clusters.map { |cluster| cluster[:representative] }).to eq(%w[hello world])
        expect(clusters.map { |cluster| cluster[:members] }).to eq([%w[hello helo], %w[world wrld]])
      end

      it 'gives every member of every cluster a path from its representative' do
        clusters.each do |cluster|
          cluster[:members].each do |member|
            path = cluster[:paths][member][:path]
            expect(path.first).to eq(cluster[:representative])
            expect(path.last).to eq(member)
          end
        end
      end
    end

    context 'with repeated and empty input' do
      it 'collapses repeated values into a single vertex' do
        clusters = described_class.clusters(%w[cat cat bat], threshold: 0.5, algorithm: :levenshtein)
        expect(clusters.length).to eq(1)
        expect(clusters.first[:members]).to eq(%w[bat cat])
      end

      it 'returns an empty array for empty input' do
        expect(described_class.clusters([], threshold: 0.5)).to eq([])
      end

      it 'treats a lone string as a one-member cluster' do
        expected = [{ representative: 'solo', members: %w[solo],
                      paths: { 'solo' => { path: %w[solo], weakest: 1.0 } } }]
        expect(described_class.clusters(%w[solo], threshold: 0.5)).to eq(expected)
      end
    end

    context 'with algorithm selection' do
      it 'uses the requested algorithm for the edges' do
        clusters = described_class.clusters(%w[night nacht], threshold: 0.2, algorithm: :dice)
        expect(clusters.length).to eq(1)
        expect(clusters.first[:paths]['night'][:weakest])
          .to eq(described_class.dice_coefficient('night', 'nacht'))
      end

      it 'defaults to jaro_winkler' do
        clusters = described_class.clusters(%w[cat bat], threshold: 0.5)
        expect(clusters.length).to eq(1)
        expect(clusters.first[:paths]['cat'][:weakest]).to eq(described_class.jaro_winkler('bat', 'cat'))
      end
    end
  end

  describe '.deduplicate with :cluster' do
    it 'returns the connected clusters instead of the greedy representatives' do
      clustered = described_class.deduplicate(%w[cat bat bag], threshold: 0.5, algorithm: :levenshtein, cluster: true)
      direct = described_class.clusters(%w[cat bat bag], threshold: 0.5, algorithm: :levenshtein)
      expect(clustered).to eq(direct)
    end

    it 'keeps the greedy representative array by default' do
      expect(described_class.deduplicate(%w[cat bat bag], threshold: 0.5, algorithm: :levenshtein)).to eq(%w[cat bag])
    end

    it 'keeps the greedy return type and input order for cluster: false' do
      result = described_class.deduplicate(%w[cat bat bag], threshold: 0.5, algorithm: :levenshtein, cluster: false)
      expect(result).to eq(%w[cat bag])
      expect(result).to all(be_a(String))
    end

    it 'honours threshold and algorithm in cluster mode' do
      loose = described_class.deduplicate(%w[cat bat bag], threshold: 0.3, algorithm: :levenshtein, cluster: true)
      tight = described_class.deduplicate(%w[cat bat bag], threshold: 0.7, algorithm: :levenshtein, cluster: true)
      expect(loose.first[:members]).to eq(%w[bag bat cat])
      expect(tight.map { |cluster| cluster[:members] }).to eq([%w[bag], %w[bat], %w[cat]])
    end

    it 'returns an empty array for empty input' do
      expect(described_class.deduplicate([], cluster: true)).to eq([])
    end

    it 'accepts connected: true as an option alias' do
      connected = described_class.deduplicate(%w[cat bat bag], threshold: 0.5, algorithm: :levenshtein, connected: true)
      clustered = described_class.clusters(%w[cat bat bag], threshold: 0.5, algorithm: :levenshtein)
      expect(connected).to eq(clustered)
    end
  end

  describe 'connected grouping aliases and readable witnesses' do
    let(:groups) do
      described_class.connected_groups(%w[bbaa baaa aaaa abaa], threshold: 0.5, algorithm: :levenshtein)
    end
    let(:group) { groups.first }
    let(:witness) { group[:paths]['bbaa'] }

    it 'exposes common representative and count key aliases' do
      expect(group[:rep]).to eq('aaaa')
      expect(group[:member_count]).to eq(4)
      expect(group[:weakest_edge_scores]['bbaa']).to eq(0.75)
    end

    it 'exposes score aliases and edge-count metadata on a witness' do
      expect(witness[:weak]).to eq(0.75)
      expect(witness[:weakest_edge_score]).to eq(0.75)
      expect(witness[:edge_count]).to eq(2)
      expect(witness[:edges].map { |edge| [edge[:from], edge[:to]] })
        .to eq([%w[aaaa abaa], %w[abaa bbaa]])
    end

    it 'accepts symbol member keys and behaves like its path array when indexed' do
      expect(group[:paths][:bbaa]).to eq(witness)
      expect(witness).to eq(%w[aaaa abaa bbaa])
      expect(witness[1]).to eq('abaa')
      expect(witness).to include('abaa')
    end

    it 'provides method aliases for the main entry point' do
      expect(described_class.deduplicate_connected(%w[cat], threshold: 0.5)).to eq(described_class.clusters(%w[cat], threshold: 0.5))
      expect(described_class.connected_clusters(%w[cat], threshold: 0.5)).to eq(described_class.clusters(%w[cat], threshold: 0.5))
    end

    it 'returns the flat four-field acceptance witness' do
      report = described_class.connection_report(
        %w[bbaa baaa aaaa abaa], 'bbaa', threshold: 0.5, algorithm: :levenshtein
      )
      expect(report.keys).to eq(%i[rep path weak member_count])
      expect(report).to include(
        rep: 'aaaa',
        path: %w[aaaa abaa bbaa],
        weak: 0.75,
        member_count: 4
      )
    end
  end
end
