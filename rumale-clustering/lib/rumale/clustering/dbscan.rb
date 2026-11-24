# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Clustering
    # DBSCAN is a class that implements DBSCAN cluster analysis.
    #
    # @example
    #   require 'rumale/clustering/dbscan'
    #
    #   analyzer = Rumale::Clustering::DBSCAN.new(eps: 0.5, min_samples: 5)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Ester, M., Kriegel, H-P., Sander, J., and Xu, X., "A density-based algorithm for discovering clusters in large spatial databases with noise," Proc. KDD' 96, pp. 266--231, 1996.
    class DBSCAN < ::Rumale::Base::Estimator
      include ::Rumale::Base::ClusterAnalyzer

      # Label value of a sample that has not been visited yet.
      UNVISITED = -2
      # Label value of a sample that is noise.
      NOISE = -1
      private_constant :UNVISITED, :NOISE

      # Return the indices of the core samples.
      # A core sample is a sample whose eps-neighborhood contains at least min_samples samples,
      # so border samples and noise samples are not included.
      # @return [Numo::Int32] (shape: [n_core_samples])
      attr_reader :core_sample_ids

      # Return the cluster labels. The negative cluster label indicates that the point is noise.
      # @return [Numo::Int32] (shape: [n_samples])
      attr_reader :labels

      # Create a new cluster analyzer with DBSCAN method.
      #
      # @param eps [Float] The radius of neighborhood.
      # @param min_samples [Integer] The number of neighbor samples to be used for the criterion whether a point is a core point.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      def initialize(eps: 0.5, min_samples: 5, metric: 'euclidean')
        super()
        @params = {
          eps: eps,
          min_samples: min_samples,
          metric: (metric == 'precomputed' ? 'precomputed' : 'euclidean')
        }
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> DBSCAN
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #     If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      #   @return [DBSCAN] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        raise ArgumentError, 'the input distance matrix should be square' if check_invalid_array_shape?(x)

        partial_fit(x)
        self
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        raise ArgumentError, 'the input distance matrix should be square' if check_invalid_array_shape?(x)

        partial_fit(x)
        labels
      end

      private

      def check_invalid_array_shape?(x)
        @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
      end

      def partial_fit(x)
        metric_mat = calc_pairwise_metrics(x)
        n_samples = metric_mat.shape[0]
        @core_sample_ids = []
        @labels = Numo::Int32.zeros(n_samples) + UNVISITED
        cluster_id = -1
        n_samples.times do |query_id|
          next unless @labels[query_id] == UNVISITED

          neighbor_ids = region_query(metric_mat[query_id, true])
          if neighbor_ids.size < @params[:min_samples]
            @labels[query_id] = NOISE
          else
            cluster_id += 1
            @core_sample_ids.push(query_id)
            @labels[query_id] = cluster_id
            expand_cluster(metric_mat, neighbor_ids, cluster_id)
          end
        end
        @core_sample_ids = Numo::Int32.cast(@core_sample_ids.sort)
        nil
      end

      def calc_pairwise_metrics(x)
        @params[:metric] == 'precomputed' ? x : ::Rumale::PairwiseMetric.euclidean_distance(x)
      end

      # Assign the samples reachable from the core samples of a cluster.
      # Only samples that have not been assigned to any cluster yet are labeled,
      # so a border sample shared by multiple clusters stays in the first cluster that reaches it
      # and is never used as a path to another cluster.
      def expand_cluster(metric_mat, neighbor_ids, cluster_id)
        until neighbor_ids.empty?
          sample_id = neighbor_ids.shift
          if @labels[sample_id] == NOISE
            @labels[sample_id] = cluster_id
          elsif @labels[sample_id] == UNVISITED
            @labels[sample_id] = cluster_id
            next_ids = region_query(metric_mat[sample_id, true])
            next if next_ids.size < @params[:min_samples]

            @core_sample_ids.push(sample_id)
            neighbor_ids.concat(next_ids.select { |id| @labels[id] <= NOISE })
          end
        end
      end

      def region_query(metric_arr)
        metric_arr.lt(@params[:eps]).where.to_a
      end
    end
  end
end
