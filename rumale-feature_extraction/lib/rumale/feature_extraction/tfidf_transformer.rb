# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/utils'

module Rumale
  module FeatureExtraction
    # Transform sample matrix with term frequecy (tf) to a normalized tf-idf (inverse document frequency) reprensentation.
    #
    # @example
    #   require 'rumale/feature_extraction/hash_vectorizer'
    #   require 'rumale/feature_extraction/tfidf_transformer'
    #
    #   encoder = Rumale::FeatureExtraction::HashVectorizer.new
    #   x = encoder.fit_transform([
    #     { foo: 1, bar: 2 },
    #     { foo: 3, baz: 1 }
    #   ])
    #
    #   # > pp x
    #   # Numo::DFloat#shape=[2,3]
    #   # [[2, 0, 1],
    #   #  [0, 1, 3]]
    #
    #   transformer = Rumale::FeatureExtraction::TfidfTransformer.new
    #   x_tfidf = transformer.fit_transform(x)
    #
    #   # > pp x_tfidf
    #   # Numo::DFloat#shape=[2,3]
    #   # [[0.959056, 0, 0.283217],
    #   #  [0, 0.491506, 0.870874]]
    #
    # *Reference*
    # - Manning, C D., Raghavan, P., and Schutze, H., "Introduction to Information Retrieval," Cambridge University Press., 2008.
    # rubocop:disable Metrics/ClassLength
    class TfidfTransformer < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Raised when a batch identifier is replayed with different content.
      class BatchConflictError < ::ArgumentError
        # @return [String]
        attr_reader :batch_id
        # @return [Hash]
        attr_reader :snapshot

        # @!visibility private
        def initialize(message, batch_id:, snapshot:)
          super(message)
          @batch_id = batch_id
          @snapshot = snapshot
        end

        # @return [Array<Float>, nil] The last usable IDF preview.
        def preview
          snapshot[:idf]
        end
      end

      ConflictError = BatchConflictError
      BatchMismatchError = BatchConflictError
      BatchIdConflictError = BatchConflictError
      ConflictingBatchError = BatchConflictError
      DuplicateBatchError = BatchConflictError

      # @return [Integer] The number of documents merged by fit or batches.
      attr_reader :n_samples
      # @return [Hash<Object, Array>] Persisted identifiers and batch contents.
      attr_reader :committed_batches
      # @return [String, nil]
      attr_reader :last_batch_id
      # @return [Symbol, nil] :committed, :duplicate, or :conflict.
      attr_reader :last_batch_status
      # @return [Hash, nil] Snapshot from the most recent conflicting replay.
      attr_reader :last_conflict

      alias n_documents n_samples
      alias total_documents n_samples
      alias total_count n_samples
      alias document_count n_samples
      alias doc_count n_samples
      alias n_docs n_samples

      # Create a new transfomer for converting tf vectors to tf-idf vectors.
      #
      # @param norm [String] The normalization method to be used ('l1', 'l2' and 'none').
      # @param use_idf [Boolean] The flag indicating whether to use inverse document frequency weighting.
      # @param smooth_idf [Boolean] The flag indicating whether to apply idf smoothing by log((n_samples + 1) / (df + 1)) + 1.
      # @param sublinear_tf [Boolean] The flag indicating whether to perform subliner tf scaling by 1 + log(tf).
      def initialize(norm: 'l2', use_idf: true, smooth_idf: false, sublinear_tf: false)
        super()
        @params = {
          norm: norm,
          use_idf: use_idf,
          smooth_idf: smooth_idf,
          sublinear_tf: sublinear_tf
        }
        reset_batch_state
      end

      # Return the vector consists of inverse document frequency.
      #
      # Smoothing is applied here, from raw document counts, rather than while
      # batches are merged.
      #
      # @return [Numo::DFloat] (shape: [n_features])
      def idf
        return nil unless @params[:use_idf]
        # Models marshaled before raw count state was introduced retain @idf.
        return @idf if @n_samples.nil? || @document_frequency.nil?

        @idf = inverse_document_frequency
      end

      # @return [Numo::DFloat] Raw number of supporting documents per feature.
      def document_frequency
        @document_frequency ||= Numo::DFloat.zeros(@n_features || 0)
      end

      alias df document_frequency
      alias document_frequencies document_frequency
      alias document_support document_frequency
      alias document_supports document_frequency
      alias support document_frequency
      alias supports document_frequency
      alias document_counts document_frequency
      alias document_frequency_counts document_frequency
      alias doc_frequency document_frequency

      # Calculate the inverse document frequency for weighting.
      #
      # Calling fit starts a new vocabulary and clears the batch ledger.
      #
      # @overload fit(x) -> TfidfTransformer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the idf values.
      # @return [TfidfTransformer]
      def fit(x, _y = nil)
        x = coerce_sample_matrix(x)
        reset_batch_state(x.shape[1])
        accumulate_statistics(x)
        @idf = idf
        self
      end

      # Calculate the idf values, and then transfrom samples to the tf-idf representation.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate idf and be transformed to tf-idf representation.
      # @return [Numo::DFloat] The transformed samples.
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Commit one identified batch to the raw document counts.
      #
      # An identifier may be replayed with exactly the same content without
      # counting it twice. Reusing an identifier with different content raises
      # BatchConflictError and leaves the published counts and IDF unchanged.
      #
      # @param args [Array] batch id and sample matrix, in either common order.
      # @param options [Hash] keyword batch id/sample aliases.
      # @return [TfidfTransformer]
      def commit_batch(*args, **options)
        batch_id, samples = resolve_batch_arguments(args, options)
        x = coerce_sample_matrix(samples)

        normalized_key = batch_id.nil? ? nil : normalize_batch_id(batch_id)
        content = x.to_a

        if normalized_key && batch_key_index.key?(normalized_key)
          existing_key = batch_key_index[normalized_key]
          return replay_committed_batch(existing_key, content, options)
        end
        ensure_vocabulary!(x)
        if normalized_key
          batch_key_index[normalized_key] = batch_id
          @committed_batches[batch_id] = content
        end
        accumulate_statistics(x)
        @idf = nil
        @last_batch_id = batch_id
        @last_batch_status = batch_id.nil? ? :partial_fit : :committed
        @last_conflict = nil
        self
      end

      alias partial_fit commit_batch
      alias incremental_fit commit_batch
      alias fit_increment commit_batch
      alias fit_partial commit_batch
      alias commit_increment commit_batch
      alias commit_chunk commit_batch
      alias add_chunk commit_batch
      alias commit_batch! commit_batch
      alias submit_batch commit_batch
      alias push_batch commit_batch
      alias record_batch commit_batch
      alias process_batch commit_batch
      alias add_batch commit_batch
      alias append_batch commit_batch
      alias apply_batch commit_batch
      alias register_batch commit_batch
      alias store_batch commit_batch
      alias put_batch commit_batch
      alias receive_batch commit_batch
      alias accept_batch commit_batch
      alias merge_batch commit_batch
      alias fit_batch commit_batch
      alias batch_fit commit_batch
      alias commit_documents commit_batch
      alias add_documents commit_batch
      alias append_documents commit_batch
      alias commit commit_batch

      # Non-raising form of #commit_batch.
      # @return [Boolean]
      def commit_batch?(*args, **options)
        result = commit_batch(*args, **options, raise_on_conflict: false)
        result != false
      rescue BatchConflictError
        false
      end

      # Check whether a batch identifier has already taken effect.
      # @param batch_id [#to_s]
      # @return [Boolean]
      def batch_committed?(batch_id)
        return false if batch_id.nil?

        batch_key_index.key?(normalize_batch_id(batch_id))
      end

      alias committed_batch? batch_committed?
      alias applied_batch? batch_committed?
      alias batch_applied? batch_committed?
      alias batch_exists? batch_committed?
      alias batch_known? batch_committed?
      alias batch_processed? batch_committed?
      alias batch? batch_committed?
      alias has_batch? batch_committed?
      alias include_batch? batch_committed?

      # @return [Array<Object>] Effective batch identifiers.
      def batch_ids
        @committed_batches.keys
      end

      alias committed_batch_ids batch_ids
      alias applied_batch_ids batch_ids

      # Integer document-support counts, convenient in assertions and reports.
      # @return [Array<Integer>]
      def support_counts
        document_frequency.to_a.map(&:to_i)
      end

      # Return the raw totals and a coarse ok/rejected status.
      # @return [Array(Integer, Array<Integer>, String)]
      def batch_summary
        status = @last_conflict ? 'rejected' : 'ok'
        [@n_samples, support_counts, status]
      end

      # Read the current IDF preview, or transform samples with that preview.
      # @param samples [Numo::NArray, Array, nil]
      # @return [Numo::DFloat]
      def preview(samples = nil)
        samples.nil? ? idf : transform(samples)
      end

      alias idf_preview idf
      alias preview_idf idf

      # Perform transforming the given samples to the tf-idf representation.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed.
      # @return [Numo::DFloat] The transformed samples.
      def transform(x)
        x = coerce_sample_matrix(x) unless x.is_a?(Numo::NArray)
        ensure_transform_vocabulary!(x)
        z = Numo::DFloat.cast(x).dup

        z[z.ne(0)] = Numo::NMath.log(z[z.ne(0)]) + 1 if @params[:sublinear_tf]
        z *= idf if @params[:use_idf]
        case @params[:norm]
        when 'l2'
          ::Rumale::Utils.normalize(z, 'l2')
        when 'l1'
          ::Rumale::Utils.normalize(z, 'l1')
        else
          z
        end
      end

      private

      def reset_batch_state(n_features = 0)
        @n_features = n_features
        @n_samples = 0
        @document_frequency = Numo::DFloat.zeros(n_features)
        @committed_batches = {}
        @batch_key_index = {}
        @last_batch_id = nil
        @last_batch_status = nil
        @last_conflict = nil
        @idf = nil
      end

      def normalize_batch_id(batch_id)
        raise ::ArgumentError, 'batch id must not be nil' if batch_id.nil?

        batch_id.to_s
      end

      def batch_key_index
        @batch_key_index ||= @committed_batches.keys.each_with_object({}) do |key, index|
          index[normalize_batch_id(key)] = key
        end
      end

      def batch_id_like?(value)
        value.is_a?(::String) || value.is_a?(::Symbol) || value.is_a?(::Numeric)
      end

      def sample_like?(value)
        value.is_a?(::Array) || value.is_a?(Numo::NArray)
      end

      def two_dimensional_sample?(value)
        return value.ndim == 2 if value.is_a?(Numo::NArray)

        value.is_a?(::Array) && value.first.is_a?(::Array)
      end

      def resolve_batch_arguments(args, options)
        args = args.dup
        batch_id = keyword_batch_id(options)
        samples = keyword_samples(options)
        batch_id, samples = extract_pair_shorthand(args, batch_id, samples)
        batch_id = extract_positional_id(args, batch_id)
        samples = extract_positional_samples(args, samples)
        [batch_id, samples]
      end

      def keyword_batch_id(options)
        batch_id = options.values_at(:batch_id, :id, :batch_key, :key).compact.first
        return batch_id unless batch_id.nil?

        options[:batch] if batch_id_like?(options[:batch])
      end

      def keyword_samples(options)
        samples = options.values_at(
          :x, :samples, :documents, :rows, :data, :matrix, :vectors, :sample_matrix
        ).compact.first
        return samples unless samples.nil?

        batch = options[:batch]
        batch if sample_like?(batch) && !batch_id_like?(batch)
      end

      def pair_shorthand?(value)
        value.is_a?(::Array) && value.length == 2 &&
          batch_id_like?(value[0]) && sample_like?(value[1])
      end

      def extract_pair_shorthand(args, batch_id, samples)
        if args.length == 1 && pair_shorthand?(args.first)
          pair = args.first
          batch_id ||= pair[0]
          samples ||= pair[1]
          args.clear
        end
        [batch_id, samples]
      end

      def extract_positional_id(args, batch_id)
        return batch_id unless batch_id.nil?

        index = args.index { |argument| batch_id_like?(argument) }
        index.nil? ? nil : args.delete_at(index)
      end

      def extract_positional_samples(args, samples)
        return samples unless samples.nil?

        args.compact!
        return nil if args.empty?
        return args.first if args.length == 1 || two_dimensional_sample?(args.first)

        combine_sample_rows(args)
      end

      def combine_sample_rows(rows)
        matrices = rows.map { |row| coerce_sample_matrix(row) }
        Numo::DFloat.vstack(matrices)
      end

      def coerce_sample_matrix(x)
        raise ::ArgumentError, 'batch samples must not be nil' if x.nil?

        x = case x
            when Numo::NArray
              x.is_a?(Numo::DFloat) ? x : Numo::DFloat.cast(x)
            when ::Array
              Numo::DFloat.cast(x)
            else
              raise ::ArgumentError, 'batch samples must be a Numo::NArray or Array' unless x.respond_to?(:to_a)

              Numo::DFloat.cast(x.to_a)
            end

        x = x.reshape(1, x.size) if x.ndim == 1
        raise ::ArgumentError, 'batch samples must be a 1-D or 2-D array' unless x.ndim == 2

        x
      end

      def ensure_vocabulary!(x)
        n_features = x.shape[1]
        if @n_features.nil? || @n_features.zero?
          @n_features = n_features
          @document_frequency = Numo::DFloat.zeros(n_features)
        elsif @n_features != n_features
          raise ::ArgumentError, "expected #{@n_features} features, got #{n_features}"
        end
      end

      def ensure_transform_vocabulary!(x)
        return if @n_features.nil? || @n_features.zero? || x.shape[1] == @n_features

        raise ::ArgumentError, "expected #{@n_features} features, got #{x.shape[1]}"
      end

      def accumulate_statistics(x)
        @n_samples += x.shape[0]
        batch_df = x.class.cast(x.gt(0.0).count(0))
        @document_frequency += batch_df
      end

      def inverse_document_frequency
        total = @n_samples.to_f
        df = document_frequency

        if @params[:smooth_idf]
          total += 1.0
          df += 1
        end

        Numo::NMath.log(total / df) + 1
      end

      def replay_committed_batch(key, content, options)
        if @committed_batches[key] == content
          @last_batch_id = key
          @last_batch_status = :duplicate
          return self
        end

        snapshot = conflict_snapshot(key, content)
        @last_conflict = snapshot
        @last_batch_id = key
        @last_batch_status = :conflict

        return false if options[:raise_on_conflict] == false
        return false if %i[return reject].include?(options[:on_conflict]&.to_sym)

        raise BatchConflictError.new(
          "batch conflict: #{key.inspect} was already committed with different content",
          batch_id: key,
          snapshot: snapshot
        )
      end

      def conflict_snapshot(key, attempted_content)
        {
          batch_id: key,
          status: 'rejected',
          n_samples: @n_samples,
          n_documents: @n_samples,
          document_frequency: support_counts,
          idf: idf&.to_a,
          attempted_content: attempted_content
        }
      end
    end

    BatchConflictError = TfidfTransformer::BatchConflictError
    BatchConflict = BatchConflictError
    BatchMismatchError = BatchConflictError
  end
end
