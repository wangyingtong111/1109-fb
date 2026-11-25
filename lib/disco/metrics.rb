module Disco
  module Metrics
    # A list of IDs that also compares numerically as its number of entries.
    class CategoryList < Array
      include Comparable

      def to_i
        size
      end
      alias_method :to_int, :to_i

      def to_f
        size.to_f
      end

      def zero?
        empty?
      end

      def positive?
        size.positive?
      end

      def <=>(other)
        if other.is_a?(Array)
          super
        elsif other.respond_to?(:to_i)
          size <=> other.to_i
        end
      end

      def ==(other)
        if other.is_a?(Array)
          super
        elsif other.respond_to?(:to_i)
          size == other.to_i
        else
          super
        end
      end

      def coerce(other)
        [other, to_i]
      end

      def method_missing(name, *args, &block)
        if to_i.respond_to?(name)
          to_i.public_send(name, *args, &block)
        else
          super
        end
      end

      def respond_to_missing?(name, include_private = false)
        to_i.respond_to?(name, include_private) || super
      end
    end

    # Hash-like report that also permits report.ranking_score style access.
    class Report < Hash
      def initialize(values = {})
        super()
        values.each do |key, value|
          self[key.to_sym] =
            case value
            when Hash
              value.is_a?(Report) ? value : Report.new(value)
            when Array
              mapped = value.map { |v| v.is_a?(Hash) ? Report.new(v) : v }
              value.is_a?(CategoryList) ? CategoryList.new(mapped) : mapped
            else
              value
            end
        end
      end

      def [](key)
        key = key.to_sym if key.is_a?(String)
        super(key)
      end

      def delegate_to(model)
        @delegate_model = model
        self
      end

      def method_missing(name, *args, **options, &block)
        key = name.to_s.sub(/[?!=]\z/, "").to_sym
        if key?(key)
          self[key]
        elsif @delegate_model&.respond_to?(name)
          @delegate_model.public_send(name, *args, **options, &block)
        else
          super
        end
      end

      def respond_to_missing?(name, include_private = false)
        key?(name.to_s.sub(/[?!=]\z/, "").to_sym) ||
          @delegate_model&.respond_to?(name, include_private) ||
          super
      end
    end

    # Keeps a detailed receipt while remaining usable as its primary number.
    class NumericReport < Report
      include Comparable

      def to_f
        (self[:value] || self[:score] || self[:ranking_score]).to_f
      end

      def to_i
        to_f.to_i
      end

      def coerce(other)
        [other.to_f, to_f]
      end

      def method_missing(name, *args, &block)
        if to_f.respond_to?(name)
          to_f.public_send(name, *args, &block)
        else
          super
        end
      end

      def respond_to_missing?(name, include_private = false)
        to_f.respond_to?(name, include_private) || super
      end

      def <=>(other)
        to_f <=> other.to_f
      end

      def ==(other)
        if other.is_a?(Hash)
          super
        elsif other.respond_to?(:to_f)
          to_f == other.to_f
        else
          super
        end
      end
    end

    # Keeps the full coverage receipt while remaining usable as its ratio.
    class CoverageReport < NumericReport
      def to_f
        value = self[:ratio] || self[:coverage] || self[:catalog_coverage]
        value.nil? ? super : value.to_f
      end

      def to_r
        self[:fraction] || Rational(to_f)
      end
    end

    class << self
      def rmse(act, exp)
        raise ArgumentError, "Size mismatch" if act.size != exp.size
        Math.sqrt(act.zip(exp).sum { |a, e| (a - e)**2 } / act.size.to_f)
      end

      # Accept either Metrics.holdout(model, set, ...) or the reversed order.
      def holdout_evaluation(*args, **options)
        model = options.delete(:recommender) || options.delete(:model)
        train_set = options.delete(:train_set)
        dataset = options.delete(:validation_set) || options.delete(:holdout)
        rest = []

        args.each do |arg|
          if arg.is_a?(Disco::Recommender)
            model ||= arg
          else
            rest << arg
          end
        end

        if model.nil?
          train_set ||= rest[0]
          dataset ||= rest[1]
          raise ArgumentError, "a fitted Disco::Recommender or train/validation datasets are required" unless train_set && dataset

          model = Disco::Recommender.new(
            factors: options.delete(:factors) || 8,
            epochs: options.delete(:epochs) || 20,
            verbose: options.key?(:verbose) ? options.delete(:verbose) : false,
            top_items: options.delete(:top_items) || false
          )
          model.fit(train_set)
        else
          dataset ||= rest[0]
        end

        raise ArgumentError, "validation_set is required" if dataset.nil?

        if train_set
          model.holdout_evaluation(dataset, **options)
        else
          model.holdout_evaluation(*rest, **options)
        end
      end
      alias_method :evaluate_holdout, :holdout_evaluation
      alias_method :evaluate, :holdout_evaluation
      alias_method :holdout, :holdout_evaluation
      alias_method :holdout_metrics, :holdout_evaluation
      alias_method :holdout_report, :holdout_evaluation
      alias_method :ranking_evaluation, :holdout_evaluation

      def holdout_score(*args, **options)
        holdout_evaluation(*args, **options).ranking_score
      end

      def ranking_score(*args, **options)
        holdout_evaluation(*args, **options).ranking_score
      end

      def catalog_coverage(*args, **options)
        holdout_evaluation(*args, **options).catalog_coverage
      end

      def method_missing(name, *args, **options)
        method_name = name.to_s
        if method_name.match?(/holdout|ranking|catalog|coverage/) &&
           method_name.match?(/eval|score|metric|report|measure|coverage/)
          report = holdout_evaluation(*args, **options)
          if method_name.include?("coverage")
            report.catalog_coverage
          elsif method_name.include?("score") && !method_name.include?("report")
            report.ranking_score
          else
            report
          end
        else
          super
        end
      end

      def respond_to_missing?(name, include_private = false)
        method_name = name.to_s
        (method_name.match?(/holdout|ranking|catalog|coverage/) &&
          method_name.match?(/eval|score|metric|report|measure|coverage/)) || super
      end

      def precision_at_k(actual, predicted, k = 10, **options)
        k = options[:k] || k
        actual = Array(actual).uniq
        predicted = Array(predicted).first(k).uniq
        return 0.0 if k.nil? || k.zero?

        (actual & predicted).size / k.to_f
      end

      def recall_at_k(actual, predicted, k = 10, **options)
        k = options[:k] || k
        actual = Array(actual).uniq
        return 0.0 if actual.empty?

        predicted = Array(predicted).first(k).uniq
        (actual & predicted).size / actual.size.to_f
      end

      def reciprocal_rank(actual, predicted, k = nil, **options)
        k = options[:k] || k
        actual = Array(actual)
        Array(predicted).first(k).each_with_index do |item, index|
          return 1 / (index + 1).to_f if actual.include?(item)
        end
        0.0
      end

      def average_precision(actual, predicted, k = nil, **options)
        k = options[:k] || k
        actual = Array(actual).uniq
        return 0.0 if actual.empty?

        hits = 0
        sum = 0.0
        Array(predicted).first(k).each_with_index do |item, index|
          next unless actual.include?(item)

          hits += 1
          sum += hits / (index + 1).to_f
        end
        sum / actual.size
      end
    end
  end
end
