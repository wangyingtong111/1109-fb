module Eps
  module Metrics
    # Default amount used to keep logarithms away from zero.
    #
    # A legal probability of zero is treated as this value so the loss stays
    # finite. A value that is not a probability at all is rejected instead
    # (see the log loss contract below).
    DEFAULT_EPS = 1e-15

    # Accumulates weighted log loss one segment at a time, so predictions and
    # weights can be read from a one-shot enumerator in pieces.
    #
    # Each segment is checked in full before the totals change. A segment
    # holding an illegal value raises and leaves the accumulator untouched, so
    # it can be retried with corrected content without the rows already
    # accumulated being counted twice.
    #
    # The loss is the sum over rows of the negative log likelihood times the
    # row's weight; the value is that sum divided by the total effective
    # weight. Segments are therefore combined by summing and normalizing once,
    # never by averaging the metric of each segment. A segment whose rows all
    # carry zero weight adds nothing and cannot shift the result.
    class LogLossAccumulator
      attr_reader :eps, :size, :effective_size

      def initialize(eps: DEFAULT_EPS)
        @eps = validate_eps(eps)
        reset
      end

      # Adds one segment of predictions and weights.
      #
      # Separate arguments:
      #
      #   accumulator.add(labels, predictions, weight: weights)
      #
      # or a single segment in one of these shapes:
      #
      #   accumulator.add([labels, predictions])
      #   accumulator.add([labels, predictions, weights])
      #   accumulator.add(y_true: labels, y_pred: predictions, weight: weights)
      #
      # Any of them may be a one-shot enumerator; each is read once.
      def add(*args, y_true: nil, y_pred: nil, weight: nil)
        if args.size >= 2
          y_true = args[0] if y_true.nil?
          y_pred = args[1] if y_pred.nil?
          weight = args[2] if args.size >= 3 && weight.nil?
        elsif args.size == 1
          seg_true, seg_pred, seg_weight = unpack_segment(args[0])
          y_true = seg_true if y_true.nil?
          y_pred = seg_pred if y_pred.nil?
          weight = seg_weight if weight.nil?
        elsif y_true.nil? || y_pred.nil?
          raise ArgumentError, "Segment needs y_true and y_pred"
        end

        true_labels = to_list(y_true)
        predictions = to_list(y_pred)
        weights = weight.nil? ? nil : to_list(weight)

        raise ArgumentError, "Different sizes" if true_labels.size != predictions.size
        raise ArgumentError, "Different sizes" if weights && weights.size != true_labels.size

        # Work against locals so a bad row leaves the accumulated totals alone.
        loss = 0.0
        loss_comp = 0.0
        total = 0.0
        total_comp = 0.0
        effective_size = 0

        true_labels.each_index do |i|
          w = weights ? weights[i] : 1
          validate_weight(w, i)

          term = row_loss(true_labels[i], predictions[i], i)

          # A zero weight contributes nothing at all, so it cannot shift the
          # normalized result.
          next if w == 0

          effective = w.to_f
          loss, loss_comp = compensated_add(loss, loss_comp, term * effective)
          total, total_comp = compensated_add(total, total_comp, effective)
          effective_size += 1
        end

        @loss, @loss_compensation = compensated_add(@loss, @loss_compensation, loss + loss_comp)
        @weight, @weight_compensation = compensated_add(@weight, @weight_compensation, total + total_comp)
        @size += true_labels.size
        @effective_size += effective_size

        self
      end
      alias << add

      # Merges another accumulator into this one and returns self.
      #
      # Both sides keep raw sums, so merging never averages normalized values.
      def merge(other)
        raise ArgumentError, "Expected a #{self.class}" unless other.is_a?(LogLossAccumulator)
        raise ArgumentError, "Different eps" unless other.eps == eps

        @loss, @loss_compensation = compensated_add(@loss, @loss_compensation, other.loss)
        @weight, @weight_compensation = compensated_add(@weight, @weight_compensation, other.total_weight)
        @size += other.size
        @effective_size += other.effective_size

        self
      end
      alias combine merge

      # Combines several accumulators into a new one, leaving the inputs as-is.
      def self.combine(accumulators)
        list = accumulators.to_a
        eps = list.first ? list.first.eps : DEFAULT_EPS
        list.reduce(new(eps: eps)) { |memo, accumulator| memo.merge(accumulator) }
      end

      # The summed weighted negative log likelihood.
      def loss
        @loss + @loss_compensation
      end

      # The summed effective weight (zero-weight rows are excluded).
      def total_weight
        @weight + @weight_compensation
      end

      def empty?
        @size == 0
      end

      # Weighted mean log loss.
      #
      # With no effective weight the result is NaN, matching a division of
      # zero by zero rather than pretending there is a value.
      def value
        weight = total_weight
        return Float::NAN if weight == 0

        loss / weight
      end
      alias log_loss value
      alias result value

      def reset
        @loss = 0.0
        @loss_compensation = 0.0
        @weight = 0.0
        @weight_compensation = 0.0
        @size = 0
        @effective_size = 0
        self
      end

      private

      def unpack_segment(segment)
        case segment
        when Hash
          unless segment.key?(:y_true) || segment.key?("y_true")
            raise ArgumentError, "Segment needs y_true and y_pred"
          end

          [fetch_key(segment, :y_true), fetch_key(segment, :y_pred), fetch_key(segment, :weight)]
        when Array
          raise ArgumentError, "Segment needs y_true and y_pred" if segment.size < 2

          segment
        else
          if segment.respond_to?(:to_a)
            list = segment.to_a
            raise ArgumentError, "Segment needs y_true and y_pred" if list.size < 2

            list
          else
            raise ArgumentError, "Invalid segment: #{segment.inspect}"
          end
        end
      end

      def fetch_key(hash, key)
        hash.key?(key) ? hash[key] : hash[key.to_s]
      end

      def to_list(value)
        if value.is_a?(Hash)
          [value]
        elsif value.nil?
          []
        elsif value.is_a?(String)
          [value]
        elsif value.respond_to?(:to_a)
          value.to_a
        else
          [value]
        end
      end

      def validate_eps(eps)
        unless eps.is_a?(Numeric) && eps.to_f.finite? && eps >= 0 && eps <= 0.5
          raise ArgumentError, "eps must be a number between 0 and 0.5"
        end

        eps.to_f
      end

      def validate_weight(weight, index)
        unless weight.is_a?(Numeric) && weight.to_f.finite? && weight >= 0
          raise ArgumentError, "Invalid weight at row #{index}: #{weight.inspect}"
        end
      end

      # Negative log likelihood for one row.
      #
      # A Hash prediction maps labels to probabilities; the probability stored
      # under the row's true label is used, so the Hash key order is
      # irrelevant. A numeric prediction keeps the historical two-class
      # contract: a true label of 1 selects the value and any other label
      # selects its complement.
      def row_loss(label, prediction, index)
        if prediction.is_a?(Hash)
          probability = probability_for(label, prediction, index)
          validate_probability(probability, index)
          -Math.log(clamp_probability(probability.to_f))
        elsif prediction.is_a?(Numeric)
          validate_probability(prediction, index)
          # Clamp before complementing, exactly as the two-class form always
          # has, so existing callers keep their results.
          probability = clamp_probability(prediction.to_f)
          label == 1 ? -Math.log(probability) : -Math.log(1 - probability)
        else
          raise ArgumentError, "Invalid prediction at row #{index}: #{prediction.inspect}"
        end
      end

      def probability_for(label, prediction, index)
        keys = [label, label.to_s]
        keys << label.to_sym if label.respond_to?(:to_sym)

        keys.uniq.each do |key|
          return prediction[key] if prediction.key?(key)
        end

        raise ArgumentError, "Missing probability for label #{label.inspect} (row #{index})"
      end

      def validate_probability(probability, index)
        unless probability.is_a?(Numeric) && probability.to_f.finite? &&
               probability >= 0 && probability <= 1
          raise ArgumentError, "Invalid probability at row #{index}: #{probability.inspect}"
        end
      end

      # Keeps a legal zero (or one) away from the logarithm while leaving the
      # rest of the interval untouched.
      def clamp_probability(probability)
        return @eps if probability < @eps
        return 1 - @eps if probability > 1 - @eps

        probability
      end

      # Neumaier compensated summation, so reordering segments does not drift.
      def compensated_add(sum, compensation, value)
        t = sum + value
        if sum.abs >= value.abs
          compensation += (sum - t) + value
        else
          compensation += (value - t) + sum
        end
        [t, compensation]
      end
    end

    # Shorter name for the same accumulator
    LogLoss = LogLossAccumulator

    class << self
      def rmse(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        Math.sqrt(mean(errors(y_true, y_pred).map { |v| v**2 }, weight: weight))
      end

      def mae(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        mean(errors(y_true, y_pred).map { |v| v.abs }, weight: weight)
      end

      def me(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        mean(errors(y_true, y_pred), weight: weight)
      end

      def accuracy(y_true, y_pred, weight: nil)
        check_size(y_true, y_pred)
        values = y_true.zip(y_pred).map { |yt, yp| yt == yp ? 1 : 0 }
        if weight
          values.each_with_index do |v, i|
            values[i] *= weight[i]
          end
          values.sum / weight.sum.to_f
        else
          values.sum / y_true.size.to_f
        end
      end

      # Weighted log loss.
      #
      # Two shapes are supported.
      #
      # The historical two-class shape, where each prediction is the
      # probability of the positive class and a true label of 1 selects it
      # while any other label selects its complement:
      #
      #   log_loss([0, 1, 0], [0.1, 0.9, 0.4], weight: [1, 2, 3])
      #
      # The multiclass shape returned by predict_probability, where each
      # prediction maps a label to its probability and the value stored under
      # the row's true label is used:
      #
      #   log_loss(
      #     ["a", "c"],
      #     [{"a" => 0.2, "b" => 0.3, "c" => 0.5},
      #      {"b" => 0.4, "c" => 0.5, "a" => 0.1}],
      #     weight: [1, 3]
      #   )
      #   # => (-log(0.2) - 3 * log(0.5)) / 4
      #
      # Hash key order does not affect the result.
      #
      # Contract for edge cases:
      # - a true label missing from its row's Hash raises ArgumentError
      # - a probability outside 0..1, or NaN/Infinity, raises ArgumentError
      # - a legal probability of 0 is treated as eps, so the loss stays finite
      # - a zero weight contributes neither loss nor denominator weight
      #
      # To accumulate a one-shot enumerator in segments, use
      # log_loss_accumulator, log_loss_segments, or pass the segments as the
      # only argument.
      def log_loss(y_true, y_pred = nil, eps: DEFAULT_EPS, weight: nil)
        if y_pred.nil?
          return log_loss_segments(y_true, eps: eps)
        end

        log_loss_accumulator(eps: eps).add(y_true, y_pred, weight: weight).value
      end

      # Builds an accumulator for summing log loss over segments:
      #
      #   accumulator = Eps::Metrics.log_loss_accumulator
      #   until predictions.empty?
      #     accumulator.add(labels.shift(100), predictions.shift(100),
      #                     weight: weights.shift(100))
      #   end
      #   accumulator.value
      #
      # A segment can also be a [labels, predictions, weights] tuple or a
      # {y_true:, y_pred:, weight:} Hash. Segments are merged by summing their
      # raw loss and weight, then normalizing once.
      def log_loss_accumulator(eps: DEFAULT_EPS)
        LogLossAccumulator.new(eps: eps)
      end

      # Accumulates a one-shot enumerator of segments and returns the value.
      #
      #   segments = Enumerator.new do |y|
      #     y << [["a"], [{"a" => 0.2}], [1]]
      #     y << [["c"], [{"c" => 0.5}], [3]]
      #   end
      #   Eps::Metrics.log_loss_segments(segments)
      def log_loss_segments(segments, eps: DEFAULT_EPS)
        accumulator = log_loss_accumulator(eps: eps)
        each_segment(segments) { |segment| accumulator.add(segment) }
        accumulator.value
      end

      private

      # Yields each segment of a one-shot enumerator exactly once.
      #
      # A single already-materialized segment (a [labels, predictions] pair)
      # is yielded as one segment; an enumerator or array of segments is
      # walked in order.
      def each_segment(segments)
        if segments.is_a?(Array) && segments_list?(segments)
          segments.each { |segment| yield segment }
        elsif segment?(segments)
          yield segments
        elsif segments.respond_to?(:each)
          segments.each { |segment| yield segment }
        else
          raise ArgumentError, "Invalid segments: #{segments.inspect}"
        end
      end

      # True for a [labels, predictions] or [labels, predictions, weights]
      # pair, which is one segment.
      def segment?(value)
        value.is_a?(Array) && value.size.between?(2, 3) &&
          value.all? { |part| list_like?(part) }
      end

      # True when every item is itself a segment, so the outer array is a
      # collection of segments rather than a single one.
      def segments_list?(value)
        !value.empty? && value.all? { |item| segment?(item) }
      end

      def list_like?(value)
        value.respond_to?(:each) && !value.is_a?(String) && !value.is_a?(Hash)
      end

      def check_size(y_true, y_pred)
        raise ArgumentError, "Different sizes" if y_true.size != y_pred.size
      end

      def mean(arr, weight: nil)
        if weight
          arr.map.with_index { |v, i| v * weight[i] }.sum / weight.sum.to_f
        else
          arr.sum / arr.size.to_f
        end
      end

      def errors(y_true, y_pred)
        y_true.zip(y_pred).map { |yt, yp| yt - yp }
      end
    end
  end
end
