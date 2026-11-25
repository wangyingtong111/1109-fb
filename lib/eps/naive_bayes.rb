module Eps
  class NaiveBayes < BaseEstimator
    attr_reader :probabilities

    def accuracy
      Eps::Metrics.accuracy(@train_set.label, predict(@train_set), weight: @train_set.weight)
    end

    private

    # TODO better summary
    def _summary(extended: false)
      str = String.new("")
      probabilities[:prior].each do |k, v|
        str += "#{k}: #{v}\n"
      end
      str
    end

    def _train(smoothing: 1)
      raise "Target must be strings" if @target_type != "categorical"
      check_missing_value(@train_set)
      check_missing_value(@validation_set) if @validation_set

      data = @train_set
      weights = normalized_weights(data.weight)

      prep_text_features(data)

      # convert boolean to strings
      data.label = data.label.map(&:to_s)

      indexes = {}
      data.label.each_with_index do |yi, i|
        next if weights && weights[i] == 0

        (indexes[yi] ||= []) << i
      end
      raise ArgumentError, "Total weight must be positive" if indexes.empty?

      grouped = {}
      indexes.each do |k, v|
        grouped[k] = data[v]
      end

      prior = {}
      grouped.sort_by { |k, _| k }.each do |k, v|
        prior[k] = weights ? v.weight.sum : v.size
      end
      labels = prior.keys

      target_counts = {}
      labels.each do |k|
        target_counts[k] = 0
      end

      conditional = {}

      @features.each do |k, type|
        prob = {}

        case type
        when "text"
          raise "Text features not supported yet for naive Bayes"
        when "categorical"
          groups = Hash.new { |hash, key| hash[key] = [] }
          data.columns[k].each_with_index do |v, i|
            next if weights && weights[i] == 0

            groups[v] << i
          end

          groups.each do |group, group_indexes|
            prob[group] = group_count(data.label, target_counts.dup, group_indexes, weights)
          end

          # smooth
          if smoothing
            labels.each do |label|
              sum = prob.sum { |k2, v2| v2[label] }.to_f
              prob.each do |k2, v|
                v[label] = (v[label] + smoothing) * sum / (sum + (prob.size * smoothing))
              end
            end
          end
        else
          labels.each do |group|
            xs = grouped[group]

            # TODO handle this case
            next unless xs

            values = xs.columns[k]
            feature_weights = weights ? xs.weight : nil
            prob[group] = {mean: mean(values, feature_weights), stdev: stdev(values, feature_weights)}
          end
        end

        conditional[k] = prob
      end

      @probabilities = {
        prior: prior,
        conditional: conditional
      }

      Evaluators::NaiveBayes.new(probabilities: probabilities, features: @features)
    end

    def normalized_weights(weights)
      return nil unless weights
      raise ArgumentError, "Number of data points differs from weight" unless weights.size == @train_set.size

      weights.each do |weight|
        unless weight.is_a?(Numeric) && weight >= 0
          raise ArgumentError, "Weights must be nonnegative numeric values"
        end
      end

      # Unit weights should use the original, unweighted calculations exactly.
      weights.all? { |weight| weight == 1 } ? nil : weights
    end

    def group_count(labels, start, indexes = nil, weights = nil)
      indexes ||= (0...labels.size).to_a
      indexes.each do |i|
        start[labels[i]] += weights ? weights[i] : 1
      end
      start
    end

    def mean(arr, weights = nil)
      if weights
        total_weight = weights.sum
        return nil if total_weight == 0

        arr.each_with_index.sum { |value, i| value * weights[i] } / total_weight.to_f
      else
        arr.sum / arr.size.to_f
      end
    end

    def stdev(arr, weights = nil)
      if weights
        total_weight = weights.sum
        return nil if total_weight <= 1

        m = mean(arr, weights)
        sum = arr.each_with_index.sum { |value, i| weights[i] * (value - m)**2 }
        # Weights are frequency weights, so sum(weights) - 1 matches expanded rows.
        Math.sqrt(sum / (total_weight - 1).to_f)
      else
        return nil if arr.size <= 1

        m = mean(arr)
        sum = arr.inject(0) { |accum, i| accum + (i - m)**2 }
        Math.sqrt(sum / (arr.length - 1).to_f)
      end
    end
  end
end
