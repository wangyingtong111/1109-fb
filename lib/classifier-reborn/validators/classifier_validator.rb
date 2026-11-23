# frozen_string_literal: true

require 'singleton'

module ClassifierReborn
  module ClassifierValidator
    # Distinct identity for predictions the model refused to make (e.g. when a
    # threshold is enabled and the best score falls below it). It is a dedicated
    # singleton object rather than a String or Symbol, so it can never collide
    # with an actual class label supplied by a caller, even if a real category
    # happens to be named "Abstained" (or "拒判").
    class AbstainedPrediction
      include Singleton

      def to_s
        'Abstained'
      end
      alias inspect to_s
    end

    # The single sentinel value stored in a confusion matrix cell when the
    # classifier abstained instead of answering.
    ABSTAINED = AbstainedPrediction.instance

    module_function

    def cross_validate(classifier, sample_data, fold = 10, *options)
      classifier = ClassifierReborn.const_get(classifier).new(options) if classifier.is_a?(String)
      sample_data.shuffle!
      partition_size = sample_data.length / fold
      partitioned_data = sample_data.each_slice(partition_size)
      conf_mats = []
      fold.times do |i|
        training_data = partitioned_data.take(fold)
        test_data = training_data.slice!(i)
        conf_mats << validate(classifier, training_data.flatten!(1), test_data)
      end
      classifier.reset
      generate_report(conf_mats)
    end

    def validate(classifier, training_data, test_data, *options)
      classifier = ClassifierReborn.const_get(classifier).new(options) if classifier.is_a?(String)
      classifier.reset
      training_data.each do |rec|
        classifier.train(rec.first, rec.last)
      end
      evaluate(classifier, test_data)
    end

    def evaluate(classifier, test_data)
      # Rows are keyed by the true class. A validation fold can contain a true
      # class the trained model has never seen, so start from the classifier's
      # categories and grow rows/columns dynamically instead of dropping them.
      conf_mat = empty_conf_mat(classifier.categories.sort)
      test_data.each do |rec|
        actual = rec.first.tr('_', ' ').capitalize
        predicted = classifier.classify(rec.last)
        # A threshold enabled classifier returns nil to signal abstention.
        # Keep these records under the dedicated ABSTAINED column instead of
        # silently discarding them.
        predicted = ABSTAINED if predicted.nil?
        ensure_conf_mat_cell(conf_mat, actual, predicted)
        conf_mat[actual][predicted] += 1
      end
      conf_mat
    end

    def generate_report(*conf_mats)
      conf_mats.flatten!
      accumulated_conf_mat = merge_conf_mats(conf_mats)
      header = 'Run     Total   Correct Incorrect Abstained  Answered  Accuracy  Coverage'
      puts
      puts ' Run Report '.center(header.length, '-')
      puts header
      puts '-' * header.length
      if conf_mats.length > 1
        conf_mats.each_with_index do |conf_mat, i|
          print_run_report(build_run_report(conf_mat), i + 1)
        end
        puts '-' * header.length
      end
      print_run_report(build_run_report(accumulated_conf_mat), 'All')
      puts
      print_conf_mat(accumulated_conf_mat)
      puts
      conf_tab = conf_mat_to_tab(accumulated_conf_mat)
      print_conf_tab(conf_tab)
    end

    def build_run_report(conf_mat)
      correct = incorrect = abstained = 0
      conf_mat.each do |actual, cols|
        cols.each do |predicted, v|
          if predicted == ABSTAINED
            abstained += v
          elsif actual == predicted
            correct += v
          else
            incorrect += v
          end
        end
      end
      total = correct + incorrect + abstained
      answered = correct + incorrect
      {
        total: total,
        correct: correct,
        incorrect: incorrect,
        abstained: abstained,
        answered: answered,
        # Accuracy over every sample; an abstention is not a correct answer.
        accuracy: divide(correct, total),
        # Accuracy restricted to samples the model actually answered.
        answered_accuracy: divide(correct, answered),
        # Share of samples the model did not refuse to classify.
        coverage: divide(answered, total)
      }
    end

    # Accumulates the raw cell counts of one or more confusion matrices.
    # The union of every true class (rows) and every prediction (columns,
    # including ABSTAINED) is built first, so folds that did not see a given
    # class still merge correctly. Counts are summed rather than averaged, and
    # the resulting keys are deterministically ordered, so merging the same
    # folds in any order yields an identical matrix.
    def merge_conf_mats(conf_mats)
      conf_mats = Array(conf_mats).flatten
      row_labels = conf_mats.flat_map(&:keys).uniq
      column_labels = conf_mats.flat_map { |cm| cm.values.flat_map(&:keys) }.uniq
      merged = empty_conf_mat(ordered_labels(row_labels), ordered_labels(column_labels))
      conf_mats.each do |conf_mat|
        conf_mat.each do |actual, cols|
          cols.each do |predicted, v|
            ensure_conf_mat_cell(merged, actual, predicted)
            merged[actual][predicted] += v
          end
        end
      end
      merged
    end

    def conf_mat_to_tab(conf_mat)
      conf_tab = Hash.new { |h, k| h[k] = { p: { t: 0, f: 0 }, n: { t: 0, f: 0 } } }
      conf_mat.each_key do |positive|
        conf_mat.each do |actual, cols|
          cols.each do |predicted, v|
            conf_tab[positive][positive == predicted ? :p : :n][actual == predicted ? :t : :f] += v
          end
        end
      end
      conf_tab
    end

    def print_run_report(stats, prefix = '', print_header = false)
      if print_header
        puts "#{'Run'.rjust([3, prefix.length].max)}     Total   Correct Incorrect Abstained  Answered  Accuracy  Coverage"
      end
      puts [
        prefix.to_s.rjust(3),
        stats[:total].to_s.rjust(9),
        stats[:correct].to_s.rjust(9),
        stats[:incorrect].to_s.rjust(9),
        stats.fetch(:abstained, 0).to_s.rjust(9),
        stats.fetch(:answered, stats[:total] - stats.fetch(:abstained, 0)).to_s.rjust(9),
        stats[:accuracy].round(5).to_s.ljust(7, '0').rjust(9),
        stats.fetch(:coverage, 1.0).round(5).to_s.ljust(7, '0').rjust(9)
      ].join(' ')
    end

    def print_conf_mat(conf_mat)
      # Rows are true classes; columns are predictions. They can differ because
      # a fold may contain a true class the model never trained on, and the
      # ABSTAINED column has no matching true-class row. Order each set
      # deterministically and always render ABSTAINED last.
      row_labels = ordered_labels(conf_mat.keys)
      column_labels = ordered_labels(conf_mat.values.flat_map(&:keys).uniq)
      header = ['Predicted ->'] + column_labels.map(&:to_s) + %w[Total Recall]
      cell_size = header.map(&:length).max
      header = header.map { |h| h.rjust(cell_size) }.join(' ')
      puts ' Confusion Matrix '.center(header.length, '-')
      puts header
      puts '-' * header.length
      predicted_totals = column_labels.map { |predicted| [predicted, 0] }.to_h
      correct = 0
      row_labels.each do |actual|
        rec = conf_mat[actual]
        actual_total = column_labels.sum { |predicted| rec[predicted].to_i }
        cells = column_labels.map { |predicted| rec[predicted].to_i.to_s.rjust(cell_size) }
        row = [actual.to_s.ljust(cell_size)] + cells +
              [actual_total.to_s.rjust(cell_size), divide(rec[actual].to_i, actual_total).round(5).to_s.rjust(cell_size)]
        puts row.join(' ')
        column_labels.each do |predicted|
          val = rec[predicted].to_i
          predicted_totals[predicted] += val
          correct += val if predicted == actual
        end
      end
      total = predicted_totals.values.reduce(:+)
      puts '-' * header.length
      puts (['Total'.ljust(cell_size)] + predicted_totals.values.map { |v| v.to_s.rjust(cell_size) } + [total.to_s.rjust(cell_size), ''.rjust(cell_size)]).join(' ')
      precision_cells = column_labels.map do |predicted|
        # ABSTAINED is never a true class, so its precision is undefined (0.0).
        matched = predicted == ABSTAINED ? 0 : conf_mat.fetch(predicted, {})[predicted].to_i
        divide(matched, predicted_totals[predicted]).round(5).to_s.rjust(cell_size)
      end
      puts (['Precision'.ljust(cell_size)] + precision_cells + ['Accuracy ->'.rjust(cell_size), divide(correct, total).round(5).to_s.rjust(cell_size)]).join(' ')
    end

    def print_conf_tab(conf_tab)
      conf_tab.each do |positive, tab|
        puts "# Positive class: #{positive}"
        derivations = conf_tab_derivations(tab)
        print_derivations(derivations)
        puts
      end
    end

    def conf_tab_derivations(tab)
      positives = tab[:p][:t] + tab[:n][:f]
      negatives = tab[:n][:t] + tab[:p][:f]
      total     = positives + negatives
      {
        total_population: positives + negatives,
        condition_positive: positives,
        condition_negative: negatives,
        true_positive: tab[:p][:t],
        true_negative: tab[:n][:t],
        false_positive: tab[:p][:f],
        false_negative: tab[:n][:f],
        prevalence: divide(positives, total),
        specificity: divide(tab[:n][:t], negatives),
        recall: divide(tab[:p][:t], positives),
        precision: divide(tab[:p][:t], tab[:p][:t] + tab[:p][:f]),
        accuracy: divide(tab[:p][:t] + tab[:n][:t], total),
        f1_score: divide(2 * tab[:p][:t], 2 * tab[:p][:t] + tab[:p][:f] + tab[:n][:f])
      }
    end

    def print_derivations(derivations)
      max_len = derivations.keys.map(&:length).max
      derivations.each do |k, v|
        puts k.to_s.tr('_', ' ').capitalize.ljust(max_len) + ' : ' + v.to_s
      end
    end

    # Builds an all-zero confusion matrix. Rows are the true classes and
    # columns are the possible predictions. Passing ABSTAINED among the column
    # categories gives threshold rejections a stable place to be recorded.
    # A separate set of column labels can be supplied for rectangular
    # matrices (e.g. when merging folds with different class sets).
    def empty_conf_mat(row_categories, column_categories = nil)
      column_categories ||= row_categories
      columns = ordered_labels(column_categories)
      ordered_labels(row_categories).map do |actual|
        [actual, columns.map { |predicted| [predicted, 0] }.to_h]
      end.to_h
    end

    # Deterministic label ordering: real labels alphabetically by their display
    # name, with the ABSTAINED sentinel always sorted last.
    def ordered_labels(labels)
      labels.uniq.partition { |label| label != ABSTAINED }
            .map { |group| group.sort_by { |label| label.to_s } }
            .inject(:+)
    end

    # Makes sure a confusion matrix has the given true-class row and prediction
    # column, adding zeroed structure on demand instead of raising on a class
    # the model was never trained with.
    def ensure_conf_mat_cell(conf_mat, actual, predicted)
      conf_mat[actual] ||= {}
      conf_mat.each_value do |cols|
        cols[predicted] ||= 0 unless cols.key?(predicted)
      end
      conf_mat[actual][predicted] ||= 0
    end

    def divide(dividend, divisor)
      divisor.zero? ? 0.0 : dividend / divisor.to_f
    end
  end
end
