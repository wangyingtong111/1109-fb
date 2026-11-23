# frozen_string_literal: true

require_relative '../category_namer'
require_relative '../extensions/hasher'
require_relative 'classifier_validator'
require_relative 'overlap_audit_report'

module ClassifierReborn
  # Overlap auditing for document classification validation.
  #
  # The audit answers one question for every holdout (test) document: does the
  # training data of this fold contain a document with the same normalized
  # features, and under which true label(s)?
  module ClassifierValidator
    module_function

    # Audits the overlap between the training data and the holdout (test) data
    # of one validation run.
    #
    # Documents are never compared as raw text. They are compared through the
    # feature extraction that is actually in effect for +classifier+ -- its
    # tokenizer, token filters, stopwords and stemmer settings -- so the
    # signature of a document is exactly the word hash the classifier would
    # train or classify on (case folded, stop words dropped, stemmed, counts
    # kept). "Green Tea Leaves" and "green tea leaves" therefore share one
    # signature, while a classifier with the stemmer disabled keeps them apart
    # from "green tea leaf".
    #
    # Every matching training record is kept as evidence together with its own
    # source position and its own true label, so a signature trained under more
    # than one label reports all of them instead of only the last one seen.
    #
    # A document whose text has no surviving term produces an empty vector.
    # Such documents are reported separately as having no comparable evidence:
    # they are never indexed as training sources and never matched against
    # anything, so a handful of unrelated empty documents can never be reported
    # as copies of each other.
    #
    # +classifier+ may be an instance that is already trained on +training_data+,
    # or a classifier name plus its initialization options as trailing arguments,
    # in which case it is instantiated and trained here so the audit sees the
    # same state the validation did.
    #
    # Returns a Hash with :feature_extraction (:classifier when the classifier
    # reports its own rule, :default otherwise), the :training and :holdout
    # record lists, and :counts carrying both the full sample and the audited
    # denominators. The validation entry points return the same Hash under
    # their :overlap key.
    def audit_overlap(classifier, training_data, test_data, *options, fold: nil)
      classifier = build_classifier(classifier, training_data, options)
      training = audit_entries(classifier, training_data, fold)
      index = overlap_index(training)
      holdout = audit_holdout(classifier, test_data, index, fold)

      { feature_extraction: audit_feature_extraction(classifier),
        training: training,
        holdout: holdout,
        counts: audit_counts(training, holdout) }
    end

    # Validates like #validate and additionally audits the overlap between the
    # training and holdout data of that run, from one training pass.
    #
    # Returns a Hash with the :conf_mat confusion matrix and the :overlap audit
    # (see #audit_overlap).
    def validate_with_audit(classifier, training_data, test_data, *options)
      classifier = ClassifierReborn.const_get(classifier).new(options) if classifier.is_a?(String)
      train_classifier(classifier, training_data)
      { conf_mat: evaluate(classifier, test_data),
        overlap: audit_overlap(classifier, training_data, test_data) }
    end

    # k-fold cross-validation with a per-fold overlap audit.
    #
    # Each fold is audited against the training data of that same fold and with
    # the feature extraction in effect for that fold, then the fold audits are
    # merged so the evidence keeps the fold and the source record it came from.
    # Reports are printed for the confusion matrices and for the merged audit.
    def cross_validate_with_audit(classifier, sample_data, fold = 10, *options)
      classifier = ClassifierReborn.const_get(classifier).new(options) if classifier.is_a?(String)
      partitions = partition_folds(sample_data, fold)
      conf_mats = []
      audits = []
      fold.times do |i|
        training_data, test_data = split_fold(partitions, i)
        train_classifier(classifier, training_data)
        conf_mats << evaluate(classifier, test_data)
        audits << audit_overlap(classifier, training_data, test_data, fold: i + 1)
      end
      classifier.reset
      merged = merge_audits(audits)
      generate_report(conf_mats)
      print_overlap_audit(merged)
      { conf_mats: conf_mats, overlap: merged }
    end

    # Returns the normalized signature of +text+ for +classifier+: the word hash
    # its effective feature extraction produces. An empty Hash means the text
    # carries no comparable evidence.
    def audit_features(classifier, text)
      if classifier.respond_to?(:word_hash)
        classifier.word_hash(text.to_s)
      else
        Hasher.word_hash(text.to_s)
      end
    end

    # Whether the audit uses the classifier's own rule or the generic one.
    def audit_feature_extraction(classifier)
      classifier.respond_to?(:word_hash) ? :classifier : :default
    end

    # The true label as the classifier itself records it.
    def audit_category(label)
      CategoryNamer.prepare_name(label).to_s
    end

    # The true label as it was supplied in the validation data. Conflict
    # detection and the reported evidence use this form, so two spellings that
    # the classifier happens to fold into one category are still reported as two
    # different true labels.
    def audit_label(label)
      label.to_s
    end

    def audit_holdout(classifier, test_data, index, fold = nil)
      audit_entries(classifier, test_data, fold).map do |entry|
        if entry[:signature].empty?
          entry.merge(status: :empty_features, sources: [], labels: [],
                      categories: [], conflicted: false)
        else
          sources = index.fetch(entry[:signature], [])
          labels = sources.map { |source| source[:label] }.uniq
          entry.merge(status: sources.empty? ? :unique : :overlap,
                      sources: sources,
                      labels: labels,
                      categories: sources.map { |source| source[:category] }.uniq,
                      conflicted: labels.length > 1)
        end
      end
    end

    def audit_entries(classifier, records, fold)
      Array(records).each_with_index.map do |rec, index|
        label, text = rec
        signature = audit_features(classifier, text)
        { index: index, fold: fold,
          label: audit_label(label), category: audit_category(label),
          text: text.to_s, signature: signature, features: signature.keys.map(&:to_s).sort,
          status: signature.empty? ? :empty_features : :indexed }
      end
    end

    def overlap_index(training_entries)
      training_entries.each_with_object({}) do |entry, index|
        next if entry[:signature].empty?

        (index[entry[:signature]] ||= []) << entry
      end
    end

    def audit_counts(training, holdout)
      empty = holdout.count { |entry| entry[:status] == :empty_features }
      { training_total: training.length,
        training_empty_features: training.count { |entry| entry[:signature].empty? },
        holdout_total: holdout.length,
        holdout_empty_features: empty,
        holdout_overlapping: holdout.count { |entry| entry[:status] == :overlap },
        holdout_unique: holdout.count { |entry| entry[:status] == :unique },
        holdout_conflicted: holdout.count { |entry| entry[:conflicted] },
        comparable_holdout: holdout.length - empty,
        audited_holdout: holdout.count { |entry| entry[:status] == :unique } }
    end

    # Combines the per-fold audits of a cross-validation run into one report.
    # Each record keeps the fold it came from, so the evidence still points at
    # the exact training source of the exact fold.
    def merge_audits(audits)
      audits = Array(audits).compact
      return nil if audits.empty?
      return audits.first if audits.length == 1

      training = audits.flat_map { |audit| audit[:training] }
      holdout = audits.flat_map { |audit| audit[:holdout] }
      { feature_extraction: audits.map { |audit| audit[:feature_extraction] }.uniq.join('/'),
        training: training,
        holdout: holdout,
        counts: audit_counts(training, holdout) }
    end

    def build_classifier(classifier, training_data, options)
      return classifier unless classifier.is_a?(String)

      classifier = ClassifierReborn.const_get(classifier).new(options)
      train_classifier(classifier, training_data)
      classifier
    end
  end
end
