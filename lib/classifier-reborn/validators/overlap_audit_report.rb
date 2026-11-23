# frozen_string_literal: true

module ClassifierReborn
  # The human-readable report of a train/holdout overlap audit.
  #
  # Kept apart from the audit itself so that the audit, its counters and its
  # evidence stay readable on their own; the report is a thin view over the
  # Hash #audit_overlap returns.
  module ClassifierValidator
    # How many entries of each section the report lists before it switches to a
    # count only.
    AUDIT_LIST_LIMIT = 20

    # How many characters of a document the report shows.
    AUDIT_TEXT_LIMIT = 60

    module_function

    # Prints the overlap audit of one validation run, or the merged audit of a
    # cross-validation run.
    #
    # Empty feature vectors are listed first and on their own: they are not
    # evidence of duplication, so they must never be counted as copies. Then
    # come the documents that do overlap with training, each with every
    # training source and every true label that source carried, so a conflict
    # between two labels of the same feature set is visible instead of being
    # reduced to whichever label happened to be trained last. Finally the
    # comparable holdout documents without overlap are listed, followed by both
    # denominators.
    def print_overlap_audit(audit, limit: AUDIT_LIST_LIMIT)
      return if audit.nil?

      print_audit_summary(audit)
      print_empty_features(audit[:holdout], limit)
      print_overlaps(audit[:holdout], limit)
      print_effective_holdout(audit[:holdout], limit)
      print_audit_denominators(audit[:counts])
    end

    def print_audit_summary(audit)
      counts = audit[:counts]
      puts
      puts ' Train/Holdout Overlap Audit '.center(80, '-')
      puts "Feature extraction : #{audit[:feature_extraction]}"
      puts "Training records   : #{counts[:training_total]} (#{counts[:training_empty_features]} with no comparable evidence)"
      puts "Holdout records    : #{counts[:holdout_total]} (full sample)"
    end

    def print_empty_features(holdout, limit)
      print_overlap_section('No comparable evidence (empty feature vectors)',
                            holdout.select { |entry| entry[:status] == :empty_features }, limit) do |entry|
        puts "   #{audit_entry_line(entry)}"
        puts '     effective features : none'
        puts '     never indexed and never matched, so it is not a copy of anything'
      end
    end

    def print_overlaps(holdout, limit)
      print_overlap_section('Overlap with training (equal normalized features)',
                            holdout.select { |entry| entry[:status] == :overlap }, limit) do |entry|
        print_overlap_entry(entry)
      end
    end

    def print_overlap_entry(entry)
      puts "   #{audit_entry_line(entry)}"
      puts "     effective features : #{entry[:features].join(', ')}"
      entry[:sources].each { |source| puts "     training source    : #{audit_entry_line(source)}" }
      puts "     training labels    : #{entry[:labels].map(&:inspect).join(', ')}"
      return unless entry[:conflicted]

      puts "     CONFLICT           : one feature set, #{entry[:labels].length} different true labels"
      puts "     classifier category: #{entry[:categories].map(&:inspect).join(', ')}"
    end

    def print_effective_holdout(holdout, limit)
      print_overlap_section('Comparable holdout without training overlap',
                            holdout.select { |entry| entry[:status] == :unique }, limit) do |entry|
        puts "   #{audit_entry_line(entry)}"
        puts "     effective features : #{entry[:features].join(', ')}"
      end
    end

    def print_overlap_section(title, entries, limit)
      puts
      puts "-- #{title} --"
      if entries.empty?
        puts '   (none)'
        return
      end
      entries.first(limit).each { |entry| yield entry }
      puts "   (#{entries.length - limit} further entries not listed)" if entries.length > limit
    end

    def print_audit_denominators(counts)
      rows = [['Full holdout sample', counts[:holdout_total]],
              ['No comparable evidence', counts[:holdout_empty_features]],
              ['Overlapping with training', counts[:holdout_overlapping]],
              ['  of which label conflicts', counts[:holdout_conflicted]],
              ['Comparable holdout', counts[:comparable_holdout]],
              ['Audited holdout denominator', counts[:audited_holdout]]]
      width = rows.map { |row| row.first.length }.max
      puts
      puts '-- Denominators --'
      rows.each { |label, value| puts "#{label.ljust(width)} : #{value}" }
      puts 'The audited denominator keeps only holdout records that are comparable'
      puts '(non-empty) and free of training overlap.'
    end

    def audit_entry_line(entry)
      position = entry[:fold] ? "fold #{entry[:fold]} ##{entry[:index]}" : "##{entry[:index]}"
      "#{position} label=#{entry[:label].inspect} text=#{audit_text(entry[:text]).inspect}"
    end

    def audit_text(text, limit = AUDIT_TEXT_LIMIT)
      text = text.to_s
      text.length > limit ? "#{text[0, limit - 3]}..." : text
    end
  end
end
