# frozen_string_literal: true

# Author::    Lucas Carlson  (mailto:lucas@rufy.com)
# Copyright:: Copyright (c) 2005 Lucas Carlson
# License::   LGPL

module ClassifierReborn
  module Summarizer
    module_function

    def summary(str, count = 10, separator = ' [...] ')
      perform_lsi split_sentences(str), count, separator
    end

    # Builds a summary from whole paragraphs rather than individual sentences.
    #
    # The candidates are always the original paragraphs of +str+, each keeping
    # its own position, so even paragraphs with identical text can be selected
    # independently. The selected paragraphs are returned in their original
    # reading order.
    #
    # Pass +redundancy_weight+ (also accepted as +diversity_weight+) to trade
    # centrality for diversity. A weight of +0+ (the default) keeps the
    # original centrality based behaviour: the most central paragraphs are
    # simply chosen. A positive weight additionally penalizes paragraphs that
    # are semantically close to a paragraph already selected, so repeated
    # themes are reduced. Larger weights remove more redundancy.
    def paragraph_summary(str, count = 1, separator = ' [...] ',
                          redundancy_weight: nil, diversity_weight: nil)
      weight = (redundancy_weight || diversity_weight || 0).to_f

      chunks = split_paragraphs(str).reject do |chunk|
        chunk.strip.empty? || chunk.strip.split.size == 1
      end
      return '' if chunks.empty?

      selected = select_paragraphs(chunks, count, weight)
      selected.map(&:strip).join(separator)
    end

    def split_sentences(str)
      str.split(/(\.|\!|\?)/) # TODO: make this less primitive
    end

    def split_paragraphs(str)
      str.split(/(\n\n|\r\r|\r\n\r\n)/) # TODO: make this less primitive
    end

    def perform_lsi(chunks, count, separator)
      lsi = ClassifierReborn::LSI.new auto_rebuild: false
      chunks.each { |chunk| lsi << chunk unless chunk.strip.empty? || chunk.strip.split.size == 1 }
      lsi.build_index
      summaries = lsi.highest_relative_content count
      summaries.select { |chunk| summaries.include? chunk }.map(&:strip).join(separator)
    end

    # Selects up to +count+ paragraph positions. Paragraphs with identical text
    # share one LSI entry (and therefore one score) but remain distinct
    # candidates through their position in the document.
    def select_paragraphs(chunks, count, redundancy_weight)
      limit = [count, chunks.size].min
      return [] if limit.zero?

      # Map every paragraph text to the positions where it occurs. Identical
      # paragraphs share an LSI entry, but each occurrence is its own slot.
      positions = Hash.new { |hash, key| hash[key] = [] }
      chunks.each_with_index { |chunk, index| positions[chunk] << index }

      texts = positions.keys

      # With a single distinct paragraph there is nothing to compare against:
      # every candidate is equally central, so just keep the leading slots.
      return chunks.first(limit) if texts.size == 1

      lsi = ClassifierReborn::LSI.new auto_rebuild: false
      texts.each { |chunk| lsi << chunk }
      lsi.build_index

      if redundancy_weight.zero?
        ranked_texts = lsi.highest_relative_content(texts.size)
      else
        # Only distinct texts can occupy slots; identical copies are expanded
        # back into their own positions afterwards.
        distinct_limit = [limit, texts.size].min
        ranked_texts = lsi.highest_relative_content_with_mmr(distinct_limit, redundancy_weight)
      end

      expand_positions(chunks, positions, ranked_texts, limit)
    end

    # Expands the ranked, distinct paragraph texts back into document
    # positions. When one text occurs more than once and more copies are
    # needed, each occurrence contributes its own position. The result keeps
    # the original reading order.
    def expand_positions(chunks, positions, ranked_texts, limit)
      remaining = Hash.new(0)
      ranked_texts.each do |text|
        # One text may need to fill several slots when paragraphs repeat.
        copies = [positions[text].size, limit - remaining.values.sum].min
        remaining[text] = copies
      end

      chosen = []
      chunks.each_with_index do |chunk, index|
        break if chosen.size == limit
        next if remaining[chunk].zero?

        remaining[chunk] -= 1
        chosen << index
      end
      chosen.map { |index| chunks[index] }
    end
  end
end
