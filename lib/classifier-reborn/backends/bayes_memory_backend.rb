# frozen_string_literal: true

require_relative 'mergeable_backend'

module ClassifierReborn
  class BayesMemoryBackend
    include MergeableBackend

    attr_reader :total_words, :total_trainings

    # This class provides Memory as the storage backend for the classifier data structures
    def initialize
      @total_words     = 0
      @total_trainings = 0
      @category_counts = {}
      @categories      = {}
    end

    def update_total_words(diff)
      @total_words += diff
    end

    def update_total_trainings(diff)
      @total_trainings += diff
    end

    # The readers below never create entries: reading the statistics of a
    # backend must not change it, so callers can inspect a model they do not own.
    def category_training_count(category)
      counts = @category_counts[category]
      counts ? counts[:training] : 0
    end

    def update_category_training_count(category, diff)
      category_counts(category)[:training] += diff
    end

    def category_has_trainings?(category)
      @category_counts.key?(category) && category_training_count(category) > 0
    end

    def category_word_count(category)
      counts = @category_counts[category]
      counts ? counts[:word] : 0
    end

    def update_category_word_count(category, diff)
      category_counts(category)[:word] += diff
    end

    def add_category(category)
      @categories[category] ||= Hash.new(0)
    end

    def category_keys
      @categories.keys
    end

    def category_word_frequency(category, word)
      bucket = @categories[category]
      bucket ? bucket[word] : 0
    end

    # Returns a copy of the word => frequency counts of +category+.
    def category_word_frequencies(category)
      (@categories[category] || {}).each_with_object({}) do |(word, count), frequencies|
        frequencies[word] = count unless count.zero?
      end
    end

    def update_category_word_frequency(category, word, diff)
      @categories[category][word] += diff
    end

    def delete_category_word(category, word)
      @categories[category].delete(word)
    end

    def word_in_category?(category, word)
      @categories[category].key?(word)
    end

    def reset
      initialize
    end

    private

    def category_counts(category)
      @category_counts[category] ||= { training: 0, word: 0 }
    end
  end
end
