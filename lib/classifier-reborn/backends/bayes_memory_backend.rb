# frozen_string_literal: true

module ClassifierReborn
  class BayesMemoryBackend
    attr_reader :total_words, :total_trainings

    # This class provides Memory as the storage backend for the classifier data structures
    def initialize
      @total_words      = 0
      @total_trainings  = 0
      @category_counts  = {}
      @categories       = {}
      @training_records = {}
    end

    def update_total_words(diff)
      @total_words += diff
    end

    def update_total_trainings(diff)
      @total_trainings += diff
    end

    def category_training_count(category)
      category_counts(category)[:training]
    end

    def update_category_training_count(category, diff)
      category_counts(category)[:training] += diff
    end

    def category_has_trainings?(category)
      @category_counts.key?(category) && category_training_count(category) > 0
    end

    def category_word_count(category)
      category_counts(category)[:word]
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
      @categories[category][word]
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

    # Records one training contribution of the sample identified by +sample_key+
    # to +category+. The same sample can be trained more than once, in which
    # case each training is a separate, revocable contribution.
    def record_training(category, sample_key)
      records = training_records(category)
      records[sample_key] = records.fetch(sample_key, 0) + 1
    end

    # Returns how many trained contributions of +sample_key+ are recorded for
    # +category+ (0 when the sample was never trained for that category).
    def training_record_count(category, sample_key)
      records = @training_records[category]
      records ? records.fetch(sample_key, 0) : 0
    end

    # Removes a single recorded contribution of +sample_key+ from +category+ and
    # returns the number of contributions left. Nothing is removed when there is
    # no contribution to remove.
    def delete_training(category, sample_key)
      records = @training_records[category]
      return 0 if records.nil? || records.fetch(sample_key, 0) <= 0

      remaining = records[sample_key] - 1
      if remaining > 0
        records[sample_key] = remaining
      else
        records.delete(sample_key)
      end
      remaining
    end

    def reset
      initialize
    end

    private

    def category_counts(category)
      @category_counts[category] ||= { training: 0, word: 0 }
    end

    def training_records(category)
      @training_records[category] ||= {}
    end
  end
end
