# frozen_string_literal: true

module ClassifierReborn
  # Shared implementation of the additive statistics every Bayes backend keeps.
  #
  # The naive Bayes model is fully described by sufficient statistics that are
  # plain sums over the training documents: the number of trainings, the total
  # number of tokens, and per category the number of trainings, the number of
  # tokens and the frequency of every word. Adding the statistics of
  # independent shards therefore yields exactly the state a single model would
  # have reached after training on all the documents.
  #
  # #absorb only uses the reader/writer methods of the backend interface, so it
  # works for the memory and the Redis backend alike, and it never touches the
  # storage of the backends it reads from.
  module MergeableBackend
    # Adds the statistics of +others+ to this backend and returns self.
    #
    # It is meant to be called on an empty destination backend: the given
    # backends are only read, while the receiver is updated. Complete shards
    # are never modified, so their owners can keep classifying and training
    # with them.
    def absorb(*others)
      others.each do |other|
        raise ArgumentError, 'cannot absorb a backend into itself' if other.equal?(self)

        absorb_statistics_of(other)
      end
      self
    end

    # Returns a new, empty backend that stores its statistics the same way as
    # this one. Backends whose storage location cannot be derived safely (for
    # example a Redis database, which would otherwise be merged into itself)
    # override this method to ask for an explicit destination.
    def merge_destination
      self.class.new
    end

    private

    def absorb_statistics_of(other)
      update_total_trainings(other.total_trainings)
      update_total_words(other.total_words)

      other.category_keys.each do |category|
        add_category(category)
        update_category_training_count(category, other.category_training_count(category))
        update_category_word_count(category, other.category_word_count(category))
        absorb_word_frequencies_of(other, category)
      end
    end

    def absorb_word_frequencies_of(other, category)
      other.category_word_frequencies(category).each do |word, count|
        next if count.zero?

        update_category_word_frequency(category, word, count)
      end
    end
  end
end