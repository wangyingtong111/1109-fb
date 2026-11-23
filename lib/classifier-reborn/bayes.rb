# frozen_string_literal: true

# Author::    Lucas Carlson  (mailto:lucas@rufy.com)
# Copyright:: Copyright (c) 2005 Lucas Carlson
# License::   LGPL

require 'set'

require_relative 'extensions/tokenizer/whitespace'
require_relative 'extensions/token_filter/stopword'
require_relative 'extensions/token_filter/stemmer'
require_relative 'category_namer'
require_relative 'backends/bayes_memory_backend'
require_relative 'backends/bayes_redis_backend'

module ClassifierReborn
  class Bayes
    CategoryNotFoundError = Class.new(StandardError)

    # Raised when classifiers that do not share the same feature extraction
    # contract (tokenizer, token filters, stemmer, stopwords) are merged.
    IncompatibleClassifierError = Class.new(StandardError)

    # The class can be created with one or more categories, each of which will be
    # initialized and given a training method. E.g.,
    #      b = ClassifierReborn::Bayes.new 'Interesting', 'Uninteresting', 'Spam'
    #
    # Options available are:
    #   language:         'en'                    Used to select language specific stop words
    #   auto_categorize:  false                   When true, enables ability to dynamically declare a category; the default is true if no initial categories are provided
    #   enable_threshold: false                   When true, enables a threshold requirement for classifition
    #   threshold:        0.0                     Default threshold, only used when enabled
    #   enable_stemmer:   true                    When false, disables word stemming
    #   stopwords:        nil                     Accepts path to a text file or an array of words, when supplied, overwrites the default stopwords; assign empty string or array to disable stopwords
    #   backend:          BayesMemoryBackend.new  Alternatively, BayesRedisBackend.new for persistent storage
    def initialize(*args)
      @initial_categories = []
      options = { language: 'en',
                  enable_threshold: false,
                  threshold: 0.0,
                  enable_stemmer: true,
                  backend: BayesMemoryBackend.new }
      args.flatten.each do |arg|
        if arg.is_a?(Hash)
          options.merge!(arg)
        else
          @initial_categories.push(arg)
        end
      end

      unless options.key?(:auto_categorize)
        options[:auto_categorize] = @initial_categories.empty? ? true : false
      end

      @language            = options[:language]
      @auto_categorize     = options[:auto_categorize]
      @enable_threshold    = options[:enable_threshold]
      @threshold           = options[:threshold]
      @enable_stemmer      = options[:enable_stemmer]
      @backend             = options[:backend]
      @tokenizer           = options[:tokenizer] || Tokenizer::Whitespace
      @token_filters       = options[:token_filters] || [TokenFilter::Stopword]
      @token_filters << TokenFilter::Stemmer if @enable_stemmer && !@token_filters.include?(TokenFilter::Stemmer)
      TokenFilter::Stopword.language = @language if @token_filters.include?(TokenFilter::Stopword)

      populate_initial_categories

      custom_stopwords options[:stopwords] if options.key?(:stopwords)

      @stopword_snapshot = stopword_snapshot
    end

    # Provides a general training method for all categories specified in Bayes#new
    # For example:
    #     b = ClassifierReborn::Bayes.new 'This', 'That', 'the_other'
    #     b.train :this, "This text"
    #     b.train "that", "That text"
    #     b.train "The other", "The other text"
    def train(category, text)
      word_hash = Hasher.word_hash(text, @enable_stemmer,
                                   tokenizer: @tokenizer, token_filters: @token_filters)
      return if word_hash.empty?

      category = CategoryNamer.prepare_name(category)

      # Add the category dynamically or raise an error
      unless category_keys.include?(category)
        if @auto_categorize
          add_category(category)
        else
          raise CategoryNotFoundError, "Cannot train; category #{category} does not exist"
        end
      end

      word_hash.each do |word, count|
        @backend.update_category_word_frequency(category, word, count)
        @backend.update_category_word_count(category, count)
        @backend.update_total_words(count)
      end
      @backend.update_total_trainings(1)
      @backend.update_category_training_count(category, 1)
    end

    # Provides a untraining method for all categories specified in Bayes#new
    # Be very careful with this method.
    #
    # For example:
    #     b = ClassifierReborn::Bayes.new 'This', 'That', 'the_other'
    #     b.train :this, "This text"
    #     b.untrain :this, "This text"
    def untrain(category, text)
      word_hash = Hasher.word_hash(text, @enable_stemmer,
                                   tokenizer: @tokenizer, token_filters: @token_filters)
      return if word_hash.empty?

      category = CategoryNamer.prepare_name(category)
      word_hash.each do |word, count|
        next if @backend.total_words < 0

        orig = @backend.category_word_frequency(category, word) || 0
        @backend.update_category_word_frequency(category, word, -count)
        if @backend.category_word_frequency(category, word) <= 0
          @backend.delete_category_word(category, word)
          count = orig
        end

        @backend.update_category_word_count(category, -count) if @backend.category_word_count(category) >= count
        @backend.update_total_words(-count)
      end
      @backend.update_total_trainings(-1)
      @backend.update_category_training_count(category, -1)
    end

    # Returns the scores in each category the provided +text+. E.g.,
    #    b.classifications "I hate bad words and you"
    #    =>  {"Uninteresting"=>-12.6997928013932, "Interesting"=>-18.4206807439524}
    # The largest of these scores (the one closest to 0) is the one picked out by #classify
    def classifications(text)
      score = {}
      word_hash = Hasher.word_hash(text, @enable_stemmer,
                                   tokenizer: @tokenizer, token_filters: @token_filters)
      if word_hash.empty?
        category_keys.each do |category|
          score[category.to_s] = Float::INFINITY
        end
        return score
      end
      category_keys.each do |category|
        score[category.to_s] = 0
        total = (@backend.category_word_count(category) || 1).to_f
        word_hash.each do |word, _count|
          s = @backend.word_in_category?(category, word) ? @backend.category_word_frequency(category, word) : 0.1
          score[category.to_s] += Math.log(s / total)
        end
        # now add prior probability for the category
        s = @backend.category_has_trainings?(category) ? @backend.category_training_count(category) : 0.1
        score[category.to_s] += Math.log(s / @backend.total_trainings.to_f)
      end
      score
    end

    # Returns the classification of the provided +text+, which is one of the
    # categories given in the initializer along with the score. E.g.,
    #    b.classify "I hate bad words and you"
    #    =>  ['Uninteresting', -4.852030263919617]
    def classify_with_score(text)
      (classifications(text).sort_by { |a| -a[1] })[0]
    end

    # Return the classification without the score
    def classify(text)
      result, score = classify_with_score(text)
      result = nil if threshold_enabled? && (score < @threshold || score == Float::INFINITY)
      result
    end

    # Retrieve the current threshold value
    attr_reader :threshold

    # Dynamically set the threshold value
    attr_writer :threshold

    # Dynamically enable threshold for classify results
    def enable_threshold
      @enable_threshold = true
    end

    # Dynamically disable threshold for classify results
    def disable_threshold
      @enable_threshold = false
    end

    # Is threshold processing enabled?
    def threshold_enabled?
      @enable_threshold
    end

    # is threshold processing disabled?
    def threshold_disabled?
      !@enable_threshold
    end

    # Is word stemming enabled?
    def stemmer_enabled?
      @enable_stemmer
    end

    # Is word stemming disabled?
    def stemmer_disabled?
      !@enable_stemmer
    end

    # Provides training and untraining methods for the categories specified in Bayes#new
    # For example:
    #     b = ClassifierReborn::Bayes.new 'This', 'That', 'the_other'
    #     b.train_this "This text"
    #     b.train_that "That text"
    #     b.untrain_that "That text"
    #     b.train_the_other "The other text"
    def method_missing(name, *args)
      cleaned_name = name.to_s.gsub(/(un)?train_([\w]+)/, '\2')
      category = CategoryNamer.prepare_name(cleaned_name)
      if category_keys.include?(category)
        args.each { |text| eval("#{Regexp.last_match(1)}train(category, text)") }
      elsif name.to_s =~ /(un)?train_([\w]+)/
        raise StandardError, "No such category: #{category}"
      else
        super # raise StandardError, "No such method: #{name}"
      end
    end

    # Provides a list of category names
    # For example:
    #     b.categories
    #     =>   ["This", "That", "The other"]
    def categories
      category_keys.collect(&:to_s)
    end

    # Provides a list of category keys as symbols
    # For example:
    #     b.categories
    #     =>   [:This, :That, :"The other"]
    def category_keys
      @backend.category_keys
    end

    # Allows you to add categories to the classifier.
    # For example:
    #     b.add_category "Not spam"
    #
    # WARNING: Adding categories to a trained classifier will
    # result in an undertrained category that will tend to match
    # more criteria than the trained selective categories. In short,
    # try to initialize your categories at initialization.
    def add_category(category)
      category = CategoryNamer.prepare_name(category)
      @backend.add_category(category)
    end

    alias append_category add_category

    def reset
      @backend.reset
      populate_initial_categories
      @stopword_snapshot = stopword_snapshot
    end

    # Combines the training statistics of +others+ with those of this
    # classifier and returns a new classifier holding the sum. The result is
    # exactly the model a single classifier would have become after training on
    # the documents of every shard, provided all shards share the same feature
    # extraction contract (tokenizer, token filters, stemmer, stopwords and
    # category naming rules); otherwise IncompatibleClassifierError is raised.
    #
    # The receivers are never modified: every shard keeps its own statistics
    # and remains usable for classification and training by its other callers.
    #
    #     merged = shard_a.merge(shard_b)
    #
    # The Redis backend stores everything in one database, so a destination
    # backend cannot be derived for it; pass an empty one explicitly:
    #
    #     merged = shard_a.merge(shard_b, into: BayesRedisBackend.new(db: 2))
    def merge(*others, into: nil)
      shards = [self, *others.flatten.compact]
      raise ArgumentError, 'merge needs at least one other classifier' if shards.size < 2

      shards.each { |shard| assert_mergeable(shard) }
      reused = shards.map { |shard| shard.backend.object_id }
      raise ArgumentError, 'the same classifier cannot be merged twice' if reused.uniq.size != shards.size

      merged = build_merge_shell(resolve_merge_destination(shards, into))
      merged.merge_from(shards)
      merged
    end

    alias combine merge

    # Class level shortcut: Bayes.merge(shard_a, shard_b, shard_c)
    def self.merge(*classifiers, into: nil)
      classifiers = classifiers.flatten.compact
      raise ArgumentError, 'merge needs at least two classifiers' if classifiers.size < 2

      first, *rest = classifiers
      first.merge(*rest, into: into)
    end

    # Read-only accessors describing this classifier, used to decide whether
    # its statistics can be added to another classifier's.
    attr_reader :backend, :language, :tokenizer

    def token_filters
      @token_filters.dup
    end

    # The categories handed to the constructor; the merged model resets to the
    # union of the shards' ones.
    attr_reader :initial_categories
    protected :initial_categories

    private

    def assert_mergeable(shard)
      raise ArgumentError, "cannot merge #{shard.class}: expected a #{self.class}" unless shard.is_a?(self.class)

      assert_stopword_table_unchanged(shard)
      return if merge_signature == shard.merge_signature && labels_are_canonical?(shard)

      raise IncompatibleClassifierError,
            'cannot merge classifiers that extract features differently ' \
            '(tokenizer, token filters, stemmer or stopwords differ)'
    end

    # The stopword table lives on the filter, not on the classifier. If another
    # classifier replaced it after this shard was created, the shard would now
    # filter differently than it did while training, and so would the merged
    # model: its statistics could no longer be reproduced.
    def assert_stopword_table_unchanged(shard)
      trained_with = shard.effective_stopword_snapshot
      return if trained_with.nil? || trained_with == shard.stopword_snapshot

      raise IncompatibleClassifierError,
            'cannot merge: the stopwords changed after the classifier was created, ' \
            'so its statistics no longer describe the current feature space'
    end

    # Categories are always stored under the symbol produced by the naming
    # rules, so merging the raw keys reproduces the labelling a single model
    # would have produced. A shard holding any other kind of key (for example
    # one written by legacy code) could silently split one label in two, so it
    # is rejected instead of being merged.
    def labels_are_canonical?(shard)
      shard.category_keys.all? { |category| category.is_a?(Symbol) }
    end

    def resolve_merge_destination(shards, destination)
      destination ||= @backend.merge_destination
      unless destination.respond_to?(:absorb) && destination.respond_to?(:total_words)
        raise ArgumentError, 'the merge destination must be a Bayes backend'
      end
      if shards.any? { |shard| shard.backend.equal?(destination) }
        raise ArgumentError, 'merge destination must be a backend no shard is using'
      end
      unless destination.total_words.zero? && destination.total_trainings.zero? && destination.category_keys.empty?
        raise ArgumentError, 'merge destination backend must be empty'
      end

      destination
    end

    def build_merge_shell(destination)
      self.class.new(merge_options.merge(backend: destination))
    end

    def merge_options
      { language: @language,
        auto_categorize: @auto_categorize,
        enable_threshold: @enable_threshold,
        threshold: @threshold,
        enable_stemmer: @enable_stemmer,
        tokenizer: @tokenizer,
        token_filters: @token_filters.dup }
    end

    # Snapshot of the stopwords this classifier was created with. The stopword
    # table itself is process wide, so two shards created with different lists
    # did not filter the same way even though they now observe the same table.
    def stopword_snapshot
      return nil unless @token_filters.include?(TokenFilter::Stopword)

      Set.new(TokenFilter::Stopword::STOPWORDS[@language])
    end

    # The stopword list the shard really filtered with. Instances restored from
    # a snapshot taken before the list was recorded fall back to the current
    # one, so old marshaled classifiers stay mergeable.
    def effective_stopword_snapshot
      return stopword_snapshot unless defined?(@stopword_snapshot) && @stopword_snapshot

      @stopword_snapshot
    end

    # The filter chain Hasher.word_hash actually applies: the stemmer is
    # appended when stemming is enabled and dropped when it is not, so the
    # effective chain is what defines the feature space.
    def feature_token_filters
      filters = @token_filters.dup
      if filters.include?(TokenFilter::Stemmer)
        filters.delete(TokenFilter::Stemmer) unless @enable_stemmer
      elsif @enable_stemmer
        filters << TokenFilter::Stemmer
      end
      filters
    end

    # Copies the statistics of +shards+ into this classifier's backend. Used by
    # #merge on the freshly built result; the shards are only read.
    def merge_from(shards)
      @backend.absorb(*shards.map { |shard| shard.backend })
      @initial_categories = shards.flat_map { |shard| shard.initial_categories }.uniq
      self
    end

    # Everything that decides how a piece of text becomes counted features. Two
    # shards may only be added when all of it agrees, because otherwise their
    # word frequencies would not live in the same feature space.
    def merge_signature
      {
        language: @language,
        tokenizer: @tokenizer,
        token_filters: feature_token_filters,
        stopwords: effective_stopword_snapshot
      }
    end

    protected :merge_from, :merge_signature, :assert_mergeable,
              :stopword_snapshot, :effective_stopword_snapshot

    def populate_initial_categories
      @initial_categories.each do |c|
        add_category(c)
      end
    end

    # Overwrites the default stopwords for current language with supplied list of stopwords or file
    def custom_stopwords(stopwords)
      unless stopwords.is_a?(Enumerable)
        if stopwords.strip.empty?
          stopwords = []
        elsif File.exist?(stopwords)
          stopwords = File.read(stopwords).force_encoding('utf-8').split
        else
          return # Do not overwrite the default
        end
      end
      TokenFilter::Stopword::STOPWORDS[@language] = Set.new stopwords
    end
  end
end
