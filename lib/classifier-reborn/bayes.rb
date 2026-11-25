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

    # Cost used for a reject action that is not given an explicit cost.
    DEFAULT_REJECT_COST = 1.0

    # Accepted orientations for a cost table.
    #
    #   :action   rows are keyed by candidate action and each row maps the true
    #             category to the cost of that action (the default)
    #   :category rows are keyed by true category and each row maps the
    #             candidate action to its cost
    COST_TABLE_ORIENTATIONS = %i[action category].freeze

    # Expected costs are rounded so equal decisions compare cleanly instead of
    # exposing floating point noise from the softmax.
    RISK_PRECISION = 10

    # Alternate keyword spellings for the decision options. The canonical names
    # match the accessors, and the extras cover the vocabulary a caller might
    # reasonably use for the same concept.
    DECISION_OPTION_ALIASES = {
      actions: :cost_table,
      action_costs: :cost_table,
      costs: :cost_table,
      cost_matrix: :cost_table,
      decision_costs: :cost_table,
      orientation: :cost_table_orientation,
      reject: :reject_action,
      abstain: :reject_action,
      abstain_action: :reject_action,
      reject_cost: :reject_costs,
      abstain_costs: :reject_costs
    }.freeze

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
    #   cost_table:       nil                     Optional cost table enabling minimum expected cost decisions
    #   reject_action:    nil                     Optional reject (abstain) action that competes with the cost table actions
    #   reject_costs:     nil                     Cost of the reject action, either a number or per category
    #   cost_table_orientation: :action           Whether the cost table is keyed by action or by true category
    def initialize(*args)
      @initial_categories = []
      options = { language: 'en',
                  enable_threshold: false,
                  threshold: 0.0,
                  enable_stemmer: true,
                  backend: BayesMemoryBackend.new,
                  cost_table: nil,
                  reject_action: nil,
                  reject_costs: nil,
                  cost_table_orientation: :action }
      args.flatten.each do |arg|
        if arg.is_a?(Hash)
          options.merge!(canonical_decision_options(arg))
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

      @cost_table_orientation = options[:cost_table_orientation]
      unless COST_TABLE_ORIENTATIONS.include?(@cost_table_orientation)
        raise ArgumentError, "cost_table_orientation must be one of #{COST_TABLE_ORIENTATIONS.inspect}"
      end

      @reject_action = options[:reject_action]
      @cost_table_source = options[:cost_table]
      @reject_costs = options[:reject_costs]

      populate_initial_categories

      custom_stopwords options[:stopwords] if options.key?(:stopwords)
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

    # Return the classification without the score.
    #
    # Without a cost table this keeps the historical behaviour: the category
    # with the highest score, filtered by the optional threshold.
    #
    # When a cost table is configured (on the classifier or per call) the
    # returned value is instead the candidate action with the lowest expected
    # cost, which may be a reject action provided by the caller. The threshold
    # is not applied to cost based decisions.
    #
    #     b.classify "I hate bad words and you"
    #     => 'Uninteresting'
    #
    #     b = ClassifierReborn::Bayes.new 'Review', 'Direct',
    #           cost_table: { 'Review' => [0, 1], 'Direct' => [10, 0] }
    #     b.classify "I hate bad words and you"
    #     => 'Review'
    def classify(text, cost_table = nil, **options)
      options = canonical_decision_options(options)
      options[:cost_table] = cost_table unless cost_table.nil?
      return legacy_classify(text) if decision_cost_table(options, categories).nil?

      classify_with_risk(text, **options).first
    end

    # Returns the lowest expected cost action together with the expected cost
    # of every candidate action. The source is a text or a score hash from
    # #classifications. E.g.,
    #    b.classify_with_risk "I hate bad words and you"
    #    => ['Review', {'Review'=>0.8, 'Direct'=>2.0}]
    def classify_with_risk(source, cost_table = nil, **options)
      options = canonical_decision_options(options)
      options[:cost_table] = cost_table unless cost_table.nil?
      scores = source.is_a?(Hash) ? source : classifications(source)
      risks = computed_costs(scores, options, scores.keys)
      [minimum_risk_action_from(risks), rounded_costs(risks)]
    end

    # Returns only the action with the smallest expected cost.
    def minimum_risk_action(source, cost_table = nil, **options)
      classify_with_risk(source, cost_table, **options).first
    end

    # Returns the expected cost of each candidate action.
    #
    # The source is either a text (which is classified with #classifications)
    # or the score hash returned by #classifications. Log scores are turned
    # into probabilities with a max shifted softmax, so moving every score by
    # the same offset does not change the probabilities or the risks.
    def expected_costs(source, cost_table = nil, **options)
      options = canonical_decision_options(options)
      options[:cost_table] = cost_table unless cost_table.nil?

      scores = source.is_a?(Hash) ? source : classifications(source)
      rounded_costs(computed_costs(scores, options, scores.keys))
    end

    alias risk_costs expected_costs
    alias risks expected_costs
    alias expected_risk expected_costs

    # Returns the probabilities implied by the log scores: a max shifted
    # softmax, so a common offset over every score cancels out.
    def weights(source)
      scores = source.is_a?(Hash) ? source : classifications(source)
      probabilities_for(scores).transform_values { |weight| weight.round(RISK_PRECISION) }
    end

    alias probabilities weights

    # Returns the cost table as given, keyed by candidate action.
    attr_reader :cost_table_source

    # Returns the expanded cost table, keyed by action and then by the true
    # category, or nil when no table is configured.
    def cost_table(**options)
      decision_cost_table(canonical_decision_options(options), categories)
    end

    # Replaces the cost table used for minimum risk decisions.
    def cost_table=(table)
      @cost_table_source = table
    end

    attr_reader :reject_action, :reject_costs

    # Sets (or clears) the reject action that competes with the cost table.
    def reject_action=(action)
      @reject_action = action
    end

    # Sets the cost of the reject action, either as a single number, a per
    # category hash or a per category array.
    def reject_costs=(costs)
      @reject_costs = costs
    end

    # Removes any configured cost table and reject action.
    def clear_cost_table
      @cost_table_source = nil
      @reject_costs = nil
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
    end

    private

    def populate_initial_categories
      @initial_categories.each do |c|
        add_category(c)
      end
    end

    # Historical classification path: highest score, optional threshold.
    def legacy_classify(text)
      result, score = classify_with_score(text)
      result = nil if threshold_enabled? && (score < @threshold || score == Float::INFINITY)
      result
    end

    # Resolves the cost table for a single decision, applying per call
    # overrides and appending the reject action when one is configured.
    #
    # Returns nil when the caller did not supply a cost table and none was
    # given to the constructor, which keeps #classify on the legacy path.
    def decision_cost_table(options, categories)
      source = options.key?(:cost_table) ? options[:cost_table] : @cost_table_source
      return nil if source.nil?

      table = expand_cost_table(source, options, categories)

      reject = options.key?(:reject_action) ? options[:reject_action] : @reject_action
      return table if reject.nil?

      reject_costs = options.key?(:reject_costs) ? options[:reject_costs] : @reject_costs
      # A reject action given without any cost defaults to DEFAULT_REJECT_COST.
      table[reject] = reject_cost_row(reject, reject_costs.nil? ? DEFAULT_REJECT_COST : reject_costs, categories)
      table
    end

    # Expected cost of every action before rounding. The winner is chosen from
    # these values so rounding can never reorder near-ties.
    def computed_costs(scores, options, categories)
      table = decision_cost_table(options, categories)
      raise ArgumentError, 'no cost table is configured' if table.nil?

      probabilities = probabilities_for(scores)
      table.each_with_object({}) do |(action, row), risks|
        risks[action] = probabilities.reduce(0.0) do |sum, (category, probability)|
          sum + (probability * cost_for(row, category, action))
        end
      end
    end

    def rounded_costs(risks)
      risks.transform_values { |risk| risk.round(RISK_PRECISION) }
    end

    # Normalizes the loose decision option spellings into the canonical keys.
    def canonical_decision_options(options)
      options.each_with_object({}) do |(key, value), canonical|
        canonical[DECISION_OPTION_ALIASES.fetch(key, key)] = value
      end
    end

    # Expands the caller's table into an action keyed table whose rows map the
    # true category to the cost of picking that action for it.
    #
    # Each action's expected cost therefore sums over every true category,
    # which is what makes the decision a proper expected risk.
    def expand_cost_table(source, options, categories)
      orientation = options.fetch(:cost_table_orientation, @cost_table_orientation)
      unless COST_TABLE_ORIENTATIONS.include?(orientation)
        raise ArgumentError, "cost_table_orientation must be one of #{COST_TABLE_ORIENTATIONS.inspect}"
      end

      rows = orientation == :category ? transpose_cost_rows(source) : source
      rows.each_with_object({}) do |(action, costs), table|
        table[action] = cost_row(costs, categories, action)
      end
    end

    # Flips a table keyed by true category into one keyed by action.
    def transpose_cost_rows(source)
      source.each_with_object({}) do |(category, costs), table|
        unless costs.is_a?(Hash)
          raise ArgumentError, "costs for category #{category.inspect} must map actions to costs"
        end

        costs.each { |action, cost| (table[action] ||= {})[category] = cost }
      end
    end

    def cost_row(costs, categories, action)
      case costs
      when Hash
        category_cost_row(costs)
      when Array
        positional_cost_row(costs, categories, action)
      else
        # A bare number is the cost of choosing this action for every category.
        flat_cost_row(costs, categories)
      end
    end

    # Positional rows line up with the classifier's categories, which is the
    # order of the keys returned by #classifications.
    def positional_cost_row(costs, categories, action)
      if costs.length != categories.length
        raise ArgumentError,
              "action #{action.inspect} has #{costs.length} costs, expected #{categories.length} " \
              "(one per category: #{categories.inspect})"
      end

      categories.each_with_index.each_with_object({}) do |(category, index), row|
        row[category] = cost_value(costs[index])
      end
    end

    def category_cost_row(costs)
      costs.each_with_object({}) do |(category, cost), row|
        row[category] = cost_value(cost)
        row[category.to_s] ||= cost_value(cost)
      end
    end

    def flat_cost_row(cost, categories)
      value = cost_value(cost)
      categories.each_with_object({}) { |category, row| row[category] = value }
    end

    def reject_cost_row(action, costs, categories)
      return category_cost_row(costs) if costs.is_a?(Hash)
      return positional_cost_row(costs, categories, action) if costs.is_a?(Array)

      flat_cost_row(costs, categories)
    end

    def cost_value(value)
      Float(value)
    rescue TypeError, ArgumentError
      raise ArgumentError, "cost #{value.inspect} is not a number"
    end

    def cost_for(row, category, action)
      return row[category] if row.key?(category)
      return row[category.to_s] if row.key?(category.to_s)

      raise ArgumentError,
            "the cost table has no cost for #{category.inspect} under action #{action.inspect}"
    end

    # Converts log scores into a probability distribution. Scores are shifted
    # by their maximum before exponentiating so a common offset cancels out
    # and the result stays finite even when the scores are very negative.
    def probabilities_for(scores)
      finite = scores.reject { |_category, score| score.nil? || !score.to_f.finite? }
      return uniform_probabilities(scores.keys) if finite.empty?

      maximum = finite.values.map(&:to_f).max
      shifted = scores.each_with_object({}) do |(category, score), mass|
        value = score.to_f
        mass[category] = value.finite? ? Math.exp(value - maximum) : 0.0
      end
      total = shifted.values.sum
      return uniform_probabilities(scores.keys) if total <= 0.0

      shifted.each_with_object({}) { |(category, mass), result| result[category] = mass / total }
    end

    def uniform_probabilities(categories)
      probability = categories.empty? ? 0.0 : 1.0 / categories.length
      categories.each_with_object({}) { |category, result| result[category] = probability }
    end

    # Picks the action with the smallest expected cost. Ties keep the first
    # action in table order.
    def minimum_risk_action_from(risks)
      risks.min_by { |_action, risk| risk }&.first
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
