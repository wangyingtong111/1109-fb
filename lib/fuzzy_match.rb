require 'fuzzy_match/rule'
require 'fuzzy_match/rule/grouping'
require 'fuzzy_match/rule/identity'
require 'fuzzy_match/result'
require 'fuzzy_match/record'
require 'fuzzy_match/similarity'
require 'fuzzy_match/score'

# See the README for more information.
class FuzzyMatch
  class << self
    def engine
      @engine
    end
    
    def engine=(alt_engine)
      @engine = alt_engine
    end
    
    def score_class
      case engine
      when :pure_ruby
        Score::PureRuby
      when :amatch
        Score::Amatch
      else
        raise ::ArgumentError, "[fuzzy_match] #{engine.inspect} is not a recognized engine."
      end
    end
  end
  
  DEFAULT_ENGINE = :pure_ruby
  
  #TODO refactor at least all the :find_X things
  DEFAULT_OPTIONS = {
    :must_match_grouping => false,
    :must_match_at_least_one_word => false,
    :gather_last_result => false,
    :find_all => false,
    :find_all_with_score => false,
    :threshold => nil,
    :find_best => false,
    :find_with_score => false,
  }

  self.engine = DEFAULT_ENGINE
  
  attr_reader :haystack
  attr_reader :source_haystack
  attr_reader :groupings
  attr_reader :identities
  attr_reader :stop_words
  attr_reader :id_reader
  attr_accessor :read
  attr_reader :default_options

  # haystack - a bunch of records that will compete to see who best matches the needle
  #
  # Rules (can only be specified at initialization or by using a setter)
  # * :<tt>identities</tt> - regexps
  # * :<tt>groupings</tt> - regexps
  # * :<tt>stop_words</tt> - regexps
  # * :<tt>read</tt> - how to interpret each record in the 'haystack', either a Proc or a symbol
  #
  # Options (can be specified at initialization or when calling #find)
  # * :<tt>must_match_grouping</tt> - don't return a match unless the needle fits into one of the groupings you specified
  # * :<tt>must_match_at_least_one_word</tt> - don't return a match unless the needle shares at least one word with the match
  # * :<tt>gather_last_result</tt> - enable <tt>last_result</tt>
  # * :<tt>threshold</tt> - set a score threshold below which not to return results (not generally recommended - please test the results of setting a threshold thoroughly - one set of results and their scores probably won't be enough to determine the appropriate number). Only checked against the Pair Distance score and ignored when one string or the other is of length 1.
  def initialize(haystack, options_and_rules = {})
    o = options_and_rules.dup

    # rules
    @read = o.delete(:read) || o.delete(:haystack_reader)
    @id_reader = o.delete(:id) || o.delete(:record_id) || o.delete(:id_key) ||
                 o.delete(:primary_key) || o.delete(:record_id_key)
    @groupings = (o.delete(:groupings) || o.delete(:blockings) || []).map { |regexp| Rule::Grouping.make(regexp) }.flatten
    @identities = (o.delete(:identities) || []).map { |regexp| Rule::Identity.new(regexp) }
    @stop_words = o.delete(:stop_words) || []
    
    # options
    if deprecated = o.delete(:must_match_blocking)
      o[:must_match_grouping] = deprecated
    end
    @default_options = DEFAULT_OPTIONS.merge(o).freeze

    @source_haystack = haystack
    @haystack = haystack.map { |original| Record.new original, :stop_words => @stop_words, :read => @read }
  end

  # Refresh one already-entered candidate by its stable record ID.
  #
  # The replacement's whole/clean/word projections are built completely before
  # the haystack slot is changed.  If reading or projecting it fails, this
  # method returns the exception and the previous projection stays active.
  # Use #refresh_record! if you prefer exceptions to propagate.
  def refresh_record(*arguments, &block)
    refresh_record!(*arguments, &block)
    true
  rescue => error
    error
  end
  alias refresh refresh_record
  alias refresh_by_id refresh_record
  alias refresh_by_record_id refresh_record
  alias refresh_record_by_id refresh_record
  alias reload_record refresh_record
  alias reload refresh_record
  alias reload_by_id refresh_record
  alias reread_record refresh_record
  alias reread refresh_record
  alias reread_by_id refresh_record
  alias replace_record refresh_record
  alias update_record refresh_record

  def refresh_record!(*arguments, &block)
    record_id, replacement, reader, options = parse_refresh_arguments(arguments, block)
    custom_id_reader = options[:id_key] || options[:primary_key] || options[:record_id_key] || self.id_reader
    slot = record_slot_for_id(record_id, custom_id_reader)
    old_record = haystack[slot]

    source = if replacement.nil?
      current_source_record(record_id, custom_id_reader) || old_record.original
    else
      replacement
    end
    candidate = Record.new(source, :read => reader || read, :stop_words => stop_words)
    candidate.project!

    haystack[slot] = candidate
    candidate
  end
  alias refresh! refresh_record!
  alias refresh_by_id! refresh_record!
  alias reload_record! refresh_record!
  alias reload! refresh_record!
  alias reread_record! refresh_record!
  alias replace_record! refresh_record!
  alias update_record! refresh_record!

  def last_result
    @last_result or raise("You can't access the last result until you've run a find with :gather_last_result => true")
  end
  
  # Return everything in sorted order
  def find_all(needle, options = {})
    options = options.merge(:find_all => true)
    find needle, options
  end

  # Return the top results with the same score
  def find_best(needle, options = {})
    options = options.merge(:find_best => true)
    find needle, options
  end

  # Return everything in sorted order with score
  def find_all_with_score(needle, options = {})
    options = options.merge(:find_all_with_score => true)
    find needle, options
  end

  # Return one with score
  def find_with_score(needle, options = {})
    options = options.merge(:find_with_score => true)
    find needle, options
  end
  
  def find(needle, options = {})
    options = default_options.merge options
    
    threshold = options[:threshold]
    gather_last_result = options[:gather_last_result]
    is_find_all_with_score = options[:find_all_with_score]
    is_find_with_score = options[:find_with_score]
    is_find_best = options[:find_best]
    is_find_all = options[:find_all] || is_find_all_with_score || is_find_best
    must_match_grouping = options[:must_match_grouping]
    must_match_at_least_one_word = options[:must_match_at_least_one_word]
    
    if gather_last_result
      @last_result = Result.new
      last_result.read = read
      last_result.haystack = haystack
      last_result.options = options
    end
    
    if gather_last_result
      last_result.identities = identities
      last_result.groupings = groupings
      last_result.stop_words = stop_words
    end
    
    needle = case needle
    when String
      Record.new needle, :stop_words => stop_words
    else
      Record.new needle, :read => read, :stop_words => stop_words
    end
    
    if gather_last_result
      last_result.needle = needle
    end

    if groupings.any?
      first_grouping = groupings.detect { |grouping| grouping.xmatch? needle }
      if gather_last_result
        if first_grouping
          last_result.timeline << "Grouping: #{first_grouping.inspect}"
        else
          last_result.timeline << "No grouping."
        end
      end
    end

    if must_match_grouping and not first_grouping
      if gather_last_result
        last_result.timeline << <<-EOS
The needle didn't match any of the #{groupings.length} groupings, which was a requirement.
\t#{groupings.map(&:inspect).join("\n\t")}
EOS
      end
      if is_find_all
        return []
      else
        return nil
      end
    end

    if groupings.any? and not first_grouping
      passed_grouping_requirement = haystack.reject do |straw|
        groupings.any? { |grouping| grouping.xmatch? straw }
      end
    else
      passed_grouping_requirement = haystack
    end

    if must_match_at_least_one_word
      passed_word_requirement = passed_grouping_requirement.select do |straw|
        (needle.words & straw.words).any?
      end
      if gather_last_result
        last_result.timeline << <<-EOS
Since :must_match_at_least_one_word => true, the competition was reduced to records sharing at least one word with the needle.
\tNeedle words: #{needle.words.map(&:inspect).join(', ')}
\tPassed (first 3): #{passed_word_requirement[0,3].map(&:inspect).join(', ')}
\tFailed (first 3): #{(passed_grouping_requirement-passed_word_requirement)[0,3].map(&:inspect).join(', ')}
EOS
      end
    else
      passed_word_requirement = passed_grouping_requirement
    end
        
    if first_grouping
      joint = passed_word_requirement.select do |straw|
        first_grouping.xjoin? needle, straw
      end
      # binding.pry      
      if gather_last_result
        last_result.timeline << <<-EOS
Since there were groupings, the competition was reduced to #{joint.length} records in the same group as the needle.
\t#{joint.map(&:inspect).join("\n\t")}
EOS
      end
    else
      joint = passed_word_requirement.dup
    end
    
    if joint.none?
      if must_match_grouping
        if gather_last_result
          last_result.timeline << <<-EOS
Since :must_match_at_least_one_word => true and none of the competition was in the same group as the needle, the search stopped.
EOS
        end
        if is_find_all
          return []
        else
          return nil
        end
      else
        joint = passed_word_requirement.dup
      end
    end
        
    if identities.any?
      possibly_identical = joint.select do |straw|
        identities.all? do |identity|
          answer = identity.identical? needle, straw
          answer.nil? or answer == true
        end
      end
      if gather_last_result
        last_result.timeline << <<-EOS
Since there were identities, the competition was reduced to records that might be identical to the needle (in other words, are not certainly different)
\tIdentities (first 10 of #{identities.length}): #{identities[0,9].map(&:inspect).join(', ')}
\tPassed (first 10 of #{possibly_identical.length}): #{possibly_identical[0,9].map(&:inspect).join(', ')}
\tFailed (first 10 of #{(joint-possibly_identical).length}): #{(joint-possibly_identical)[0,9].map(&:inspect).join(', ')}
EOS
      end
    else
      possibly_identical = joint.dup
    end
    
    similarities = possibly_identical.map { |straw| needle.similarity straw }.sort.reverse
        
    if gather_last_result
      last_result.timeline << <<-EOS
The competition was sorted in order of similarity to the needle.
\t#{similarities[0,9].map { |s| "#{s.record2.similarity(needle).inspect}" }.join("\n\t")}
EOS
    end
    
    if is_find_all_with_score
      memo = []
      similarities.each do |similarity|
        if similarity.satisfy?(needle, threshold)
          bs = similarity.best_score
          memo << [similarity.record2.original, bs.dices_coefficient_similar, bs.levenshtein_similar]
        end
      end
      return memo
    end

    if is_find_best
      memo = []
      best_bs = nil
      similarities.each do |similarity|
        if similarity.satisfy?(needle, threshold)
          bs = similarity.best_score
          best_bs ||= bs
          if bs >= best_bs
            memo << similarity.record2.original
          else
            break
          end
        end
      end
      return memo
    end

    if is_find_all
      memo = []
      similarities.each do |similarity|
        if similarity.satisfy?(needle, threshold)
          memo << similarity.record2.original
        end
      end
      return memo
    end
    
    best_similarity = similarities.first
    winner = nil

    if best_similarity and best_similarity.satisfy?(needle, threshold)
      winner = best_similarity.record2.original
      if gather_last_result
        last_result.winner = winner
        last_result.score = best_similarity.best_score.dices_coefficient_similar
        last_result.timeline << <<-EOS
A winner was determined because the Dice's Coefficient similarity (#{'%0.5f' % best_similarity.best_score.dices_coefficient_similar}) is greater than zero or because it shared a word with the needle.
EOS
      end
      if is_find_with_score
        bs = best_similarity.best_score
        return [winner, bs.dices_coefficient_similar, bs.levenshtein_similar]
      else
        return winner
      end
    elsif gather_last_result
      best_similarity_record = if best_similarity and best_similarity.record2
        best_similarity.record2.original
      end
      last_result.timeline << <<-EOS
No winner assigned because the score of the best similarity (#{best_similarity_record.inspect}) was zero and it didn't match any words with the needle (#{needle.inspect}).
EOS
    end

    nil # ugly
  end
  
  # Explain is like mysql's EXPLAIN command. You give it a needle and it tells you about how it was located (successfully or not) in the haystack.
  #
  #     d = FuzzyMatch.new ['737', '747', '757' ]
  #     d.explain 'boeing 737-100'
  def explain(needle, options = {})
    find needle, options.merge(:gather_last_result => true)
    last_result.explain
  end

  private

  REFRESH_OPTION_KEYS = [
    :with, :replacement, :new_record, :record, :source,
    :read, :reader, :read_with, :id_key, :primary_key, :record_id_key
  ].freeze
  private_constant :REFRESH_OPTION_KEYS

  def parse_refresh_arguments(arguments, block)
    options = {}

    if arguments.last.is_a?(::Hash) && refresh_options_hash?(arguments.last)
      options.merge!(arguments.pop)
    end

    if arguments.length == 1 && arguments.first.is_a?(::Hash)
      hash = arguments.first
      if refresh_options_hash?(hash)
        options.merge!(arguments.shift)
      else
        arguments.shift
        arguments = [id_from_hash(hash), id_only_hash?(hash) ? nil : hash]
      end
    end

    record_id = arguments.shift unless arguments.empty?
    replacement = arguments.shift unless arguments.empty?

    if record_id.is_a?(::Hash)
      hash = record_id
      record_id = id_from_hash(hash)
      replacement ||= hash unless id_only_hash?(hash)
    end

    record_id ||= options[:id] || options['id'] || options[:record_id] || options['record_id']
    replacement ||= options[:with] || options[:replacement] || options[:new_record] ||
                    options[:record] || options[:source]
    reader = options[:read] || options[:reader] || options[:read_with]
    reader ||= block

    [record_id, replacement, reader, options]
  end

  def refresh_options_hash?(hash)
    hash.keys.any? do |key|
      REFRESH_OPTION_KEYS.include?(key) ||
        (key.respond_to?(:to_sym) && REFRESH_OPTION_KEYS.include?(key.to_sym))
    end
  end

  def id_key_name?(key)
    %i[id record_id].include?(key) || %w[id record_id].include?(key)
  end

  def id_only_hash?(hash)
    hash.keys.all? { |key| id_key_name?(key) }
  end

  def id_from_hash(hash)
    hash[:id] || hash['id'] || hash[:record_id] || hash['record_id']
  end

  def record_slot_for_id(record_id, custom_id_reader)
    raise ::ArgumentError, 'A record ID is required to refresh a FuzzyMatch record' if record_id.nil?

    slot = haystack.index do |record|
      ids_match?(record_identifier(record.original, custom_id_reader), record_id)
    end

    raise ::ArgumentError, "No FuzzyMatch record found with ID #{record_id.inspect}" unless slot
    slot
  end

  def current_source_record(record_id, custom_id_reader)
    return nil unless source_haystack.is_a?(::Array)

    source_haystack.find do |original|
      ids_match?(record_identifier(original, custom_id_reader), record_id)
    end
  end

  def record_identifier(original, custom_id_reader)
    if custom_id_reader
      Record.read_value(original, custom_id_reader)
    else
      infer_record_id(original)
    end
  end

  def infer_record_id(original)
    case original
    when ::Hash
      original.key?(:id) ? original[:id] : original['id']
    else
      original.id if original.respond_to?(:id)
    end
  end

  def ids_match?(actual, expected)
    actual == expected || (!actual.nil? && actual.to_s == expected.to_s)
  rescue ::NoMethodError
    false
  end
end
