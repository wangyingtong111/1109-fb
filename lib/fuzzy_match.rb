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
  attr_reader :groupings
  attr_reader :identities
  attr_reader :stop_words
  attr_accessor :read
  attr_reader :fields
  attr_reader :default_options

  # haystack - a bunch of records that will compete to see who best matches the needle
  #
  # Rules (can only be specified at initialization or by using a setter)
  # * :<tt>identities</tt> - regexps
  # * :<tt>groupings</tt> - regexps
  # * :<tt>stop_words</tt> - regexps
  # * :<tt>read</tt> - how to interpret each record in the 'haystack', either a Proc or a symbol. If it returns an Array, each element is treated as a separate positional field and only matching positions are compared.
  # * :<tt>fields</tt> (also :projections) - an explicit Array of field readers (symbols/keys/indices/Procs) giving the structured projection. Equivalent to passing <tt>:read => [:a, :b]</tt>.
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
    @fields = o.delete(:fields) || o.delete(:projections)
    if @fields.nil? and @read.is_a?(::Array)
      @fields = @read
      @read = nil
    end
    @groupings = (o.delete(:groupings) || o.delete(:blockings) || []).map { |regexp| Rule::Grouping.make(regexp) }.flatten
    @identities = (o.delete(:identities) || []).map { |regexp| Rule::Identity.new(regexp) }
    @stop_words = o.delete(:stop_words) || []

    # options
    if deprecated = o.delete(:must_match_blocking)
      o[:must_match_grouping] = deprecated
    end
    @default_options = DEFAULT_OPTIONS.merge(o).freeze

    @haystack = haystack.map { |original| Record.new original, :stop_words => @stop_words, :read => @read, :fields => @fields }
  end
    
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
      Record.new needle, :read => read, :fields => fields, :stop_words => stop_words
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
    
    similarities = possibly_identical.each_with_index.map do |straw, index|
      similarity = needle.similarity(straw)
      similarity.tie_index = index
      similarity
    end.sort.reverse
        
    if gather_last_result
      last_result.timeline << <<-EOS
The competition was sorted in order of similarity to the needle.
\t#{similarities[0,9].map { |s| "#{s.record2.similarity(needle).inspect}" }.join("\n\t")}
EOS
    end
    
    if gather_last_result
      last_result.field_details = similarities.map { |similarity| build_detail_row(similarity) }
      last_result.details = last_result.field_details
      if best = similarities.first
        last_result.field_scores = best.best_score.respond_to?(:field_dices_coefficient_similar) ? best.best_score.field_dices_coefficient_similar : nil
      end
      if needle.structured?
        last_result.timeline << structured_detail_timeline(similarities)
      end
    end

    if is_find_all_with_score
      memo = []
      if structured_projection?(needle)
        # Structured projections return one positional score per field.  The
        # detail tuples are reported in candidate/projection order because the
        # order between non-winning candidates is not a ranking statement;
        # #find still selects via the sorted `similarities` above.
        possibly_identical.each do |straw|
          similarity = needle.similarity(straw)
          next unless similarity.satisfy?(needle, threshold)
          memo << score_tuple(similarity)
        end
      else
        similarities.each do |similarity|
          if similarity.satisfy?(needle, threshold)
            memo << score_tuple(similarity)
          end
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
        if best_similarity.structured?
          last_result.winner_field_details = best_similarity.field_details
          last_result.field_scores = best_similarity.best_score.field_dices_coefficient_similar
        end
        last_result.timeline << <<-EOS
A winner was determined because the Dice's Coefficient similarity (#{'%0.5f' % best_similarity.best_score.dices_coefficient_similar}) is greater than zero or because it shared a word with the needle.
EOS
      end
      if is_find_with_score
        return score_tuple(best_similarity, winner)
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

  # Per-field comparison details for the most recent run (requires
  # :gather_last_result => true).  Each entry describes one candidate and its
  # positional field statuses (:compared / :not_compared).
  def details
    last_result.details
  end

  def field_details
    last_result.field_details
  end

  private

  def structured_projection?(needle)
    needle.respond_to?(:structured?) && needle.structured?
  end

  # The score tuple returned by find_*_with_score.  Ordinary records keep the
  # historical [original, dice, levenshtein] shape.  Structured projections
  # return one positional dice score per field instead:
  # [original, field_0_score, field_1_score, ...] where a +nil+ field score
  # means that position was "not compared".
  def score_tuple(similarity, original = nil)
    original ||= similarity.record2.original
    bs = similarity.best_score
    if bs.structured?
      [original, *bs.field_dices_coefficient_similar]
    else
      [original, bs.dices_coefficient_similar, bs.levenshtein_similar]
    end
  end

  def build_detail_row(similarity)
    bs = similarity.best_score
    row = {
      :record => similarity.record2.original,
      :score => bs.dices_coefficient_similar,
      :dice => bs.dices_coefficient_similar,
      :levenshtein => bs.levenshtein_similar,
      :structured => bs.structured?
    }
    if bs.structured?
      row[:field_scores] = bs.field_dices_coefficient_similar
      row[:field_levenshtein] = bs.field_levenshtein_similar
      row[:fields] = similarity.field_details
    end
    row
  end

  def structured_detail_timeline(similarities)
    lines = similarities[0, 9].map do |similarity|
      id = similarity.record2.original.respond_to?(:[]) ? (similarity.record2.original[:id] || similarity.record2.original['id'] || similarity.record2.original) : similarity.record2.original
      fields = similarity.field_details.map do |detail|
        if detail[:compared]
          "field##{detail[:index]}=#{ '%0.3f' % detail[:dice] }"
        else
          "field##{detail[:index]}=#{Record::NOT_COMPARED_TEXT_ZH}/#{Record::NOT_COMPARED}"
        end
      end.join(', ')
      "\t#{id.inspect}: #{fields}"
    end.join("\n")
    "Positional field comparison (missing positions are marked not compared / 未比较):\n#{lines}\n"
  end
end
