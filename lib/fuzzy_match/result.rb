# encoding: utf-8
require 'erb'
require 'pp'

class FuzzyMatch
  class Result #:nodoc: all
    EXPLANATION = <<-ERB
#####################################################
# SUMMARY
#####################################################

Needle: <%= needle.inspect %>
Match:  <%= winner.inspect %>

#####################################################
# OPTIONS
#####################################################

<%= PP.pp(options, '') %>

<% timeline.each_with_index do |event, index| %>
(<%= index+1 %>) <%= event %>
<% end %>
ERB

    attr_accessor :needle
    attr_accessor :read
    attr_accessor :haystack
    attr_accessor :options
    attr_accessor :groupings
    attr_accessor :identities
    attr_accessor :stop_words
    attr_accessor :winner
    attr_accessor :score
    # Structured multi-field details.  +field_details+ / +details+ hold one
    # row per candidate (in ranked order); +winner_field_details+ holds the
    # positional statuses of the winner; +field_scores+ is the winner's
    # positional score array (+nil+ entries mean "not compared").
    attr_accessor :field_details
    attr_accessor :details
    attr_accessor :winner_field_details
    attr_accessor :field_scores
    attr_reader :timeline

    def initialize
      @timeline = []
    end

    # Convenience aliases for callers looking for structured score detail.
    def score_details
      @field_details
    end

    def per_field_scores
      @field_scores
    end

    def not_compared_fields
      return [] unless @winner_field_details
      @winner_field_details.select { |d| d && d[:compared] == false }
    end

    def explain
      $stdout.puts ::ERB.new(EXPLANATION, trim_mode: '<>').result(binding)
    end
  end
end
