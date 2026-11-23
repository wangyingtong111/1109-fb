# frozen_string_literal: true

# Author::    Lucas Carlson  (mailto:lucas@rufy.com)
# Copyright:: Copyright (c) 2005 Lucas Carlson
# License::   LGPL

require 'set'

module ClassifierReborn
  module TokenFilter
    # This filter removes stopwords in the language, from given tokens.
    #
    # A classifier owns its own Stopword instance, so each classifier's
    # language selection and custom stopword list are kept isolated from any
    # other classifier running in the same process. The Stopword module itself
    # is retained as a shared, backward-compatible facade for callers that use
    # the filter directly (e.g. through Hasher) without instantiating one.
    class StopwordFilter
      # Default locations searched for language stopword files
      DEFAULT_STOPWORDS_PATH = [File.expand_path(File.join(File.dirname(__FILE__), '../../../../data/stopwords'))].freeze

      # Lazily-loaded stopword sets shared by all instances that use the
      # bundled language files; custom per-instance lists never get written here
      STOPWORDS = Hash.new do |hash, language|
        hash[language] = load_language_stopwords(language)
      end

      # Paths searched when lazily loading language stopword files
      @stopwords_path = DEFAULT_STOPWORDS_PATH.dup

      class << self
        # Paths searched when lazily loading language stopword files
        attr_accessor :stopwords_path

        # Add custom path to a new stopword file created by user
        def add_custom_stopword_path(path)
          stopwords_path.unshift(path)
        end

        # Load the stopword set for a language from the configured search paths
        def load_language_stopwords(language)
          stopwords_path.each do |path|
            file = File.join(path, language.to_s)
            if File.exist?(file)
              return Set.new File.read(file).force_encoding('utf-8').split
            end
          end
          Set.new
        end
      end

      attr_accessor :language

      # A custom stopword list overrides the bundled language file for this
      # instance only; +nil+ means fall back to the language stopword files.
      attr_reader :custom_stopwords

      def initialize(language: 'en')
        @language = language
        @custom_stopwords = nil
      end

      def call(tokens)
        tokens.reject do |token|
          token.maybe_stopword? &&
            (token.length <= 2 || stopwords.include?(token))
        end
      end

      # Overwrites the default stopwords for this instance's language with the
      # supplied list of words. Assign an empty list to disable stopwords.
      def custom_stopwords=(stopwords)
        @custom_stopwords = Set.new(stopwords)
      end

      # The active stopword set used while filtering tokens
      def stopwords
        @custom_stopwords || self.class::STOPWORDS[@language]
      end
    end

    # Shared facade preserving the historical module-level API. State lives on
    # a single shared instance; classifiers should build their own Stopword
    # instance instead of relying on this global one.
    module Stopword
      STOPWORDS_PATH = StopwordFilter.stopwords_path

      # Lazily-loaded hash of the bundled language stopword data, shared with
      # every StopwordFilter instance that uses the bundled language files
      STOPWORDS = StopwordFilter::STOPWORDS

      module_function

      def call(tokens)
        default_instance.call(tokens)
      end

      def add_custom_stopword_path(path)
        StopwordFilter.add_custom_stopword_path(path)
      end

      # Changes the language of the shared stopword filter
      def language=(language)
        default_instance.language = language
      end

      def language
        default_instance.language
      end

      def default_instance
        @default_instance ||= StopwordFilter.new
      end
    end
  end
end
