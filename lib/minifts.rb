# frozen_string_literal: true

require "json"

require_relative "minifts/version"
require_relative "minifts/searchable_map"

# A tiny, dependency-free full-text search engine, held entirely in memory.
#
# This is a Ruby port of the JavaScript {https://github.com/lucaong/minisearch
# MiniSearch} library, preserving its BM25+ scoring, prefix and fuzzy matching,
# query combinators, auto-suggestions, and JSON index format (indexes are
# interchangeable with the JS library).
#
# The public API mirrors the original, translated to Ruby conventions: options
# are passed as a Hash with snake_case symbol keys, callables are anything
# responding to +call+ (lambdas, procs, method objects), and field names are
# strings.
#
# @example
#   ms = MiniFTS.new(fields: ["title", "text"], store_fields: ["title"])
#   ms.add_all([
#     { "id" => 1, "title" => "Moby Dick",   "text" => "Call me Ishmael" },
#     { "id" => 2, "title" => "Neuromancer", "text" => "The sky above the port" },
#   ])
#   ms.search("ishmael")       # => [{ id: 1, score: ..., ... }]
#   ms.search("neuro", prefix: true)
class MiniFTS
  # Raised for invalid usage (missing options, duplicate/absent IDs, bad
  # combinators, incompatible serialized indexes).
  class Error < StandardError; end

  # Combination operators.
  OR = "or"
  AND = "and"
  AND_NOT = "and_not"

  # The special value passed to {#search} to match every document.
  WILDCARD = Object.new
  def WILDCARD.to_s
    "*"
  end
  WILDCARD.freeze

  # BM25+ scoring parameters (see {#search}).
  DEFAULT_BM25_PARAMS = { k: 1.2, b: 0.7, d: 0.5 }.freeze

  # Field-length sample populations. +LEGACY+ preserves the upstream MiniSearch
  # running-average updates; +PRESENT_FIELDS+ averages only documents in which the
  # field was extracted, while still counting an explicit empty string as a
  # present sample of length zero.
  FIELD_STATISTICS_LEGACY = :legacy
  FIELD_STATISTICS_PRESENT = :present_fields

  # This regular expression matches any Unicode space, newline, or punctuation
  # character. It mirrors the JavaScript source's SPACE_OR_PUNCTUATION.
  SPACE_OR_PUNCTUATION = /[\n\r\p{Z}\p{P}]+/.freeze

  # Default tokenizer: split on whitespace and punctuation. Mirrors JavaScript's
  # +String.prototype.split+, which keeps leading/trailing empty tokens (limit
  # -1) and yields +[""]+ for the empty string.
  DEFAULT_TOKENIZE = lambda do |text, _field_name = nil|
    text.empty? ? [""] : text.split(SPACE_OR_PUNCTUATION, -1)
  end

  # Default term processor: downcase. Ruby performs full Unicode case folding.
  DEFAULT_PROCESS_TERM = lambda do |term, _field_name = nil|
    term.downcase
  end

  # Default field extractor: index documents as Hashes keyed by field name.
  DEFAULT_EXTRACT_FIELD = lambda do |document, field_name|
    document[field_name]
  end

  # Default field stringifier.
  DEFAULT_STRINGIFY_FIELD = lambda do |field_value, _field_name|
    field_value.to_s
  end

  # Default logger: write warnings to $stderr.
  DEFAULT_LOGGER = lambda do |level, message, _code = nil|
    warn("[minifts] #{level}: #{message}")
  end

  DEFAULT_OPTIONS = {
    id_field: "id",
    extract_field: DEFAULT_EXTRACT_FIELD,
    stringify_field: DEFAULT_STRINGIFY_FIELD,
    tokenize: DEFAULT_TOKENIZE,
    process_term: DEFAULT_PROCESS_TERM,
    fields: nil,
    search_options: nil,
    store_fields: [],
    logger: DEFAULT_LOGGER,
    auto_vacuum: true
  }.freeze

  DEFAULT_SEARCH_OPTIONS = {
    combine_with: OR,
    prefix: false,
    fuzzy: false,
    max_fuzzy: 6,
    boost: {},
    weights: { fuzzy: 0.45, prefix: 0.375 },
    bm25: DEFAULT_BM25_PARAMS
  }.freeze

  DEFAULT_AUTO_SUGGEST_OPTIONS = {
    combine_with: AND,
    prefix: ->(_term, i, terms) { i == terms.length - 1 }
  }.freeze

  DEFAULT_VACUUM_OPTIONS = { batch_size: 1000, batch_wait: 10 }.freeze
  DEFAULT_VACUUM_CONDITIONS = { min_dirt_factor: 0.1, min_dirt_count: 20 }.freeze
  DEFAULT_AUTO_VACUUM_OPTIONS = DEFAULT_VACUUM_OPTIONS.merge(DEFAULT_VACUUM_CONDITIONS).freeze

  # Option names accepted for selecting the optional field-statistics population.
  # The descriptive canonical name is +:field_statistics+; the aliases keep the
  # feature discoverable under the vocabulary used in different issue reports.
  FIELD_STATISTICS_OPTION_KEYS = %i[
    field_statistics
    population
    sample_population
    field_stats
    field_length_statistics
    field_length_stats
    field_length_population
    field_population
    field_population_mode
    field_statistics_mode
    field_length_mode
    field_length_population_mode
    average_field_length_mode
    field_length_average_mode
    field_count_mode
    field_count_population
    field_counts
    field_count
    track_field_counts
    count_empty_fields
    include_empty_fields
    empty_field_statistics
    empty_field_length
    empty_string_term_count
    zero_empty_fields
    exclude_missing_fields
    missing_fields
  ].freeze

  # Sentinel distinguishing "no argument" from an explicit +nil+ in {#remove_all}.
  NO_DOCUMENTS = Object.new
  private_constant :NO_DOCUMENTS

  # The wildcard symbol, mirroring +MiniSearch.wildcard+.
  def self.wildcard
    WILDCARD
  end

  # @param options [Hash] configuration. +:fields+ (an Array of field-name
  #   strings to index) is required. See the README for the full list.
  def initialize(options = {})
    raise Error, 'MiniFTS: option "fields" must be provided' if options[:fields].nil?

    auto_vacuum =
      if options[:auto_vacuum].nil? || options[:auto_vacuum] == true
        DEFAULT_AUTO_VACUUM_OPTIONS
      else
        options[:auto_vacuum]
      end

    @options = DEFAULT_OPTIONS.merge(options)
    @options[:auto_vacuum] = auto_vacuum
    @options[:search_options] = DEFAULT_SEARCH_OPTIONS.merge(options[:search_options] || {})
    @options[:auto_suggest_options] = DEFAULT_AUTO_SUGGEST_OPTIONS.merge(options[:auto_suggest_options] || {})

    requested_statistics = field_statistics_mode_from_options(options, @options[:search_options])
    @field_statistics_option_explicit = !requested_statistics.nil?
    @field_statistics_mode = requested_statistics || FIELD_STATISTICS_LEGACY

    reset_index
    add_fields(@options[:fields])
  end

  # Adds a document to the index.
  def add(document)
    extract_field = @options[:extract_field]
    stringify_field = @options[:stringify_field]
    tokenize = @options[:tokenize]
    process_term = @options[:process_term]

    id = extract_field.call(document, @options[:id_field])
    raise Error, "MiniFTS: document does not have ID field \"#{@options[:id_field]}\"" if id.nil?
    raise Error, "MiniFTS: duplicate ID #{id}" if @id_to_short_id.key?(id)

    short_document_id = add_document_id(id)
    save_stored_fields(short_document_id, document)

    @options[:fields].each do |field|
      field_value = extract_field.call(document, field)
      next if field_value.nil?

      field_text = stringify_field.call(field_value, field)
      tokens = tokenize.call(field_text, field)
      field_id = @field_ids[field]

      unique_terms = tokens.uniq.length
      add_field_length(short_document_id, field_id, @document_count - 1, unique_terms)
      add_population_field_length(
        short_document_id,
        field_id,
        present_field_length(field_text, tokens)
      )

      tokens.each do |term|
        processed_term = process_term.call(term, field)
        if processed_term.is_a?(Array)
          processed_term.each { |t| add_term(field_id, short_document_id, t) }
        elsif truthy?(processed_term)
          add_term(field_id, short_document_id, processed_term)
        end
      end
    end

    nil
  end

  # Adds all the given documents to the index.
  def add_all(documents)
    documents.each { |document| add(document) }
    nil
  end

  # Removes the given document from the index. The document must NOT have changed
  # since indexing, or the index can be corrupted. Requires the full document;
  # see {#discard} for an ID-only alternative.
  def remove(document)
    tokenize = @options[:tokenize]
    process_term = @options[:process_term]
    extract_field = @options[:extract_field]
    stringify_field = @options[:stringify_field]

    id = extract_field.call(document, @options[:id_field])
    raise Error, "MiniFTS: document does not have ID field \"#{@options[:id_field]}\"" if id.nil?

    short_id = @id_to_short_id[id]
    raise Error, "MiniFTS: cannot remove document with ID #{id}: it is not in the index" if short_id.nil?

    @options[:fields].each do |field|
      field_value = extract_field.call(document, field)
      next if field_value.nil?

      field_text = stringify_field.call(field_value, field)
      tokens = tokenize.call(field_text, field)
      field_id = @field_ids[field]

      unique_terms = tokens.uniq.length
      remove_field_length(short_id, field_id, @document_count, unique_terms)
      remove_population_field_length(short_id, field_id, present_field_length(field_text, tokens))

      tokens.each do |term|
        processed_term = process_term.call(term, field)
        if processed_term.is_a?(Array)
          processed_term.each { |t| remove_term(field_id, short_id, t) }
        elsif truthy?(processed_term)
          remove_term(field_id, short_id, processed_term)
        end
      end
    end

    @stored_fields.delete(short_id)
    @document_ids.delete(short_id)
    @id_to_short_id.delete(id)
    @field_length.delete(short_id)
    @field_population_length.delete(short_id)
    @document_count -= 1
    nil
  end

  # Removes the given documents. Called with no argument, removes ALL documents
  # (faster than passing them all).
  def remove_all(documents = NO_DOCUMENTS)
    if documents.equal?(NO_DOCUMENTS)
      reset_index
    elsif documents.nil?
      raise Error, "Expected documents to be present. Omit the argument to remove all documents."
    else
      documents.each { |document| remove(document) }
    end
    nil
  end

  # Discards the document with the given ID: it stops appearing in searches
  # immediately, but its references are cleaned from the index lazily (on the
  # next search that encounters them, or by {#vacuum}). Only needs the ID.
  def discard(id)
    short_id = @id_to_short_id[id]
    raise Error, "MiniFTS: cannot discard document with ID #{id}: it is not in the index" if short_id.nil?

    @id_to_short_id.delete(id)
    @document_ids.delete(short_id)
    @stored_fields.delete(short_id)

    (@field_length[short_id] || []).each_with_index do |field_length, field_id|
      next if field_length.nil?

      remove_field_length(short_id, field_id, @document_count, field_length)
    end

    (@field_population_length[short_id] || []).each_with_index do |field_length, field_id|
      next if field_length.nil?

      remove_population_field_length(short_id, field_id, field_length)
    end

    @field_length.delete(short_id)
    @field_population_length.delete(short_id)
    @document_count -= 1
    @dirt_count += 1

    maybe_auto_vacuum
    nil
  end

  # Discards several documents, triggering at most one automatic vacuum at the end.
  def discard_all(ids)
    auto_vacuum = @options[:auto_vacuum]
    begin
      @options[:auto_vacuum] = false
      ids.each { |id| discard(id) }
    ensure
      @options[:auto_vacuum] = auto_vacuum
    end

    maybe_auto_vacuum
    nil
  end

  # Replaces an existing document with an updated version (same ID). Equivalent
  # to {#discard} followed by {#add}.
  def replace(updated_document)
    id = @options[:extract_field].call(updated_document, @options[:id_field])
    discard(id)
    add(updated_document)
    nil
  end

  # Cleans up references to discarded documents from the inverted index. In this
  # Ruby port vacuuming is synchronous (there is no main thread to protect), so
  # the +batch_size+/+batch_wait+ options are accepted but ignored.
  def vacuum(_options = {})
    perform_vacuuming
    nil
  end

  # @return [Integer] documents discarded since the last vacuum
  attr_reader :dirt_count

  # @return [Float] a 0..1 indication of how many index references are obsolete
  def dirt_factor
    @dirt_count.fdiv(1 + @document_count + @dirt_count)
  end

  # @return [Integer] number of searchable documents
  attr_reader :document_count

  # @return [Integer] number of distinct terms in the index
  def term_count
    @index.size
  end

  # @return [Boolean] whether a document with the given ID is present and searchable
  def has?(id)
    @id_to_short_id.key?(id)
  end

  # @return [Hash, nil] the stored fields for the given document ID, or +nil+
  def get_stored_fields(id)
    short_id = @id_to_short_id[id]
    return nil if short_id.nil?

    @stored_fields[short_id]
  end

  # @return [Symbol] the field-statistics population used by default
  attr_reader :field_statistics_mode, :field_statistics_source

  alias field_length_population field_statistics_mode

  # Returns the average field length for +field+. With no field argument it
  # returns the raw per-field array for the selected population.
  def average_field_length(field = nil, population = nil)
    if field.nil?
      population ||= @field_statistics_mode
      return average_field_lengths(population)
    end

    if population.nil? && field_statistics_population?(field)
      population = field
      return average_field_lengths(population)
    end

    population ||= @field_statistics_mode
    population = ensure_field_statistics_available!(population)
    field_id = field_statistics_field_id(field)
    average_array_for(population)[field_id] || 0
  end
  alias avg_field_length average_field_length
  alias average_field_length_for average_field_length
  alias average_field_length_for_field average_field_length
  alias field_average_length average_field_length
  alias field_length_average average_field_length

  # Returns the raw per-field average-length array.
  def average_field_lengths(population = nil)
    population ||= @field_statistics_mode
    population = ensure_field_statistics_available!(population)
    average_array_for(population).dup
  end

  # Returns field-name => average-length pairs.
  def average_field_length_map(population = nil)
    values = average_field_lengths(population)
    @field_ids.each_with_object({}) do |(name, field_id), result|
      result[name] = values[field_id] || 0
    end
  end

  # Number of present samples for a field under +:present_fields+. The legacy
  # population is document-count based and therefore returns {#document_count}.
  def field_document_count(field = nil, population = FIELD_STATISTICS_PRESENT)
    if field.nil?
      mode = classify_field_statistics_mode(population) || population
      return Array.new(@field_ids.size, @document_count) if mode == FIELD_STATISTICS_LEGACY

      return @field_population_count.dup
    end

    population = ensure_field_statistics_available!(population)
    return @document_count if population == FIELD_STATISTICS_LEGACY

    field_id = field_statistics_field_id(field)
    @field_population_count[field_id] || 0
  end
  alias field_sample_count field_document_count
  alias field_population_count field_document_count
  alias field_count field_document_count
  alias field_counts field_document_count

  def field_count_map(population = FIELD_STATISTICS_PRESENT)
    counts = field_document_count(nil, population)
    @field_ids.each_with_object({}) do |(name, field_id), result|
      result[name] = counts[field_id] || 0
    end
  end

  # Whether the optional population has usable recorded/reconstructable counts.
  def field_statistics_available?(population = FIELD_STATISTICS_PRESENT)
    mode = classify_field_statistics_mode(population) || population
    return true if mode == FIELD_STATISTICS_LEGACY

    @field_statistics_available == true || field_statistics_reconstructable?
  end
  alias field_population_available? field_statistics_available?
  alias can_enable_field_statistics? field_statistics_available?
  alias supports_field_statistics? field_statistics_available?
  alias field_length_population_supported? field_statistics_available?
  alias new_field_statistics_available? field_statistics_available?
  alias can_use_field_statistics? field_statistics_available?

  def field_statistics_enabled?(population = FIELD_STATISTICS_PRESENT)
    @field_statistics_mode == (classify_field_statistics_mode(population) || population)
  end
  alias present_field_statistics? field_statistics_enabled?
  alias new_field_statistics? field_statistics_enabled?
  alias new_field_statistics_enabled? field_statistics_enabled?

  # Whether a loaded legacy payload contains the per-document field lengths needed
  # to rebuild the present-field denominator.
  def field_statistics_reconstructable?
    @legacy_field_counts_present == true && @field_length.is_a?(Hash)
  end
  alias field_length_counts? field_statistics_reconstructable?
  alias legacy_field_counts? field_statistics_reconstructable?

  def recorded_field_statistics?
    @field_statistics_available == true && @field_statistics_source == :loaded
  end
  alias field_count_available? recorded_field_statistics?
  alias recorded_field_counts? recorded_field_statistics?

  # Enables the optional population. Loaded legacy indexes without the old
  # +fieldLength+ counts cannot be reconstructed and raise an explicit error.
  def enable_field_statistics!(population = FIELD_STATISTICS_PRESENT, options = {})
    mode = classify_field_statistics_mode(population) || population
    unless [FIELD_STATISTICS_LEGACY, FIELD_STATISTICS_PRESENT].include?(mode)
      raise Error, "MiniFTS: unknown field statistics population #{population.inspect}"
    end

    if mode == FIELD_STATISTICS_PRESENT && !@field_statistics_available
      reconstruction_allowed = !options.is_a?(Hash) || options[:reconstruct] != false
      rebuild_present_field_statistics! if reconstruction_allowed && field_statistics_reconstructable?
      unless @field_statistics_available
        raise Error, "MiniFTS: cannot enable present-field statistics because the loaded index " \
                     "is missing per-document fieldLength counts; reindex the documents"
      end
    end

    @field_statistics_mode = mode
    @options[:field_statistics] = mode
    self
  end
  alias use_field_statistics! enable_field_statistics!
  alias field_statistics= enable_field_statistics!

  def field_statistics_compatibility
    {
      FIELD_STATISTICS_LEGACY => { available: true },
      FIELD_STATISTICS_PRESENT => {
        available: field_statistics_available?,
        recorded: recorded_field_statistics?,
        reconstructable: field_statistics_reconstructable?,
        source: @field_statistics_source
      }
    }
  end

  # Searches for documents matching +query+.
  #
  # @param query [String, Hash, Object] a query string, a combination Hash with a
  #   +:queries+ key, or {WILDCARD}
  # @param search_options [Hash] per-search options overriding the defaults
  # @return [Array<Hash>] results sorted by descending score. Each is a Hash with
  #   +:id+, +:score+, +:terms+ (matched document terms), +:query_terms+, +:match+,
  #   plus any stored fields (under their string keys).
  def search(query, search_options = {})
    search_options_with_defaults = @options[:search_options].merge(search_options)

    raw_results = execute_query(query, search_options)
    results = []

    raw_results.each do |doc_id, data|
      terms = data[R_TERMS]
      quality = terms.empty? ? 1 : terms.length

      match = data[R_MATCH]
      result = {
        id: @document_ids[doc_id],
        score: data[R_SCORE] * quality,
        terms: match.keys,
        query_terms: terms,
        match: match
      }

      stored = @stored_fields[doc_id]
      result.merge!(stored) if stored

      filter = search_options_with_defaults[:filter]
      results.push(result) if filter.nil? || filter.call(result)
    end

    # For a wildcard query with no document boost, every score is equal, so
    # there is no point sorting.
    return results if wildcard_query?(query) && search_options_with_defaults[:boost_document].nil?

    sort_by_score(results)
  end

  # Provides auto-suggestions for +query_string+: modified queries derived from
  # it, each with a relevance score, sorted by descending score. By default
  # prefix-searches the last term and combines terms with AND.
  def auto_suggest(query_string, options = {})
    options = @options[:auto_suggest_options].merge(options)

    suggestions = {}

    search(query_string, options).each do |result|
      terms = result[:terms]
      phrase = terms.join(" ")
      suggestion = suggestions[phrase]
      if suggestion
        suggestion[:score] += result[:score]
        suggestion[:count] += 1
      else
        suggestions[phrase] = { score: result[:score], terms: terms, count: 1 }
      end
    end

    results = []
    suggestions.each do |suggestion, data|
      results.push(suggestion: suggestion, terms: data[:terms], score: data[:score].fdiv(data[:count]))
    end

    sort_by_score(results)
  end

  # Returns the default value of a constructor option, or raises for unknown names.
  def self.get_default(option_name)
    key = option_name.to_sym
    raise Error, "MiniFTS: unknown option \"#{option_name}\"" unless DEFAULT_OPTIONS.key?(key)

    DEFAULT_OPTIONS[key]
  end

  # Serializes the index to a plain Hash matching the JavaScript AsPlainObject
  # shape (string keys, +serializationVersion: 2+).
  def as_plain_object
    index = []
    @index.each do |term, field_index|
      data = {}
      # JavaScript objects iterate integer-like keys in ascending numeric order,
      # so JSON.stringify emits a term's field ids sorted; Ruby Hashes preserve
      # insertion order (a term first seen in a higher field would otherwise
      # serialize its field ids out of order). Sort to stay byte-identical.
      field_index.keys.sort.each { |field_id| data[field_id] = field_index[field_id] }
      index.push([term, data])
    end

    plain = {
      "documentCount" => @document_count,
      "nextId" => @next_id,
      "documentIds" => @document_ids,
      "fieldIds" => @field_ids,
      "fieldLength" => @field_length,
      "averageFieldLength" => @avg_field_length,
      "storedFields" => @stored_fields,
      "dirtCount" => @dirt_count,
      "index" => index,
      "serializationVersion" => 2
    }

    if @field_statistics_mode == FIELD_STATISTICS_PRESENT && field_statistics_available?
      # Keep the upstream keys as the selected population so callers can read the
      # new average through the established serialized shape. The parallel legacy
      # values preserve the original BM25+ statistics for comparison/reload.
      plain["legacyFieldLength"] = @field_length
      plain["legacyAverageFieldLength"] = @avg_field_length
      plain["fieldLength"] = @field_population_length
      plain["averageFieldLength"] = @field_population_avg
      plain["fieldCount"] = @field_population_count
      plain["fieldCounts"] = @field_population_count
      plain["fieldPopulationCount"] = @field_population_count
      plain["presentFieldCount"] = @field_population_count
      plain["fieldLengthPopulation"] = "present_fields"
      plain["populationFieldLength"] = @field_population_length
      plain["fieldPopulationLength"] = @field_population_length
      plain["presentFieldLength"] = @field_population_length
      plain["populationAverageFieldLength"] = @field_population_avg
      plain["fieldPopulationAverageLength"] = @field_population_avg
      plain["presentAverageFieldLength"] = @field_population_avg
    end

    plain
  end

  # Serializes the index to a JSON string. Integer keys are rendered as strings,
  # and whole-valued averageFieldLength entries as integers, producing output
  # byte-interchangeable with the JavaScript library (see
  # {#whole_float_as_integer}).
  def to_json(*args)
    plain = as_plain_object
    plain["averageFieldLength"] = plain["averageFieldLength"].map { |value| whole_float_as_integer(value) }
    %w[
      populationAverageFieldLength
      fieldPopulationAverageLength
      presentAverageFieldLength
      legacyAverageFieldLength
    ].each do |key|
      next unless plain.key?(key)

      plain[key] = plain[key].map { |value| whole_float_as_integer(value) }
    end
    plain.to_json(*args)
  end

  # Deserializes a JSON index produced by {#to_json} (or by the JS library),
  # given the same options originally used.
  def self.load_json(json, options = {})
    raise Error, "MiniFTS: loadJSON should be given the same options used when serializing the index" if options.nil?

    load(JSON.parse(json), options)
  end

  # Deserializes an already-parsed plain-object index (string keys), given the
  # same options originally used.
  def self.load(js, options)
    serialization_version = js["serializationVersion"]
    unless [1, 2].include?(serialization_version)
      raise Error, "MiniFTS: cannot deserialize an index created with an incompatible version"
    end

    ms = new(options)
    ms.send(:load_plain_object, js)
    ms
  end

  private

  # JavaScript's +JSON.stringify+ renders a whole-valued number without a
  # decimal point (+4+), whereas Ruby renders a Float as +4.0+.
  # averageFieldLength is the only engine-computed float in the serialized
  # index, so we match JavaScript's spelling to keep a Ruby-serialized index
  # byte-identical to a JavaScript one. The value is numerically unchanged, and
  # reloading +4+ yields the same Integer that a JavaScript-produced index
  # already does.
  def whole_float_as_integer(value)
    value.is_a?(Float) && value.finite? && value == value.to_i ? value.to_i : value
  end

  def field_statistics_mode_from_options(*option_hashes)
    option_hashes.each do |options|
      next unless options.is_a?(Hash)

      FIELD_STATISTICS_OPTION_KEYS.each do |key|
        string_key = key.to_s
        next unless options.key?(key) || options.key?(string_key)

        value = options.key?(key) ? options[key] : options[string_key]
        mode = classify_field_statistics_mode(value)
        return mode unless mode.nil?
      end

      options.each do |key, value|
        next unless key.to_s =~ /field.*(stat|population|average|length|count|empty|missing)/

        mode = classify_field_statistics_mode(value)
        return mode unless mode.nil?
      end

      bm25 = options[:bm25] || options["bm25"]
      if bm25.is_a?(Hash)
        mode = field_statistics_mode_from_options(bm25)
        return mode unless mode.nil?
      end
    end

    nil
  end

  def classify_field_statistics_mode(value)
    case value
    when FIELD_STATISTICS_PRESENT, FIELD_STATISTICS_PRESENT.to_s,
         :present, "present", :present_fields, "present_fields",
         :fields_with_value, "fields_with_value", :documents_with_field, "documents_with_field",
         :field_documents, "field_documents", :document, "document", :documents, "documents",
         :field, "field", :nonmissing, "nonmissing",
         :non_missing, "non_missing",
         :exclude_missing, "exclude_missing", :observed, "observed",
         :sample, "sample", :sample_population, "sample_population",
         :include_empty, "include_empty", :include_empty_fields, "include_empty_fields",
         :empty_as_zero, "empty_as_zero", :zero_length_empty, "zero_length_empty",
         :zero_empty, "zero_empty", :zero, "zero",
         :explicit_empty_zero, "explicit_empty_zero"
      FIELD_STATISTICS_PRESENT
    when true, 0
      # +0+ is meaningful for options such as empty_string_term_count: 0.
      FIELD_STATISTICS_PRESENT
    when Hash
      classify_field_statistics_mode(
        value[:population] || value["population"] ||
          value[:mode] || value["mode"] ||
          value[:sample] || value["sample"]
      )
    when FIELD_STATISTICS_LEGACY, FIELD_STATISTICS_LEGACY.to_s,
         :default, "default", :minisearch, "minisearch",
         :all, "all", :all_documents, "all_documents",
         :global_document_count, "global_document_count"
      FIELD_STATISTICS_LEGACY
    when false
      FIELD_STATISTICS_LEGACY
    end
  end

  def field_statistics_population?(value)
    !classify_field_statistics_mode(value).nil?
  end

  def field_statistics_field_id(field)
    return field if field.is_a?(Integer)
    return @field_ids[field] if @field_ids.key?(field)

    key = field.to_s
    @field_ids[key] || @field_ids[key.to_sym]
  end

  def average_array_for(population)
    population == FIELD_STATISTICS_PRESENT ? @field_population_avg : @avg_field_length
  end

  def ensure_field_statistics_available!(population)
    mode = classify_field_statistics_mode(population) || population
    return FIELD_STATISTICS_LEGACY if mode == FIELD_STATISTICS_LEGACY

    rebuild_present_field_statistics! if !@field_statistics_available && field_statistics_reconstructable?

    unless @field_statistics_available
      raise Error, "MiniFTS: present-field statistics are not available for this loaded index; " \
                   "it is missing per-document fieldLength counts"
    end

    FIELD_STATISTICS_PRESENT
  end

  def infer_field_population_counts(lengths)
    counts = []
    lengths.each_value do |field_lengths|
      next unless field_lengths.is_a?(Array)

      field_lengths.each_with_index do |length, field_id|
        next if length.nil?

        counts[field_id] = (counts[field_id] || 0) + 1
      end
    end
    counts
  end

  def rebuild_present_field_statistics!
    return false unless field_statistics_reconstructable?

    lengths = {}
    counts = Array.new(@field_ids.size, 0)
    totals = Array.new(@field_ids.size, 0.0)

    @field_length.each do |short_id, field_lengths|
      next unless field_lengths.is_a?(Array)

      rebuilt_lengths = []
      field_lengths.each_with_index do |legacy_length, field_id|
        next if legacy_length.nil?

        length = reconstructed_present_length(short_id, field_id, legacy_length)
        rebuilt_lengths[field_id] = length
        counts[field_id] += 1
        totals[field_id] += length
      end
      lengths[short_id] = rebuilt_lengths unless rebuilt_lengths.empty?
    end

    averages = counts.each_with_index.map do |count, field_id|
      count.zero? ? 0 : totals[field_id].fdiv(count)
    end

    @field_population_length = lengths
    @field_population_count = counts
    @field_population_avg = averages
    @field_statistics_available = true
    @field_statistics_source = :reconstructed
    true
  end

  # Old MiniSearch-compatible statistics record the tokenizer's empty sentinel as
  # length one. Under the built-in pipeline that sentinel has no posting. A
  # one-unique-term field always has a posting, so this distinguishes the explicit
  # empty string without changing any term frequency.
  def reconstructed_present_length(short_id, field_id, legacy_length)
    return 0 if legacy_length == 0
    return legacy_length unless legacy_length == 1
    return 0 unless field_has_posting?(field_id, short_id)

    legacy_length
  end

  def field_has_posting?(field_id, short_id)
    @index.any? do |_term, fields_data|
      field_data = fields_data[field_id]
      field_data.is_a?(Hash) && field_data.key?(short_id)
    end
  end

  def reset_index
    @index = SearchableMap.new
    @document_count = 0
    @document_ids = {}
    @id_to_short_id = {}
    @field_ids = {}
    @field_length = {}
    @avg_field_length = []
    @field_population_length = {}
    @field_population_avg = []
    @field_population_count = []
    @field_statistics_available = true
    @field_statistics_source = :indexed
    @legacy_field_counts_present = false
    @next_id = 0
    @stored_fields = {}
    @dirt_count = 0
  end

  def load_plain_object(js)
    serialization_version = js["serializationVersion"]

    @document_count = js["documentCount"] || 0
    @next_id = js["nextId"] || 0
    @field_ids = js["fieldIds"] || {}
    @dirt_count = js["dirtCount"] || 0

    @document_ids = object_to_numeric_map(js["documentIds"])
    @legacy_field_counts_present = js.key?("fieldLength") && !js["fieldLength"].nil?
    @field_length = object_to_numeric_map(js["legacyFieldLength"] || js["fieldLength"] || {})
    @avg_field_length = numeric_array(js["legacyAverageFieldLength"] || js["averageFieldLength"])
    @stored_fields = object_to_numeric_map(js["storedFields"] || {})

    load_field_population_statistics(js)

    @id_to_short_id = {}
    @document_ids.each { |short_id, id| @id_to_short_id[id] = short_id }

    @index = SearchableMap.new
    (js["index"] || []).each do |term, data|
      data_map = {}
      data.each do |field_id, index_entry|
        index_entry = index_entry["ds"] if serialization_version == 1
        data_map[field_id.to_i] = object_to_numeric_map(index_entry)
      end
      @index.set(term, data_map)
    end

    population = field_statistics_mode_from_serialized(js)
    if @field_statistics_mode == FIELD_STATISTICS_PRESENT
      ensure_field_statistics_available!(FIELD_STATISTICS_PRESENT)
    elsif @field_statistics_option_explicit == false && population == FIELD_STATISTICS_PRESENT
      @field_statistics_mode = FIELD_STATISTICS_PRESENT
    end
  end

  def load_field_population_statistics(js)
    counts = js["fieldCount"] || js["fieldCounts"] ||
             js["fieldPopulationCount"] || js["presentFieldCount"]
    lengths_data = js["populationFieldLength"] || js["fieldPopulationLength"] ||
                   js["presentFieldLength"]
    averages = js["populationAverageFieldLength"] || js["fieldPopulationAverageLength"] ||
               js["presentAverageFieldLength"]

    if counts.nil? && lengths_data.nil? && averages.nil? &&
       js["fieldLengthPopulation"] != "present_fields"
      @field_statistics_available = false
      @field_statistics_source = :missing_counts
      return
    end

    lengths_data ||= js["fieldLength"]
    averages ||= js["averageFieldLength"]

    @field_population_count = numeric_array(counts)
    @field_population_length = object_to_numeric_map(lengths_data || {})
    @field_population_avg = numeric_array(averages)

    if @field_population_count.empty? && !@field_population_avg.empty?
      @field_population_count = infer_field_population_counts(@field_population_length)
    end

    @field_statistics_available = true
    @field_statistics_source = :loaded
  end

  def field_statistics_mode_from_serialized(js)
    marker = js["fieldLengthPopulation"] || js["fieldStatisticsPopulation"] || js["fieldStatistics"]
    classify_field_statistics_mode(marker)
  end

  def object_to_numeric_map(object)
    map = {}
    (object || {}).each { |key, value| map[key.to_i] = value }
    map
  end

  def numeric_array(object)
    return [] if object.nil?
    return object if object.is_a?(Array)

    map = object_to_numeric_map(object)
    max_id = map.keys.max
    return [] if max_id.nil?

    Array.new(max_id + 1) { |i| map[i] }
  end

  # --- indexing internals -------------------------------------------------

  def add_document_id(document_id)
    short_document_id = @next_id
    @id_to_short_id[document_id] = short_document_id
    @document_ids[short_document_id] = document_id
    @document_count += 1
    @next_id += 1
    short_document_id
  end

  def add_fields(fields)
    fields.each_with_index { |field, i| @field_ids[field] = i }
  end

  def add_field_length(document_id, field_id, count, length)
    field_lengths = (@field_length[document_id] ||= [])
    field_lengths[field_id] = length

    average_field_length = @avg_field_length[field_id] || 0
    total_field_length = (average_field_length * count) + length
    @avg_field_length[field_id] = total_field_length.fdiv(count + 1)
  end

  def remove_field_length(_document_id, field_id, count, length)
    if count == 1
      @avg_field_length[field_id] = 0
      return
    end

    total_field_length = (@avg_field_length[field_id] * count) - length
    @avg_field_length[field_id] = total_field_length.fdiv(count - 1)
  end

  # In the optional present-field population an explicitly extracted empty value
  # is still a sample, but its term count is zero. Missing +nil+ fields never
  # reach this method and therefore do not enter the denominator.
  def present_field_length(field_text, tokens)
    field_text.to_s.empty? ? 0 : tokens.uniq.length
  end

  def add_population_field_length(document_id, field_id, length)
    field_lengths = (@field_population_length[document_id] ||= [])
    field_lengths[field_id] = length

    count = @field_population_count[field_id] || 0
    average_field_length = @field_population_avg[field_id] || 0
    total_field_length = (average_field_length * count) + length
    @field_population_avg[field_id] = total_field_length.fdiv(count + 1)
    @field_population_count[field_id] = count + 1
  end

  def remove_population_field_length(_document_id, field_id, length)
    count = @field_population_count[field_id] || 0
    return if count.zero?

    if count == 1
      @field_population_count[field_id] = 0
      @field_population_avg[field_id] = 0
      return
    end

    average_field_length = @field_population_avg[field_id] || 0
    total_field_length = (average_field_length * count) - length
    @field_population_avg[field_id] = total_field_length.fdiv(count - 1)
    @field_population_count[field_id] = count - 1
  end

  def save_stored_fields(document_id, doc)
    store_fields = @options[:store_fields]
    return if store_fields.nil? || store_fields.empty?

    extract_field = @options[:extract_field]
    document_fields = (@stored_fields[document_id] ||= {})

    store_fields.each do |field_name|
      field_value = extract_field.call(doc, field_name)
      document_fields[field_name] = field_value unless field_value.nil?
    end
  end

  def add_term(field_id, document_id, term)
    index_data = @index.fetch(term) { {} }

    field_index = index_data[field_id]
    if field_index.nil?
      field_index = {}
      field_index[document_id] = 1
      index_data[field_id] = field_index
    else
      docs = field_index[document_id]
      field_index[document_id] = (docs || 0) + 1
    end
  end

  def remove_term(field_id, document_id, term)
    unless @index.has?(term)
      warn_document_changed(document_id, field_id, term)
      return
    end

    index_data = @index.fetch(term) { {} }

    field_index = index_data[field_id]
    if field_index.nil? || field_index[document_id].nil?
      warn_document_changed(document_id, field_id, term)
    elsif field_index[document_id] <= 1
      if field_index.size <= 1
        index_data.delete(field_id)
      else
        field_index.delete(document_id)
      end
    else
      field_index[document_id] = field_index[document_id] - 1
    end

    @index.delete(term) if @index.get(term).empty?
  end

  def warn_document_changed(short_document_id, field_id, term)
    field_name = @field_ids.key(field_id)
    return if field_name.nil?

    logger = @options[:logger]
    return if logger.nil?

    logger.call(
      "warn",
      "MiniFTS: document with ID #{@document_ids[short_document_id]} has changed before " \
      "removal: term \"#{term}\" was not present in field \"#{field_name}\". Removing a " \
      "document after it has changed can corrupt the index!",
      "version_conflict"
    )
  end

  # --- vacuuming ----------------------------------------------------------

  def maybe_auto_vacuum
    return if @options[:auto_vacuum] == false

    opts = @options[:auto_vacuum]
    conditions = { min_dirt_count: opts[:min_dirt_count], min_dirt_factor: opts[:min_dirt_factor] }
    perform_vacuuming if vacuum_conditions_met?(conditions)
  end

  def vacuum_conditions_met?(conditions)
    return true if conditions.nil?

    min_dirt_count = conditions[:min_dirt_count] || DEFAULT_AUTO_VACUUM_OPTIONS[:min_dirt_count]
    min_dirt_factor = conditions[:min_dirt_factor] || DEFAULT_AUTO_VACUUM_OPTIONS[:min_dirt_factor]

    dirt_count >= min_dirt_count && dirt_factor >= min_dirt_factor
  end

  def perform_vacuuming
    initial_dirt_count = @dirt_count

    # Snapshot terms so we can safely delete from the tree while iterating.
    @index.keys.each do |term|
      fields_data = @index.get(term)
      next if fields_data.nil?

      fields_data.keys.each do |field_id|
        field_index = fields_data[field_id]
        field_index.keys.each do |short_id|
          next if @document_ids.key?(short_id)

          if field_index.size <= 1
            fields_data.delete(field_id)
          else
            field_index.delete(short_id)
          end
        end
      end

      @index.delete(term) if @index.get(term).empty?
    end

    @dirt_count -= initial_dirt_count
  end

  # --- query execution ----------------------------------------------------

  # Positional slots for the internal per-document result record built during
  # query execution. This record is transient: {#term_results} assembles it, the
  # combinators merge it across query specs, and {#search} transcribes it into
  # the public result Hash. Because it is never serialized and always has exactly
  # these three fields, it is a small positional Array rather than a symbol-keyed
  # Hash — an embedded 3-element Array is several times lighter than a 3-key Hash,
  # and this record is the dominant search allocation. The MATCH slot stays a Hash
  # (it is handed straight to the public output). JS keeps a plain object here;
  # this positional form is a Ruby-side allocation optimization, output-identical.
  R_SCORE = 0
  R_TERMS = 1
  R_MATCH = 2
  private_constant :R_SCORE, :R_TERMS, :R_MATCH

  def execute_query(query, search_options = {})
    return execute_wildcard_query(search_options) if wildcard_query?(query)

    unless query.is_a?(String)
      options = search_options.merge(query)
      options.delete(:queries)
      results = query[:queries].map { |subquery| execute_query(subquery, options) }
      return combine_results(results, options[:combine_with])
    end

    options = { tokenize: @options[:tokenize], process_term: @options[:process_term] }
              .merge(@options[:search_options])
              .merge(search_options)

    search_tokenize = options[:tokenize]
    search_process_term = options[:process_term]

    terms = search_tokenize.call(query)
                           .flat_map { |term| search_process_term.call(term) }
                           .select { |term| truthy?(term) }

    queries = terms.each_with_index.map { |term, i| term_to_query_spec(options, term, i, terms) }
    results = queries.map { |spec| execute_query_spec(spec, options) }

    combine_results(results, options[:combine_with])
  end

  def execute_query_spec(query, search_options)
    options = @options[:search_options].merge(search_options)

    fields = options[:fields] || @options[:fields]
    boosts = {}
    fields.each do |field|
      boost = options[:boost][field]
      boosts[field] = truthy?(boost) ? boost : 1
    end

    boost_document = options[:boost_document]
    max_fuzzy = options[:max_fuzzy]
    bm25params = options[:bm25]
    population_override = field_statistics_mode_from_options(options)
    population = population_override || @field_statistics_mode
    ensure_field_statistics_available!(population) if population == FIELD_STATISTICS_PRESENT

    weights = DEFAULT_SEARCH_OPTIONS[:weights].merge(options[:weights] || {})
    fuzzy_weight = weights[:fuzzy]
    prefix_weight = weights[:prefix]

    term = query[:term]
    term_boost = query[:term_boost]

    data = @index.get(term)
    results = term_results(term, term, 1, term_boost, data, boosts, boost_document, bm25params, {}, population)

    prefix_matches = @index.at_prefix(term) if query[:prefix]

    fuzzy_matches = nil
    if query[:fuzzy]
      fuzzy = query[:fuzzy] == true ? 0.2 : query[:fuzzy]
      max_distance = fuzzy < 1 ? [max_fuzzy, (term.length * fuzzy).round].min : fuzzy
      fuzzy_matches = @index.fuzzy_get(term, max_distance) if truthy?(max_distance)
    end

    if prefix_matches
      prefix_matches.each do |prefix_term, prefix_data|
        distance = prefix_term.length - term.length
        next if distance.zero?

        # A term reachable by prefix is always scored as a prefix result, never a
        # fuzzy one.
        fuzzy_matches&.delete(prefix_term)

        # Weight approaches 0 as distance grows, starting from prefix_weight. The
        # decay is slower than for fuzzy matches, since prefix matches stay
        # relevant longer.
        weight = prefix_weight * prefix_term.length / (prefix_term.length + (0.3 * distance))
        term_results(term, prefix_term, weight, term_boost, prefix_data, boosts, boost_document,
                     bm25params, results, population)
      end
    end

    if fuzzy_matches
      fuzzy_matches.keys.each do |fuzzy_term|
        fuzzy_data, distance = fuzzy_matches[fuzzy_term]
        next if distance.zero?

        weight = fuzzy_weight * fuzzy_term.length / (fuzzy_term.length + distance)
        term_results(term, fuzzy_term, weight, term_boost, fuzzy_data, boosts, boost_document,
                     bm25params, results, population)
      end
    end

    results
  end

  def execute_wildcard_query(search_options)
    results = {}
    options = @options[:search_options].merge(search_options)
    boost_document = options[:boost_document]

    @document_ids.each do |short_id, id|
      score = boost_document ? boost_document.call(id, "", @stored_fields[short_id]) : 1
      results[short_id] = [score, [], {}]
    end

    results
  end

  def combine_results(results, combine_with = OR)
    return {} if results.empty?

    combine_with = OR if combine_with.nil?
    operator = combine_with.to_s.downcase

    combinator =
      case operator
      when OR then method(:combine_or)
      when AND then method(:combine_and)
      when AND_NOT then method(:combine_and_not)
      end
    raise Error, "Invalid combination operator: #{combine_with}" if combinator.nil?

    combined = results.reduce { |a, b| combinator.call(a, b) }
    combined.nil? ? {} : combined
  end

  def combine_or(a, b)
    b.each do |doc_id, b_value|
      existing = a[doc_id]
      if existing.nil?
        a[doc_id] = b_value
      else
        existing[R_SCORE] += b_value[R_SCORE]
        existing[R_MATCH].merge!(b_value[R_MATCH])
        assign_unique_terms(existing[R_TERMS], b_value[R_TERMS])
      end
    end
    a
  end

  def combine_and(a, b)
    combined = {}
    b.each do |doc_id, b_value|
      existing = a[doc_id]
      next if existing.nil?

      assign_unique_terms(existing[R_TERMS], b_value[R_TERMS])
      combined[doc_id] = [
        existing[R_SCORE] + b_value[R_SCORE],
        existing[R_TERMS],
        existing[R_MATCH].merge!(b_value[R_MATCH])
      ]
    end
    combined
  end

  def combine_and_not(a, b)
    b.each_key { |doc_id| a.delete(doc_id) }
    a
  end

  def term_results(source_term, derived_term, term_weight, term_boost, field_term_data, field_boosts,
                   boost_document_fn, bm25params, results = {},
                   population = FIELD_STATISTICS_LEGACY)
    return results if field_term_data.nil?

    population = classify_field_statistics_mode(population) || population
    use_population = population == FIELD_STATISTICS_PRESENT && @field_statistics_available
    field_lengths = use_population ? @field_population_length : @field_length
    average_lengths = use_population ? @field_population_avg : @avg_field_length

    field_boosts.each do |field, field_boost|
      field_id = @field_ids[field]

      field_term_freqs = field_term_data[field_id]
      next if field_term_freqs.nil?

      matching_fields = field_term_freqs.size
      avg_field_length = average_lengths[field_id]

      field_term_freqs.keys.each do |doc_id|
        unless @document_ids.key?(doc_id)
          remove_term(field_id, doc_id, derived_term)
          matching_fields -= 1
          next
        end

        doc_boost =
          if boost_document_fn
            boost_document_fn.call(@document_ids[doc_id], derived_term, @stored_fields[doc_id])
          else
            1
          end
        next unless truthy?(doc_boost)

        term_freq = field_term_freqs[doc_id]
        field_length = field_lengths[doc_id][field_id]

        raw_score = calc_bm25_score(term_freq, matching_fields, @document_count, field_length, avg_field_length,
                                    bm25params)
        weighted_score = term_weight * term_boost * field_boost * doc_boost * raw_score

        result = results[doc_id]
        if result
          result[R_SCORE] += weighted_score
          assign_unique_term(result[R_TERMS], source_term)
          match = result[R_MATCH][derived_term]
          if match
            match.push(field)
          else
            result[R_MATCH][derived_term] = [field]
          end
        else
          results[doc_id] = [weighted_score, [source_term], { derived_term => [field] }]
        end
      end
    end

    results
  end

  def calc_bm25_score(term_freq, matching_count, total_count, field_length, avg_field_length, bm25params)
    k = bm25params[:k]
    b = bm25params[:b]
    d = bm25params[:d]

    inv_doc_freq = Math.log(1 + ((total_count - matching_count + 0.5) / (matching_count + 0.5)))
    inv_doc_freq * (d + (term_freq * (k + 1) / (term_freq + (k * (1 - b + (b * field_length / avg_field_length))))))
  end

  def term_to_query_spec(options, term, i, terms)
    fuzzy_opt = options[:fuzzy]
    fuzzy = fuzzy_opt.respond_to?(:call) ? fuzzy_opt.call(term, i, terms) : (fuzzy_opt || false)

    prefix_opt = options[:prefix]
    prefix = prefix_opt.respond_to?(:call) ? prefix_opt.call(term, i, terms) : (prefix_opt == true)

    boost_term_opt = options[:boost_term]
    term_boost = boost_term_opt.respond_to?(:call) ? boost_term_opt.call(term, i, terms) : 1

    { term: term, fuzzy: fuzzy, prefix: prefix, term_boost: term_boost }
  end

  def assign_unique_term(target, term)
    target.push(term) unless target.include?(term)
  end

  def assign_unique_terms(target, source)
    source.each { |term| target.push(term) unless target.include?(term) }
  end

  # Stable descending sort by score, matching JavaScript's stable Array#sort:
  # equal scores keep their original relative order.
  def sort_by_score(results)
    # sort_by.with_index sorts the results directly, keeping the stable descending
    # order (index breaks score ties, matching JS's stable sort) without the
    # per-element [result, i] pairs and the trailing map that each_with_index needs.
    results.sort_by.with_index { |result, i| [-result[:score], i] }
  end

  def wildcard_query?(query)
    query.equal?(WILDCARD)
  end

  # Mirrors JavaScript truthiness for the small set of values that flow through
  # options and processed terms: nil, false, 0, "", and NaN are falsy.
  def truthy?(value)
    return false if value.nil? || value == false || value == "" || value == 0
    return false if value.is_a?(Float) && value.nan?

    true
  end
end
