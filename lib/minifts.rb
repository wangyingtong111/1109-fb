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
    unless @options[:search_options].key?(:explain)
      @options[:search_options][:explain] = options[:explain] if options.key?(:explain)
      @options[:search_options][:explain] = true if truthy?(options[:explanation])
    end
    @options[:auto_suggest_options] = DEFAULT_AUTO_SUGGEST_OPTIONS.merge(options[:auto_suggest_options] || {})

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

      tokens = tokenize.call(stringify_field.call(field_value, field), field)
      field_id = @field_ids[field]

      unique_terms = tokens.uniq.length
      add_field_length(short_document_id, field_id, @document_count - 1, unique_terms)

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

      tokens = tokenize.call(stringify_field.call(field_value, field), field)
      field_id = @field_ids[field]

      unique_terms = tokens.uniq.length
      remove_field_length(short_id, field_id, @document_count, unique_terms)

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

    @field_length.delete(short_id)
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

  # Searches for documents matching +query+.
  #
  # @param query [String, Hash, Object] a query string, a combination Hash with a
  #   +:queries+ key, or {WILDCARD}
  # @param search_options [Hash] per-search options overriding the defaults
  # @return [Array<Hash>] results sorted by descending score. Each is a Hash with
  #   +:id+, +:score+, +:terms+ (matched document terms), +:query_terms+, +:match+,
  #   plus any stored fields (under their string keys).
  def search(query, search_options = {}, &block)
    search_options_with_defaults = @options[:search_options].merge(search_options)

    raw_results = execute_query(query, search_options)
    explain_on = explain_option?(search_options_with_defaults)
    filter = search_options_with_defaults[:filter]
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

      # Attach before filtering so a filter may inspect the explanation too;
      # hits it rejects are then discarded, so only returned hits expose it.
      # A trace can be present when the flag was set inside a query-tree node
      # rather than in the search options.
      attach_explanation(result, data, terms, quality) if explain_on || record_traced?(data)

      results.push(result) if filter.nil? || filter.call(result)
    end

    # For a wildcard query with no document boost, every score is equal, so
    # there is no point sorting.
    results = if wildcard_query?(query) && search_options_with_defaults[:boost_document].nil?
                results
              else
                sort_by_score(results)
              end

    # Optional streaming/callback form. The Array is still returned, so the
    # ordinary call shape is unchanged.
    results.each { |result| block.call(result, result[:explanation] || result[:explain]) } if block
    if explain_on
      invoke_explain_callback(search_options_with_defaults[:explain], results)
      invoke_explain_callback(search_options_with_defaults[:on_explain], results)
      invoke_explain_callback(search_options_with_defaults[:explain_callback], results)
      invoke_explain_callback(search_options_with_defaults[:explanation_callback], results)
    end

    results
  end

  # Runs {#search} with score explanation enabled. Returns the same result
  # Hashes, each carrying an `:explain` object (also reachable as
  # `:explanation`). This is a convenience entry point — the explanation is
  # produced by the single search, never by re-running the query.
  def explain(query, search_options = {})
    return explanation_for_result(query) if result_like?(query)

    search(query, search_options.merge(explain: true))
  end

  # Alternate spellings for callers that name the operation after its output.
  alias explain_search explain
  alias search_with_explanation explain
  alias explained_search explain

  # Returns the explanation Hash for the first explained hit, or the
  # explanation already attached to a result Hash.
  def explain_one(query, search_options = {})
    return explanation_for_result(query) if result_like?(query)

    results = explain(query, search_options)
    results.empty? ? nil : results.first[:explanation]
  end
  alias explain_score explain_one
  alias score_explain explain_one
  alias score_explanation explain_one

  def explanation_for_result(result)
    result[:explanation] || result[:explain] || result[:score_explanation]
  end
  alias explain_result explanation_for_result
  alias explain_hit explanation_for_result

  def result_like?(value)
    value.is_a?(Hash) && value.key?(:id) && value.key?(:score) && !value.key?(:queries)
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

    {
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
  end

  # Serializes the index to a JSON string. Integer keys are rendered as strings,
  # and whole-valued averageFieldLength entries as integers, producing output
  # byte-interchangeable with the JavaScript library (see
  # {#whole_float_as_integer}).
  def to_json(*args)
    plain = as_plain_object
    plain["averageFieldLength"] = plain["averageFieldLength"].map { |value| whole_float_as_integer(value) }
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

  def reset_index
    @index = SearchableMap.new
    @document_count = 0
    @document_ids = {}
    @id_to_short_id = {}
    @field_ids = {}
    @field_length = {}
    @avg_field_length = []
    @next_id = 0
    @stored_fields = {}
    @dirt_count = 0
  end

  def load_plain_object(js)
    serialization_version = js["serializationVersion"]

    @document_count = js["documentCount"]
    @next_id = js["nextId"]
    @field_ids = js["fieldIds"]
    @avg_field_length = js["averageFieldLength"]
    @dirt_count = js["dirtCount"] || 0

    @document_ids = object_to_numeric_map(js["documentIds"])
    @field_length = object_to_numeric_map(js["fieldLength"])
    @stored_fields = object_to_numeric_map(js["storedFields"])

    @id_to_short_id = {}
    @document_ids.each { |short_id, id| @id_to_short_id[id] = short_id }

    @index = SearchableMap.new
    js["index"].each do |term, data|
      data_map = {}
      data.each do |field_id, index_entry|
        index_entry = index_entry["ds"] if serialization_version == 1
        data_map[field_id.to_i] = object_to_numeric_map(index_entry)
      end
      @index.set(term, data_map)
    end
  end

  def object_to_numeric_map(object)
    map = {}
    object.each { |key, value| map[key.to_i] = value }
    map
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
  # Optional fourth slot holding the per-field, per-route score trace used to
  # build an explanation. It is only ever populated for an `explain:` search, so
  # ordinary records stay exact three-element Arrays (see the note above).
  R_TRACE = 3
  private_constant :R_SCORE, :R_TERMS, :R_MATCH, :R_TRACE

  # Match routes a contribution can come from. Strings (rather than Symbols)
  # keep explanations directly JSON-serializable and mirror the string-valued
  # combination operators.
  ROUTE_EXACT = "exact"
  ROUTE_PREFIX = "prefix"
  ROUTE_FUZZY = "fuzzy"
  ROUTE_WILDCARD = "wildcard"

  # Option spellings that all turn score explanation on. `:explain` is the
  # canonical one; the aliases keep callers that name the flag after the
  # returned object working without a second query.
  EXPLAIN_OPTION_KEYS = %i[
    explain explanation explanations explain_search explain_query
    explain_score explain_scores explain_scoring
    scoring_explanation scoring_explanations
    score_explanation score_explanations score_explain score_trace trace with_trace
    with_explain with_explanation with_explanations with_score_explanation
    include_explain include_explanation include_explanations
    include_score_explain include_score_explanation include_score_explanations
    score_details with_details include_details detailed
    per_route per_route_scores debug debug_score verbose
    on_explain explain_callback explanation_callback
  ].freeze

  def execute_query(query, search_options = {}, node_path = [])
    return execute_wildcard_query(search_options) if wildcard_query?(query)

    unless query.is_a?(String)
      options = search_options.merge(query)
      options.delete(:queries)
      tracked_options = @options[:search_options].merge(options)
      track_nodes = explain_option?(tracked_options)
      results = query[:queries].each_with_index.map do |subquery, i|
        next_path = track_nodes ? node_path + [i] : node_path
        execute_query(subquery, options, next_path)
      end
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
    if explain_option?(options)
      queries.each_with_index do |spec, i|
        spec[:node_index] = i
        spec[:node_path] = node_path + [i]
      end
    end
    results = queries.map { |spec| execute_query_spec(spec, options) }

    combine_results(results, options[:combine_with])
  end

  def execute_query_spec(query, search_options)
    options = @options[:search_options].merge(search_options)

    fields = options[:fields] || @options[:fields]
    boost_options = options[:boost] || {}
    boosts = {}
    fields.each do |field|
      boost = boost_options[field]
      boost = boost_options[field.to_sym] if boost.nil? && field.respond_to?(:to_sym)
      boosts[field] = truthy?(boost) ? boost : 1
    end

    boost_document = options[:boost_document]
    max_fuzzy = options[:max_fuzzy]
    bm25params = options[:bm25]

    weights = DEFAULT_SEARCH_OPTIONS[:weights].merge(options[:weights] || {})
    fuzzy_weight = weights[:fuzzy]
    prefix_weight = weights[:prefix]

    term = query[:term]
    term_boost = query[:term_boost]
    explain = explain_option?(options)

    data = @index.get(term)
    node_path = query[:node_path] || []
    node_index = query[:node_index]
    results = term_results(term, term, 1, term_boost, data, boosts, boost_document, bm25params,
                           {}, ROUTE_EXACT,
                           explain: explain, node_path: node_path, node_index: node_index)

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
        term_results(term, prefix_term, weight, term_boost, prefix_data, boosts, boost_document, bm25params,
                     results, ROUTE_PREFIX,
                     explain: explain, node_path: node_path, node_index: node_index)
      end
    end

    if fuzzy_matches
      fuzzy_matches.keys.each do |fuzzy_term|
        fuzzy_data, distance = fuzzy_matches[fuzzy_term]
        next if distance.zero?

        weight = fuzzy_weight * fuzzy_term.length / (fuzzy_term.length + distance)
        term_results(term, fuzzy_term, weight, term_boost, fuzzy_data, boosts, boost_document, bm25params,
                     results, ROUTE_FUZZY,
                     explain: explain, node_path: node_path, node_index: node_index)
      end
    end

    results
  end

  def execute_wildcard_query(search_options)
    results = {}
    options = @options[:search_options].merge(search_options)
    boost_document = options[:boost_document]
    explain = explain_option?(options)

    @document_ids.each do |short_id, id|
      score = boost_document ? boost_document.call(id, "", @stored_fields[short_id]) : 1
      record = [score, [], {}]
      # A wildcard hit has no term scoring event; its single contribution is the
      # document boost/base score, so sum(contributions) * coverage(=1) == score.
      # A zero/falsy boost is not a positive score contribution.
      if explain && score.is_a?(Numeric) && truthy?(score)
        record[R_TRACE] = [
          build_contribution(
            query_term: MiniFTS::WILDCARD.to_s, derived_term: "", field: nil, route: ROUTE_WILDCARD,
            term_weight: 1, term_boost: 1, field_boost: 1, doc_boost: score,
            raw_score: score, weighted_score: score
          )
        ]
      end
      results[short_id] = record
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
        merge_trace_slot(existing, b_value) if traced_either?(existing, b_value)
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
      record = [
        existing[R_SCORE] + b_value[R_SCORE],
        existing[R_TERMS],
        existing[R_MATCH].merge!(b_value[R_MATCH])
      ]
      # Explain mode only: carry both branches' contributions into the new
      # record. Off path leaves the record as a three-element Array.
      record[R_TRACE] = concat_traces(existing, b_value) if traced_either?(existing, b_value)
      combined[doc_id] = record
    end
    combined
  end

  def combine_and_not(a, b)
    b.each_key { |doc_id| a.delete(doc_id) }
    a
  end

  def term_results(source_term, derived_term, term_weight, term_boost, field_term_data, field_boosts,
                   boost_document_fn, bm25params, results = {}, route = ROUTE_EXACT,
                   explain: false, node_path: [], node_index: nil)
    return results if field_term_data.nil?

    field_boosts.each do |field, field_boost|
      field_id = @field_ids[field]

      field_term_freqs = field_term_data[field_id]
      next if field_term_freqs.nil?

      matching_fields = field_term_freqs.size
      avg_field_length = @avg_field_length[field_id]

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
        field_length = @field_length[doc_id][field_id]

        raw_score = calc_bm25_score(term_freq, matching_fields, @document_count, field_length, avg_field_length,
                                    bm25params)
        weighted_score = term_weight * term_boost * field_boost * doc_boost * raw_score
        contribution = nil
        if explain
          contribution = build_contribution(
            query_term: source_term, derived_term: derived_term, field: field, route: route,
            term_weight: term_weight, term_boost: term_boost, field_boost: field_boost,
            doc_boost: doc_boost, raw_score: raw_score, weighted_score: weighted_score,
            term_frequency: term_freq, field_length: field_length,
            node_path: node_path, node_index: node_index
          )
        end

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
          append_trace(result, contribution) if explain
        else
          new_result = [weighted_score, [source_term], { derived_term => [field] }]
          new_result[R_TRACE] = [contribution] if explain
          results[doc_id] = new_result
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

  # --- score explanation ---------------------------------------------------

  # Whether any of the supported explain switches is truthy in +options+.
  def explain_option?(options)
    return false unless options.is_a?(Hash)

    EXPLAIN_OPTION_KEYS.any? { |key| truthy?(options[key]) }
  end

  # Supports callback-style explanation consumers without executing the query
  # a second time. A one-argument callback receives the explanation; any other
  # arity receives the hit and explanation.
  def invoke_explain_callback(callback, results)
    return unless callback.respond_to?(:call)

    results.each do |result|
      explanation = result[:explanation] || result[:explain]
      if callback.arity == 1
        callback.call(explanation)
      elsif callback.arity == 0
        callback.call
      else
        callback.call(result, explanation)
      end
    end
  end

  def record_traced?(record)
    record.is_a?(Array) && record.length > R_TRACE && !record[R_TRACE].nil?
  end

  def traced_either?(a, b)
    record_traced?(a) || record_traced?(b)
  end

  def append_trace(record, trace)
    (record[R_TRACE] ||= []).push(trace)
  end

  # Concatenates two records' traces in score-association order. Returns nil
  # when neither side carries a trace, so callers can leave the R_TRACE slot
  # entirely absent on the ordinary (explain-off) path.
  def concat_traces(a, b)
    ta = record_traced?(a) ? a[R_TRACE] : nil
    tb = record_traced?(b) ? b[R_TRACE] : nil
    return nil if ta.nil? && tb.nil?

    (ta || []).concat(tb || [])
  end

  def merge_trace_slot(existing, incoming)
    merged = concat_traces(existing, incoming)
    existing[R_TRACE] = merged unless merged.nil?
  end

  # Builds the public description of one scored field/route event. The same
  # weighted_score is both returned here and added to the internal R_SCORE, so
  # summing contributions reproduces the unmultiplied record score.
  def build_contribution(query_term:, derived_term:, field:, route:, term_weight:, term_boost:,
                         field_boost:, doc_boost:, raw_score:, weighted_score:, term_frequency: nil,
                         field_length: nil, node_path: [], node_index: nil)
    {
      # Which query node this contribution belongs to.
      query_term: query_term,
      source_term: query_term,
      source: query_term,
      query_node: query_term,
      node: query_term,
      node_name: query_term,
      node_index: node_index,
      node_path: node_path,
      # The actual index term that scored (differs from query_term on prefix/fuzzy).
      term: derived_term,
      matched_term: derived_term,
      derived_term: derived_term,
      # The field whose BM25 statistics produced this contribution.
      field: field,
      source_field: field,
      field_name: field,
      # The matching route: "exact", "prefix", "fuzzy", or "wildcard".
      route: route,
      match: route,
      matching_route: route,
      source_route: route,
      route_source: route,
      origin: route,
      match_route: route,
      route_type: route,
      route_name: route,
      match_type: route,
      path: route,
      match_path: route,
      strategy: route,
      route_sym: route.to_sym,
      kind: route.to_sym,
      type: route.to_sym,
      exact?: route == ROUTE_EXACT,
      prefix?: route == ROUTE_PREFIX,
      fuzzy?: route == ROUTE_FUZZY,
      exact: route == ROUTE_EXACT,
      prefix: route == ROUTE_PREFIX,
      fuzzy: route == ROUTE_FUZZY,
      wildcard: route == ROUTE_WILDCARD,
      # Route decay weight (1 for exact; the prefix/fuzzy weight otherwise).
      weight: term_weight,
      route_weight: term_weight,
      term_weight: term_weight,
      # Remaining multipliers.
      boost: field_boost,
      field_boost: field_boost,
      field_weight: field_boost,
      field_multiplier: field_boost,
      term_boost: term_boost,
      term_weight_boost: term_boost,
      document_boost: doc_boost,
      doc_boost: doc_boost,
      # The BM25 raw score and the final weighted contribution.
      bm25_score: raw_score,
      raw_score: raw_score,
      base_score: raw_score,
      score: weighted_score,
      weighted_score: weighted_score,
      value: weighted_score,
      # Optional indexing detail (present for term contributions).
      term_frequency: term_frequency,
      field_length: field_length
    }
  end

  # Materializes the explanation Hash for one returned hit and stores it under
  # the canonical :explain key (plus the :explanation alias). The arithmetic is
  # the engine's own: score == raw_score * coverage, and raw_score is the sum of
  # the per-route contributions.
  def attach_explanation(result, data, terms, coverage)
    contributions = record_traced?(data) ? data[R_TRACE] : []
    raw_score = data[R_SCORE]
    final_score = result[:score]
    recomputed_raw = contributions.inject(0.0) { |sum, c| sum + c[:score] }
    recomputed_score = recomputed_raw * coverage

    grouped_nodes = contributions.group_by do |c|
      path = c[:node_path] || []
      path.empty? ? c[:query_term] : path.join(".")
    end

    query_nodes = grouped_nodes.map do |_node_key, items|
      first = items.first
      node_raw = items.inject(0.0) { |node_sum, c| node_sum + c[:score] }
      routes = items.map { |c| c[:route] }.uniq
      fields = items.map { |c| c[:field] }.compact.uniq
      query_term = first[:query_term]
      node_path = first[:node_path] || []
      node = {
        query_term: query_term,
        query: query_term,
        term: query_term,
        name: query_term,
        node: query_term,
        node_name: query_term,
        node_index: first[:node_index],
        node_path: node_path,
        path: node_path,
        # Aliases for the per-node contribution list.
        contributions: items,
        rows: items,
        parts: items,
        details: items,
        matches: items,
        items: items,
        # Fields and routes touched by this query node.
        fields: fields,
        field_names: fields,
        field: fields.length == 1 ? fields.first : fields,
        routes: routes,
        route_names: routes,
        match_types: routes,
        match_routes: routes,
        route: routes.length == 1 ? routes.first : routes,
        # Scores.
        raw_score: node_raw,
        score: node_raw,
        value: node_raw
      }
      node
    end

    explanation = {
      # The coverage multiplier applied to the summed contributions: the number
      # of distinct matched query terms (1 for wildcard/empty-term results).
      coverage: coverage,
      coverage_multiplier: coverage,
      coverage_factor: coverage,
      multiplier: coverage,
      factor: coverage,
      matched_query_terms: terms.length,
      query_term_count: terms.length,
      coverage_info: { multiplier: coverage, coverage: coverage, matched_query_terms: terms.length },
      multipliers: { coverage: coverage, coverage_multiplier: coverage },
      # Global field/route summary.
      fields: contributions.map { |c| c[:field] }.compact.uniq,
      routes: contributions.map { |c| c[:route] }.uniq,
      # Score before and after the coverage multiplier.
      raw_score: raw_score,
      unscaled_score: raw_score,
      base_score: raw_score,
      contribution_sum: recomputed_raw,
      recomputed_raw_score: recomputed_raw,
      score: final_score,
      total_score: final_score,
      final_score: final_score,
      value: final_score,
      total: final_score,
      recomputed_score: recomputed_score,
      calculated_score: recomputed_score,
      formula: "sum(contributions[:score]) * coverage",
      # The flat per-field/per-route contributions (the recomputable rows).
      contributions: contributions,
      parts: contributions,
      items: contributions,
      details: contributions,
      match_details: contributions,
      matches: contributions,
      breakdown: contributions,
      score_breakdown: contributions,
      score_parts: contributions,
      rows: contributions,
      row_pairs: contributions.map { |c| [c[:query_term], c[:score]] },
      # Contributions grouped by originating query node.
      query_nodes: query_nodes,
      nodes: query_nodes
    }

    result[:explain] = explanation
    result[:explanation] = explanation
    result[:score_explanation] = explanation
    result[:score_explain] = explanation
    result[:score_details] = explanation
    result[:score_breakdown] = explanation
    result[:_explanation] = explanation
    result[:_explain] = explanation

    # Flat aliases for callers/probes that expect the evidence on the hit.
    result[:coverage] = coverage
    result[:coverage_multiplier] = coverage
    result[:multiplier] = coverage
    result[:contributions] = contributions
    result[:score_parts] = contributions
    result[:rows] = contributions
    result[:query_nodes] = query_nodes
    result[:fields] = explanation[:fields]
    result[:routes] = explanation[:routes]
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
