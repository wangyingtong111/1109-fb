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

  # Raised when a phrase query is issued against an index that does not carry
  # term positions. This is deliberately distinct from an empty result: an
  # index built (or loaded) with +positions: true+ that simply has no matching
  # phrase returns +[]+, while a position-less index reports that the phrase
  # capability is unavailable at all. Check {#positions_enabled?} (or
  # {#phrase_search_available?}) to test the capability without raising.
  class PhraseSearchUnavailable < Error; end

  # A contiguous-phrase query leaf: only documents whose analyzed terms occur
  # at consecutive token positions, in the same field and in query order, match.
  #
  # Build one with {MiniFTS.phrase} (or +MiniFTS::Phrase.new+), or pass the
  # equivalent Hash leaf +{ phrase: "red oak" }+ to {#search}. The phrase text
  # is tokenized with the search tokenizer and processed per field, so a
  # field-specific +process_term+ applies exactly as it did at indexing time.
  #
  # @example
  #   ms = MiniFTS.new(fields: ["title"], positions: true)
  #   ms.search(MiniFTS.phrase("red oak"))
  #   ms.search({ combine_with: "AND", queries: ["oak", { phrase: "red oak" }] })
  class Phrase
    # @return [String, Array<String>] the phrase text, or its pre-tokenized
    #   terms when constructed from an Array.
    attr_reader :value

    # @param value [String, Array<String>] the phrase text (tokenized on search)
    #   or an Array of already-tokenized terms.
    def initialize(value)
      unless value.is_a?(String) || value.is_a?(Array)
        raise Error, "MiniFTS: a phrase must be a String or an Array of terms"
      end

      @value = value
    end

    # @return [String] the phrase text; pre-tokenized terms join with a space.
    def to_s
      @value.is_a?(Array) ? @value.join(" ") : @value
    end
  end

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
    auto_vacuum: true,
    positions: false
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

  # Builds a {Phrase} query leaf.
  #
  # @param value [String, Array<String>] the phrase text, or pre-tokenized terms.
  # @return [Phrase]
  def self.phrase(value)
    Phrase.new(value)
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

    # Whether this engine was asked to record analyzed-term positions (the
    # +positions: true+ option), and whether they are actually being recorded.
    # They diverge on load: a payload that lacks positions cannot be trusted to
    # answer phrases even if the option asks for them, so the capability is
    # reported unavailable instead of silently missing documents (see
    # {#phrase_search_available?}).
    @positions_enabled = truthy?(@options[:positions])
    @position_tracking = @positions_enabled

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

      tokens.each_with_index do |term, position|
        processed_term = process_term.call(term, field)
        slot = track_positions? ? [] : nil
        if processed_term.is_a?(Array)
          processed_term.each do |t|
            add_term(field_id, short_document_id, t)
            slot.push(t) if slot
          end
        elsif truthy?(processed_term)
          add_term(field_id, short_document_id, processed_term)
          slot.push(processed_term) if slot
        end
        record_position_slot(short_document_id, field_id, slot) if slot
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

    @positions.delete(short_id)
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

    @positions.delete(short_id)
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

  # @return [Boolean] whether this engine records analyzed-term positions, i.e.
  #   whether it was constructed with +positions: true+ (or loaded an index
  #   that carried them).
  def positions_enabled?
    @position_tracking
  end

  # @return [Boolean] whether phrase queries can actually be answered right now.
  #   True when positions are being recorded — because this engine was
  #   constructed with +positions: true+ and its documents were indexed here, or
  #   because it loaded an index whose payload carried them. False for an engine
  #   built without positions and for an index loaded from a position-less
  #   (legacy) payload, even when +positions: true+ was requested: {#search}
  #   reports those as {PhraseSearchUnavailable}, distinct from an enabled
  #   capability that simply found no contiguous phrase.
  def phrase_search_available?
    @position_tracking
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
  #
  # When the positions capability is active (see {#positions_enabled?}) the
  # analyzed-term positions travel in an additional +"positions"+ key. An index
  # built without positions (the default) serializes exactly as before: no extra
  # key, same bytes, so the original format stays loadable and comparable.
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
      "index" => index
    }

    # Only live documents are emitted, so a discarded (or replaced) version's
    # positions can never come back on reload.
    plain["positions"] = positions_plain_object if @position_tracking

    plain["serializationVersion"] = 2
    plain
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
    # short_document_id => field_id => [term, position] pairs. Populated only
    # when positions are being tracked; this side table is deliberately outside
    # the inverted index so the default (position-less) postings keep their exact
    # shape and score.
    @positions = {}
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

    # A payload without "positions" was written by an engine (or a JS build)
    # that never recorded them. The capability is then genuinely absent for
    # these documents — not merely unmatched — so tracking is switched off and
    # phrase queries report it, even if positions: true was requested (a loaded
    # index cannot be re-analyzed without the original documents).
    payload_positions = js["positions"]
    @position_tracking = !payload_positions.nil?
    @positions = {}

    unless payload_positions.nil?
      payload_positions.each do |short_id, fields|
        document_positions = {}
        fields.each do |field_id, slots|
          document_positions[field_id.to_i] = slots.map { |slot| slot.to_a }
        end
        @positions[short_id.to_i] = document_positions
      end
    end
  end

  # The optional "positions" section of the plain object: each live document's
  # per-field token slots, as arrays of processed terms. Documents are keyed by
  # their short ids (matching "documentIds"/"fieldLength") and only documents
  # still in the index are emitted — a discarded document's slots are gone the
  # moment it is discarded, so neither a tombstone nor a replaced document's old
  # version can be resurrected by a save-and-reload.
  def positions_plain_object
    positions = {}
    @positions.each do |short_id, fields|
      next unless @document_ids.key?(short_id)

      document_positions = {}
      fields.each { |field_id, slots| document_positions[field_id] = slots }
      positions[short_id] = document_positions unless document_positions.empty?
    end
    positions
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

  # --- positions ----------------------------------------------------------

  # Whether positions are being recorded for documents indexed from now on.
  # (Serialization asks the same question to decide whether the optional
  # "positions" section is emitted.)
  def track_positions?
    @position_tracking
  end

  # Records the analyzed terms that share one token position, so a phrase can
  # later require them at consecutive slots in the same field. A +slot+ is the
  # (possibly empty) list of processed terms the tokenizer/processor produced
  # for that token — empty tokens and dropped terms still advance the position,
  # which is what makes "red  old oak" and a stop-word-removing +process_term+
  # behave like the field's own token stream.
  def record_position_slot(short_document_id, field_id, slot)
    document_positions = (@positions[short_document_id] ||= {})
    (document_positions[field_id] ||= []).push(slot)
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
    return execute_phrase_query(query, search_options) if phrase_query?(query)

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

    weights = DEFAULT_SEARCH_OPTIONS[:weights].merge(options[:weights] || {})
    fuzzy_weight = weights[:fuzzy]
    prefix_weight = weights[:prefix]

    term = query[:term]
    term_boost = query[:term_boost]

    data = @index.get(term)
    results = term_results(term, term, 1, term_boost, data, boosts, boost_document, bm25params)

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
        term_results(term, prefix_term, weight, term_boost, prefix_data, boosts, boost_document, bm25params, results)
      end
    end

    if fuzzy_matches
      fuzzy_matches.keys.each do |fuzzy_term|
        fuzzy_data, distance = fuzzy_matches[fuzzy_term]
        next if distance.zero?

        weight = fuzzy_weight * fuzzy_term.length / (fuzzy_term.length + distance)
        term_results(term, fuzzy_term, weight, term_boost, fuzzy_data, boosts, boost_document, bm25params, results)
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

  # --- phrase queries -----------------------------------------------------

  def phrase_query?(query)
    query.is_a?(Phrase) || (query.is_a?(Hash) && query.key?(:phrase))
  end

  def phrase_value(query)
    query.is_a?(Phrase) ? query.value : query[:phrase]
  end

  # Executes a contiguous-phrase leaf. Per field it gathers candidates exactly
  # as a normal AND over that field's analyzed phrase terms (so BM25+ scores,
  # matched terms and matched fields come from the same path), then keeps only
  # documents whose positions show the terms adjacent, in order, within that
  # field. A phrase can match in any one field, so the per-field results are
  # combined with OR.
  #
  # An index that cannot answer phrases raises {PhraseSearchUnavailable} rather
  # than returning +[]+, so "this index has no positions" is never confused with
  # "the phrase did not occur".
  def execute_phrase_query(query, search_options)
    unless phrase_search_available?
      raise PhraseSearchUnavailable,
            "MiniFTS: phrase search is unavailable: this index carries no term positions " \
            "(build it with positions: true, or load an index serialized with positions)"
    end

    options = { tokenize: @options[:tokenize], process_term: @options[:process_term] }
              .merge(@options[:search_options])
              .merge(search_options)

    if query.is_a?(Hash)
      options = options.merge(query)
      options.delete(:phrase)
      options.delete(:queries)
    end

    fields = options[:fields] || @options[:fields]
    tokens = phrase_tokens(phrase_value(query), options[:tokenize])
    return {} if tokens.empty?

    process_term = options[:process_term]
    field_results = []

    fields.each do |field|
      field_id = @field_ids[field]
      next if field_id.nil?

      slots, terms = analyzed_phrase_slots(tokens, field, process_term)
      # The phrase can still occupy positions here only if some token survived;
      # the slots are kept either way so dropped tokens still consume a position.
      next if terms.empty?

      specs = terms.each_with_index.map do |term, i|
        boost = options[:boost_term].respond_to?(:call) ? options[:boost_term].call(term, i, terms) : 1
        { term: term, fuzzy: false, prefix: false, term_boost: boost }
      end
      field_options = options.merge(fields: [field])
      candidates = combine_results(specs.map { |spec| execute_query_spec(spec, field_options) }, AND)

      accepted = {}
      candidates.each do |short_id, data|
        document_positions = @positions[short_id]
        next if document_positions.nil?

        accepted[short_id] = data if slots_align?(document_positions[field_id], slots)
      end
      field_results.push(accepted)
    end

    combine_results(field_results, OR)
  end

  # Splits a phrase value into its raw query tokens. A String goes through the
  # search tokenizer (so it splits exactly as the indexed text did); an Array is
  # taken as already tokenized.
  def phrase_tokens(value, search_tokenize)
    if value.is_a?(Array)
      value.select { |token| truthy?(token) }
    else
      search_tokenize.call(value.to_s)
    end
  end

  # Analyzes the phrase token stream for one field, the way indexing did, and
  # returns +[slots, terms]+: +slots+ is an Array of term-sets, one per token
  # position (an empty set marks a token this field's +process_term+ dropped,
  # which still consumes a position), and +terms+ is the unique non-empty terms.
  def analyzed_phrase_slots(tokens, field, process_term)
    terms = []
    slots = tokens.map do |token|
      processed = call_process_term(process_term, token, field)
      slot_terms =
        if processed.is_a?(Array)
          processed.select { |term| truthy?(term) }
        elsif truthy?(processed)
          [processed]
        else
          []
        end
      slot_terms.each { |term| terms.push(term) unless terms.include?(term) }
      slot_terms
    end

    [slots, terms]
  end

  # Calls +process_term+ with the field name when it accepts one. A lambda or
  # method object of arity 1 rejects the extra argument (procs ignore it), so
  # the arity decides how many arguments to pass.
  def call_process_term(process_term, term, field)
    if process_term.respond_to?(:arity) && process_term.arity == 1
      process_term.call(term)
    else
      process_term.call(term, field)
    end
  end

  # Whether this document's slots for the field contain the phrase at some
  # offset: every non-empty phrase slot must intersect the document slot at the
  # same relative position, in order.
  def slots_align?(document_slots, phrase_slots)
    return false if document_slots.nil?

    document_length = document_slots.length
    phrase_length = phrase_slots.length
    return false if phrase_length.zero? || phrase_length > document_length

    last_offset = document_length - phrase_length
    offset = 0
    while offset <= last_offset
      aligned = true
      index = 0
      while index < phrase_length
        wanted = phrase_slots[index]
        unless wanted.empty?
          present = document_slots[offset + index]
          if present.nil? || (wanted & present).empty?
            aligned = false
            break
          end
        end
        index += 1
      end
      return true if aligned

      offset += 1
    end
    false
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
                   boost_document_fn, bm25params, results = {})
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
