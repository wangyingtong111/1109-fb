## [Unreleased]

- Optional highlight ranges. Build an index with `ranges: true` and search with
  `ranges: true` to receive a `:ranges` entry on each result mapping field name
  to the sorted, unique half-open `[start, stop)` character offsets of the
  original text that matched. Ranges are tracked in a side structure parallel to
  the inverted index (the field text need not be in `store_fields`), attribute
  every derived term back to its source token (so one token expanded by
  `process_term` into several terms yields a single span), and survive
  `to_json` / `load_json` via a new `termRanges` serialization entry.
- `MiniFTS#ranges?` reports whether an index carries range data. Indexes built
  without the option, or archives written before this feature, return `false`;
  a `ranges: true` search against them raises `MiniFTS::RangesUnavailableError`.
- Default searches are unchanged: no `:ranges` key, identical scores and result
  shape, and an index built without `ranges: true` serializes byte-identically.

## [1.0.1] - 2026-07-18

- Require MFA for gem pushes (`rubygems_mfa_required` gemspec metadata). No
  library changes; the packaged files and public API are identical to 1.0.0.

## [1.0.0] - 2026-07-18

- Initial release: a pure-Ruby port of the JavaScript
  [MiniSearch](https://github.com/lucaong/minisearch) full-text search engine.
- In-memory inverted index backed by a radix tree (`MiniFTS::SearchableMap`)
  with exact, prefix, and fuzzy (Levenshtein) lookup.
- BM25+ relevance scoring with field boosting, term boosting, document boosting,
  and per-field weights.
- Query combinators (`OR`, `AND`, `AND_NOT`), combination-tree queries, result
  filtering, and wildcard queries.
- Auto-suggestions (`auto_suggest`).
- Incremental `add` / `remove` / `discard` / `replace`, with lazy index cleanup
  and synchronous (auto-)vacuuming.
- Byte-identical JSON serialization with the JavaScript library (`to_json` /
  `load_json`): a Ruby-serialized index loads and searches identically in
  JavaScript, and vice versa, for any corpus without astral-plane (above
  U+FFFF) characters. Verified bidirectionally against the real
  `minisearch@7.2.0` across 32 scenarios (`rake compat`).
- Runs on every Ruby since 2.4; no runtime dependencies.
- Configurable `logger` for index-corruption warnings; setting `logger: nil`
  silences them without raising.
- Verified bit-for-bit against the reference JavaScript implementation via
  golden and randomized differential tests, plus a full parity suite mirroring
  the reference's own (non-async) test cases, including the movie/song ranking
  sets.
