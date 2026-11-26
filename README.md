# philiprehberger-fuzzy_match

[![Tests](https://github.com/philiprehberger/rb-fuzzy-match/actions/workflows/ci.yml/badge.svg)](https://github.com/philiprehberger/rb-fuzzy-match/actions/workflows/ci.yml)
[![Gem Version](https://badge.fury.io/rb/philiprehberger-fuzzy_match.svg)](https://rubygems.org/gems/philiprehberger-fuzzy_match)
[![Last updated](https://img.shields.io/github/last-commit/philiprehberger/rb-fuzzy-match)](https://github.com/philiprehberger/rb-fuzzy-match/commits/main)

Fuzzy string matching with Levenshtein, Damerau-Levenshtein, Jaro-Winkler, Hamming, LCS, token-based, and phonetic algorithms

## Requirements

- Ruby >= 3.1

## Installation

Add to your Gemfile:

```ruby
gem "philiprehberger-fuzzy_match"
```

Or install directly:

```bash
gem install philiprehberger-fuzzy_match
```

## Usage

```ruby
require "philiprehberger/fuzzy_match"

# Individual algorithms
Philiprehberger::FuzzyMatch.levenshtein('kitten', 'sitting')   # => 3
Philiprehberger::FuzzyMatch.jaro_winkler('martha', 'marhta')   # => ~0.96
Philiprehberger::FuzzyMatch.dice_coefficient('night', 'nacht') # => 0.25

# Normalized ratio (0.0 to 1.0)
Philiprehberger::FuzzyMatch.ratio('kitten', 'sitting')  # => ~0.57
```

### Damerau-Levenshtein (Transposition-Aware)

```ruby
# Counts adjacent transpositions as 1 edit (Levenshtein counts them as 2)
Philiprehberger::FuzzyMatch.damerau_levenshtein('teh', 'the')   # => 1
Philiprehberger::FuzzyMatch.damerau_ratio('teh', 'the')         # => ~0.667
```

### Jaro-Winkler Prefix Gate

By default the Winkler common-prefix bonus is applied unconditionally. Pass
`prefix_threshold:` to gate it on the plain Jaro base score: the bonus is only
added when the base score reaches the threshold.

```ruby
# Base Jaro is 0.9167 >= 0.7, so the classic formula still boosts to 0.95
Philiprehberger::FuzzyMatch.jaro_winkler('abcdxxxx', 'abcdxxxy', prefix_threshold: 0.7)
# => 0.95

# Base Jaro is 2/3 < 0.7, so the shared prefix is not rewarded
Philiprehberger::FuzzyMatch.jaro_winkler('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)
# => 0.6666666666666666

# Without the option the historical score is unchanged (0.8)
Philiprehberger::FuzzyMatch.jaro_winkler('abcdxxxx', 'abcdyyyy')
# => 0.8
```

Inspect the full breakdown behind a score — the Jaro base score, the common
prefix length (still capped by the original rule at 4), whether the bonus was
applied, and the final score:

```ruby
Philiprehberger::FuzzyMatch.jaro_winkler_components('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)
# => { base_score: 0.6666666666666666, prefix_length: 4, bonus_applied: false, score: 0.6666666666666666 }

# Same hash via the explain alias
Philiprehberger::FuzzyMatch.jaro_winkler_explain('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)
Philiprehberger::FuzzyMatch.jaro_winkler_bonus?('abcdxxxx', 'abcdyyyy', prefix_threshold: 0.7)  # => false
```

`.rank` (and `.closest_n` / `.top_n`) score every candidate with the same
components, so passing the same `prefix_threshold:` keeps ranking and direct
scoring in agreement. Ties keep input order, and `explain: true` attaches the
per-candidate breakdown:

```ruby
candidates = %w[abcdyyyy ABCDXXXY abcdxxxy]
results = Philiprehberger::FuzzyMatch.rank('abcdxxxx', candidates, prefix_threshold: 0.7)
# => [{ value: "ABCDXXXY", score: 0.95 },
#     { value: "abcdxxxy", score: 0.95 },
#     { value: "abcdyyyy", score: 0.6666666666666666 }]

results = Philiprehberger::FuzzyMatch.rank('abcdxxxx', candidates, prefix_threshold: 0.7, explain: true)
results.last[:components]
# => { base_score: 0.6666666666666666, prefix_length: 4, bonus_applied: false, score: 0.6666666666666666 }
```

### Longest Common Subsequence

```ruby
Philiprehberger::FuzzyMatch.lcs('kitten', 'sitting')       # => 4
Philiprehberger::FuzzyMatch.lcs_ratio('kitten', 'sitting')  # => ~0.615
```

### Best Match

```ruby
candidates = %w[Ruby Python Rust JavaScript]
result = Philiprehberger::FuzzyMatch.best('rubyy', candidates)
result[:match]  # => "Ruby"
result[:score]  # => 0.8
```

### Ranked Search

```ruby
candidates = %w[commit comment command compare]
results = Philiprehberger::FuzzyMatch.search('comit', candidates, threshold: 0.5)
# => [{ match: "commit", score: 0.8333 }, { match: "comment", score: 0.7143 }, ...]
```

### Rank candidates

```ruby
candidates = %w[commit comment command compare]
Philiprehberger::FuzzyMatch.rank('comit', candidates)
# => [{ value: "commit", score: ... }, { value: "comment", score: ... }, ...]

# Choose algorithm (:jaro_winkler default, :dice, or :levenshtein)
Philiprehberger::FuzzyMatch.rank('comit', candidates, algorithm: :levenshtein)
```

### Top N Matches

```ruby
candidates = %w[commit comment command compare zebra]
results = Philiprehberger::FuzzyMatch.closest_n('comit', candidates, n: 3)
# => [{ match: "commit", score: ... }, { match: "comment", score: ... }, { match: "command", score: ... }]

# Choose algorithm (:jaro_winkler default, :dice, or :levenshtein)
Philiprehberger::FuzzyMatch.closest_n('comit', candidates, n: 2, algorithm: :levenshtein)
```

### Top-N Matches

```ruby
candidates = %w[commit comment command compare zebra]
results = Philiprehberger::FuzzyMatch.top_n('comit', candidates, n: 3)
# => [{ value: "commit", similarity: ... }, { value: "comment", similarity: ... }, { value: "command", similarity: ... }]

# Filter out low-similarity entries with min_similarity (default 0.0)
Philiprehberger::FuzzyMatch.top_n('comit', candidates, n: 5, min_similarity: 0.7)
# => only entries whose similarity >= 0.7

# Choose algorithm (:jaro_winkler default, :dice, or :levenshtein)
Philiprehberger::FuzzyMatch.top_n('comit', candidates, n: 2, algorithm: :levenshtein)
```

### Did-You-Mean Suggestions

```ruby
Philiprehberger::FuzzyMatch.suggest('comit', %w[commit comment zebra], threshold: 0.6, max: 3)
# => ["commit", "comment"]
```

### Phonetic Matching

```ruby
Philiprehberger::FuzzyMatch.soundex('Robert')    # => "R163"
Philiprehberger::FuzzyMatch.metaphone('Smith')    # => "SM0"
Philiprehberger::FuzzyMatch.phonetic_match?('Robert', 'Rupert')  # => true
```

### Similarity Matrix

```ruby
strings = %w[hello helo world]
matrix = Philiprehberger::FuzzyMatch.similarity_matrix(strings)
# => { "hello" => { "hello" => 1.0, "helo" => 0.9333, "world" => 0.4667 }, ... }

# Filter to only high-similarity pairs
matrix = Philiprehberger::FuzzyMatch.similarity_matrix(strings, threshold: 0.8)
# => { "hello" => { "hello" => 1.0, "helo" => 0.9333 }, ... }

# Choose algorithm (:jaro_winkler default, :dice, or :levenshtein)
Philiprehberger::FuzzyMatch.similarity_matrix(strings, algorithm: :dice)
```

### Deduplication

```ruby
Philiprehberger::FuzzyMatch.deduplicate(%w[hello helo world wrld], threshold: 0.8)
# => ["hello", "world"]
```

### Hamming Distance

```ruby
Philiprehberger::FuzzyMatch.hamming('karolin', 'kathrin')  # => 3
Philiprehberger::FuzzyMatch.hamming('abc', 'abc')          # => 0
# Raises Error for different-length strings
```

### Token-Based Matching

```ruby
# Token sort: reorder tokens alphabetically before comparing
Philiprehberger::FuzzyMatch.token_sort_ratio('john smith jr', 'jr john smith')  # => 1.0

# Token set: compare based on token set intersection/union
Philiprehberger::FuzzyMatch.token_set_ratio('new york mets', 'new york mets vs atlanta braves')
# => high score (shared tokens boost similarity)
```

### Partial (Substring) Ratio

```ruby
# "Does string A appear approximately inside string B?"
Philiprehberger::FuzzyMatch.partial_ratio('the cat', 'the cat sat on the mat')
# => 1.0

Philiprehberger::FuzzyMatch.partial_ratio('cat', 'a black cat sat on a mat')
# => 1.0

Philiprehberger::FuzzyMatch.partial_ratio('helo', 'why hello there')
# => ~0.75
```

Slides the shorter string across every same-length window of the longer
one and returns the maximum Levenshtein-based ratio. FuzzyWuzzy parity.

### Weighted Scoring

```ruby
Philiprehberger::FuzzyMatch.weighted_score('kitten', 'sitting',
  weights: { jaro_winkler: 0.5, dice: 0.3, levenshtein_ratio: 0.2 })
# => weighted combination of algorithm scores
# Supported keys: :jaro_winkler, :dice, :levenshtein_ratio, :lcs_ratio, :damerau_ratio
# Weights must sum to 1.0
```

## API

### `Philiprehberger::FuzzyMatch`

| Method | Description |
|--------|-------------|
| `.levenshtein(a, b)` | Levenshtein edit distance (integer) |
| `.jaro_winkler(a, b, prefix_threshold: nil)` | Jaro-Winkler similarity (0.0 to 1.0); optional Jaro base-score gate for the prefix bonus |
| `.jaro_winkler_components(a, b, prefix_threshold: nil)` | Jaro-Winkler breakdown: `{ base_score:, prefix_length:, bonus_applied:, score: }` (aliases: `.jaro_winkler_explain`, `.jaro_winkler_detail`, `.jaro_winkler_breakdown`) |
| `.jaro_winkler_bonus?(a, b, prefix_threshold: nil)` | Whether the common-prefix bonus applies for the pair |
| `.dice_coefficient(a, b)` | Dice coefficient from bigram overlap (0.0 to 1.0) |
| `.damerau_levenshtein(a, b)` | Damerau-Levenshtein distance with transpositions (integer) |
| `.damerau_ratio(a, b)` | Normalized Damerau-Levenshtein similarity (0.0 to 1.0) |
| `.lcs(a, b)` | Longest common subsequence length (integer) |
| `.lcs_ratio(a, b)` | Normalized LCS similarity (0.0 to 1.0) |
| `.ratio(a, b)` | Normalized Levenshtein ratio (0.0 to 1.0) |
| `.partial_ratio(a, b)` | Substring-style similarity: max Levenshtein ratio over windows of the longer string (0.0 to 1.0) |
| `.best(query, candidates, threshold: 0.0)` | Best match as `{ match:, score: }` |
| `.search(query, candidates, threshold: 0.3)` | Ranked array of `{ match:, score: }` |
| `.suggest(query, candidates, threshold: 0.6, max: 5)` | Array of match strings |
| `.rank(query, candidates, algorithm: :jaro_winkler, prefix_threshold: nil, explain: false)` | All candidates sorted desc as `{ value:, score: }` (stable); forwards the Jaro gate and can attach `:components` |
| `.closest_n(query, candidates, n:, algorithm: :jaro_winkler, prefix_threshold: nil, explain: false)` | Top N matches as `{ match:, score: }` sorted by score descending |
| `.top_n(query, candidates, n:, algorithm: :jaro_winkler, min_similarity: 0.0, prefix_threshold: nil, explain: false)` | Top-N matches as `{ value:, similarity: }` sorted by similarity descending, with optional similarity floor |
| `.soundex(string)` | Generate 4-character Soundex code |
| `.metaphone(string)` | Generate Metaphone phonetic code |
| `.phonetic_match?(a, b)` | Check if two strings match phonetically |
| `.hamming(a, b)` | Hamming distance for equal-length strings (integer) |
| `.token_sort_ratio(a, b)` | Token-sorted Jaro-Winkler similarity (0.0 to 1.0) |
| `.token_set_ratio(a, b)` | Token-set-based similarity (0.0 to 1.0) |
| `.weighted_score(a, b, weights:)` | Weighted multi-algorithm score (0.0 to 1.0) |
| `.similarity_matrix(strings, algorithm:, threshold:)` | Pairwise similarity hash-of-hashes for batch comparison |
| `.deduplicate(array, threshold:, algorithm:)` | Group and deduplicate similar strings |

All methods are case-insensitive by default.

## Development

```bash
bundle install
bundle exec rspec
bundle exec rubocop
```

## Support

If you find this project useful:

⭐ [Star the repo](https://github.com/philiprehberger/rb-fuzzy-match)

🐛 [Report issues](https://github.com/philiprehberger/rb-fuzzy-match/issues?q=is%3Aissue+is%3Aopen+label%3Abug)

💡 [Suggest features](https://github.com/philiprehberger/rb-fuzzy-match/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)

❤️ [Sponsor development](https://github.com/sponsors/philiprehberger)

🌐 [All Open Source Projects](https://philiprehberger.com/open-source-packages)

💻 [GitHub Profile](https://github.com/philiprehberger)

🔗 [LinkedIn Profile](https://www.linkedin.com/in/philiprehberger)

## License

[MIT](LICENSE)
