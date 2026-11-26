require "minifts"
processor = ->(word, field = nil) { field == "title" ? word.upcase : word.downcase }
opts = { fields: %w[title text], positions: true, auto_vacuum: false,
         tokenize: ->(v, *_) { v.split }, process_term: processor }
docs = [
  { "id" => 1, "title" => "red oak" },
  { "id" => 2, "title" => "oak red" },
  { "id" => 3, "title" => "red old oak" },
  { "id" => 4, "title" => "red", "text" => "oak" },
]
ms = MiniFTS.new(opts)
ms.add_all(docs)
p phrase: ms.search(MiniFTS.phrase("red oak")).map { |r| r[:id] }
p hash_leaf: ms.search(phrase: "red oak").map { |r| r[:id] }
p and_plain: ms.search("red oak", combine_with: "AND").map { |r| r[:id] }
p tree: ms.search(combine_with: "AND", queries: ["oak", { phrase: "red oak" }]).map { |r| r[:id] }
p avail: ms.phrase_search_available?
r = ms.search(MiniFTS.phrase("red oak")).first
p match: r[:match], terms: r[:terms], query_terms: r[:query_terms], score: r[:score]
