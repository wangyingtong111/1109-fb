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
legacy = ms.to_json   # has positions actually; build a position-less legacy from same docs elsewhere

ms.discard(1)
ms.replace("id" => 2, "title" => "red oak")
ms.add("id" => 1, "title" => "oak red")

puts "before reload: #{ms.search(MiniFTS.phrase("red oak")).map { |r| r[:id] }.inspect}"
json = ms.to_json
reloaded = MiniFTS.load_json(json, opts)
puts "after reload:  #{reloaded.search(MiniFTS.phrase("red oak")).map { |r| r[:id] }.inspect}"
puts "reload avail: #{reloaded.phrase_search_available?}"

# Legacy index: same corpus built WITHOUT positions
legacy_ms = MiniFTS.new(opts.merge(positions: false))
legacy_ms.add_all(docs)
legacy_json = legacy_ms.to_json
puts "legacy payload has positions key: #{JSON.parse(legacy_json).key?("positions")}"
loaded_legacy = MiniFTS.load_json(legacy_json, opts)
begin
  p loaded_legacy.search(MiniFTS.phrase("red oak"))
rescue MiniFTS::PhraseSearchUnavailable => e
  puts "legacy raised: #{e.class}: #{e.message}"
end
puts "legacy avail: #{loaded_legacy.phrase_search_available?} / enabled?: #{loaded_legacy.positions_enabled?}"
puts "legacy plain: #{loaded_legacy.search("red oak", combine_with: "AND").map { |r| r[:id] }.inspect}"

# enabled-but-no-match must be [] not raise
puts "enabled no match: #{reloaded.search(MiniFTS.phrase("blue elm")).inspect}"
puts "same json reload->dump has positions: #{JSON.parse(MiniFTS.load_json(json, opts).to_json).key?("positions")}"
