require "minifts"
processor = ->(word, field = nil) { field == "title" ? word.upcase : word.downcase }
opts = { fields: %w[title text], positions: true, auto_vacuum: false,
         tokenize: ->(v, *_) { v.split }, process_term: processor }
docs = [{ "id" => 1, "title" => "red oak" }, { "id" => 2, "title" => "oak red" },
        { "id" => 3, "title" => "red old oak" }, { "id" => 4, "title" => "red", "text" => "oak" }]
ms = MiniFTS.new(opts); ms.add_all(docs)
p direct_phrase: ms.search(MiniFTS.phrase("red oak")).map { |r| r[:id] }
p hash_leaf: ms.search({ phrase: "red oak" }).map { |r| r[:id] }
p tree: ms.search(combine_with: "AND", queries: ["oak", { phrase: "red oak" }]).map { |r| r[:id] }
p tree2: ms.search(combine_with: "AND", queries: [{ phrase: "red oak" }, "oak"]).map { |r| r[:id] }
p only_phrase_leaf_in_tree: ms.search(combine_with: "AND", queries: [{ phrase: "red oak" }]).map { |r| r[:id] }
p oak_alone: ms.search("oak").map { |r| r[:id] }
