---
title: Latent Semantic Indexer (LSI)
layout: default
---

# {{ page.title }}

Latent Semantic Indexing engines are not as fast or as small as Bayesian classifiers, but are more flexible, providing fast search, and clustering detection as well as semantic analysis of the text that
theoretically simulates human learning.

## Usage

```ruby
require 'classifier-reborn'
lsi = ClassifierReborn::LSI.new
strings = [["This text deals with dogs. Dogs.", :dog],
           ["This text involves dogs too. Dogs!", :dog],
           ["This text revolves around cats. Cats.", :cat],
           ["This text also involves cats. Cats!", :cat],
           ["This text involves birds. Birds.", :bird]]
strings.each { |x| lsi.add_item x.first, x.last }

lsi.search("dog", 3)
#=> ["This text deals with dogs. Dogs.",
#    "This text involves dogs too. Dogs!",
#    "This text also involves cats. Cats!"]

lsi.find_related(strings[2], 2)
#=> ["This text revolves around cats. Cats.",
#    "This text also involves cats. Cats!"]

lsi.classify "This text is also about dogs!"
#=> :dog
```

## Projection search

`search` is lexical-first by default: the query is compared in the full term
space, and terms the index does not know are simply ignored. You can instead
measure an external query strictly inside the reduced (truncated) SVD space
chosen when the index was built by passing `projection: true`.

```ruby
# Build an index whose truncation keeps only the dominant term direction.
lsi = ClassifierReborn::LSI.new(auto_rebuild: false)
lsi.add_item "maple syrup", "Tree"
lsi.add_item "maple leaf", "Tree"
lsi.add_item "maple tree", "Tree"
lsi.add_item "maple wood", "Tree"
lsi.add_item "maple bark", "Tree"
lsi.add_item "orange citrus", "Fruit"
lsi.build_index(1.0 / 6.0) # keeps a single singular direction

# The query is encoded with the index's existing word list and projected onto
# the retained singular directions, so it never adds terms or retrains the
# index.
lsi.search("maple", 10, projection: true)
#=> the maple documents, scored along the retained direction

# "orange" lives in a direction the truncation discarded, so adding it to the
# query changes neither the projected vector nor the main-direction scores.
lsi.search("maple", 10, projection: true) ==
  lsi.search("maple orange", 10, projection: true)
#=> true

# A query with no component in the retained space has no valid projection.
lsi.search("orange", 10, projection: true)
#=> []
```

The default `lsi.search("maple")` call keeps the historical lexical-first
behavior, so the two modes can be compared against each other. Projection
searches do not modify the index: the word list, version and indexed vectors
are identical before and after repeated queries.

Please see the [ClassifierReborn::LSI documentation](http://www.rubydoc.info/gems/classifier-reborn/ClassifierReborn/LSI) for more information.
It is possible to index, search and classify with more than just simple strings.

**Note:** *A Redis backend is not implemented for LSI yet.*

## References

* [Latent Semantic Indexing](http://www.c2.com/cgi/wiki?LatentSemanticIndexing)
* [Smart Bookmarking](http://web.archive.org/web/20061126153713/http://www.chadfowler.com/index.cgi/Computing/LatentSemanticIndexing.rdoc)
* [Latent semantic analysis](http://en.wikipedia.org/wiki/Latent_semantic_analysis)
