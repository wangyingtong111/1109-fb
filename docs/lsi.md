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

### Multi-label documents and quota voting

A document may carry more than one label by passing several categories to
`#add_item`. When classifying, every selected neighbor normally casts one full
vote per label entry, so listing the same label repeatedly on a document
inflates that label's score:

```ruby
lsi = ClassifierReborn::LSI.new
lsi.add_item "A document about dogs and cats", :dog, :cat, :cat
```

Enable quota voting to give every selected neighbor a single vote, worth its
similarity score, that is split equally across its **unique** labels.
Repeating a label on the same document then never increases its share.
Neighbors without labels cast no vote, and neighbors outside the nearest
neighbor cutoff are never backfilled:

```ruby
lsi = ClassifierReborn::LSI.new(quota_voting: true)
# ...or per call, without changing the instance default:
lsi.scored_categories("some text", 0.30, true)
# the :quota keyword is accepted as well:
lsi.scored_categories("some text", 0.30, quota: true)
```

With two selected neighbors scored `0.8` and `0.5` carrying `[:a, :b, :b]` and
`[:b]` respectively, the full view scores `:a` at `0.4` (`0.8 / 2`) and `:b`
at `0.9` (`0.8 / 2 + 0.5`). Looking only at `:b` at the same cutoff still
yields `0.9`; the hidden `:a` share is not reallocated to `:b`.

Please see the [ClassifierReborn::LSI documentation](http://www.rubydoc.info/gems/classifier-reborn/ClassifierReborn/LSI) for more information.
It is possible to index, search and classify with more than just simple strings.

**Note:** *A Redis backend is not implemented for LSI yet.*

## References

* [Latent Semantic Indexing](http://www.c2.com/cgi/wiki?LatentSemanticIndexing)
* [Smart Bookmarking](http://web.archive.org/web/20061126153713/http://www.chadfowler.com/index.cgi/Computing/LatentSemanticIndexing.rdoc)
* [Latent semantic analysis](http://en.wikipedia.org/wiki/Latent_semantic_analysis)
