# concurrent-map

<a href="https://docs.rs/concurrent-map"><img src="https://docs.rs/concurrent-map/badge.svg"></a>

Lock-free linearizable map.

* `get`, `insert`, `cas`, `remove`, `iter`, `range`, `get_gt`, `get_gte`, `get_lt`, `get_lte`, `first`, `last`, `pop_first`, `pop_last`
* fully lock-free node splits and merges based on the [sled](https://sled.rs) battle-tested implementation. `concurrent-map` can be though of in some ways as a simplified, in-memory `sled` that supports high-level types.
* initially designed for use in sled's next generation object store, [marble](https://github.com/komora-io/marble).

The `ConcurrentMap` allows users to tune the tree fan-out (`FANOUT`)
and the underlying memory reclamation granularity (`LOCAL_GC_BUFFER_SIZE`)
for achieving desired performance properties. The defaults are pretty good
for most use cases but if you want to squeeze every bit of performance out
for your particular workload, tweaking them based on realistic measurements
may be beneficial. See the `ConcurrentMap` docs for more details.

If you want to use a custom key type, you must
implement the `Minimum` trait,
allowing the left-most side of the tree to be
created before inserting any data.

This is an ordered data structure, and supports very high throughput iteration over
lexicographically sorted ranges of values. If you are looking for simple point operation
performance, you may find a better option among one of the many concurrent
hashmap implementations that are floating around. Pay for what you actually use :)
