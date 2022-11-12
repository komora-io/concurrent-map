# concurrent-map

<a href="https://docs.rs/concurrent-map"><img src="https://docs.rs/concurrent-map/badge.svg"></a>

Lock-free linearizable map.

* `get`, `insert`, `cas`, `remove`, `iter`, `range`
* fully lock-free node splits and merges based on the [sled](https://sled.rs) battle-tested implementation. `concurrent-map` can be though of in some ways as a simplified, in-memory `sled` that supports high-level types.
* initially designed for use in sled's next generation object store, [marble](https://github.com/komora-io/marble).
