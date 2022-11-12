# concurrent-map

<a href="https://docs.rs/concurrent-map"><img src="https://docs.rs/concurrent-map/badge.svg"></a>

Lock-free linearizable map.

* `get`, `insert`, `cas`, `remove`, `iter`, `range`
* fully lock-free node splits and merges based on [sled](https://sled.rs) battle-tested implementation
