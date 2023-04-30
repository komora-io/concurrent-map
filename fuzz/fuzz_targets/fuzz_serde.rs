#![no_main]
#[macro_use]
extern crate libfuzzer_sys;
extern crate concurrent_map;

fuzz_target!(|data: Vec<u64>| {
    let tree = concurrent_map::ConcurrentMap::<_, _, 4>::default();

    for item in data {
        tree.insert(item, item);
    }

    let serialized = bincode::serialize(&tree).unwrap();
    let deserialized = bincode::deserialize(&serialized).unwrap();
    assert_eq!(tree, deserialized);
});
