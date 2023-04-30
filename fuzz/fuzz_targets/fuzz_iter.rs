#![no_main]
#[macro_use]
extern crate libfuzzer_sys;
extern crate concurrent_map;

fuzz_target!(|data: Vec<u64>| {
    let mut model = std::collections::BTreeMap::default();

    for item in &data {
        model.insert(*item, *item);
    }

    let model_iter: Vec<_> = model.into_iter().collect();

    {
        let tree = concurrent_map::ConcurrentMap::<_, _, 4>::default();

        for item in &data {
            tree.insert(*item, *item);
        }

        let self_iter: Vec<_> = tree.iter().collect();

        assert_eq!(self_iter, model_iter);
    }
    {
        let tree = concurrent_map::ConcurrentMap::<_, _, 5>::default();

        for item in &data {
            tree.insert(*item, *item);
        }

        let self_iter: Vec<_> = tree.iter().collect();

        assert_eq!(self_iter, model_iter);
    }
    {
        let tree = concurrent_map::ConcurrentMap::<_, _, 7>::default();

        for item in &data {
            tree.insert(*item, *item);
        }

        let self_iter: Vec<_> = tree.iter().collect();

        assert_eq!(self_iter, model_iter);
    }
    {
        let tree = concurrent_map::ConcurrentMap::<_, _, 8>::default();

        for item in &data {
            tree.insert(*item, *item);
        }

        let self_iter: Vec<_> = tree.iter().collect();

        assert_eq!(self_iter, model_iter);
    }
});
