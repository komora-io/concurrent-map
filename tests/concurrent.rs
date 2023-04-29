use concurrent_map::ConcurrentMap;

#[test]
fn concurrent_tree() {
    let n: u16 = 1024;
    let concurrency = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(8)
        * 2;

    let run = |tree: ConcurrentMap<u16, u16, 8>, barrier: &std::sync::Barrier, low_bits| {
        let shift = concurrency.next_power_of_two().trailing_zeros();
        let unique_key = |key| (key << shift) | low_bits;

        barrier.wait();
        for key in 0..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i), None);
            tree.insert(i, i);
            assert_eq!(tree.get(&i), Some(i), "failed to get key {i}");
        }
        for key in 0_u16..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i), Some(i), "failed to get key {i}");
        }
        for key in 0_u16..n {
            let i = unique_key(key);
            assert_eq!(
                tree.cas(i, Some(&i), Some(unique_key(key * 2))),
                Ok(Some(i)),
                "failed to get key {i}"
            );
        }
        let visible: std::collections::HashMap<u16, u16> = tree.iter().collect();
        let visible_rev: std::collections::HashMap<u16, u16> = tree.iter().rev().collect();

        for key in 0_u16..n {
            let i = unique_key(key);
            let v = unique_key(key * 2);
            assert_eq!(visible.get(&i).copied(), Some(v), "failed to get key {i}");
            assert_eq!(
                visible_rev.get(&i).copied(),
                Some(v),
                "failed to get key {i}"
            );
        }

        for key in 0..n {
            let i = unique_key(key);
            let v = unique_key(key * 2);
            assert_eq!(tree.remove(&i), Some(v));
        }
        for key in 0..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i), None, "failed to get key {i}");
        }
    };

    let tree = ConcurrentMap::default();

    std::thread::scope(|s| {
        for _ in 0..64 {
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(concurrency));
            let mut threads = vec![];
            for i in 0..concurrency {
                let tree_2 = tree.clone();
                let barrier_2 = barrier.clone();

                let thread = s.spawn(move || run(tree_2, &barrier_2, u16::try_from(i).unwrap()));
                threads.push(thread);
            }
            for thread in threads {
                thread.join().unwrap();
            }
        }
    });
}

#[test]
fn big_scan() {
    let n: u32 = 16 * 1024 * 1024;
    let concurrency = 8;
    let stride = n / concurrency;

    let fill =
        |tree: ConcurrentMap<u32, u32>, barrier: &std::sync::Barrier, start_fill, stop_fill| {
            barrier.wait();
            let insert = std::time::Instant::now();
            for i in start_fill..stop_fill {
                tree.insert(i, i);
            }
            let insert_elapsed = insert.elapsed();
            println!(
                "{} inserts/s, total {} in {:?}",
                (u64::from(stride) * 1000)
                    / u64::try_from(insert_elapsed.as_millis()).unwrap_or(u64::MAX),
                stop_fill - start_fill,
                insert_elapsed
            );
        };

    let read = |tree: ConcurrentMap<u32, u32>, barrier: &std::sync::Barrier| {
        barrier.wait();
        let scan = std::time::Instant::now();
        let count = tree.range(..).take(stride as _).count();
        assert_eq!(count, stride as _);
        let scan_elapsed = scan.elapsed();
        let scan_micros = scan_elapsed.as_micros().max(1) as f64;
        println!(
            "{} scanned items/s, total {:?}",
            ((f64::from(stride) * 1_000_000.0) / scan_micros) as u64,
            scan_elapsed
        );
    };

    let tree = ConcurrentMap::default();
    let barrier = std::sync::Barrier::new(concurrency as _);

    std::thread::scope(|s| {
        let mut threads = vec![];
        for i in 0..concurrency {
            let tree_2 = tree.clone();
            let barrier_2 = &barrier;

            let start_fill = i * stride;
            let stop_fill = (i + 1) * stride;

            let thread = s.spawn(move || fill(tree_2, barrier_2, start_fill, stop_fill));
            threads.push(thread);
        }
        for thread in threads {
            thread.join().unwrap();
        }
    });

    std::thread::scope(|s| {
        let mut threads = vec![];
        for _ in 0..concurrency {
            let tree_2 = tree.clone();
            let barrier_2 = &barrier;

            let thread = s.spawn(move || read(tree_2, barrier_2));
            threads.push(thread);
        }
        for thread in threads {
            thread.join().unwrap();
        }
    });
}

#[test]
fn bulk_load() {
    let n: u64 = 16 * 1024 * 1024;

    let concurrency = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(8) as u64;

    let run = |tree: ConcurrentMap<u64, u64>, barrier: &std::sync::Barrier, low_bits| {
        let shift = concurrency.next_power_of_two().trailing_zeros();
        let unique_key = |key| (key << shift) | low_bits;

        barrier.wait();
        for key in 0..n / concurrency {
            let i = unique_key(key);
            tree.insert(i, i);
        }
    };

    let tree = ConcurrentMap::default();

    std::thread::scope(|s| {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(1 + concurrency as usize));
        let mut threads = vec![];
        for i in 0..concurrency {
            let tree_2 = tree.clone();
            let barrier_2 = barrier.clone();

            let thread = s.spawn(move || run(tree_2, &barrier_2, i));
            threads.push(thread);
        }
        barrier.wait();
        let insert = std::time::Instant::now();
        for thread in threads {
            thread.join().unwrap();
        }
        let insert_elapsed = insert.elapsed();
        println!(
            "{} bulk inserts/s, total {:?}",
            (n * 1000) / u64::try_from(insert_elapsed.as_millis()).unwrap_or(u64::MAX),
            insert_elapsed
        );
    });
}
