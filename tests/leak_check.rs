use std::time::Instant;

use concurrent_map::ConcurrentMap;

mod alloc {
    use std::alloc::{Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[global_allocator]
    static ALLOCATOR: Alloc = Alloc;

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    static FREED: AtomicUsize = AtomicUsize::new(0);
    static RESIDENT: AtomicUsize = AtomicUsize::new(0);

    pub fn allocated() -> usize {
        ALLOCATED.swap(0, Ordering::Relaxed) / 1_000_000
    }

    pub fn freed() -> usize {
        FREED.swap(0, Ordering::Relaxed) / 1_000_000
    }

    pub fn resident() -> usize {
        RESIDENT.load(Ordering::Relaxed) / 1_000_000
    }

    #[derive(Default, Debug, Clone, Copy)]
    struct Alloc;

    unsafe impl std::alloc::GlobalAlloc for Alloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ret = System.alloc(layout);
            assert_ne!(
                ret,
                std::ptr::null_mut(),
                "alloc returned null pointer for layout {layout:?}"
            );
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            RESIDENT.fetch_add(layout.size(), Ordering::Relaxed);
            std::ptr::write_bytes(ret, 0xa1, layout.size());
            ret
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            std::ptr::write_bytes(ptr, 0xde, layout.size());
            FREED.fetch_add(layout.size(), Ordering::Relaxed);
            RESIDENT.fetch_sub(layout.size(), Ordering::Relaxed);
            System.dealloc(ptr, layout)
        }
    }
}

#[test]
fn leak_check() {
    let n: u32 = 16 * 1024;

    let concurrency = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(8)
        * 2;

    let run = |tree: ConcurrentMap<u32, u32, 5>, barrier: &std::sync::Barrier, low_bits| {
        let shift = concurrency.next_power_of_two().trailing_zeros();
        let unique_key = |key| (key << shift) | low_bits;

        barrier.wait();
        for key in 0..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i), None);
            tree.insert(i, i);
            assert_eq!(tree.get(&i), Some(i), "failed to get key {i}");
        }
        for key in 0_u32..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i), Some(i), "failed to get key {i}");
        }
        for key in 0_u32..n {
            let i = unique_key(key);
            assert_eq!(
                tree.cas(i, Some(&i), Some(unique_key(key * 2))),
                Ok(Some(i)),
                "failed to get key {i}"
            );
        }
        let visible: std::collections::HashMap<u32, u32> = tree.iter().collect();

        for key in 0_u32..n {
            let i = unique_key(key);
            let v = unique_key(key * 2);
            assert_eq!(visible.get(&i).copied(), Some(v), "failed to get key {i}");
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

    let before = Instant::now();
    let resident_before = alloc::resident();

    let tree = ConcurrentMap::default();
    std::thread::scope(|s| {
        for _ in 0..64 {
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(concurrency));
            let mut threads = vec![];
            for i in 0..concurrency {
                let tree_2 = tree.clone();
                let barrier_2 = barrier.clone();

                let thread = s.spawn(move || run(tree_2, &barrier_2, u32::try_from(i).unwrap()));
                threads.push(thread);
            }
            for thread in threads {
                thread.join().unwrap();
            }
        }
    });

    drop(tree);

    let resident_after = alloc::resident();

    println!(
        "{:.2} million wps {} mb allocated {} mb freed {} mb resident to insert {} items",
        n as f64 / (before.elapsed().as_micros().max(1)) as f64,
        alloc::allocated(),
        alloc::freed(),
        resident_after,
        n,
    );

    assert_eq!(
        resident_after - resident_before,
        0,
        "leaked {}mb",
        resident_after
    );
}
