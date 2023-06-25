use std::thread::scope;
use std::time::Instant;

use concurrent_map::ConcurrentMap;

const PRODUCERS: usize = 128;
const CONSUMERS: usize = 128;
const N: usize = 1024 * 1024;
const PRODUCER_N: usize = N / PRODUCERS;
const CONSUMER_N: usize = N / CONSUMERS;

fn producer(cm: ConcurrentMap<usize, usize>, min: usize, max: usize) {
    for i in min..max {
        cm.insert(i, i);
    }
}

fn consumer(cm: ConcurrentMap<usize, usize>, n: usize) {
    let mut popped = 0;
    while popped < n {
        if let Some((k, v)) = cm.pop_first() {
            assert_eq!(k, v);
            popped += 1;
        }
    }
}

fn main() {
    let cm = ConcurrentMap::default();

    let before = Instant::now();
    scope(|s| {
        let mut handles = vec![];

        for i in 0..PRODUCERS {
            let min = i * PRODUCER_N;
            let max = (i + 1) * PRODUCER_N;
            let cm = cm.clone();
            let handle = s.spawn(move || producer(cm, min, max));
            handles.push(handle);
        }

        for _ in 0..CONSUMERS {
            let cm = cm.clone();
            let handle = s.spawn(move || consumer(cm, CONSUMER_N));
            handles.push(handle);
        }

        for handle in handles.into_iter() {
            handle.join().unwrap()
        }
    });

    let elapsed = before.elapsed();

    let per_second = N as u128 * 1000 / elapsed.as_millis();

    println!(
        "with {} producers and {} consumers, took {:?} to transfer {} items ({} per second)",
        PRODUCERS, CONSUMERS, elapsed, N, per_second
    );
}
