#![no_main]
#[macro_use]
extern crate libfuzzer_sys;
extern crate arbitrary;
extern crate concurrent_map;

use arbitrary::Arbitrary;

const KEYSPACE: u64 = 255;

#[derive(Debug)]
enum Op {
    Insert { key: u64, value: u64 },
    Remove { key: u64 },
}

impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(if bool::arbitrary(u).unwrap_or(true) {
            Op::Insert {
                key: u.int_in_range(0..=KEYSPACE as u64).unwrap_or(0),
                value: u.int_in_range(0..=KEYSPACE as u64).unwrap_or(0),
            }
        } else {
            Op::Remove {
                key: u.int_in_range(0..=KEYSPACE as u64).unwrap_or(0),
            }
        })
    }
}

fuzz_target!(|ops: Vec<Op>| {
    let mut tree = concurrent_map::ConcurrentMap::default();
    let mut model = std::collections::BTreeMap::new();

    for op in ops {
        match op {
            Op::Insert { key, value } => {
                assert_eq!(
                    tree.insert(key, value).map(|arc| *arc),
                    model.insert(key, value)
                );
            }
            Op::Remove { key } => {
                assert_eq!(tree.remove(&key).map(|arc| *arc), model.remove(&key));
            }
        };

        for (key, value) in &model {
            assert_eq!(tree.get(key).as_deref(), Some(value));
        }

        /* TODO
        for (key, value) in &tree {
            assert_eq!(model.get(key), Some(value));
        }
        */
    }
});
