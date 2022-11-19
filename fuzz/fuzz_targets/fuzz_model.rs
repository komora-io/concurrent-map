#![no_main]
#[macro_use]
extern crate libfuzzer_sys;
extern crate arbitrary;
extern crate concurrent_map;

use arbitrary::Arbitrary;

const KEYSPACE: u64 = 128;

#[derive(Debug)]
enum Op {
    Insert {
        key: u64,
        value: u64,
    },
    Remove {
        key: u64,
    },
    Cas {
        key: u64,
        old: Option<u64>,
        new: Option<u64>,
    },
    Range {
        start: u64,
        end: u64,
    },
}

impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(if u.ratio(1, 2)? {
            Op::Insert {
                key: u.int_in_range(0..=KEYSPACE as u64)?,
                value: u.int_in_range(0..=KEYSPACE as u64)?,
            }
        } else if u.ratio(1, 2)? {
            Op::Remove {
                key: u.int_in_range(0..=KEYSPACE as u64)?,
            }
        } else if u.ratio(1, 2)? {
            Op::Cas {
                key: u.int_in_range(0..=KEYSPACE as u64)?,
                old: if u.ratio(1, 2)? {
                    Some(u.int_in_range(0..=KEYSPACE as u64)?)
                } else {
                    None
                },
                new: if u.ratio(1, 2)? {
                    Some(u.int_in_range(0..=KEYSPACE as u64)?)
                } else {
                    None
                },
            }
        } else {
            let start = u.int_in_range(0..=KEYSPACE as u64)?;
            let end = (start + 1).max(u.int_in_range(0..=KEYSPACE as u64)?);
            Op::Range { start, end }
        })
    }
}

fuzz_target!(|ops: Vec<Op>| {
    let tree = concurrent_map::ConcurrentMap::<_, _, 8>::default();
    let mut model = std::collections::BTreeMap::new();

    for (_i, op) in ops.into_iter().enumerate() {
        match op {
            Op::Insert { key, value } => {
                assert_eq!(tree.insert(key, value), model.insert(key, value));
            }
            Op::Remove { key } => {
                assert_eq!(tree.remove(&key), model.remove(&key));
            }
            Op::Range { start, end } => {
                let mut model_iter = model.range(start..end);
                let mut tree_iter = tree.range(start..end);

                for (k1, v1) in &mut model_iter {
                    let (k2, v2) = tree_iter.next().unwrap();
                    assert_eq!((k1, v1), (&k2, &v2));
                }

                assert_eq!(tree_iter.next(), None);
            }
            Op::Cas { key, old, new } => {
                let succ = if old == model.get(&key).copied() {
                    if let Some(n) = new {
                        model.insert(key, n);
                    } else {
                        model.remove(&key);
                    }
                    true
                } else {
                    false
                };

                let res = tree.cas(key, old.as_ref(), new);

                if succ {
                    assert!(res.is_ok());
                } else {
                    assert!(res.is_err());
                }
            }
        };

        for (key, value) in &model {
            assert_eq!(tree.get(key), Some(*value));
        }

        /* TODO
        for (key, value) in &tree {
            assert_eq!(model.get(key), Some(value));
        }
        */
    }

    let mut model_iter = model.iter();
    let mut tree_iter = tree.iter();

    for (k1, v1) in &mut model_iter {
        let (k2, v2) = tree_iter.next().unwrap();
        assert_eq!((k1, v1), (&k2, &v2));
    }

    assert_eq!(tree_iter.next(), None);
});
