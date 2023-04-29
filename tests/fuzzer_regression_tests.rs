/// Tests are placed here when the fuzzer finds bugs, so we can re-play them deterministically
/// even if tested on a machine without a fuzzer corpus loaded etc...

#[test]
fn test_00() {
    let map = concurrent_map::ConcurrentMap::<_, _, 8>::default();
    map.insert(0, 0);
    map.insert(103, 103);

    let mut range = map.range(0..1);

    let expected = (0, 0);

    assert_eq!(range.next().unwrap(), expected);
    assert_eq!(range.next_back().unwrap(), expected);
}

#[test]
fn test_01() {
    let map = concurrent_map::ConcurrentMap::<_, _, 8>::default();
    let mut model = std::collections::BTreeMap::new();

    let items = [95, 126, 2, 73, 0, 106, 54];

    for item in items {
        map.insert(item, item);
        model.insert(item, item);
    }

    let bounds = 81..124;

    let expected = model
        .range(bounds.clone())
        .map(|(k, _v)| *k)
        .collect::<Vec<_>>();
    let actual = map
        .range(bounds.clone())
        .map(|(k, _v)| k)
        .collect::<Vec<_>>();
    assert_eq!(expected, actual);

    let expected_rev = model
        .range(bounds.clone())
        .rev()
        .map(|(k, _v)| *k)
        .collect::<Vec<_>>();
    let actual_rev = map.range(bounds).rev().map(|(k, _v)| k).collect::<Vec<_>>();

    assert_eq!(expected_rev, actual_rev);
}
