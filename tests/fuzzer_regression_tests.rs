/// Tests are placed here when the fuzzer finds bugs, so we can re-play them deterministically
/// even if tested on a machine without a fuzzer corpus loaded etc...
use concurrent_map::ConcurrentMap;
use std::collections::BTreeMap;

fn map_model<const FANOUT: usize>(
    items: &[u64],
) -> (ConcurrentMap<u64, u64, FANOUT>, BTreeMap<u64, u64>) {
    let map = concurrent_map::ConcurrentMap::<_, _, FANOUT>::default();
    let mut model = std::collections::BTreeMap::new();

    for item in items {
        map.insert(*item, *item);
        model.insert(*item, *item);
    }

    (map, model)
}

fn prop_iter_matches<const FANOUT: usize>(
    map: &ConcurrentMap<u64, u64, FANOUT>,
    model: &BTreeMap<u64, u64>,
) {
    let expected = model.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
    let actual = map.iter().collect::<Vec<_>>();
    assert_eq!(expected, actual);
}

fn prop_rev_iter_matches<const FANOUT: usize>(
    map: &ConcurrentMap<u64, u64, FANOUT>,
    model: &BTreeMap<u64, u64>,
) {
    let expected = model
        .iter()
        .rev()
        .map(|(k, v)| (*k, *v))
        .collect::<Vec<_>>();
    let actual = map.iter().rev().collect::<Vec<_>>();
    assert_eq!(expected, actual);
}

fn prop_range_matches<const FANOUT: usize>(
    map: &ConcurrentMap<u64, u64, FANOUT>,
    model: &BTreeMap<u64, u64>,
    bounds: std::ops::Range<u64>,
) {
    let expected = model
        .range(bounds.clone())
        .map(|(k, _v)| *k)
        .collect::<Vec<_>>();
    let actual = map
        .range(bounds.clone())
        .map(|(k, _v)| k)
        .collect::<Vec<_>>();
    assert_eq!(expected, actual);
}

fn prop_rev_range_matches<const FANOUT: usize>(
    map: &ConcurrentMap<u64, u64, FANOUT>,
    model: &BTreeMap<u64, u64>,
    bounds: std::ops::Range<u64>,
) {
    let expected_rev = model
        .range(bounds.clone())
        .rev()
        .map(|(k, _v)| *k)
        .collect::<Vec<_>>();
    let actual_rev = map.range(bounds).rev().map(|(k, _v)| k).collect::<Vec<_>>();

    assert_eq!(expected_rev, actual_rev);
}

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
    let items = [95, 126, 2, 73, 0, 106, 54];

    let (map, model) = map_model::<4>(&items);

    let bounds = 81..124;

    prop_iter_matches(&map, &model);
    prop_rev_iter_matches(&map, &model);
    prop_range_matches(&map, &model, bounds.clone());
    prop_rev_range_matches(&map, &model, bounds.clone());
}

#[test]
fn test_02() {
    let items = [2365587456, 12989, 18446742974197923840, 1099511627775];

    let (map, model) = map_model::<4>(&items);

    prop_iter_matches(&map, &model);
    prop_rev_iter_matches(&map, &model);
}

#[test]
fn test_03() {
    let items = [
        838873789,
        49478023249920,
        5859553998519926784,
        0,
        11936128518274744320,
        165,
    ];

    let (map, model) = map_model::<4>(&items);

    prop_iter_matches(&map, &model);
    prop_rev_iter_matches(&map, &model);
}
