#![cfg_attr(
    test,
    deny(
        missing_docs,
        future_incompatible,
        nonstandard_style,
        rust_2018_idioms,
        missing_copy_implementations,
        trivial_casts,
        trivial_numeric_casts,
        unused_qualifications,
    )
)]
#![cfg_attr(test, deny(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::decimal_literal_representation,
    clippy::doc_markdown,
    // clippy::else_if_without_else,
    clippy::empty_enum,
    clippy::explicit_into_iter_loop,
    clippy::explicit_iter_loop,
    clippy::expl_impl_clone_on_copy,
    clippy::fallible_impl_from,
    clippy::filter_map_next,
    clippy::float_arithmetic,
    clippy::get_unwrap,
    clippy::if_not_else,
    clippy::indexing_slicing,
    clippy::inline_always,
    clippy::integer_arithmetic,
    clippy::invalid_upcast_comparisons,
    clippy::items_after_statements,
    clippy::manual_find_map,
    clippy::map_entry,
    clippy::map_flatten,
    clippy::match_like_matches_macro,
    clippy::match_same_arms,
    clippy::maybe_infinite_iter,
    clippy::mem_forget,
    // clippy::missing_docs_in_private_items,
    clippy::module_name_repetitions,
    clippy::multiple_inherent_impl,
    clippy::mut_mut,
    clippy::needless_borrow,
    clippy::needless_continue,
    clippy::needless_pass_by_value,
    clippy::non_ascii_literal,
    clippy::path_buf_push_overwrite,
    // clippy::print_stdout,
    clippy::redundant_closure_for_method_calls,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::single_match_else,
    clippy::string_add,
    clippy::string_add_assign,
    clippy::type_repetition_in_bounds,
    clippy::unicode_not_nfc,
    clippy::unimplemented,
    clippy::unseparated_literal_suffix,
    clippy::used_underscore_binding,
    clippy::wildcard_dependencies,
))]
#![cfg_attr(
    test,
    warn(
        clippy::missing_const_for_fn,
        clippy::multiple_crate_versions,
        clippy::wildcard_enum_match_arm,
    )
)]

//! A lock-free B+ tree based on [sled](https://github.com/spacejam/sled)'s internal
//! index structure, but supporting richer Rust types as keys and values than raw bytes.
//!
//! This structure supports atomic compare and swap operations with the
//! [`ConcurrentMap::cas`] method.
//!
//! The [`ConcurrentMap`] allows users to tune the tree fan-out (`FANOUT`)
//! and the underlying memory reclamation granularity (`LOCAL_GC_BUFFER_SIZE`)
//! for achieving desired performance properties. The defaults are pretty good
//! for most use cases but if you want to squeeze every bit of performance out
//! for your particular workload, tweaking them based on realistic measurements
//! may be beneficial. See the [`ConcurrentMap`] docs for more details.
//!
//! If you want to use a custom key type, you must
//! implement the [`Minimum`] trait,
//! allowing the left-most side of the tree to be
//! created before inserting any data. If you wish
//! to perform scans in reverse lexicographical order,
//! you may instead implement [`Maximum`] for your key
//! type and use [`std::cmp::Reverse`].
//!
//! This is an ordered data structure, and supports very high throughput iteration over
//! lexicographically sorted ranges of values. If you are looking for simple point operation
//! performance, you may find a better option among one of the many concurrent
//! hashmap implementations that are floating around.

#[cfg(not(feature = "fault_injection"))]
#[inline]
const fn debug_delay() -> bool {
    false
}

/// This function is useful for inducing random jitter into
/// our atomic operations, shaking out more possible
/// interleavings quickly. It gets fully eliminated by the
/// compiler in non-test code.
#[cfg(feature = "fault_injection")]
fn debug_delay() -> bool {
    use std::thread;
    use std::time::Duration;

    use rand::{thread_rng, Rng};

    let mut rng = thread_rng();

    match rng.gen_range(0..100) {
        0..=98 => {
            //thread::yield_now();
            false
        }
        _ => {
            //thread::sleep(Duration::from_millis(1));
            true
        }
    }
}

use stack_map::StackMap;

use std::borrow::Borrow;
use std::fmt;
use std::ops::{Bound, Deref};
use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicPtr, AtomicUsize, Ordering},
    Arc,
};

#[cfg(feature = "timing")]
use std::time::{Duration, Instant};

use ebr::{Ebr, Guard};

// NB this must always be 1
const MERGE_SIZE: usize = 1;

#[derive(Debug)]
enum Deferred<
    K: 'static + Clone + Minimum + Send + Sync + Ord,
    V: 'static + Clone + Send + Sync,
    const FANOUT: usize,
> {
    Node(Box<Node<K, V, FANOUT>>),
    BoxedAtomicPtr(BoxedAtomicPtr<K, V, FANOUT>),
}

impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > Drop for Deferred<K, V, FANOUT>
{
    fn drop(&mut self) {
        if let Deferred::BoxedAtomicPtr(id) = self {
            assert!(!id.0.is_null());
            let reclaimed: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                unsafe { Box::from_raw(id.0 as *mut _) };
            drop(reclaimed);
        }
    }
}

#[derive(Debug, Clone, Eq)]
struct BoxedAtomicPtr<
    K: 'static + Clone + Minimum + Send + Sync + Ord,
    V: 'static + Clone + Send + Sync,
    const FANOUT: usize,
>(*const AtomicPtr<Node<K, V, FANOUT>>);

impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > Copy for BoxedAtomicPtr<K, V, FANOUT>
{
}

impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > PartialEq for BoxedAtomicPtr<K, V, FANOUT>
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

unsafe impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > Send for BoxedAtomicPtr<K, V, FANOUT>
{
}

unsafe impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > Sync for BoxedAtomicPtr<K, V, FANOUT>
{
}

impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > Deref for BoxedAtomicPtr<K, V, FANOUT>
{
    type Target = AtomicPtr<Node<K, V, FANOUT>>;

    fn deref(&self) -> &AtomicPtr<Node<K, V, FANOUT>> {
        unsafe { &*self.0 }
    }
}

impl<
        K: 'static + Clone + Minimum + Send + Sync + Ord,
        V: 'static + Clone + Send + Sync,
        const FANOUT: usize,
    > BoxedAtomicPtr<K, V, FANOUT>
{
    fn new(node: Box<Node<K, V, FANOUT>>) -> BoxedAtomicPtr<K, V, FANOUT> {
        let pointee_ptr = Box::into_raw(node);
        let pointer_ptr = Box::into_raw(Box::new(AtomicPtr::new(pointee_ptr)));
        BoxedAtomicPtr(pointer_ptr)
    }

    fn node_view<const LOCAL_GC_BUFFER_SIZE: usize>(
        &self,
        _guard: &mut Guard<'_, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    ) -> Option<NodeView<K, V, FANOUT>> {
        let ptr = NonNull::new(self.load(Ordering::Acquire))?;

        Some(NodeView { ptr, id: *self })
    }
}

/// Error type for the [`ConcurrentMap::cas`] operation.
#[derive(Debug, PartialEq, Eq)]
pub struct CasFailure<V> {
    /// The current actual value that failed the comparison
    pub actual: Option<V>,
    /// The value that was proposed as a new value, which could
    /// not be installed due to the comparison failure.
    pub returned_new_value: Option<V>,
}

#[derive(Debug)]
struct NodeView<K, V, const FANOUT: usize>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    ptr: NonNull<Node<K, V, FANOUT>>,
    id: BoxedAtomicPtr<K, V, FANOUT>,
}

impl<K, V, const FANOUT: usize> NodeView<K, V, FANOUT>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    /// Try to replace. If the node has been deleted since we got our view,
    /// an Err(None) is returned.
    fn cas<const LOCAL_GC_BUFFER_SIZE: usize>(
        &self,
        replacement: Box<Node<K, V, FANOUT>>,
        guard: &mut Guard<'_, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    ) -> Result<NodeView<K, V, FANOUT>, Option<NodeView<K, V, FANOUT>>> {
        assert!(
            !(replacement.hi.is_some() ^ replacement.next.is_some()),
            "hi and next must both either be None or Some"
        );

        if debug_delay() {
            return Err(Some(NodeView {
                ptr: self.ptr,
                id: self.id,
            }));
        }

        let replacement_ptr = Box::into_raw(replacement);
        let res = self.id.compare_exchange(
            self.ptr.as_ptr(),
            replacement_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        match res {
            Ok(_) => {
                let replaced: Box<Node<K, V, FANOUT>> = unsafe { Box::from_raw(self.ptr.as_ptr()) };
                guard.defer_drop(Deferred::Node(replaced));
                Ok(NodeView {
                    ptr: NonNull::new(replacement_ptr).unwrap(),
                    id: self.id,
                })
            }
            Err(actual) => {
                let failed_value: Box<Node<K, V, FANOUT>> =
                    unsafe { Box::from_raw(replacement_ptr) };
                drop(failed_value);

                if actual.is_null() {
                    Err(None)
                } else {
                    Err(Some(NodeView {
                        ptr: NonNull::new(actual).unwrap(),
                        id: self.id,
                    }))
                }
            }
        }
    }
}

impl<K, V, const FANOUT: usize> Deref for NodeView<K, V, FANOUT>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    type Target = Node<K, V, FANOUT>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

/// Trait for types for which a minimum possible value exists.
///
/// This trait must be implemented for any `K` key type in the [`ConcurrentMap`].
pub trait Minimum: Ord {
    /// The returned value must be less than or equal
    /// to all possible values for this type.
    const MIN: Self;
}

/// Trait for types for which a maximum possible value exists.
///
/// This exists primarily to play nicely with [`std::cmp::Reverse`] keys
/// for achieving high performance reverse iteration.
pub trait Maximum: Ord {
    /// The returned value must be greater than or equal
    /// to all possible values for this type.
    const MAX: Self;
}

impl Minimum for () {
    const MIN: Self = ();
}

impl<T: Maximum> Minimum for std::cmp::Reverse<T> {
    const MIN: Self = std::cmp::Reverse(T::MAX);
}

macro_rules! impl_integer {
    ($($t:ty),+) => {
        $(
            impl Minimum for $t {
                const MIN: Self = <$t>::MIN;
            }

            impl Maximum for $t {
                const MAX: Self = <$t>::MAX;
            }
        )*
    }
}

impl_integer!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128);

impl<T: Ord> Minimum for Vec<T> {
    const MIN: Self = Vec::new();
}

impl<T: Ord> Minimum for &[T] {
    const MIN: Self = &[];
}

impl<T: Minimum, const LEN: usize> Minimum for [T; LEN] {
    const MIN: Self = [T::MIN; LEN];
}

impl Minimum for String {
    const MIN: Self = String::new();
}

impl Minimum for &str {
    const MIN: Self = "";
}

/// A lock-free B+ tree.
///
/// Note that this structure is `Send` but NOT `Sync`,
/// despite being a lock-free tree. This is because the
/// inner reclamation system, provided by the `ebr` crate
/// completely avoids atomic operations in its hot path
/// for efficiency. If you want to share [`ConcurrentMap`]
/// between threads, simply clone it, and this will set up
/// a new efficient thread-local memory reclamation state.
///
/// If you want to use a custom key type, you must
/// implement the [`Minimum`] trait,
/// allowing the left-most side of the tree to be
/// created before inserting any data. If you wish
/// to perform scans in reverse lexicographical order,
/// you may instead implement [`Maximum`] for your key
/// type and use [`std::cmp::Reverse`].
///
/// The `FANOUT` const generic must be greater than 3.
/// This const generic controls how large the fixed-size
/// array for either child pointers (for index nodes) or
/// values (for leaf nodes) will be. A higher value may
/// make reads and scans faster, but writes will be slower
/// because each modification performs a read-copy-update
/// of the full node. In some cases, it may be preferable
/// to wrap complex values in an `Arc` to avoid higher
/// copy costs.
///
/// The `LOCAL_GC_BUFFER_SIZE` const generic must be greater than 0.
/// This controls the epoch-based reclamation granularity.
/// Garbage is placed into fixed-size arrays, and garbage collection
/// only happens after this array fills up and a final timestamp is
/// assigned to it. Lower values will cause replaced values to be dropped
/// more quickly, but the efficiency will be lower. Values that are
/// extremely high may cause undesirable memory usage because it will
/// take more time to fill up each thread-local garbage segment.
///
/// # Examples
///
/// ```
/// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
///
/// // insert and remove atomically returns the last value, if it was set,
/// // similarly to a BTreeMap
/// assert_eq!(map.insert(1, 10), None);
/// assert_eq!(map.insert(1, 11), Some(10));
/// assert_eq!(map.remove(&1), Some(11));
///
/// // get also functions similarly to BTreeMap, except it
/// // returns a cloned version of the value rather than a
/// // reference to it, so that no locks need to be maintained.
/// // For this reason, it can be a good idea to use types that
/// // are cheap to clone for values, which can be easily handled
/// // with `Arc` etc...
/// assert_eq!(map.insert(1, 12), None);
/// assert_eq!(map.get(&1), Some(12));
///
/// // compare and swap from value 12 to value 20
/// map.cas(1, Some(&12_usize), Some(20)).unwrap();
///
/// assert_eq!(map.get(&1).unwrap(), 20);
///
/// // there are a lot of methods that are not covered
/// // here - check out the docs!
/// ```
#[derive(Clone)]
pub struct ConcurrentMap<K, V, const FANOUT: usize = 64, const LOCAL_GC_BUFFER_SIZE: usize = 128>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    // epoch-based reclamation
    ebr: Ebr<Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    // the tree structure, separate from the other
    // types so that we can mix mutable references
    // to ebr with immutable references to other
    // things.
    inner: Arc<Inner<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>>,
    // an eventually consistent, lagging count of the
    // number of items in this structure.
    len: Arc<AtomicUsize>,
}

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> fmt::Debug
    for ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + fmt::Debug + Clone + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ConcurrentMap ")?;
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> Default
    for ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    fn default() -> ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE> {
        assert!(FANOUT > 3, "ConcurrentMap FANOUT must be greater than 3");
        assert!(
            LOCAL_GC_BUFFER_SIZE > 0,
            "LOCAL_GC_BUFFER_SIZE must be greater than 0"
        );
        let mut root_node = Node::<K, V, FANOUT>::new_root();
        let root_node_lo = root_node.lo.clone();
        let leaf_node = Node::<K, V, FANOUT>::new_leaf(root_node.lo.clone());
        let leaf = BoxedAtomicPtr::new(leaf_node);
        root_node.index_mut().insert(root_node_lo, leaf);

        let root = BoxedAtomicPtr::new(root_node);

        let inner = Arc::new(Inner {
            root,
            #[cfg(feature = "timing")]
            slowest_op: u64::MIN.into(),
            #[cfg(feature = "timing")]
            fastest_op: u64::MAX.into(),
        });

        ConcurrentMap {
            ebr: Ebr::default(),
            inner,
            len: Arc::new(0.into()),
        }
    }
}

struct Inner<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    root: BoxedAtomicPtr<K, V, FANOUT>,
    #[cfg(feature = "timing")]
    slowest_op: AtomicU64,
    #[cfg(feature = "timing")]
    fastest_op: AtomicU64,
}

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> Drop
    for Inner<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    fn drop(&mut self) {
        #[cfg(feature = "timing")]
        self.print_timing();

        let ebr = Ebr::default();
        let mut guard = ebr.pin();

        let mut cursor: NodeView<K, V, FANOUT> = self.root(&mut guard);

        let mut lhs_chain: Vec<BoxedAtomicPtr<K, V, FANOUT>> = vec![];

        loop {
            lhs_chain.push(cursor.id);
            if cursor.is_leaf() {
                break;
            }
            let child_ptr: BoxedAtomicPtr<K, V, FANOUT> = cursor.index().get_index(0).unwrap().1;

            cursor = child_ptr.node_view(&mut guard).unwrap();
        }

        let mut layer = 0;
        for lhs_ptr in lhs_chain {
            layer += 1;

            let mut min_fill_physical: f64 = 1.0;
            let mut max_fill_physical: f64 = 0.0;
            let mut fill_sum_physical: f64 = 0.0;

            let mut min_fill_logical: f64 = 1.0;
            let mut max_fill_logical: f64 = 0.0;
            let mut fill_sum_logical: f64 = 0.0;
            let mut nodes_counted: usize = 0;

            let mut next_opt: Option<BoxedAtomicPtr<K, V, FANOUT>> = Some(lhs_ptr);
            while let Some(next) = next_opt {
                assert!(!next.0.is_null());
                let sibling_cursor = next.node_view(&mut guard).unwrap();

                let fill_phy = ((std::mem::size_of::<K>() + std::mem::size_of::<V>())
                    * sibling_cursor.len()) as f64
                    / std::mem::size_of::<Node<K, V, FANOUT>>() as f64;
                min_fill_physical = min_fill_physical.min(fill_phy);
                max_fill_physical = max_fill_physical.max(fill_phy);
                fill_sum_physical += fill_phy;

                let fill_log = sibling_cursor.len() as f64 / FANOUT as f64;
                min_fill_logical = min_fill_logical.min(fill_log);
                max_fill_logical = max_fill_logical.max(fill_log);
                fill_sum_logical += fill_log;
                nodes_counted += 1;

                next_opt = sibling_cursor.next;
                let node_box = unsafe { Box::from_raw(sibling_cursor.ptr.as_ptr()) };
                drop(node_box);

                let reclaimed_ptr: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                    unsafe { Box::from_raw(next.0 as *mut _) };
                drop(reclaimed_ptr);
            }

            if cfg!(feature = "print_utilization_on_drop") {
                println!("layer {layer} count {nodes_counted}");
                println!(
                    "logical: min: {min_fill_logical} max: {max_fill_logical} avg: {}",
                    fill_sum_logical / nodes_counted as f64
                );
                println!(
                    "physical: min: {min_fill_physical} max: {max_fill_physical} avg: {}",
                    fill_sum_physical / nodes_counted as f64
                );
            }
        }
    }
}

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize>
    ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    /// Atomically get a value out of the map that is associated with this key.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    ///
    /// let actual = map.get(&0);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get(&1);
    /// let expected = Some(1);
    /// assert_eq!(expected, actual);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut guard = self.ebr.pin();

        let leaf = self.inner.leaf_for_key(LeafSearch::Eq(key), &mut guard);

        leaf.get(key)
    }

    /// Atomically get a key and value out of the map that is associated with the key that
    /// is lexicographically less than the provided key.
    ///
    /// This will always return `None` if the key passed to `get_lt` == `K::MIN`.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    ///
    /// let actual = map.get_lt(&0);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_lt(&1);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_lt(&2);
    /// let expected = Some((1, 1));
    /// assert_eq!(expected, actual);
    /// ```
    pub fn get_lt<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord + PartialEq,
    {
        if key == K::MIN.borrow() {
            return None;
        }

        let start = Bound::Unbounded;
        let end = Bound::Excluded(key);
        self.range((start, end)).next_back()
    }

    /// Atomically get a key and value out of the map that is associated with the key that
    /// is lexicographically less than or equal to the provided key.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    ///
    /// let actual = map.get_lte(&0);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_lte(&1);
    /// let expected = Some((1, 1));
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_lte(&2);
    /// let expected = Some((1, 1));
    /// assert_eq!(expected, actual);
    /// ```
    pub fn get_lte<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord + PartialEq,
    {
        let start = Bound::Unbounded;
        let end = Bound::Included(key);
        self.range((start, end)).next_back()
    }

    /// Atomically get a key and value out of the map that is associated with the key
    /// that is lexicographically greater than the provided key.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    ///
    /// let actual = map.get_gt(&0);
    /// let expected = Some((1, 1));
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_gt(&1);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_gt(&2);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    /// ```
    pub fn get_gt<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord + PartialEq,
    {
        self.range((Bound::Excluded(key), Bound::Unbounded)).next()
    }

    /// Atomically get a key and value out of the map that is associated with the key
    /// that is lexicographically greater than or equal to the provided key.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    ///
    /// let actual = map.get_gte(&0);
    /// let expected = Some((1, 1));
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_gte(&1);
    /// let expected = Some((1, 1));
    /// assert_eq!(expected, actual);
    ///
    /// let actual = map.get_gte(&2);
    /// let expected = None;
    /// assert_eq!(expected, actual);
    /// ```
    pub fn get_gte<Q>(&self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Ord + PartialEq,
    {
        self.range((Bound::Included(key.borrow()), Bound::Unbounded))
            .next()
    }

    /// Get the minimum item stored in this structure.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    /// map.insert(2, 2);
    /// map.insert(3, 3);
    ///
    /// let actual = map.first();
    /// let expected = Some((1, 1));
    /// assert_eq!(actual, expected);
    /// ```
    pub fn first(&self) -> Option<(K, V)> {
        self.iter().next()
    }

    /// Atomically remove the minimum item stored in this structure.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    /// map.insert(2, 2);
    /// map.insert(3, 3);
    ///
    /// let actual = map.pop_first();
    /// let expected = Some((1, 1));
    /// assert_eq!(actual, expected);
    ///
    /// assert_eq!(map.get(&1), None);
    /// ```
    pub fn pop_first(&self) -> Option<(K, V)>
    where
        V: PartialEq,
    {
        loop {
            let (k, v) = self.first()?;
            if self.cas(k.clone(), Some(&v), None).is_ok() {
                return Some((k, v));
            }
        }
    }

    /// Get the maximum item stored in this structure.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    /// map.insert(2, 2);
    /// map.insert(3, 3);
    ///
    /// let actual = map.last();
    /// let expected = Some((3, 3));
    /// assert_eq!(actual, expected);
    /// ```
    pub fn last(&self) -> Option<(K, V)> {
        self.iter().next_back()
    }

    /// Atomically remove the maximum item stored in this structure.
    ///
    /// # Examples
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// map.insert(1, 1);
    /// map.insert(2, 2);
    /// map.insert(3, 3);
    ///
    /// let actual = map.pop_last();
    /// let expected = Some((3, 3));
    /// assert_eq!(actual, expected);
    ///
    /// assert_eq!(map.get(&3), None);
    /// ```
    pub fn pop_last(&self) -> Option<(K, V)>
    where
        V: PartialEq,
    {
        loop {
            let (k, v) = self.last()?;
            if self.cas(k.clone(), Some(&v), None).is_ok() {
                return Some((k, v));
            }
        }
    }

    /// Atomically insert a key-value pair into the map, returning the previous value associated with this key if one existed.
    ///
    /// # Examples
    ///
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// assert_eq!(map.insert(1, 1), None);
    /// assert_eq!(map.insert(1, 1), Some(1));
    /// ```
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        loop {
            let mut guard = self.ebr.pin();
            let leaf = self.inner.leaf_for_key(LeafSearch::Eq(&key), &mut guard);
            let mut leaf_clone: Box<Node<K, V, FANOUT>> = Box::new((*leaf).clone());
            assert!(
                leaf_clone.len() < (FANOUT - MERGE_SIZE),
                "bad leaf: should split: {}",
                leaf_clone.should_split(),
            );
            let ret = leaf_clone.insert(key.clone(), value.clone());

            let rhs_ptr_opt = if leaf_clone.should_split() {
                Some(leaf_clone.split())
            } else {
                None
            };

            let install_attempt = leaf.cas(leaf_clone, &mut guard);

            if install_attempt.is_ok() {
                if ret.is_none() {
                    self.len.fetch_add(1, Ordering::Relaxed);
                }
                return ret;
            } else if let Some(new_ptr) = rhs_ptr_opt {
                // clear dangling BoxedAtomicPtr (cas freed the pointee already)
                let reclaimed_ptr: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                    unsafe { Box::from_raw(new_ptr.0 as *mut _) };

                let _dropping_reclaimed_rhs: Box<Node<K, V, FANOUT>> =
                    unsafe { Box::from_raw(reclaimed_ptr.load(Ordering::Acquire)) };
            }
        }
    }

    /// Atomically remove the value associated with this key from the map, returning the previous value if one existed.
    ///
    /// # Examples
    ///
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// assert_eq!(map.remove(&1), None);
    /// assert_eq!(map.insert(1, 1), None);
    /// assert_eq!(map.remove(&1), Some(1));
    /// ```
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        loop {
            let mut guard = self.ebr.pin();
            let leaf = self.inner.leaf_for_key(LeafSearch::Eq(key), &mut guard);
            let mut leaf_clone: Box<Node<K, V, FANOUT>> = Box::new((*leaf).clone());
            let ret = leaf_clone.remove(key);
            let install_attempt = leaf.cas(leaf_clone, &mut guard);
            if install_attempt.is_ok() {
                if ret.is_some() {
                    self.len.fetch_sub(1, Ordering::Relaxed);
                }
                return ret;
            }
        }
    }

    /// Atomically compare and swap the value associated with this key from the old value to the
    /// new one. An old value of `None` means "only create this value if it does not already
    /// exist". A new value of `None` means "delete this value, if it matches the provided old value".
    /// If successful, returns the old value if it existed. If unsuccessful, returns the proposed
    /// new value.
    ///
    /// # Examples
    ///
    /// ```
    /// let map = concurrent_map::ConcurrentMap::<usize, usize>::default();
    ///
    /// // key 1 does not yet exist
    /// assert_eq!(map.get(&1), None);
    ///
    /// // uniquely create value 10
    /// map.cas(1, None, Some(10)).unwrap();
    ///
    /// assert_eq!(map.get(&1).unwrap(), 10);
    ///
    /// // compare and swap from value 10 to value 20
    /// map.cas(1, Some(&10_usize), Some(20)).unwrap();
    ///
    /// assert_eq!(map.get(&1).unwrap(), 20);
    ///
    /// // if we guess the wrong current value, a CasFailure is returned
    /// // which will tell us what the actual current value is (which we
    /// // failed to provide) and it will give us back our proposed new
    /// // value.
    /// let cas_result = map.cas(1, Some(&999999_usize), Some(30));
    ///
    /// let expected_cas_failure = Err(concurrent_map::CasFailure {
    ///     actual: Some(20),
    ///     returned_new_value: Some(30),
    /// });
    ///
    /// assert_eq!(cas_result, expected_cas_failure);
    ///
    /// // conditionally delete
    /// map.cas(1, Some(&20_usize), None).unwrap();
    ///
    /// assert_eq!(map.get(&1), None);
    /// ```
    pub fn cas<VRef>(
        &self,
        key: K,
        old: Option<&VRef>,
        new: Option<V>,
    ) -> Result<Option<V>, CasFailure<V>>
    where
        V: Borrow<VRef>,
        VRef: PartialEq + ?Sized,
    {
        loop {
            let mut guard = self.ebr.pin();
            let leaf = self.inner.leaf_for_key(LeafSearch::Eq(&key), &mut guard);
            let mut leaf_clone: Box<Node<K, V, FANOUT>> = Box::new((*leaf).clone());
            let ret = leaf_clone.cas(key.clone(), old, new.clone());

            let rhs_ptr_opt = if leaf_clone.should_split() {
                Some(leaf_clone.split())
            } else {
                None
            };

            let install_attempt = leaf.cas(leaf_clone, &mut guard);

            if install_attempt.is_ok() {
                if matches!(ret, Ok(Some(_))) && new.is_none() {
                    self.len.fetch_sub(1, Ordering::Relaxed);
                } else if matches!(ret, Ok(None)) && new.is_some() {
                    self.len.fetch_add(1, Ordering::Relaxed);
                }
                return ret;
            } else if let Some(new_ptr) = rhs_ptr_opt {
                // clear dangling BoxedAtomicPtr (cas freed pointee already)
                let reclaimed_ptr: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                    unsafe { Box::from_raw(new_ptr.0 as *mut _) };

                let _dropping_reclaimed_rhs: Box<Node<K, V, FANOUT>> =
                    unsafe { Box::from_raw(reclaimed_ptr.load(Ordering::Acquire)) };
            }
        }
    }

    /// A **lagging**, eventually-consistent length count. This is NOT atomically
    /// updated with [`insert`] / [`remove`] / [`cas`], but is updated after those
    /// operations complete their atomic modifications to the shared map.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// A **lagging**, eventually-consistent check for emptiness, based on the correspondingly
    /// non-atomic `len` method.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over the map.
    ///
    /// This is not an atomic snapshot, and it caches B+tree leaf
    /// nodes as it iterates through them to achieve high throughput.
    /// As a result, the following behaviors are possible:
    ///
    /// * may (or may not!) return values that were concurrently added to the map after the
    ///   iterator was created
    /// * may (or may not!) return items that were concurrently deleted from the map after
    ///   the iterator was created
    /// * If a key's value is changed from one value to another one after this iterator
    ///   is created, this iterator might return the old or the new value.
    ///
    /// But, you can be certain that any key that existed prior to the creation of this
    /// iterator, and was not changed during iteration, will be observed as expected.
    pub fn iter(&self) -> Iter<'_, K, V, FANOUT, LOCAL_GC_BUFFER_SIZE> {
        let mut guard = self.ebr.pin();

        let current = self.inner.leaf_for_key(LeafSearch::Eq(&K::MIN), &mut guard);
        let current_back = self.inner.leaf_for_key(LeafSearch::Max, &mut guard);
        let next_index_from_back = 0;

        Iter {
            guard,
            inner: &self.inner,
            current,
            range: std::ops::RangeFull,
            next_index: 0,
            current_back,
            next_index_from_back,
            q: std::marker::PhantomData,
        }
    }

    /// Iterate over a range of the map.
    ///
    /// This is not an atomic snapshot, and it caches B+tree leaf
    /// nodes as it iterates through them to achieve high throughput.
    /// As a result, the following behaviors are possible:
    ///
    /// * may (or may not!) return values that were concurrently added to the map after the
    ///   iterator was created
    /// * may (or may not!) return items that were concurrently deleted from the map after
    ///   the iterator was created
    /// * If a key's value is changed from one value to another one after this iterator
    ///   is created, this iterator might return the old or the new value.
    ///
    /// But, you can be certain that any key that existed prior to the creation of this
    /// iterator, and was not changed during iteration, will be observed as expected.
    ///
    /// # Panics
    ///
    /// This will panic if the provided range's end_bound() == Bound::Excluded(K::MIN).
    pub fn range<Q, R>(&self, range: R) -> Iter<'_, K, V, FANOUT, LOCAL_GC_BUFFER_SIZE, R, Q>
    where
        R: std::ops::RangeBounds<Q>,
        K: Borrow<Q>,
        Q: ?Sized + Ord + PartialEq,
    {
        let mut guard = self.ebr.pin();

        let kmin = &K::MIN;
        let min = kmin.borrow();
        let start = match range.start_bound() {
            Bound::Unbounded => min,
            Bound::Included(k) | Bound::Excluded(k) => k,
        };

        let end = match range.end_bound() {
            Bound::Unbounded => LeafSearch::Max,
            Bound::Included(k) => LeafSearch::Eq(k),
            Bound::Excluded(k) => {
                assert!(k != K::MIN.borrow());
                LeafSearch::Lt(k)
            }
        };

        let current = self.inner.leaf_for_key(LeafSearch::Eq(start), &mut guard);
        let current_back = self.inner.leaf_for_key(end, &mut guard);

        Iter {
            guard,
            inner: &self.inner,
            range,
            current,
            current_back,
            next_index: 0,
            next_index_from_back: 0,
            q: std::marker::PhantomData,
        }
    }
}

/// An iterator over a [`ConcurrentMap`]. Note that this is
/// not an atomic snapshot of the overall shared state, but
/// it will contain any data that existed before the iterator
/// was created.
///
/// Note that this iterator contains an epoch-based reclamation
/// guard, and the overall concurrent structure will be unable
/// to free any memory until this iterator drops again.
///
/// There are a lot of generics on this struct. Most of them directly
/// correspond to the generics of the [`ConcurrentMap`] itself. But
/// there are two that don't:
///
/// * `R` is The type of the range that is stored in the iterator
/// * `Q` is the type that exists INSIDE of `R`
///
/// So, if an `Iter` is created from:
///
/// ```
/// let map = concurrent_map::ConcurrentMap::default::<usize, usize>();
/// let start = std::ops::Bound::Excluded(0_usize);
/// let end = std::ops::Bound::Included(5_usize);
/// let iter = map.range((start, end));
/// ```
///
/// then the type of `R` is `(std::ops::Bound, std::ops::Bound)`
/// (a 2-tuple of `std::ops::Bound`), and the type of `Q` is `usize`.
pub struct Iter<
    'a,
    K,
    V,
    const FANOUT: usize,
    const LOCAL_GC_BUFFER_SIZE: usize,
    R = std::ops::RangeFull,
    Q = K,
> where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
    R: std::ops::RangeBounds<Q>,
    K: Borrow<Q>,
    Q: ?Sized,
{
    inner: &'a Inner<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>,
    guard: Guard<'a, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    range: R,
    current: NodeView<K, V, FANOUT>,
    next_index: usize,
    current_back: NodeView<K, V, FANOUT>,
    next_index_from_back: usize,
    q: std::marker::PhantomData<&'a Q>,
}

impl<'a, K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize, R, Q> Iterator
    for Iter<'a, K, V, FANOUT, LOCAL_GC_BUFFER_SIZE, R, Q>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
    R: std::ops::RangeBounds<Q>,
    K: Borrow<Q>,
    Q: ?Sized + PartialEq + Ord,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((k, v)) = self.current.leaf().get_index(self.next_index) {
                // iterate over current cached b+ tree leaf node
                self.next_index += 1;
                if !self.range.contains(k.borrow()) {
                    // we might hit this on the first iteration
                    continue;
                }
                return Some((k.clone(), v.clone()));
            } else if let Some(next_ptr) = self.current.next {
                if !self
                    .range
                    .contains(self.current.hi.as_ref().unwrap().borrow())
                {
                    // we have reached the end of our range
                    return None;
                }
                if let Some(next_view) = next_ptr.node_view(&mut self.guard) {
                    // we were able to take the fast path by following the sibling pointer
                    self.current = next_view;
                    self.next_index = 0;
                } else if let Some(ref hi) = self.current.hi {
                    // we have to take the slow path by traversing the
                    // map due to a concurrent merge that deleted the
                    // right sibling. we are protected from a use after
                    // free of the ID itself due to holding an ebr Guard
                    // on the Iter struct, holding a barrier against re-use.
                    self.current = self
                        .inner
                        .leaf_for_key(LeafSearch::Eq(hi.borrow()), &mut self.guard);
                    self.next_index = 0;
                } else {
                    panic!("somehow hit a node that has a next but not a hi key");
                }
            } else {
                // end of the collection
                return None;
            }
        }
    }
}

impl<'a, K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize, R, Q> DoubleEndedIterator
    for Iter<'a, K, V, FANOUT, LOCAL_GC_BUFFER_SIZE, R, Q>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
    R: std::ops::RangeBounds<Q>,
    K: Borrow<Q>,
    Q: ?Sized + PartialEq + Ord,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_index_from_back >= self.current_back.leaf().len() {
                if !self.range.contains(&self.current_back.lo.borrow())
                    || self.current_back.lo == K::MIN
                {
                    // finished
                    return None;
                }

                let next_current_back = self.inner.leaf_for_key(
                    LeafSearch::Lt(&self.current_back.lo.borrow()),
                    &mut self.guard,
                );
                assert!(next_current_back.lo != self.current_back.lo);
                self.current_back = next_current_back;

                self.next_index_from_back = 0;

                if self.current_back.leaf().is_empty() {
                    continue;
                }
            }

            let offset_to_return = self.current_back.leaf().len() - (1 + self.next_index_from_back);
            let (k, v) = self
                .current_back
                .leaf()
                .get_index(offset_to_return)
                .unwrap();

            self.next_index_from_back += 1;
            if !self.range.contains(k.borrow()) {
                continue;
            } else {
                return Some((k.clone(), v.clone()));
            }
        }
    }
}

enum LeafSearch<K> {
    // For finding a leaf that would contain this key, if present.
    Eq(K),
    // For finding the direct left sibling of a node during reverse
    // iteration. The actual semantic is to find a leaf that has a lo key
    // that is less than K and a hi key that is >= K
    Lt(K),
    Max,
}

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize>
    Inner<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    fn root(
        &self,
        _guard: &mut Guard<'_, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    ) -> NodeView<K, V, FANOUT> {
        loop {
            if let Some(ptr) = NonNull::new(self.root.load(Ordering::Acquire)) {
                return NodeView { ptr, id: self.root };
            }
        }
    }

    // lock-free merging:
    // 1. try to mark the parent's merging_child
    //  a. must not be the left-most child
    //  b. if unsuccessful, give up
    // 2. mark the child as merging
    // 3. find the left sibling
    // 4. cas the left sibling to eat the right sibling
    //  a. loop until successful
    //  b. go right if the left-most child split and no-longer points to merging child
    //  c. split the new larger left sibling if it is at the split threshold
    // 5. cas the parent to remove the merged child
    // 6. remove the child's pointer in the page table
    // 7. defer the reclamation of the BoxedAtomicPtr
    // 8. defer putting the child's ID into the free stack
    //
    // merge threshold must be >= 1, because otherwise index nodes with 1 empty
    // child will never be compactible.

    fn install_parent_merge<'a>(
        &'a self,
        parent: &NodeView<K, V, FANOUT>,
        child: &NodeView<K, V, FANOUT>,
        guard: &mut Guard<'a, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    ) -> Result<NodeView<K, V, FANOUT>, ()> {
        // 1. try to mark the parent's merging_child
        //  a. must not be the left-most child
        //  b. if unsuccessful, give up
        let is_leftmost_child = parent.index().get_index(0).unwrap().0 == child.lo;

        if is_leftmost_child {
            return Err(());
        }

        if !parent.index().contains_key(&child.lo) {
            // can't install a parent merge if the child is unknown to the parent.
            return Err(());
        }

        if parent.merging_child.is_some() {
            // there is already a merge in-progress for a different child.
            return Err(());
        }

        let mut parent_clone: Box<Node<K, V, FANOUT>> = Box::new((*parent).clone());
        parent_clone.merging_child = Some(child.id);
        parent.cas(parent_clone, guard).map_err(|_| ())
    }

    fn merge_child<'a>(
        &'a self,
        parent: &mut NodeView<K, V, FANOUT>,
        child: &mut NodeView<K, V, FANOUT>,
        guard: &mut Guard<'a, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    ) {
        // 2. mark child as merging
        while !child.is_merging {
            let mut child_clone: Box<Node<K, V, FANOUT>> = Box::new((*child).clone());
            child_clone.is_merging = true;
            *child = match child.cas(child_clone, guard) {
                Ok(new_child) | Err(Some(new_child)) => new_child,
                Err(None) => {
                    // child already removed
                    return;
                }
            };
        }

        // 3. find the left sibling
        let first_left_sibling_guess = parent
            .index()
            .iter()
            .filter(|(k, _v)| (..&child.lo).contains(&k))
            .next_back()
            .unwrap()
            .1;

        let mut left_sibling = if let Some(view) = first_left_sibling_guess.node_view(guard) {
            view
        } else {
            // the merge already completed and this left sibling has already also been merged
            return;
        };

        loop {
            if left_sibling.next.is_none() {
                // the merge completed and the left sibling became the infinity node in the mean
                // time
                return;
            }

            if child.hi.is_some() && left_sibling.hi.is_some() && left_sibling.hi >= child.hi {
                // step 4 happened concurrently
                break;
            }

            let next = left_sibling.next.unwrap();
            if next != child.id {
                left_sibling = if let Some(view) = next.node_view(guard) {
                    view
                } else {
                    // the merge already completed and this left sibling has already also been merged
                    return;
                };
                continue;
            }

            // 4. cas the left sibling to eat the right sibling
            //  a. loop until successful
            //  b. go right if the left-most child split and no-longer points to merging child
            //  c. split the new larger left sibling if it is at the split threshold
            let mut left_sibling_clone: Box<Node<K, V, FANOUT>> = Box::new((*left_sibling).clone());
            left_sibling_clone.merge(child);

            let rhs_ptr_opt = if left_sibling_clone.should_split() {
                // we have to try to split the sibling, funny enough.
                // this is the consequence of using fixed-size arrays
                // for storing items with no flexibility.

                Some(left_sibling_clone.split())
            } else {
                None
            };

            let cas_result = left_sibling.cas(left_sibling_clone, guard);
            if let (Err(_), Some(rhs_ptr)) = (&cas_result, rhs_ptr_opt) {
                // We need to free the split right sibling that we installed
                let reclaimed_ptr: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                    unsafe { Box::from_raw(rhs_ptr.0 as *mut _) };

                let _dropping_reclaimed_rhs: Box<Node<K, V, FANOUT>> =
                    unsafe { Box::from_raw(reclaimed_ptr.load(Ordering::Acquire)) };
            }

            match cas_result {
                Ok(_) => {
                    break;
                }
                Err(Some(actual)) => left_sibling = actual,
                Err(None) => {
                    return;
                }
            }
        }

        // 5. cas the parent to remove the merged child
        while parent.merging_child == Some(child.id) {
            let mut parent_clone: Box<Node<K, V, FANOUT>> = Box::new((*parent).clone());

            assert!(parent_clone.merging_child.is_some());
            assert!(parent_clone.index().contains_key(&child.lo));

            parent_clone.merging_child = None;
            parent_clone.index_mut().remove(&child.lo).unwrap();

            let cas_result = parent.cas(parent_clone, guard);
            match cas_result {
                Ok(new_parent) | Err(Some(new_parent)) => *parent = new_parent,
                Err(None) => {
                    return;
                }
            }
        }

        // 6. remove the child's pointer in the page table
        if child
            .id
            .compare_exchange(
                child.ptr.as_ptr(),
                std::ptr::null_mut(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // only the thread that uninstalls this pointer gets to
            // mark resources for reuse.
            return;
        }

        // 7. defer the reclamation of the BoxedAtomicPtr
        guard.defer_drop(Deferred::BoxedAtomicPtr(child.id));

        // 8. defer the reclamation of the child node
        let replaced: Box<Node<K, V, FANOUT>> = unsafe { Box::from_raw(child.ptr.as_ptr()) };
        guard.defer_drop(Deferred::Node(replaced));
    }

    #[cfg(feature = "timing")]
    fn print_timing(&self) {
        println!(
            "min : {:?}",
            Duration::from_nanos(self.fastest_op.load(Ordering::Acquire))
        );
        println!(
            "max : {:?}",
            Duration::from_nanos(self.slowest_op.load(Ordering::Acquire))
        );
    }

    #[cfg(feature = "timing")]
    fn record_timing(&self, time: Duration) {
        let nanos = time.as_nanos() as u64;
        let min = self.fastest_op.load(Ordering::Relaxed);
        if nanos < min {
            self.fastest_op.fetch_min(nanos, Ordering::Relaxed);
        }

        let max = self.slowest_op.load(Ordering::Relaxed);
        if nanos > max {
            self.slowest_op.fetch_max(nanos, Ordering::Relaxed);
        }
    }

    fn leaf_for_key<'a, Q>(
        &'a self,
        search: LeafSearch<&Q>,
        guard: &mut Guard<'a, Deferred<K, V, FANOUT>, LOCAL_GC_BUFFER_SIZE>,
    ) -> NodeView<K, V, FANOUT>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut parent_cursor_opt: Option<NodeView<K, V, FANOUT>> = None;
        let mut cursor = self.root(guard);
        let mut root_cursor = NodeView {
            ptr: cursor.ptr,
            id: cursor.id,
        };

        macro_rules! reset {
            ($reason:expr) => {
                // println!("resetting because of {:?}", $reason);
                parent_cursor_opt = None;
                cursor = self.root(guard);
                root_cursor = NodeView {
                    ptr: cursor.ptr,
                    id: cursor.id,
                };
                continue;
            };
        }

        #[cfg(feature = "timing")]
        let before = Instant::now();

        loop {
            if let Some(merging_child_ptr) = cursor.merging_child {
                let mut child = if let Some(view) = merging_child_ptr.node_view(guard) {
                    view
                } else {
                    reset!("merging child of marked parent already freed");
                };
                self.merge_child(&mut cursor, &mut child, guard);
                reset!("cooperatively performed merge_child after detecting parent");
            }

            if cursor.is_merging {
                reset!("resetting after detected child merging without corresponding parent child_merge");
            }

            if cursor.should_merge() {
                if let Some(ref mut parent_cursor) = parent_cursor_opt {
                    let is_leftmost_child =
                        parent_cursor.index().get_index(0).unwrap().0 == cursor.lo;

                    if !is_leftmost_child {
                        if let Ok(new_parent) =
                            self.install_parent_merge(parent_cursor, &cursor, guard)
                        {
                            *parent_cursor = new_parent;
                        } else {
                            reset!("failed to install parent merge");
                        }

                        self.merge_child(parent_cursor, &mut cursor, guard);
                        reset!("completed merge_child");
                    }
                } else {
                    assert!(!cursor.is_leaf());
                }
            }

            match search {
                LeafSearch::Eq(k) | LeafSearch::Lt(k) => assert!(k >= cursor.lo.borrow()),
                LeafSearch::Max => {}
            }

            if let Some(hi) = &cursor.hi {
                let go_right = match search {
                    LeafSearch::Eq(k) => k >= hi.borrow(),
                    // Lt looks for a node with lo < K, hi >= K
                    LeafSearch::Lt(k) => k > hi.borrow(),
                    LeafSearch::Max => true,
                };
                if go_right {
                    // go right to the tree sibling
                    let next = cursor.next.unwrap();
                    let rhs = if let Some(view) = next.node_view(guard) {
                        view
                    } else {
                        reset!("right child already freed");
                    };

                    if let Some(ref mut parent_cursor) = parent_cursor_opt {
                        if parent_cursor.is_viable_parent_for(&rhs) {
                            let mut parent_clone: Box<Node<K, V, FANOUT>> =
                                Box::new((*parent_cursor).clone());
                            assert!(!parent_clone.is_leaf());
                            parent_clone.index_mut().insert(rhs.lo.clone(), next);

                            let rhs_ptr_opt = if parent_clone.should_split() {
                                Some(parent_clone.split())
                            } else {
                                None
                            };

                            if let Ok(new_parent_view) = parent_cursor.cas(parent_clone, guard) {
                                parent_cursor_opt = Some(new_parent_view);
                            } else if let Some(rhs_ptr) = rhs_ptr_opt {
                                let reclaimed_ptr: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                                    unsafe { Box::from_raw(rhs_ptr.0 as *mut _) };

                                let _dropping_reclaimed_rhs: Box<Node<K, V, FANOUT>> =
                                    unsafe { Box::from_raw(reclaimed_ptr.load(Ordering::Acquire)) };
                            }
                        }
                    } else {
                        // root hoist
                        let current_root_ptr: AtomicPtr<_> = root_cursor.ptr.as_ptr().into();
                        let new_index_ptr =
                            BoxedAtomicPtr(Box::into_raw(Box::new(current_root_ptr)));

                        let mut new_root_node = Node::<K, V, FANOUT>::new_root();
                        new_root_node
                            .index_mut()
                            .insert(cursor.lo.clone(), new_index_ptr);
                        new_root_node.index_mut().insert(rhs.lo.clone(), next);
                        let new_root_ptr = Box::into_raw(new_root_node);

                        let worked = !debug_delay()
                            && self
                                .root
                                .compare_exchange(
                                    root_cursor.ptr.as_ptr(),
                                    new_root_ptr,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                )
                                .is_ok();

                        if worked {
                            let parent_view = NodeView {
                                id: self.root,
                                ptr: NonNull::new(new_root_ptr).unwrap(),
                            };
                            parent_cursor_opt = Some(parent_view);
                        } else {
                            let dangling_root = unsafe { Box::from_raw(new_root_ptr) };
                            drop(dangling_root);

                            let reclaimed_ptr: Box<AtomicPtr<Node<K, V, FANOUT>>> =
                                unsafe { Box::from_raw(new_index_ptr.0 as *mut _) };
                            drop(reclaimed_ptr);
                        }
                    }

                    cursor = rhs;
                    continue;
                }
            }

            if cursor.is_leaf() {
                assert!(!cursor.is_merging);
                assert!(cursor.merging_child.is_none());
                if let Some(ref hi) = cursor.hi {
                    match search {
                        LeafSearch::Eq(k) => assert!(k < hi.borrow()),
                        LeafSearch::Lt(k) => assert!(k <= hi.borrow()),
                        LeafSearch::Max => {
                            unreachable!("leaf should have no hi key if we're searching for Max")
                        }
                    }
                }
                break;
            }

            // go down the tree
            let index = cursor.index();
            let child_ptr = match search {
                LeafSearch::Eq(k) => index.get_less_than_or_equal(k).unwrap().1,
                LeafSearch::Lt(k) => {
                    // Lt looks for a node with lo < K and hi >= K
                    // so we find the first child with a lo key > K and
                    // return its left sibling
                    index.get_less_than(k).unwrap().1
                }
                LeafSearch::Max => {
                    index
                        .get_index(index.len().checked_sub(1).unwrap())
                        .unwrap()
                        .1
                }
            };

            parent_cursor_opt = Some(cursor);
            cursor = if let Some(view) = child_ptr.node_view(guard) {
                view
            } else {
                reset!("attempt to traverse to child failed because the child has been freed");
            };
        }

        #[cfg(feature = "timing")]
        self.record_timing(before.elapsed());

        cursor
    }
}

#[derive(Debug, Clone)]
#[repr(u8)]
enum Data<K, V, const FANOUT: usize>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    Leaf(StackMap<K, V, FANOUT>),
    Index(StackMap<K, BoxedAtomicPtr<K, V, FANOUT>, FANOUT>),
}

impl<K, V, const FANOUT: usize> Data<K, V, FANOUT>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    const fn len(&self) -> usize {
        match self {
            Data::Leaf(ref leaf) => leaf.len(),
            Data::Index(ref index) => index.len(),
        }
    }
}

#[derive(Debug)]
struct Node<K, V, const FANOUT: usize>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    next: Option<BoxedAtomicPtr<K, V, FANOUT>>,
    merging_child: Option<BoxedAtomicPtr<K, V, FANOUT>>,
    data: Data<K, V, FANOUT>,
    lo: K,
    hi: Option<K>,
    is_merging: bool,
}

impl<K, V, const FANOUT: usize> Clone for Node<K, V, FANOUT>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    fn clone(&self) -> Node<K, V, FANOUT> {
        Node {
            lo: self.lo.clone(),
            hi: self.hi.clone(),
            next: self.next,
            data: self.data.clone(),
            merging_child: self.merging_child,
            is_merging: self.is_merging,
        }
    }
}

impl<K, V, const FANOUT: usize> Node<K, V, FANOUT>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    const fn index(&self) -> &StackMap<K, BoxedAtomicPtr<K, V, FANOUT>, FANOUT> {
        if let Data::Index(ref index) = self.data {
            index
        } else {
            unreachable!()
        }
    }

    fn index_mut(&mut self) -> &mut StackMap<K, BoxedAtomicPtr<K, V, FANOUT>, FANOUT> {
        if let Data::Index(ref mut index) = self.data {
            index
        } else {
            unreachable!()
        }
    }

    const fn leaf(&self) -> &StackMap<K, V, FANOUT> {
        if let Data::Leaf(ref leaf) = self.data {
            leaf
        } else {
            unreachable!()
        }
    }

    fn leaf_mut(&mut self) -> &mut StackMap<K, V, FANOUT> {
        if let Data::Leaf(ref mut leaf) = self.data {
            leaf
        } else {
            unreachable!()
        }
    }

    const fn is_leaf(&self) -> bool {
        matches!(self.data, Data::Leaf(..))
    }

    fn new_root() -> Box<Node<K, V, FANOUT>> {
        let min_key = K::MIN;
        Box::new(Node {
            lo: min_key,
            hi: None,
            next: None,
            data: Data::Index(Default::default()),
            merging_child: None,
            is_merging: false,
        })
    }

    fn new_leaf(lo: K) -> Box<Node<K, V, FANOUT>> {
        Box::new(Node {
            lo,
            hi: None,
            next: None,
            data: Data::Leaf(Default::default()),
            merging_child: None,
            is_merging: false,
        })
    }

    fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());
        assert!(self.is_leaf());

        self.leaf().get(key).cloned()
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());

        self.leaf_mut().remove(key)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());
        assert!(!self.should_split());

        self.leaf_mut().insert(key, value)
    }

    fn cas<V2>(
        &mut self,
        key: K,
        old: Option<&V2>,
        new: Option<V>,
    ) -> Result<Option<V>, CasFailure<V>>
    where
        V: Borrow<V2>,
        V2: ?Sized + PartialEq,
    {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());

        // anything that should be split should have been split
        // prior to becoming globally visible via this codepath.
        assert!(!self.should_split());

        match (old, self.leaf().get(&key)) {
            (expected, actual) if expected == actual.map(Borrow::borrow) => {
                if let Some(to_insert) = new {
                    Ok(self.leaf_mut().insert(key, to_insert))
                } else {
                    Ok(self.leaf_mut().remove(&key))
                }
            }
            (_, actual) => Err(CasFailure {
                actual: actual.cloned(),
                returned_new_value: new,
            }),
        }
    }

    const fn should_merge(&self) -> bool {
        if self.merging_child.is_some() || self.is_merging {
            return false;
        }
        self.len() <= MERGE_SIZE
    }

    const fn should_split(&self) -> bool {
        if self.merging_child.is_some() || self.is_merging {
            return false;
        }
        self.len() >= FANOUT - MERGE_SIZE
    }

    const fn len(&self) -> usize {
        self.data.len()
    }

    fn split(&mut self) -> BoxedAtomicPtr<K, V, FANOUT> {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());

        let split_idx = if self.hi.is_none() {
            // the infinity node should split almost at the end to improve fill ratio
            self.len() - 2
        } else if self.lo == K::MIN {
            // the right-most node should split almost at the beginning to improve fill ratio
            1
        } else {
            FANOUT / 2
        };

        let (split_point, rhs_data) = match self.data {
            Data::Leaf(ref mut leaf) => {
                let (split_point, rhs_leaf) = leaf.split_off(split_idx);
                (split_point, Data::Leaf(rhs_leaf))
            }
            Data::Index(ref mut index) => {
                let (split_point, rhs_index) = index.split_off(split_idx);
                (split_point, Data::Index(rhs_index))
            }
        };

        assert!(rhs_data.len() < FANOUT - MERGE_SIZE);

        let rhs_hi = std::mem::replace(&mut self.hi, Some(split_point.clone()));

        let rhs = BoxedAtomicPtr::new(Box::new(Node {
            lo: split_point,
            hi: rhs_hi,
            next: self.next,
            data: rhs_data,
            merging_child: None,
            is_merging: false,
        }));

        self.next = Some(rhs);

        assert!(!self.should_split());

        rhs
    }

    fn merge(&mut self, rhs: &NodeView<K, V, FANOUT>) {
        assert!(rhs.is_merging);
        assert!(!self.is_merging);

        self.hi = rhs.hi.clone();
        self.next = rhs.next;

        match self.data {
            Data::Leaf(ref mut leaf) => {
                for (k, v) in rhs.leaf().iter() {
                    let prev = leaf.insert(k.clone(), v.clone());
                    assert!(prev.is_none());
                }
            }
            Data::Index(ref mut index) => {
                for (k, v) in rhs.index().iter() {
                    let prev = index.insert(k.clone(), *v);
                    assert!(prev.is_none());
                }
            }
        }
    }

    fn is_viable_parent_for(&self, possible_child: &NodeView<K, V, FANOUT>) -> bool {
        match (&self.hi, &possible_child.hi) {
            (Some(_), None) => return false,
            (Some(parent_hi), Some(child_hi)) if parent_hi < child_hi => return false,
            _ => {}
        }
        self.lo <= possible_child.lo
    }
}

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> FromIterator<(K, V)>
    for ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let map = ConcurrentMap::default();

        for (k, v) in iter {
            map.insert(k, v);
        }

        map
    }
}

impl<'a, K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> IntoIterator
    for &'a ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Clone + Send + Sync,
{
    type Item = (K, V);
    type IntoIter = Iter<'a, K, V, FANOUT, LOCAL_GC_BUFFER_SIZE, std::ops::RangeFull>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

const fn _test_impls() {
    const fn send<T: Send>() {}
    const fn clone<T: Clone>() {}
    send::<ConcurrentMap<usize, usize>>();
    clone::<ConcurrentMap<usize, usize>>();
}

#[test]
fn basic_map() {
    let map = ConcurrentMap::<usize, usize>::default();

    let n = 64; // SPLIT_SIZE
    for i in 0..=n {
        assert_eq!(map.get(&i), None);
        map.insert(i, i);
        assert_eq!(map.get(&i), Some(i), "failed to get key {i}");
    }

    for (i, (k, _v)) in map.range(..).enumerate() {
        assert_eq!(i, k);
    }

    for (i, (k, _v)) in map.range(..).rev().enumerate() {
        assert_eq!(n - i, k);
    }

    for (i, (k, _v)) in map.iter().enumerate() {
        assert_eq!(i, k);
    }

    for (i, (k, _v)) in map.iter().rev().enumerate() {
        assert_eq!(n - i, k);
    }

    for (i, (k, _v)) in map.range(0..).enumerate() {
        assert_eq!(i, k);
    }

    for (i, (k, _v)) in map.range(0..).rev().enumerate() {
        assert_eq!(n - i, k);
    }

    for (i, (k, _v)) in map.range(0..n).enumerate() {
        assert_eq!(i, k);
    }

    for (i, (k, _v)) in map.range(0..n).rev().enumerate() {
        assert_eq!((n - 1) - i, k);
    }

    for (i, (k, _v)) in map.range(0..=n).enumerate() {
        assert_eq!(i, k);
    }

    for (i, (k, _v)) in map.range(0..=n).rev().enumerate() {
        assert_eq!(n - i, k);
    }

    for i in 0..=n {
        assert_eq!(map.get(&i), Some(i), "failed to get key {i}");
    }
}

#[test]
fn timing_map() {
    use std::time::Instant;

    let map = ConcurrentMap::<u64, u64>::default();

    let n = 1024 * 1024;

    let insert = Instant::now();
    for i in 0..n {
        map.insert(i, i);
    }
    let insert_elapsed = insert.elapsed();
    println!(
        "{} inserts/s, total {:?}",
        (n * 1_000_000) / u64::try_from(insert_elapsed.as_micros().max(1)).unwrap_or(u64::MAX),
        insert_elapsed
    );

    let scan = Instant::now();
    let count = map.range(..).count();
    assert_eq!(count as u64, n);
    let scan_elapsed = scan.elapsed();
    println!(
        "{} scanned items/s, total {:?}",
        (n * 1_000_000) / u64::try_from(scan_elapsed.as_micros().max(1)).unwrap_or(u64::MAX),
        scan_elapsed
    );

    let scan_rev = Instant::now();
    let count = map.range(..).rev().count();
    assert_eq!(count as u64, n);
    let scan_rev_elapsed = scan_rev.elapsed();
    println!(
        "{} reverse-scanned items/s, total {:?}",
        (n * 1_000_000) / u64::try_from(scan_rev_elapsed.as_micros().max(1)).unwrap_or(u64::MAX),
        scan_rev_elapsed
    );

    let gets = Instant::now();
    for i in 0..n {
        map.get(&i);
    }
    let gets_elapsed = gets.elapsed();
    println!(
        "{} gets/s, total {:?}",
        (n * 1_000_000) / u64::try_from(gets_elapsed.as_micros().max(1)).unwrap_or(u64::MAX),
        gets_elapsed
    );
}
