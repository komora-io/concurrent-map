#![allow(unused)]
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Bound, Deref};
use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicPtr, AtomicU64, Ordering},
    Arc,
};

use ebr::{Ebr, Guard};
use pagetable::PageTable;

use crate::{ConcurrentStack, ConcurrentStackPusher};

#[cfg(test)]
const SPLIT_SIZE: usize = 4;

#[cfg(test)]
const MERGE_SIZE: usize = 1;

#[cfg(not(test))]
const SPLIT_SIZE: usize = 16;

#[cfg(not(test))]
const MERGE_SIZE: usize = 3;

type Id = u64;

enum Deferred<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    DropNode(Box<Node<K, V>>),
    MarkIdReusable {
        stack: ConcurrentStackPusher<u64>,
        id: Id,
    },
}

impl<K, V> Drop for Deferred<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn drop(&mut self) {
        if let Deferred::MarkIdReusable { stack, id } = self {
            stack.push(*id);
        }
    }
}

#[derive(Debug)]
struct NodeView<'a, K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    ptr: NonNull<Node<K, V>>,
    slot: &'a AtomicPtr<Node<K, V>>,
    id: u64,
}

impl<'a, K, V> NodeView<'a, K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    /// Try to replace. If the node has been deleted since we got our view,
    /// an Err(None) is returned.
    fn cas(
        &self,
        replacement: Box<Node<K, V>>,
        guard: &mut Guard<'a, Deferred<K, V>>,
    ) -> Result<NodeView<'a, K, V>, Option<NodeView<'a, K, V>>> {
        // println!("replacing:");
        // println!("nodeview:     {:?}", *self);
        // println!("current:      {:?}", **self);
        // println!("new:          {:?}", *replacement);
        assert!(
            !(replacement.hi.is_some() ^ replacement.next.is_some()),
            "hi and next must both either be None or Some"
        );
        let replacement_ptr = Box::into_raw(replacement);

        let res = self.slot.compare_exchange(
            self.ptr.as_ptr(),
            replacement_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        match res {
            Ok(_) => {
                let replaced: Box<Node<K, V>> = unsafe { Box::from_raw(self.ptr.as_ptr()) };
                guard.defer_drop(Deferred::DropNode(replaced));
                Ok(NodeView {
                    slot: self.slot,
                    ptr: NonNull::new(replacement_ptr).unwrap(),
                    id: self.id,
                })
            }
            Err(actual) => {
                let failed_value = unsafe { Box::from_raw(replacement_ptr) };
                drop(failed_value);

                if actual.is_null() {
                    Err(None)
                } else {
                    Err(Some(NodeView {
                        ptr: NonNull::new(actual).unwrap(),
                        slot: self.slot,
                        id: self.id,
                    }))
                }
            }
        }
    }
}

impl<'a, K, V> Deref for NodeView<'a, K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    type Target = Node<K, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

/// This trait should be implemented for anything you wish to use
/// as a key in the `ConcurrentMap`.
pub trait Minimum {
    fn minimum() -> Self;
}

macro_rules! impl_min {
    ($($t:ty),+) => {
        $(
            impl Minimum for $t {
                #[inline]
                fn minimum() -> Self {
                    <$t>::MIN
                }
            }
        )*
    }
}

impl_min!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128);

macro_rules! impl_collection_default {
    ($($t:ty),+) => {
        $(
            impl<T> Minimum for $t {
                #[inline]
                fn minimum() -> Self {
                    <$t>::default()
                }
            }
        )*
    }
}

impl_collection_default!(
    Vec<T>,
    std::collections::VecDeque<T>,
    std::collections::HashSet<T>,
    std::collections::BTreeSet<T>,
    std::collections::LinkedList<T>
);

impl<T> Minimum for &[T] {
    fn minimum() -> Self {
        &[]
    }
}

impl<T: Minimum, const LEN: usize> Minimum for [T; LEN] {
    fn minimum() -> Self {
        core::array::from_fn(|_i| T::minimum())
    }
}

impl<T: Minimum> Minimum for Box<T> {
    fn minimum() -> Self {
        Box::new(T::minimum())
    }
}

impl Minimum for String {
    fn minimum() -> Self {
        String::new()
    }
}

impl Minimum for &str {
    fn minimum() -> Self {
        ""
    }
}

/// A lock-free B+ tree.
///
/// If you want to use a custom key type, you must
/// implement the `concurrent_map::Minimum` trait,
/// allowing the left-most side of the tree to be
/// created before inserting any data.
#[derive(Clone)]
pub struct ConcurrentMap<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    ebr: Ebr<Deferred<K, V>>,
    idgen: Arc<AtomicU64>,
    inner: Arc<Inner<K, V>>,
    free_ids: ConcurrentStack<u64>,
}

impl<K, V> Default for ConcurrentMap<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn default() -> ConcurrentMap<K, V> {
        let initial_root_id = 0;
        let initial_leaf_id = 1;

        let table = PageTable::<AtomicPtr<Node<K, V>>>::default();

        let mut root_node = Node::<K, V>::new_root();
        root_node
            .index
            .insert(root_node.lo.clone(), initial_leaf_id);
        let leaf_node = Node::<K, V>::new_leaf(root_node.lo.clone());

        let root_ptr = Box::into_raw(root_node);
        let leaf_ptr = Box::into_raw(leaf_node);

        table
            .get(initial_root_id)
            .store(root_ptr, Ordering::Release);
        table
            .get(initial_leaf_id)
            .store(leaf_ptr, Ordering::Release);

        let root_id = initial_root_id.into();

        let inner = Arc::new(Inner { root_id, table });

        ConcurrentMap {
            ebr: Ebr::default(),
            idgen: Arc::new(2.into()),
            free_ids: ConcurrentStack::default(),
            inner,
        }
    }
}

#[derive(Default)]
pub struct Inner<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    root_id: AtomicU64,
    table: PageTable<AtomicPtr<Node<K, V>>>,
}

impl<K, V> Drop for Inner<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn drop(&mut self) {
        let mut ebr = Ebr::default();
        let mut guard = ebr.pin();

        let mut cursor = self.root(&mut guard);

        let mut lhs_chain = vec![];
        loop {
            lhs_chain.push(cursor.id);
            if cursor.is_leaf {
                break;
            }
            let child_id = *cursor.index.iter().next().unwrap().1;

            cursor = self.view_for_id(child_id, &mut guard).unwrap();
        }

        for lhs_id in lhs_chain {
            let mut next_opt = Some(lhs_id);
            while let Some(next) = next_opt {
                let cursor = self.view_for_id(next, &mut guard).unwrap();
                next_opt = cursor.next;
                let node_box = unsafe { Box::from_raw(cursor.ptr.as_ptr()) };
                drop(node_box);
            }
        }
    }
}

impl<K, V> ConcurrentMap<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    pub fn get(&mut self, key: &K) -> Option<Arc<V>> {
        let mut guard = self.ebr.pin();

        let leaf = self
            .inner
            .leaf_for_key(key, &self.idgen, &mut self.free_ids, &mut guard);

        leaf.get(key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<Arc<V>> {
        let key_arc = Arc::new(key);
        let value_arc = Arc::new(value);
        loop {
            let mut guard = self.ebr.pin();
            let leaf =
                self.inner
                    .leaf_for_key(&*key_arc, &self.idgen, &mut self.free_ids, &mut guard);
            let mut leaf_clone: Box<Node<K, V>> = Box::new((*leaf).clone());
            let ret = leaf_clone.insert(key_arc.clone(), value_arc.clone());

            let mut rhs_id = None;

            if leaf_clone.should_split() {
                let new_id = self
                    .free_ids
                    .pop()
                    .unwrap_or_else(|| self.idgen.fetch_add(1, Ordering::Relaxed));

                let (lhs, rhs) = leaf_clone.split(new_id);

                let rhs_ptr = Box::into_raw(Box::new(rhs));

                let atomic_ptr_ref = self.inner.table.get(new_id);

                let prev = atomic_ptr_ref.swap(rhs_ptr, Ordering::Release);

                assert!(prev.is_null());

                rhs_id = Some(new_id);

                leaf_clone = Box::new(lhs);
            };

            let install_attempt = leaf.cas(leaf_clone, &mut guard);

            if install_attempt.is_ok() {
                return ret;
            } else {
                if let Some(new_id) = rhs_id {
                    let atomic_ptr_ref = self.inner.table.get(new_id);

                    // clear dangling ptr (cas freed it already)
                    let _rhs_ptr = atomic_ptr_ref.swap(std::ptr::null_mut(), Ordering::AcqRel);
                    self.free_ids.push(new_id);
                }
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        loop {
            let mut guard = self.ebr.pin();
            let leaf = self
                .inner
                .leaf_for_key(key, &self.idgen, &mut self.free_ids, &mut guard);
            let mut leaf_clone: Box<Node<K, V>> = Box::new((*leaf).clone());
            let ret = leaf_clone.remove(key);
            let install_attempt = leaf.cas(leaf_clone, &mut guard);
            if install_attempt.is_ok() {
                return ret;
            }
        }
    }

    fn iter(&mut self) -> Iter<'_, K, V>
    where
        K: Clone,
    {
        self.range::<K, _>(..)
    }

    fn range<'a, T: ?Sized, R>(&mut self, range: R) -> Iter<'_, K, V>
    where
        K: Clone,
        R: 'static + std::ops::RangeBounds<K>,
    {
        let mut guard = self.ebr.pin();

        let current = match range.start_bound() {
            Bound::Included(t) => {
                self.inner
                    .leaf_for_key(t, &self.idgen, &mut self.free_ids, &mut guard)
            }
            Bound::Excluded(t) => {
                self.inner
                    .leaf_for_key(t, &self.idgen, &mut self.free_ids, &mut guard)
            }
            Bound::Unbounded => self.inner.leftmost_leaf(&mut guard),
        };

        Iter {
            guard,
            inner: &self.inner,
            lo: bound_map(range.start_bound()),
            hi: bound_map(range.end_bound()),
            current,
        }
    }
}

fn bound_map<T, K>(bound: Bound<T>) -> Bound<Arc<K>>
where
    K: Clone,
    T: std::borrow::Borrow<K> + Ord,
{
    match bound {
        Bound::Included(t) => Bound::Included(t.borrow().clone().into()),
        Bound::Excluded(t) => Bound::Excluded(t.borrow().clone().into()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

struct Iter<'a, K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    inner: &'a Inner<K, V>,
    guard: Guard<'a, Deferred<K, V>>,
    lo: Bound<Arc<K>>,
    hi: Bound<Arc<K>>,
    current: NodeView<'a, K, V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    Bound<Arc<K>>: PartialOrd,
    V: 'static + fmt::Debug + Send + Sync,
{
    type Item = (Arc<K>, Arc<V>);
    fn next(&mut self) -> Option<Self::Item> {
        /*
                if Bound::Included(self.current.lo.clone()) > self.hi {
                    return None;
                }
                let mut iter = self.current.leaf.range((&self.lo, &self.hi));
                iter.next().map(|(k, v)| (k.clone(), v.clone()))
        */
        todo!()
    }
}

/*
impl<'a, K, V> IntoIterator for &'a mut ConcurrentMap<K, V>
where
    K: 'static + Clone + fmt::Debug + Minimum + Ord + Send + Sync,
    Bound<Arc<K>>: PartialOrd,
    V: 'static + fmt::Debug + Send + Sync,
{
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}
*/

/*
impl<K, V> fmt::Debug for Inner<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Inner").finish()
    }
}
*/

impl<K, V> Inner<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn view_for_id<'a>(
        &'a self,
        id: Id,
        _guard: &mut Guard<'a, Deferred<K, V>>,
    ) -> Option<NodeView<'a, K, V>> {
        let slot = self.table.get(id);
        let ptr = NonNull::new(slot.load(Ordering::Acquire))?;

        Some(NodeView { ptr, slot, id })
    }

    fn root<'a>(&'a self, guard: &mut Guard<'a, Deferred<K, V>>) -> NodeView<'a, K, V> {
        loop {
            let root_id = self.root_id.load(Ordering::Acquire);

            if let Some(view) = self.view_for_id(root_id, guard) {
                return view;
            }
        }
    }

    fn leftmost_leaf<'a>(&'a self, guard: &mut Guard<'a, Deferred<K, V>>) -> NodeView<'a, K, V> {
        let mut cursor = self.root(guard);

        let mut lhs_chain = vec![];
        loop {
            lhs_chain.push(cursor.id);
            if cursor.is_leaf {
                return cursor;
            }
            let child_id = *cursor.index.iter().next().unwrap().1;

            cursor = self.view_for_id(child_id, guard).unwrap();
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
    // 5. cas the parent to remove the merged child
    // 6. remove the child's pointer in the page table
    // 7. defer the reclamation of the child node
    // 8. defer putting the child's ID into the free stack
    //
    // merge threshold must be >= 1, because otherwise index nodes with 1 empty
    // child will never be compactible.

    fn install_parent_merge<'a>(
        &'a self,
        parent: &NodeView<'a, K, V>,
        child: &NodeView<'a, K, V>,
        guard: &mut Guard<'a, Deferred<K, V>>,
    ) -> Result<NodeView<'a, K, V>, ()> {
        // 1. try to mark the parent's merging_child
        //  a. must not be the left-most child
        //  b. if unsuccessful, give up
        let is_leftmost_child = parent.index.iter().next().unwrap().0 == &child.lo;

        if is_leftmost_child {
            return Err(());
        }

        if !parent.index.contains_key(&child.lo) {
            // can't install a parent merge if the child is unknown to the parent.
            return Err(());
        }

        if parent.merging_child.is_some() {
            // there is already a merge in-progress for a different child.
            return Err(());
        }

        let mut parent_clone: Box<Node<K, V>> = Box::new((*parent).clone());
        parent_clone.merging_child = Some(child.id);
        parent.cas(parent_clone, guard).map_err(|_| ())
    }

    fn merge_child<'a>(
        &'a self,
        parent: &mut NodeView<'a, K, V>,
        child: &mut NodeView<'a, K, V>,
        free_ids: &ConcurrentStack<u64>,
        guard: &mut Guard<'a, Deferred<K, V>>,
    ) {
        // 2. mark child as merging
        while !child.is_merging {
            let mut child_clone: Box<Node<K, V>> = Box::new((*child).clone());
            child_clone.is_merging = true;
            *child = match child.cas(child_clone, guard) {
                Ok(new_child) | Err(Some(new_child)) => new_child,
                Err(None) => {
                    // child already removed
                    /*
                    println!(
                        "returning early from merge_child because \
                        the merging child has already been freed"
                    );
                    */
                    return;
                }
            };
        }

        // 3. find the left sibling
        let first_left_sibling_guess = parent
            .index
            .range::<Arc<K>, _>(..&child.lo)
            .next_back()
            .unwrap()
            .1;
        let mut left_sibling =
            if let Some(view) = self.view_for_id(*first_left_sibling_guess, guard) {
                view
            } else {
                // the merge already completed and this left sibling has already also been merged
                /*
                println!(
                    "returning early from merge_child because the \
                    first left sibling guess is freed, meaning the \
                    merge already succeeded"
                );
                */
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
                // println!("breaking from the left sibling install loop because left_sibling.hi ({:?}) >= child.hi ({:?})", left_sibling.hi, child.hi);
                break;
            }

            let next = left_sibling.next.unwrap();
            if next != child.id {
                left_sibling = if let Some(view) = self.view_for_id(next, guard) {
                    view
                } else {
                    // the merge already completed and this left sibling has already also been merged
                    /*
                    println!(
                        "returning early from merge_child because one of the left siblings \
                        of the merging child is already freed, meaning \
                        the merge must have completed already"
                    );
                    */
                    return;
                };
                continue;
            }

            // 4. cas the left sibling to eat the right sibling
            //  a. loop until successful
            //  b. go right if the left-most child split and no-longer points to merging child
            let mut left_sibling_clone: Box<Node<K, V>> = Box::new((*left_sibling).clone());
            left_sibling_clone.merge(child);
            let cas_result = left_sibling.cas(left_sibling_clone, guard);
            match cas_result {
                Ok(_) => {
                    // println!("successfully merged child into its left sibling");
                    break;
                }
                Err(Some(actual)) => left_sibling = actual,
                Err(None) => {
                    /*
                    println!(
                        "returning early from merge_child because \
                        one of the left siblings has already been \
                        freed, implying the original merge completed."
                    );
                    */
                    return;
                }
            }
        }

        // 5. cas the parent to remove the merged child
        while parent.merging_child == Some(child.id) {
            let mut parent_clone: Box<Node<K, V>> = Box::new((*parent).clone());

            assert!(parent_clone.merging_child.is_some());
            assert!(parent_clone.index.contains_key(&child.lo));

            parent_clone.merging_child = None;
            parent_clone.index.remove(&child.lo).unwrap();

            let cas_result = parent.cas(parent_clone, guard);
            match cas_result {
                Ok(new_parent) | Err(Some(new_parent)) => *parent = new_parent,
                Err(None) => {
                    /*
                    println!(
                        "returning early from merge_child because the parent \
                        of the merged child has already been freed, indicating \
                        that the merge already completed."
                    );
                    */
                    return;
                }
            }
        }

        // 6. remove the child's pointer in the page table
        if child
            .slot
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
            /*
            println!(
                "returning early from merge_child because we lost \
                the race to remove the child pointer from the \
                page table."
            );
            */
            return;
        }

        // 7. defer putting the child's ID into the free stack
        guard.defer_drop(Deferred::MarkIdReusable {
            stack: free_ids.get_pusher(),
            id: child.id,
        });

        // 8. defer the reclamation of the child node
        let replaced: Box<Node<K, V>> = unsafe { Box::from_raw(child.ptr.as_ptr()) };
        guard.defer_drop(Deferred::DropNode(replaced));

        // println!("merge_child fully completed");
    }

    fn leaf_for_key<'a>(
        &'a self,
        key: &K,
        idgen: &AtomicU64,
        free_ids: &mut ConcurrentStack<u64>,
        guard: &mut Guard<'a, Deferred<K, V>>,
    ) -> NodeView<'a, K, V> {
        // println!("looking for key {key:?}");
        let mut parent_cursor_opt: Option<NodeView<'a, K, V>> = None;
        let mut cursor = self.root(guard);
        let mut root_id = cursor.id;

        macro_rules! reset {
            ($reason:expr) => {
                // println!("reset at {} due to {}", line!(), $reason);
                parent_cursor_opt = None;
                cursor = self.root(guard);
                root_id = cursor.id;
                continue;
            };
        }

        loop {
            // println!("cursor is: {} -> {:?}", cursor.id, *cursor);
            if let Some(merging_child_id) = cursor.merging_child {
                let mut child = if let Some(view) = self.view_for_id(merging_child_id, guard) {
                    view
                } else {
                    reset!("merging child of marked parent already freed");
                };
                self.merge_child(&mut cursor, &mut child, free_ids, guard);
                reset!("cooperatively performed merge_child after detecting parent");
            }

            if cursor.is_merging {
                reset!("resetting after detected child merging without corresponding parent child_merge");
            }

            if cursor.should_merge() {
                if let Some(ref mut parent_cursor) = parent_cursor_opt {
                    let is_leftmost_child =
                        parent_cursor.index.iter().next().unwrap().0 == &cursor.lo;

                    if !is_leftmost_child {
                        if let Ok(new_parent) =
                            self.install_parent_merge(parent_cursor, &cursor, guard)
                        {
                            *parent_cursor = new_parent;
                        } else {
                            reset!("failed to install parent merge");
                        }

                        self.merge_child(parent_cursor, &mut cursor, free_ids, guard);
                        reset!("completed merge_child");
                    }
                }
            }

            assert!(key >= &*cursor.lo);
            if let Some(hi) = &cursor.hi {
                if key >= hi {
                    // go right to the tree sibling
                    let next = cursor.next.unwrap();
                    let rhs = if let Some(view) = self.view_for_id(next, guard) {
                        view
                    } else {
                        // println!("right child {next} not found");
                        reset!("right child already freed");
                    };

                    if let Some(ref mut parent_cursor) = parent_cursor_opt {
                        if parent_cursor.is_viable_parent_for(&rhs) {
                            let mut parent_clone: Box<Node<K, V>> =
                                Box::new((*parent_cursor).clone());
                            assert!(!parent_clone.is_leaf);
                            parent_clone.index.insert(rhs.lo.clone(), next);

                            let mut rhs_id = None;

                            if parent_clone.should_split() {
                                let new_id = free_ids
                                    .pop()
                                    .unwrap_or_else(|| idgen.fetch_add(1, Ordering::Relaxed));

                                let (lhs, rhs) = parent_clone.split(new_id);

                                let rhs_ptr = Box::into_raw(Box::new(rhs));

                                let atomic_ptr_ref = self.table.get(new_id);

                                let prev = atomic_ptr_ref.swap(rhs_ptr, Ordering::Release);

                                assert!(prev.is_null());

                                rhs_id = Some(new_id);

                                parent_clone = Box::new(lhs);
                            };

                            if let Ok(new_parent_view) = parent_cursor.cas(parent_clone, guard) {
                                parent_cursor_opt = Some(new_parent_view);
                            } else if let Some(new_id) = rhs_id {
                                let atomic_ptr_ref = self.table.get(new_id);

                                // clear dangling ptr (cas freed it already)
                                let _rhs_ptr =
                                    atomic_ptr_ref.swap(std::ptr::null_mut(), Ordering::AcqRel);
                                free_ids.push(new_id);
                            }
                        }
                    } else {
                        // root hoist
                        let new_id = free_ids
                            .pop()
                            .unwrap_or_else(|| idgen.fetch_add(1, Ordering::Relaxed));

                        let mut new_root_node = Node::<K, V>::new_root();
                        new_root_node.index.insert(cursor.lo.clone(), root_id);
                        new_root_node.index.insert(rhs.lo.clone(), next);
                        let new_root_ptr = Box::into_raw(new_root_node);

                        let atomic_ptr_ref = self.table.get(new_id);
                        let prev = atomic_ptr_ref.swap(new_root_ptr, Ordering::Release);
                        assert!(prev.is_null());

                        if self
                            .root_id
                            .compare_exchange(root_id, new_id, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                        {
                            let parent_view = NodeView {
                                id: new_id,
                                ptr: NonNull::new(new_root_ptr).unwrap(),
                                slot: atomic_ptr_ref,
                            };
                            parent_cursor_opt = Some(parent_view);
                        } else {
                            let atomic_ptr_ref = self.table.get(new_id);

                            let root_ptr =
                                atomic_ptr_ref.swap(std::ptr::null_mut(), Ordering::AcqRel);
                            let dangling_root = unsafe { Box::from_raw(root_ptr) };
                            drop(dangling_root);

                            // it's safe to directly push this id into the free list because
                            // it was never accessible via the atomic root marker.
                            free_ids.push(new_id);
                        }
                    }

                    cursor = rhs;
                    continue;
                }
            }

            if cursor.is_leaf {
                assert!(!cursor.is_merging);
                assert!(cursor.merging_child.is_none());
                assert!(&*cursor.lo <= &*key);
                if let Some(ref hi) = cursor.hi {
                    assert!(&**hi > &*key);
                }
                break;
            }

            // go down the tree
            let child_id = *cursor.index.range::<K, _>(..=key).next_back().unwrap().1;
            parent_cursor_opt = Some(cursor);
            cursor = if let Some(view) = self.view_for_id(child_id, guard) {
                view
            } else {
                reset!("attempt to traverse to child failed because the child has been freed");
            };
        }
        // println!("final leaf is: {:?}", *cursor);

        cursor
    }
}

#[derive(Debug)]
struct Node<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    lo: Arc<K>,
    hi: Option<Arc<K>>,
    next: Option<Id>,
    merging_child: Option<Id>,
    is_merging: bool,
    is_leaf: bool,
    leaf: BTreeMap<Arc<K>, Arc<V>>,
    index: BTreeMap<Arc<K>, Id>,
}

impl<K, V> Clone for Node<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn clone(&self) -> Node<K, V> {
        Node {
            lo: self.lo.clone(),
            hi: self.hi.clone(),
            next: self.next,
            is_leaf: self.is_leaf,
            leaf: self.leaf.clone(),
            index: self.index.clone(),
            merging_child: self.merging_child.clone(),
            is_merging: self.is_merging.clone(),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: 'static + fmt::Debug + Minimum + Ord + Send + Sync,
    V: 'static + fmt::Debug + Send + Sync,
{
    fn new_root() -> Box<Node<K, V>> {
        let min_key = Arc::new(K::minimum());
        Box::new(Node {
            lo: min_key,
            hi: None,
            next: None,
            is_leaf: false,
            leaf: Default::default(),
            index: Default::default(),
            merging_child: None,
            is_merging: false,
        })
    }

    fn new_leaf(lo: Arc<K>) -> Box<Node<K, V>> {
        Box::new(Node {
            lo,
            hi: None,
            next: None,
            is_leaf: true,
            leaf: Default::default(),
            index: Default::default(),
            merging_child: None,
            is_merging: false,
        })
    }

    fn get(&self, key: &K) -> Option<Arc<V>> {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());
        assert!(self.is_leaf);

        self.leaf.get(key).cloned()
    }

    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());

        self.leaf.remove(key)
    }

    fn insert(&mut self, key: Arc<K>, value: Arc<V>) -> Option<Arc<V>> {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());

        self.leaf.insert(key, value)
    }

    fn should_merge(&self) -> bool {
        if self.merging_child.is_some() || self.is_merging {
            return false;
        }
        if self.is_leaf {
            self.leaf.len() <= MERGE_SIZE
        } else {
            self.index.len() <= MERGE_SIZE
        }
    }

    fn should_split(&self) -> bool {
        if self.merging_child.is_some() || self.is_merging {
            return false;
        }
        if self.is_leaf {
            self.leaf.len() >= SPLIT_SIZE
        } else {
            self.index.len() >= SPLIT_SIZE
        }
    }

    fn split(mut self, new_id: u64) -> (Node<K, V>, Node<K, V>) {
        assert!(!self.is_merging);
        assert!(self.merging_child.is_none());

        let split_idx = SPLIT_SIZE / 2;

        let (split_point, rhs_leaf, rhs_index) = if self.is_leaf {
            let split_point = self.leaf.keys().nth(split_idx).unwrap().clone();

            let rhs_leaf = self.leaf.split_off(&split_point);

            (split_point, rhs_leaf, Default::default())
        } else {
            let split_point = self.index.keys().nth(split_idx).unwrap().clone();

            let rhs_index = self.index.split_off(&split_point);

            (split_point, Default::default(), rhs_index)
        };

        let rhs_hi = std::mem::replace(&mut self.hi, Some(split_point.clone()));
        let rhs_next = std::mem::replace(&mut self.next, Some(new_id));

        let rhs = Node {
            lo: split_point.clone(),
            hi: rhs_hi,
            next: rhs_next,
            is_leaf: self.is_leaf,
            index: rhs_index,
            leaf: rhs_leaf,
            merging_child: None,
            is_merging: false,
        };

        assert_eq!(self.hi, Some(split_point));

        (self, rhs)
    }

    fn merge(&mut self, rhs: &NodeView<'_, K, V>) {
        assert!(rhs.is_merging);
        assert!(!self.is_merging);

        self.hi = rhs.hi.clone();
        self.next = rhs.next;

        if self.is_leaf {
            for (k, v) in &rhs.leaf {
                let prev = self.leaf.insert(k.clone(), v.clone());
                assert!(prev.is_none());
            }
        } else {
            for (k, v) in &rhs.leaf {
                let prev = self.leaf.insert(k.clone(), v.clone());
                assert!(prev.is_none());
            }
        };
    }

    fn is_viable_parent_for(&self, possible_child: &NodeView<'_, K, V>) -> bool {
        match (&self.hi, &possible_child.hi) {
            (Some(_), None) => return false,
            (Some(parent_hi), Some(child_hi)) if parent_hi < child_hi => return false,
            _ => {}
        }
        self.lo <= possible_child.lo
    }
}

#[test]
fn basic_tree() {
    let mut tree = ConcurrentMap::default();

    let n = 64; // SPLIT_SIZE
    for i in 0..n {
        assert_eq!(tree.get(&i), None);
        tree.insert(i, i);
        assert_eq!(
            tree.get(&i).map(|arc| *arc),
            Some(i),
            "failed to get key {i}"
        );
    }
    for i in 0..n {
        assert_eq!(
            tree.get(&i).map(|arc| *arc),
            Some(i),
            "failed to get key {i}"
        );
    }
}

#[test]
fn concurrent_tree() {
    let n: u16 = 1024;
    let concurrency: u16 = 32;

    let run = |mut tree: ConcurrentMap<u16, u16>, barrier: &std::sync::Barrier, low_bits| {
        let shift = concurrency.next_power_of_two().trailing_zeros();
        let unique_key = |key| (key << shift) | low_bits;

        barrier.wait();
        for key in 0..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i), None);
            tree.insert(i, i);
            assert_eq!(
                tree.get(&i).map(|arc| *arc),
                Some(i),
                "failed to get key {i}"
            );
        }
        for key in 0_u16..n {
            let i = unique_key(key);
            assert_eq!(
                tree.get(&i).map(|arc| *arc),
                Some(i),
                "failed to get key {i}"
            );
        }
        for key in 0..n {
            let i = unique_key(key);
            assert_eq!(tree.remove(&i).map(|arc| *arc), Some(i));
        }
        for key in 0..n {
            let i = unique_key(key);
            assert_eq!(tree.get(&i).map(|arc| *arc), None, "failed to get key {i}");
        }
        println!("done");
    };

    let mut tree = ConcurrentMap::default();

    std::thread::scope(|s| {
        for _ in 0..64 {
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(concurrency as usize));
            let mut threads = vec![];
            for i in 0..concurrency {
                let tree = tree.clone();
                let barrier = barrier.clone();

                let thread = s.spawn(move || run(tree, &barrier, i));
                threads.push(thread);
            }
            for thread in threads.into_iter() {
                thread.join().unwrap();
            }
        }
    });
}
