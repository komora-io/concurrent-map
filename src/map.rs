#![allow(unused)]
use std::collections::BTreeMap;
use std::ops::{Bound, Deref};
use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicPtr, AtomicU64, Ordering},
    Arc,
};

use ebr::{Ebr, Guard};
use pagetable::PageTable;

use crate::ConcurrentStack;

const SPLIT_SIZE: usize = 4;

type Id = u64;

struct NodeView<'a, K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    ptr: NonNull<Node<K, V>>,
    slot: &'a AtomicPtr<Node<K, V>>,
    id: u64,
}

impl<'a, K, V> NodeView<'a, K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    /// Try to replace
    fn cas(
        &self,
        replacement: Box<Node<K, V>>,
        guard: &mut Guard<'a, Box<Node<K, V>>>,
    ) -> Result<NodeView<'a, K, V>, ()> {
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
                guard.defer_drop(replaced);
                Ok(NodeView {
                    slot: self.slot,
                    ptr: NonNull::new(replacement_ptr).unwrap(),
                    id: self.id,
                })
            }
            Err(_) => {
                let failed_value = unsafe { Box::from_raw(replacement_ptr) };
                drop(failed_value);

                Err(())
            }
        }
    }
}

impl<'a, K, V> Deref for NodeView<'a, K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    type Target = Node<K, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

#[derive(Clone)]
pub struct ConcurrentMap<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    ebr: Ebr<Box<Node<K, V>>>,
    idgen: Arc<AtomicU64>,
    inner: Arc<Inner<K, V>>,
    free_ids: ConcurrentStack<u64>,
}

impl<K, V> Default for ConcurrentMap<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    fn default() -> ConcurrentMap<K, V> {
        let initial_root_id = 0;
        let initial_leaf_id = 1;

        let table = PageTable::<AtomicPtr<Node<K, V>>>::default();

        let mut root_node = Node::<K, V>::new_root(Arc::default());
        root_node.index.insert(Arc::default(), initial_leaf_id);
        let leaf_node = Node::<K, V>::new_leaf(Arc::default());

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
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    root_id: AtomicU64,
    table: PageTable<AtomicPtr<Node<K, V>>>,
}

impl<K, V> Drop for Inner<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    fn drop(&mut self) {
        let mut ebr = Ebr::default();
        let mut guard = ebr.pin();

        let root_id = self.root_id.load(Ordering::Acquire);
        let mut cursor = self.view_for_id(root_id, &mut guard);

        let mut lhs_chain = vec![];
        loop {
            lhs_chain.push(cursor.id);
            if cursor.is_leaf {
                break;
            }
            let child_id = *cursor.index.iter().next().unwrap().1;

            cursor = self.view_for_id(child_id, &mut guard);
        }

        for lhs_id in lhs_chain {
            let mut next_opt = Some(lhs_id);
            while let Some(next) = next_opt {
                let cursor = self.view_for_id(next, &mut guard);
                next_opt = cursor.next;
                let node_box = unsafe { Box::from_raw(cursor.ptr.as_ptr()) };
                drop(node_box);
            }
        }
    }
}

impl<K, V> ConcurrentMap<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
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
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    inner: &'a Inner<K, V>,
    guard: Guard<'a, Box<Node<K, V>>>,
    lo: Bound<Arc<K>>,
    hi: Bound<Arc<K>>,
    current: NodeView<'a, K, V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    Bound<Arc<K>>: PartialOrd,
    V: 'static + std::fmt::Debug + Send + Sync,
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
    K: 'static + Clone + std::fmt::Debug + Default + Ord + Send + Sync,
    Bound<Arc<K>>: PartialOrd,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}
*/

impl<K, V> Inner<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    fn view_for_id<'a>(
        &'a self,
        id: Id,
        _guard: &mut Guard<'a, Box<Node<K, V>>>,
    ) -> NodeView<'a, K, V> {
        let slot = self.table.get(id);
        let ptr = NonNull::new(slot.load(Ordering::Acquire)).unwrap();

        NodeView { ptr, slot, id }
    }

    fn leftmost_leaf<'a>(&'a self, guard: &mut Guard<'a, Box<Node<K, V>>>) -> NodeView<'a, K, V> {
        let root_id = self.root_id.load(Ordering::Acquire);

        let mut cursor = self.view_for_id(root_id, guard);

        let mut lhs_chain = vec![];
        loop {
            lhs_chain.push(cursor.id);
            if cursor.is_leaf {
                return cursor;
            }
            let child_id = *cursor.index.iter().next().unwrap().1;

            cursor = self.view_for_id(child_id, guard);
        }
    }

    fn leaf_for_key<'a>(
        &'a self,
        key: &K,
        idgen: &AtomicU64,
        free_ids: &mut ConcurrentStack<u64>,
        guard: &mut Guard<'a, Box<Node<K, V>>>,
    ) -> NodeView<'a, K, V> {
        let root_id = self.root_id.load(Ordering::Acquire);

        let mut parent_cursor_opt: Option<NodeView<'a, K, V>> = None;
        let mut cursor = self.view_for_id(root_id, guard);

        const MAX_LOOPS: usize = 8;
        for i in 0.. {
            // println!("cursor is: {} -> {:?}", cursor.id, *cursor);
            if i >= MAX_LOOPS {
                panic!("exceeded max loops");
            }
            assert!(key >= &*cursor.lo);
            if let Some(hi) = &cursor.hi {
                if key >= hi {
                    // go right to the tree sibling
                    let next = cursor.next.unwrap();
                    let rhs = self.view_for_id(next, guard);

                    if let Some(ref mut parent_cursor) = parent_cursor_opt {
                        let mut parent_clone: Box<Node<K, V>> = Box::new((*parent_cursor).clone());
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
                    } else {
                        // root hoist
                        let new_id = free_ids
                            .pop()
                            .unwrap_or_else(|| idgen.fetch_add(1, Ordering::Relaxed));

                        let mut new_root_node = Node::<K, V>::new_root(Arc::default());
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
                            free_ids.push(new_id);
                        }
                    }

                    cursor = rhs;
                    continue;
                }
            }

            if cursor.is_leaf {
                break;
            }

            // go down the tree
            let child_id = *cursor.index.range::<K, _>(..=key).next_back().unwrap().1;
            parent_cursor_opt = Some(cursor);
            cursor = self.view_for_id(child_id, guard);
        }
        // println!("final leaf is: {:?}", *cursor);

        cursor
    }
}

#[derive(Debug)]
struct Node<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    lo: Arc<K>,
    hi: Option<Arc<K>>,
    next: Option<Id>,
    is_leaf: bool,
    leaf: BTreeMap<Arc<K>, Arc<V>>,
    index: BTreeMap<Arc<K>, Id>,
}

impl<K, V> Clone for Node<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    fn clone(&self) -> Node<K, V> {
        Node {
            lo: self.lo.clone(),
            hi: self.hi.clone(),
            next: self.next,
            is_leaf: self.is_leaf,
            leaf: self.leaf.clone(),
            index: self.index.clone(),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: 'static + std::fmt::Debug + Default + Ord + Send + Sync,
    V: 'static + std::fmt::Debug + Send + Sync,
{
    fn new_root(lo: Arc<K>) -> Box<Node<K, V>> {
        Box::new(Node {
            lo,
            hi: None,
            next: None,
            is_leaf: false,
            leaf: Default::default(),
            index: Default::default(),
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
        })
    }

    fn get(&self, key: &K) -> Option<Arc<V>> {
        self.leaf.get(key).cloned()
    }

    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        self.leaf.remove(key)
    }

    fn insert(&mut self, key: Arc<K>, value: Arc<V>) -> Option<Arc<V>> {
        self.leaf.insert(key, value)
    }

    fn should_split(&self) -> bool {
        if self.is_leaf {
            self.leaf.len() >= SPLIT_SIZE
        } else {
            self.index.len() >= SPLIT_SIZE
        }
    }

    fn split(mut self, new_id: u64) -> (Node<K, V>, Node<K, V>) {
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
        };

        assert_eq!(self.hi, Some(split_point));

        (self, rhs)
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
