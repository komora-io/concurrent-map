use std::sync::Arc;

// TODO make it a const array
#[derive(Debug)]
pub struct ArrayMap<K: Ord, V> {
    inner: Vec<(Arc<K>, V)>,
}

impl<K: Ord, V: Clone> Clone for ArrayMap<K, V> {
    fn clone(&self) -> Self {
        ArrayMap {
            inner: self.iter().cloned().collect(),
        }
    }
}

impl<K: Ord, V> Default for ArrayMap<K, V> {
    fn default() -> Self {
        ArrayMap { inner: vec![] }
    }
}

impl<K: Ord, V> ArrayMap<K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Ok(index) = self.inner.binary_search_by_key(&key, |(k, _v)| k) {
            Some(unsafe { &self.inner.get_unchecked(index).1 })
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: Arc<K>, value: V) -> Option<V> {
        match self.inner.binary_search_by_key(&&key, |(k, _v)| k) {
            Ok(index) => {
                let slot = unsafe { &mut self.inner.get_unchecked_mut(index).1 };
                Some(std::mem::replace(slot, value))
            }
            Err(index) => {
                self.inner.insert(index, (key, value));
                None
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Ok(index) = self.inner.binary_search_by_key(&key, |(k, _v)| k) {
            Some(self.inner.remove(index).1)
        } else {
            None
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.binary_search_by_key(&key, |(k, _v)| k).is_ok()
    }

    pub fn is_leftmost(&self, key: &K) -> bool {
        &*self.inner[0].0 == key
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &(Arc<K>, V)> {
        self.inner.iter()
    }

    /// Similar to `BTreeMap::split_off`
    /// TODO just do it by index and also return split key
    pub fn split_off(&mut self, key: &K) -> Self {
        let index = self.inner.binary_search_by_key(&key, |(k, _v)| k).unwrap();
        let rhs = self.inner.split_off(index);
        ArrayMap { inner: rhs }
    }

    pub fn get_index(&self, index: usize) -> Option<&(Arc<K>, V)> {
        self.inner.get(index)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

pub(crate) fn binary_search_lub<K: Ord, V>(key: &K, s: &[(Arc<K>, V)]) -> Option<usize> {
    match s.binary_search_by_key(&key, |(k, _v)| k) {
        Ok(i) => Some(i),
        Err(0) => None,
        Err(i) => Some(i - 1),
    }
}
