use std::mem::MaybeUninit;
use std::sync::Arc;

const MAX_LEN: usize = super::SPLIT_SIZE;

// TODO make it a const array
#[derive(Debug)]
pub struct ArrayMap<K: Ord, V> {
    inner: [MaybeUninit<(Arc<K>, V)>; 16],
    len: u8,
}

impl<K: Ord, V> Drop for ArrayMap<K, V> {
    fn drop(&mut self) {
        for i in 0..self.len() {
            let ptr = self.inner[i].as_mut_ptr();
            unsafe {
                std::ptr::drop_in_place(ptr);
            }
        }
    }
}

impl<K: Ord, V: Clone> Clone for ArrayMap<K, V> {
    fn clone(&self) -> Self {
        let mut inner: [MaybeUninit<(Arc<K>, V)>; 16] =
            core::array::from_fn(|_i| MaybeUninit::uninit());

        for (i, item) in self.iter().cloned().enumerate() {
            inner[i].write(item);
        }

        ArrayMap {
            inner,
            len: self.len,
        }
    }
}

impl<K: Ord, V> Default for ArrayMap<K, V> {
    fn default() -> Self {
        ArrayMap {
            inner: core::array::from_fn(|_i| MaybeUninit::uninit()),
            len: 0,
        }
    }
}

impl<K: Ord, V> ArrayMap<K, V> {
    fn binary_search(&self, key: &K) -> Result<usize, usize> {
        self.inner[..self.len()]
            .binary_search_by_key(&key, |slot| unsafe { &*slot.assume_init_ref().0 })
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if let Ok(index) = self.binary_search(&key) {
            Some(unsafe { &self.inner.get_unchecked(index).assume_init_ref().1 })
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: Arc<K>, value: V) -> Option<V> {
        match self.binary_search(&*key) {
            Ok(index) => {
                let slot = unsafe { &mut self.inner.get_unchecked_mut(index).assume_init_mut().1 };
                Some(std::mem::replace(slot, value))
            }
            Err(index) => {
                assert!(self.len() < MAX_LEN);

                unsafe {
                    if index < self.len() {
                        let src = self.inner.get_unchecked(index).as_ptr();
                        let dst = self.inner.get_unchecked_mut(index + 1).as_mut_ptr();

                        std::ptr::copy(src, dst, self.len() - index);
                    }

                    self.len += 1;

                    self.inner.get_unchecked_mut(index).write((key, value));
                }
                None
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Ok(index) = self.binary_search(&key) {
            unsafe {
                let ret = std::ptr::read(self.inner.get_unchecked(index).as_ptr()).1;

                if index + 1 < self.len() {
                    let src = self.inner.get_unchecked(index + 1).as_ptr();
                    let dst = self.inner.get_unchecked_mut(index).as_mut_ptr();

                    std::ptr::copy(src, dst, self.len() - index);
                }

                self.len -= 1;

                Some(ret)
            }
        } else {
            None
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.binary_search(&key).is_ok()
    }

    pub fn is_leftmost(&self, key: &K) -> bool {
        assert!(self.len > 0);
        unsafe { &*self.inner[0].assume_init_ref().0 == key }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &(Arc<K>, V)> {
        self.inner[..self.len()]
            .iter()
            .map(|slot| unsafe { slot.assume_init_ref() })
    }

    // returns the split key and the right hand split
    pub fn split_off(&mut self, split_idx: usize) -> (Arc<K>, Self) {
        assert!(split_idx < self.len());

        let split_key = unsafe { self.inner[split_idx].assume_init_ref().0.clone() };

        let mut rhs = Self::default();

        for i in split_idx..self.len() {
            let src = self.inner[i].as_ptr();
            let dst = rhs.inner[i - split_idx].as_mut_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }
        }

        rhs.len = self.len - split_idx as u8;
        self.len = u8::try_from(split_idx).unwrap();

        (split_key, rhs)
    }

    pub fn get_index(&self, index: usize) -> Option<&(Arc<K>, V)> {
        if index < self.len() {
            Some(unsafe { self.inner.get_unchecked(index).assume_init_ref() })
        } else {
            None
        }
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }
}

impl<K: Ord, V: Copy> ArrayMap<K, V> {
    pub fn index_next_child(&self, key: &K) -> V {
        // binary_search_lub
        let index = match self.binary_search(key) {
            Ok(i) => i,
            Err(0) => unreachable!(),
            Err(i) => i - 1,
        };
        self.get_index(index).as_ref().unwrap().1
    }
}
