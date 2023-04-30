use std::marker::PhantomData;

use serde::de::{Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};

use crate::{ConcurrentMap, Minimum};

impl<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> Serialize
    for ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Serialize + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Serialize + Clone + Send + Sync,
{
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(self.len()))?;
        //let mut map = s.serialize_map(None)?;
        for (k, v) in self.iter() {
            map.serialize_entry(&k, &v)?;
        }
        map.end()
    }
}

struct ConcurrentMapVisitor<K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> {
    pd: PhantomData<(K, V)>,
}

impl<'de, K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> Visitor<'de>
    for ConcurrentMapVisitor<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Deserialize<'de> + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Deserialize<'de> + Clone + Send + Sync,
{
    type Value = ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a StackMap<IVec, IVec, 1024>")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let map = ConcurrentMap::default();

        while let Some((key, value)) = access.next_entry()? {
            map.insert(key, value);
        }

        Ok(map)
    }
}

impl<'de, K, V, const FANOUT: usize, const LOCAL_GC_BUFFER_SIZE: usize> Deserialize<'de>
    for ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>
where
    K: 'static + Deserialize<'de> + Clone + Minimum + Ord + Send + Sync,
    V: 'static + Deserialize<'de> + Clone + Send + Sync,
{
    fn deserialize<D>(d: D) -> Result<ConcurrentMap<K, V, FANOUT, LOCAL_GC_BUFFER_SIZE>, D::Error>
    where
        D: Deserializer<'de>,
    {
        d.deserialize_map(ConcurrentMapVisitor { pd: PhantomData })
    }
}
