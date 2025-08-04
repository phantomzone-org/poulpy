use std::hash::Hash;

use fnv::FnvHashMap;

pub struct Map<K, V>(pub FnvHashMap<K, V>);

impl<K: Eq + Hash, V> Map<K, V> {
    pub fn new() -> Self {
        Self {
            0: FnvHashMap::<K, V>::default(),
        }
    }

    pub fn insert(&mut self, k: K, data: V) -> Option<V> {
        self.0.insert(k, data)
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.0.get(k)
    }
}
