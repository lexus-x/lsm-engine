use crate::{Entry, Key, Value, Sequence};
use rand::Rng;

/// Arena-based skip-list node
struct Node {
    entry: Entry,
    forward: Vec<usize>, // indices into arena
}

/// Skip-list based memtable for in-memory sorted key-value storage
pub struct Memtable {
    arena: Vec<Node>,
    head: usize,
    max_level: usize,
    current_level: usize,
    len: usize,
    size_bytes: usize,
    max_size_bytes: usize,
}

const NIL: usize = usize::MAX;

impl Memtable {
    pub fn new(max_size_bytes: usize) -> Self {
        let max_level = 16;
        let head = 0;
        let mut arena = Vec::with_capacity(1024);
        arena.push(Node {
            entry: Entry::new(Vec::new(), None, 0),
            forward: vec![NIL; max_level],
        });
        Self {
            arena,
            head,
            max_level,
            current_level: 0,
            len: 0,
            size_bytes: 0,
            max_size_bytes,
        }
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while level < self.max_level - 1 && rng.gen::<f64>() < 0.5 {
            level += 1;
        }
        level
    }

    /// Find predecessors at each level for a given key.
    /// We want the rightmost node at each level whose key < target key,
    /// or whose key == target key but sequence >= target sequence
    /// (so we insert after it).
    fn find_predecessors(&self, key: &[u8]) -> Vec<usize> {
        let mut update = vec![0usize; self.max_level];
        let mut current = self.head;
        
        for i in (0..=self.current_level).rev() {
            while self.arena[current].forward[i] != NIL {
                let next_idx = self.arena[current].forward[i];
                let next_key = &self.arena[next_idx].entry.key;
                
                if next_key.as_slice() < key {
                    current = next_idx;
                } else {
                    break;
                }
            }
            update[i] = current;
        }
        
        update
    }

    pub fn put(&mut self, key: Key, value: Option<Value>, sequence: Sequence) {
        let update = self.find_predecessors(&key);
        
        // Check if the immediate successor has the same key; if so we still insert
        // (multiple versions of same key coexist, newest sequence wins on read)
        let new_level = self.random_level();
        
        if new_level > self.current_level {
            self.current_level = new_level;
        }
        
        let entry_size = key.len() + value.as_ref().map_or(0, |v| v.len()) + 8;
        
        let new_idx = self.arena.len();
        let mut forward = vec![NIL; self.max_level];
        
        for i in 0..=new_level {
            forward[i] = self.arena[update[i]].forward[i];
            self.arena[update[i]].forward[i] = new_idx;
        }
        
        self.arena.push(Node {
            entry: Entry::new(key, value, sequence),
            forward,
        });
        
        self.len += 1;
        self.size_bytes += entry_size;
    }

    pub fn get(&self, key: &[u8]) -> Option<Entry> {
        let mut current = self.head;
        
        for i in (0..=self.current_level).rev() {
            while self.arena[current].forward[i] != NIL {
                let next_idx = self.arena[current].forward[i];
                let next_key = &self.arena[next_idx].entry.key;
                
                match next_key.as_slice().cmp(key) {
                    std::cmp::Ordering::Less => current = next_idx,
                    _ => break,
                }
            }
        }
        
        // Now advance one step in level 0
        if self.arena[current].forward[0] != NIL {
            let next_idx = self.arena[current].forward[0];
            if self.arena[next_idx].entry.key.as_slice() == key {
                // May have multiple versions; walk forward to find the one with highest sequence
                let mut best = self.arena[next_idx].entry.clone();
                let mut walk = self.arena[next_idx].forward[0];
                while walk != NIL && self.arena[walk].entry.key.as_slice() == key {
                    if self.arena[walk].entry.sequence > best.sequence {
                        best = self.arena[walk].entry.clone();
                    }
                    walk = self.arena[walk].forward[0];
                }
                return Some(best);
            }
        }
        
        None
    }

    pub fn range_scan(&self, start: &[u8], end: &[u8]) -> Vec<Entry> {
        let mut results = Vec::new();
        let mut current = self.head;
        
        // Navigate to the start
        for i in (0..=self.current_level).rev() {
            while self.arena[current].forward[i] != NIL {
                let next_idx = self.arena[current].forward[i];
                if self.arena[next_idx].entry.key.as_slice() < start {
                    current = next_idx;
                } else {
                    break;
                }
            }
        }
        
        // Now scan forward
        let mut idx = self.arena[current].forward[0];
        
        while idx != NIL {
            let node = &self.arena[idx];
            if node.entry.key.as_slice() > end {
                break;
            }
            results.push(node.entry.clone());
            idx = node.forward[0];
        }
        
        results
    }

    pub fn iter(&self) -> MemtableIterator<'_> {
        MemtableIterator {
            arena: &self.arena,
            current: self.arena[self.head].forward[0],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn clear(&mut self) {
        self.arena.clear();
        self.arena.push(Node {
            entry: Entry::new(Vec::new(), None, 0),
            forward: vec![NIL; self.max_level],
        });
        self.current_level = 0;
        self.len = 0;
        self.size_bytes = 0;
    }
}

pub struct MemtableIterator<'a> {
    arena: &'a Vec<Node>,
    current: usize,
}

impl<'a> Iterator for MemtableIterator<'a> {
    type Item = &'a Entry;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == NIL {
            return None;
        }
        
        let node = &self.arena[self.current];
        let entry = &node.entry;
        self.current = node.forward[0];
        Some(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memtable_put_get() {
        let mut mt = Memtable::new(1024 * 1024);
        mt.put(b"key1".to_vec(), Some(b"val1".to_vec()), 1);
        mt.put(b"key2".to_vec(), Some(b"val2".to_vec()), 2);
        
        let entry = mt.get(b"key1").unwrap();
        assert_eq!(entry.value.as_ref().unwrap(), b"val1");
        
        let entry = mt.get(b"key2").unwrap();
        assert_eq!(entry.value.as_ref().unwrap(), b"val2");
        
        assert!(mt.get(b"key3").is_none());
    }

    #[test]
    fn test_memtable_tombstone() {
        let mut mt = Memtable::new(1024 * 1024);
        mt.put(b"key1".to_vec(), Some(b"val1".to_vec()), 1);
        mt.put(b"key1".to_vec(), None, 2); // tombstone
        
        let entry = mt.get(b"key1").unwrap();
        assert!(entry.is_tombstone());
    }

    #[test]
    fn test_memtable_range_scan() {
        let mut mt = Memtable::new(1024 * 1024);
        for i in 0..10u64 {
            let key = format!("key{:02}", i);
            let val = format!("val{}", i);
            mt.put(key.into_bytes(), Some(val.into_bytes()), i);
        }
        
        let results = mt.range_scan(b"key03", b"key07");
        assert_eq!(results.len(), 5); // key03..key07 inclusive
    }

    #[test]
    fn test_memtable_overwrite() {
        let mut mt = Memtable::new(1024 * 1024);
        mt.put(b"key1".to_vec(), Some(b"v1".to_vec()), 1);
        mt.put(b"key1".to_vec(), Some(b"v2".to_vec()), 2);
        
        let entry = mt.get(b"key1").unwrap();
        assert_eq!(entry.value.as_ref().unwrap(), b"v2");
        assert_eq!(entry.sequence, 2);
    }
}
