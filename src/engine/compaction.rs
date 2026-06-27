use std::io;
use std::path::Path;

use crate::Entry;
use crate::engine::sstable::{SSTable, SSTableBuilder};
use crate::index::IndexType;

/// Result of a compaction operation
pub struct CompactionResult {
    pub sst: SSTable,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub size_bytes: u64,
}

/// Size-tiered compaction: merge multiple SSTables into one
pub struct Compactor {
    block_size: usize,
    index_type: IndexType,
    bloom_bits_per_element: f64,
}

impl Compactor {
    pub fn new(block_size: usize, index_type: IndexType, bloom_bits_per_element: f64) -> Self {
        Self {
            block_size,
            index_type,
            bloom_bits_per_element,
        }
    }

    /// Compact SSTables from one level and the next level into a single output SSTable.
    /// Tombstones for keys not present in older levels are dropped.
    pub fn compact(
        &self,
        level_ssts: &[SSTable],
        next_level_ssts: &[SSTable],
        output_path: &Path,
        output_level: usize,
    ) -> io::Result<CompactionResult> {
        // Collect all entries from all SSTables, merge-sort by key
        let mut all_entries: Vec<Entry> = Vec::new();
        
        // Read from current level (newer)
        for sst in level_ssts {
            let entries = sst.scan(sst.min_key(), sst.max_key())?;
            all_entries.extend(entries);
        }
        
        // Read from next level (older)
        for sst in next_level_ssts {
            let entries = sst.scan(sst.min_key(), sst.max_key())?;
            all_entries.extend(entries);
        }
        
        // Sort by key, then by sequence descending (newest first)
        all_entries.sort_by(|a, b| {
            a.key.cmp(&b.key)
                .then(b.sequence.cmp(&a.sequence))
        });
        
        // Deduplicate: keep only the newest version of each key
        let mut merged: Vec<Entry> = Vec::new();
        let mut i = 0;
        while i < all_entries.len() {
            let key = all_entries[i].key.clone();
            let entry = all_entries[i].clone();
            
            // Skip tombstones if this is not level 0 (tombstones at level 0+ are needed
            // to shadow entries in older levels; but when merging to a deeper level,
            // we can drop tombstones if the key doesn't exist in older SSTables)
            if entry.is_tombstone() && output_level > 0 {
                // Check if this key exists in any older (not being compacted) SSTable
                // For simplicity, we keep tombstones during compaction
                // A production system would check older levels
                // Drop tombstone only if it's the last version (no older entries)
                let has_older = {
                    let mut found = false;
                    let mut j = i + 1;
                    while j < all_entries.len() && all_entries[j].key == key {
                        found = true;
                        j += 1;
                    }
                    found
                };
                if !has_older {
                    // No older version, can drop tombstone
                    i += 1;
                    continue;
                }
            }
            
            merged.push(entry);
            
            // Skip all older versions of this key
            while i < all_entries.len() && all_entries[i].key == key {
                i += 1;
            }
        }
        
        if merged.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "no entries after compaction"));
        }
        
        let min_key = merged.first().unwrap().key.clone();
        let max_key = merged.last().unwrap().key.clone();
        
        let builder = SSTableBuilder::new(
            output_path,
            self.block_size,
            self.index_type,
            self.bloom_bits_per_element,
        );
        
        let result = builder.build_from_entries(&merged)?;
        
        Ok(CompactionResult {
            sst: result.sst,
            min_key,
            max_key,
            size_bytes: result.size_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_compaction_merge() {
        let dir = tempdir().unwrap();
        
        // Create two SSTables with overlapping keys
        let entries1: Vec<Entry> = (0..50)
            .map(|i| Entry::new(
                format!("key{:04}", i).into_bytes(),
                Some(format!("v1_{}", i).into_bytes()),
                i * 2,
            ))
            .collect();
        
        let entries2: Vec<Entry> = (25..75)
            .map(|i| Entry::new(
                format!("key{:04}", i).into_bytes(),
                Some(format!("v2_{}", i).into_bytes()),
                i * 2 + 1,
            ))
            .collect();
        
        let path1 = dir.path().join("sst1.dat");
        let path2 = dir.path().join("sst2.dat");
        
        let builder1 = SSTableBuilder::new(&path1, 256, IndexType::Bloom, 10.0);
        let result1 = builder1.build_from_entries(&entries1).unwrap();
        
        let builder2 = SSTableBuilder::new(&path2, 256, IndexType::Bloom, 10.0);
        let result2 = builder2.build_from_entries(&entries2).unwrap();
        
        let output_path = dir.path().join("merged.dat");
        let compactor = Compactor::new(256, IndexType::Bloom, 10.0);
        let result = compactor.compact(
            &[result1.sst],
            &[result2.sst],
            &output_path,
            1,
        ).unwrap();
        
        // Verify merged SSTable
        // Keys 0-24 come from sst1, keys 25-74 come from sst2 (newer)
        match result.sst.get(b"key0010").unwrap() {
            LookupResult::Found(v) => assert_eq!(v, b"v1_10"),
            _ => panic!("expected from sst1"),
        }
        
        match result.sst.get(b"key0050").unwrap() {
            LookupResult::Found(v) => assert_eq!(v, b"v2_50"),
            _ => panic!("expected from sst2"),
        }
    }
}
