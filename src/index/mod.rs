pub mod bloom;
pub mod learned;

/// Hint returned by an index about where to look for a key
#[derive(Debug, Clone)]
pub enum LookupHint {
    /// Key is definitely not in the SSTable
    NotFound,
    /// Search within this byte offset range
    SearchRange { start: usize, end: usize },
}

/// Common trait for SSTable indexes (bloom filter or learned index)
pub trait Index: Send + Sync {
    /// Insert a key with its byte offset in the SSTable
    fn insert(&mut self, key: &[u8], offset: usize);
    
    /// Finalize the index after all insertions
    fn build(&mut self);
    
    /// Get a lookup hint for the given key
    fn lookup_hint(&self, key: &[u8]) -> LookupHint;
    
    /// Serialize the index to bytes
    fn serialize(&self) -> Vec<u8>;
    
    /// Return the type name
    fn index_type_name(&self) -> &str;
}

// Re-export concrete types
pub use crate::IndexType;
pub use bloom::BloomFilterIndex;
pub use learned::LearnedIndexModel as LearnedIndex;
