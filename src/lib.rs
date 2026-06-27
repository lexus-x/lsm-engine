pub mod engine;
pub mod index;
pub mod benchmark;

/// Common types used across the engine
pub type Key = Vec<u8>;
pub type Value = Vec<u8>;
pub type Sequence = u64;

/// Result of a point lookup
#[derive(Debug, Clone)]
pub enum LookupResult {
    Found(Vec<u8>),
    NotFound,
    Tombstone,
}

/// Index type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexType {
    Bloom,
    Learned,
}

/// Configuration for the LSM engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub data_dir: String,
    pub memtable_size_bytes: usize,
    pub block_size_bytes: usize,
    pub index_type: IndexType,
    pub sstable_level0_max: usize,
    pub compaction_size_ratio: usize,
    pub bloom_bits_per_element: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            data_dir: "./lsm_data".to_string(),
            memtable_size_bytes: 4 * 1024 * 1024, // 4MB
            block_size_bytes: 4096,
            index_type: IndexType::Bloom,
            sstable_level0_max: 4,
            compaction_size_ratio: 10,
            bloom_bits_per_element: 10.0,
        }
    }
}

/// Entry in the memtable / WAL
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub key: Key,
    pub value: Option<Value>, // None = tombstone
    pub sequence: Sequence,
}

impl Entry {
    pub fn new(key: Key, value: Option<Value>, sequence: Sequence) -> Self {
        Self { key, value, sequence }
    }

    pub fn is_tombstone(&self) -> bool {
        self.value.is_none()
    }

    pub fn size_bytes(&self) -> usize {
        self.key.len() + self.value.as_ref().map_or(0, |v| v.len()) + 8
    }
}
