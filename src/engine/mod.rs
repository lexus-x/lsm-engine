pub mod memtable;
pub mod sstable;
pub mod wal;
pub mod compaction;
pub mod manifest;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::index::IndexType;
use crate::{Entry, EngineConfig, Key, LookupResult, Value};

/// A complete SSTable entry for the engine to manage
#[derive(Debug, Clone)]
pub struct SSTableMeta {
    pub id: u64,
    pub level: usize,
    pub path: PathBuf,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub size_bytes: u64,
    pub index_type: IndexType,
}

/// Main engine struct tying all components together
pub struct Engine {
    config: EngineConfig,
    memtable: memtable::Memtable,
    wal: wal::Wal,
    sstables: Arc<Mutex<Vec<Vec<sstable::SSTable>>>>, // levels -> sstables
    manifest: manifest::Manifest,
    next_sequence: AtomicU64,
    next_sst_id: AtomicU64,
    flush_count: AtomicU64,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self, String> {
        std::fs::create_dir_all(&config.data_dir).map_err(|e| e.to_string())?;
        
        let data_dir = PathBuf::from(&config.data_dir);
        let wal_path = data_dir.join("wal.log");
        let manifest_path = data_dir.join("manifest.json");
        
        let manifest = manifest::Manifest::load_or_create(&manifest_path);
        let wal = wal::Wal::open(&wal_path).map_err(|e| e.to_string())?;
        
        // Reconstruct SSTables from manifest
        let max_level = manifest.max_level();
        let mut levels: Vec<Vec<sstable::SSTable>> = (0..=max_level).map(|_| Vec::new()).collect();
        for entry in manifest.entries() {
            let sst_path = data_dir.join(format!("sst_{}_{}.dat", entry.level, entry.id));
            if sst_path.exists() {
                match sstable::SSTable::open(&sst_path, entry.index_type) {
                    Ok(sst) => levels[entry.level].push(sst),
                    Err(_) => continue,
                }
            }
        }
        
        // Replay WAL
        let memtable = memtable::Memtable::new(config.memtable_size_bytes);
        let mut engine = Self {
            config,
            memtable,
            wal,
            sstables: Arc::new(Mutex::new(levels)),
            manifest,
            next_sequence: AtomicU64::new(1),
            next_sst_id: AtomicU64::new(1),
            flush_count: AtomicU64::new(0),
        };
        
        engine.replay_wal()?;
        Ok(engine)
    }
    
    fn replay_wal(&mut self) -> Result<(), String> {
        let entries = self.wal.replay().map_err(|e| e.to_string())?;
        let mut max_seq = 0u64;
        for entry in entries {
            if entry.sequence > max_seq {
                max_seq = entry.sequence;
            }
            self.memtable.put(entry.key, entry.value, entry.sequence);
        }
        if max_seq > 0 {
            self.next_sequence.store(max_seq + 1, Ordering::SeqCst);
        }
        Ok(())
    }
    
    pub fn put(&mut self, key: Key, value: Value) -> Result<(), String> {
        let seq = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let entry = Entry::new(key.clone(), Some(value.clone()), seq);
        self.wal.append(&entry).map_err(|e| e.to_string())?;
        self.memtable.put(key, Some(value), seq);
        
        if self.memtable.size_bytes() >= self.config.memtable_size_bytes {
            self.flush_memtable()?;
        }
        Ok(())
    }
    
    pub fn get(&self, key: &[u8]) -> Result<LookupResult, String> {
        // Check memtable first
        if let Some(entry) = self.memtable.get(key) {
            return if entry.is_tombstone() {
                Ok(LookupResult::Tombstone)
            } else {
                Ok(LookupResult::Found(entry.value.clone().unwrap()))
            };
        }
        
        // Search SSTables from newest to oldest
        let levels = self.sstables.lock().map_err(|e| e.to_string())?;
        for level in levels.iter() {
            for sst in level.iter().rev() {
                if key < sst.min_key() || key > sst.max_key() {
                    continue;
                }
                match sst.get(key).map_err(|e| e.to_string())? {
                    LookupResult::Found(v) => return Ok(LookupResult::Found(v)),
                    LookupResult::Tombstone => return Ok(LookupResult::Tombstone),
                    LookupResult::NotFound => continue,
                }
            }
        }
        
        Ok(LookupResult::NotFound)
    }
    
    pub fn delete(&mut self, key: Key) -> Result<(), String> {
        let seq = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let entry = Entry::new(key.clone(), None, seq);
        self.wal.append(&entry).map_err(|e| e.to_string())?;
        self.memtable.put(key, None, seq);
        
        if self.memtable.size_bytes() >= self.config.memtable_size_bytes {
            self.flush_memtable()?;
        }
        Ok(())
    }
    
    pub fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<Entry>, String> {
        let mut results = Vec::new();
        
        // Collect from memtable
        for entry in self.memtable.range_scan(start, end) {
            results.push(entry);
        }
        
        // Collect from SSTables
        let levels = self.sstables.lock().map_err(|e| e.to_string())?;
        for level in levels.iter() {
            for sst in level.iter().rev() {
                if sst.max_key() < start || sst.min_key() > end {
                    continue;
                }
                let sst_entries = sst.scan(start, end).map_err(|e| e.to_string())?;
                results.extend(sst_entries);
            }
        }
        
        // Sort by key then by sequence (descending) to get newest first
        results.sort_by(|a, b| {
            a.key.cmp(&b.key)
                .then(b.sequence.cmp(&a.sequence))
        });
        
        // Deduplicate, keeping only the newest entry per key
        let mut deduped = Vec::new();
        let mut i = 0;
        while i < results.len() {
            let key = results[i].key.clone();
            // Skip tombstones
            if !results[i].is_tombstone() {
                deduped.push(results[i].clone());
            }
            // Skip older entries for the same key
            while i < results.len() && results[i].key == key {
                i += 1;
            }
        }
        
        Ok(deduped)
    }
    
    fn flush_memtable(&mut self) -> Result<(), String> {
        if self.memtable.is_empty() {
            return Ok(());
        }
        
        let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
        let data_dir = PathBuf::from(&self.config.data_dir);
        let sst_path = data_dir.join(format!("sst_0_{}.dat", sst_id));
        
        let entries: Vec<Entry> = self.memtable.iter().cloned().collect();
        
        let builder = sstable::SSTableBuilder::new(
            &sst_path,
            self.config.block_size_bytes,
            self.config.index_type,
            self.config.bloom_bits_per_element,
        );
        
        let result = builder.build_from_entries(&entries).map_err(|e| e.to_string())?;
        
        let min_key = result.min_key.clone();
        let max_key = result.max_key.clone();
        
        let meta = SSTableMeta {
            id: sst_id,
            level: 0,
            path: sst_path.clone(),
            min_key: min_key.clone(),
            max_key: max_key.clone(),
            size_bytes: result.size_bytes,
            index_type: self.config.index_type,
        };
        let sst = result.sst;
        
        {
            let mut levels = self.sstables.lock().map_err(|e| e.to_string())?;
            if levels.is_empty() {
                levels.push(Vec::new());
            }
            levels[0].push(sst);
        }
        
        self.manifest.add(meta);
        self.manifest.persist().map_err(|e| e.to_string())?;
        
        // Clear memtable and WAL
        self.memtable.clear();
        self.wal.truncate().map_err(|e| e.to_string())?;
        
        self.flush_count.fetch_add(1, Ordering::SeqCst);
        
        // Check if compaction is needed
        self.maybe_compact()?;
        
        Ok(())
    }
    
    fn maybe_compact(&mut self) -> Result<(), String> {
        let levels = self.sstables.lock().map_err(|e| e.to_string())?;
        if levels.is_empty() {
            return Ok(());
        }
        
        // Check level 0
        if levels[0].len() >= self.config.sstable_level0_max {
            drop(levels);
            self.compact_level(0)?;
        }
        
        Ok(())
    }
    
    fn compact_level(&mut self, level: usize) -> Result<(), String> {
        let data_dir = PathBuf::from(&self.config.data_dir);
        let sstables_to_compact;
        let next_level_sstables;
        
        {
            let mut levels = self.sstables.lock().map_err(|e| e.to_string())?;
            while levels.len() <= level + 1 {
                levels.push(Vec::new());
            }
            
            sstables_to_compact = std::mem::take(&mut levels[level]);
            next_level_sstables = std::mem::take(&mut levels[level + 1]);
            
            if sstables_to_compact.is_empty() {
                return Ok(());
            }
            
            let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
            let output_path = data_dir.join(format!("sst_{}_{}.dat", level + 1, sst_id));
            
            let compactor = compaction::Compactor::new(
                self.config.block_size_bytes,
                self.config.index_type,
                self.config.bloom_bits_per_element,
            );
            
            let result = compactor.compact(
                &sstables_to_compact,
                &next_level_sstables,
                &output_path,
                level + 1,
            ).map_err(|e| e.to_string())?;
            
            // Remove old SSTables
            for sst in &sstables_to_compact {
                let _ = std::fs::remove_file(sst.path());
            }
            for sst in &next_level_sstables {
                let _ = std::fs::remove_file(sst.path());
            }
            
            // Update manifest
            for sst in &sstables_to_compact {
                self.manifest.remove(sst.id());
            }
            for sst in &next_level_sstables {
                self.manifest.remove(sst.id());
            }
            
            levels[level].clear();
            levels[level + 1] = vec![result.sst];
            
            let meta = SSTableMeta {
                id: sst_id,
                level: level + 1,
                path: output_path,
                min_key: result.min_key,
                max_key: result.max_key,
                size_bytes: result.size_bytes,
                index_type: self.config.index_type,
            };
            self.manifest.add(meta);
            self.manifest.persist().map_err(|e| e.to_string())?;
        }
        
        Ok(())
    }
    
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
    
    pub fn memtable_entry_count(&self) -> usize {
        self.memtable.len()
    }
    
    pub fn sstable_count(&self) -> usize {
        self.sstables.lock().map_or(0, |levels| {
            levels.iter().map(|l| l.len()).sum()
        })
    }
    
    pub fn flush(&mut self) -> Result<(), String> {
        self.flush_memtable()
    }
}
