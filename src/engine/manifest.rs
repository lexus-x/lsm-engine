use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::index::IndexType;
use crate::engine::SSTableMeta;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestEntry {
    id: u64,
    level: usize,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
    size_bytes: u64,
    index_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestData {
    version: u32,
    entries: Vec<ManifestEntry>,
}

/// Tracks all SSTables, their levels, and key ranges.
/// Persisted to disk as JSON.
pub struct Manifest {
    path: PathBuf,
    entries: Vec<ManifestEntry>,
}

impl Manifest {
    pub fn load_or_create(path: &PathBuf) -> Self {
        let entries = if path.exists() {
            match fs::read_to_string(path) {
                Ok(data) => {
                    match serde_json::from_str::<ManifestData>(&data) {
                        Ok(manifest) => manifest.entries,
                        Err(_) => Vec::new(),
                    }
                }
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };

        Self {
            path: path.clone(),
            entries,
        }
    }

    pub fn add(&mut self, meta: SSTableMeta) {
        let entry = ManifestEntry {
            id: meta.id,
            level: meta.level,
            min_key: meta.min_key,
            max_key: meta.max_key,
            size_bytes: meta.size_bytes,
            index_type: format!("{:?}", meta.index_type),
        };
        self.entries.push(entry);
    }

    pub fn remove(&mut self, id: u64) {
        self.entries.retain(|e| e.id != id);
    }

    pub fn persist(&self) -> Result<(), String> {
        let data = ManifestData {
            version: 1,
            entries: self.entries.clone(),
        };
        let json = serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?;
        fs::write(&self.path, json).map_err(|e| e.to_string())
    }

    pub fn max_level(&self) -> usize {
        self.entries.iter().map(|e| e.level).max().unwrap_or(0)
    }

    pub fn entries(&self) -> Vec<SSTableMeta> {
        self.entries.iter().map(|e| SSTableMeta {
            id: e.id,
            level: e.level,
            path: PathBuf::new(), // path will be reconstructed
            min_key: e.min_key.clone(),
            max_key: e.max_key.clone(),
            size_bytes: e.size_bytes,
            index_type: match e.index_type.as_str() {
                "Bloom" => IndexType::Bloom,
                "Learned" => IndexType::Learned,
                _ => IndexType::Bloom,
            },
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_manifest_persist_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest.json");
        
        let mut manifest = Manifest::load_or_create(&path);
        
        manifest.add(SSTableMeta {
            id: 1,
            level: 0,
            path: PathBuf::from("sst_0_1.dat"),
            min_key: b"aaa".to_vec(),
            max_key: b"zzz".to_vec(),
            size_bytes: 1024,
            index_type: IndexType::Bloom,
        });
        
        manifest.persist().unwrap();
        
        let loaded = Manifest::load_or_create(&path);
        assert_eq!(loaded.entries().len(), 1);
        assert_eq!(loaded.max_level(), 0);
    }
}
