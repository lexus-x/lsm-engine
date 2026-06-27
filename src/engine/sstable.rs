use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write, BufWriter};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt, ByteOrder};

use crate::{Entry, LookupResult};
use crate::index::{IndexType, Index, BloomFilterIndex, LearnedIndex, LookupHint};

const MAGIC: u32 = 0x4C534D45; // "LSME"
const VERSION: u32 = 1;
const FOOTER_MAX_SIZE: usize = 4096;

/// A single data block in the SSTable
#[derive(Debug, Clone)]
struct Block {
    entries: Vec<Entry>,
    first_key: Vec<u8>,
    offset: u64,
    size: u64,
}

impl Block {
    fn new(entries: Vec<Entry>, offset: u64, size: u64) -> Self {
        let first_key = entries.first().map(|e| e.key.clone()).unwrap_or_default();
        Self { entries, first_key, offset, size }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(self.entries.len() as u32).unwrap();
        for entry in &self.entries {
            buf.write_u32::<LittleEndian>(entry.key.len() as u32).unwrap();
            buf.extend_from_slice(&entry.key);
            match &entry.value {
                Some(v) => {
                    buf.write_u8(1).unwrap();
                    buf.write_u32::<LittleEndian>(v.len() as u32).unwrap();
                    buf.extend_from_slice(v);
                }
                None => {
                    buf.write_u8(0).unwrap();
                }
            }
            buf.write_u64::<LittleEndian>(entry.sequence).unwrap();
        }
        buf
    }

    fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut cursor = io::Cursor::new(data);
        let num_entries = cursor.read_u32::<LittleEndian>()? as usize;
        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let key_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut key = vec![0u8; key_len];
            cursor.read_exact(&mut key)?;
            let has_value = cursor.read_u8()?;
            let value = if has_value == 1 {
                let val_len = cursor.read_u32::<LittleEndian>()? as usize;
                let mut val = vec![0u8; val_len];
                cursor.read_exact(&mut val)?;
                Some(val)
            } else {
                None
            };
            let sequence = cursor.read_u64::<LittleEndian>()?;
            entries.push(Entry::new(key, value, sequence));
        }
        Ok(Self::new(entries, 0, 0))
    }

    fn get(&self, key: &[u8]) -> Option<&Entry> {
        // Binary search within block
        match self.entries.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            Ok(mut idx) => {
                // Found a match, but there might be multiple versions;
                // walk back to find all, pick highest sequence
                let mut best = &self.entries[idx];
                // Walk forward to find any later version
                while idx + 1 < self.entries.len() && self.entries[idx + 1].key.as_slice() == key {
                    idx += 1;
                    if self.entries[idx].sequence > best.sequence {
                        best = &self.entries[idx];
                    }
                }
                Some(best)
            }
            Err(_) => None,
        }
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Vec<Entry> {
        let mut results = Vec::new();
        for entry in &self.entries {
            if entry.key.as_slice() >= start && entry.key.as_slice() <= end {
                results.push(entry.clone());
            }
            if entry.key.as_slice() > end {
                break;
            }
        }
        results
    }
}

/// Block index entry mapping first_key -> block position
#[derive(Debug, Clone)]
struct BlockIndexEntry {
    first_key: Vec<u8>,
    block_offset: u64,
    block_size: u64,
}

/// SSTable on disk
pub struct SSTable {
    path: PathBuf,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
    block_index: Vec<BlockIndexEntry>,
    index_type: IndexType,
    index: Box<dyn Index>,
    size: u64,
}

impl SSTable {
    pub fn open(path: &Path, _index_type: IndexType) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();
        
        if file_size < 12 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }
        
        // Read footer_size(4) + MAGIC(4) from the very end
        file.seek(SeekFrom::Start(file_size - 8))?;
        let mut end_buf = [0u8; 8];
        file.read_exact(&mut end_buf)?;
        
        let footer_size = LittleEndian::read_u32(&end_buf[0..4]) as u64;
        let magic = LittleEndian::read_u32(&end_buf[4..8]);
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid SSTable magic"));
        }
        
        // Read footer data (before footer_size+magic)
        let footer_start = file_size - 8 - footer_size;
        file.seek(SeekFrom::Start(footer_start))?;
        let mut footer = vec![0u8; footer_size as usize];
        file.read_exact(&mut footer)?;
        
        let mut cursor = io::Cursor::new(footer.as_slice());
        let version = cursor.read_u32::<LittleEndian>()?;
        let idx_type_byte = cursor.read_u8()?;
        let min_key_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut min_key = vec![0u8; min_key_len];
        cursor.read_exact(&mut min_key)?;
        let max_key_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut max_key = vec![0u8; max_key_len];
        cursor.read_exact(&mut max_key)?;
        let block_index_offset = cursor.read_u64::<LittleEndian>()?;
        let index_offset = cursor.read_u64::<LittleEndian>()?;
        let index_size = cursor.read_u64::<LittleEndian>()?;
        
        let _ = version;
        
        let idx_type = match idx_type_byte {
            0 => IndexType::Bloom,
            1 => IndexType::Learned,
            _ => IndexType::Bloom,
        };
        
        // Read block index
        file.seek(SeekFrom::Start(block_index_offset))?;
        let mut bi_buf = vec![0u8; (index_offset - block_index_offset) as usize];
        file.read_exact(&mut bi_buf)?;
        let block_index = Self::deserialize_block_index(&bi_buf)?;
        
        // Read index data
        file.seek(SeekFrom::Start(index_offset))?;
        let mut idx_buf = vec![0u8; index_size as usize];
        file.read_exact(&mut idx_buf)?;
        
        let index: Box<dyn Index> = match idx_type {
            IndexType::Bloom => {
                let bloom = BloomFilterIndex::deserialize(&idx_buf);
                Box::new(bloom)
            }
            IndexType::Learned => {
                let learned = LearnedIndex::deserialize(&idx_buf);
                Box::new(learned)
            }
        };
        
        Ok(Self {
            path: path.to_path_buf(),
            min_key,
            max_key,
            block_index,
            index_type: idx_type,
            index,
            size: file_size,
        })
    }

    fn deserialize_block_index(data: &[u8]) -> io::Result<Vec<BlockIndexEntry>> {
        let mut cursor = io::Cursor::new(data);
        let num_blocks = cursor.read_u32::<LittleEndian>()? as usize;
        let mut entries = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let key_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut first_key = vec![0u8; key_len];
            cursor.read_exact(&mut first_key)?;
            let block_offset = cursor.read_u64::<LittleEndian>()?;
            let block_size = cursor.read_u64::<LittleEndian>()?;
            entries.push(BlockIndexEntry { first_key, block_offset, block_size });
        }
        Ok(entries)
    }

    fn read_block(&self, entry: &BlockIndexEntry) -> io::Result<Block> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.block_offset))?;
        let mut buf = vec![0u8; entry.block_size as usize];
        file.read_exact(&mut buf)?;
        let mut block = Block::deserialize(&buf)?;
        block.offset = entry.block_offset;
        block.size = entry.block_size;
        Ok(block)
    }

    fn find_block(&self, key: &[u8]) -> Option<usize> {
        if self.block_index.is_empty() {
            return None;
        }
        // Binary search for the block that might contain this key
        let idx = self.block_index.partition_point(|e| e.first_key.as_slice() <= key);
        if idx == 0 {
            Some(0)
        } else {
            Some(idx - 1)
        }
    }

    pub fn get(&self, key: &[u8]) -> io::Result<LookupResult> {
        if key < self.min_key.as_slice() || key > self.max_key.as_slice() {
            return Ok(LookupResult::NotFound);
        }
        
        // Use index to check if key might exist
        match self.index.lookup_hint(key) {
            LookupHint::NotFound => return Ok(LookupResult::NotFound),
            LookupHint::SearchRange { .. } => {
                // Proceed with lookup
            }
        }
        
        let block_idx = self.find_block(key);
        match block_idx {
            Some(idx) => {
                let block = self.read_block(&self.block_index[idx])?;
                match block.get(key) {
                    Some(entry) => {
                        if entry.is_tombstone() {
                            Ok(LookupResult::Tombstone)
                        } else {
                            Ok(LookupResult::Found(entry.value.clone().unwrap()))
                        }
                    }
                    None => Ok(LookupResult::NotFound),
                }
            }
            None => Ok(LookupResult::NotFound),
        }
    }

    pub fn scan(&self, start: &[u8], end: &[u8]) -> io::Result<Vec<Entry>> {
        let mut results = Vec::new();
        let start_block = self.find_block(start).unwrap_or(0);
        
        for i in start_block..self.block_index.len() {
            let block = self.read_block(&self.block_index[i])?;
            // If block's first key > end, we're done
            if block.first_key.as_slice() > end {
                break;
            }
            let entries = block.scan(start, end);
            results.extend(entries);
        }
        
        Ok(results)
    }

    pub fn min_key(&self) -> &[u8] {
        &self.min_key
    }

    pub fn max_key(&self) -> &[u8] {
        &self.max_key
    }

    pub fn size_bytes(&self) -> u64 {
        self.size
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn id(&self) -> u64 {
        // Extract id from filename
        self.path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('_').last())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    pub fn index_type(&self) -> IndexType {
        self.index_type
    }
}

/// Builder for constructing SSTables from sorted entries
pub struct SSTableBuilder {
    path: PathBuf,
    block_size: usize,
    index_type: IndexType,
    bloom_bits_per_element: f64,
}

pub struct SSTableBuildResult {
    pub sst: SSTable,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub size_bytes: u64,
}

impl SSTableBuilder {
    pub fn new(
        path: &Path,
        block_size: usize,
        index_type: IndexType,
        bloom_bits_per_element: f64,
    ) -> Self {
        Self {
            path: path.to_path_buf(),
            block_size,
            index_type,
            bloom_bits_per_element,
        }
    }

    pub fn build_from_entries(&self, entries: &[Entry]) -> io::Result<SSTableBuildResult> {
        if entries.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "no entries"));
        }

        let mut file = BufWriter::new(File::create(&self.path)?);
        let mut block_index_entries = Vec::new();
        let mut current_block_entries = Vec::new();
        let mut current_block_size = 0u64;
        let mut offset = 0u64;

        // Create index
        let mut index: Box<dyn Index> = match self.index_type {
            IndexType::Bloom => Box::new(BloomFilterIndex::new(entries.len(), self.bloom_bits_per_element)),
            IndexType::Learned => Box::new(LearnedIndex::new()),
        };

        // Build blocks
        for entry in entries {
            let entry_size = 4 + entry.key.len() + 1 + entry.value.as_ref().map_or(0, |v| 4 + v.len()) + 8;
            
            if current_block_size + entry_size as u64 > self.block_size as u64 && !current_block_entries.is_empty() {
                // Flush current block
                let block = Block::new(current_block_entries.clone(), offset, 0);
                let block_data = block.serialize();
                let block_size = block_data.len() as u64;
                file.write_all(&block_data)?;
                
                block_index_entries.push(BlockIndexEntry {
                    first_key: block.first_key.clone(),
                    block_offset: offset,
                    block_size,
                });
                
                // Add entries to index
                for e in &current_block_entries {
                    index.insert(&e.key, offset as usize);
                }
                
                offset += block_size;
                current_block_entries.clear();
                current_block_size = 0;
            }
            
            current_block_entries.push(entry.clone());
            current_block_size += entry_size as u64;
        }

        // Flush last block
        if !current_block_entries.is_empty() {
            let block = Block::new(current_block_entries.clone(), offset, 0);
            let block_data = block.serialize();
            let block_size = block_data.len() as u64;
            file.write_all(&block_data)?;
            
            block_index_entries.push(BlockIndexEntry {
                first_key: block.first_key.clone(),
                block_offset: offset,
                block_size,
            });
            
            for e in &current_block_entries {
                index.insert(&e.key, offset as usize);
            }
            
            offset += block_size;
        }

        // Build index
        index.build();
        
        // For learned index, also train with offset data
        if self.index_type == IndexType::Learned {
            // Rebuild learned index with proper offset training
            let mut learned = LearnedIndex::new();
            for entry in entries {
                // Find which block this entry belongs to
                for bi in &block_index_entries {
                    if entry.key >= bi.first_key {
                        learned.insert(&entry.key, bi.block_offset as usize);
                    }
                }
            }
            learned.build();
            index = Box::new(learned);
        }

        let block_index_offset = offset;
        
        // Write block index
        let bi_data = Self::serialize_block_index(&block_index_entries);
        file.write_all(&bi_data)?;
        let index_offset = block_index_offset + bi_data.len() as u64;
        
        // Write index data
        let index_data = index.serialize();
        file.write_all(&index_data)?;
        let index_size = index_data.len() as u64;
        
        let min_key = entries.first().unwrap().key.clone();
        let max_key = entries.last().unwrap().key.clone();
        
        // Write footer
        // Format: version(4) + index_type(1) + min_key_len(4) + min_key + max_key_len(4) + max_key + block_index_offset(8) + index_offset(8) + index_size(8) | footer_size(4) | magic(4)
        let mut footer = Vec::new();
        footer.write_u32::<LittleEndian>(VERSION)?;
        footer.write_u8(self.index_type as u8)?;
        footer.write_u32::<LittleEndian>(min_key.len() as u32)?;
        footer.extend_from_slice(&min_key);
        footer.write_u32::<LittleEndian>(max_key.len() as u32)?;
        footer.extend_from_slice(&max_key);
        footer.write_u64::<LittleEndian>(block_index_offset)?;
        footer.write_u64::<LittleEndian>(index_offset)?;
        footer.write_u64::<LittleEndian>(index_size)?;
        file.write_all(&footer)?;
        file.write_u32::<LittleEndian>(footer.len() as u32)?;
        file.write_u32::<LittleEndian>(MAGIC)?;
        
        file.flush()?;
        drop(file);
        
        let total_size = std::fs::metadata(&self.path)?.len();
        
        // Re-open the SSTable
        let sst = SSTable::open(&self.path, self.index_type)?;
        
        Ok(SSTableBuildResult {
            sst,
            min_key,
            max_key,
            size_bytes: total_size,
        })
    }

    fn serialize_block_index(entries: &[BlockIndexEntry]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(entries.len() as u32).unwrap();
        for entry in entries {
            buf.write_u32::<LittleEndian>(entry.first_key.len() as u32).unwrap();
            buf.extend_from_slice(&entry.first_key);
            buf.write_u64::<LittleEndian>(entry.block_offset).unwrap();
            buf.write_u64::<LittleEndian>(entry.block_size).unwrap();
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_sstable_write_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.dat");
        
        let entries: Vec<Entry> = (0..100)
            .map(|i| {
                let key = format!("key{:04}", i).into_bytes();
                let val = format!("value{}", i).into_bytes();
                Entry::new(key, Some(val), i as u64)
            })
            .collect();
        
        let builder = SSTableBuilder::new(&path, 256, IndexType::Bloom, 10.0);
        let result = builder.build_from_entries(&entries).unwrap();
        
        // Point lookup
        match result.sst.get(b"key0050").unwrap() {
            LookupResult::Found(v) => assert_eq!(v, b"value50"),
            _ => panic!("expected found"),
        }
        
        // Not found
        match result.sst.get(b"key9999").unwrap() {
            LookupResult::NotFound => {}
            _ => panic!("expected not found"),
        }
        
        // Scan
        let scanned = result.sst.scan(b"key0010", b"key0019").unwrap();
        assert_eq!(scanned.len(), 10);
    }
}
