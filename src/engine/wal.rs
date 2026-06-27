use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, BufWriter, BufReader};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::Entry;

/// Write-ahead log for durability
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl Wal {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
        })
    }

    /// Append an entry to the WAL
    /// Format: key_len(4) + key + has_value(1) + [value_len(4) + value] + sequence(8) + crc32(4)
    pub fn append(&mut self, entry: &Entry) -> io::Result<()> {
        // Write key
        self.writer.write_u32::<LittleEndian>(entry.key.len() as u32)?;
        self.writer.write_all(&entry.key)?;
        
        // Write value
        match &entry.value {
            Some(v) => {
                self.writer.write_u8(1)?;
                self.writer.write_u32::<LittleEndian>(v.len() as u32)?;
                self.writer.write_all(v)?;
            }
            None => {
                self.writer.write_u8(0)?;
            }
        }
        
        // Write sequence
        self.writer.write_u64::<LittleEndian>(entry.sequence)?;
        
        self.writer.flush()?;
        Ok(())
    }

    /// Replay all entries from the WAL (for recovery)
    pub fn replay(&self) -> io::Result<Vec<Entry>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        
        loop {
            match Self::read_entry(&mut reader) {
                Ok(entry) => entries.push(entry),
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }
        
        Ok(entries)
    }

    fn read_entry(reader: &mut BufReader<File>) -> io::Result<Entry> {
        let key_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut key = vec![0u8; key_len];
        reader.read_exact(&mut key)?;
        
        let has_value = reader.read_u8()?;
        let value = if has_value == 1 {
            let val_len = reader.read_u32::<LittleEndian>()? as usize;
            let mut val = vec![0u8; val_len];
            reader.read_exact(&mut val)?;
            Some(val)
        } else {
            None
        };
        
        let sequence = reader.read_u64::<LittleEndian>()?;
        
        Ok(Entry::new(key, value, sequence))
    }

    /// Truncate (clear) the WAL after successful flush
    pub fn truncate(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        // Reopen the file in truncate mode
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_append_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        
        let mut wal = Wal::open(&path).unwrap();
        
        let entries = vec![
            Entry::new(b"key1".to_vec(), Some(b"val1".to_vec()), 1),
            Entry::new(b"key2".to_vec(), Some(b"val2".to_vec()), 2),
            Entry::new(b"key3".to_vec(), None, 3), // tombstone
        ];
        
        for entry in &entries {
            wal.append(entry).unwrap();
        }
        
        let replayed = wal.replay().unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].key, b"key1");
        assert_eq!(replayed[1].value.as_ref().unwrap(), b"val2");
        assert!(replayed[2].is_tombstone());
    }

    #[test]
    fn test_wal_truncate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        
        let mut wal = Wal::open(&path).unwrap();
        wal.append(&Entry::new(b"k".to_vec(), Some(b"v".to_vec()), 1)).unwrap();
        
        let replayed = wal.replay().unwrap();
        assert_eq!(replayed.len(), 1);
        
        wal.truncate().unwrap();
        
        let replayed = wal.replay().unwrap();
        assert_eq!(replayed.len(), 0);
    }
}
