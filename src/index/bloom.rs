use crate::index::{Index, LookupHint};

/// Standard Bloom filter implementation
pub struct BloomFilterIndex {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
    num_inserted: usize,
}

impl BloomFilterIndex {
    /// Create a new Bloom filter
    /// `expected_elements`: expected number of insertions
    /// `bits_per_element`: controls false positive rate (10 ≈ 1% FPR)
    pub fn new(expected_elements: usize, bits_per_element: f64) -> Self {
        let num_bits = ((expected_elements as f64 * bits_per_element) as usize).max(64);
        let num_hashes = ((bits_per_element * std::f64::consts::LN_2) as usize).max(1).min(30);
        let num_words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            num_inserted: 0,
        }
    }

    fn hash(&self, key: &[u8], seed: u32) -> u64 {
        // FNV-1a inspired hash
        let mut h: u64 = 0xcbf29ce484222325 ^ (seed as u64);
        for &b in key {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn insert(&mut self, key: &[u8]) {
        for i in 0..self.num_hashes {
            let h = self.hash(key, i as u32);
            let bit = (h % self.num_bits as u64) as usize;
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
        self.num_inserted += 1;
    }

    pub fn might_contain(&self, key: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let h = self.hash(key, i as u32);
            let bit = (h % self.num_bits as u64) as usize;
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        // num_bits(8) + num_hashes(4) + num_inserted(8) + bits_len(8) + bits
        data.extend_from_slice(&(self.num_bits as u64).to_le_bytes());
        data.extend_from_slice(&(self.num_hashes as u32).to_le_bytes());
        data.extend_from_slice(&(self.num_inserted as u64).to_le_bytes());
        data.extend_from_slice(&(self.bits.len() as u64).to_le_bytes());
        for word in &self.bits {
            data.extend_from_slice(&word.to_le_bytes());
        }
        data
    }

    pub fn deserialize(data: &[u8]) -> Self {
        if data.len() < 28 {
            return Self::new(1, 10.0);
        }
        let num_bits = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let num_hashes = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let num_inserted = u64::from_le_bytes(data[12..20].try_into().unwrap()) as usize;
        let bits_len = u64::from_le_bytes(data[20..28].try_into().unwrap()) as usize;
        let mut bits = Vec::with_capacity(bits_len);
        for i in 0..bits_len {
            let offset = 28 + i * 8;
            if offset + 8 <= data.len() {
                bits.push(u64::from_le_bytes(data[offset..offset+8].try_into().unwrap()));
            } else {
                bits.push(0);
            }
        }
        Self { bits, num_bits, num_hashes, num_inserted }
    }
}

impl Index for BloomFilterIndex {
    fn insert(&mut self, key: &[u8], _offset: usize) {
        self.insert(key);
    }

    fn build(&mut self) {
        // Bloom filter is built incrementally, nothing to finalize
    }

    fn lookup_hint(&self, key: &[u8]) -> LookupHint {
        if self.might_contain(key) {
            LookupHint::SearchRange { start: 0, end: usize::MAX }
        } else {
            LookupHint::NotFound
        }
    }

    fn serialize(&self) -> Vec<u8> {
        self.serialize()
    }

    fn index_type_name(&self) -> &str {
        "bloom"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut bf = BloomFilterIndex::new(1000, 10.0);
        for i in 0..100 {
            let key = format!("key{}", i);
            bf.insert(key.as_bytes());
        }
        
        // All inserted keys should be found
        for i in 0..100 {
            let key = format!("key{}", i);
            assert!(bf.might_contain(key.as_bytes()));
        }
        
        // Non-inserted keys should mostly not be found (with some false positives)
        let mut false_positives = 0;
        for i in 100..200 {
            let key = format!("key{}", i);
            if bf.might_contain(key.as_bytes()) {
                false_positives += 1;
            }
        }
        // False positive rate should be low
        assert!(false_positives < 10, "too many false positives: {}", false_positives);
    }

    #[test]
    fn test_bloom_filter_serialize() {
        let mut bf = BloomFilterIndex::new(100, 10.0);
        bf.insert(b"hello");
        bf.insert(b"world");
        
        let data = bf.serialize();
        let bf2 = BloomFilterIndex::deserialize(&data);
        
        assert!(bf2.might_contain(b"hello"));
        assert!(bf2.might_contain(b"world"));
        assert!(!bf2.might_contain(b"missing"));
    }
}
