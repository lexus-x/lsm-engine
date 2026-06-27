use rand::Rng;
use crate::index::{Index, LookupHint};

const HIDDEN_SIZE: usize = 32;
const EPOCHS: usize = 100;
const LEARNING_RATE: f64 = 0.01;

/// Convert first 8 bytes of a key to a u64 for numeric comparison
fn key_to_u64(key: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    let len = key.len().min(8);
    bytes[..len].copy_from_slice(&key[..len]);
    u64::from_be_bytes(bytes)
}

/// Learned index model: single-hidden-layer MLP that predicts byte offsets
pub struct LearnedIndexModel {
    // Training data
    samples: Vec<(u64, f64)>, // (key_value, byte_offset)

    // Normalization
    key_min: u64,
    key_max: u64,
    offset_max: f64,

    // Quantized MLP weights (INT8)
    w1_q: Vec<i8>,      // [HIDDEN_SIZE] - input to hidden
    b1: Vec<f64>,        // [HIDDEN_SIZE] - hidden bias
    w1_scale: f64,
    w2_q: Vec<i8>,      // [HIDDEN_SIZE] - hidden to output
    b2: f64,             // output bias
    w2_scale: f64,

    // Error bound
    max_error: f64,

    // State
    trained: bool,
}

impl LearnedIndexModel {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            key_min: u64::MAX,
            key_max: 0,
            offset_max: 0.0,
            w1_q: vec![0; HIDDEN_SIZE],
            b1: vec![0.0; HIDDEN_SIZE],
            w1_scale: 1.0,
            w2_q: vec![0; HIDDEN_SIZE],
            b2: 0.0,
            w2_scale: 1.0,
            max_error: 0.0,
            trained: false,
        }
    }

    pub fn add_sample(&mut self, key: &[u8], offset: usize) {
        let key_val = key_to_u64(key);
        if key_val < self.key_min { self.key_min = key_val; }
        if key_val > self.key_max { self.key_max = key_val; }
        if offset as f64 > self.offset_max { self.offset_max = offset as f64; }
        self.samples.push((key_val, offset as f64));
    }

    fn normalize_key(&self, key_val: u64) -> f64 {
        if self.key_max == self.key_min {
            return 0.5;
        }
        (key_val - self.key_min) as f64 / (self.key_max - self.key_min) as f64
    }

    fn normalize_offset(&self, offset: f64) -> f64 {
        if self.offset_max <= 0.0 {
            return 0.0;
        }
        offset / self.offset_max
    }

    fn denormalize_offset(&self, normalized: f64) -> f64 {
        normalized * self.offset_max
    }

    /// Train the MLP on collected samples
    pub fn train(&mut self) {
        if self.samples.is_empty() || self.key_max == self.key_min {
            self.trained = true;
            return;
        }

        let mut rng = rand::thread_rng();

        // Initialize weights randomly
        let mut w1: Vec<f64> = (0..HIDDEN_SIZE).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let mut b1: Vec<f64> = vec![0.0; HIDDEN_SIZE];
        let mut w2: Vec<f64> = (0..HIDDEN_SIZE).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let mut b2: f64 = 0.0;

        // Prepare normalized training data
        let train_data: Vec<(f64, f64)> = self.samples.iter().map(|(k, o)| {
            (self.normalize_key(*k), self.normalize_offset(*o))
        }).collect();

        // SGD training
        for _epoch in 0..EPOCHS {
            for &(x, target) in &train_data {
                // Forward pass
                let mut a1 = [0.0f64; HIDDEN_SIZE];
                let mut z1 = [0.0f64; HIDDEN_SIZE];
                for i in 0..HIDDEN_SIZE {
                    z1[i] = w1[i] * x + b1[i];
                    a1[i] = if z1[i] > 0.0 { z1[i] } else { 0.0 }; // ReLU
                }

                let mut output = b2;
                for i in 0..HIDDEN_SIZE {
                    output += w2[i] * a1[i];
                }

                // Backward pass (MSE loss: d_loss/d_output = 2*(output - target))
                let d_output = output - target; // simplified gradient (drop the 2, absorbed into lr)

                // Gradients for output layer
                for i in 0..HIDDEN_SIZE {
                    let dw2 = d_output * a1[i];
                    w2[i] -= LEARNING_RATE * dw2;
                }
                b2 -= LEARNING_RATE * d_output;

                // Gradients for hidden layer
                for i in 0..HIDDEN_SIZE {
                    let d_a1 = d_output * w2[i];
                    let d_z1 = if z1[i] > 0.0 { d_a1 } else { 0.0 }; // ReLU derivative
                    let dw1 = d_z1 * x;
                    let db1 = d_z1;
                    w1[i] -= LEARNING_RATE * dw1;
                    b1[i] -= LEARNING_RATE * db1;
                }
            }
        }

        // Quantize weights to INT8
        let w1_max = w1.iter().map(|w| w.abs()).fold(0.0f64, f64::max);
        let w2_max = w2.iter().map(|w| w.abs()).fold(0.0f64, f64::max);

        let w1_scale = if w1_max > 0.0 { w1_max / 127.0 } else { 1.0 };
        let w2_scale = if w2_max > 0.0 { w2_max / 127.0 } else { 1.0 };

        let w1_q: Vec<i8> = w1.iter().map(|&w| {
            (w / w1_scale).round().clamp(-128.0, 127.0) as i8
        }).collect();
        let w2_q: Vec<i8> = w2.iter().map(|&w| {
            (w / w2_scale).round().clamp(-128.0, 127.0) as i8
        }).collect();

        // Compute max prediction error on training data
        let mut max_err = 0.0f64;
        for &(key_val, _) in &self.samples {
            let x = self.normalize_key(key_val);
            let predicted_norm = self.forward_f32(x, &w1, &b1, &w2, b2);
            let predicted = self.denormalize_offset(predicted_norm);

            // Find actual offset for this key
            if let Some(&(_, actual)) = self.samples.iter().find(|(k, _)| *k == key_val) {
                let err = (predicted - actual).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        self.w1_q = w1_q;
        self.b1 = b1;
        self.w1_scale = w1_scale;
        self.w2_q = w2_q;
        self.b2 = b2;
        self.w2_scale = w2_scale;
        self.max_error = max_err;
        self.trained = true;
    }

    fn forward_f32(&self, x: f64, w1: &[f64], b1: &[f64], w2: &[f64], b2: f64) -> f64 {
        let mut a1 = [0.0f64; HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            let z = w1[i] * x + b1[i];
            a1[i] = if z > 0.0 { z } else { 0.0 };
        }
        let mut output = b2;
        for i in 0..HIDDEN_SIZE {
            output += w2[i] * a1[i];
        }
        output
    }

    /// Forward pass with quantized weights (INT8 with dequantization)
    fn forward_quantized(&self, x: f64) -> f64 {
        let mut a1 = [0.0f64; HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            let w = self.w1_q[i] as f64 * self.w1_scale;
            let z = w * x + self.b1[i];
            a1[i] = if z > 0.0 { z } else { 0.0 }; // ReLU
        }
        let mut output = self.b2;
        for i in 0..HIDDEN_SIZE {
            let w = self.w2_q[i] as f64 * self.w2_scale;
            output += w * a1[i];
        }
        output
    }

    /// Predict the byte offset range for a given key
    pub fn predict(&self, key: &[u8]) -> Option<(usize, usize)> {
        if !self.trained || self.samples.is_empty() || self.key_max == self.key_min {
            return None;
        }

        let key_val = key_to_u64(key);
        let x = self.normalize_key(key_val);
        let predicted_norm = self.forward_quantized(x);
        let predicted = self.denormalize_offset(predicted_norm);

        let err = self.max_error;
        let start = (predicted - err).max(0.0) as usize;
        let end = (predicted + err) as usize;

        Some((start, end))
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header: trained flag + key_min + key_max + offset_max
        data.push(if self.trained { 1 } else { 0 });
        data.extend_from_slice(&self.key_min.to_le_bytes());
        data.extend_from_slice(&self.key_max.to_le_bytes());
        data.extend_from_slice(&self.offset_max.to_le_bytes());
        data.extend_from_slice(&self.max_error.to_le_bytes());

        // Quantized weights
        data.extend_from_slice(&(HIDDEN_SIZE as u32).to_le_bytes());
        data.extend_from_slice(&self.w1_scale.to_le_bytes());
        for &w in &self.w1_q {
            data.push(w as u8);
        }
        for &b in &self.b1 {
            data.extend_from_slice(&b.to_le_bytes());
        }
        data.extend_from_slice(&self.w2_scale.to_le_bytes());
        for &w in &self.w2_q {
            data.push(w as u8);
        }
        data.extend_from_slice(&self.b2.to_le_bytes());

        data
    }

    pub fn deserialize(data: &[u8]) -> Self {
        if data.len() < 41 {
            return Self::new();
        }

        let mut pos = 0;
        let trained = data[pos] == 1; pos += 1;
        let key_min = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;
        let key_max = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;
        let offset_max = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;
        let max_error = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;

        let hidden_size = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize; pos += 4;
        let hs = hidden_size.min(HIDDEN_SIZE);

        let w1_scale = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;
        let mut w1_q = vec![0i8; hs];
        for i in 0..hs {
            if pos < data.len() {
                w1_q[i] = data[pos] as i8; pos += 1;
            }
        }
        let mut b1 = vec![0.0f64; hs];
        for i in 0..hs {
            if pos + 8 <= data.len() {
                b1[i] = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;
            }
        }
        let w2_scale = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap()); pos += 8;
        let mut w2_q = vec![0i8; hs];
        for i in 0..hs {
            if pos < data.len() {
                w2_q[i] = data[pos] as i8; pos += 1;
            }
        }
        let mut b2 = 0.0f64;
        if pos + 8 <= data.len() {
            b2 = f64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
        }

        Self {
            samples: Vec::new(),
            key_min,
            key_max,
            offset_max,
            w1_q,
            b1,
            w1_scale,
            w2_q,
            b2,
            w2_scale,
            max_error,
            trained,
        }
    }
}

impl Index for LearnedIndexModel {
    fn insert(&mut self, key: &[u8], offset: usize) {
        self.add_sample(key, offset);
    }

    fn build(&mut self) {
        self.train();
    }

    fn lookup_hint(&self, key: &[u8]) -> LookupHint {
        match self.predict(key) {
            Some((start, end)) => LookupHint::SearchRange { start, end },
            None => LookupHint::SearchRange { start: 0, end: usize::MAX },
        }
    }

    fn serialize(&self) -> Vec<u8> {
        self.serialize()
    }

    fn index_type_name(&self) -> &str {
        "learned"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_index_basic() {
        let mut model = LearnedIndexModel::new();
        
        // Add training data: keys 0..100 map to offsets 0..1000
        for i in 0..100u64 {
            let key = format!("key{:04}", i).into_bytes();
            model.add_sample(&key, (i * 10) as usize);
        }
        
        model.train();
        assert!(model.trained);
        
        // Predictions should be reasonable
        let key = format!("key{:04}", 50u64).into_bytes();
        let (start, end) = model.predict(&key).unwrap();
        // The actual offset is 500; prediction should be within error bounds
        assert!(start <= 600, "start {} too high", start);
        assert!(end >= 400, "end {} too low", end);
    }

    #[test]
    fn test_learned_index_serialize() {
        let mut model = LearnedIndexModel::new();
        for i in 0..50u64 {
            let key = format!("k{:04}", i).into_bytes();
            model.add_sample(&key, (i * 20) as usize);
        }
        model.train();
        
        let data = model.serialize();
        let model2 = LearnedIndexModel::deserialize(&data);
        
        assert!(model2.trained);
        assert_eq!(model2.key_min, model.key_min);
        assert_eq!(model2.key_max, model.key_max);
    }

    #[test]
    fn test_learned_index_empty() {
        let model = LearnedIndexModel::new();
        assert!(model.predict(b"anything").is_none());
    }
}
