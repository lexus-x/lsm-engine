use std::time::{Duration, Instant};
use rand::Rng;
use rand::distributions::Distribution;
use rand::distributions::weighted::WeightedIndex;

use crate::benchmark::BenchmarkResult;
use crate::engine::Engine;
use crate::{EngineConfig, IndexType};

/// Key distribution types
#[derive(Debug, Clone, Copy)]
pub enum DistributionType {
    Uniform,
    Zipfian,
    Sequential,
    Latest,
}

/// YCSB-style workload generator
struct WorkloadConfig {
    read_proportion: f64,
    update_proportion: f64,
    insert_proportion: f64,
    delete_proportion: f64,
    distribution: DistributionType,
}

fn get_workload_config(workload: &str) -> WorkloadConfig {
    match workload {
        "A" => WorkloadConfig {
            read_proportion: 0.5,
            update_proportion: 0.5,
            insert_proportion: 0.0,
            delete_proportion: 0.0,
            distribution: DistributionType::Zipfian,
        },
        "B" => WorkloadConfig {
            read_proportion: 0.95,
            update_proportion: 0.05,
            insert_proportion: 0.0,
            delete_proportion: 0.0,
            distribution: DistributionType::Zipfian,
        },
        "C" => WorkloadConfig {
            read_proportion: 1.0,
            update_proportion: 0.0,
            insert_proportion: 0.0,
            delete_proportion: 0.0,
            distribution: DistributionType::Zipfian,
        },
        "D" => WorkloadConfig {
            read_proportion: 0.95,
            update_proportion: 0.0,
            insert_proportion: 0.05,
            delete_proportion: 0.0,
            distribution: DistributionType::Latest,
        },
        _ => WorkloadConfig {
            read_proportion: 0.5,
            update_proportion: 0.5,
            insert_proportion: 0.0,
            delete_proportion: 0.0,
            distribution: DistributionType::Uniform,
        },
    }
}

/// Zipfian distribution sampler (simplified)
struct ZipfianSampler {
    n: usize,
    theta: f64,
    zetan: f64,
}

impl ZipfianSampler {
    fn new(n: usize) -> Self {
        let theta = 0.99;
        let zetan = Self::zeta(n, theta);
        Self { n, theta, zetan }
    }

    fn zeta(n: usize, theta: f64) -> f64 {
        let mut sum = 0.0;
        for i in 1..=n {
            sum += 1.0 / (i as f64).powf(theta);
        }
        sum
    }

    fn sample(&self, rng: &mut impl Rng) -> usize {
        let u: f64 = rng.gen_range(0.0..1.0);
        let uz = u * self.zetan;
        
        let mut sum = 0.0;
        for i in 1..=self.n {
            sum += 1.0 / (i as f64).powf(self.theta);
            if sum >= uz {
                return i - 1;
            }
        }
        self.n - 1
    }
}

/// Generate a key for the given index
fn generate_key(idx: usize) -> Vec<u8> {
    format!("user{:010}", idx).into_bytes()
}

/// Generate a random value
fn generate_value(rng: &mut impl Rng) -> Vec<u8> {
    let len = rng.gen_range(64..256);
    let mut val = vec![0u8; len];
    rng.fill(&mut val[..]);
    val
}

/// Run a single workload benchmark
pub fn run_workload(
    workload: &str,
    engine_type: &str,
    num_keys: usize,
    num_ops: usize,
) -> BenchmarkResult {
    let config = get_workload_config(workload);
    let index_type = match engine_type {
        "learned" => IndexType::Learned,
        _ => IndexType::Bloom,
    };

    let tmp_dir = tempfile::tempdir().unwrap();
    let mut engine_config = EngineConfig::default();
    engine_config.data_dir = tmp_dir.path().to_str().unwrap().to_string();
    engine_config.index_type = index_type;
    engine_config.memtable_size_bytes = 2 * 1024 * 1024; // 2MB for benchmarks
    
    let mut engine = Engine::new(engine_config).unwrap();
    let mut rng = rand::thread_rng();

    // Phase 1: Load initial data
    println!("  Loading {} keys...", num_keys);
    for i in 0..num_keys {
        let key = generate_key(i);
        let val = generate_value(&mut rng);
        engine.put(key, val).unwrap();
    }
    engine.flush().unwrap();

    // Phase 2: Run operations
    let zipfian = ZipfianSampler::new(num_keys);
    let mut latencies: Vec<Duration> = Vec::with_capacity(num_ops);
    let mut read_ops = 0u64;
    let mut write_ops = 0u64;
    let mut delete_ops = 0u64;
    let read_errors = 0u64;
    let mut next_insert_key = num_keys;

    let op_weights = [
        config.read_proportion,
        config.update_proportion,
        config.insert_proportion,
        config.delete_proportion,
    ];
    let op_dist = WeightedIndex::new(&op_weights).unwrap();

    println!("  Running {} operations...", num_ops);
    let start_time = Instant::now();

    for _ in 0..num_ops {
        let op = op_dist.sample(&mut rng);
        
        let key_idx = match config.distribution {
            DistributionType::Uniform => rng.gen_range(0..num_keys),
            DistributionType::Zipfian => zipfian.sample(&mut rng).min(num_keys - 1),
            DistributionType::Sequential => rng.gen_range(0..num_keys),
            DistributionType::Latest => {
                let offset = rng.gen_range(0..num_keys / 10);
                num_keys.saturating_sub(1 + offset)
            }
        };

        let op_start = Instant::now();

        match op {
            0 => {
                // Read
                let key = generate_key(key_idx);
                let _ = engine.get(&key);
                read_ops += 1;
            }
            1 => {
                // Update
                let key = generate_key(key_idx);
                let val = generate_value(&mut rng);
                let _ = engine.put(key, val);
                write_ops += 1;
            }
            2 => {
                // Insert
                let key = generate_key(next_insert_key);
                let val = generate_value(&mut rng);
                let _ = engine.put(key, val);
                next_insert_key += 1;
                write_ops += 1;
            }
            3 => {
                // Delete
                let key = generate_key(key_idx);
                let _ = engine.delete(key);
                delete_ops += 1;
            }
            _ => unreachable!(),
        }

        latencies.push(op_start.elapsed());
    }

    let total_duration = start_time.elapsed();

    // Compute statistics
    latencies.sort();
    let p50_idx = latencies.len() / 2;
    let p99_idx = (latencies.len() * 99) / 100;
    
    let p50_latency_us = latencies[p50_idx].as_secs_f64() * 1_000_000.0;
    let p99_latency_us = latencies[p99_idx].as_secs_f64() * 1_000_000.0;
    let throughput = num_ops as f64 / total_duration.as_secs_f64();

    BenchmarkResult {
        workload: workload.to_string(),
        engine_type: engine_type.to_string(),
        total_ops: num_ops as u64,
        duration_ms: total_duration.as_millis() as u64,
        throughput_ops_per_sec: throughput,
        p50_latency_us,
        p99_latency_us,
        read_ops,
        write_ops,
        delete_ops,
        read_errors,
    }
}
