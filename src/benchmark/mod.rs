pub mod harness;

use serde::{Deserialize, Serialize};

/// Result of a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub workload: String,
    pub engine_type: String,
    pub total_ops: u64,
    pub duration_ms: u64,
    pub throughput_ops_per_sec: f64,
    pub p50_latency_us: f64,
    pub p99_latency_us: f64,
    pub read_ops: u64,
    pub write_ops: u64,
    pub delete_ops: u64,
    pub read_errors: u64,
}

/// Run a single benchmark workload
pub fn run_benchmark(
    workload: &str,
    engine_type: &str,
    num_keys: usize,
    num_ops: usize,
) -> BenchmarkResult {
    harness::run_workload(workload, engine_type, num_keys, num_ops)
}

/// Run all workloads and return results
pub fn run_all_benchmarks(num_keys: usize, num_ops: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    for workload in &["A", "B", "C", "D"] {
        for engine in &["bloom", "learned"] {
            println!("Running workload {} with {} engine...", workload, engine);
            let result = run_benchmark(workload, engine, num_keys, num_ops);
            results.push(result);
        }
    }
    
    results
}

/// Format results as a comparison table
pub fn format_comparison_table(results: &[BenchmarkResult]) -> String {
    let mut output = String::new();
    output.push_str(&format!("{:<12} {:<10} {:<15} {:<15} {:<15} {:<15}\n",
        "Workload", "Engine", "Throughput", "P50 Latency", "P99 Latency", "Total Ops"));
    output.push_str(&"-".repeat(85));
    output.push('\n');
    
    for r in results {
        output.push_str(&format!("{:<12} {:<10} {:<12.0} ops/s {:<12.1} us {:<12.1} us {:<15}\n",
            r.workload, r.engine_type, r.throughput_ops_per_sec,
            r.p50_latency_us, r.p99_latency_us, r.total_ops));
    }
    
    output
}

/// Format results as JSON
pub fn format_json(results: &[BenchmarkResult]) -> String {
    serde_json::to_string_pretty(results).unwrap_or_else(|_| "[]".to_string())
}
