use clap::{Parser, Subcommand};

use lsm_engine::benchmark;

#[derive(Parser)]
#[command(name = "bench")]
#[command(about = "LSM-Engine Benchmark Tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single benchmark workload
    Run {
        /// Workload type: A, B, C, or D
        #[arg(short, long, default_value = "A")]
        workload: String,
        
        /// Number of keys to pre-load
        #[arg(short, long, default_value = "100000")]
        keys: usize,
        
        /// Number of operations to run
        #[arg(short, long, default_value = "1000000")]
        ops: usize,
        
        /// Engine type: bloom or learned
        #[arg(short, long, default_value = "bloom")]
        engine: String,
    },
    
    /// Run all workloads and compare bloom vs learned
    Compare {
        /// Number of keys to pre-load
        #[arg(short, long, default_value = "100000")]
        keys: usize,
        
        /// Number of operations to run
        #[arg(short, long, default_value = "1000000")]
        ops: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Run { workload, keys, ops, engine } => {
            println!("=== LSM-Engine Benchmark ===");
            println!("Workload: {}", workload);
            println!("Engine: {}", engine);
            println!("Keys: {}", keys);
            println!("Operations: {}", ops);
            println!();
            
            let result = benchmark::run_benchmark(workload, engine, *keys, *ops);
            
            println!("\n=== Results ===");
            println!("Throughput: {:.0} ops/sec", result.throughput_ops_per_sec);
            println!("P50 Latency: {:.1} µs", result.p50_latency_us);
            println!("P99 Latency: {:.1} µs", result.p99_latency_us);
            println!("Duration: {} ms", result.duration_ms);
            println!("Read ops: {}, Write ops: {}, Delete ops: {}", 
                result.read_ops, result.write_ops, result.delete_ops);
            
            // Output JSON
            println!("\n=== JSON ===");
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
        
        Commands::Compare { keys, ops } => {
            println!("=== LSM-Engine Comparison Benchmark ===");
            println!("Keys: {}", keys);
            println!("Operations: {}", ops);
            println!();
            
            let results = benchmark::run_all_benchmarks(*keys, *ops);
            
            println!("\n=== Comparison Table ===");
            println!("{}", benchmark::format_comparison_table(&results));
            
            // Save JSON results
            let json = benchmark::format_json(&results);
            std::fs::write("benchmark_results.json", &json).unwrap_or_else(|e| {
                eprintln!("Failed to write results: {}", e);
            });
            println!("\nResults saved to benchmark_results.json");
        }
    }
}
