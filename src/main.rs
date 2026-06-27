use lsm_engine::engine::Engine;
use lsm_engine::EngineConfig;

fn main() {
    println!("LSM-Engine v0.1.0");
    println!("A key-value store with learned index models");
    println!("Use the `bench` binary for benchmarks: cargo run --bin bench");
    
    let config = EngineConfig::default();
    let _engine = Engine::new(config).expect("Failed to create engine");
    println!("Engine initialized successfully.");
}
