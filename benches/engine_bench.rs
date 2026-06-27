use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lsm_engine::engine::Engine;
use lsm_engine::{EngineConfig, IndexType, LookupResult};

fn bench_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("put");
    
    for index_type in &[IndexType::Bloom, IndexType::Learned] {
        let name = format!("{:?}", index_type);
        
        group.bench_with_input(BenchmarkId::new("put", &name), index_type, |b, &idx_type| {
            let tmp_dir = tempfile::tempdir().unwrap();
            let mut config = EngineConfig::default();
            config.data_dir = tmp_dir.path().to_str().unwrap().to_string();
            config.index_type = idx_type;
            config.memtable_size_bytes = 4 * 1024 * 1024;
            
            let mut engine = Engine::new(config).unwrap();
            let mut counter = 0u64;
            
            b.iter(|| {
                let key = format!("key{:012}", counter).into_bytes();
                let val = vec![0u8; 100];
                engine.put(key, val).unwrap();
                counter += 1;
            });
        });
    }
    
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");
    
    for index_type in &[IndexType::Bloom, IndexType::Learned] {
        let name = format!("{:?}", index_type);
        
        group.bench_with_input(BenchmarkId::new("get", &name), index_type, |b, &idx_type| {
            let tmp_dir = tempfile::tempdir().unwrap();
            let mut config = EngineConfig::default();
            config.data_dir = tmp_dir.path().to_str().unwrap().to_string();
            config.index_type = idx_type;
            config.memtable_size_bytes = 4 * 1024 * 1024;
            
            let mut engine = Engine::new(config).unwrap();
            
            // Pre-populate
            for i in 0..10000u64 {
                let key = format!("key{:012}", i).into_bytes();
                let val = vec![0u8; 100];
                engine.put(key, val).unwrap();
            }
            engine.flush().unwrap();
            
            let mut counter = 0u64;
            
            b.iter(|| {
                let key = format!("key{:012}", counter % 10000).into_bytes();
                let _ = black_box(engine.get(&key));
                counter += 1;
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_put, bench_get);
criterion_main!(benches);
