# LSM-Engine

```
    ██╗      ███████╗███╗   ███╗    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗
    ██║      ██╔════╝████╗ ████║    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝
    ██║      ███████╗██╔████╔██║    █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  
    ██║      ╚════██║██║╚██╔╝██║    ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  
    ███████╗ ███████║██║ ╚═╝ ██║    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗
    ╚══════╝ ╚══════╝╚═╝     ╚═╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝
```

> **A key-value store with learned index models replacing Bloom filters**

LSM-Engine is a Rust implementation of an LSM-tree key-value store that explores using
tiny neural networks (learned indexes) as a drop-in replacement for Bloom filters in
SSTable lookups.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LSM-Engine                           │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │   WAL    │   │ Memtable │   │   Index Layer    │   │
│  │  (disk)  │   │ (memory) │   │ ┌──────┐ ┌─────┐│   │
│  │          │   │ skip-list│   │ │Bloom │ │MLP  ││   │
│  └────┬─────┘   └────┬─────┘   │ │Filter│ │Learn││   │
│       │              │         │ └──────┘ └─────┘│   │
│       └──────┬───────┘         └────────┬────────┘   │
│              │                          │             │
│              ▼                          ▼             │
│  ┌──────────────────────────────────────────────┐    │
│  │              SSTable Layer                    │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │    │
│  │  │L0-SST│  │L0-SST│  │L0-SST│  │L0-SST│    │    │
│  │  └──────┘  └──────┘  └──────┘  └──────┘    │    │
│  │  ┌────────────────┐  ┌────────────────┐     │    │
│  │  │    L1-SST      │  │    L1-SST      │     │    │
│  │  └────────────────┘  └────────────────┘     │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build the project
cargo build --release

# Run a single benchmark
cargo run --release --bin bench -- run --workload A --keys 100000 --ops 1000000 --engine bloom

# Compare bloom vs learned
cargo run --release --bin bench -- compare --keys 100000 --ops 1000000

# Run criterion benchmarks
cargo bench

# Generate charts (requires matplotlib)
python3 assets/generate_charts.py benchmark_results.json
```

## Benchmark Results

| Workload | Bloom Throughput | Learned Throughput | Speedup | Bloom P50 | Learned P50 |
|----------|-----------------|-------------------|---------|-----------|-------------|
| A (50/50 R/W) | ~50K ops/s | ~65K ops/s | ~1.3x | 15 µs | 10 µs |
| B (95/5 R/W) | ~80K ops/s | ~110K ops/s | ~1.4x | 10 µs | 7 µs |
| C (100% R) | ~90K ops/s | ~125K ops/s | ~1.4x | 8 µs | 6 µs |
| D (95R/5I) | ~75K ops/s | ~100K ops/s | ~1.3x | 12 µs | 9 µs |

## Technical Deep-Dive

### The Learned Index Concept

Traditional Bloom filters answer "is this key NOT here?" with a fixed false positive rate.
A learned index answers "look HERE" by predicting the byte offset of a key in the SSTable.

```
Bloom Filter:  key → {maybe, no}    → scan entire block
Learned Index: key → [start, end]   → scan small window
```

### MLP Architecture

```
Input (1)  →  Hidden (32, ReLU)  →  Output (1)
   key           w1[32], b1[32]       predicted offset
```

- **Training**: SGD on sorted key→offset pairs, ~100 epochs, lr=0.01
- **Quantization**: INT8 weights for fast inference (1KB per model)
- **Error bound**: Max prediction error tracked → search window guarantee

### Why It Works

Keys in an SSTable are sorted. The mapping `key → byte_offset` is a monotonically
increasing function — exactly what a small MLP can learn. The error bound ensures
we never miss a key, and the search window is typically 5-10% of the SSTable.

## Comparison vs RocksDB

| Aspect | RocksDB | LSM-Engine |
|--------|---------|------------|
| Language | C++ | Rust |
| Memtable | Skip-list, B-tree, Hash | Skip-list |
| Index | Bloom filter | Bloom + Learned |
| Compaction | Level, Universal, FIFO | Size-tiered |
| WAL | Group commit | Sequential append |
| Maturity | Production (Facebook) | Research prototype |
| Innovation | Battle-tested | Learned indexes for SSTables |

LSM-Engine is not meant to replace RocksDB. It's a research prototype exploring whether
learned indexes can improve LSM-tree lookups. The answer: **yes, for read-heavy workloads
with sorted key distributions**.

## Project Structure

```
lsm-engine/
├── src/
│   ├── lib.rs              # Core types and config
│   ├── main.rs             # Entry point
│   ├── engine/
│   │   ├── mod.rs          # Engine orchestration
│   │   ├── memtable.rs     # Skip-list memtable
│   │   ├── sstable.rs      # SSTable with block index
│   │   ├── wal.rs          # Write-ahead log
│   │   ├── compaction.rs   # Size-tiered compaction
│   │   └── manifest.rs     # SSTable tracking
│   ├── index/
│   │   ├── mod.rs          # Index trait
│   │   ├── bloom.rs        # Bloom filter
│   │   └── learned.rs      # MLP learned index
│   ├── benchmark/
│   │   ├── mod.rs          # Benchmark runner
│   │   └── harness.rs      # YCSB workloads
│   └── bin/
│       └── bench.rs        # CLI benchmark tool
├── benches/
│   └── engine_bench.rs     # Criterion benchmarks
├── docs/
│   ├── architecture.md     # Architecture docs
│   ├── benchmarks.md       # Benchmark methodology
│   └── talking_points.md   # Interview talking points
└── assets/
    └── generate_charts.py  # Chart generation
```

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
