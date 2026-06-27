# Benchmark Methodology

## Overview

The benchmark suite implements YCSB (Yahoo! Cloud Serving Benchmark) style workloads
to evaluate the performance of Bloom filter vs. Learned Index approaches.

## Workloads

| Workload | Read | Update | Insert | Distribution | Description |
|----------|------|--------|--------|--------------|-------------|
| A        | 50%  | 50%    | 0%     | Zipfian      | Read-heavy with updates |
| B        | 95%  | 5%     | 0%     | Zipfian      | Read-dominated |
| C        | 100% | 0%     | 0%     | Zipfian      | Read-only |
| D        | 95%  | 0%     | 5%     | Latest       | Read with inserts |

## Key Distributions

### Zipfian
- Models real-world access patterns where some keys are "hot"
- θ = 0.99 (skewness parameter)
- Approximates the 80/20 rule: ~20% of keys account for ~80% of accesses

### Uniform
- Equal probability for all keys
- Used as a baseline comparison

### Latest
- Newer keys are more likely to be accessed
- Models time-series or log data patterns

## Metrics

### Throughput
- Measured in operations per second (ops/sec)
- Total operations / total elapsed time

### Latency Percentiles
- **P50**: Median latency (50th percentile)
- **P99**: Tail latency (99th percentile)
- Measured in microseconds (µs)

### Memory Usage
- Peak memory consumption during benchmark
- Includes memtable, indexes, and cached SSTable blocks

## Expected Results

### Throughput Comparison

```
Workload    Bloom (ops/s)    Learned (ops/s)    Speedup
────────────────────────────────────────────────────────
A           ~50,000          ~65,000            ~1.3x
B           ~80,000          ~110,000           ~1.4x
C           ~90,000          ~125,000           ~1.4x
D           ~75,000          ~100,000           ~1.3x
```

### Latency Comparison

```
Workload    Bloom P50    Bloom P99    Learned P50    Learned P99
────────────────────────────────────────────────────────────────
A           15 µs        120 µs       10 µs          80 µs
B           10 µs        90 µs        7 µs           60 µs
C           8 µs         75 µs        6 µs           50 µs
D           12 µs        100 µs       9 µs           70 µs
```

### Why Learned Index Wins

1. **Smaller search window**: Instead of scanning the entire SSTable, the learned index
   narrows the search to a small window around the predicted offset.

2. **No false positives**: Bloom filters can say "maybe present" for keys that don't exist,
   leading to unnecessary disk reads. The learned index never produces false positives.

3. **Compact representation**: A 32-neuron MLP with INT8 weights is smaller than a Bloom
   filter at 10 bits/element for large datasets.

4. **CPU cache friendly**: The MLP inference is a tight loop over small weight arrays,
   which fits well in L1/L2 cache.

## Running Benchmarks

```bash
# Single workload
cargo run --bin bench -- run --workload A --keys 100000 --ops 1000000 --engine bloom
cargo run --bin bench -- run --workload A --keys 100000 --ops 1000000 --engine learned

# Full comparison
cargo run --bin bench -- compare --keys 100000 --ops 1000000

# Criterion benchmarks
cargo bench
```
