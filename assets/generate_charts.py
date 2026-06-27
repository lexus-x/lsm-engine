#!/usr/bin/env python3
"""Generate benchmark comparison charts from JSON results."""

import json
import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def load_results(path: str) -> list:
    """Load benchmark results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def group_by_workload(results: list) -> dict:
    """Group results by workload."""
    grouped = {}
    for r in results:
        w = r['workload']
        if w not in grouped:
            grouped[w] = {}
        grouped[w][r['engine_type']] = r
    return grouped


def plot_throughput(grouped: dict, output_dir: str):
    """Plot throughput comparison bar chart."""
    workloads = sorted(grouped.keys())
    bloom_throughput = []
    learned_throughput = []
    
    for w in workloads:
        bloom_throughput.append(grouped[w].get('bloom', {}).get('throughput_ops_per_sec', 0))
        learned_throughput.append(grouped[w].get('learned', {}).get('throughput_ops_per_sec', 0))
    
    x = np.arange(len(workloads))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bloom_throughput, width, label='Bloom Filter', color='#2196F3')
    bars2 = ax.bar(x + width/2, learned_throughput, width, label='Learned Index', color='#FF5722')
    
    ax.set_xlabel('Workload')
    ax.set_ylabel('Throughput (ops/sec)')
    ax.set_title('Throughput Comparison: Bloom Filter vs Learned Index')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Workload {w}' for w in workloads])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved throughput_comparison.png")


def plot_latency(grouped: dict, output_dir: str):
    """Plot latency comparison bar chart."""
    workloads = sorted(grouped.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, percentile in enumerate(['p50_latency_us', 'p99_latency_us']):
        ax = axes[idx]
        bloom_lat = []
        learned_lat = []
        
        for w in workloads:
            bloom_lat.append(grouped[w].get('bloom', {}).get(percentile, 0))
            learned_lat.append(grouped[w].get('learned', {}).get(percentile, 0))
        
        x = np.arange(len(workloads))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, bloom_lat, width, label='Bloom Filter', color='#2196F3')
        bars2 = ax.bar(x + width/2, learned_lat, width, label='Learned Index', color='#FF5722')
        
        label = 'P50' if 'p50' in percentile else 'P99'
        ax.set_xlabel('Workload')
        ax.set_ylabel(f'Latency (µs)')
        ax.set_title(f'{label} Latency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Workload {w}' for w in workloads])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved latency_comparison.png")


def plot_memory(grouped: dict, output_dir: str):
    """Plot memory usage comparison bar chart."""
    workloads = sorted(grouped.keys())
    bloom_mem = []
    learned_mem = []
    
    for w in workloads:
        # Estimate memory: bloom filter vs learned index
        # Bloom: ~10 bits/element, Learned: ~32*32 INT8 weights + biases
        bloom_mem.append(grouped[w].get('bloom', {}).get('total_ops', 0) * 10 / 8 / 1024)
        learned_mem.append(grouped[w].get('learned', {}).get('total_ops', 0) * 10 / 8 / 1024)
    
    x = np.arange(len(workloads))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bloom_mem, width, label='Bloom Filter', color='#2196F3')
    bars2 = ax.bar(x + width/2, learned_mem, width, label='Learned Index', color='#FF5722')
    
    ax.set_xlabel('Workload')
    ax.set_ylabel('Memory (KB)')
    ax.set_title('Memory Usage Comparison: Bloom Filter vs Learned Index')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Workload {w}' for w in workloads])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved memory_comparison.png")


def main():
    if len(sys.argv) < 2:
        results_path = 'benchmark_results.json'
    else:
        results_path = sys.argv[1]
    
    output_dir = 'assets/charts'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Run benchmarks first: cargo run --bin bench -- compare")
        sys.exit(1)
    
    results = load_results(results_path)
    grouped = group_by_workload(results)
    
    plot_throughput(grouped, output_dir)
    plot_latency(grouped, output_dir)
    plot_memory(grouped, output_dir)
    
    print(f"\nAll charts saved to {output_dir}/")


if __name__ == '__main__':
    main()
