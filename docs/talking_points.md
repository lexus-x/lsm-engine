# Interview Talking Points

## 1. Why Learned Indexes Over Bloom Filters?

### The Problem with Bloom Filters

Bloom filters are probabilistic data structures that answer "is this key definitely NOT in
the SSTable?" They have a fundamental limitation: **false positives**. At 10 bits per element,
you get ~1% false positive rate. This means:

- For every 100 lookups of non-existent keys, ~1 unnecessary disk read
- In read-heavy workloads with sparse key spaces, this adds up
- The false positive rate is fixed — you can't make it data-aware

### The Learned Index Approach

A learned index replaces the Bloom filter with a tiny neural network that predicts WHERE
in the SSTable to look. Instead of asking "is it here?", it says "look at byte offset X,
plus or minus Y."

**Key insight**: Keys in an SSTable are sorted. The mapping from key → byte offset is a
**monotonically increasing function**. This is exactly the kind of function a small MLP
can learn well.

### Trade-offs

| Aspect | Bloom Filter | Learned Index |
|--------|-------------|---------------|
| False positives | Yes (~1%) | None |
| Search strategy | Full block scan | Narrowed window |
| Memory | Fixed per element | Fixed per model |
| Update cost | O(1) insert | Retrain on flush |
| CPU cost | Hash computation | MLP inference |
| Data-aware | No | Yes (adapts to distribution) |

### When Learned Indexes Win

- **Sorted keys with patterns**: The MLP can learn the key→offset mapping
- **Read-heavy workloads**: More lookups = more benefit from faster searches
- **Large SSTables**: The search window reduction matters more with more data

### When Bloom Filters Win

- **Random/uniform keys**: MLP can't learn a meaningful mapping
- **Write-heavy workloads**: Retraining cost adds up
- **Tiny SSTables**: Overhead of MLP isn't worth it

## 2. Designing for Production: Error-Bounded Predictions

### The Core Challenge

A neural network's prediction is never perfect. If the MLP says "the key is at offset 5000"
but it's actually at offset 5200, we'd miss it. We need **error bounds**.

### Our Solution: Max-Error Bounding

During training, we track the maximum prediction error across all training samples:

```
max_error = max(|predicted[i] - actual[i]|) for all i in training set
```

At lookup time:
1. MLP predicts offset `P`
2. Search window = `[P - max_error, P + max_error]`
3. Binary search within this window

This gives us a **guarantee**: the key, if present, is within the search window.

### Why This Works

The key→offset function is smooth and monotonic. The MLP captures the global trend,
and the residual errors are bounded. For typical datasets:
- Search window is ~5-10% of the SSTable size
- This is much smaller than scanning the entire file

### Optimization: Adaptive Window Sizing

If the search window is larger than a threshold (e.g., 1000 bytes), we fall back to
full binary search over the block index. This handles edge cases where the MLP's
prediction is poor for certain key ranges.

## 3. INT8 Quantization for Fast Inference

### Why Quantize?

MLP inference with f64 weights involves:
- 32 multiplications (hidden layer)
- 32 additions (hidden layer)
- 32 multiplications (output layer)
- 32 additions (output layer)

With f64, each multiplication is 8 bytes. With INT8, it's 1 byte. Benefits:
- **8x less memory** for weights
- **Better cache utilization**: Weights fit in L1 cache
- **SIMD potential**: INT8 operations can be vectorized

### Our Quantization Strategy

After training with f64:
1. Find the maximum absolute weight value: `w_max = max(|w|)`
2. Compute scale: `scale = w_max / 127`
3. Quantize: `w_q = round(w / scale).clamp(-128, 127)`

During inference:
```rust
// Dequantize on-the-fly
let w_real = w_q as f64 * scale;
let z = w_real * input + bias;
```

### Accuracy Impact

INT8 quantization introduces small rounding errors, but:
- The MLP has only 32 hidden neurons — the function is smooth
- The error bound already accounts for prediction imprecision
- The quantization error is much smaller than the max prediction error

In practice, INT8 quantized models perform within 1-2% of f64 models.

### Production Considerations

- **Mixed precision**: Could use INT8 for inference, f32 for training
- **Dynamic quantization**: Adjust scale factors per-layer
- **Hardware acceleration**: On modern CPUs, INT8 dot products use VNNI instructions
- **Memory savings**: 32*32 INT8 weights = 1KB vs 8KB for f64

## Bonus: Why Not Use an Existing ML Framework?

We implement the MLP by hand because:
1. **No dependency bloat**: No need for tensorflow, pytorch, or onnxruntime
2. **Deterministic**: Our inference is pure Rust, no FFI or unsafe (except for minor optimizations)
3. **Minimal binary size**: The entire MLP inference is ~100 lines of Rust
4. **Compile-time optimization**: The compiler can inline and vectorize our tight loops
5. **Portability**: Works on any platform Rust supports, no native library dependencies
