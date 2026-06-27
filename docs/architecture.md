# LSM-Engine Architecture

## Overview

LSM-Engine is a key-value store implementing a Log-Structured Merge-tree (LSM-tree) with
**learned index models** replacing traditional Bloom filters for SSTable lookups.

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
│  │  ┌──────────────────────────────────┐       │    │
│  │  │           L2-SST                  │       │    │
│  │  └──────────────────────────────────┘       │    │
│  └──────────────────────────────────────────────┘    │
│              │                                        │
│              ▼                                        │
│  ┌──────────────────┐                                │
│  │  Compaction      │                                │
│  │  (size-tiered)   │                                │
│  └──────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Write-Ahead Log (WAL)

The WAL provides durability. Every write is first appended to the WAL before being applied
to the memtable. On recovery, the WAL is replayed to reconstruct the memtable.

**Format:**
```
[key_len:u32][key:bytes][has_value:u8][value_len:u32][value:bytes][sequence:u64]
```

### 2. Memtable

An in-memory sorted data structure implemented as a skip-list. Supports:
- **put**: Insert or update a key-value pair
- **get**: Point lookup with O(log n) complexity
- **delete**: Insert a tombstone marker
- **range_scan**: Iterate over a key range

When the memtable exceeds the configured size threshold, it is flushed to an SSTable.

### 3. SSTable (Sorted String Table)

An immutable, sorted file on disk. The SSTable uses a block-based format:

```
┌─────────────────────────────────────────────┐
│              Data Block 0                    │
│  [num_entries:u32][entry0][entry1]...       │
├─────────────────────────────────────────────┤
│              Data Block 1                    │
│  [num_entries:u32][entry0][entry1]...       │
├─────────────────────────────────────────────┤
│                   ...                        │
├─────────────────────────────────────────────┤
│              Block Index                     │
│  [num_blocks:u32][first_key][offset]...     │
├─────────────────────────────────────────────┤
│           Index Data                         │
│  (Bloom filter or Learned index bytes)       │
├─────────────────────────────────────────────┤
│              Footer                          │
│  [version][index_type][min_key][max_key]     │
│  [block_index_offset][index_offset]          │
│  [index_size][magic]                         │
└─────────────────────────────────────────────┘
```

### 4. Index Layer

Two index implementations share a common `Index` trait:

#### Bloom Filter (Baseline)
- Standard Bloom filter with configurable bits-per-element
- Returns "maybe present" or "definitely not present"
- False positive rate: ~1% at 10 bits/element

#### Learned Index (Core Innovation)
- Single-hidden-layer MLP (32 neurons, ReLU activation)
- Input: normalized key value (f64)
- Output: predicted byte offset in SSTable
- **Training**: SGD on sorted key array, ~100 epochs, lr=0.01
- **INT8 Quantization**: Weights quantized to INT8 for fast inference
- **Error Bounding**: Max prediction error tracked during training
- **Lookup**: Predict offset ± error → search window with binary search

```
                Learned Index Lookup Flow
                
   key ──► normalize ──► MLP forward pass ──► predicted offset
                                                    │
                                          ┌─────────┴─────────┐
                                          │   ± max_error      │
                                          └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │  Search Window     │
                                          │  [start, end]      │
                                          └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │  Binary Search     │
                                          │  within window     │
                                          └───────────────────┘
```

### 5. Compaction

Size-tiered compaction strategy:
- Level 0 accumulates flushed memtables
- When Level 0 reaches capacity, merge into Level 1
- Tombstones are dropped when safe (no older levels contain the key)
- Merging is a sorted merge of all entries

### 6. Manifest

JSON-based tracking of all SSTables:
```json
{
  "version": 1,
  "entries": [
    {
      "id": 1,
      "level": 0,
      "min_key": "...",
      "max_key": "...",
      "size_bytes": 1024,
      "index_type": "Bloom"
    }
  ]
}
```

## Data Flow

### Write Path
```
Client PUT(key, value)
  │
  ├─► Write to WAL
  │
  ├─► Insert into Memtable
  │
  └─► If memtable full:
       ├─► Flush Memtable → SSTable (with index)
       ├─► Clear WAL
       └─► Maybe trigger compaction
```

### Read Path
```
Client GET(key)
  │
  ├─► Check Memtable
  │     ├─► Found → return value
  │     └─► Tombstone → return NotFound
  │
  └─► Search SSTables (newest → oldest)
        ├─► Check key range
        ├─► Query index (Bloom/Learned)
        │     ├─► NotFound → skip SSTable
        │     └─► SearchRange → binary search
        └─► Found → return value
```
