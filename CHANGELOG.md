# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release
- LSM-tree key-value storage engine
- Learned index implementation (MLP-based)
- Bloom filter baseline implementation
- YCSB-style benchmark harness
- CLI tool for running benchmarks
- Comprehensive documentation

### Technical Details
- **Memtable**: Skip-list based in-memory sorted structure
- **SSTable**: Block-based format with block index
- **WAL**: Write-ahead log for crash recovery
- **Compaction**: Size-tiered compaction strategy
- **Learned Index**: Single hidden layer MLP with INT8 quantization
- **Error Bounding**: Prediction error tracking with fallback to binary search

## [0.1.0] - 2024-01-01

### Added
- Core engine implementation
- Basic benchmark infrastructure
- Initial documentation

---

## Version History

- **0.1.0**: Initial release with core functionality
- **Unreleased**: Development version with latest features

## Future Plans

- [ ] SIMD-optimized inference path
- [ ] Adaptive fallback mechanisms
- [ ] Range scan support
- [ ] Multi-threaded compaction
- [ ] Additional benchmark datasets
- [ ] Performance optimizations
- [ ] Extended documentation
