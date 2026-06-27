# Contributing to LSM-Engine

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/lsm-engine.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes
5. Run tests: `cargo test`
6. Run clippy: `cargo clippy -- -D warnings`
7. Format code: `cargo fmt`
8. Commit with a descriptive message
9. Push and create a Pull Request

## Development Setup

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build

# Test
cargo test

# Run benchmarks
cargo run --release --bin bench -- compare --keys 100000 --ops 1000000
```

## Code Style

- Follow Rust naming conventions
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Add doc comments for public APIs
- Write tests for new functionality

## Areas for Contribution

- **Performance**: SIMD optimizations, cache-aware data structures
- **ML Models**: Alternative model architectures, better quantization
- **Storage**: Range tombstones, prefix compression, bloom filter alternatives
- **Benchmarks**: New workloads, real-world datasets
- **Documentation**: Tutorials, architecture deep-dives

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure CI passes
4. Request review from maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.
