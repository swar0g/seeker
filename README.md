# Seeker - Bitcoin Address Collision Seeker

A high-performance Rust application that searches for collisions in Bitcoin addresses by generating random secp256k1 keypairs, computing HASH160 (RIPEMD-160(SHA-256)) public key digests, and binary-searching a precomputed index of known Bitcoin address hashes.

It supports modular compute backends:
- **CPU Backend**: Multithreaded key derivation and hashing using `rayon` and thread-local `secp256k1` contexts.
- **CUDA Backend**: Feature-gated GPU acceleration using CUDA C kernels compiled at runtime via NVRTC — secp256k1 public key derivation, SHA-256 and RIPEMD-160 all run on the GPU, with CPU verification sampling.

---

## Architecture & How It Works

1. **Address Index Construction (`build_index`)**:
   - Reads P2PKH Bitcoin addresses in Base58Check format from `addresses.txt`.
   - Validates length, network prefix (mainnet `0x00`), and Base58 checksums.
   - Converts addresses to 20-byte HASH160 values.
   - Parallel-sorts (`rayon`) and deduplicates entries.
   - Outputs a binary file `addresses.bin` with a fixed-length memory-mapped layout.

2. **Address Index Memory Mapping (`AddressIndex`)**:
   - Binary header structure:
     - `0..8`: Magic bytes `BTCIDX01`
     - `8..16`: Number of entries (`u64` little-endian)
     - `16..31`: Reserved padding (total header size = 32 bytes)
     - `32..`: Contiguous 20-byte HASH160 entries sorted in ascending order
   - Fast $O(\log N)$ binary search lookup over the memory-mapped file (`memmap2`).

3. **Collision Seeking Loop (`seeker`)**:
   - Generates random 32-byte secret keys in batches using `getrandom`.
   - Passes secret keys to the configured `HashBackend` (`cpu` or `cuda`).
   - Checks generated 20-byte HASH160 public key digests against `AddressIndex`.
   - Reports throughput statistics (~1M attempts interval) and outputs private key if a collision occurs.

---

## Binary Targets & CLI Usage

### 1. `build_index`
Converts `addresses.txt` into `addresses.bin`.

```bash
# Prepare addresses.txt (one Base58Check address per line)
cargo run --release --bin build_index
```

### 2. `seeker`
Main collision detection application. Requires `addresses.bin` in the working directory.

```bash
# Run with default CPU backend
cargo run --release

# Run with CUDA backend enabled (requires CUDA toolchain and drivers)
cargo run --release --features cuda
```

---

## Features & Configuration

### Features (`Cargo.toml`)
- `cpu` (default): Standard CPU hashing pipeline.
- `cuda`: Enables `cudarc` and runtime NVRTC CUDA compilation for GPU key derivation and hashing.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SEEKER_BATCH_SIZE` | `32768` | Number of secret keys per batch |
| `SEEKER_BACKEND` | `cpu` | Backend choice (`cpu` or `cuda`) |
| `SEEKER_CUDA_VERIFY_SAMPLE` | `64` | Number of CPU verification samples per CUDA batch (`0` checks all entries) |
| `SEEKER_CUDA_VERIFY_SECP` | `1` | Set to `0` to disable CPU secp256k1 pubkey parity verification for CUDA |
| `SEEKER_CUDA_VERIFY_SHA256` | `1` | Set to `0` to disable CPU SHA-256 parity verification for CUDA |
| `SEEKER_CUDA_VERIFY_HASH160` | `1` | Set to `0` to disable CPU HASH160 parity verification for CUDA |

---

## Docker Support

Build and run using Docker or Docker Compose:

```bash
# Build Docker image
docker build -t seeker .

# Run container (requires addresses.bin mounted into working directory)
docker run --rm -v $(pwd)/addresses.bin:/addresses.bin seeker

# Or using docker-compose
docker compose up
```

---

## Testing & Verification

Run tests:

```bash
# Run standard test suite
cargo test

# Run single reference match test
cargo test batch_hash_matches_legacy_reference

# Run CUDA tests (requires CUDA environment)
cargo test --features cuda
```

---

## Mathematical & Practical Context

- Total possible 160-bit Bitcoin address hashes: $2^{160} \approx 1.46 \times 10^{48}$.
- Finding a collision against even hundreds of millions of target addresses is theoretically infeasible with classical compute resources. This software is built for research, performance benchmarking, and educational exploration of cryptographic pipelines and GPU hardware acceleration.

---

## License

[MIT License](LICENSE)

## Acknowledgements

- [madler/bitprim](https://github.com/madler/bitprim) - Used for the secp256k1 implementation.
- [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) (MIT) - Source of the CUDA secp256k1 field arithmetic and fixed-window generator multiply vendored into `src/backend/cuda_kernels.cu`.
