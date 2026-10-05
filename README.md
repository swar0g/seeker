# Seeker - Bitcoin Address Collision Seeker

A high-performance Rust application that searches for collisions in Bitcoin addresses by generating random secp256k1 keypairs, computing HASH160 (RIPEMD-160(SHA-256)) public key digests, and binary-searching a precomputed index of known Bitcoin address hashes.

It supports modular compute backends:
- **CPU Backend**: Multithreaded key derivation and hashing using `rayon` and thread-local `secp256k1` contexts.
- **CUDA Backend**: Feature-gated GPU acceleration using CUDA C kernels compiled at runtime via NVRTC — secp256k1 public keys are derived by *incremental key walking* (each GPU thread derives one random start key, then walks `k, k+1, k+2, …` by repeated point addition of G, normalizing accumulated points with one shared modular inversion per 256-point group), and SHA-256 / RIPEMD-160 also run on the GPU, with CPU verification sampling.

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
   - Generates `batch / walk_steps` random 32-byte *start* keys per batch using `getrandom`; walk `i` covers the consecutive secrets `start_i, start_i + 1, …, start_i + walk_steps − 1` (mod the curve order), so keys stay uniformly random as a batch while the GPU pays one full scalar multiplication per walk instead of per key.
   - Passes the walk starts to the configured `HashBackend` (`cpu` or `cuda`); the returned digests are in step-major order and a matched secret is reconstructed on the host as `(start_i + j) mod n`.
   - Checks generated 20-byte HASH160 public key digests against `AddressIndex` in parallel (`rayon`).
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
| `SEEKER_BATCH_SIZE` | `32768` (CPU) / `4194304` (CUDA) | Keys per batch, rounded down to a whole number of walks |
| `SEEKER_WALK_STEPS` | `256` | Consecutive keys per walk; each batch is split into `batch / walk_steps` walks |
| `SEEKER_BACKEND` | `cpu` | Backend choice (`cpu` or `cuda`) |
| `SEEKER_CUDA_VERIFY_SAMPLE` | `64` | Number of CPU verification samples per CUDA batch (`0` checks all entries) |
| `SEEKER_CUDA_VERIFY_SECP` | `1` | Set to `0` to disable CPU secp256k1 pubkey parity verification for CUDA |
| `SEEKER_CUDA_VERIFY_SHA256` | `1` | Set to `0` to disable CPU SHA-256 parity verification for CUDA |
| `SEEKER_CUDA_VERIFY_HASH160` | `1` | Set to `0` to disable CPU HASH160 parity verification for CUDA |

---

## Benchmarks

Measured 2026-10-05, release build (`cargo build --release --features cuda`), ~60 s runs (marginal rate between steady-state progress prints), index of 536,740 addresses.

**Hardware**: NVIDIA GeForce RTX 3060 Ti (8 GB GDDR6, 38 SMs, sm_86), AMD Ryzen 5 5600X (6C/12T), Windows 11, CUDA 13.3 (runtime NVRTC compilation), driver 616.56.

**Parameters**: end-to-end throughput counts keys fully processed per second (random start generation → public key derivation → SHA-256 + RIPEMD-160 → index check), with CUDA verification sampling at its default of 64 keys per batch unless noted otherwise.

| Configuration | Parameters | Throughput |
|---|---|---|
| CUDA, incremental walk (default) | `SEEKER_BATCH_SIZE=4194304`, `SEEKER_WALK_STEPS=256`, sampling on | **~35M keys/s** |
| CUDA, previous defaults (2^21 batch, 128 steps, 32-point inversion group) | `SEEKER_BATCH_SIZE=2097152`, `SEEKER_WALK_STEPS=128` | ~31.1M keys/s |
| CUDA, one scalar multiply per key | `SEEKER_WALK_STEPS=1`, verification disabled (fused hash kernel) | ~1.8M keys/s |
| CPU backend | `SEEKER_BATCH_SIZE=32768` | ~320K keys/s |

The end-to-end speedup over the pre-walk pipeline is ~128×. The pipeline is host-bound now: per 1M keys the GPU spends ~4.3 ms in the walk kernel, ~1.7 ms in SHA-256 and ~0.6 ms in RIPEMD-160 (both hash kernels are fully unrolled, so their message words stay in registers), leaving the device idle for roughly three quarters of the wall time — the remaining bottleneck is the host-side digest scan and per-batch transfers, not the EC stage.

### EC stage only (nsys kernel medians)

| Derivation strategy | Time per 1M keys | Public keys/s |
|---|---|---|
| Incremental walk (default parameters) | ~4.3 ms | ~232M |
| Per-key w=8 scalar multiply + inversion (`SEEKER_WALK_STEPS=1`) | ~378 ms | ~2.6M |

That is a **~88× speedup of the EC stage**: one full scalar multiply (248 doublings, 31 mixed additions, one Fermat inversion) per *walk* instead of per key, ~7 field multiplications + one shared group inversion per key (a single Fermat inversion amortized over `WALK_GROUP = 256` walked keys), plus SHA-256/RIPEMD-160 unchanged.

### Inversion-group sweep (end-to-end, marginal rates, ~16K walker threads)

The walk kernel batch-normalizes accumulated Jacobian points with Montgomery's trick: one Fermat inversion (~255 squarings) shared across `WALK_GROUP` consecutive points. Widening the group amortizes the inversion further but grows the per-thread local-memory working set (4 × `WALK_GROUP` × 32 B), so the win needs the walk length to follow the group:

| `WALK_GROUP` | Batch size | Steps/walk | End-to-end |
|---|---|---|---|
| 32 (pre-2026-10-05) | 2^21 | 128 | ~31.1M keys/s |
| 128 | 2^21 | 128 | ~29.8M keys/s |
| **256 (default)** | 2^22 | 256 | **~34.5M keys/s** |
| 512 | 2^23 | 512 | ~34.4M keys/s |

Group 128 alone (walk length unchanged) measures *below* the 32-point baseline: the larger per-thread working set falls out of L1 before the back-substitution pass re-reads it. Group 256 with matching 256-step walks gains ~11% end-to-end (the EC kernel itself improves ~1.4×, from ~6.0 to ~4.3 ms per 1M keys); group 512 is within noise of 256 while doubling the batch buffers, so 256 ships. ~16K walker threads remains the sweet spot for the walk kernel's occupancy-vs-amortization trade-off.

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
