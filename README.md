# Seeker - Bitcoin Address Collision Seeker

A high-performance Rust application that searches for collisions in Bitcoin addresses by generating random secp256k1 keypairs, computing HASH160 (RIPEMD-160(SHA-256)) public key digests, and binary-searching a precomputed index of known Bitcoin address hashes.

It supports modular compute backends:
- **CPU Backend**: Multithreaded key derivation and hashing using `rayon` and thread-local `secp256k1` contexts.
- **CUDA Backend**: Feature-gated GPU acceleration using CUDA C kernels compiled at runtime via NVRTC — secp256k1 public keys are derived by *incremental key walking* (each GPU thread derives one random start key, then walks `k, k+1, k+2, …` by repeated point addition of G, normalizing accumulated points with one shared modular inversion per chunk), and SHA-256 / RIPEMD-160 also run on the GPU, with CPU verification sampling.

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
| `SEEKER_BATCH_SIZE` | `32768` (CPU) / `2097152` (CUDA) | Keys per batch, rounded down to a whole number of walks |
| `SEEKER_WALK_STEPS` | `128` | Consecutive keys per walk; each batch is split into `batch / walk_steps` walks |
| `SEEKER_BACKEND` | `cpu` | Backend choice (`cpu` or `cuda`) |
| `SEEKER_CUDA_VERIFY_SAMPLE` | `64` | Number of CPU verification samples per CUDA batch (`0` checks all entries) |
| `SEEKER_CUDA_VERIFY_SECP` | `1` | Set to `0` to disable CPU secp256k1 pubkey parity verification for CUDA |
| `SEEKER_CUDA_VERIFY_SHA256` | `1` | Set to `0` to disable CPU SHA-256 parity verification for CUDA |
| `SEEKER_CUDA_VERIFY_HASH160` | `1` | Set to `0` to disable CPU HASH160 parity verification for CUDA |

---

## Benchmarks

Measured 2026-10-04, release build (`cargo build --release --features cuda`), ~30 s runs, index of 536,740 addresses.

**Hardware**: NVIDIA GeForce RTX 3060 Ti (8 GB GDDR6, 38 SMs, sm_86), AMD Ryzen 5 5600X (6C/12T), Windows 11, CUDA 13.3 (runtime NVRTC compilation), driver 616.56.

**Parameters**: end-to-end throughput counts keys fully processed per second (random start generation → public key derivation → SHA-256 + RIPEMD-160 → index check), with CUDA verification sampling at its default of 64 keys per batch unless noted otherwise.

| Configuration | Parameters | Throughput |
|---|---|---|
| CUDA, incremental walk (default) | `SEEKER_BATCH_SIZE=2097152`, `SEEKER_WALK_STEPS=128`, sampling on | **~32.1M keys/s** |
| CUDA, one scalar multiply per key | `SEEKER_WALK_STEPS=1`, verification disabled (fused hash kernel) | ~1.8M keys/s |
| CPU backend | `SEEKER_BATCH_SIZE=32768` | ~320K keys/s |

The end-to-end speedup over the pre-walk pipeline is ~119×. The pipeline is no longer EC-bound: per 1M keys the GPU spends ~6 ms in the walk kernel, ~11.3 ms in SHA-256 (its `w[64]` message schedule still spills to local memory), and ~3.6 ms in RIPEMD-160.

### EC stage only (nsys kernel medians)

| Derivation strategy | Time per 1M keys | Public keys/s |
|---|---|---|
| Incremental walk (default parameters) | ~6.0 ms | ~168M |
| Per-key w=8 scalar multiply + inversion (`SEEKER_WALK_STEPS=1`) | ~378 ms | ~2.6M |

That is a **~63× speedup of the EC stage**: one full scalar multiply (248 doublings, 31 mixed additions, one Fermat inversion) per *walk* instead of per key, ~7 field multiplications + a shared chunk inversion per key, plus SHA-256/RIPEMD-160 unchanged.

### Walk parameter sweep (EC kernel only, per 1M keys)

| Batch size | Steps/walk | Walker threads | Kernel time |
|---|---|---|---|
| 2^20 | 16 | 65,536 | 22.7 ms |
| 2^20 | 32 | 32,768 | 13.4 ms |
| 2^20 | 64 | 16,384 | 8.2 ms |
| 2^20 | 128 | 8,192 | 23.3 ms |
| 2^21 (default) | 128 | 16,384 | **6.0 ms** |

The walker count trades kernel occupancy against scalar-multiply amortization: fewer walkers underutilize the SMs, more walkers spend proportionally more time in per-walk scalar multiplies. ~16K walkers is the measured sweet spot on this GPU.

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
