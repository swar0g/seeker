# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands

```bash
# Build (release profile is required for meaningful performance)
cargo build --release

# Build with CUDA backend
cargo build --release --features cuda

# Run the collision seeker (requires addresses.bin in working directory)
cargo run --release

# Build the binary address index from a text file (addresses.txt → addresses.bin)
cargo run --release --bin build_index

# Run all tests
cargo test

# Run a single test
cargo test batch_hash_matches_legacy_reference

# Inspect CUDA kernel register usage / spills (needs nvcc + MSVC cl.exe via -ccbin)
nvcc -cubin -arch=sm_86 --ptxas-options=-v -ccbin <path-to-cl.exe> -o kernels.cubin src/backend/cuda_kernels.cu
```

## Architecture

### Backend abstraction (`src/backend/`)

The hashing pipeline is abstracted behind the `HashBackend` trait (`src/backend/mod.rs`):

```rust
fn hash_batch_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>>
fn hash_batch_walking(&self, starts: &[[u8; 32]], steps: usize) -> Result<Vec<[u8; 20]>>
```

`hash_batch_walking` hashes consecutive key walks: walk `i` covers the secrets `(starts[i] + j) mod n` for `j in 0..steps`, digest order is **step-major** (`digest[j * starts.len() + i]` belongs to `(starts[i] + j) mod n`), and the main loop reconstructs a matched secret with `add_small_mod_n` (`src/backend/mod.rs`, curve-order modular addition, verified against the curve group law in tests). The default implementation expands the walks into explicit keys; the CPU backend uses it unchanged (still one scalar multiplication per key).

Two implementations exist:
- **`CpuBackend`** (`src/backend/cpu.rs`): delegates to the CPU path in `main.rs` using `rayon`-friendly thread-local `Secp256k1` contexts.
- **`CudaBackend`** (`src/backend/cuda.rs`, feature-gated `cuda`): runs the whole pipeline on GPU, compiled at runtime via NVRTC. Kernel source lives in `src/backend/cuda_kernels.cu` (embedded with `include_str!`; the field/point arithmetic is vendored from UltrafastSecp256k1, MIT). Two derivation kernels exist:
  - `secp256k1_pubkey33_batch` (per-key): one fixed-base w=8 windowed multiply over the precomputed `[0..255]*G` `__constant__` table plus one Fermat inversion per key. Used by `hash_batch_from_secret_bytes` (self-check, tests).
  - `secp256k1_walk33_batch` (incremental, used by `hash_batch_walking`): each thread derives its start key once, then walks `k, k+1, k+2, …` by repeated Jacobian mixed addition of G (7M+4S per key). `WALK_CHUNK = 32` consecutive points are collected in Jacobian form and their Z coordinates are normalized with a single shared inversion (Montgomery batch inversion). Measures ~63× the per-key kernel's EC throughput at the default 128 steps per walk.

  Device buffers persist across batches: a walking batch costs one bulk host→device upload of the packed walk starts (`batch / steps` keys, not the whole batch) and one bulk device→host download of the HASH160 digests; public keys and intermediate SHA-256 digests are chained on the device and never touch host memory. All kernels are enqueued before any blocking copy. Verification sampling per batch: contiguous front sample (pubkeys / SHA-256, cheap D2H) plus a spread sample checked against the already-downloaded HASH160 digests, so walk steps across all threads are covered end to end. Launch config: 128 threads/block for the derivation kernels (per their `__launch_bounds__`), 256 for the hash kernels.

Backend selection: `create_default_backend()` picks CPU by default; set `SEEKER_BACKEND=cuda` to force CUDA.

### Address index format (`src/main.rs` — `AddressIndex`)

The binary file `addresses.bin` is memory-mapped and has a fixed layout:
- Bytes 0–7: magic `BTCIDX01`
- Bytes 8–15: entry count as little-endian `u64`
- Bytes 16–31: reserved (padding to 32-byte header)
- Bytes 32+: packed 20-byte HASH160 values, **sorted ascending** for binary search

`build_index` (`src/bin/build_index.rs`) reads `addresses.txt` (one Base58Check P2PKH address per line), strips checksums, parallel-sorts with `rayon`, deduplicates, and writes this format.

### Main loop (`src/main.rs`)

Each iteration generates `batch / walk_steps` random 32-byte start keys, calls `hash_batch_walking(starts, walk_steps)`, then checks all digests against the index in parallel (`rayon`, first match wins). On a match the secret is reconstructed as `(starts[idx % walks] + idx / walks) mod n` and printed. Prints throughput every ~1 M attempts. Measured on an RTX 3060 Ti with default settings: ~32 M keys/s (CUDA) vs ~320 K keys/s (CPU); the remaining CUDA bottleneck is the SHA-256 kernel (its `w[64]` message schedule spills to local memory) and per-batch host↔device sync, not the EC stage.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SEEKER_BATCH_SIZE` | `32768` (CPU) / `2097152` (CUDA) | Keys per batch (rounded down to a multiple of the walk length) |
| `SEEKER_WALK_STEPS` | `128` | Consecutive keys per walk; batch is split into `batch / steps` walks |
| `SEEKER_BACKEND` | CPU | Set to `cuda` to use CUDA backend |
| `SEEKER_CUDA_VERIFY_SAMPLE` | `64` | CPU-check this many keys per batch for CUDA parity (`0` = all) |
| `SEEKER_CUDA_VERIFY_SECP` | enabled | Set to `0` to skip GPU pubkey vs CPU derivation verification |
| `SEEKER_CUDA_VERIFY_SHA256` | enabled | Set to `0` to skip SHA-256 GPU verification |
| `SEEKER_CUDA_VERIFY_HASH160` | enabled | Set to `0` to skip HASH160 GPU verification |
