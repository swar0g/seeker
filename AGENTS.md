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
```

Two implementations exist:
- **`CpuBackend`** (`src/backend/cpu.rs`): delegates to the CPU path in `main.rs` using `rayon`-friendly thread-local `Secp256k1` contexts.
- **`CudaBackend`** (`src/backend/cuda.rs`, feature-gated `cuda`): runs the whole pipeline on GPU — secp256k1 public key derivation (fixed-base w=8 windowed multiply over a precomputed `[0..255]*G` table in `__constant__` memory), then SHA-256 and RIPEMD-160, all compiled at runtime via NVRTC. Kernel source lives in `src/backend/cuda_kernels.cu` (embedded with `include_str!`; the derivation code is vendored from UltrafastSecp256k1, MIT). Device buffers persist across batches: each batch costs exactly one bulk host→device upload of the packed secret keys and one bulk device→host download of the HASH160 digests; public keys and intermediate SHA-256 digests are chained on the device and never touch host memory. All kernels are enqueued before any blocking copy, so the GPU stays busy for the whole batch. Includes per-batch CPU verification sampling of GPU-derived public keys, SHA-256 and HASH160 digests (only the sampled bytes are copied back). Launch config: 128 threads/block for the derivation kernel (per its `__launch_bounds__`), 256 for the hash kernels.

Backend selection: `create_default_backend()` picks CPU by default; set `SEEKER_BACKEND=cuda` to force CUDA.

### Address index format (`src/main.rs` — `AddressIndex`)

The binary file `addresses.bin` is memory-mapped and has a fixed layout:
- Bytes 0–7: magic `BTCIDX01`
- Bytes 8–15: entry count as little-endian `u64`
- Bytes 16–31: reserved (padding to 32-byte header)
- Bytes 32+: packed 20-byte HASH160 values, **sorted ascending** for binary search

`build_index` (`src/bin/build_index.rs`) reads `addresses.txt` (one Base58Check P2PKH address per line), strips checksums, parallel-sorts with `rayon`, deduplicates, and writes this format.

### Main loop (`src/main.rs`)

Generates random 32-byte secret keys in batches, calls the active backend to produce HASH160 values, then binary-searches the index. Prints throughput every ~1 M attempts. `SEEKER_BATCH_SIZE` env var controls batch size (default 32768).

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SEEKER_BATCH_SIZE` | `32768` | Keys per batch |
| `SEEKER_BACKEND` | CPU | Set to `cuda` to use CUDA backend |
| `SEEKER_CUDA_VERIFY_SAMPLE` | `64` | CPU-check this many keys per batch for CUDA parity (`0` = all) |
| `SEEKER_CUDA_VERIFY_SECP` | enabled | Set to `0` to skip GPU pubkey vs CPU derivation verification |
| `SEEKER_CUDA_VERIFY_SHA256` | enabled | Set to `0` to skip SHA-256 GPU verification |
| `SEEKER_CUDA_VERIFY_HASH160` | enabled | Set to `0` to skip HASH160 GPU verification |
