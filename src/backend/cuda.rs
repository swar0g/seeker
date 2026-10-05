use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use ripemd::Ripemd160;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use sha2::Digest;

use super::HashBackend;

thread_local! {
    static SECP: Secp256k1<secp256k1::SignOnly> = Secp256k1::signing_only();
}

const SECP256K1_KERNEL_NAME: &str = "secp256k1_pubkey33_batch";
const SECP_WALK_KERNEL_NAME: &str = "secp256k1_walk33_batch";
const SHA256_KERNEL_NAME: &str = "sha256_pubkey33_batch";
const RIPEMD160_KERNEL_NAME: &str = "ripemd160_sha256_batch";
const HASH160_KERNEL_NAME: &str = "hash160_pubkey33_batch";

/// CUDA kernel source, compiled through NVRTC at startup. Kept in a separate
/// file so it can also be compiled directly with nvcc, e.g. to inspect
/// register usage with `nvcc -cubin --ptxas-options=-v -arch=sm_86`.
///
/// Contains the whole GPU pipeline: secp256k1 public key derivation (vendored
/// from UltrafastSecp256k1, see the file header) plus the SHA-256 /
/// RIPEMD-160 / fused HASH160 kernels.
const CUDA_KERNEL_SRC: &str = include_str!("cuda_kernels.cu");

/// Block size for the secp256k1 derivation kernel. Its `__launch_bounds__`
/// declares a 128-thread maximum; two 128-thread blocks fit per SM within
/// the kernel's 230-register budget.
const SECP_BLOCK: u32 = 128;
/// Block size for the hash kernels. Measured faster than the 1024-thread
/// blocks of cudarc's `for_num_elems` default at these batch sizes.
const HASH_BLOCK: u32 = 256;

/// Persistent device-side scratch space for the hashing pipeline.
///
/// The buffers are allocated once for the largest batch seen so far and
/// reused for every batch afterwards, so steady-state operation performs
/// exactly one bulk host-to-device upload (the packed secret keys) and one
/// bulk device-to-host download (the HASH160 digests) per batch. Secret keys
/// are turned into public keys *on the GPU*, the SHA-256 intermediates are
/// chained on the device, and no `cudaMalloc`/`cudaFree` happens between
/// batches.
struct DeviceBuffers {
    /// Largest total key count hashed so far; sizes the pubkey, SHA and
    /// HASH160 buffers.
    total_capacity: usize,
    /// Largest walk-start (or plain secret key) count uploaded so far;
    /// sizes the packed secrets buffer.
    secrets_capacity: usize,
    /// `secrets_capacity * 32` bytes: packed 32-byte big-endian secret keys,
    /// or the walk start keys for the incremental kernel.
    secrets: CudaSlice<u8>,
    /// `total_capacity * 33` bytes: packed 33-byte compressed public keys,
    /// written by the secp256k1 kernels, consumed by the hash kernels.
    pubkeys: CudaSlice<u8>,
    /// `total_capacity * 32` bytes: SHA-256 intermediates, consumed on the device.
    sha: CudaSlice<u8>,
    /// `total_capacity * 20` bytes: HASH160 digests.
    hash160: CudaSlice<u8>,
    /// Host staging buffer reused by the final device-to-host download.
    hash160_host: Vec<u8>,
}

impl DeviceBuffers {
    fn empty(stream: &Arc<CudaStream>) -> Result<Self> {
        let alloc = |what: &str| {
            stream
                .null::<u8>()
                .map_err(|e| anyhow::anyhow!("failed to allocate CUDA {what} buffer: {e:?}"))
        };
        Ok(Self {
            total_capacity: 0,
            secrets_capacity: 0,
            secrets: alloc("secret key")?,
            pubkeys: alloc("pubkey")?,
            sha: alloc("SHA")?,
            hash160: alloc("HASH160")?,
            hash160_host: Vec::new(),
        })
    }
}

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    secp256k1_fn: CudaFunction,
    secp_walk_fn: CudaFunction,
    sha256_fn: CudaFunction,
    ripemd160_fn: CudaFunction,
    hash160_fn: CudaFunction,
    buffers: Mutex<DeviceBuffers>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0)
            .map_err(|e| anyhow::anyhow!("failed to open CUDA device 0: {e:?}"))?;

        let ptx = compile_ptx(CUDA_KERNEL_SRC)
            .map_err(|e| anyhow::anyhow!("failed to compile CUDA kernels via NVRTC: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| anyhow::anyhow!("failed to load CUDA PTX module: {e:?}"))?;

        let secp256k1_fn = module
            .load_function(SECP256K1_KERNEL_NAME)
            .map_err(|e| anyhow::anyhow!("failed to find CUDA secp256k1 kernel: {e:?}"))?;
        let secp_walk_fn = module
            .load_function(SECP_WALK_KERNEL_NAME)
            .map_err(|e| anyhow::anyhow!("failed to find CUDA secp256k1 walk kernel: {e:?}"))?;
        let sha256_fn = module
            .load_function(SHA256_KERNEL_NAME)
            .map_err(|e| anyhow::anyhow!("failed to find CUDA SHA-256 kernel: {e:?}"))?;
        let ripemd160_fn = module
            .load_function(RIPEMD160_KERNEL_NAME)
            .map_err(|e| anyhow::anyhow!("failed to find CUDA RIPEMD-160 kernel: {e:?}"))?;
        let hash160_fn = module
            .load_function(HASH160_KERNEL_NAME)
            .map_err(|e| anyhow::anyhow!("failed to find CUDA HASH160 kernel: {e:?}"))?;

        let stream = ctx.default_stream();
        let buffers = DeviceBuffers::empty(&stream)?;

        Ok(Self {
            ctx,
            secp256k1_fn,
            secp_walk_fn,
            sha256_fn,
            ripemd160_fn,
            hash160_fn,
            buffers: Mutex::new(buffers),
        })
    }

    /// Derives compressed public keys on the CPU. Used only for the
    /// verification samples; the full batch is derived on the GPU.
    fn derive_pubkeys_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 33]>> {
        secret_keys
            .par_iter()
            .map(|secret| {
                SECP.with(|secp| {
                    let key = SecretKey::from_byte_array(*secret)?;
                    let pubkey = PublicKey::from_secret_key(secp, &key).serialize();
                    Ok(pubkey)
                })
            })
            .collect()
    }

    fn verification_sample_limit(total: usize) -> usize {
        let configured = std::env::var("SEEKER_CUDA_VERIFY_SAMPLE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64);

        if configured == 0 {
            total
        } else {
            configured.min(total)
        }
    }

    fn verification_flags() -> (bool, bool, bool) {
        let verify_secp = std::env::var("SEEKER_CUDA_VERIFY_SECP").ok().as_deref() != Some("0");
        let verify_sha = std::env::var("SEEKER_CUDA_VERIFY_SHA256").ok().as_deref() != Some("0");
        let verify_hash160 =
            std::env::var("SEEKER_CUDA_VERIFY_HASH160").ok().as_deref() != Some("0");
        (verify_secp, verify_sha, verify_hash160)
    }

    /// Grows the persistent buffers when a batch larger than any seen before
    /// arrives (in practice only on the first batch). `starts` is the number
    /// of packed 32-byte keys / walk starts, `total` the total key count the
    /// hash kernels run over. The whole set is built before assigning so a
    /// failed allocation leaves existing buffers intact.
    fn ensure_buffers(
        stream: &Arc<CudaStream>,
        bufs: &mut DeviceBuffers,
        starts: usize,
        total: usize,
    ) -> Result<()> {
        if bufs.total_capacity >= total && bufs.secrets_capacity >= starts {
            return Ok(());
        }

        let total_cap = total.max(bufs.total_capacity);
        let starts_cap = starts.max(bufs.secrets_capacity);
        let alloc = |what: &str, len: usize| {
            stream
                .alloc_zeros(len)
                .map_err(|e| anyhow::anyhow!("failed to allocate CUDA {what} buffer: {e:?}"))
        };
        let grown = DeviceBuffers {
            total_capacity: total_cap,
            secrets_capacity: starts_cap,
            secrets: alloc("secret key", starts_cap * 32)?,
            pubkeys: alloc("pubkey", total_cap * 33)?,
            sha: alloc("SHA", total_cap * 32)?,
            hash160: alloc("HASH160", total_cap * 20)?,
            hash160_host: vec![0; total_cap * 20],
        };
        *bufs = grown;
        Ok(())
    }

    fn launch_cfg(count: usize, block: u32) -> LaunchConfig {
        let blocks = (count as u32).div_ceil(block);
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launches a batch kernel over the first `count` entries of `input`,
    /// writing `count` results to the front of `output`. The kernels bounds-
    /// check every access against `count`, so buffers sized for the persistent
    /// capacity are safe to pass in full.
    fn launch_batch_kernel(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        kernel_name: &str,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<u8>,
        count: usize,
        block: u32,
    ) -> Result<()> {
        let cfg = Self::launch_cfg(count, block);
        unsafe {
            stream
                .launch_builder(func)
                .arg(input)
                .arg(output)
                .arg(&(count as u32))
                .launch(cfg)
                .map_err(|e| {
                    anyhow::anyhow!("failed launching CUDA {kernel_name} kernel: {e:?}")
                })?;
        }
        Ok(())
    }

    /// CPU-checks sampled GPU-derived public keys against the host reference.
    fn verify_secp_sample(cpu_pubkeys: &[[u8; 33]], gpu_pubkeys: &[u8]) -> Result<()> {
        for (i, (cpu, gpu)) in cpu_pubkeys
            .iter()
            .zip(gpu_pubkeys.chunks_exact(33))
            .enumerate()
        {
            if cpu.as_slice() != gpu {
                bail!("CUDA secp256k1 pubkey mismatch at sampled index {i}");
            }
        }

        Ok(())
    }

    /// CPU-checks sampled GPU SHA-256 digests against the host reference.
    fn verify_sha_sample(pubkeys: &[[u8; 33]], gpu_sha_sample: &[u8]) -> Result<()> {
        for (i, (pubkey, gpu_digest)) in pubkeys
            .iter()
            .zip(gpu_sha_sample.chunks_exact(32))
            .enumerate()
        {
            if &sha2::Sha256::digest(pubkey)[..] != gpu_digest {
                bail!("CUDA SHA-256 mismatch at sampled index {i}");
            }
        }

        Ok(())
    }

    /// CPU-checks sampled GPU HASH160 digests against the host reference.
    fn verify_hash160_sample(pubkeys: &[[u8; 33]], gpu_hash160_sample: &[u8]) -> Result<()> {
        for (i, (pubkey, gpu_digest)) in pubkeys
            .iter()
            .zip(gpu_hash160_sample.chunks_exact(20))
            .enumerate()
        {
            let cpu_hash160 = Ripemd160::digest(sha2::Sha256::digest(pubkey));
            if &cpu_hash160[..] != gpu_digest {
                bail!("CUDA HASH160 mismatch at sampled index {i}");
            }
        }

        Ok(())
    }

    /// CPU-checks GPU HASH160 digests at arbitrary batch indices against the
    /// host reference. `cpu_pubkeys[k]` must be the reference public key for
    /// batch index `indices[k]`; `digests` is the full downloaded digest
    /// array in step-major order.
    fn verify_hash160_sample_at(
        cpu_pubkeys: &[[u8; 33]],
        indices: &[usize],
        digests: &[u8],
    ) -> Result<()> {
        for (k, (pubkey, &idx)) in cpu_pubkeys.iter().zip(indices.iter()).enumerate() {
            let cpu_hash160 = Ripemd160::digest(sha2::Sha256::digest(pubkey));
            let gpu_digest = &digests[idx * 20..idx * 20 + 20];
            if cpu_hash160.as_slice() != gpu_digest {
                bail!("CUDA HASH160 mismatch at batch index {idx} (sample {k})");
            }
        }

        Ok(())
    }

    /// Hashes one batch entirely on the GPU: a single bulk upload of the
    /// packed secret keys, secp256k1 derivation into the device pubkey
    /// buffer, then hashing through the device buffers, and a single bulk
    /// download of the resulting HASH160 digests.
    ///
    /// All kernels are enqueued before any blocking copy waits, so the GPU
    /// never idles between the upload and the final download; the CPU
    /// verification references are computed while the kernels run.
    fn hash_batch_inner(
        &self,
        secret_keys: &[[u8; 32]],
        verify_secp: bool,
        verify_sha: bool,
        verify_hash160: bool,
    ) -> Result<Vec<[u8; 20]>> {
        let count = secret_keys.len();
        if count == 0 {
            return Ok(Vec::new());
        }

        let sample = Self::verification_sample_limit(count);

        let stream = self.ctx.default_stream();
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| anyhow::anyhow!("CUDA device buffer lock poisoned"))?;
        Self::ensure_buffers(&stream, &mut guard, count, count)?;
        let bufs = &mut *guard;

        // One bulk host-to-device transfer for the whole packed batch.
        let mut host_flat = Vec::with_capacity(count * 32);
        for secret in secret_keys {
            host_flat.extend_from_slice(secret);
        }
        stream
            .memcpy_htod(&host_flat, &mut bufs.secrets)
            .map_err(|e| anyhow::anyhow!("failed to copy secret keys to CUDA device: {e:?}"))?;

        // Public key derivation on the GPU: secrets in, compressed pubkeys
        // out, straight into device memory.
        Self::launch_batch_kernel(
            &stream,
            &self.secp256k1_fn,
            SECP256K1_KERNEL_NAME,
            &bufs.secrets,
            &mut bufs.pubkeys,
            count,
            SECP_BLOCK,
        )?;

        // Enqueue the hash kernels before doing any blocking work so the
        // pipeline stays full. The fused kernel skips the SHA-256 buffer
        // entirely; the two-kernel path chains through it on the device.
        let fused = !verify_sha && !verify_hash160;
        if fused {
            Self::launch_batch_kernel(
                &stream,
                &self.hash160_fn,
                HASH160_KERNEL_NAME,
                &bufs.pubkeys,
                &mut bufs.hash160,
                count,
                HASH_BLOCK,
            )?;
        } else {
            Self::launch_batch_kernel(
                &stream,
                &self.sha256_fn,
                SHA256_KERNEL_NAME,
                &bufs.pubkeys,
                &mut bufs.sha,
                count,
                HASH_BLOCK,
            )?;

            // RIPEMD-160 consumes the SHA-256 digests directly in device
            // memory; the intermediate never round-trips through the host.
            Self::launch_batch_kernel(
                &stream,
                &self.ripemd160_fn,
                RIPEMD160_KERNEL_NAME,
                &bufs.sha,
                &mut bufs.hash160,
                count,
                HASH_BLOCK,
            )?;
        }

        // CPU reference for the verification samples, computed while the
        // GPU churns through the kernels enqueued above.
        let sample_pubkeys = self.derive_pubkeys_from_secret_bytes(&secret_keys[..sample])?;

        // From here on the blocking copies wait on the stream, but every
        // kernel is already enqueued, so the GPU stays busy throughout.
        if verify_secp {
            let mut gpu_sample = vec![0u8; sample * 33];
            stream
                .memcpy_dtoh(&bufs.pubkeys.slice(0..sample * 33), &mut gpu_sample)
                .map_err(|e| anyhow::anyhow!("failed to copy CUDA pubkey sample to host: {e:?}"))?;
            Self::verify_secp_sample(&sample_pubkeys, &gpu_sample)?;
        }

        if verify_sha && !fused {
            let mut sha_sample = vec![0u8; sample * 32];
            stream
                .memcpy_dtoh(&bufs.sha.slice(0..sample * 32), &mut sha_sample)
                .map_err(|e| {
                    anyhow::anyhow!("failed to copy CUDA SHA sample to host: {e:?}")
                })?;
            Self::verify_sha_sample(&sample_pubkeys, &sha_sample)?;
        }

        // One bulk device-to-host transfer for this batch's HASH160 digests.
        stream
            .memcpy_dtoh(
                &bufs.hash160.slice(0..count * 20),
                &mut bufs.hash160_host,
            )
            .map_err(|e| anyhow::anyhow!("failed to copy CUDA HASH160 output to host: {e:?}"))?;

        if verify_hash160 {
            Self::verify_hash160_sample(
                &sample_pubkeys,
                &bufs.hash160_host[..count * 20],
            )?;
        }

        Ok(bufs.hash160_host[..count * 20]
            .chunks_exact(20)
            .map(|chunk| {
                let mut digest = [0u8; 20];
                digest.copy_from_slice(chunk);
                digest
            })
            .collect())
    }

    /// Hashes one batch of consecutive key walks entirely on the GPU.
    ///
    /// Instead of one uploaded key per hashed entry, only `starts.len()`
    /// packed 32-byte walk starts go over the bus: the walk kernel derives
    /// each start's public key with the w=8 fixed-window multiply and then
    /// walks `steps` consecutive keys per thread by repeated mixed addition
    /// of G, batch-normalizing the accumulated Jacobian points per chunk
    /// with one shared field inversion. The SHA-256 / RIPEMD-160 stage is
    /// unchanged and runs over the full `starts.len() * steps` keys.
    ///
    /// Digest order is step-major: digest `idx` belongs to secret
    /// `(starts[idx % starts.len()] + idx / starts.len()) mod n`.
    ///
    /// Verification sampling: the contiguous front sample (cheap single-copy
    /// D2H) checks the scalar-multiply prologue and the hash kernels; a
    /// spread sample checked against the already-downloaded HASH160 digests
    /// covers walk steps across all threads end to end.
    fn hash_walk_inner(
        &self,
        starts: &[[u8; 32]],
        steps: usize,
        verify_secp: bool,
        verify_sha: bool,
        verify_hash160: bool,
    ) -> Result<Vec<[u8; 20]>> {
        let n_starts = starts.len();
        if n_starts == 0 || steps == 0 {
            return Ok(Vec::new());
        }
        let total = n_starts * steps;
        let sample = Self::verification_sample_limit(total);

        let stream = self.ctx.default_stream();
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| anyhow::anyhow!("CUDA device buffer lock poisoned"))?;
        Self::ensure_buffers(&stream, &mut guard, n_starts, total)?;
        let bufs = &mut *guard;

        // One bulk host-to-device transfer for the packed walk starts.
        let mut host_flat = Vec::with_capacity(n_starts * 32);
        for start in starts {
            host_flat.extend_from_slice(start);
        }
        stream
            .memcpy_htod(&host_flat, &mut bufs.secrets)
            .map_err(|e| anyhow::anyhow!("failed to copy walk starts to CUDA device: {e:?}"))?;

        // Public key walk on the GPU: starts in, steps compressed pubkeys
        // per thread out (step-major), straight into device memory.
        let cfg = Self::launch_cfg(n_starts, SECP_BLOCK);
        unsafe {
            stream
                .launch_builder(&self.secp_walk_fn)
                .arg(&bufs.secrets)
                .arg(&mut bufs.pubkeys)
                .arg(&(n_starts as u32))
                .arg(&(steps as u32))
                .launch(cfg)
                .map_err(|e| {
                    anyhow::anyhow!("failed launching CUDA secp256k1 walk kernel: {e:?}")
                })?;
        }

        // Same hash stage as hash_batch_inner, over the full walk.
        let fused = !verify_sha && !verify_hash160;
        if fused {
            Self::launch_batch_kernel(
                &stream,
                &self.hash160_fn,
                HASH160_KERNEL_NAME,
                &bufs.pubkeys,
                &mut bufs.hash160,
                total,
                HASH_BLOCK,
            )?;
        } else {
            Self::launch_batch_kernel(
                &stream,
                &self.sha256_fn,
                SHA256_KERNEL_NAME,
                &bufs.pubkeys,
                &mut bufs.sha,
                total,
                HASH_BLOCK,
            )?;
            Self::launch_batch_kernel(
                &stream,
                &self.ripemd160_fn,
                RIPEMD160_KERNEL_NAME,
                &bufs.sha,
                &mut bufs.hash160,
                total,
                HASH_BLOCK,
            )?;
        }

        // CPU reference secrets for the sampled indices, computed while the
        // GPU churns through the kernels enqueued above.
        let secret_at = |idx: usize| -> [u8; 32] {
            let start_index = idx % n_starts;
            let step = (idx / n_starts) as u32;
            super::add_small_mod_n(&starts[start_index], step)
        };
        let sample_secrets: Vec<[u8; 32]> = (0..sample).map(secret_at).collect();
        let sample_pubkeys = self.derive_pubkeys_from_secret_bytes(&sample_secrets)?;

        // From here on the blocking copies wait on the stream, but every
        // kernel is already enqueued, so the GPU stays busy throughout.
        if verify_secp {
            let mut gpu_sample = vec![0u8; sample * 33];
            stream
                .memcpy_dtoh(&bufs.pubkeys.slice(0..sample * 33), &mut gpu_sample)
                .map_err(|e| anyhow::anyhow!("failed to copy CUDA pubkey sample to host: {e:?}"))?;
            Self::verify_secp_sample(&sample_pubkeys, &gpu_sample)?;
        }

        if verify_sha && !fused {
            let mut sha_sample = vec![0u8; sample * 32];
            stream
                .memcpy_dtoh(&bufs.sha.slice(0..sample * 32), &mut sha_sample)
                .map_err(|e| {
                    anyhow::anyhow!("failed to copy CUDA SHA sample to host: {e:?}")
                })?;
            Self::verify_sha_sample(&sample_pubkeys, &sha_sample)?;
        }

        // One bulk device-to-host transfer for this batch's HASH160 digests.
        stream
            .memcpy_dtoh(
                &bufs.hash160.slice(0..total * 20),
                &mut bufs.hash160_host,
            )
            .map_err(|e| anyhow::anyhow!("failed to copy CUDA HASH160 output to host: {e:?}"))?;

        if verify_hash160 {
            let indices: Vec<usize> = (0..sample).map(|k| k * total / sample).collect();
            let spread_secrets: Vec<[u8; 32]> = indices.iter().map(|&idx| secret_at(idx)).collect();
            let spread_pubkeys = self.derive_pubkeys_from_secret_bytes(&spread_secrets)?;
            Self::verify_hash160_sample_at(
                &spread_pubkeys,
                &indices,
                &bufs.hash160_host[..total * 20],
            )?;
        }

        Ok(bufs.hash160_host[..total * 20]
            .chunks_exact(20)
            .map(|chunk| {
                let mut digest = [0u8; 20];
                digest.copy_from_slice(chunk);
                digest
            })
            .collect())
    }
}

impl HashBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn hash_batch_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>> {
        let (verify_secp, verify_sha, verify_hash160) = Self::verification_flags();
        self.hash_batch_inner(secret_keys, verify_secp, verify_sha, verify_hash160)
    }

    fn hash_batch_walking(&self, starts: &[[u8; 32]], steps: usize) -> Result<Vec<[u8; 20]>> {
        let (verify_secp, verify_sha, verify_hash160) = Self::verification_flags();
        self.hash_walk_inner(starts, steps, verify_secp, verify_sha, verify_hash160)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn cpu_reference(secrets: &[[u8; 32]]) -> Vec<[u8; 20]> {
        let secp = Secp256k1::signing_only();
        secrets
            .iter()
            .map(|secret| {
                let key = SecretKey::from_byte_array(*secret).unwrap();
                let pubkey = PublicKey::from_secret_key(&secp, &key).serialize();
                let sha = sha2::Sha256::digest(pubkey);
                let hash160 = Ripemd160::digest(sha);
                let mut out = [0u8; 20];
                out.copy_from_slice(&hash160);
                out
            })
            .collect()
    }

    #[test]
    fn cuda_paths_match_cpu_reference_across_batch_sizes() {
        let Ok(backend) = CudaBackend::new() else {
            eprintln!("CUDA not available; skipping CUDA path assertion in this environment");
            return;
        };

        let secrets: Vec<[u8; 32]> = (1u8..=8).map(|i| [i; 32]).collect();
        let reference = cpu_reference(&secrets);

        // Fused single-kernel hash path (both hash verifications off), with
        // and without the GPU-vs-CPU pubkey sample check.
        let fused = backend.hash_batch_inner(&secrets, false, false, false).unwrap();
        assert_eq!(fused, reference);

        let fused_checked = backend.hash_batch_inner(&secrets, true, false, false).unwrap();
        assert_eq!(fused_checked, reference);

        // Two-kernel path with sampling verification (default configuration).
        let verified = backend.hash_batch_inner(&secrets, true, true, true).unwrap();
        assert_eq!(verified, reference);

        // Mixed verification flags exercise the same device-side chaining.
        let sha_only = backend.hash_batch_inner(&secrets, true, true, false).unwrap();
        assert_eq!(sha_only, reference);

        let hash160_only = backend.hash_batch_inner(&secrets, true, false, true).unwrap();
        assert_eq!(hash160_only, reference);

        // Smaller batch against the now-larger persistent buffers: only the
        // front of each buffer may be used, and exactly `count` digests must
        // come back.
        let small_secrets: Vec<[u8; 32]> = vec![[9u8; 32], [10u8; 32], [11u8; 32]];
        let small_reference = cpu_reference(&small_secrets);
        let small = backend.hash_batch_inner(&small_secrets, true, true, true).unwrap();
        assert_eq!(small, small_reference);

        // Growing again after the smaller batch must still be correct.
        let grown = backend.hash_batch_inner(&secrets, false, false, false).unwrap();
        assert_eq!(grown, reference);
    }

    #[test]
    fn cuda_derivation_matches_generator_point() {
        let Ok(backend) = CudaBackend::new() else {
            eprintln!("CUDA not available; skipping CUDA path assertion in this environment");
            return;
        };

        // k = 1 must derive the documented secp256k1 generator point:
        // 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798.
        // `verify_secp = true` makes the backend compare the GPU-derived
        // pubkey for this key against the CPU one byte for byte.
        let mut one = [0u8; 32];
        one[31] = 1;
        let reference = cpu_reference(&[one]);
        let gpu = backend.hash_batch_inner(&[one], true, true, true).unwrap();
        assert_eq!(gpu, reference);
    }

    fn walk_reference(starts: &[[u8; 32]], steps: usize) -> Vec<[u8; 20]> {
        let secrets: Vec<[u8; 32]> = (0..steps)
            .flat_map(|j| {
                starts
                    .iter()
                    .map(move |s| crate::backend::add_small_mod_n(s, j as u32))
            })
            .collect();
        cpu_reference(&secrets)
    }

    fn varied_starts(count: u8) -> Vec<[u8; 32]> {
        (0..count)
            .map(|i| {
                let mut k = [0u8; 32];
                k[0] = 0xa0 + i;
                k[15] = i.wrapping_mul(37);
                k[31] = i.wrapping_mul(7) + 1;
                k
            })
            .collect()
    }

    #[test]
    fn cuda_walk_matches_cpu_reference_random_starts() {
        let Ok(backend) = CudaBackend::new() else {
            eprintln!("CUDA not available; skipping CUDA walk assertion in this environment");
            return;
        };

        // Uniform-random starts: the scalar-multiply prologue and the walk
        // must hold for arbitrary scalars, not just the repeated-byte
        // patterns of the other vectors.
        let mut flat = vec![0u8; 64 * 32];
        getrandom::fill(&mut flat).unwrap();
        let starts: Vec<[u8; 32]> = flat
            .chunks_exact(32)
            .map(|c| {
                let mut k = [0u8; 32];
                k.copy_from_slice(c);
                k
            })
            .filter(|k| SecretKey::from_byte_array(*k).is_ok())
            .take(64)
            .collect();
        assert_eq!(starts.len(), 64);

        let out = backend.hash_batch_walking(&starts, 96).unwrap();
        assert_eq!(out, walk_reference(&starts, 96));
    }

    #[test]
    fn cuda_walk_matches_cpu_reference() {
        let Ok(backend) = CudaBackend::new() else {
            eprintln!("CUDA not available; skipping CUDA walk assertion in this environment");
            return;
        };

        let starts = varied_starts(4);

        // 70 steps = two full WALK_CHUNK chunks plus a remainder chunk, and
        // the default verification flags are on, so every batch is also
        // checked by the in-band sampling (contiguous + spread).
        let steps = 70;
        let out = backend.hash_batch_walking(&starts, steps).unwrap();
        assert_eq!(out.len(), starts.len() * steps);
        assert_eq!(out, walk_reference(&starts, steps));

        // Single-key walks (steps = 1) degenerate to one scalar multiply per
        // thread through the same chunked normalization code.
        let single = backend.hash_batch_walking(&starts, 1).unwrap();
        assert_eq!(single, walk_reference(&starts, 1));

        // One thread walking past a chunk boundary.
        let lone = varied_starts(1);
        let lonely = backend.hash_batch_walking(&lone, 33).unwrap();
        assert_eq!(lonely, walk_reference(&lone, 33));

        // Larger batch over the persistent buffers: 64 threads x 128 steps.
        let many = varied_starts(64);
        let big = backend.hash_batch_walking(&many, 128).unwrap();
        assert_eq!(big, walk_reference(&many, 128));
    }

    #[test]
    fn cuda_walk_matches_reference_with_all_verifications_off() {
        let Ok(backend) = CudaBackend::new() else {
            eprintln!("CUDA not available; skipping CUDA walk assertion in this environment");
            return;
        };

        let starts = varied_starts(3);
        let fused = backend.hash_walk_inner(&starts, 64, false, false, false).unwrap();
        assert_eq!(fused, walk_reference(&starts, 64));

        let split = backend.hash_walk_inner(&starts, 64, false, true, true).unwrap();
        assert_eq!(split, walk_reference(&starts, 64));
    }
}
