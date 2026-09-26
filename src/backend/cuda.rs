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
    capacity: usize,
    /// `capacity * 32` bytes: packed 32-byte big-endian secret keys.
    secrets: CudaSlice<u8>,
    /// `capacity * 33` bytes: packed 33-byte compressed public keys,
    /// written by the secp256k1 kernel, consumed by the hash kernels.
    pubkeys: CudaSlice<u8>,
    /// `capacity * 32` bytes: SHA-256 intermediates, consumed on the device.
    sha: CudaSlice<u8>,
    /// `capacity * 20` bytes: HASH160 digests.
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
            capacity: 0,
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

    /// Grows the persistent buffers when a batch larger than any seen before
    /// arrives (in practice only on the first batch). The whole set is built
    /// before assigning so a failed allocation leaves existing buffers intact.
    fn ensure_buffers(
        stream: &Arc<CudaStream>,
        bufs: &mut DeviceBuffers,
        count: usize,
    ) -> Result<()> {
        if bufs.capacity >= count {
            return Ok(());
        }

        let alloc = |what: &str, len: usize| {
            stream
                .alloc_zeros(len)
                .map_err(|e| anyhow::anyhow!("failed to allocate CUDA {what} buffer: {e:?}"))
        };
        let grown = DeviceBuffers {
            capacity: count,
            secrets: alloc("secret key", count * 32)?,
            pubkeys: alloc("pubkey", count * 33)?,
            sha: alloc("SHA", count * 32)?,
            hash160: alloc("HASH160", count * 20)?,
            hash160_host: vec![0; count * 20],
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
        Self::ensure_buffers(&stream, &mut guard, count)?;
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
}

impl HashBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn hash_batch_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>> {
        let verify_secp = std::env::var("SEEKER_CUDA_VERIFY_SECP").ok().as_deref() != Some("0");
        let verify_sha = std::env::var("SEEKER_CUDA_VERIFY_SHA256").ok().as_deref() != Some("0");
        let verify_hash160 =
            std::env::var("SEEKER_CUDA_VERIFY_HASH160").ok().as_deref() != Some("0");

        self.hash_batch_inner(secret_keys, verify_secp, verify_sha, verify_hash160)
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
}
