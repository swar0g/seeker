use std::{
    fs::File,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use anyhow::{bail, Result};
use getrandom::fill;
use memmap2::Mmap;
use rayon::prelude::*;
use ripemd::Ripemd160;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};

mod backend;

const HEADER_SIZE: usize = 32;
const ENTRY_SIZE: usize = 20;
const MAGIC: &[u8; 8] = b"BTCIDX01";

// One context per thread: initialised once, reused for every key derivation on
// that thread. `signing_only()` builds half the lookup tables that `new()` does.
thread_local! {
    static SECP: Secp256k1<secp256k1::SignOnly> = Secp256k1::signing_only();
}

static ATTEMPTS: AtomicU64 = AtomicU64::new(0);

pub struct AddressIndex {
    mmap: Mmap,
    len: usize,
    bits: Vec<u64>,
}

impl AddressIndex {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: The file is opened read-only and no other process holds a
        // writable handle to it. The OS keeps the mapping alive independently
        // of the file descriptor; the mmap is valid for the full lifetime of
        // the returned `AddressIndex`.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            bail!("File too small");
        }

        if &mmap[0..8] != MAGIC {
            bail!("Invalid index file (bad magic)");
        }

        let count_u64 = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        let count = usize::try_from(count_u64)?;

        let expected_size = HEADER_SIZE + count * ENTRY_SIZE;

        if mmap.len() != expected_size {
            bail!("Corrupted index file (size mismatch)");
        }

        // build Bloom filter: 1 bit per entry, two hash functions (default hasher + prefix)
        let num_bits = count;
        let bits_len = num_bits.div_ceil(64);
        let mut bits = vec![0u64; bits_len];
        let _base_hasher = std::collections::hash_map::DefaultHasher::new();
        for i in 0..count {
            let start = HEADER_SIZE + i * ENTRY_SIZE;
            let end = start + ENTRY_SIZE;
            let entry = &mmap[start..end];
            let mut h1_hasher = std::collections::hash_map::DefaultHasher::new();
            let mut h2_hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hasher::write(&mut h1_hasher, entry);
            let h1 = std::hash::Hasher::finish(&h1_hasher) as usize;
            std::hash::Hasher::write(&mut h2_hasher, entry);
            std::hash::Hasher::write(&mut h2_hasher, &[1u8; 4]);
            let h2 = std::hash::Hasher::finish(&h2_hasher) as usize;
            let pos1 = h1 % num_bits;
            let pos2 = h2 % num_bits;
            bits[pos1 / 64] |= 1u64 << (pos1 % 64);
            bits[pos2 / 64] |= 1u64 << (pos2 % 64);
        }
        Ok(Self {
            mmap,
            len: count,
            bits,
        })
    }

    /// Fast probabilistic pre‑check. Returns `true` only if the filter
    /// believes the hash *might* be present; callers should still fall
    /// back to the real binary‑search.
    fn might_contain(&self, target: &[u8; 20]) -> bool {
        let mut h1_hasher = std::collections::hash_map::DefaultHasher::new();
        let mut h2_hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hasher::write(&mut h1_hasher, target);
        let h1 = std::hash::Hasher::finish(&h1_hasher) as usize;
        std::hash::Hasher::write(&mut h2_hasher, target);
        std::hash::Hasher::write(&mut h2_hasher, &[1u8; 4]);
        let h2 = std::hash::Hasher::finish(&h2_hasher) as usize;

        let num_bits = self.len;
        let pos1_mod = h1 % num_bits;
        let pos2_mod = h2 % num_bits;
        let pos1 = pos1_mod / 64;
        let pos2 = pos2_mod / 64;
        let bit1 = 1u64 << (pos1_mod % 64);
        let bit2 = 1u64 << (pos2_mod % 64);
        self.bits[pos1] & bit1 != 0 && self.bits[pos2] & bit2 != 0
    }

    fn data(&self) -> &[[u8; 20]] {
        // SAFETY: The constructor validated that the data region contains exactly
        // `self.len` contiguous 20-byte entries. `[u8; 20]` has alignment 1, so
        // casting from raw mmap bytes is always valid for any byte sequence.
        unsafe {
            std::slice::from_raw_parts(
                self.mmap[HEADER_SIZE..].as_ptr() as *const [u8; 20],
                self.len,
            )
        }
    }

    pub fn contains(&self, target: &[u8; 20]) -> bool {
        if self.len == 0 {
            return false;
        }
        // Bloom‑filter short‑circuit ( >95 % of non‑matches are rejected )
        if !self.might_contain(target) {
            return false; // definitely not in the index
        }
        // fallback to the exact binary search
        self.data().binary_search(target).is_ok()
    }
}

fn generate_secret_key_bytes_batch(batch_size: usize, out: &mut Vec<[u8; 32]>, buffer: &mut [u8]) {
    assert!(
        buffer.len() >= batch_size * 32,
        "buffer must be large enough to hold batch_size keys"
    );
    // Optimized batch generation: fill a large buffer once per loop iteration
    // instead of generating one key at a time. This reduces the number of
    // calls to `getrandom::fill` and the overhead of repeatedly allocating
    // temporary arrays.
    out.clear();

    while out.len() < batch_size {
        // Fill the entire buffer with random bytes.
        fill(buffer).unwrap();
        // Interpret the buffer as a slice of `[u8; 32]`.
        let chunks =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const [u8; 32], batch_size) };
        for chunk in chunks {
            if SecretKey::from_byte_array(*chunk).is_ok() {
                out.push(*chunk);
                if out.len() == batch_size {
                    break;
                }
            }
        }
    }
}

fn hash_pubkey_bytes(serialized_pubkey: &[u8]) -> [u8; 20] {
    let sha = Sha256::digest(serialized_pubkey);
    let ripemd = Ripemd160::digest(sha);

    let mut hash = [0u8; 20];
    hash.copy_from_slice(&ripemd);
    hash
}

fn hash_from_secret(secp: &Secp256k1<secp256k1::SignOnly>, secret_key: &SecretKey) -> [u8; 20] {
    let public_key = PublicKey::from_secret_key(secp, secret_key);
    let serialized = public_key.serialize();
    hash_pubkey_bytes(&serialized)
}

// Parallelize public key derivation and hashing across CPU threads using Rayon.
// Each worker thread reuses its own thread-local `secp256k1` context via `SECP.with`.
// This scales CPU backend throughput linearly with available CPU cores.
pub(crate) fn hash_batch_from_secret_bytes(secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>> {
    secret_keys
        .par_iter()
        .map(|secret| {
            SECP.with(|secp| {
                let key = SecretKey::from_byte_array(*secret)?;
                Ok(hash_from_secret(secp, &key))
            })
        })
        .collect()
}

fn main() -> Result<()> {
    let backend = backend::create_default_backend()?;
    println!("Backend selected: {}", backend.name());

    // Walking backends amortize one scalar multiplication over `walk_steps`
    // consecutive keys, so the GPU path needs much larger batches than the
    // CPU path to keep the device busy. 2^21 keys at 128 steps per walk
    // measures as the fastest EC-kernel configuration on a 38-SM GPU
    // (16384 walker threads); smaller batches starve it, larger ones only
    // grow the host-side digest scan.
    let default_batch = if backend.name() == "cuda" { 1 << 21 } else { 32768 };
    let batch_size = std::env::var("SEEKER_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_batch);

    let walk_steps = std::env::var("SEEKER_WALK_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(128)
        .min(u32::MAX as usize)
        .min(batch_size);
    // Batch actually scanned per iteration: the batch size rounded down to a
    // whole number of walks.
    let walk_threads = batch_size / walk_steps;
    let batch_size = walk_threads * walk_steps;
    println!(
        "Batch size: {batch_size} ({} walks x {walk_steps} steps)",
        walk_threads
    );

    let startup_vectors = [[1u8; 32], [2u8; 32]];
    let startup_hashes = backend.hash_batch_from_secret_bytes(&startup_vectors)?;
    if startup_hashes.len() != startup_vectors.len() {
        bail!("Backend startup self-check failed: hash count mismatch");
    }

    let index = AddressIndex::open("addresses.bin")?;
    let start = Instant::now();

    println!("Index loaded with {} entries", index.len);

    let mut walk_starts = Vec::with_capacity(walk_threads);
    let mut buffer = vec![0u8; walk_threads * 32];
    let mut next_report = 1_000_000u64;

    loop {
        generate_secret_key_bytes_batch(walk_threads, &mut walk_starts, &mut buffer);
        let hashes = match backend.hash_batch_walking(&walk_starts, walk_steps) {
            Ok(h) => h,
            Err(err) => {
                eprintln!("Backend hashing failed for current batch: {err}");
                continue;
            }
        };

        let attempts = ATTEMPTS.fetch_add(batch_size as u64, Ordering::Relaxed) + batch_size as u64;

        if attempts >= next_report {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (attempts as f64 / elapsed) as u64;
            println!("Attempts: {attempts}, throughput: {rate} keys/s");
            next_report = attempts + 1_000_000;
        }

        // The index check is host-side and linear in the digest count, which
        // makes it a real fraction of wall time once the backend produces
        // millions of digests per second — spread it across cores. Digest
        // order is step-major: idx = step * walk_threads + start_index
        // (see HashBackend::hash_batch_walking).
        if let Some((idx, _)) = hashes
            .par_iter()
            .enumerate()
            .find_first(|(_, hash)| index.contains(hash))
        {
            let start_index = idx % walk_threads;
            let step = (idx / walk_threads) as u32;
            let secret_bytes = backend::add_small_mod_n(&walk_starts[start_index], step);
            let secret_key = SecretKey::from_byte_array(secret_bytes)
                .expect("reconstructed walk secret key must be a valid scalar");
            println!("Collision detected!");
            println!("Private key: {}", secret_key.display_secret());
            return Ok(());
        }
    }

    #[allow(unreachable_code)]
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_might_contain() {
        use std::io::Write;

        let mut temp_path = std::env::temp_dir();
        temp_path.push(format!("test_might_contain_{}.bin", std::process::id()));

        let mut entries = Vec::new();
        for i in 0..1000u32 {
            let mut entry = [0u8; 20];
            entry[0..4].copy_from_slice(&i.to_le_bytes());
            entries.push(entry);
        }

        {
            let mut f = std::fs::File::create(&temp_path).unwrap();
            f.write_all(b"BTCIDX01").unwrap();
            f.write_all(&(entries.len() as u64).to_le_bytes()).unwrap();
            f.write_all(&[0u8; 16]).unwrap();
            for e in &entries {
                f.write_all(e).unwrap();
            }
            f.sync_all().unwrap();
        }

        let index = AddressIndex::open(&temp_path).unwrap();

        // Assert true positives
        for e in &entries {
            assert!(index.might_contain(e), "must contain inserted elements");
        }

        // Assert negatives
        // The bloom filter might have false positives, but it is extremely unlikely for a small set
        // to collide on 0x44. Let's make sure it returns false for at least one out of a few.
        let mut all_false_positives = true;
        for i in 1000..2000u32 {
            let mut candidate = [0u8; 20];
            candidate[0..4].copy_from_slice(&i.to_le_bytes());
            if !index.might_contain(&candidate) {
                all_false_positives = false;
                break;
            }
        }
        assert!(!all_false_positives, "should have some true negatives");

        std::fs::remove_file(temp_path).unwrap();
    }
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempFileGuard {
        path: std::path::PathBuf,
    }

    impl TempFileGuard {
        fn new_with_content(content: &[u8]) -> Self {
            let mut path = std::env::temp_dir();
            let count = TEST_FILE_COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!("test_index_{}_{}.bin", time, count));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(content).unwrap();
            Self { path }
        }
    }

    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn legacy_hash_from_secret_bytes(secret: [u8; 32]) -> [u8; 20] {
        let secp = Secp256k1::signing_only();
        let secret_key = SecretKey::from_byte_array(secret).unwrap();
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        let serialized = public_key.serialize();
        let sha = Sha256::digest(serialized);
        let ripemd = Ripemd160::digest(sha);

        let mut hash = [0u8; 20];
        hash.copy_from_slice(&ripemd);
        hash
    }

    #[test]
    fn batch_hash_matches_legacy_reference() {
        let vectors = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [0x11u8; 32]];
        let actual = hash_batch_from_secret_bytes(&vectors).unwrap();
        let expected: Vec<[u8; 20]> = vectors
            .into_iter()
            .map(legacy_hash_from_secret_bytes)
            .collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn walking_hashes_match_reference() {
        let backend = backend::create_default_backend().unwrap();
        let starts = vec![[1u8; 32], [2u8; 32], [0x11u8; 32]];
        let steps = 5;

        let out = backend.hash_batch_walking(&starts, steps).unwrap();
        assert_eq!(out.len(), starts.len() * steps);

        // Step-major digest order; walk i, step j must hash (starts[i] + j).
        for j in 0..steps {
            for i in 0..starts.len() {
                let secret = backend::add_small_mod_n(&starts[i], j as u32);
                let expected = legacy_hash_from_secret_bytes(secret);
                assert_eq!(out[j * starts.len() + i], expected, "step {j}, walk {i}");
            }
        }
    }

    #[test]
    fn batch_hash_rejects_invalid_secret_keys() {
        let vectors = vec![[0u8; 32], [1u8; 32]];
        let result = hash_batch_from_secret_bytes(&vectors);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_backend_matches_reference_path() {
        let backend = backend::create_default_backend().unwrap();
        let vectors = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];

        let backend_out = backend.hash_batch_from_secret_bytes(&vectors).unwrap();
        let reference_out = hash_batch_from_secret_bytes(&vectors).unwrap();

        assert_eq!(backend_out, reference_out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_matches_reference_path_when_available() {
        let Ok(backend) = backend::create_cuda_backend() else {
            eprintln!("CUDA not available; skipping CUDA parity assertion in this environment");
            return;
        };

        let vectors = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32]];
        let backend_out = backend.hash_batch_from_secret_bytes(&vectors).unwrap();
        let reference_out = hash_batch_from_secret_bytes(&vectors).unwrap();

        assert_eq!(backend_out, reference_out);
    }

    #[test]
    fn test_generate_secret_key_bytes_batch() {
        let batch_size = 100;
        let mut out = Vec::with_capacity(batch_size);
        out.push([0u8; 32]); // Test that existing elements are cleared
        let mut buffer = vec![0u8; batch_size * 32];

        generate_secret_key_bytes_batch(batch_size, &mut out, &mut buffer);
        assert_eq!(out.len(), batch_size);

        for key in &out {
            assert!(SecretKey::from_byte_array(*key).is_ok());
        }
        assert_ne!(out[0], out[1], "keys should be reasonably distinct");
    }

    #[test]
    fn test_generate_secret_key_bytes_batch_zero_size() {
        let mut out = vec![[1u8; 32]];
        let mut buffer = vec![];
        generate_secret_key_bytes_batch(0, &mut out, &mut buffer);
        assert!(
            out.is_empty(),
            "out should be cleared even if batch_size is 0"
        );
    }

    #[test]
    #[should_panic(expected = "buffer must be large enough to hold batch_size keys")]
    fn test_generate_secret_key_bytes_batch_buffer_too_small() {
        let mut out = vec![];
        let mut buffer = vec![0u8; 31];
        generate_secret_key_bytes_batch(1, &mut out, &mut buffer);
    }

    #[test]
    fn test_address_index_open_file_too_small() {
        let content = [0u8; 10]; // Smaller than HEADER_SIZE (32)
        let guard = TempFileGuard::new_with_content(&content);
        let result = AddressIndex::open(&guard.path);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().to_string(), "File too small");
    }

    #[test]
    fn test_address_index_open_invalid_magic() {
        let mut content = [0u8; 32];
        content[0..8].copy_from_slice(b"BADMAGIC"); // Not "BTCIDX01"
        let guard = TempFileGuard::new_with_content(&content);
        let result = AddressIndex::open(&guard.path);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "Invalid index file (bad magic)"
        );
    }

    #[test]
    fn test_address_index_open_size_mismatch() {
        let mut content = vec![0u8; 32]; // Header only
        content[0..8].copy_from_slice(b"BTCIDX01"); // Magic
        content[8..16].copy_from_slice(&1u64.to_le_bytes()); // Expects 1 entry (20 bytes)
                                                             // Missing the 20 bytes for the entry, total size should be 32 + 20 = 52, but it is 32.
        let guard = TempFileGuard::new_with_content(&content);
        let result = AddressIndex::open(&guard.path);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "Corrupted index file (size mismatch)"
        );
    }

    #[test]
    fn test_address_index_open_success() {
        let mut content = vec![0u8; 32 + 20 * 2]; // Header + 2 entries
        content[0..8].copy_from_slice(b"BTCIDX01"); // Magic
        content[8..16].copy_from_slice(&2u64.to_le_bytes()); // 2 entries
                                                             // Populate first entry with dummy hash
        let hash1 = [1u8; 20];
        content[32..52].copy_from_slice(&hash1);
        // Populate second entry with dummy hash
        let hash2 = [2u8; 20];
        content[52..72].copy_from_slice(&hash2);

        let guard = TempFileGuard::new_with_content(&content);
        let result = AddressIndex::open(&guard.path);
        assert!(result.is_ok());
        let index = result.unwrap();
        assert_eq!(index.len, 2);
    }
}
