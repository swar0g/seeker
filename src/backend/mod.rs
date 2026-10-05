use anyhow::Result;

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;

/// secp256k1 group order n, as 4 little-endian u64 limbs.
const CURVE_ORDER: [u64; 4] = [
    0xBFD25E8C_D0364141,
    0xBAAEDCE6_AF48A03B,
    0xFFFFFFFF_FFFFFFFE,
    0xFFFFFFFF_FFFFFFFF,
];

/// 2^256 - n (fits in one limb).
const ORDER_COMPLEMENT: u64 = 0x1_0000_03D1;

/// Little-endian limb arrays compare lexicographically from the *least*
/// significant limb, so a numeric `>=` must walk limbs from the top.
fn geq_curve_order(value: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if value[i] != CURVE_ORDER[i] {
            return value[i] > CURVE_ORDER[i];
        }
    }
    true
}

/// Computes `(key + addend) mod n` on 256-bit big-endian scalars.
///
/// Walk bookkeeping on the host: the secret covered by step `j` of a walk
/// started at `key` is `(key + j) mod n`. Valid secret keys are `< n`, and
/// `addend` is `< 2^32 < 2^256 - n`, so the sum can only leave `[0, n)` by
/// one conditional subtraction; the wraparound branch handles out-of-range
/// inputs defensively anyway.
pub(crate) fn add_small_mod_n(key: &[u8; 32], addend: u32) -> [u8; 32] {
    debug_assert!(u64::from(addend) < ORDER_COMPLEMENT);

    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        *limb = u64::from_be_bytes(key[(3 - i) * 8..(4 - i) * 8].try_into().unwrap());
    }

    // limbs += addend (zero-extended), with carry propagation.
    let mut carry = u64::from(addend);
    for limb in &mut limbs {
        let (sum, c) = limb.overflowing_add(carry);
        *limb = sum;
        carry = u64::from(c);
    }

    if carry != 0 {
        // Sum wrapped past 2^256: subtracting 2^256 mod n means adding
        // 2^256 - n. The result stays far below n for addend < 2^32.
        let mut carry = ORDER_COMPLEMENT;
        for limb in &mut limbs {
            let (sum, next) = limb.overflowing_add(carry);
            *limb = sum;
            carry = u64::from(next);
        }
        debug_assert_eq!(carry, 0);
    } else if geq_curve_order(&limbs) {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (sub, b1) = limbs[i].overflowing_sub(CURVE_ORDER[i]);
            let (sub, b2) = sub.overflowing_sub(borrow);
            limbs[i] = sub;
            borrow = u64::from(b1 || b2);
        }
        debug_assert_eq!(borrow, 0);
    }

    let mut out = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        out[(3 - i) * 8..(4 - i) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    out
}

pub trait HashBackend {
    fn name(&self) -> &'static str;
    fn hash_batch_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>>;

    /// Hashes a batch of consecutive key walks.
    ///
    /// Walk `i` starts at the 32-byte big-endian secret `starts[i]` and covers
    /// the consecutive secrets `(starts[i] + j) mod n` for `j in 0..steps`.
    /// Backends that support it derive the whole walk with one point addition
    /// per key instead of one scalar multiplication per key.
    ///
    /// Digest order is step-major: `digest[j * starts.len() + i]` belongs to
    /// secret `(starts[i] + j) mod n`.
    ///
    /// The default implementation expands the walks into explicit keys and
    /// falls back to [`HashBackend::hash_batch_from_secret_bytes`].
    fn hash_batch_walking(&self, starts: &[[u8; 32]], steps: usize) -> Result<Vec<[u8; 20]>> {
        let mut secret_keys = Vec::with_capacity(starts.len() * steps);
        for j in 0..steps {
            for start in starts {
                secret_keys.push(add_small_mod_n(start, j as u32));
            }
        }
        self.hash_batch_from_secret_bytes(&secret_keys)
    }
}

pub fn create_default_backend() -> Result<Box<dyn HashBackend>> {
    #[cfg(feature = "cuda")]
    {
        if std::env::var("SEEKER_BACKEND").ok().as_deref() == Some("cuda") {
            return create_cuda_backend();
        }
    }

    Ok(Box::new(cpu::CpuBackend::new()))
}

#[cfg(feature = "cuda")]
pub fn create_cuda_backend() -> Result<Box<dyn HashBackend>> {
    Ok(Box::new(cuda::CudaBackend::new()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use secp256k1::{PublicKey, Secp256k1, SecretKey};

    fn key_from_u64(value: u64) -> [u8; 32] {
        let mut key = [0u8; 32];
        key[24..].copy_from_slice(&value.to_be_bytes());
        key
    }

    fn curve_order_bytes() -> [u8; 32] {
        let mut key = [0u8; 32];
        for i in 0..4 {
            key[(3 - i) * 8..(4 - i) * 8].copy_from_slice(&CURVE_ORDER[i].to_be_bytes());
        }
        key
    }

    fn order_minus_one() -> [u8; 32] {
        let mut key = curve_order_bytes();
        key[31] -= 1;
        key
    }

    #[test]
    fn add_small_mod_n_basic() {
        assert_eq!(add_small_mod_n(&key_from_u64(1), 1), key_from_u64(2));
        assert_eq!(add_small_mod_n(&key_from_u64(0), 0), key_from_u64(0));

        let mut expected = [0u8; 32];
        expected[23] = 1; // u64::MAX + 1 == 2^64
        assert_eq!(add_small_mod_n(&key_from_u64(u64::MAX), 1), expected);
    }

    #[test]
    fn add_small_mod_n_wraps_at_curve_order() {
        let n_minus_one = order_minus_one();
        assert_eq!(add_small_mod_n(&n_minus_one, 1), [0u8; 32]);

        let mut n_minus_two = n_minus_one;
        n_minus_two[31] = 0x3f;
        let mut one = [0u8; 32];
        one[31] = 1;
        assert_eq!(add_small_mod_n(&n_minus_two, 3), one);
    }

    #[test]
    fn add_small_mod_n_handles_carry_past_256_bits() {
        // start = 2^256 - 1, +1 -> 2^256 == (2^256 - n) (mod n).
        let start = [0xffu8; 32];
        let mut expected = [0u8; 32];
        expected[24..].copy_from_slice(&ORDER_COMPLEMENT.to_be_bytes());
        assert_eq!(add_small_mod_n(&start, 1), expected);
    }

    #[test]
    fn add_small_mod_n_crosses_modulus_near_order() {
        let n_bytes = curve_order_bytes();
        for delta in 1u64..=5 {
            for add in 1u32..=10 {
                // key = n - delta, a valid start right below the modulus.
                let mut key = n_bytes;
                let (limb, borrow) = CURVE_ORDER[0].overflowing_sub(delta);
                assert!(!borrow);
                key[24..].copy_from_slice(&limb.to_be_bytes());

                let expected = if u64::from(add) >= delta {
                    key_from_u64(u64::from(add) - delta)
                } else {
                    let mut e = n_bytes;
                    let (limb, b) = CURVE_ORDER[0].overflowing_sub(delta - u64::from(add));
                    assert!(!b);
                    e[24..].copy_from_slice(&limb.to_be_bytes());
                    e
                };
                assert_eq!(add_small_mod_n(&key, add), expected, "delta {delta}, add {add}");
            }
        }
    }

    #[test]
    fn add_small_mod_n_matches_curve_group_law() {
        // Group-law oracle: for any valid scalar k and addend a,
        // pk((k + a) mod n) must equal pk(k) + pk(a). This checks the full
        // modular reduction on pseudo-random 256-bit inputs, where
        // limb-wise comparisons and partial reductions go wrong.
        let secp = Secp256k1::signing_only();
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for case in 0..128 {
            let mut key = [0u8; 32];
            for limb_bytes in key.chunks_exact_mut(8) {
                limb_bytes.copy_from_slice(&next().to_be_bytes());
            }
            let addend = (next() % 4096) as u32 + 1;

            let Ok(secret) = SecretKey::from_byte_array(key) else {
                continue; // start >= n: outside the helper's contract
            };
            let sum = add_small_mod_n(&key, addend);

            let mut addend_bytes = [0u8; 32];
            addend_bytes[28..].copy_from_slice(&addend.to_be_bytes());
            let pk_start = PublicKey::from_secret_key(&secp, &secret);
            let pk_add = PublicKey::from_secret_key(
                &secp,
                &SecretKey::from_byte_array(addend_bytes).unwrap(),
            );
            let combined = pk_start.combine(&pk_add).unwrap();
            let pk_sum = PublicKey::from_secret_key(
                &secp,
                &SecretKey::from_byte_array(sum)
                    .expect("reduced sum must be a valid scalar"),
            );
            assert_eq!(pk_sum, combined, "case {case}, addend {addend}");
        }
    }
}
