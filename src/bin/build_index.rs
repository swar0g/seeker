use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use anyhow::{bail, Result};
use rayon::slice::ParallelSliceMut;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"BTCIDX01";

fn decode_pubkey_hash(addr: &str) -> Result<[u8; 20]> {
    let decoded = bs58::decode(addr).into_vec()?;

    if decoded.len() != 25 {
        bail!("Invalid address length: {}", addr);
    }

    if decoded[0] != 0x00 {
        bail!("Not a mainnet P2PKH address: {}", addr);
    }

    let checksum = &decoded[21..25];
    let computed = Sha256::digest(Sha256::digest(&decoded[0..21]));
    if checksum != &computed[0..4] {
        bail!("Bad checksum for address: {}", addr);
    }

    let mut hash = [0u8; 20];
    hash.copy_from_slice(&decoded[1..21]);
    Ok(hash)
}

fn write_header(file: &mut impl Write, count: u64) -> Result<()> {
    file.write_all(MAGIC)?;
    file.write_all(&count.to_le_bytes())?;
    file.write_all(&[0u8; 16])?;
    Ok(())
}

fn main() -> Result<()> {
    let input_path = "addresses.txt";
    let output_path = "addresses.bin";

    println!("Reading addresses from {}", input_path);

    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut hashes: Vec<[u8; 20]> = Vec::new();

    for line in reader.lines() {
        let addr = line?;
        let addr = addr.trim();
        if addr.is_empty() {
            continue;
        }
        let hash = decode_pubkey_hash(addr)?;
        hashes.push(hash);
    }

    println!("Loaded {} addresses", hashes.len());

    hashes.par_sort_unstable();
    hashes.dedup();

    println!("After deduplication: {}", hashes.len());

    let mut output = BufWriter::new(File::create(output_path)?);

    write_header(&mut output, u64::try_from(hashes.len())?)?;

    for hash in &hashes {
        output.write_all(hash)?;
    }

    output.flush()?;

    println!(
        "Index successfully written to {} ({} entries)",
        output_path,
        hashes.len()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_pubkey_hash_valid() {
        // Genesis address
        let addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";
        let result = decode_pubkey_hash(addr);
        assert!(
            result.is_ok(),
            "Valid address should be decoded successfully"
        );
    }

    #[test]
    fn test_decode_pubkey_hash_invalid_base58() {
        let addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfN0"; // '0' is not a valid base58 character
        let result = decode_pubkey_hash(addr);
        assert!(result.is_err(), "Invalid base58 should return an error");
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("provided string contained invalid character"));
    }

    #[test]
    fn test_decode_pubkey_hash_invalid_length() {
        // Just the prefix + some bytes
        let addr = bs58::encode([0x00, 0x01, 0x02, 0x03]).into_string();
        let result = decode_pubkey_hash(&addr);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid address length"));
    }

    #[test]
    fn test_decode_pubkey_hash_wrong_prefix() {
        // Create a fake address with testnet prefix (0x6f)
        let mut data = vec![0x6f];
        data.extend_from_slice(&[0u8; 20]);
        let checksum = Sha256::digest(Sha256::digest(&data));
        data.extend_from_slice(&checksum[0..4]);

        let addr = bs58::encode(data).into_string();
        let result = decode_pubkey_hash(&addr);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Not a mainnet P2PKH address"));
    }

    #[test]
    fn test_decode_pubkey_hash_bad_checksum() {
        // Genesis address with last character changed
        let addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb"; // Changed 'a' to 'b'
        let result = decode_pubkey_hash(addr);
        assert!(result.is_err());
        // Could be invalid base58 or bad checksum depending on what 'b' decodes to.
        // Let's create a guaranteed bad checksum address instead.

        let mut data = vec![0x00];
        data.extend_from_slice(&[0u8; 20]);
        let checksum = Sha256::digest(Sha256::digest(&data));
        let mut bad_checksum = checksum[0..4].to_vec();
        bad_checksum[0] ^= 0xff; // flip bits to make it bad
        data.extend_from_slice(&bad_checksum);

        let bad_addr = bs58::encode(data).into_string();
        let bad_result = decode_pubkey_hash(&bad_addr);
        assert!(bad_result.is_err());
        assert!(bad_result
            .unwrap_err()
            .to_string()
            .contains("Bad checksum for address"));
    }
}
