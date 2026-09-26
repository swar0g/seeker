use anyhow::Result;

use super::HashBackend;

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl HashBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn hash_batch_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>> {
        crate::hash_batch_from_secret_bytes(secret_keys)
    }
}
