use anyhow::Result;

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;

pub trait HashBackend {
    fn name(&self) -> &'static str;
    fn hash_batch_from_secret_bytes(&self, secret_keys: &[[u8; 32]]) -> Result<Vec<[u8; 20]>>;
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
