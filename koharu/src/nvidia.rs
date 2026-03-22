use std::fmt;

use anyhow::{Context, Result};

const CUDA_SUCCESS: i32 = 0;
const CUDA_12_9_DRIVER_VERSION: i32 = 12090;

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CuDriverGetVersion = unsafe extern "C" fn(driver_version: *mut i32) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudaDriverVersion {
    raw: i32,
}

impl CudaDriverVersion {
    pub const fn from_raw(raw: i32) -> Self {
        Self { raw }
    }

    pub const fn raw(self) -> i32 {
        self.raw
    }

    pub const fn major(self) -> i32 {
        self.raw / 1000
    }

    pub const fn minor(self) -> i32 {
        (self.raw % 1000) / 10
    }

    pub const fn supports_cuda_12_9(self) -> bool {
        self.raw >= CUDA_12_9_DRIVER_VERSION
    }
}

impl fmt::Display for CudaDriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

pub fn supports_cuda_12_9() -> Result<bool> {
    Ok(driver_version()?.supports_cuda_12_9())]
}

pub fn driver_version() -> Result<CudaDriverVersion> {
    let lib_name = if cfg!(target_os = "windows") {
        "nvcuda.dll"
    } else {
        "libcuda.so"
    };

    unsafe {
        let library = libloading::Library::new(lib_name)
            .with_context(|| format!("Failed to load NVIDIA driver library {lib_name}"))?;
        let cu_init = *library
            .get::<CuInit>(b"cuInit\0")
            .context("Failed to load cuInit from NVIDIA driver")?;
        let cu_driver_get_version = *library
            .get::<CuDriverGetVersion>(b"cuDriverGetVersion\0")
            .context("Failed to load cuDriverGetVersion from NVIDIA driver")?;

        let init_status = cu_init(0);
        ensure_cuda_success(init_status, "cuInit")?;

        let mut raw = 0;
        let version_status = cu_driver_get_version(&mut raw);
        ensure_cuda_success(version_status, "cuDriverGetVersion")?;

        Ok(CudaDriverVersion::from_raw(raw))
    }
}

fn ensure_cuda_success(code: i32, op: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }

    anyhow::bail!("{op} failed with CUDA driver error code {code}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_major_minor_from_driver_version() {
        let version = CudaDriverVersion::from_raw(12090);
        assert_eq!(version.major(), 12);
        assert_eq!(version.minor(), 9);
        assert_eq!(version.to_string(), "12.9");
    }

    #[test]
    fn checks_cuda_12_9_threshold() {
        assert!(CudaDriverVersion::from_raw(12090).supports_cuda_12_9());
        assert!(CudaDriverVersion::from_raw(12010).supports_cuda_12_9());
        assert!(!CudaDriverVersion::from_raw(12080).supports_cuda_12_9());
        assert!(!CudaDriverVersion::from_raw(11000).supports_cuda_12_9());
    }
]
