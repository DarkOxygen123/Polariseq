//! Compute backend abstraction.
//!
//! This is the architectural keystone of Polariseq. The same high-level API
//! (Python or Rust) targets a [`ComputeBackend`], and the concrete backend
//! decides where the work runs: the host CPU, an NVIDIA GPU (CUDA), any
//! wgpu-compatible device (Metal / Vulkan / DX12), or a multi-node cluster.
//!
//! Today (Phase 0) only the trait and a stub [`CpuBackend`] exist. Real kernels
//! land per the roadmap:
//!
//! | Backend            | Phase | Backing tech                       |
//! |--------------------|-------|------------------------------------|
//! | `CpuBackend`       | 2+    | `faer` + `sprs` + `rayon`          |
//! | `CudaBackend`      | 7     | cuBLAS / cuSOLVER FFI              |
//! | `WgpuBackend`      | 8     | `wgpu` (Metal/Vulkan/DX12)         |
//! | `DistributedBackend` | 9   | MPI via `rsmpi`                    |
//! | `HpuBackend`       | 10    | oneAPI / SYCL (Intel Gaudi)        |
//!
//! The trait is intentionally minimal now: it exists so that the crate
//! boundaries are right from day one, not so that it is feature-complete.

use crate::matrix::Matrix;
use crate::neighbors::NeighborsOptions;
use crate::pca::PcaOptions;

/// The result of running a canonical PCA: a cells × components embedding plus
/// the per-component variance ratio (what scanpy stores in `obsm['X_pca']` and
/// `uns['pca']['variance_ratio']`).
///
/// Fields are owned `Vec<f32>` because PCA output is small relative to the
/// input matrix and owning it keeps the API simple for Phase 0. Phase 1+ will
/// introduce zero-copy / mmap-backed variants for very large `n_obs`.
#[derive(Debug, Clone, PartialEq)]
pub struct PcaResult {
    /// `n_obs * n_components` row-major embedding.
    pub embedding: Vec<f32>,
    /// `n_components` ratio of variance explained by each PC.
    pub variance_ratio: Vec<f32>,
    /// Number of observations (cells) in the input.
    pub n_obs: usize,
    /// Number of components computed.
    pub n_components: usize,
}

/// A k-nearest-neighbor graph over cells, the output of `neighbors`. Matches
/// the scanpy convention: a sparse adjacency stored as CSR-style triple in
/// `obsp['connectivities']` and `obsp['distances']`.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborGraph {
    /// CSR `indptr` (length `n_obs + 1`).
    pub indptr: Vec<u32>,
    /// CSR `indices` (length = number of directed edges).
    pub indices: Vec<u32>,
    /// CSR `data` (connectivity weights; aligned with `indices`).
    pub data: Vec<f32>,
    /// Number of cells (rows of the graph).
    pub n_obs: usize,
}

/// Where work executes. The user-facing knob on the Python side
/// (`ps.set_backend("cpu")`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Host CPU: SIMD + `rayon` parallelism. Always available.
    Cpu,
    /// NVIDIA GPU via CUDA. Phase 7.
    Cuda,
    /// Portable GPU via wgpu (Metal/Vulkan/DX12). Phase 8.
    Wgpu,
    /// Multi-node distributed via MPI. Phase 9.
    Distributed,
    /// HPU (Intel Gaudi) via oneAPI/SYCL. Phase 10.
    Hpu,
}

/// Trait every execution backend implements. Today this is a stub: it declares
/// the shape of the API (a method per canonical op) so future phases can fill
/// in real kernels without restructuring call sites.
///
/// Inputs are intentionally raw slices for now. Phase 1 introduces a proper
/// `Matrix` type (dense / CSR / CSC, chunked, mmap-aware) and the signatures
/// will take `&Matrix` instead.
pub trait ComputeBackend {
    /// Which hardware this backend targets.
    fn kind(&self) -> BackendKind;

    /// Human-readable name (e.g. `"cpu"`, `"cuda:0"`).
    fn name(&self) -> &'static str;

    /// Truncated / randomized PCA.
    ///
    /// # Errors
    /// Returns [`BackendError::Pca`] on invalid inputs or SVD failure.
    fn pca(&self, matrix: &Matrix, options: &PcaOptions) -> Result<PcaResult, BackendError>;

    /// kNN graph over a cell embedding. Returns the distance graph (scanpy's
    /// `obsp['distances']`); connectivity weights are derived downstream.
    ///
    /// # Errors
    /// Returns [`BackendError::Neighbors`] on invalid inputs.
    fn neighbors(
        &self,
        embedding: &[f32],
        n_obs: usize,
        n_dim: usize,
        options: &NeighborsOptions,
    ) -> Result<NeighborGraph, BackendError>;
}

/// Failures raised by compute backends.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The operation is not implemented yet (gated behind a future phase).
    #[error("`{op}` is not implemented on this backend (lands in phase {phase})")]
    NotImplemented {
        /// Operation name, e.g. `"neighbors"`.
        op: &'static str,
        /// Roadmap phase in which this op ships.
        phase: u8,
    },
    /// PCA failed (bad inputs or SVD convergence).
    #[error("PCA failed: {0}")]
    Pca(#[from] crate::pca::PcaError),
    /// Neighbor graph construction failed.
    #[error("neighbors failed: {0}")]
    Neighbors(#[from] crate::neighbors::NeighborsError),
}

/// The default CPU backend. Backed by `faer` (SVD) and `rayon` (parallelism).
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn pca(&self, matrix: &Matrix, options: &PcaOptions) -> Result<PcaResult, BackendError> {
        crate::pca::pca(matrix, options).map_err(BackendError::from)
    }

    fn neighbors(
        &self,
        embedding: &[f32],
        n_obs: usize,
        n_dim: usize,
        options: &NeighborsOptions,
    ) -> Result<NeighborGraph, BackendError> {
        crate::neighbors::neighbors(embedding, n_obs, n_dim, options).map_err(BackendError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_reports_correct_kind() {
        let b = CpuBackend;
        assert_eq!(b.kind(), BackendKind::Cpu);
        assert_eq!(b.name(), "cpu");
    }

    #[test]
    fn cpu_backend_runs_pca() {
        use crate::matrix::{DenseMatrix, Matrix};
        use crate::pca::PcaOptions;
        // 50×5 dense matrix; request 3 components.
        let data: Vec<f32> = (0..250).map(|i| (i as f32) * 0.01).collect();
        let m = Matrix::Dense(DenseMatrix::from_row_major(data, 50, 5).unwrap());
        let b = CpuBackend;
        let result = b
            .pca(
                &m,
                &PcaOptions {
                    n_components: 3,
                    ..Default::default()
                },
            )
            .expect("PCA should run on CpuBackend");
        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_components, 3);
        assert_eq!(result.embedding.len(), 50 * 3);
    }

    #[test]
    fn cpu_backend_runs_neighbors() {
        use crate::neighbors::NeighborsOptions;
        // 4 cells in 2D; ask for 3 neighbors (incl. self).
        let emb = [0.0_f32, 0.0, 1.0, 0.0, 10.0, 10.0, 0.0, 1.0];
        let b = CpuBackend;
        let result = b
            .neighbors(
                &emb,
                4,
                2,
                &NeighborsOptions {
                    n_neighbors: 3,
                    ..Default::default()
                },
            )
            .expect("neighbors should run on CpuBackend");
        assert_eq!(result.n_obs, 4);
    }
}
