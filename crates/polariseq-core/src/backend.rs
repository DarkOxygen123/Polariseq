//! The compute backend trait: two canonical operations (PCA and the neighbor
//! graph) behind one interface, and [`CpuBackend`], which runs them on the host
//! CPU with `faer` and `rayon`.
//!
//! Every analysis step of Polariseq runs on the CPU and calls its kernels
//! directly (`pca`, `neighbors`, ...). The trait is the boundary at which an
//! implementation of those operations for another device would plug in.
//! [`BackendKind`] names the device classes, and only [`BackendKind::Cpu`] has
//! an implementation.

use crate::matrix::Matrix;
use crate::neighbors::NeighborsOptions;
use crate::pca::PcaOptions;

/// The result of running a canonical PCA: a cells × components embedding plus
/// the per-component variance ratio (what scanpy stores in `obsm['X_pca']` and
/// `uns['pca']['variance_ratio']`).
///
/// Fields are owned `Vec<f32>`: the embedding is small beside the input
/// matrix.
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
    /// CSR `data`: the connectivity weights, aligned with `indices`.
    pub data: Vec<f32>,
    /// Number of cells (rows of the graph).
    pub n_obs: usize,
}

/// The kind of device a backend runs on. Only `Cpu` has an implementation
/// ([`CpuBackend`]). The others name device classes another backend could
/// target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Host CPU: SIMD + `rayon` parallelism. Always available.
    Cpu,
    /// NVIDIA GPU via CUDA (no implementation).
    Cuda,
    /// Portable GPU via wgpu: Metal, Vulkan, DX12 (no implementation).
    Wgpu,
    /// Several nodes via MPI (no implementation).
    Distributed,
    /// Intel Gaudi via oneAPI/SYCL (no implementation).
    Hpu,
}

/// The trait every execution backend implements, with one method per canonical
/// operation. PCA takes a [`Matrix`], and the neighbor graph takes the embedding
/// as a row-major slice, the layout PCA returns.
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

    /// The k-nearest-neighbor graph of a cell embedding. Returns the distance
    /// graph, Scanpy's `obsp['distances']`, from which the connectivity weights
    /// are derived later.
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
    /// The operation has no implementation on this backend.
    #[error("`{op}` is not implemented on this backend (lands in phase {phase})")]
    NotImplemented {
        /// Operation name, e.g. `"neighbors"`.
        op: &'static str,
        /// Development phase the message names.
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
