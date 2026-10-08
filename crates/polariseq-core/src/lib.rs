//! # polariseq-core
//!
//! Pure-Rust computational core for Polariseq. No Python bindings, no IO, no
//! filesystem — only data structures, algorithms, and the [`ComputeBackend`]
//! abstraction that lets the same high-level API target CPU / CUDA / wgpu /
//! distributed execution later.
//!
//! See `docs/architecture.md` for the rationale behind these types.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
// Numeric code routinely casts usize <-> u32/f64. These lints fire on every
// such cast; they're correct in principle but noisy for a numerics crate where
// the cast ranges are known-bounded (matrices don't exceed 2^32 elements).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

pub mod backend;
pub mod clustering;
pub mod compressed;
pub mod de;
pub mod disk_matrix;
pub mod doublets;
pub mod file_matrix;
pub mod harmony;
pub mod hierarchy;
mod leiden;
pub mod louvain;
pub mod lowess;
pub mod matrix;
pub mod neighbors;
pub mod nndescent;
pub mod pca;
pub mod preprocess;
pub mod pseudobulk;
pub mod rowblocks;
pub mod rsvd;
pub mod simd;
pub mod store;
pub mod trajectory;
pub mod umap;

use rayon::prelude::*;

/// Re-export the most-used items at the crate root for ergonomic access.
pub use backend::{ComputeBackend, CpuBackend};
pub use clustering::{
    leiden, leiden_connectivities, ClusteringError, ClusteringResult, LeidenOptions,
};
pub use hierarchy::{cut_balanced, cut_straight, paris, HierarchyError, Merge, ParisOptions};
pub use matrix::{CsrMatrix, DenseMatrix, Matrix, MatrixError, MatrixKind};
pub use neighbors::{neighbors, Metric, NeighborsError, NeighborsOptions};
pub use pca::{pca, pca_csr, PcaError, PcaOptions};

/// Crate version (matches the workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A tiny smoke-test function used by the Phase 0 round-trip test. It doubles a
/// slice in place using rayon. This is *not* a real kernel — it exists only to
/// prove the Rust core is reachable and that parallelism works. Real compute
/// kernels (PCA, neighbors, Leiden) arrive in later phases.
///
/// # Panics
/// Never panics; returns the original slice unchanged if empty.
pub fn double_in_place(slice: &mut [f64]) {
    slice.par_iter_mut().for_each(|x| *x *= 2.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_a_slice() {
        let mut v = vec![1.0_f64, 2.0, 3.0, 4.0];
        double_in_place(&mut v);
        assert_eq!(v, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn handles_empty_slice() {
        let mut v: Vec<f64> = Vec::new();
        double_in_place(&mut v);
        assert_eq!(v, Vec::<f64>::new());
    }

    #[test]
    fn version_is_set() {
        assert_ne!(VERSION, "");
    }
}
