//! # polariseq-core
//!
//! Single-cell RNA-seq analysis on sparse count matrices: quality control and
//! normalization, gene selection, PCA, the neighbor graph, Leiden clustering,
//! UMAP, Harmony integration, doublet detection, marker genes, cell type scores
//! and pseudobulk comparisons. Pure Rust, parallel with rayon, and the same
//! result at any thread count. It reads no files: [`polariseq-io`] does.
//!
//! [`polariseq-io`]: https://github.com/DarkOxygen123/Polariseq/tree/main/crates/polariseq-io
//!
//! # Examples
//!
//! Normalize a count matrix, then cluster its cells:
//!
//! ```
//! use polariseq_core::preprocess::{log1p_inplace, normalize_total_inplace};
//! use polariseq_core::{leiden, neighbors, pca_csr, CsrMatrix};
//! use polariseq_core::{LeidenOptions, NeighborsOptions, PcaOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // 8 cells × 4 genes as compressed sparse rows: the first four cells
//! // express genes 0 and 1, the last four genes 2 and 3.
//! let indptr = vec![0, 2, 4, 6, 8, 10, 12, 14, 16];
//! let indices = vec![0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3];
//! let data = vec![5.0, 3.0, 4.0, 2.0, 6.0, 3.0, 5.0, 4.0,
//!                 3.0, 6.0, 2.0, 5.0, 4.0, 4.0, 3.0, 5.0];
//! let mut counts = CsrMatrix::new(indptr, indices, data, 8, 4)?;
//!
//! normalize_total_inplace(&mut counts, 1e4);
//! log1p_inplace(&mut counts);
//!
//! let pcs = pca_csr(&counts, &PcaOptions { n_components: 2, ..Default::default() })?;
//! let graph = neighbors(
//!     &pcs.embedding,
//!     pcs.n_obs,
//!     pcs.n_components,
//!     &NeighborsOptions { n_neighbors: 3, ..Default::default() },
//! )?;
//! let clusters = leiden(&graph, &LeidenOptions::default())?;
//!
//! assert_eq!(clusters.labels.len(), 8);
//! # Ok(())
//! # }
//! ```
//!
//! # Where to find each step
//!
//! | step | module |
//! |---|---|
//! | the matrix types | [`matrix`]: [`CsrMatrix`], [`DenseMatrix`] |
//! | quality control, normalization, gene selection | [`preprocess`] |
//! | PCA | [`pca`](mod@pca), with the randomized SVD of [`rsvd`] |
//! | the neighbor graph | [`neighbors`](mod@neighbors) (exact), [`nndescent`] (approximate) |
//! | clustering | [`clustering`] (Leiden), [`louvain`], [`hierarchy`] (Paris) |
//! | UMAP | [`umap`] |
//! | batch correction | [`harmony`] |
//! | doublets | [`doublets`] (Scrublet) |
//! | marker genes | [`de`] |
//! | cell type scores | [`annotate`] |
//! | conditions with donors as replicates | [`pseudobulk`] |
//! | trajectories | [`trajectory`] (PAGA, pseudotime) |
//! | matrices larger than memory | [`rowblocks`], [`store`], [`file_matrix`] |
//!
//! Most kernels also take a [`rowblocks::RowBlocks`] source, so the same call
//! runs on a matrix in memory or on one read from disk a block at a time.
//!
//! # Requirements
//!
//! Rust 1.82 or later. No system libraries.
//!
//! # Documentation and citation
//!
//! These crates are the computational core of Polariseq, built for single-cell
//! RNA-seq analysis. The [Polariseq documentation](../../../index.html)
//! explains the analysis steps, the methods they follow and the Python
//! interface.
//!
//! If you use Polariseq, cite it with the version you used:
//!
//! > Uthamkumar M, Dutta A, Basak T. Polariseq: atlas-scale single-cell
//! > analysis on the computer you have.
#![doc = concat!("> Version ", env!("CARGO_PKG_VERSION"), " (2026).")]
//! > <https://github.com/DarkOxygen123/Polariseq>

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

pub mod annotate;
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

/// Doubles every value in place, in parallel: the build's round-trip check.
/// `polariseq._polariseq.triple` (from Python) and the `core_smoke` benchmark
/// call it to show that the core is linked and its thread pool runs.
///
/// # Panics
/// Never panics. An empty slice is left as it is.
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
