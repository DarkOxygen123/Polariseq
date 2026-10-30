//! # polariseq-io
//!
//! Read single-cell data into [`polariseq_core`] matrices: `.h5ad` files
//! (`AnnData` on HDF5) and the 10x Genomics `.h5` file and matrix folder, whole
//! or in blocks of rows, and several files as one dataset. Out of core, it
//! copies the counts once into a compressed block store and streams passes over
//! it.
//!
//! # Examples
//!
//! Read a file's matrix and gene names:
//!
//! ```no_run
//! use polariseq_core::Matrix;
//! use polariseq_io::AnnDataFile;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file = AnnDataFile::open("pbmc3k.h5ad")?;
//! let genes = file.read_var_names()?;
//! let Matrix::Csr(counts) = file.read_x()? else {
//!     return Err("X is stored dense".into());
//! };
//! println!("{} cells × {} genes", counts.nrows, genes.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Where to find each reader
//!
//! | task | module |
//! |---|---|
//! | `.h5ad` files | [`h5ad`]: [`AnnDataFile`], [`read_h5ad`] |
//! | 10x Genomics files | [`tenx`] |
//! | the `obs` and `var` tables | [`frame`] |
//! | several files as one dataset | [`merge`] |
//! | a file's shape and size without reading it | [`lazy`] |
//! | copying the counts into the block store | [`spill`] |
//! | passes over a file on disk | [`stream`], [`lazy_compute`] |
//! | several datasets integrated | [`integrate`], [`stream_correct`] |
//!
//! # Requirements
//!
//! Rust 1.82 or later. HDF5 is built from source with the crate (the `static`
//! feature of `hdf5-metno`), so the first build takes a few minutes and needs a
//! C compiler and CMake.
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

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
// Stylistic pedantic lints that add noise, not safety, in numeric IO code:
// graph/matrix loops read best with the textbook (i, j, k, s, e) names.
#![allow(clippy::many_single_char_names, clippy::items_after_statements)]

mod fastread;
pub mod frame;
pub mod h5ad;
pub mod integrate;
pub mod lazy;
pub mod lazy_compute;
pub mod merge;
pub mod spill;
pub mod stream;
pub mod stream_correct;
pub mod tenx;

pub use h5ad::{read_h5ad, AnnDataFile};
pub use polariseq_core::{matrix::MatrixError, CsrMatrix, DenseMatrix, Matrix};

/// A file to read and the format of its bytes. The readers of this crate take
/// paths directly ([`read_h5ad`], and the 10x readers in [`tenx`]), and this
/// pairing is used by its tests.
#[derive(Debug, Clone)]
pub struct DataSource {
    /// The file's path.
    pub uri: String,
    /// What format the bytes are in.
    pub format: DataFormat,
}

/// The container format of a [`DataSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// `AnnData` on HDF5 (`.h5ad`). The single most common on-disk format.
    H5ad,
    /// `AnnData` on Zarr v2 or v3 (`.zarr`). No reader in this crate.
    AnnDataZarr,
    /// `MuData` on HDF5 (`.h5mu`): multimodal, each `/mod/<modality>/` subgroup
    /// an `AnnData`. No reader in this crate.
    H5mu,
    /// 10x Cell Ranger matrix market (`matrix.mtx` + barcodes + features).
    Mtx,
}

/// Failures raised by IO operations.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// The format has no reader in this crate.
    #[error("reading `{format}` is not implemented yet (lands in phase {phase})")]
    NotImplemented {
        /// Format name.
        format: &'static str,
        /// Development phase the message names.
        phase: u8,
    },
    /// The HDF5 layer returned an error.
    #[error("hdf5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),
    /// The matrix layer returned an error.
    #[error("matrix error: {0}")]
    Matrix(#[from] MatrixError),
    /// An IO error (file creation, write).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The arguments ask for something the data cannot give (a gene a filter
    /// removes, a filter that removes everything). The message says what and why.
    #[error("{0}")]
    Rejected(String),
    /// The file does not match the expected anndata layout.
    #[error("invalid anndata layout: {0}")]
    InvalidLayout(String),
}

/// Result alias for IO operations.
pub type Result<T> = std::result::Result<T, IoError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_source_is_constructible() {
        let src = DataSource {
            uri: "s3://bucket/adata.zarr".into(),
            format: DataFormat::AnnDataZarr,
        };
        assert_eq!(src.uri, "s3://bucket/adata.zarr");
        assert_eq!(src.format, DataFormat::AnnDataZarr);
    }
}
