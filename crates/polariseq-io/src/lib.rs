//! # polariseq-io
//!
//! Readers and writers for the on-disk formats biologists already use. Adoption
//! in bioinformatics is gated on format interop: a new tool that can't read the
//! `.h5ad` / Zarr files people already have is dead on arrival.
//!
//! ## The h5ad format
//!
//! An `.h5ad` is an HDF5 file with the `anndata` element-encoding spec. Every
//! element carries `encoding-type` / `encoding-version` attributes so readers
//! can dispatch:
//!
//! | Element            | Group / dataset                                          |
//! |--------------------|----------------------------------------------------------|
//! | `X` dense          | dataset, attrs `{encoding-type: "array"}`                |
//! | `X` CSR sparse     | group `{data, indices, indptr}` + attr `shape`           |
//! | `obs`/`var`        | dataframe group: `_index` + `column-order` attr          |
//! | categorical column | group `{codes, categories}` + attr `ordered`             |
//!
//! This reader pulls CSR triples directly into [`polariseq_core::CsrMatrix`] and
//! dense datasets into [`polariseq_core::DenseMatrix`], with zero sparse-libs
//! dependency in the loop — we control the storage layout end to end.
//!
//! Backed by `hdf5-metno` 0.14 (the maintained fork of `hdf5-rust`; the
//! original `aldanor/hdf5-rust` 0.8 cannot parse HDF5 2.x).

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

/// A path or URL we can read from. Cloud object stores (`s3://`, `gcs://`) land
/// in Phase 6 alongside the Zarr-v3 cloud backend.
#[derive(Debug, Clone)]
pub struct DataSource {
    /// The on-disk path, `s3://...`/`gcs://...` URL, or local file.
    pub uri: String,
    /// What format the bytes are in.
    pub format: DataFormat,
}

/// The container format of a [`DataSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// `AnnData` on HDF5 (`.h5ad`). The single most common on-disk format.
    H5ad,
    /// `AnnData` on Zarr v2 or v3 (`.h5ad.zarr`). Cloud-native; the recommended
    /// format going forward, and what Vitessce consumes.
    AnnDataZarr,
    /// `MuData` on HDF5 (`.h5mu`). Multimodal; each `/mod/<modality>/` subgroup
    /// is itself an `AnnData`.
    H5mu,
    /// 10x Cell Ranger matrix market (`matrix.mtx` + barcodes + features).
    Mtx,
}

/// Failures raised by IO operations.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// The reader for this format is not implemented yet.
    #[error("reading `{format}` is not implemented yet (lands in phase {phase})")]
    NotImplemented {
        /// Format name.
        format: &'static str,
        /// Roadmap phase in which the reader ships.
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
