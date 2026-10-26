//! `.h5ad` (`AnnData` on HDF5) reader.
//!
//! Reads the X matrix (dense or sparse CSR/CSC) plus obs/var into Polariseq's
//! native types. The on-disk layout follows the anndata element-encoding spec:
//! every element has `encoding-type` / `encoding-version` attributes.
//!
//! Currently supported:
//! - `X` as dense `array` (float32, float64)
//! - `X` as `csr_matrix` / `csc_matrix` (the sparse triple `{data,indices,indptr}` + `shape`)
//! - `obs`/`var` row names (the `_index`) and numeric columns
//!
//! Not yet supported (incremental, as needed): categorical obs columns, nullable
//! arrays, string columns beyond the index, `uns`, `obsm`/`varm`. Each lands as
//! a downstream phase requires it; the X-matrix path — the hot path — is
//! complete.

use hdf5_metno::{File, Group};
use polariseq_core::{CsrMatrix, DenseMatrix, Matrix};

use crate::{IoError, Result};

/// An opened `.h5ad` file. Owns the HDF5 handle (file stays open while you read).
///
/// `AnnDataFile` is the entry point: open with [`AnnDataFile::open`], then read
/// the X matrix with [`AnnDataFile::read_x`] and metadata with
/// [`AnnDataFile::read_obs_names`] / [`AnnDataFile::read_var_names`].
pub struct AnnDataFile {
    file: File,
}

/// The encoding-type of the X element, as declared in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XEncoding {
    /// Dense array (`encoding-type: "array"`).
    Dense,
    /// CSR sparse matrix.
    CsrMatrix,
    /// CSC sparse matrix.
    CscMatrix,
}

/// Metadata describing the X matrix before materializing it.
#[derive(Debug, Clone, PartialEq)]
pub struct XInfo {
    /// Encoding (dense / CSR / CSC).
    pub encoding: XEncoding,
    /// Number of rows (cells).
    pub n_obs: usize,
    /// Number of columns (features).
    pub n_vars: usize,
    /// Number of stored nonzeros (sparse) or `n_obs * n_vars` (dense).
    pub n_elements: usize,
}

impl AnnDataFile {
    /// Open a `.h5ad` file read-only.
    ///
    /// # Errors
    /// Returns [`IoError::Hdf5`] if the file cannot be opened.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self { file })
    }

    /// The underlying HDF5 file handle (for advanced use).
    #[must_use]
    pub fn inner(&self) -> &File {
        &self.file
    }

    /// Inspect X's encoding and shape without materializing the matrix.
    /// Cheap (only reads attributes + dataset shapes).
    ///
    /// # Errors
    /// Returns [`IoError::InvalidLayout`] if X is missing or malformed.
    pub fn x_info(&self) -> Result<XInfo> {
        // X may be a group (sparse) or a dataset (dense). Try group first.
        if let Ok(x) = self.file.group("X") {
            let encoding_type = read_attr_string(&x, "encoding-type")?;
            match encoding_type.as_deref() {
                Some("csr_matrix" | "csc_matrix") => {
                    let encoding = if encoding_type.as_deref() == Some("csr_matrix") {
                        XEncoding::CsrMatrix
                    } else {
                        XEncoding::CscMatrix
                    };
                    let (n_obs, n_vars) = read_shape_attr(&x)?;
                    let nnz = x.dataset("data")?.shape().iter().product();
                    Ok(XInfo {
                        encoding,
                        n_obs,
                        n_vars,
                        n_elements: nnz,
                    })
                }
                Some("array") => {
                    // Dense X stored as a group containing the array (rare).
                    Err(IoError::InvalidLayout(
                        "X is a group with encoding array — unsupported layout".into(),
                    ))
                }
                other => Err(IoError::InvalidLayout(format!(
                    "unknown X encoding-type: {other:?}"
                ))),
            }
        } else {
            // Dense X is a top-level dataset.
            let ds = self.file.dataset("X")?;
            let shape = ds.shape();
            if shape.len() != 2 {
                return Err(IoError::InvalidLayout(format!(
                    "dense X must be 2-D, got {}-D",
                    shape.len()
                )));
            }
            Ok(XInfo {
                encoding: XEncoding::Dense,
                n_obs: shape[0],
                n_vars: shape[1],
                n_elements: shape[0] * shape[1],
            })
        }
    }

    /// Read the X matrix into a Polariseq [`Matrix`].
    ///
    /// Returns dense or sparse depending on what's on disk. Float64 data is
    /// narrowed to float32 (our default — see `matrix.rs` rationale).
    ///
    /// # Errors
    /// Returns [`IoError`] on any read or layout failure.
    pub fn read_x(&self) -> Result<Matrix> {
        let info = self.x_info()?;
        match info.encoding {
            XEncoding::CsrMatrix => Ok(Matrix::Csr(self.read_csr()?)),
            XEncoding::CscMatrix => Ok(Matrix::Csr(self.read_csc_as_csr()?)),
            XEncoding::Dense => Ok(Matrix::Dense(self.read_dense()?)),
        }
    }

    /// Read a CSR X matrix.
    fn read_csr(&self) -> Result<CsrMatrix> {
        read_sparse_as_csr(&self.file.group("X")?, "csr")
    }

    /// Read a CSC X matrix and convert to CSR (canonical for our pipeline)
    /// by counting sort — O(nnz), never densified.
    fn read_csc_as_csr(&self) -> Result<CsrMatrix> {
        let g = self.file.group("X")?;
        let (nrows, ncols) = read_shape_attr(&g)?;
        let data: Vec<f32> = read_dataset_f32(&g, "data")?;
        let indices: Vec<u32> = read_dataset_u32(&g, "indices")?;
        let indptr: Vec<u32> = read_dataset_u32(&g, "indptr")?;
        CsrMatrix::from_csc(&data, &indices, &indptr, nrows, ncols).map_err(IoError::from)
    }

    /// Read a dense X matrix.
    fn read_dense(&self) -> Result<DenseMatrix> {
        let ds = self.file.dataset("X")?;
        let shape = ds.shape();
        if shape.len() != 2 {
            return Err(IoError::InvalidLayout(format!(
                "dense X must be 2-D, got {}-D",
                shape.len()
            )));
        }
        let (nrows, ncols) = (shape[0], shape[1]);
        let data = read_dataset_f32(&self.file, "X")?;
        Ok(DenseMatrix { data, nrows, ncols })
    }

    /// Read the obs row names (the `_index` column of `/obs`).
    ///
    /// # Errors
    /// Returns [`IoError`] if obs is missing or has no index.
    pub fn read_obs_names(&self) -> Result<Vec<String>> {
        read_index(&self.file.group("obs")?)
    }

    /// Read the var row names (the `_index` column of `/var`).
    ///
    /// # Errors
    /// Returns [`IoError`] if var is missing or has no index.
    pub fn read_var_names(&self) -> Result<Vec<String>> {
        read_index(&self.file.group("var")?)
    }

    /// The columns of `obs` or `var`, or of those named in `want`, each in
    /// its narrowest form (see [`crate::frame`]). A file from anndata before
    /// 0.7, whose dataframe is one compound dataset, has none that are read.
    ///
    /// # Errors
    /// Returns [`IoError`] if a named column is absent (unless `missing_ok`)
    /// or a read fails.
    pub fn read_frame(
        &self,
        axis: &str,
        want: Option<&[String]>,
        missing_ok: bool,
    ) -> Result<crate::frame::Frame> {
        match self.file.group(axis) {
            Ok(g) => crate::frame::read_frame(&g, want, missing_ok),
            Err(_) if self.file.dataset(axis).is_ok() => Ok(crate::frame::Frame::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Read a numeric obs/var column into a `Vec<f32>`. Returns `None` if the
    /// column is absent or not numeric.
    ///
    /// # Errors
    /// Returns [`IoError::InvalidLayout`] only if the group itself is missing.
    pub fn read_numeric_column(&self, axis: &str, name: &str) -> Result<Option<Vec<f32>>> {
        let g = self.file.group(axis)?;
        // The column may be a plain dataset or a group (nullable/categorical).
        if let Ok(ds) = g.dataset(name) {
            let v = read_container_f32(&ds)?;
            return Ok(Some(v));
        }
        Ok(None)
    }
}

/// Top-level convenience: open a `.h5ad` and read its X matrix.
///
/// # Errors
/// Returns [`IoError`] on any failure.
pub fn read_h5ad<P: AsRef<std::path::Path>>(path: P) -> Result<Matrix> {
    let f = AnnDataFile::open(path)?;
    f.read_x()
}

// --- helpers -----------------------------------------------------------------

/// Read the `shape` attribute on a sparse X group as `(nrows, ncols)`.
fn read_shape_attr(group: &Group) -> Result<(usize, usize)> {
    let attr = group
        .attr("shape")
        .map_err(|e| IoError::InvalidLayout(format!("X has no shape attr: {e}")))?;
    let v: Vec<u64> = attr.read_raw::<u64>()?;
    if v.len() != 2 {
        return Err(IoError::InvalidLayout(format!(
            "X/shape must be length 2, got {}",
            v.len()
        )));
    }
    Ok((v[0] as usize, v[1] as usize))
}

/// Read a string attribute, returning `None` if the attribute is absent.
pub(crate) fn read_attr_string(group: &Group, name: &str) -> Result<Option<String>> {
    let Ok(attr) = group.attr(name) else {
        return Ok(None);
    };
    // String attrs are stored as VarLenUnicode.
    let s = attr
        .read_scalar::<hdf5_metno::types::VarLenUnicode>()
        .or_else(|_| {
            // Fall back: some writers use fixed-length unicode.
            let v: Vec<hdf5_metno::types::VarLenUnicode> = attr.read_raw()?;
            Ok::<_, hdf5_metno::Error>(v.into_iter().next().unwrap_or_default())
        })?;
    Ok(Some(s.to_string()))
}

/// Read the sparse `{data,indices,indptr}` triple of a CSR/CSC group into our
/// `CsrMatrix`. Caller has verified this is CSR (CSC is handled separately).
fn read_sparse_as_csr(group: &Group, _kind: &str) -> Result<CsrMatrix> {
    let (nrows, ncols) = read_shape_attr(group)?;
    let data: Vec<f32> = read_dataset_f32(group, "data")?;
    let indices: Vec<u32> = read_dataset_u32(group, "indices")?;
    let indptr: Vec<u32> = read_dataset_u32(group, "indptr")?;
    CsrMatrix::new(indptr, indices, data, nrows, ncols).map_err(IoError::from)
}

/// Read a 1-D dataset belonging to `group`, as `Vec<f32>`. Tries the
/// chunk-parallel fast path first (gzipped datasets), then the library
/// path (f32 then f64 narrowing — the two dtypes anndata writes for
/// expression data).
fn read_dataset_f32(group: &Group, name: &str) -> Result<Vec<f32>> {
    let ds = group.dataset(name)?;
    if std::env::var_os("POLARISEQ_FASTREAD").is_some() {
        if let Some(v) = crate::fastread::read_f32(&ds) {
            return Ok(v);
        }
    }
    read_container_f32(&ds)
}

/// Read a `Container` (dataset or attribute) as `Vec<f32>`, trying f32 then f64.
fn read_container_f32(c: &hdf5_metno::Container) -> Result<Vec<f32>> {
    let v = c.read_raw::<f32>().or_else(|_| {
        let v = c.read_raw::<f64>()?;
        Ok::<_, hdf5_metno::Error>(v.into_iter().map(|x| x as f32).collect::<Vec<_>>())
    })?;
    Ok(v)
}

/// Read a 1-D dataset belonging to `group` as `Vec<u32>`. anndata writes
/// `indices`/`indptr` as int32 (sometimes int64); we widen.
fn read_dataset_u32(group: &Group, name: &str) -> Result<Vec<u32>> {
    let ds = group.dataset(name)?;
    if std::env::var_os("POLARISEQ_FASTREAD").is_some() {
        if let Some(v) = crate::fastread::read_u32(&ds) {
            return Ok(v);
        }
    }
    widen_integer_dataset(&ds, name)
}

/// Read an integer dataset (`indices`/`indptr`) as `Vec<u32>`, dispatching on
/// the dtype stored in the file. anndata widens `indptr` to int64 once a file
/// holds more than 2^31 nonzeros, so both layouts occur in the wild.
///
/// The memory type must follow the file type. Reading an int64 dataset
/// through an int32 memory type does not fail — HDF5 converts silently, and
/// on overflow the converted value no longer matches the stored one. Measured
/// on the 1,306,127-cell mouse-brain file (indptr int64, 2,624,828,308
/// nonzeros): the whole-dataset `read_raw::<i32>` returned 2,147,483,647 for
/// the final row pointer, the CSR triple then failed shape validation, and
/// the error named two lengths that agreed — the corruption was invisible
/// from the message. Dispatching on the stored dtype removes the conversion
/// entirely; values are range-checked on the way in.
pub(crate) fn widen_integer_dataset(ds: &hdf5_metno::Dataset, name: &str) -> Result<Vec<u32>> {
    use hdf5_metno::types::{IntSize, TypeDescriptor};
    let is_64 = matches!(
        ds.dtype()
            .map_err(IoError::Hdf5)?
            .to_descriptor()
            .map_err(IoError::Hdf5)?,
        TypeDescriptor::Integer(IntSize::U8) | TypeDescriptor::Unsigned(IntSize::U8)
    );
    if is_64 {
        let v: Vec<i64> = ds.read_raw().map_err(IoError::Hdf5)?;
        return v
            .into_iter()
            .map(|x| {
                u32::try_from(x).map_err(|_| {
                    IoError::InvalidLayout(format!(
                        "'{name}' holds {x}, above the u32 index limit {}",
                        u32::MAX
                    ))
                })
            })
            .collect();
    }
    // int32 (the common case): widen. Negative values have no meaning as
    // indices; widening them unchanged keeps the CSR validation as the place
    // that rejects a malformed triple.
    let v: Vec<i32> = ds.read_raw().map_err(IoError::Hdf5)?;
    Ok(v.into_iter().map(|x| x as u32).collect())
}

/// Read the `_index` of an obs/var dataframe group. Handles the modern
/// `nullable-string-array` layout (a group with `values`+`mask`) and the legacy
/// plain-string-array layout.
fn read_index(group: &Group) -> Result<Vec<String>> {
    let index_name = read_attr_string(group, "_index")?
        .ok_or_else(|| IoError::InvalidLayout("obs/var has no _index attr".into()))?;
    // Modern: _index is a group with a `values` string-array sub-dataset.
    if let Ok(idx_group) = group.group(&index_name) {
        if let Ok(values) = idx_group.dataset("values") {
            let v: Vec<hdf5_metno::types::VarLenUnicode> = values.read_raw()?;
            return Ok(v.into_iter().map(|s| s.to_string()).collect());
        }
    }
    // Legacy: _index is itself a string dataset.
    if let Ok(ds) = group.dataset(&index_name) {
        let v: Vec<hdf5_metno::types::VarLenUnicode> = ds.read_raw()?;
        return Ok(v.into_iter().map(|s| s.to_string()).collect());
    }
    Err(IoError::InvalidLayout(format!(
        "could not read index `{index_name}`"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../tests/data");
        p.push(name);
        p
    }

    #[test]
    fn int64_indptr_above_i32_max_round_trips() {
        // anndata widens indptr to int64 past 2^31 nonzeros, and the 1.3M
        // mouse-brain file's tail (2,624,828,308) sits above i32::MAX.
        // Reading that dataset through an int32 memory type used to return a
        // converted vector of the right length with a corrupted tail.
        let mut path = std::env::temp_dir();
        path.push(format!("polariseq_i64_indptr_{}.h5", std::process::id()));
        let tail: i64 = 2_624_828_308;
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            let group = file.create_group("X").unwrap();
            let ds = group
                .new_dataset::<i64>()
                .shape(3)
                .create("indptr")
                .unwrap();
            ds.write_raw(&[0i64, 2_147_483_648, tail]).unwrap();
        }
        let file = File::open(&path).unwrap();
        let group = file.group("X").unwrap();
        let v = widen_integer_dataset(&group.dataset("indptr").unwrap(), "indptr").unwrap();
        assert_eq!(v, vec![0u32, 2_147_483_648, u32::try_from(tail).unwrap()]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn int64_index_above_u32_limit_is_rejected() {
        let mut path = std::env::temp_dir();
        path.push(format!("polariseq_i64_over_{}.h5", std::process::id()));
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            let ds = file.new_dataset::<i64>().shape(2).create("indptr").unwrap();
            ds.write_raw(&[0i64, i64::from(u32::MAX) + 1]).unwrap();
        }
        let file = File::open(&path).unwrap();
        let ds = file.dataset("indptr").unwrap();
        let err = widen_integer_dataset(&ds, "indptr").unwrap_err();
        assert!(err.to_string().contains("u32 index limit"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_sparse_x_info() {
        let f = AnnDataFile::open(fixture("sparse_100x200.h5ad")).unwrap();
        let info = f.x_info().unwrap();
        assert_eq!(info.encoding, XEncoding::CsrMatrix);
        assert_eq!(info.n_obs, 100);
        assert_eq!(info.n_vars, 200);
        assert_eq!(info.n_elements, 1719);
    }

    #[test]
    fn read_dense_x_info() {
        let f = AnnDataFile::open(fixture("tiny_dense.h5ad")).unwrap();
        let info = f.x_info().unwrap();
        assert_eq!(info.encoding, XEncoding::Dense);
        assert_eq!(info.n_obs, 3);
        assert_eq!(info.n_vars, 3);
        assert_eq!(info.n_elements, 9);
    }

    #[test]
    fn read_dense_x_values() {
        let f = AnnDataFile::open(fixture("tiny_dense.h5ad")).unwrap();
        let m = f.read_x().unwrap();
        match m {
            Matrix::Dense(d) => {
                assert_eq!(d.shape(), (3, 3));
                assert_eq!(d.data, vec![1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]);
            }
            Matrix::Csr(_) => panic!("expected dense"),
        }
    }

    #[test]
    fn read_sparse_x_values_and_recover_via_densify() {
        let f = AnnDataFile::open(fixture("sparse_100x200.h5ad")).unwrap();
        let m = f.read_x().unwrap();
        let csr = match m {
            Matrix::Csr(c) => c,
            Matrix::Dense(_) => panic!("expected sparse"),
        };
        assert_eq!(csr.shape(), (100, 200));
        assert_eq!(csr.nnz(), 1719);
        // Every stored value should be > 0 (we sparsified zeros out).
        assert!(csr.data.iter().all(|&v| v > 0.0));
        // Row sums should all be > 0 (each cell had at least one count).
        let sums = csr.row_sums();
        assert!(sums.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn read_obs_and_var_names() {
        let f = AnnDataFile::open(fixture("sparse_100x200.h5ad")).unwrap();
        let obs = f.read_obs_names().unwrap();
        let var = f.read_var_names().unwrap();
        assert_eq!(obs.len(), 100);
        assert_eq!(var.len(), 200);
        assert_eq!(obs[0], "cell_0");
        assert_eq!(var[0], "gene_0");
        assert_eq!(obs[99], "cell_99");
        assert_eq!(var[199], "gene_199");
    }

    #[test]
    fn read_numeric_obs_column() {
        let f = AnnDataFile::open(fixture("sparse_100x200.h5ad")).unwrap();
        let n_counts = f
            .read_numeric_column("obs", "n_counts")
            .unwrap()
            .expect("n_counts column should exist");
        assert_eq!(n_counts.len(), 100);
        // Should match the row sums of X exactly.
        let m = f.read_x().unwrap();
        let Matrix::Csr(csr) = m else {
            unreachable!();
        };
        let row_sums = csr.row_sums();
        for (a, b) in n_counts.iter().zip(&row_sums) {
            assert!((a - b).abs() < 1e-3, "mismatch {a} vs {b}");
        }
    }

    #[test]
    fn read_var_n_cells_column() {
        let f = AnnDataFile::open(fixture("sparse_100x200.h5ad")).unwrap();
        let n_cells = f
            .read_numeric_column("var", "n_cells")
            .unwrap()
            .expect("n_cells column should exist");
        assert_eq!(n_cells.len(), 200);
        // Match column counts of nonzero entries.
        let m = f.read_x().unwrap();
        let Matrix::Csr(csr) = m else {
            unreachable!();
        };
        // Count nonzeros per column.
        let mut col_nnz = vec![0_u32; 200];
        for i in 0..100 {
            let (cols, _) = csr.row(i);
            for &c in cols {
                col_nnz[c as usize] += 1;
            }
        }
        for (got, want) in n_cells.iter().zip(&col_nnz) {
            assert!(
                (*got - *want as f32).abs() < 0.5,
                "n_cells mismatch {got} vs {want}"
            );
        }
    }
}
