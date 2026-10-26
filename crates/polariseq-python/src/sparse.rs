//! `SparseMatrix` — the Python handle to a Rust-owned CSR matrix.
//!
//! The expression matrix lives in Rust for the whole pipeline; Python holds
//! this object and passes it back to kernels **by reference**. Nothing
//! proportional to `nnz` crosses the FFI boundary unless the user explicitly
//! asks for it (`to_scipy`, `toarray`, `data`/`indices`/`indptr`).
//!
//! A small scipy-like surface (`shape`, `nnz`, `dtype`, `sum`, `mean`,
//! `toarray`, boolean/integer/slice `__getitem__`) keeps existing notebooks
//! working without a scipy round-trip.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyReadonlyArray1, PyReadonlyArray2};
use polariseq_core::{preprocess, CsrMatrix, MatrixError};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use pyo3::IntoPyObjectExt;
use rayon::prelude::*;

/// A cells × features CSR matrix owned by Rust.
#[pyclass(name = "SparseMatrix", module = "polariseq._polariseq")]
pub struct PySparseMatrix {
    pub(crate) inner: CsrMatrix,
}

impl PySparseMatrix {
    pub(crate) fn new(inner: CsrMatrix) -> Self {
        Self { inner }
    }
}

fn matrix_err(e: MatrixError) -> PyErr {
    match e {
        MatrixError::IndexOutOfBounds { .. } => PyIndexError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// One axis of a `__getitem__` key.
enum Selection {
    All,
    Indices(Vec<usize>),
}

fn mask_to_indices<I: Iterator<Item = bool>>(mask: I, len: usize, n: usize) -> PyResult<Selection> {
    if len != n {
        return Err(PyValueError::new_err(format!(
            "boolean index of length {len} does not match axis of length {n}"
        )));
    }
    Ok(Selection::Indices(
        mask.enumerate()
            .filter_map(|(i, keep)| keep.then_some(i))
            .collect(),
    ))
}

fn ints_to_indices<I: Iterator<Item = i64>>(idx: I, n: usize) -> PyResult<Selection> {
    let n_i = n as i64;
    idx.map(|i| {
        let j = if i < 0 { i + n_i } else { i };
        if j < 0 || j >= n_i {
            Err(PyIndexError::new_err(format!(
                "index {i} is out of bounds for axis of length {n}"
            )))
        } else {
            Ok(j as usize)
        }
    })
    .collect::<PyResult<Vec<usize>>>()
    .map(Selection::Indices)
}

/// Interpret one axis of an index key: a slice (step 1), a boolean mask, an
/// integer array/list, or a single integer.
fn parse_selection(key: &Bound<'_, PyAny>, n: usize) -> PyResult<Selection> {
    if let Ok(slice) = key.cast::<PySlice>() {
        let idx = slice.indices(n as isize)?;
        if idx.step != 1 {
            return Err(PyValueError::new_err("only step-1 slices are supported"));
        }
        let start = idx.start.max(0) as usize;
        let stop = (idx.stop.max(0) as usize).max(start);
        if start == 0 && stop >= n {
            return Ok(Selection::All);
        }
        return Ok(Selection::Indices((start..stop).collect()));
    }
    if let Ok(mask) = key.extract::<PyReadonlyArray1<bool>>() {
        let a = mask.as_array();
        return mask_to_indices(a.iter().copied(), a.len(), n);
    }
    if let Ok(mask) = key.extract::<Vec<bool>>() {
        return mask_to_indices(mask.iter().copied(), mask.len(), n);
    }
    if let Ok(idx) = key.extract::<PyReadonlyArray1<i64>>() {
        return ints_to_indices(idx.as_array().iter().copied(), n);
    }
    if let Ok(idx) = key.extract::<PyReadonlyArray1<i32>>() {
        return ints_to_indices(idx.as_array().iter().map(|&i| i64::from(i)), n);
    }
    if let Ok(idx) = key.extract::<PyReadonlyArray1<u32>>() {
        return ints_to_indices(idx.as_array().iter().map(|&i| i64::from(i)), n);
    }
    if let Ok(idx) = key.extract::<PyReadonlyArray1<u64>>() {
        return ints_to_indices(idx.as_array().iter().map(|&i| i as i64), n);
    }
    if let Ok(i) = key.extract::<i64>() {
        return ints_to_indices(std::iter::once(i), n);
    }
    if let Ok(idx) = key.extract::<Vec<i64>>() {
        return ints_to_indices(idx.into_iter(), n);
    }
    Err(PyTypeError::new_err(
        "index must be a slice, an integer, a boolean mask or an integer array",
    ))
}

fn row_sums_f64(m: &CsrMatrix) -> Vec<f64> {
    (0..m.nrows)
        .into_par_iter()
        .map(|i| m.row(i).1.iter().map(|&v| f64::from(v)).sum())
        .collect()
}

fn col_sums_f64(m: &CsrMatrix) -> Vec<f64> {
    let ncols = m.ncols;
    (0..m.nrows)
        .into_par_iter()
        .fold(
            || vec![0.0_f64; ncols],
            |mut acc, i| {
                let (cols, vals) = m.row(i);
                for (&c, &v) in cols.iter().zip(vals) {
                    acc[c as usize] += f64::from(v);
                }
                acc
            },
        )
        .reduce(
            || vec![0.0_f64; ncols],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += *y;
                }
                a
            },
        )
}

/// The CSR triple as numpy arrays: `(data, indices, indptr)`.
type CsrArrays<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
);

#[pymethods]
impl PySparseMatrix {
    /// Build from a CSR triple. The arrays are copied once into Rust-owned
    /// memory; pass `int32` indices as a `uint32` view to avoid a cast copy.
    #[staticmethod]
    fn from_arrays(
        data: PyReadonlyArray1<'_, f32>,
        indices: PyReadonlyArray1<'_, u32>,
        indptr: PyReadonlyArray1<'_, u32>,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<Self> {
        let inner = CsrMatrix::new(
            indptr.as_array().to_vec(),
            indices.as_array().to_vec(),
            data.as_array().to_vec(),
            n_rows,
            n_cols,
        )
        .map_err(matrix_err)?;
        Ok(Self { inner })
    }

    /// Build from a dense row-major `float32` array, dropping exact zeros.
    #[staticmethod]
    fn from_dense(array: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        let a = array.as_array();
        let (n_rows, n_cols) = (a.nrows(), a.ncols());
        let owned = a.as_standard_layout();
        let slice = owned
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("array is not contiguous"))?;
        Ok(Self {
            inner: CsrMatrix::from_dense_row_major(slice, n_rows, n_cols),
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    /// Element dtype (`numpy.float32`).
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        numpy::dtype::<f32>(py)
    }

    /// Copy of the stored values.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.data.clone().into_pyarray(py)
    }

    /// Copy of the column indices (`uint32`).
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.indices.clone().into_pyarray(py)
    }

    /// Copy of the row pointers (`uint32`).
    #[getter]
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.indptr.clone().into_pyarray(py)
    }

    /// `(data, indices, indptr)` copies — the input to `scipy.sparse.csr_matrix`.
    fn to_arrays<'py>(&self, py: Python<'py>) -> CsrArrays<'py> {
        (self.data(py), self.indices(py), self.indptr(py))
    }

    /// Densify into a `(n_rows × n_cols)` `float32` array.
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let d = self.inner.to_dense();
        crate::flat_to_pyarray2(py, d.data, d.nrows, d.ncols)
    }

    /// Alias of `toarray` (scipy spelling).
    fn todense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.toarray(py)
    }

    /// Deep copy.
    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Sum of all values (`axis=None`), per row (`axis=1`) or per column
    /// (`axis=0`). Accumulated in `float64`.
    #[pyo3(signature = (axis=None))]
    fn sum(&self, py: Python<'_>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
        match axis {
            None => {
                let total: f64 = self.inner.data.par_iter().map(|&v| f64::from(v)).sum();
                total.into_py_any(py)
            }
            Some(1 | -1) => row_sums_f64(&self.inner).into_pyarray(py).into_py_any(py),
            Some(0 | -2) => col_sums_f64(&self.inner).into_pyarray(py).into_py_any(py),
            Some(a) => Err(PyValueError::new_err(format!("axis {a} out of range"))),
        }
    }

    /// Mean of all values, per row or per column (zeros included).
    #[pyo3(signature = (axis=None))]
    fn mean(&self, py: Python<'_>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
        let (nrows, ncols) = self.inner.shape();
        match axis {
            None => {
                let total: f64 = self.inner.data.par_iter().map(|&v| f64::from(v)).sum();
                (total / (nrows * ncols).max(1) as f64).into_py_any(py)
            }
            Some(1 | -1) => {
                let n = ncols.max(1) as f64;
                let v: Vec<f64> = row_sums_f64(&self.inner)
                    .into_iter()
                    .map(|s| s / n)
                    .collect();
                v.into_pyarray(py).into_py_any(py)
            }
            Some(0 | -2) => {
                let n = nrows.max(1) as f64;
                let v: Vec<f64> = col_sums_f64(&self.inner)
                    .into_iter()
                    .map(|s| s / n)
                    .collect();
                v.into_pyarray(py).into_py_any(py)
            }
            Some(a) => Err(PyValueError::new_err(format!("axis {a} out of range"))),
        }
    }

    /// Indexing: `X[rows]`, `X[rows, cols]`, `X[:, cols]` with slices, boolean
    /// masks, integer arrays or single integers. Two arrays select the outer
    /// product rows × cols (AnnData semantics), not scipy's pointwise pairs.
    /// Always returns a new `SparseMatrix`; rows are gathered and columns kept
    /// in the given order.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (nrows, ncols) = self.inner.shape();
        let (row_key, col_key) = if let Ok(t) = key.cast::<PyTuple>() {
            match t.len() {
                1 => (t.get_item(0)?, None),
                2 => (t.get_item(0)?, Some(t.get_item(1)?)),
                n => {
                    return Err(PyIndexError::new_err(format!(
                        "too many indices for a 2-D matrix: {n}"
                    )))
                }
            }
        } else {
            (key.clone(), None)
        };
        let rows = parse_selection(&row_key, nrows)?;
        let cols = match col_key {
            Some(c) => parse_selection(&c, ncols)?,
            None => Selection::All,
        };

        // The selections are plain index vectors now, so the gathers run
        // with the GIL released — subsetting to HVGs is a real cost at
        // 100k cells (~100 ms), not a Python bookkeeping step.
        let inner = &self.inner;
        let mut out: Option<CsrMatrix> = match rows {
            Selection::All => None,
            Selection::Indices(r) => Some(py.detach(|| inner.select_rows(&r)).map_err(matrix_err)?),
        };
        if let Selection::Indices(c) = cols {
            let base = out.as_ref().unwrap_or(inner);
            out = Some(py.detach(|| base.select_cols(&c)).map_err(matrix_err)?);
        }
        Ok(Self {
            inner: out.unwrap_or_else(|| self.inner.clone()),
        })
    }

    fn __repr__(&self) -> String {
        let (n, m) = self.inner.shape();
        format!(
            "<SparseMatrix {n}x{m}, {} stored float32 (Rust-owned CSR)>",
            self.inner.nnz()
        )
    }

    /// Keep only the rows where `mask` is true, compacting the stored
    /// arrays in place (no full-matrix copy — peak stays at ~1x). Returns
    /// the number of rows kept.
    fn retain_rows_mask(
        &mut self,
        py: Python<'_>,
        mask: PyReadonlyArray1<'_, bool>,
    ) -> PyResult<usize> {
        let mask: Vec<bool> = mask.as_array().to_vec();
        let inner = &mut self.inner;
        py.detach(|| inner.retain_rows_mask(&mask))
            .map_err(matrix_err)
    }

    // ── preprocessing kernels ────────────────────────────────────────────

    /// Per-cell and per-gene QC in one pass. `mito` is an optional per-gene
    /// mitochondrial mask. Returns a dict of numpy arrays.
    #[pyo3(signature = (mito=None))]
    fn qc_metrics<'py>(
        &self,
        py: Python<'py>,
        mito: Option<PyReadonlyArray1<'py, bool>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mask: Option<Vec<bool>> = mito.map(|m| m.as_array().to_vec());
        if let Some(m) = &mask {
            if m.len() != self.inner.ncols {
                return Err(PyValueError::new_err(format!(
                    "mito mask has length {}, expected {}",
                    m.len(),
                    self.inner.ncols
                )));
            }
        }
        let inner = &self.inner;
        let qc = py.detach(|| preprocess::qc_metrics(inner, mask.as_deref()));
        let d = PyDict::new(py);
        d.set_item("n_counts", qc.n_counts.into_pyarray(py))?;
        d.set_item("n_genes", qc.n_genes.into_pyarray(py))?;
        match qc.mt_counts {
            Some(mt) => d.set_item("mt_counts", mt.into_pyarray(py))?,
            None => d.set_item("mt_counts", py.None())?,
        }
        d.set_item("gene_n_cells", qc.gene_n_cells.into_pyarray(py))?;
        d.set_item("gene_total_counts", qc.gene_total_counts.into_pyarray(py))?;
        Ok(d)
    }

    /// Scale each cell to `target_sum` total counts, in place. Returns the
    /// per-cell totals before scaling.
    fn normalize_total<'py>(
        &mut self,
        py: Python<'py>,
        target_sum: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let inner = &mut self.inner;
        let totals = py.detach(|| preprocess::normalize_total_inplace(inner, target_sum));
        totals.into_pyarray(py)
    }

    /// Normalize each cell again over the current genes, in place, from
    /// values normalized over all genes (see `preprocess::renormalize_inplace`).
    fn renormalize_inplace(
        &mut self,
        py: Python<'_>,
        back: PyReadonlyArray1<'_, f32>,
        from_log: bool,
        target_sum: f32,
        log: bool,
    ) -> PyResult<()> {
        let back = back.as_slice()?.to_vec();
        if back.len() != self.inner.nrows {
            return Err(PyValueError::new_err(format!(
                "{} back-scales for {} cells",
                back.len(),
                self.inner.nrows
            )));
        }
        let inner = &mut self.inner;
        py.detach(|| preprocess::renormalize_inplace(inner, &back, from_log, target_sum, log));
        Ok(())
    }

    /// `ln(1 + x)` on every stored value, in place.
    fn log1p(&mut self, py: Python<'_>) {
        let inner = &mut self.inner;
        py.detach(|| preprocess::log1p_inplace(inner));
    }

    /// Seurat-flavour highly-variable genes (scanpy-compatible). With
    /// `expm1=True` (the default, for log1p data) statistics are computed on
    /// `expm1(X)`. `flavor="scarf"` selects them as Scarf does instead (see
    /// `preprocess::hvg_scarf`; `n_bins` then defaults to 200 and genes must be
    /// detected in more than `min_cells` cells).
    #[pyo3(signature = (n_top_genes, n_bins=20, expm1=true, flavor="seurat", min_cells=0, lowess_frac=0.1))]
    #[allow(clippy::too_many_arguments)]
    fn highly_variable_genes<'py>(
        &self,
        py: Python<'py>,
        n_top_genes: usize,
        n_bins: usize,
        expm1: bool,
        flavor: &str,
        min_cells: u64,
        lowess_frac: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner;
        if flavor == "scarf" {
            let r = py.detach(|| {
                let (stats, det) = preprocess::gene_stats_detected_blocks(inner, expm1);
                preprocess::hvg_scarf(&stats, &det, n_top_genes, n_bins, lowess_frac, min_cells)
            });
            return crate::hvg_scarf_dict(py, r);
        }
        if flavor != "seurat" {
            return Err(PyValueError::new_err(format!("unknown flavor {flavor:?}")));
        }
        if min_cells > 0 {
            let hvg = py.detach(|| {
                let (stats, det) = preprocess::gene_stats_detected_blocks(inner, expm1);
                preprocess::hvg_seurat_floor(&stats, &det, n_top_genes, n_bins, min_cells)
            });
            return crate::hvg_seurat_dict(py, hvg);
        }
        let hvg = py.detach(|| {
            let stats = preprocess::gene_mean_var(inner, expm1);
            preprocess::hvg_seurat(&stats, n_top_genes, n_bins)
        });
        let d = PyDict::new(py);
        d.set_item("mask", hvg.mask.into_pyarray(py))?;
        d.set_item("means", hvg.means.into_pyarray(py))?;
        d.set_item("dispersions", hvg.dispersions.into_pyarray(py))?;
        d.set_item("dispersions_norm", hvg.dispersions_norm.into_pyarray(py))?;
        d.set_item("n_hvg", hvg.n_hvg)?;
        Ok(d)
    }
}
