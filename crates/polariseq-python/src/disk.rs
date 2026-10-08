//! `DiskMatrix`: the expression matrix of an out-of-core analysis.
//!
//! The counts live in a block store on the SSD, in the project folder the
//! user declared; nothing larger than a block is ever resident. The
//! preprocessing steps that rewrite the matrix in memory are recorded here
//! instead, as views applied while a block is read: a cell filter
//! (`FilteredRows`), a gene filter (`FilteredCols`), and `normalize_total` +
//! `log1p` (`NormalizedView`). The steps that read the matrix (quality control,
//! gene statistics, PCA, marker statistics) stream it through those views with
//! the same kernels as the in-memory mode and as `process_diskbacked`.
//!
//! Every mask and scale is indexed by the rows and columns of the file, so the
//! views compose in one fixed order whatever order the steps ran in: genes
//! innermost, then the normalisation (whose scale is per file row), then the
//! cells.
//!
//! The method names match `SparseMatrix` where the operation is the same
//! (`qc_metrics`, `normalize_total`, `log1p`, `highly_variable_genes`,
//! `retain_rows_mask`), so the Python steps work on either with one branch.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use polariseq_core::file_matrix::FileCsr;
use polariseq_core::preprocess::{
    cluster_stats_blocks, gene_mean_var_blocks, hvg_seurat, qc_metrics_blocks, row_totals_blocks,
    FilteredCols, FilteredRows, NormalizedView,
};
use polariseq_core::rowblocks::RowBlocks;
use polariseq_core::{pca::pca_blocks, PcaOptions};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use crate::{de_result_dict, flat_to_pyarray2};

#[pyclass(name = "DiskMatrix", module = "polariseq._polariseq")]
pub struct PyDiskMatrix {
    /// Shared by every subset of the same store, so its row pointers are
    /// held once however many subsets there are.
    csr: Arc<FileCsr>,
    prefix: PathBuf,
    block_rows: usize,
    /// Kept cells, one flag per file row.
    row_keep: Vec<bool>,
    /// Kept genes, one flag per file column.
    col_keep: Vec<bool>,
    n_rows_kept: usize,
    n_cols_kept: usize,
    /// `normalize_total` scale, one factor per file row.
    scale: Option<Vec<f32>>,
    log1p: bool,
}

fn io_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

impl PyDiskMatrix {
    fn from_csr(csr: FileCsr, prefix: PathBuf, block_rows: usize) -> Self {
        let (n, g) = (csr.nrows, csr.ncols);
        Self {
            csr: Arc::new(csr),
            prefix,
            block_rows,
            row_keep: vec![true; n],
            col_keep: vec![true; g],
            n_rows_kept: n,
            n_cols_kept: g,
            scale: None,
            log1p: false,
        }
    }

    /// Run `f` on the current matrix: the file's counts with the recorded
    /// gene filter, normalisation and (when `rows`) cell filter applied.
    fn with_view<R>(&self, src: &FileCsr, rows: bool, f: impl FnOnce(&dyn RowBlocks) -> R) -> R {
        let cols = (src.ncols == self.csr.ncols && self.n_cols_kept < self.csr.ncols)
            .then(|| FilteredCols::new(src, &self.col_keep));
        let base: &dyn RowBlocks = match &cols {
            Some(c) => c,
            None => src,
        };
        let norm = (self.scale.is_some() || self.log1p).then(|| {
            let scale = self
                .scale
                .clone()
                .unwrap_or_else(|| vec![1.0; self.csr.nrows]);
            NormalizedView::new(base, scale, self.log1p)
        });
        let mid: &dyn RowBlocks = match &norm {
            Some(n) => n,
            None => base,
        };
        let filt = (rows && self.n_rows_kept < self.csr.nrows)
            .then(|| FilteredRows::new(mid, &self.row_keep));
        let top: &dyn RowBlocks = match &filt {
            Some(r) => r,
            None => mid,
        };
        f(top)
    }

    /// Run `f` on the current matrix: the file's counts with every recorded
    /// filter and the normalization applied.
    pub(crate) fn current_view<R>(&self, f: impl FnOnce(&dyn RowBlocks) -> R) -> R {
        self.with_view(&self.csr, true, f)
    }

    /// Run `f` on the raw counts of the current cells and genes: the recorded
    /// filters apply, the normalization does not.
    pub(crate) fn raw_view<R>(&self, f: impl FnOnce(&dyn RowBlocks) -> R) -> R {
        let cols = (self.n_cols_kept < self.csr.ncols)
            .then(|| FilteredCols::new(&*self.csr, &self.col_keep));
        let base: &dyn RowBlocks = match &cols {
            Some(c) => c,
            None => &*self.csr,
        };
        let rows =
            (self.n_rows_kept < self.csr.nrows).then(|| FilteredRows::new(base, &self.row_keep));
        let top: &dyn RowBlocks = match &rows {
            Some(r) => r,
            None => base,
        };
        f(top)
    }

    /// Values of the given current columns for every current cell, row-major.
    fn columns_flat(&self, py: Python<'_>, columns: &[usize]) -> PyResult<Vec<f32>> {
        let (n, k) = (self.n_rows_kept, columns.len());
        let mut pos = vec![usize::MAX; self.n_cols_kept];
        for (p, &c) in columns.iter().enumerate() {
            if c >= self.n_cols_kept {
                return Err(PyValueError::new_err(format!("column {c} out of range")));
            }
            pos[c] = p;
        }
        let out = std::sync::Mutex::new(vec![0.0_f32; n * k]);
        py.detach(|| {
            self.with_view(&self.csr, true, |v| {
                v.for_each_block(self.block_rows, &|off, blk| {
                    let mut local: Vec<(usize, f32)> = Vec::new();
                    for i in 0..blk.n_rows {
                        let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                        for (&c, &val) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                            let p = pos[c as usize];
                            if p != usize::MAX {
                                local.push(((off + i) * k + p, val));
                            }
                        }
                    }
                    if let Ok(mut o) = out.lock() {
                        for (idx, val) in local {
                            o[idx] = val;
                        }
                    }
                });
            });
        });
        out.into_inner().map_err(io_err)
    }

    /// Narrow a file-indexed keep mask by a mask over its kept entries.
    fn narrow(keep: &mut [bool], mask: &[bool]) -> usize {
        let mut it = mask.iter();
        let mut n = 0;
        for k in keep.iter_mut() {
            if *k {
                *k = *it.next().unwrap_or(&false);
                n += usize::from(*k);
            }
        }
        n
    }
}

#[pymethods]
impl PyDiskMatrix {
    /// Copy the counts of an `.h5ad` file into a block store at `prefix`
    /// (`.data`, `.indices`, `.indptr`), streaming, and open it.
    #[staticmethod]
    fn from_h5ad(py: Python<'_>, path: &str, prefix: &str, block_rows: usize) -> PyResult<Self> {
        let pfx = PathBuf::from(prefix);
        let csr = py
            .detach(|| {
                let s = polariseq_io::spill::spill_h5ad_with(path, &pfx, block_rows)?;
                s.open_with(block_rows)
            })
            .map_err(io_err)?;
        Ok(Self::from_csr(csr, pfx, block_rows))
    }

    /// Copy the counts of several `.h5ad` files into one block store: their
    /// cells one after another, over one set of genes (`gene_maps[k][j]`: the
    /// merged column of file `k`'s gene `j`, or `2**32 - 1` to drop it).
    #[staticmethod]
    fn from_h5ad_many(
        py: Python<'_>,
        paths: Vec<String>,
        gene_maps: Vec<PyReadonlyArray1<'_, u32>>,
        n_cols: usize,
        prefix: &str,
        block_rows: usize,
    ) -> PyResult<Self> {
        let maps: Vec<Vec<u32>> = gene_maps
            .iter()
            .map(|m| m.as_slice().map(<[u32]>::to_vec))
            .collect::<Result<_, _>>()?;
        let pfx = PathBuf::from(prefix);
        let csr = py
            .detach(|| {
                let s =
                    polariseq_io::spill::spill_h5ad_many(&paths, &maps, n_cols, &pfx, block_rows)?;
                s.open_with(block_rows)
            })
            .map_err(io_err)?;
        Ok(Self::from_csr(csr, pfx, block_rows))
    }

    /// Open a block store written earlier (reused across sessions).
    #[staticmethod]
    fn open(prefix: &str, n_rows: usize, n_cols: usize, block_rows: usize) -> PyResult<Self> {
        let pfx = PathBuf::from(prefix);
        let csr = FileCsr::open_at(&pfx, n_rows, n_cols)
            .map_err(io_err)?
            .with_max_block_rows(block_rows);
        Ok(Self::from_csr(csr, pfx, block_rows))
    }

    /// A new matrix over the same counts, keeping only the current rows and
    /// columns where `rows` and `cols` are true (`None`: all). Nothing is
    /// copied: the store and its row pointers are shared, and the filters and
    /// normalization recorded so far carry over; later steps on either matrix
    /// do not affect the other.
    #[pyo3(signature = (rows=None, cols=None))]
    fn subset(
        &self,
        rows: Option<PyReadonlyArray1<'_, bool>>,
        cols: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Self> {
        let mut new = Self {
            csr: Arc::clone(&self.csr),
            prefix: self.prefix.clone(),
            block_rows: self.block_rows,
            row_keep: self.row_keep.clone(),
            col_keep: self.col_keep.clone(),
            n_rows_kept: self.n_rows_kept,
            n_cols_kept: self.n_cols_kept,
            scale: self.scale.clone(),
            log1p: self.log1p,
        };
        if let Some(m) = rows {
            let m = m.as_slice()?;
            if m.len() != self.n_rows_kept {
                return Err(PyValueError::new_err(format!(
                    "row mask has {} entries for {} rows",
                    m.len(),
                    self.n_rows_kept
                )));
            }
            new.n_rows_kept = Self::narrow(&mut new.row_keep, m);
        }
        if let Some(m) = cols {
            let m = m.as_slice()?;
            if m.len() != self.n_cols_kept {
                return Err(PyValueError::new_err(format!(
                    "column mask has {} entries for {} columns",
                    m.len(),
                    self.n_cols_kept
                )));
            }
            new.n_cols_kept = Self::narrow(&mut new.col_keep, m);
        }
        Ok(new)
    }

    /// `(cells, genes)` of the current matrix.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.n_rows_kept, self.n_cols_kept)
    }

    /// `(cells, genes)` of the file.
    #[getter]
    fn file_shape(&self) -> (usize, usize) {
        (self.csr.nrows, self.csr.ncols)
    }

    /// Stored values in the file's counts.
    #[getter]
    fn nnz(&self) -> usize {
        self.csr.nnz
    }

    /// Where the counts are stored.
    #[getter]
    fn prefix(&self) -> String {
        self.prefix.to_string_lossy().into_owned()
    }

    /// Rows per block of every pass.
    #[getter]
    fn block_rows(&self) -> usize {
        self.block_rows
    }

    /// Which file rows are kept.
    fn row_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.row_keep.clone().into_pyarray(py)
    }

    /// Which file columns are kept.
    fn col_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.col_keep.clone().into_pyarray(py)
    }

    /// The recorded transforms: whether cells are scaled, whether log1p.
    fn transforms(&self) -> (bool, bool) {
        (self.scale.is_some(), self.log1p)
    }

    /// Keep only the current rows where `mask` is true (a view; the store is
    /// not rewritten). Returns the number of rows kept.
    fn retain_rows_mask(&mut self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<usize> {
        let m = mask.as_slice()?;
        if m.len() != self.n_rows_kept {
            return Err(PyValueError::new_err(format!(
                "mask has {} entries for {} rows",
                m.len(),
                self.n_rows_kept
            )));
        }
        self.n_rows_kept = Self::narrow(&mut self.row_keep, m);
        Ok(self.n_rows_kept)
    }

    /// Keep only the current columns where `mask` is true (a view).
    fn retain_cols_mask(&mut self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<usize> {
        let m = mask.as_slice()?;
        if m.len() != self.n_cols_kept {
            return Err(PyValueError::new_err(format!(
                "mask has {} entries for {} columns",
                m.len(),
                self.n_cols_kept
            )));
        }
        self.n_cols_kept = Self::narrow(&mut self.col_keep, m);
        Ok(self.n_cols_kept)
    }

    /// Per-cell and per-gene QC of the current matrix, as
    /// `SparseMatrix.qc_metrics`.
    #[pyo3(signature = (mito=None))]
    fn qc_metrics<'py>(
        &self,
        py: Python<'py>,
        mito: Option<PyReadonlyArray1<'py, bool>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mask: Option<Vec<bool>> = mito.map(|m| m.as_array().to_vec());
        if let Some(m) = &mask {
            if m.len() != self.n_cols_kept {
                return Err(PyValueError::new_err(format!(
                    "mito mask has length {}, expected {}",
                    m.len(),
                    self.n_cols_kept
                )));
            }
        }
        let qc = py
            .detach(|| self.with_view(&self.csr, true, |v| qc_metrics_blocks(v, mask.as_deref())));
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

    /// Scale each cell to `target_sum`, as a view. The totals are those of the
    /// current genes, summed as `SparseMatrix.normalize_total` sums them.
    /// Returns the totals of the current cells.
    fn normalize_total<'py>(
        &mut self,
        py: Python<'py>,
        target_sum: f32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if self.log1p {
            return Err(PyValueError::new_err(
                "normalize_total after log1p is not supported out of core",
            ));
        }
        let totals = py.detach(|| self.with_view(&self.csr, false, |v| row_totals_blocks(v, None)));
        let step = NormalizedView::scale_from_totals(&totals, target_sum);
        self.scale = Some(match self.scale.take() {
            Some(old) => old.iter().zip(&step).map(|(a, b)| a * b).collect(),
            None => step,
        });
        let kept: Vec<f32> = totals
            .iter()
            .zip(&self.row_keep)
            .filter(|(_, &k)| k)
            .map(|(&t, _)| t)
            .collect();
        Ok(kept.into_pyarray(py))
    }

    /// Scale each cell to `target_sum` over the current genes, reading their
    /// counts from the copy of the current genes at `subset_prefix` (made here
    /// when absent, and the one `pca` then reuses), not from the whole store:
    /// the totals are the same, and the copy holds a fraction of the values.
    /// Returns the totals of the current cells.
    fn normalize_total_subset<'py>(
        &mut self,
        py: Python<'py>,
        target_sum: f32,
        subset_prefix: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if self.log1p || self.scale.is_some() {
            return Err(PyValueError::new_err(
                "normalize_total_subset needs the raw counts: call reset_normalization first",
            ));
        }
        let pfx = PathBuf::from(subset_prefix);
        let n_cols = self.n_cols_kept;
        let totals = py
            .detach(|| -> Result<Vec<f32>, String> {
                let sub = if pfx.with_extension("indptr").exists() {
                    FileCsr::open_at(&pfx, self.csr.nrows, n_cols)
                        .map(|f| f.with_max_block_rows(self.block_rows))
                } else {
                    self.csr.select_columns(&self.col_keep, &pfx, self.block_rows)
                }
                .map_err(|e| e.to_string())?;
                Ok(row_totals_blocks(&sub, None))
            })
            .map_err(PyRuntimeError::new_err)?;
        self.scale = Some(NormalizedView::scale_from_totals(&totals, target_sum));
        let kept: Vec<f32> = totals
            .iter()
            .zip(&self.row_keep)
            .filter(|(_, &k)| k)
            .map(|(&t, _)| t)
            .collect();
        Ok(kept.into_pyarray(py))
    }

    /// Drop the normalization view (scaling and log), so that the current
    /// genes' raw counts can be normalized again, as Scarf normalizes over the
    /// selected genes only.
    fn reset_normalization(&mut self) {
        self.scale = None;
        self.log1p = false;
    }

    /// `ln(1 + x)`, as a view.
    fn log1p(&mut self) -> PyResult<()> {
        if self.log1p {
            return Err(PyValueError::new_err("log1p has already been applied"));
        }
        self.log1p = true;
        Ok(())
    }

    /// Seurat-flavour highly-variable genes of the current matrix, as
    /// `SparseMatrix.highly_variable_genes`.
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
        if flavor == "scarf" {
            let r = py.detach(|| {
                self.with_view(&self.csr, true, |v| {
                    let (stats, det) = polariseq_core::preprocess::gene_stats_detected_blocks(v, expm1);
                    polariseq_core::preprocess::hvg_scarf(&stats, &det, n_top_genes, n_bins, lowess_frac, min_cells)
                })
            });
            return crate::hvg_scarf_dict(py, r);
        }
        if flavor != "seurat" {
            return Err(PyValueError::new_err(format!("unknown flavor {flavor:?}")));
        }
        if min_cells > 0 {
            let hvg = py.detach(|| {
                self.with_view(&self.csr, true, |v| {
                    let (stats, det) = polariseq_core::preprocess::gene_stats_detected_blocks(v, expm1);
                    polariseq_core::preprocess::hvg_seurat_floor(&stats, &det, n_top_genes, n_bins, min_cells)
                })
            });
            return crate::hvg_seurat_dict(py, hvg);
        }
        let hvg = py.detach(|| {
            self.with_view(&self.csr, true, |v| {
                hvg_seurat(&gene_mean_var_blocks(v, expm1), n_top_genes, n_bins)
            })
        });
        let d = PyDict::new(py);
        d.set_item("mask", hvg.mask.into_pyarray(py))?;
        d.set_item("means", hvg.means.into_pyarray(py))?;
        d.set_item("dispersions", hvg.dispersions.into_pyarray(py))?;
        d.set_item("dispersions_norm", hvg.dispersions_norm.into_pyarray(py))?;
        d.set_item("n_hvg", hvg.n_hvg)?;
        Ok(d)
    }

    /// PCA of the current matrix. The counts of the current genes are first
    /// copied to `subset_prefix` (reused when that copy exists), the gene
    /// subset every later pass then reads, as `process_diskbacked` does.
    /// With `out` (`cells × n_components` float32, such as a memory-mapped
    /// file) each block's rows are projected straight into it and the dict's
    /// `embedding` is `None`.
    #[pyo3(signature = (n_components, subset_prefix, center=true, oversample=10, n_iter=4, seed=0, out=None, scale=false, max_value=None))]
    #[allow(clippy::too_many_arguments)]
    fn pca<'py>(
        &self,
        py: Python<'py>,
        n_components: usize,
        subset_prefix: &str,
        center: bool,
        oversample: usize,
        n_iter: usize,
        seed: u64,
        out: Option<numpy::PyReadwriteArray2<'py, f32>>,
        scale: bool,
        max_value: Option<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pfx = PathBuf::from(subset_prefix);
        let n_cols = self.n_cols_kept;
        let mut out = out;
        let buf: Option<&mut [f32]> = match out.as_mut() {
            Some(o) => Some(o.as_slice_mut()?),
            None => None,
        };
        let result = py
            .detach(|| -> Result<_, String> {
                let sub = if pfx.with_extension("indptr").exists() {
                    FileCsr::open_at(&pfx, self.csr.nrows, n_cols)
                        .map(|f| f.with_max_block_rows(self.block_rows))
                } else {
                    self.csr
                        .select_columns(&self.col_keep, &pfx, self.block_rows)
                }
                .map_err(|e| e.to_string())?;
                let opts = PcaOptions {
                    n_components,
                    center,
                    oversample,
                    n_iter,
                    seed,
                };
                // with `scale`, each gene is standardized first (scanpy's
                // pp.scale), over the same view, without forming the scaled matrix
                match (buf, scale) {
                    (Some(buf), false) => self
                        .with_view(&sub, true, |v| {
                            polariseq_core::pca::pca_blocks_into(v, &opts, buf)
                        })
                        .map(|s| (None, s)),
                    (Some(buf), true) => self
                        .with_view(&sub, true, |v| {
                            polariseq_core::pca::pca_scaled_blocks_into(v, &opts, max_value, buf)
                        })
                        .map(|s| (None, s)),
                    (None, false) => self
                        .with_view(&sub, true, |v| pca_blocks(v, &opts))
                        .map(|r| {
                            let (e, s) = r.into_parts();
                            (Some(e), s)
                        }),
                    (None, true) => self
                        .with_view(&sub, true, |v| {
                            polariseq_core::pca::pca_scaled_blocks(v, &opts, max_value)
                        })
                        .map(|r| {
                            let (e, s) = r.into_parts();
                            (Some(e), s)
                        }),
                }
                .map_err(|e| format!("PCA failed: {e}"))
            })
            .map_err(PyRuntimeError::new_err)?;
        let (embedding, summary) = result;
        let d = PyDict::new(py);
        let n_obs = self.n_rows_kept;
        let k = summary.n_components;
        match embedding {
            Some(e) => d.set_item("embedding", flat_to_pyarray2(py, e, n_obs, k)?)?,
            None => d.set_item("embedding", py.None())?,
        }
        d.set_item("variance_ratio", summary.variance_ratio.into_pyarray(py))?;
        d.set_item("n_components", k)?;
        Ok(d)
    }

    /// Marker ranking of the current matrix by groups of the current cells,
    /// with the Welch test of the in-memory `rank_genes_groups`.
    fn rank_genes_groups<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<'py, u32>,
        n_groups: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let labels = labels.as_slice()?.to_vec();
        if labels.len() != self.n_rows_kept {
            return Err(PyValueError::new_err(format!(
                "{} labels for {} cells",
                labels.len(),
                self.n_rows_kept
            )));
        }
        let result = py
            .detach(|| {
                let stats = self.with_view(&self.csr, true, |v| {
                    cluster_stats_blocks(v, &labels, n_groups)
                });
                polariseq_core::de::rank_genes_groups_from_stats(&stats, None)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("DE failed: {e}")))?;
        de_result_dict(py, &result)
    }

    /// Values of the current matrix in the given current columns, for every
    /// current cell (`cells × columns`), in one streamed pass. For plots.
    fn column_values<'py>(
        &self,
        py: Python<'py>,
        columns: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let flat = self.columns_flat(py, &columns)?;
        flat_to_pyarray2(py, flat, self.n_rows_kept, columns.len())
    }

    /// `X[:, gene]` (one column, 1-D) or `X[:, [genes]]` (2-D): what plots
    /// ask a matrix for. An out-of-core matrix is read by columns only.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let err = || {
            PyTypeError::new_err(
                "an out-of-core matrix is read by columns: X[:, gene] or X[:, [genes]]",
            )
        };
        let key = key.cast::<PyTuple>().map_err(|_| err())?;
        if key.len() != 2 {
            return Err(err());
        }
        let rows = key.get_item(0)?;
        let rows = rows.cast::<PySlice>().map_err(|_| err())?;
        for part in ["start", "stop", "step"] {
            if !rows.getattr(part)?.is_none() {
                return Err(err());
            }
        }
        let cols = key.get_item(1)?;
        if let Ok(c) = cols.extract::<usize>() {
            let flat = self.columns_flat(py, &[c])?;
            return Ok(flat.into_pyarray(py).into_any());
        }
        let cs: Vec<usize> = cols.extract().map_err(|_| err())?;
        let flat = self.columns_flat(py, &cs)?;
        Ok(flat_to_pyarray2(py, flat, self.n_rows_kept, cs.len())?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "DiskMatrix({} x {} of {} x {}, {} stored values, at {})",
            self.n_rows_kept,
            self.n_cols_kept,
            self.csr.nrows,
            self.csr.ncols,
            self.csr.nnz,
            self.prefix.display()
        )
    }
}
