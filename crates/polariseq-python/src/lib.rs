//! # polariseq-python
//!
//! PyO3 bindings exposing the Rust core to Python. This crate compiles to a
//! `cdylib` named `_polariseq` that `maturin` packages; the pure-Python
//! `polariseq` package re-exports from here and adds the ergonomic API layer.
//!
//! The binding layer stays thin: Rust functions, Python signatures, zero-copy
//! buffer passing via the `numpy` crate.
//!
//! **The GIL is released around every multi-millisecond kernel.** Each hot
//! function copies its Python inputs into owned Rust first, then runs the
//! compute inside `py.detach(...)`, then touches Python again only to build
//! the result. Two reasons, the second the more important:
//!
//! 1. A multi-second Rust call that holds the GIL blocks every other Python
//!    thread in the user's process — their progress bar, their notebook's
//!    interrupt handler, their web server's request loop. Nobody asked for
//!    that.
//! 2. The telemetry harness samples memory from a Python thread inside the
//!    measured process; with the GIL held, that thread cannot run, so the
//!    sampler was blind for 95 % of a Polariseq run (chapter 09). A blind
//!    sampler draws a flat line through the expensive step, which is exactly
//!    the curve we would like to see, so it cannot be allowed to.
//!
//! The compiler enforces the discipline: a closure passed to `detach` may not
//! capture a `Bound`, a `PyRef` or a `PyReadonlyArray`, so anything that
//! needs Python is forced above or below the call.

// PyO3 #[pyfunction] signatures naturally have many args (each maps to a Python
// kwarg); the pedantic arg-count lint doesn't apply here.
#![allow(
    clippy::too_many_arguments,
    clippy::missing_errors_doc,
    clippy::must_use_candidate
)]

mod compare;
mod disk;
mod doublets;
mod frame;
mod sparse;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyReadwriteArray2, ToPyArray,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sparse::PySparseMatrix;

/// Convert a flat row-major `Vec<f32>` into a `(nrows × ncols)` `PyArray2`.
pub(crate) fn flat_to_pyarray2<'py>(
    py: Python<'py>,
    flat: Vec<f32>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr = numpy::ndarray::Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| PyRuntimeError::new_err(format!("shape error: {e}")))?;
    Ok(arr.to_pyarray(py))
}

/// A PCA result dict: the embedding (or `None` when it was written into the
/// caller's array), the variance ratio and the component count.
fn pca_dict<'py>(
    py: Python<'py>,
    embedding: Option<Vec<f32>>,
    summary: polariseq_core::pca::PcaSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match embedding {
        Some(e) => dict.set_item(
            "embedding",
            flat_to_pyarray2(py, e, summary.n_obs, summary.n_components)?,
        )?,
        None => dict.set_item("embedding", py.None())?,
    }
    dict.set_item("variance_ratio", summary.variance_ratio.into_pyarray(py))?;
    dict.set_item("n_components", summary.n_components)?;
    Ok(dict)
}

/// Triple every element of a 1-D float64 numpy array (Phase 0 round-trip canary).
#[pyfunction]
fn triple<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = input.as_slice()?;
    let mut out: Vec<f64> = slice.to_vec();
    polariseq_core::double_in_place(&mut out);
    for x in &mut out {
        *x *= 1.5;
    }
    Ok(out.into_pyarray(py))
}

/// Size the global worker pool.
///
/// Called once from Python when the budget is first resolved. Rayon's global
/// pool can only be built once per process, so a second call with a different
/// count is reported rather than silently ignored — otherwise a user who set
/// a thread count would have no way to know it did not take effect.
///
/// Returns the number of threads now in use.
#[pyfunction]
fn set_threads(n: usize) -> PyResult<usize> {
    let n = n.max(1);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => Ok(n),
        Err(_) => {
            let current = rayon::current_num_threads();
            if current == n {
                Ok(current)
            } else {
                Err(PyRuntimeError::new_err(format!(
                    "the worker pool is already running with {current} threads; \
                     set the thread count before the first parallel call"
                )))
            }
        }
    }
}

/// Threads currently in the global worker pool.
#[pyfunction]
fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// Report the Rust core version.
#[pyfunction]
fn core_version() -> &'static str {
    polariseq_core::VERSION
}

/// Read the X matrix from a `.h5ad` file into a dense numpy array.
///
/// Returns `(data, n_obs, n_vars)` where data is a row-major `(n_obs × n_vars)`
/// float32 numpy array. Densifies sparse data.
#[pyfunction]
fn read_h5ad_x<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize, usize)> {
    let file = polariseq_io::AnnDataFile::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("h5ad read failed: {e}")))?;
    let matrix = file
        .read_x()
        .map_err(|e| PyRuntimeError::new_err(format!("read_x failed: {e}")))?;
    let dense = polariseq_core::Matrix::to_dense(&matrix);
    let (n_obs, n_vars) = dense.shape();
    let data2d = flat_to_pyarray2(py, dense.data, n_obs, n_vars)?;
    Ok((data2d, n_obs, n_vars))
}

/// Scan .h5ad metadata WITHOUT loading the data matrix. Returns shape,
/// obs_names, var_names, encoding, and estimated memory. Fast (<100ms even
/// for huge files) because it only reads HDF5 attributes and small datasets.
#[pyfunction]
fn scan_h5ad<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let file = polariseq_io::AnnDataFile::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot open: {e}")))?;

    // Get X info (shape + encoding) without materializing.
    let x_info = file
        .x_info()
        .map_err(|e| PyRuntimeError::new_err(format!("cannot scan X: {e}")))?;

    // Read obs/var names (small string arrays — fast).
    let obs_names = file.read_obs_names().unwrap_or_default();
    let var_names = file.read_var_names().unwrap_or_default();

    let encoding_str = match x_info.encoding {
        polariseq_io::h5ad::XEncoding::Dense => "dense",
        polariseq_io::h5ad::XEncoding::CsrMatrix => "csr",
        polariseq_io::h5ad::XEncoding::CscMatrix => "csc",
    };

    // Estimate memory if materialized: nnz × 8 bytes (data f32 + indices u32).
    let est_memory_gb = (x_info.n_elements * 8) as f64 / 1e9;

    let dict = PyDict::new(py);
    dict.set_item("n_obs", x_info.n_obs)?;
    dict.set_item("n_vars", x_info.n_vars)?;
    dict.set_item("n_elements", x_info.n_elements)?;
    dict.set_item("encoding", encoding_str)?;
    dict.set_item("est_memory_gb", est_memory_gb)?;
    dict.set_item("obs_names", obs_names)?;
    dict.set_item("var_names", var_names)?;
    Ok(dict)
}

/// Scan 10x .h5 metadata without loading the matrix.
#[pyfunction]
fn scan_10x_h5<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    use hdf5_metno::File;

    let file =
        File::open(path).map_err(|e| PyRuntimeError::new_err(format!("cannot open: {e}")))?;
    let matrix = file
        .group("matrix")
        .map_err(|e| PyRuntimeError::new_err(format!("no /matrix: {e}")))?;

    // Read shape (tiny dataset — instant).
    let shape_ds = matrix
        .dataset("shape")
        .map_err(|e| PyRuntimeError::new_err(format!("no shape: {e}")))?;
    let shape: Vec<u64> = shape_ds
        .read_raw::<u32>()
        .map(|v| v.into_iter().map(u64::from).collect())
        .or_else(|_| shape_ds.read_raw::<u64>())
        .map_err(|e| PyRuntimeError::new_err(format!("cannot read shape: {e}")))?;

    let n_genes = shape.first().copied().unwrap_or(0) as usize;
    let n_cells = shape.get(1).copied().unwrap_or(0) as usize;

    // Read nnz from data dataset shape (metadata only, no data).
    let nnz = matrix
        .dataset("data")
        .map(|d| d.shape().first().copied().unwrap_or(0))
        .unwrap_or(0);

    // Read barcodes and gene names.
    let features = matrix.group("features").ok();
    let gene_names: Vec<String> = features
        .and_then(|f| f.dataset("name").ok())
        .and_then(|d| {
            d.read_raw::<hdf5_metno::types::VarLenAscii>()
                .ok()
                .map(|v| v.into_iter().map(|s| s.to_string()).collect())
        })
        .unwrap_or_default();
    let barcodes: Vec<String> = matrix
        .dataset("barcodes")
        .ok()
        .and_then(|d| {
            d.read_raw::<hdf5_metno::types::VarLenAscii>()
                .ok()
                .map(|v| v.into_iter().map(|s| s.to_string()).collect())
        })
        .unwrap_or_default();

    let dict = PyDict::new(py);
    dict.set_item("n_cells", n_cells)?;
    dict.set_item("n_genes", n_genes)?;
    dict.set_item("nnz", nnz)?;
    dict.set_item("est_memory_gb", (nnz * 8) as f64 / 1e9)?;
    dict.set_item("barcodes", barcodes)?;
    dict.set_item("gene_names", gene_names)?;
    Ok(dict)
}

/// Read a sparse X matrix from a `.h5ad` file into a Rust-owned
/// [`PySparseMatrix`] — no densification, no copy into Python.
#[pyfunction]
fn read_h5ad_csr(py: Python<'_>, path: &str) -> PyResult<PySparseMatrix> {
    let matrix = py
        .detach(|| {
            let file = polariseq_io::AnnDataFile::open(path)
                .map_err(|e| format!("h5ad read failed: {e}"))?;
            file.read_x().map_err(|e| format!("read_x failed: {e}"))
        })
        .map_err(PyRuntimeError::new_err)?;
    match matrix {
        polariseq_core::Matrix::Csr(csr) => Ok(PySparseMatrix::new(csr)),
        polariseq_core::Matrix::Dense(_) => Err(PyRuntimeError::new_err(
            "X is dense, not sparse; use read_h5ad_x",
        )),
    }
}

/// Read several `.h5ad` files as one matrix in memory: their cells one after
/// another, over one set of genes (`gene_maps[k][j]`: the merged column of
/// file `k`'s gene `j`, or `2**32 - 1` to drop it). Each row is read once and
/// remapped as it arrives, so only the merged matrix is ever resident.
#[pyfunction]
#[pyo3(signature = (paths, gene_maps, n_cols, block_rows=16_384, nnz_hint=0))]
fn read_h5ad_many(
    py: Python<'_>,
    paths: Vec<String>,
    gene_maps: Vec<PyReadonlyArray1<'_, u32>>,
    n_cols: usize,
    block_rows: usize,
    nnz_hint: usize,
) -> PyResult<PySparseMatrix> {
    let maps: Vec<Vec<u32>> = gene_maps
        .iter()
        .map(|m| m.as_slice().map(<[u32]>::to_vec))
        .collect::<Result<_, _>>()?;
    let csr = py
        .detach(|| polariseq_io::merge::read_many(&paths, &maps, n_cols, block_rows, nnz_hint))
        .map_err(|e| PyRuntimeError::new_err(format!("h5ad read failed: {e}")))?;
    Ok(PySparseMatrix::new(csr))
}

/// The shape, stored entries and gene names of a `.h5ad` file, without its
/// cell names or matrix: what planning several datasets needs of each.
#[pyfunction]
fn scan_h5ad_genes<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let err =
        |e: polariseq_io::IoError| PyRuntimeError::new_err(format!("cannot scan {path}: {e}"));
    let file = polariseq_io::AnnDataFile::open(path).map_err(err)?;
    let x = file.x_info().map_err(err)?;
    let dict = PyDict::new(py);
    dict.set_item("n_obs", x.n_obs)?;
    dict.set_item("n_vars", x.n_vars)?;
    dict.set_item("nnz", x.n_elements)?;
    dict.set_item("var_names", file.read_var_names().map_err(err)?)?;
    Ok(dict)
}

/// Read obs/var names and numeric columns from a `.h5ad` file.
///
/// Returns a dict with keys: `obs_names`, `var_names`, `obs` (dict of numeric
/// arrays), `var` (dict of numeric arrays).
#[pyfunction]
fn read_h5ad_metadata<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let file = polariseq_io::AnnDataFile::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("h5ad read failed: {e}")))?;
    let result = PyDict::new(py);

    if let Ok(names) = file.read_obs_names() {
        result.set_item("obs_names", names)?;
    }
    if let Ok(names) = file.read_var_names() {
        result.set_item("var_names", names)?;
    }
    let obs = PyDict::new(py);
    // Try common QC columns.
    for col in &["n_counts", "n_genes", "n_genes_by_counts", "total_counts"] {
        if let Ok(Some(vals)) = file.read_numeric_column("obs", col) {
            obs.set_item(*col, vals.into_pyarray(py))?;
        }
    }
    result.set_item("obs", obs)?;

    let var = PyDict::new(py);
    for col in &["n_cells", "n_cells_by_counts", "mean_counts"] {
        if let Ok(Some(vals)) = file.read_numeric_column("var", col) {
            var.set_item(*col, vals.into_pyarray(py))?;
        }
    }
    result.set_item("var", var)?;

    Ok(result)
}

/// Run PCA on a dense `(n_obs × n_features)` float32 matrix.
///
/// Returns a dict with `embedding` (`n_obs × n_components`), `variance_ratio`
/// (`n_components`), and `n_components`.
#[pyfunction]
fn pca_dense<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_components: usize,
    center: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let array = data.as_array();
    let (n_obs, n_features) = (array.shape()[0], array.shape()[1]);
    let flat: Vec<f32> = array.iter().copied().collect();
    let matrix = polariseq_core::DenseMatrix::from_row_major(flat, n_obs, n_features)
        .map_err(|e| PyRuntimeError::new_err(format!("matrix error: {e}")))?;
    let opts = polariseq_core::PcaOptions {
        n_components,
        center,
        seed,
        ..Default::default()
    };
    let result = py
        .detach(|| polariseq_core::pca(&polariseq_core::Matrix::Dense(matrix), &opts))
        .map_err(|e| PyRuntimeError::new_err(format!("PCA failed: {e}")))?;

    let dict = PyDict::new(py);
    let emb = flat_to_pyarray2(py, result.embedding, result.n_obs, result.n_components)?;
    dict.set_item("embedding", emb)?;
    dict.set_item("variance_ratio", result.variance_ratio.into_pyarray(py))?;
    dict.set_item("n_components", result.n_components)?;
    Ok(dict)
}

/// Run PCA on a Rust-owned sparse matrix, by reference (no copy). With `out`
/// (an `n_obs × n_components` float32 array, such as a memory-mapped file)
/// the embedding is written there and the dict's `embedding` is `None`.
///
/// Returns the same dict shape as `pca_dense`.
#[pyfunction]
#[pyo3(signature = (matrix, n_components, center, oversample, n_iter, seed, out=None))]
#[allow(clippy::too_many_arguments)]
fn pca_sparse<'py>(
    py: Python<'py>,
    matrix: PyRef<'py, PySparseMatrix>,
    n_components: usize,
    center: bool,
    oversample: usize,
    n_iter: usize,
    seed: u64,
    out: Option<PyReadwriteArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = polariseq_core::PcaOptions {
        n_components,
        center, // implicit centering: the matrix is never densified
        oversample,
        n_iter,
        seed,
    };
    // Bind the plain-Rust reference outside the closure so the `PyRef`
    // itself is not captured (`&CsrMatrix: Send` since `CsrMatrix: Sync`).
    let inner = &matrix.inner;
    let err =
        |e: polariseq_core::pca::PcaError| PyRuntimeError::new_err(format!("PCA failed: {e}"));
    if let Some(mut out) = out {
        let buf = out.as_slice_mut()?;
        let summary = py
            .detach(|| polariseq_core::pca::pca_csr_into(inner, &opts, buf))
            .map_err(err)?;
        return pca_dict(py, None, summary);
    }
    let result = py
        .detach(|| polariseq_core::pca_csr(inner, &opts))
        .map_err(err)?;
    let (emb, summary) = result.into_parts();
    pca_dict(py, Some(emb), summary)
}

/// PCA of the per-gene standardized sparse matrix (scanpy's `pp.scale`
/// followed by `tl.pca`), by reference and without densifying. `max_value`
/// clips the scaled values to `±max_value`, as scanpy does.
///
/// Returns the same dict shape as `pca_dense`.
#[pyfunction]
#[pyo3(signature = (matrix, n_components, max_value, seed, out=None))]
fn pca_sparse_scaled<'py>(
    py: Python<'py>,
    matrix: PyRef<'py, PySparseMatrix>,
    n_components: usize,
    max_value: Option<f32>,
    seed: u64,
    out: Option<PyReadwriteArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = polariseq_core::PcaOptions {
        n_components,
        seed,
        ..Default::default()
    };
    let inner = &matrix.inner;
    let err =
        |e: polariseq_core::pca::PcaError| PyRuntimeError::new_err(format!("PCA failed: {e}"));
    if let Some(mut out) = out {
        let buf = out.as_slice_mut()?;
        let summary = py
            .detach(|| polariseq_core::pca::pca_scaled_blocks_into(inner, &opts, max_value, buf))
            .map_err(err)?;
        return pca_dict(py, None, summary);
    }
    let result = py
        .detach(|| polariseq_core::pca::pca_scaled_blocks(inner, &opts, max_value))
        .map_err(err)?;
    let (emb, summary) = result.into_parts();
    pca_dict(py, Some(emb), summary)
}

/// Build a kNN neighbor graph from a dense embedding.
///
/// Returns a dict with `indptr`, `indices`, `data` (the CSR distance graph).
#[pyfunction]
fn neighbors_dense<'py>(
    py: Python<'py>,
    embedding: PyReadonlyArray2<'py, f32>,
    n_neighbors: usize,
    metric: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let array = embedding.as_array();
    let (n_obs, n_dim) = (array.shape()[0], array.shape()[1]);
    let flat: Vec<f32> = array.iter().copied().collect();
    let m = polariseq_core::Metric::parse(metric)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown metric: {metric}")))?;
    let opts = polariseq_core::NeighborsOptions {
        n_neighbors,
        metric: m,
        ..Default::default()
    };
    let graph = py
        .detach(|| polariseq_core::neighbors(&flat, n_obs, n_dim, &opts))
        .map_err(|e| PyRuntimeError::new_err(format!("neighbors failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("indptr", graph.indptr.into_pyarray(py))?;
    dict.set_item("indices", graph.indices.into_pyarray(py))?;
    dict.set_item("data", graph.data.into_pyarray(py))?;
    dict.set_item("n_obs", graph.n_obs)?;
    Ok(dict)
}

/// Build an approximate kNN graph via NN-Descent (for large datasets).
///
/// `max_iterations`, `max_candidates` and `n_trees` accept 0 for "auto".
/// Returns the same CSR distance graph format as `neighbors_dense`.
#[pyfunction]
#[pyo3(signature = (embedding, k, max_iterations=0, seed=0, max_candidates=0, n_trees=0, delta=0.001, out=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn nndescent_dense<'py>(
    py: Python<'py>,
    embedding: PyReadonlyArray2<'py, f32>,
    k: usize,
    max_iterations: usize,
    seed: u64,
    max_candidates: usize,
    n_trees: usize,
    delta: f64,
    out: Option<(
        PyReadwriteArray1<'py, u32>,
        PyReadwriteArray1<'py, u32>,
        PyReadwriteArray1<'py, f32>,
    )>,
) -> PyResult<Bound<'py, PyDict>> {
    let (n_obs, n_dim) = embedding.as_array().dim();
    // Read in place when contiguous (a memory-mapped embedding stays on disk).
    let owned: Vec<f32>;
    let flat: &[f32] = if let Ok(s) = embedding.as_slice() {
        s
    } else {
        owned = embedding.as_array().iter().copied().collect();
        &owned
    };
    let opts = polariseq_core::nndescent::NndescentOptions {
        k,
        max_iterations,
        seed,
        max_candidates,
        n_trees,
        delta,
        ..Default::default()
    };
    let dict = PyDict::new(py);
    if let Some((mut ip, mut ix, mut dx)) = out {
        let (ip, ix, dx) = (ip.as_slice_mut()?, ix.as_slice_mut()?, dx.as_slice_mut()?);
        let cap = n_obs * polariseq_core::nndescent::effective_k(n_obs, k);
        if ip.len() != n_obs + 1 || ix.len() < cap || dx.len() < cap {
            return Err(PyValueError::new_err(format!(
                "outputs must hold {} row pointers and {cap} neighbours",
                n_obs + 1
            )));
        }
        let nnz = py.detach(|| {
            polariseq_core::nndescent::nndescent_into(flat, n_obs, n_dim, &opts, ip, ix, dx)
        });
        dict.set_item("nnz", nnz)?;
        dict.set_item("n_obs", n_obs)?;
        return Ok(dict);
    }
    let graph = py.detach(|| polariseq_core::nndescent::nndescent(flat, n_obs, n_dim, &opts));
    dict.set_item("indptr", graph.indptr.into_pyarray(py))?;
    dict.set_item("indices", graph.indices.into_pyarray(py))?;
    dict.set_item("data", graph.data.into_pyarray(py))?;
    dict.set_item("n_obs", graph.n_obs)?;
    Ok(dict)
}

/// Symmetrised fuzzy connectivities of a kNN distance graph (scanpy's
/// `obsp['connectivities']` analogue): symmetric CSR, both directions,
/// rows sorted, no self-loops. Clustering and UMAP share this graph.
#[pyfunction]
fn fuzzy_connectivities<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
) -> PyResult<Bound<'py, PyDict>> {
    // Borrowed, not copied: the distance graph can be large (4M cells x 14
    // neighbours), and it is only read.
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let (c_indptr, c_indices, c_weights) =
        py.detach(|| polariseq_core::umap::fuzzy_connectivities_from(ip, ix, dx, n_obs));
    let dict = PyDict::new(py);
    dict.set_item("indptr", c_indptr.into_pyarray(py))?;
    dict.set_item("indices", c_indices.into_pyarray(py))?;
    dict.set_item("data", c_weights.into_pyarray(py))?;
    Ok(dict)
}

/// The fuzzy connectivities of a kNN distance graph, written into arrays the
/// caller allocates once their size is known: `alloc(n_rows, nnz)` returns
/// `(indptr, indices, weights)` (uint32 `n_rows + 1`, uint32 `nnz`, float32
/// `nnz`), such as memory-mapped result files. The rows are counted in one
/// pass and written in a second, in parallel, so the graph is never built
/// anywhere else. Returns `nnz`.
#[pyfunction]
fn fuzzy_connectivities_into<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    alloc: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    if ip.len() != n_obs + 1 {
        return Err(PyValueError::new_err("indptr must have n_obs + 1 entries"));
    }
    let (plan, lens) = py.detach(|| {
        let plan = polariseq_core::umap::FuzzyPlan::new(ip, ix, dx, n_obs);
        let lens = plan.row_lens();
        (plan, lens)
    });
    let nnz: u64 = lens.iter().map(|&l| u64::from(l)).sum();
    let nnz = u32::try_from(nnz)
        .map_err(|_| PyValueError::new_err("the connectivity graph exceeds 2^32 entries"))?
        as usize;
    let arrays = alloc.call1((n_obs, nnz))?;
    let (out_ip, out_ix, out_w): (
        PyReadwriteArray1<'py, u32>,
        PyReadwriteArray1<'py, u32>,
        PyReadwriteArray1<'py, f32>,
    ) = arrays.extract()?;
    let (mut out_ip, mut out_ix, mut out_w) = (out_ip, out_ix, out_w);
    let (oip, oix, ow) = (
        out_ip.as_slice_mut()?,
        out_ix.as_slice_mut()?,
        out_w.as_slice_mut()?,
    );
    if oip.len() != n_obs + 1 || oix.len() != nnz || ow.len() != nnz {
        return Err(PyValueError::new_err(
            "alloc returned arrays of the wrong size",
        ));
    }
    oip[0] = 0;
    for (i, &l) in lens.iter().enumerate() {
        oip[i + 1] = oip[i] + l;
    }
    let oip: &[u32] = oip;
    py.detach(|| plan.fill(oip, oix, ow));
    Ok(nnz)
}

/// Run Leiden clustering on the symmetric connectivity graph (both
/// directions stored, rows sorted by column), used as given: the graph in
/// `obsp['connectivities']`, not the kNN distances.
///
/// Returns a dict with `labels` (per-cell community assignment) and
/// `n_communities`.
#[pyfunction]
fn leiden_cluster<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    resolution: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    // Read in place: no copy of the graph, which may be a memory-mapped file.
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let opts = polariseq_core::LeidenOptions {
        resolution,
        seed,
        ..Default::default()
    };
    let result = py
        .detach(|| polariseq_core::clustering::leiden_connectivities_ref(ip, ix, dx, n_obs, &opts))
        .map_err(|e| PyRuntimeError::new_err(format!("leiden failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("labels", result.labels.into_pyarray(py))?;
    dict.set_item("n_communities", result.n_communities)?;
    dict.set_item("quality", result.quality)?;
    dict.set_item("n_edges", result.n_edges)?;
    dict.set_item("total_weight", result.total_weight)?;
    Ok(dict)
}

/// Fast community detection (parallel label propagation). Much faster than
/// Leiden at scale — O(edges) per iteration, no graph rebuilding.
#[pyfunction]
fn fast_community<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    resolution: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let opts = polariseq_core::louvain::FastCommunityOptions {
        resolution,
        seed,
        ..Default::default()
    };
    let result = py.detach(|| {
        polariseq_core::louvain::fast_community_connectivities_ref(ip, ix, dx, n_obs, &opts)
    });

    let dict = PyDict::new(py);
    dict.set_item("labels", result.labels.into_pyarray(py))?;
    dict.set_item("n_communities", result.n_communities)?;
    dict.set_item("quality", result.modularity)?;
    dict.set_item("n_edges", result.n_edges)?;
    dict.set_item("total_weight", result.total_weight)?;
    Ok(dict)
}

/// Paris hierarchical clustering of a CSR graph (rows sorted by column,
/// no duplicates, weights positive). Returns the `(n - 1) × 4` dendrogram
/// `[a, b, height, size]` (scipy `linkage` layout). `numpy_sum_block`
/// selects NumPy 1.x summation (8192) for reference tests only.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_obs, reorder=false, numpy_sum_block=None))]
fn paris_dendrogram<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f64>,
    n_obs: usize,
    reorder: bool,
    numpy_sum_block: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let opts = polariseq_core::ParisOptions {
        reorder,
        numpy_sum_block,
    };
    let merges = py
        .detach(|| polariseq_core::paris(ip, ix, dx, n_obs, &opts))
        .map_err(|e| PyRuntimeError::new_err(format!("paris failed: {e}")))?;
    let rows = merges.len();
    let flat: Vec<f64> = merges.into_iter().flatten().collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, 4), flat)
        .map_err(|e| PyRuntimeError::new_err(format!("shape error: {e}")))?;
    Ok(arr.into_pyarray(py))
}

fn to_merges(dendrogram: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<polariseq_core::Merge>> {
    let a = dendrogram.as_array();
    if a.ncols() != 4 {
        return Err(PyRuntimeError::new_err(format!(
            "a dendrogram has 4 columns, got {}",
            a.ncols()
        )));
    }
    Ok(a.rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect())
}

/// Cut a dendrogram into (at least) `n_clusters` clusters or at a height
/// `threshold` (scikit-network's `cut_straight`); labels by decreasing size.
#[pyfunction]
#[pyo3(signature = (dendrogram, n_clusters=None, threshold=None))]
fn dendrogram_cut_straight<'py>(
    py: Python<'py>,
    dendrogram: PyReadonlyArray2<'py, f64>,
    n_clusters: Option<usize>,
    threshold: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let merges = to_merges(&dendrogram)?;
    let labels = polariseq_core::cut_straight(&merges, n_clusters, threshold)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    Ok(labels.into_pyarray(py))
}

/// Scarf's balanced cut of a dendrogram; labels start at 1, as in Scarf.
#[pyfunction]
#[pyo3(signature = (dendrogram, max_size, min_size, max_distance_fc=2.0, numpy_sum_block=None))]
fn dendrogram_cut_balanced<'py>(
    py: Python<'py>,
    dendrogram: PyReadonlyArray2<'py, f64>,
    max_size: u64,
    min_size: u64,
    max_distance_fc: f64,
    numpy_sum_block: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let merges = to_merges(&dendrogram)?;
    let labels = py
        .detach(|| {
            polariseq_core::cut_balanced(
                &merges,
                max_size,
                min_size,
                max_distance_fc,
                numpy_sum_block,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    Ok(labels.into_pyarray(py))
}

/// Harmony batch correction of an embedding (cells × dimensions, C order),
/// read in place; the corrected embedding is written into `out` (same
/// shape, e.g. a memory-mapped result file). `codes` holds one array of
/// level codes per batch variable and `n_levels` their numbers of levels;
/// `theta` and `lamb` give one value per variable or one for all. Returns
/// what the run did.
#[pyfunction]
#[pyo3(signature = (embedding, codes, n_levels, out, *, theta, sigma, lamb, n_clusters, tau,
    block_size, max_iter_harmony, max_iter_kmeans, epsilon_cluster, epsilon_harmony, alpha,
    batch_prop_cutoff, seed))]
#[allow(clippy::too_many_arguments)]
fn harmony_correct<'py>(
    py: Python<'py>,
    embedding: PyReadonlyArray2<'py, f32>,
    codes: Vec<PyReadonlyArray1<'py, u32>>,
    n_levels: Vec<usize>,
    mut out: PyReadwriteArray2<'py, f32>,
    theta: Vec<f32>,
    sigma: f32,
    lamb: Option<Vec<f32>>,
    n_clusters: Option<usize>,
    tau: f32,
    block_size: f32,
    max_iter_harmony: usize,
    max_iter_kmeans: usize,
    epsilon_cluster: f32,
    epsilon_harmony: f32,
    alpha: f32,
    batch_prop_cutoff: f32,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    use polariseq_core::harmony::{harmony_into, Covariate, HarmonyError, HarmonyOptions};
    let (n, d) = embedding.as_array().dim();
    let z = embedding.as_slice()?;
    let buf = out.as_slice_mut()?;
    if codes.len() != n_levels.len() {
        return Err(PyValueError::new_err(
            "one number of levels per batch variable",
        ));
    }
    let slices = codes
        .iter()
        .map(|c| c.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let covariates: Vec<Covariate<'_>> = slices
        .iter()
        .zip(&n_levels)
        .map(|(&codes, &n_levels)| Covariate { codes, n_levels })
        .collect();
    let opts = HarmonyOptions {
        n_clusters,
        theta,
        sigma,
        lambda: lamb,
        alpha,
        tau,
        block_size,
        max_iter_harmony,
        max_iter_kmeans,
        epsilon_cluster,
        epsilon_harmony,
        batch_prop_cutoff,
        seed,
    };
    let report = py
        .detach(|| harmony_into(z, n, d, &covariates, &opts, buf))
        .map_err(|e| match e {
            HarmonyError::Numerical(_) => PyRuntimeError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string()),
        })?;
    let dict = PyDict::new(py);
    dict.set_item("n_clusters", report.n_clusters)?;
    dict.set_item("objective", report.objective)?;
    dict.set_item("kmeans_rounds", report.kmeans_rounds)?;
    dict.set_item("converged", report.converged)?;
    Ok(dict)
}

/// Run the whole disk-backed preprocessing pass on a `.h5ad` too large for
/// RAM: stream-spill to SSD, QC, `normalize_total`, `log1p`, HVG, PCA.
///
/// Nothing is ever fully materialised — the previous pair of functions
/// (`extract_csr_to_disk` + `pca_disk`) read the entire matrix into RAM and
/// then built a second complete copy of it as bytes before writing. Returns
/// the embedding plus the QC and HVG results computed along the way.
#[pyfunction]
#[pyo3(signature = (path, spill_dir, n_hvg=2000, n_pcs=50, target_sum=1e4, seed=0, block_rows=0,
                    min_genes=0, max_genes=None, min_counts=0.0, max_counts=None,
                    max_pct_mt=None, mito=None, min_cells=0, hvg_floor_frac=0.0,
                    renormalize=None, scale_pcs=false))]
#[allow(clippy::too_many_arguments)]
fn disk_backed_preprocess<'py>(
    py: Python<'py>,
    path: &str,
    spill_dir: &str,
    n_hvg: usize,
    n_pcs: usize,
    target_sum: f32,
    seed: u64,
    block_rows: usize,
    min_genes: u32,
    max_genes: Option<u32>,
    min_counts: f32,
    max_counts: Option<f32>,
    max_pct_mt: Option<f32>,
    mito: Option<PyReadonlyArray1<'py, bool>>,
    min_cells: u32,
    hvg_floor_frac: f32,
    renormalize: Option<f32>,
    scale_pcs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mito: Option<Vec<bool>> = match mito {
        Some(m) => Some(m.as_slice()?.to_vec()),
        None => None,
    };
    let defaults = polariseq_io::spill::DiskPipelineOptions::default();
    let opts = polariseq_io::spill::DiskPipelineOptions {
        target_sum,
        n_hvg,
        n_pcs,
        min_genes,
        max_genes: max_genes.unwrap_or(u32::MAX),
        min_counts,
        max_counts: max_counts.unwrap_or(f32::INFINITY),
        max_pct_mt: max_pct_mt.unwrap_or(f32::INFINITY),
        mito,
        min_cells,
        seed,
        spill_dir: std::path::PathBuf::from(spill_dir),
        // 0 means "no budget supplied": keep the built-in default.
        block_rows: if block_rows == 0 {
            defaults.block_rows
        } else {
            block_rows
        },
        hvg_floor_frac,
        renormalize,
        scale_pcs,
    };
    let r = py
        .detach(|| polariseq_io::spill::disk_backed_preprocess(path, &opts))
        .map_err(|e| PyRuntimeError::new_err(format!("disk-backed preprocessing failed: {e}")))?;

    let dict = PyDict::new(py);
    let emb = flat_to_pyarray2(py, r.embedding, r.n_rows, r.n_components)?;
    dict.set_item("embedding", emb)?;
    dict.set_item("variance_ratio", r.variance_ratio.into_pyarray(py))?;
    dict.set_item("n_components", r.n_components)?;
    dict.set_item("n_obs", r.n_rows)?;
    dict.set_item("n_obs_input", r.n_rows_input)?;
    // The keep mask lets the caller line the embedding back up with the
    // file's cell barcodes; the disk path filters, so row i of `embedding`
    // is not input row i.
    dict.set_item("cell_keep", r.cell_keep)?;
    dict.set_item("gene_keep", r.gene_keep.into_pyarray(py))?;
    if let Some(p) = r.pct_counts_mt {
        dict.set_item("pct_counts_mt", p.into_pyarray(py))?;
    }
    dict.set_item("n_vars", r.n_cols)?;
    dict.set_item("nnz", r.nnz)?;
    dict.set_item("n_counts", r.qc.n_counts.into_pyarray(py))?;
    dict.set_item("n_genes", r.qc.n_genes.into_pyarray(py))?;
    dict.set_item("gene_n_cells", r.qc.gene_n_cells.into_pyarray(py))?;
    dict.set_item("gene_total_counts", r.qc.gene_total_counts.into_pyarray(py))?;
    dict.set_item("highly_variable", r.hvg.mask)?;
    dict.set_item("hvg_means", r.hvg.means.into_pyarray(py))?;
    dict.set_item(
        "hvg_dispersions_norm",
        r.hvg.dispersions_norm.into_pyarray(py),
    )?;
    Ok(dict)
}

/// Run UMAP on a kNN distance graph. Returns the 2D embedding (n × 2).
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_obs, min_dist, spread, n_epochs, seed, deterministic=false, connectivities=false, out=None, init="spectral"))]
fn umap_embed<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: usize,
    seed: u64,
    deterministic: bool,
    connectivities: bool,
    out: Option<PyReadwriteArray2<'py, f32>>,
    init: &str,
) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
    // The graph is read in place (it may be a memory-mapped file); with
    // `connectivities` it is the fuzzy set itself, used as it is.
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let init = match init {
        "spectral" => polariseq_core::umap::UmapInit::Spectral,
        "random" => polariseq_core::umap::UmapInit::Random,
        // the start is what `out` holds when the call is made
        "given" => polariseq_core::umap::UmapInit::Given,
        other => return Err(PyValueError::new_err(format!("unknown init {other:?}"))),
    };
    if init == polariseq_core::umap::UmapInit::Given && out.is_none() {
        return Err(PyValueError::new_err("init='given' needs `out` holding the start"));
    }
    let opts = polariseq_core::umap::UmapOptions {
        min_dist,
        spread,
        n_epochs: if n_epochs > 0 { Some(n_epochs) } else { None },
        seed,
        deterministic,
        init,
        ..Default::default()
    };
    let run = |buf: &mut [f32]| {
        if connectivities {
            polariseq_core::umap::umap_from_connectivities_into(ip, ix, dx, n_obs, &opts, buf);
        } else {
            polariseq_core::umap::umap_into(ip, ix, dx, n_obs, &opts, buf);
        }
    };
    if let Some(mut out) = out {
        let buf = out.as_slice_mut()?;
        if buf.len() != n_obs * 2 {
            return Err(PyValueError::new_err("out must be n_obs × 2"));
        }
        py.detach(|| run(buf));
        return Ok(None);
    }
    let mut emb = vec![0.0_f32; n_obs * 2];
    py.detach(|| run(&mut emb));
    Ok(Some(flat_to_pyarray2(py, emb, n_obs, 2)?))
}

/// A seurat-flavour result as a dict (used by the detection-floor variant).
pub(crate) fn hvg_seurat_dict(
    py: Python<'_>,
    r: polariseq_core::preprocess::HvgResult,
) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mask", r.mask.into_pyarray(py))?;
    d.set_item("means", r.means.into_pyarray(py))?;
    d.set_item("dispersions", r.dispersions.into_pyarray(py))?;
    d.set_item("dispersions_norm", r.dispersions_norm.into_pyarray(py))?;
    d.set_item("n_hvg", r.n_hvg)?;
    Ok(d)
}

/// The result of Scarf-flavour gene selection as a dict (shared by the
/// in-memory and out-of-core matrices).
pub(crate) fn hvg_scarf_dict(
    py: Python<'_>,
    r: polariseq_core::preprocess::HvgScarf,
) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mask", r.mask.into_pyarray(py))?;
    d.set_item("log_mean", r.log_mean.into_pyarray(py))?;
    d.set_item("log_var", r.log_var.into_pyarray(py))?;
    d.set_item("corrected_var", r.corrected_var.into_pyarray(py))?;
    d.set_item("detected", r.detected.into_pyarray(py))?;
    d.set_item("min_cells", r.min_cells)?;
    d.set_item("n_hvg", r.n_hvg)?;
    Ok(d)
}

/// The spectral start of UMAP alone, for a symmetric fuzzy connectivity graph
/// (n × 2, each axis in [-10, 10]).
#[pyfunction]
fn umap_spectral_layout<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (ip, ix, dx) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    let emb = py.detach(|| polariseq_core::umap::spectral_layout(ip, ix, dx, n_obs, seed));
    flat_to_pyarray2(py, emb, n_obs, 2)
}

/// Scarf's UMAP start from cell coordinates (n x d float32, typically the
/// principal components): k-means centroids laid out by their own PCA, each
/// cell at its centroid. Returns n x 2.
#[pyfunction]
#[pyo3(signature = (data, n_centroids=1000, seed=0))]
fn umap_kmeans_start<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_centroids: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (n, d) = {
        let a = data.as_array();
        (a.shape()[0], a.shape()[1])
    };
    // a C-contiguous array is read in place; anything else must be made so first
    let x = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be a C-contiguous float32 array"))?;
    let emb = py.detach(|| polariseq_core::umap::kmeans_start(x, n, d, n_centroids, seed));
    flat_to_pyarray2(py, emb, n, 2)
}

/// k-means groups of cell coordinates (n x d float32): each cell's group, as
/// Scarf's UMAP start computes them (see `umap::kmeans_labels`).
#[pyfunction]
#[pyo3(signature = (data, n_groups=1000, seed=0))]
fn umap_kmeans_labels<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    n_groups: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let (n, d) = {
        let a = data.as_array();
        (a.shape()[0], a.shape()[1])
    };
    let x = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be a C-contiguous float32 array"))?;
    let labels = py.detach(|| polariseq_core::umap::kmeans_labels(x, n, d, n_groups, seed).0);
    Ok(labels.into_pyarray(py))
}

/// Polariseq's multilevel spectral start for a symmetric fuzzy connectivity
/// graph, through the graph of the given groups (see `umap::multilevel_layout`).
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_obs, labels, n_groups, smooth=30))]
#[allow(clippy::too_many_arguments)]
fn umap_multilevel_layout<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    labels: PyReadonlyArray1<'py, u32>,
    n_groups: usize,
    smooth: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (ip, ix, dx, lb) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?, labels.as_slice()?);
    if lb.len() != n_obs || lb.iter().any(|&g| g as usize >= n_groups) {
        return Err(PyValueError::new_err("labels must hold one group in 0..n_groups per cell"));
    }
    let emb = py.detach(|| polariseq_core::umap::multilevel_layout(ip, ix, dx, n_obs, lb, n_groups, smooth));
    flat_to_pyarray2(py, emb, n_obs, 2)
}

/// Wrap a 10x read result as a dict: `matrix` (Rust-owned `SparseMatrix`),
/// `n_cells`, `n_genes`, `barcodes`, `gene_names`.
fn tenx_to_dict(
    py: Python<'_>,
    result: polariseq_io::tenx::TenxData,
) -> PyResult<Bound<'_, PyDict>> {
    let csr = match result.matrix {
        polariseq_core::Matrix::Csr(c) => c,
        polariseq_core::Matrix::Dense(_) => return Err(PyRuntimeError::new_err("expected sparse")),
    };
    let dict = PyDict::new(py);
    dict.set_item("matrix", PySparseMatrix::new(csr))?;
    dict.set_item("n_cells", result.n_cells)?;
    dict.set_item("n_genes", result.n_genes)?;
    dict.set_item("barcodes", result.barcodes)?;
    dict.set_item("gene_names", result.gene_names)?;
    Ok(dict)
}

/// Read a 10x Genomics .h5 file (cells × genes, Rust-owned sparse matrix).
#[pyfunction]
fn read_10x_h5<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let result = polariseq_io::tenx::read_10x_h5(path)
        .map_err(|e| PyRuntimeError::new_err(format!("10x h5 read failed: {e}")))?;
    tenx_to_dict(py, result)
}

/// Read a 10x MatrixMarket directory (matrix.mtx + barcodes.tsv + features.tsv).
/// Returns the same dict format as `read_10x_h5`.
#[pyfunction]
fn read_10x_mtx<'py>(py: Python<'py>, dir: &str) -> PyResult<Bound<'py, PyDict>> {
    let result = polariseq_io::tenx::read_10x_mtx(dir)
        .map_err(|e| PyRuntimeError::new_err(format!("10x mtx read failed: {e}")))?;
    tenx_to_dict(py, result)
}

/// Scan an .h5ad for metadata WITHOUT loading matrix data (instant).
/// Returns dict with n_cells, n_genes, nnz, is_sparse, gene_names, cell_names,
/// estimated_ram_bytes, file_size_bytes.
#[pyfunction]
fn scan_h5ad_meta<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let meta = polariseq_io::lazy::scan_h5ad(path)
        .map_err(|e| PyRuntimeError::new_err(format!("scan failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("n_cells", meta.n_cells)?;
    dict.set_item("n_genes", meta.n_genes)?;
    dict.set_item("nnz", meta.nnz)?;
    dict.set_item("is_sparse", meta.is_sparse)?;
    dict.set_item("gene_names", meta.gene_names)?;
    dict.set_item("cell_names", meta.cell_names)?;
    dict.set_item("estimated_ram_bytes", meta.estimated_ram_bytes)?;
    dict.set_item("file_size_bytes", meta.file_size_bytes)?;
    Ok(dict)
}

/// Read rows [start, end) from a sparse .h5ad — partial load via HDF5 hyperslab.
/// Returns dict with data/indices/indptr/n_cells/n_genes (CSR triple).
#[pyfunction]
fn read_h5ad_rows<'py>(
    py: Python<'py>,
    path: &str,
    start: usize,
    end: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let csr = polariseq_io::lazy::read_rows_h5ad(path, start, end)
        .map_err(|e| PyRuntimeError::new_err(format!("row read failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("data", csr.data.into_pyarray(py))?;
    dict.set_item("indices", csr.indices.into_pyarray(py))?;
    dict.set_item("indptr", csr.indptr.into_pyarray(py))?;
    dict.set_item("n_cells", csr.nrows)?;
    dict.set_item("n_genes", csr.ncols)?;
    Ok(dict)
}

/// Validate an .h5ad file: check structure, shapes, sanity.
#[pyfunction]
fn validate_h5ad_file<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let report = polariseq_io::lazy::validate_h5ad(path)
        .map_err(|e| PyRuntimeError::new_err(format!("validation failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("valid", report.valid)?;
    dict.set_item("issues", report.issues)?;
    dict.set_item("n_cells", report.meta.n_cells)?;
    dict.set_item("n_genes", report.meta.n_genes)?;
    Ok(dict)
}

/// Streaming multi-dataset batch correction: processes each .h5ad independently
/// (never in RAM together), merges embeddings, runs Harmony, clusters, UMAP.
#[pyfunction]
fn streaming_merge_correct<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    n_components: usize,
    chunk_size: usize,
    n_harmony_clusters: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    let opts = polariseq_io::stream_correct::StreamingCorrectOptions {
        n_components,
        chunk_size,
        n_harmony_clusters,
        seed,
        ..Default::default()
    };
    let result = polariseq_io::stream_correct::streaming_merge_correct(&path_refs, &opts)
        .map_err(|e| PyRuntimeError::new_err(format!("streaming correction failed: {e}")))?;

    let total_cells: usize = result.dataset_sizes.iter().sum();
    let dict = PyDict::new(py);
    let emb = flat_to_pyarray2(py, result.embedding, total_cells, n_components)?;
    dict.set_item("embedding", emb)?;
    dict.set_item("batch_labels", result.batch_labels.into_pyarray(py))?;
    dict.set_item("cluster_labels", result.cluster_labels.into_pyarray(py))?;
    dict.set_item("umap", result.umap.into_pyarray(py))?;
    dict.set_item("dataset_sizes", result.dataset_sizes)?;
    dict.set_item("n_clusters", result.n_clusters)?;
    dict.set_item(
        "peak_streaming_mb",
        result.peak_streaming_bytes as f64 / 1e6,
    )?;
    dict.set_item("elapsed_s", result.elapsed_secs)?;
    Ok(dict)
}

/// Streaming multi-dataset integration with batch correction.
/// Processes chunk-by-chunk, never loads full matrices.
#[pyfunction]
fn integrate_datasets<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    n_hvg: usize,
    n_pcs: usize,
    chunk_size: usize,
    reference_size: usize,
    harmony_clusters: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let config = polariseq_io::integrate::IntegrateConfig {
        n_hvg,
        n_pcs,
        chunk_size,
        reference_size,
        harmony_clusters,
        seed,
    };
    let result = polariseq_io::integrate::integrate_datasets(&paths, &config)
        .map_err(|e| PyRuntimeError::new_err(format!("integration failed: {e}")))?;

    let n_total = result.stats.n_total_cells;
    let k = n_pcs;

    let dict = PyDict::new(py);
    let emb = flat_to_pyarray2(py, result.embedding, n_total, k)?;
    dict.set_item("embedding", emb)?;
    dict.set_item("batch_labels", result.batch_labels.into_pyarray(py))?;
    dict.set_item("dataset_sizes", result.dataset_sizes)?;
    dict.set_item("shared_genes", result.shared_genes)?;
    dict.set_item("hvg_indices", result.hvg_indices)?;

    let stats = PyDict::new(py);
    stats.set_item("scan_time_s", result.stats.scan_time_s)?;
    stats.set_item("hvg_time_s", result.stats.hvg_time_s)?;
    stats.set_item("pca_time_s", result.stats.reference_pca_time_s)?;
    stats.set_item("harmony_time_s", result.stats.harmony_time_s)?;
    stats.set_item("total_time_s", result.stats.total_time_s)?;
    stats.set_item("n_total_cells", result.stats.n_total_cells)?;
    dict.set_item("stats", stats)?;

    Ok(dict)
}

/// Differential expression: rank genes per cluster vs rest. Takes the
/// Rust-owned sparse matrix by reference.
#[pyfunction]
fn rank_genes_groups<'py>(
    py: Python<'py>,
    matrix: PyRef<'py, PySparseMatrix>,
    labels: PyReadonlyArray1<'py, u32>,
    n_groups: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let labels_vec: Vec<u32> = labels.as_slice()?.to_vec();
    let result = polariseq_core::de::rank_genes_groups(&matrix.inner, &labels_vec, n_groups)
        .map_err(|e| PyRuntimeError::new_err(format!("DE failed: {e}")))?;

    de_result_dict(py, &result)
}

/// A ranking result as the dict `rank_genes_groups` returns to Python.
pub(crate) fn de_result_dict<'py>(
    py: Python<'py>,
    result: &polariseq_core::de::DeResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let mut groups_list: Vec<Bound<'py, PyDict>> = Vec::with_capacity(result.groups.len());
    for g in &result.groups {
        let gd = PyDict::new(py);
        gd.set_item("group_id", g.group_id)?;
        gd.set_item("gene_indices", g.gene_indices.clone().into_pyarray(py))?;
        gd.set_item("scores", g.scores.clone().into_pyarray(py))?;
        gd.set_item("logfoldchanges", g.logfoldchanges.clone().into_pyarray(py))?;
        gd.set_item("pvals", g.pvals.clone().into_pyarray(py))?;
        gd.set_item("pvals_adj", g.pvals_adj.clone().into_pyarray(py))?;
        gd.set_item("pts", g.pts.clone().into_pyarray(py))?;
        groups_list.push(gd);
    }
    dict.set_item("groups", groups_list)?;
    dict.set_item("n_genes", result.n_genes)?;
    dict.set_item("n_cells", result.n_cells)?;
    Ok(dict)
}

/// Marker ranking for a matrix that stays in its file: per-cluster
/// statistics of the normalised, log-transformed matrix streamed from
/// `path` (rows normalised over the genes in `gene_keep`), then the same
/// Welch test as the in-memory `rank_genes_groups`. Labels `>= n_groups`
/// mark cells left out (filtered in quality control). Gene indices refer to
/// the file's genes.
#[pyfunction]
#[pyo3(signature = (path, labels, n_groups, target_sum, gene_keep=None))]
fn rank_genes_groups_file<'py>(
    py: Python<'py>,
    path: &str,
    labels: PyReadonlyArray1<'py, u32>,
    n_groups: usize,
    target_sum: f32,
    gene_keep: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyDict>> {
    let labels_vec: Vec<u32> = labels.as_slice()?.to_vec();
    let keep: Option<Vec<bool>> = match gene_keep {
        Some(k) => Some(k.as_slice()?.to_vec()),
        None => None,
    };
    let result = py
        .detach(|| -> Result<polariseq_core::de::DeResult, String> {
            let stats = polariseq_io::lazy_compute::streaming_cluster_stats_normalized_masked(
                path,
                &labels_vec,
                n_groups,
                target_sum,
                keep.as_deref(),
            )
            .map_err(|e| format!("streaming statistics failed: {e}"))?;
            polariseq_core::de::rank_genes_groups_from_stats(&stats, keep.as_deref())
                .map_err(|e| format!("DE failed: {e}"))
        })
        .map_err(PyRuntimeError::new_err)?;
    de_result_dict(py, &result)
}

/// Diffusion pseudotime via Dijkstra from root cell.
#[pyfunction]
fn pseudotime_dijkstra<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    root: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let graph = polariseq_core::backend::NeighborGraph {
        indptr: indptr.as_slice()?.to_vec(),
        indices: indices.as_slice()?.to_vec(),
        data: data.as_slice()?.to_vec(),
        n_obs,
    };
    let result = polariseq_core::trajectory::pseudotime(&graph, root)
        .map_err(|e| PyRuntimeError::new_err(format!("pseudotime failed: {e}")))?;
    let dict = PyDict::new(py);
    dict.set_item("pseudotime", result.pseudotime.into_pyarray(py))?;
    dict.set_item("root", result.root)?;
    dict.set_item("n_reachable", result.n_reachable)?;
    Ok(dict)
}

/// PAGA connectivities between clusters.
#[pyfunction]
fn paga_connectivities<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, f32>,
    n_obs: usize,
    labels: PyReadonlyArray1<'py, u32>,
    n_clusters: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let graph = polariseq_core::backend::NeighborGraph {
        indptr: indptr.as_slice()?.to_vec(),
        indices: indices.as_slice()?.to_vec(),
        data: data.as_slice()?.to_vec(),
        n_obs,
    };
    let labels_vec: Vec<u32> = labels.as_slice()?.to_vec();
    let result = polariseq_core::trajectory::paga(&graph, &labels_vec, n_clusters)
        .map_err(|e| PyRuntimeError::new_err(format!("paga failed: {e}")))?;
    let dict = PyDict::new(py);
    dict.set_item("connectivities", result.connectivities.into_pyarray(py))?;
    dict.set_item("n_clusters", result.n_clusters)?;
    dict.set_item("cluster_sizes", result.cluster_sizes)?;
    Ok(dict)
}

/// Streaming cluster means: per-gene per-cluster expression WITHOUT loading
/// the full matrix. Memory: O(n_genes × n_clusters) + O(chunk).
#[pyfunction]
fn streaming_cluster_means<'py>(
    py: Python<'py>,
    path: &str,
    labels: PyReadonlyArray1<'py, u32>,
    n_clusters: usize,
    chunk_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let labels_vec: Vec<u32> = labels.as_slice()?.to_vec();
    let (means, n_genes, nc) = polariseq_io::lazy_compute::streaming_cluster_means(
        path,
        &labels_vec,
        n_clusters,
        chunk_size,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("streaming means failed: {e}")))?;

    let dict = PyDict::new(py);
    let means_arr = numpy::ndarray::Array2::from_shape_vec((n_genes, nc), means)
        .map_err(|e| PyRuntimeError::new_err(format!("reshape: {e}")))?;
    dict.set_item("means", means_arr.to_pyarray(py))?;
    dict.set_item("n_genes", n_genes)?;
    dict.set_item("n_clusters", nc)?;
    Ok(dict)
}

/// Streaming per-cluster statistics over the NORMALISED (normalize-total +
/// log1p) matrix: sums, sums of squares, expressing-cell counts and cluster
/// sizes, per (gene, cluster) — without materialising the matrix. Memory is
/// O(n_genes × n_clusters) plus one block. Label ids >= n_clusters are
/// ignored, which is how a caller excludes quality-filtered cells.
#[pyfunction]
fn streaming_cluster_stats_normalized<'py>(
    py: Python<'py>,
    path: &str,
    labels: PyReadonlyArray1<'py, u32>,
    n_clusters: usize,
    target_sum: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let labels_vec: Vec<u32> = labels.as_slice()?.to_vec();
    let stats = polariseq_io::lazy_compute::streaming_cluster_stats_normalized(
        path,
        &labels_vec,
        n_clusters,
        target_sum,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("streaming normalised stats failed: {e}")))?;

    let (ng, nc) = (stats.n_genes, stats.n_clusters);
    let shape_err =
        |e: numpy::ndarray::ShapeError| PyRuntimeError::new_err(format!("reshape: {e}"));
    let dict = PyDict::new(py);
    let sums = numpy::ndarray::Array2::from_shape_vec((ng, nc), stats.sums)
        .map_err(shape_err)?
        .to_pyarray(py);
    dict.set_item("sums", sums)?;
    let sq_sums = numpy::ndarray::Array2::from_shape_vec((ng, nc), stats.sq_sums)
        .map_err(shape_err)?
        .to_pyarray(py);
    dict.set_item("sq_sums", sq_sums)?;
    let counts = numpy::ndarray::Array2::from_shape_vec((ng, nc), stats.counts)
        .map_err(shape_err)?
        .to_pyarray(py);
    dict.set_item("counts", counts)?;
    dict.set_item(
        "cluster_sizes",
        numpy::ndarray::Array1::from(stats.cluster_sizes).to_pyarray(py),
    )?;
    dict.set_item("n_genes", ng)?;
    dict.set_item("n_clusters", nc)?;
    Ok(dict)
}

/// Module entry point.
#[pymodule]
fn _polariseq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparseMatrix>()?;
    m.add_class::<disk::PyDiskMatrix>()?;
    m.add_function(wrap_pyfunction!(triple, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad_x, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad_csr, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad_many, m)?)?;
    m.add_function(wrap_pyfunction!(scan_h5ad_genes, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(frame::read_h5ad_frame, m)?)?;
    m.add_function(wrap_pyfunction!(doublets::scrublet, m)?)?;
    m.add_function(wrap_pyfunction!(compare::pseudobulk_sums, m)?)?;
    m.add_function(wrap_pyfunction!(compare::compare_pseudobulk, m)?)?;
    m.add_function(wrap_pyfunction!(compare::compare_cells, m)?)?;
    m.add_function(wrap_pyfunction!(compare::group_means, m)?)?;
    m.add_function(wrap_pyfunction!(pca_dense, m)?)?;
    m.add_function(wrap_pyfunction!(pca_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(pca_sparse_scaled, m)?)?;
    m.add_function(wrap_pyfunction!(neighbors_dense, m)?)?;
    m.add_function(wrap_pyfunction!(nndescent_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_connectivities, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_connectivities_into, m)?)?;
    m.add_function(wrap_pyfunction!(leiden_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(fast_community, m)?)?;
    m.add_function(wrap_pyfunction!(harmony_correct, m)?)?;
    m.add_function(wrap_pyfunction!(disk_backed_preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(umap_embed, m)?)?;
    m.add_function(wrap_pyfunction!(umap_spectral_layout, m)?)?;
    m.add_function(wrap_pyfunction!(umap_kmeans_start, m)?)?;
    m.add_function(wrap_pyfunction!(umap_kmeans_labels, m)?)?;
    m.add_function(wrap_pyfunction!(umap_multilevel_layout, m)?)?;
    m.add_function(wrap_pyfunction!(read_10x_h5, m)?)?;
    m.add_function(wrap_pyfunction!(read_10x_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(scan_h5ad_meta, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad_rows, m)?)?;
    m.add_function(wrap_pyfunction!(validate_h5ad_file, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_datasets, m)?)?;
    m.add_function(wrap_pyfunction!(rank_genes_groups, m)?)?;
    m.add_function(wrap_pyfunction!(rank_genes_groups_file, m)?)?;
    m.add_function(wrap_pyfunction!(pseudotime_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(paga_connectivities, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_cluster_means, m)?)?;
    m.add_function(wrap_pyfunction!(paris_dendrogram, m)?)?;
    m.add_function(wrap_pyfunction!(dendrogram_cut_straight, m)?)?;
    m.add_function(wrap_pyfunction!(dendrogram_cut_balanced, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_cluster_stats_normalized, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_merge_correct, m)?)?;
    m.add_function(wrap_pyfunction!(scan_h5ad, m)?)?;
    m.add_function(wrap_pyfunction!(scan_10x_h5, m)?)?;
    m.add("__version__", polariseq_core::VERSION)?;
    Ok(())
}
