//! `scrublet`: doublet scores of the rows of an in-memory or out-of-core
//! matrix of raw counts. The scores and calls are written into arrays the
//! caller owns (memory-mapped files in the project folder, or RAM), so
//! nothing per cell is copied back.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadwriteArray1};
use polariseq_core::doublets::{scrublet_blocks, BatchResult, ScrubletOptions};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::disk::PyDiskMatrix;
use crate::sparse::PySparseMatrix;

fn batch_dict<'py>(py: Python<'py>, r: BatchResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("batch", r.batch)?;
    d.set_item("n_cells", r.n_cells)?;
    d.set_item("n_genes", r.n_genes)?;
    d.set_item("n_sim", r.n_sim)?;
    d.set_item("n_neighbors", r.n_neighbors)?;
    d.set_item("threshold", r.threshold)?;
    d.set_item("doublet_scores_sim", r.scores_sim.into_pyarray(py))?;
    d.set_item("detected_doublet_rate", r.detected_doublet_rate)?;
    d.set_item("detectable_doublet_fraction", r.detectable_doublet_fraction)?;
    d.set_item("overall_doublet_rate", r.overall_doublet_rate)?;
    d.set_item("note", r.note)?;
    Ok(d)
}

/// Scrublet doublet scores of every row of `matrix` (raw counts), batch by
/// batch: `batch[i]` is row `i`'s batch, below `n_batches`, or `2**32 - 1`
/// to leave it unscored. Fills `scores` and `calls` (one entry per row) and
/// returns one dict per batch scored.
#[pyfunction]
#[pyo3(signature = (matrix, batch, n_batches, scores, calls, *, sim_doublet_ratio=2.0,
    expected_doublet_rate=0.05, stdev_doublet_rate=0.02, n_neighbors=None, n_prin_comps=30,
    min_counts=3.0, min_cells=3, min_gene_variability_pctl=85.0, threshold=None, seed=0,
    exact_knn_max=60_000, max_batch_bytes=1_073_741_824))]
#[allow(clippy::too_many_arguments)]
pub fn scrublet<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    batch: PyReadonlyArray1<'py, u32>,
    n_batches: usize,
    mut scores: PyReadwriteArray1<'py, f32>,
    mut calls: PyReadwriteArray1<'py, bool>,
    sim_doublet_ratio: f64,
    expected_doublet_rate: f64,
    stdev_doublet_rate: f64,
    n_neighbors: Option<usize>,
    n_prin_comps: usize,
    min_counts: f64,
    min_cells: usize,
    min_gene_variability_pctl: f64,
    threshold: Option<f64>,
    seed: u64,
    exact_knn_max: usize,
    max_batch_bytes: usize,
) -> PyResult<Bound<'py, PyList>> {
    let opts = ScrubletOptions {
        sim_doublet_ratio,
        expected_doublet_rate,
        stdev_doublet_rate,
        n_neighbors,
        n_prin_comps,
        min_counts,
        min_cells,
        min_gene_variability_pctl,
        threshold,
        seed,
        exact_knn_max,
        max_batch_bytes,
    };
    let batch = batch.as_slice()?;
    let scores = scores.as_slice_mut()?;
    let calls = calls.as_slice_mut()?;
    let n = batch.len();
    if scores.len() != n || calls.len() != n {
        return Err(PyValueError::new_err(
            "one batch code, score and call per row",
        ));
    }
    let results = if let Ok(m) = matrix.cast::<PySparseMatrix>() {
        let m = m.borrow();
        let inner = &m.inner;
        if inner.nrows != n {
            return Err(PyValueError::new_err(format!(
                "{n} batch codes for {} rows",
                inner.nrows
            )));
        }
        py.detach(|| scrublet_blocks(inner, batch, n_batches, &opts, scores, calls))
    } else if let Ok(d) = matrix.cast::<PyDiskMatrix>() {
        let d = d.borrow();
        let dm: &PyDiskMatrix = &d;
        py.detach(|| {
            dm.raw_view(|v| {
                if v.n_rows() == n {
                    Ok(scrublet_blocks(v, batch, n_batches, &opts, scores, calls))
                } else {
                    Err(format!("{n} batch codes for {} rows", v.n_rows()))
                }
            })
        })
        .map_err(PyValueError::new_err)?
    } else {
        return Err(PyTypeError::new_err(
            "scrublet reads a SparseMatrix or a DiskMatrix",
        ));
    };
    let out = PyList::empty(py);
    for r in results {
        out.append(batch_dict(py, r)?)?;
    }
    Ok(out)
}
