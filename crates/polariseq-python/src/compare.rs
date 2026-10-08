//! Comparing conditions: pseudobulk sums and tests, all in the core.
//!
//! The raw counts are summed per unit (replicate, condition, cell type) from
//! whichever source holds them: the in-memory matrix while it is still raw,
//! the on-disk store (its raw counts, under any recorded normalization), or,
//! for an in-memory analysis already normalized, the source files themselves.
//! Each cell type is then tested in parallel, and only the result table comes
//! back to Python; the sums cross only when asked for (`pseudobulk_sums`).

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use polariseq_core::preprocess::cluster_stats_blocks;
use polariseq_core::pseudobulk::{
    limma_trend, unit_sums_blocks, welch_compare, CellStats, PseudobulkOptions, UnitSums,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::disk::PyDiskMatrix;
use crate::sparse::PySparseMatrix;

/// The raw counts summed per unit, from a matrix or from files.
#[allow(clippy::too_many_arguments)]
fn sums_of(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    unit: &[u32],
    n_units: usize,
    gene_maps: Option<Vec<PyReadonlyArray1<'_, u32>>>,
    n_cols: Option<usize>,
    block_rows: usize,
) -> PyResult<UnitSums> {
    if let Ok(m) = source.cast::<PySparseMatrix>() {
        let m = m.borrow();
        let inner = &m.inner;
        if inner.nrows != unit.len() {
            return Err(PyValueError::new_err(format!(
                "{} units for {} rows",
                unit.len(),
                inner.nrows
            )));
        }
        return Ok(py.detach(|| unit_sums_blocks(inner, unit, n_units)));
    }
    if let Ok(d) = source.cast::<PyDiskMatrix>() {
        let d = d.borrow();
        let dm: &PyDiskMatrix = &d;
        return py
            .detach(|| {
                dm.raw_view(|v| {
                    if v.n_rows() == unit.len() {
                        Ok(unit_sums_blocks(v, unit, n_units))
                    } else {
                        Err(format!("{} units for {} rows", unit.len(), v.n_rows()))
                    }
                })
            })
            .map_err(PyValueError::new_err);
    }
    if let Ok(paths) = source.extract::<Vec<String>>() {
        let maps: Vec<Vec<u32>> = gene_maps
            .ok_or_else(|| PyValueError::new_err("reading files needs gene_maps"))?
            .iter()
            .map(|m| m.as_slice().map(<[u32]>::to_vec))
            .collect::<Result<_, _>>()?;
        let n_cols = n_cols.ok_or_else(|| PyValueError::new_err("reading files needs n_cols"))?;
        return py
            .detach(|| {
                polariseq_io::merge::unit_sums(&paths, &maps, n_cols, unit, n_units, block_rows)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }
    Err(PyTypeError::new_err(
        "the counts come from a SparseMatrix, a DiskMatrix or a list of files",
    ))
}

/// Keep the given columns of the sums, in that order.
fn select_cols(s: UnitSums, cols: &[u32]) -> UnitSums {
    let g = s.n_genes;
    let k = cols.len();
    let mut sums = vec![0.0; s.n_units * k];
    let mut expressing = vec![0; s.n_units * k];
    for u in 0..s.n_units {
        for (t, &c) in cols.iter().enumerate() {
            sums[u * k + t] = s.sums[u * g + c as usize];
            expressing[u * k + t] = s.expressing[u * g + c as usize];
        }
    }
    UnitSums {
        n_units: s.n_units,
        n_genes: k,
        sums,
        expressing,
        cells: s.cells,
    }
}

/// Raw counts summed per unit: `{"sums": (units × genes), "expressing":
/// (units × genes, cells with the gene), "cells": (units,)}`.
#[pyfunction]
#[pyo3(signature = (source, unit, n_units, *, cols=None, gene_maps=None, n_cols=None, block_rows=16_384))]
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_sums<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    unit: PyReadonlyArray1<'py, u32>,
    n_units: usize,
    cols: Option<PyReadonlyArray1<'py, u32>>,
    gene_maps: Option<Vec<PyReadonlyArray1<'py, u32>>>,
    n_cols: Option<usize>,
    block_rows: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let unit = unit.as_slice()?;
    let mut s = sums_of(py, source, unit, n_units, gene_maps, n_cols, block_rows)?;
    if let Some(c) = cols {
        s = select_cols(s, c.as_slice()?);
    }
    let d = PyDict::new(py);
    let (u, g) = (s.n_units, s.n_genes);
    d.set_item("sums", s.sums.into_pyarray(py).reshape([u, g])?)?;
    d.set_item("expressing", s.expressing.into_pyarray(py).reshape([u, g])?)?;
    d.set_item("cells", s.cells.into_pyarray(py))?;
    Ok(d)
}

/// One cell type's results, flattened.
#[derive(Default)]
struct Rows {
    group: Vec<u32>,
    condition: Vec<u32>,
    gene: Vec<u32>,
    log_fc: Vec<f64>,
    ave_expr: Vec<f64>,
    mean_reference: Vec<f64>,
    mean_condition: Vec<f64>,
    pct_reference: Vec<f64>,
    pct_condition: Vec<f64>,
    t: Vec<f64>,
    p_value: Vec<f64>,
    adj_p_value: Vec<f64>,
}

impl Rows {
    fn extend(&mut self, o: Self) {
        self.group.extend(o.group);
        self.condition.extend(o.condition);
        self.gene.extend(o.gene);
        self.log_fc.extend(o.log_fc);
        self.ave_expr.extend(o.ave_expr);
        self.mean_reference.extend(o.mean_reference);
        self.mean_condition.extend(o.mean_condition);
        self.pct_reference.extend(o.pct_reference);
        self.pct_condition.extend(o.pct_condition);
        self.t.extend(o.t);
        self.p_value.extend(o.p_value);
        self.adj_p_value.extend(o.adj_p_value);
    }

    fn into_dict<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("group", self.group.into_pyarray(py))?;
        d.set_item("condition", self.condition.into_pyarray(py))?;
        d.set_item("gene", self.gene.into_pyarray(py))?;
        d.set_item("log_fc", self.log_fc.into_pyarray(py))?;
        d.set_item("ave_expr", self.ave_expr.into_pyarray(py))?;
        d.set_item("mean_reference", self.mean_reference.into_pyarray(py))?;
        d.set_item("mean_condition", self.mean_condition.into_pyarray(py))?;
        d.set_item("pct_reference", self.pct_reference.into_pyarray(py))?;
        d.set_item("pct_condition", self.pct_condition.into_pyarray(py))?;
        d.set_item("t", self.t.into_pyarray(py))?;
        d.set_item("p_value", self.p_value.into_pyarray(py))?;
        d.set_item("adj_p_value", self.adj_p_value.into_pyarray(py))?;
        Ok(d)
    }
}

/// What testing one cell type found.
struct GroupInfo {
    group: u32,
    samples: usize,
    paired: bool,
    df_residual: f64,
    df_prior: f64,
    note: Option<String>,
}

/// Pseudobulk comparison of conditions within each cell type: sums per unit,
/// then limma-trend per cell type, in parallel. `unit_group`,
/// `unit_condition` and `unit_replicate` describe each unit; units with fewer
/// than `min_cells` cells take no part. Returns the result rows (one per
/// cell type, condition and gene tested) and one summary per cell type.
#[pyfunction]
#[pyo3(signature = (source, unit, n_units, unit_group, unit_condition, unit_replicate, reference,
    *, min_cells=10, cols=None, gene_maps=None, n_cols=None, block_rows=16_384, min_count=10.0,
    min_total_count=15.0, large_n=10.0, min_prop=0.7, prior_count=3.0))]
#[allow(clippy::too_many_arguments)]
pub fn compare_pseudobulk<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    unit: PyReadonlyArray1<'py, u32>,
    n_units: usize,
    unit_group: PyReadonlyArray1<'py, u32>,
    unit_condition: PyReadonlyArray1<'py, u32>,
    unit_replicate: PyReadonlyArray1<'py, u32>,
    reference: u32,
    min_cells: u32,
    cols: Option<PyReadonlyArray1<'py, u32>>,
    gene_maps: Option<Vec<PyReadonlyArray1<'py, u32>>>,
    n_cols: Option<usize>,
    block_rows: usize,
    min_count: f64,
    min_total_count: f64,
    large_n: f64,
    min_prop: f64,
    prior_count: f64,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
    let unit = unit.as_slice()?;
    let (ug, uc, ur) = (
        unit_group.as_slice()?,
        unit_condition.as_slice()?,
        unit_replicate.as_slice()?,
    );
    if ug.len() != n_units || uc.len() != n_units || ur.len() != n_units {
        return Err(PyValueError::new_err(
            "one group, condition and replicate per unit",
        ));
    }
    let mut s = sums_of(py, source, unit, n_units, gene_maps, n_cols, block_rows)?;
    if let Some(c) = cols {
        s = select_cols(s, c.as_slice()?);
    }
    let opts = PseudobulkOptions {
        min_count,
        min_total_count,
        large_n,
        min_prop,
        prior_count,
    };
    let mut groups: Vec<u32> = ug.to_vec();
    groups.sort_unstable();
    groups.dedup();
    let g = s.n_genes;
    let per: Vec<(Rows, GroupInfo)> = py.detach(|| {
        groups
            .par_iter()
            .map(|&grp| {
                let units: Vec<usize> = (0..n_units)
                    .filter(|&u| ug[u] == grp && s.cells[u] >= min_cells)
                    .collect();
                let mut info = GroupInfo {
                    group: grp,
                    samples: units.len(),
                    paired: false,
                    df_residual: f64::NAN,
                    df_prior: f64::NAN,
                    note: None,
                };
                let y: Vec<f64> = units
                    .iter()
                    .flat_map(|&u| s.sums[u * g..(u + 1) * g].iter().copied())
                    .collect();
                let cond: Vec<u32> = units.iter().map(|&u| uc[u]).collect();
                let rep: Vec<u32> = units.iter().map(|&u| ur[u]).collect();
                let mut rows = Rows::default();
                match limma_trend(&y, units.len(), g, &cond, reference, &rep, &opts) {
                    Err(e) => info.note = Some(e),
                    Ok(fit) => {
                        info.paired = fit.paired;
                        info.df_residual = fit.df_residual;
                        info.df_prior = fit.df_prior;
                        // Share of cells expressing each gene, per condition.
                        let pct = |code: u32| -> Vec<f64> {
                            let us: Vec<usize> =
                                units.iter().copied().filter(|&u| uc[u] == code).collect();
                            let cells: f64 = us.iter().map(|&u| f64::from(s.cells[u])).sum();
                            (0..g)
                                .map(|j| {
                                    us.iter()
                                        .map(|&u| f64::from(s.expressing[u * g + j]))
                                        .sum::<f64>()
                                        / cells
                                })
                                .collect()
                        };
                        let pct_ref = pct(reference);
                        for c in fit.comparisons {
                            let pct_c = pct(c.condition);
                            let n = c.genes.len();
                            rows.group.extend(std::iter::repeat_n(grp, n));
                            rows.condition.extend(std::iter::repeat_n(c.condition, n));
                            rows.pct_reference
                                .extend(c.genes.iter().map(|&j| pct_ref[j as usize]));
                            rows.pct_condition
                                .extend(c.genes.iter().map(|&j| pct_c[j as usize]));
                            rows.gene.extend(c.genes);
                            rows.log_fc.extend(c.log_fc);
                            rows.ave_expr.extend(c.ave_expr);
                            rows.mean_reference.extend(c.mean_reference);
                            rows.mean_condition.extend(c.mean_condition);
                            rows.t.extend(c.t);
                            rows.p_value.extend(c.p_value);
                            rows.adj_p_value.extend(c.adj_p_value);
                        }
                    }
                }
                (rows, info)
            })
            .collect()
    });
    let mut all = Rows::default();
    let infos = PyList::empty(py);
    for (rows, info) in per {
        all.extend(rows);
        let d = PyDict::new(py);
        d.set_item("group", info.group)?;
        d.set_item("samples", info.samples)?;
        d.set_item("paired", info.paired)?;
        d.set_item("df_residual", info.df_residual)?;
        d.set_item("df_prior", info.df_prior)?;
        d.set_item("note", info.note)?;
        infos.append(d)?;
    }
    Ok((all.into_dict(py)?, infos))
}

/// Mean of the current matrix per group of cells: `n_groups x genes`, row-major
/// (`label[i]` the group of cell `i`, or `u32::MAX` to leave it out), in one
/// streamed pass in memory or out of core. Used for groups' expression profiles.
#[pyfunction]
pub fn group_means<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    label: PyReadonlyArray1<'py, u32>,
    n_groups: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let label = label.as_slice()?;
    let stats = if let Ok(m) = matrix.cast::<PySparseMatrix>() {
        let m = m.borrow();
        let inner = &m.inner;
        py.detach(|| cluster_stats_blocks(inner, label, n_groups))
    } else if let Ok(d) = matrix.cast::<PyDiskMatrix>() {
        let d = d.borrow();
        let dm: &PyDiskMatrix = &d;
        py.detach(|| dm.current_view(|v| cluster_stats_blocks(v, label, n_groups)))
    } else {
        return Err(PyTypeError::new_err(
            "group_means reads a SparseMatrix or a DiskMatrix",
        ));
    };
    let (g, means) = (stats.n_genes, stats.means());
    // stats are per (gene, group); return groups as rows
    let mut out = vec![0.0_f32; n_groups * g];
    for j in 0..g {
        for l in 0..n_groups {
            out[l * g + j] = means[j * n_groups + l];
        }
    }
    crate::flat_to_pyarray2(py, out, n_groups, g)
}

/// Cell-level comparison: Welch's t-test of each condition's cells against the
/// reference's, within each group, on the matrix's current (normalized)
/// values. `label[i]` is `group * n_conditions + condition` or `u32::MAX`.
#[pyfunction]
#[pyo3(signature = (matrix, label, n_groups, n_conditions, reference, *, min_cells=3))]
pub fn compare_cells<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    label: PyReadonlyArray1<'py, u32>,
    n_groups: usize,
    n_conditions: usize,
    reference: u32,
    min_cells: u32,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
    let label = label.as_slice()?;
    let n_labels = n_groups * n_conditions;
    let stats = if let Ok(m) = matrix.cast::<PySparseMatrix>() {
        let m = m.borrow();
        let inner = &m.inner;
        py.detach(|| cluster_stats_blocks(inner, label, n_labels))
    } else if let Ok(d) = matrix.cast::<PyDiskMatrix>() {
        let d = d.borrow();
        let dm: &PyDiskMatrix = &d;
        py.detach(|| dm.current_view(|v| cluster_stats_blocks(v, label, n_labels)))
    } else {
        return Err(PyTypeError::new_err(
            "compare_cells reads a SparseMatrix or a DiskMatrix",
        ));
    };
    let g = stats.n_genes;
    let column =
        |v: &[f64], l: usize| -> Vec<f64> { (0..g).map(|j| v[j * n_labels + l]).collect() };
    let columnu =
        |v: &[u32], l: usize| -> Vec<u32> { (0..g).map(|j| v[j * n_labels + l]).collect() };
    let mut all = Rows::default();
    let infos = PyList::empty(py);
    for grp in 0..n_groups {
        let lr = grp * n_conditions + reference as usize;
        let nr = stats.cluster_sizes[lr];
        let (sr, qr, zr) = (
            column(&stats.sums, lr),
            column(&stats.sq_sums, lr),
            columnu(&stats.counts, lr),
        );
        for c in (0..n_conditions).filter(|&c| c != reference as usize) {
            let lc = grp * n_conditions + c;
            let nc = stats.cluster_sizes[lc];
            let d = PyDict::new(py);
            d.set_item("group", grp)?;
            d.set_item("condition", c)?;
            d.set_item("cells_reference", nr)?;
            d.set_item("cells_condition", nc)?;
            if nr < min_cells || nc < min_cells {
                d.set_item("note", format!("{nr} reference and {nc} condition cells; at least {min_cells} each are needed"))?;
                infos.append(d)?;
                continue;
            }
            d.set_item("note", py.None())?;
            infos.append(d)?;
            let (sc, qc, zc) = (
                column(&stats.sums, lc),
                column(&stats.sq_sums, lc),
                columnu(&stats.counts, lc),
            );
            let r = welch_compare(
                &CellStats {
                    n: f64::from(nc),
                    sum: &sc,
                    sq: &qc,
                    nonzero: &zc,
                },
                &CellStats {
                    n: f64::from(nr),
                    sum: &sr,
                    sq: &qr,
                    nonzero: &zr,
                },
            );
            all.group.extend(std::iter::repeat_n(grp as u32, g));
            all.condition.extend(std::iter::repeat_n(c as u32, g));
            all.gene.extend(0..g as u32);
            all.ave_expr.extend(
                r.mean_condition
                    .iter()
                    .zip(&r.mean_reference)
                    .map(|(a, b)| (a + b) / 2.0),
            );
            all.log_fc.extend(r.log_fc);
            all.mean_reference.extend(r.mean_reference);
            all.mean_condition.extend(r.mean_condition);
            all.pct_reference.extend(r.pct_reference);
            all.pct_condition.extend(r.pct_condition);
            all.t.extend(r.t);
            all.p_value.extend(r.p_value);
            all.adj_p_value.extend(r.adj_p_value);
        }
    }
    Ok((all.into_dict(py)?, infos))
}
