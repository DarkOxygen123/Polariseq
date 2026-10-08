//! Preprocessing kernels for single-cell count matrices.
//!
//! The canonical pipeline before PCA is QC → filter → `normalize_total` →
//! `log1p` → highly-variable genes. Every kernel here works on the CSR matrix
//! in place or in a single parallel pass, and none of them allocate anything
//! proportional to `nnz` beyond the matrix itself:
//!
//! - [`qc_metrics`] — per-cell and per-gene QC statistics in one pass.
//! - [`normalize_total_inplace`] / [`scale_rows_inplace`] — per-cell scaling.
//! - [`log1p_inplace`] — `ln(1 + x)` on the stored values (sparsity preserved).
//! - [`gene_mean_var`] + [`hvg_seurat`] — **the** highly-variable-gene
//!   selection, following scanpy's `flavor="seurat"` step for step so that
//!   HVG sets agree with the reference implementation.
//!
//! Scanpy is BSD-3-Clause (Copyright (c) 2025 scverse, Copyright (c) 2017
//! F. Alexander Wolf, P. Angerer, Theis Lab); `THIRD_PARTY_NOTICES.md` keeps
//! its notice.

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use crate::matrix::{split_rows_mut, CsrMatrix, ROW_BLOCK};
use crate::rowblocks::{BlockFold, RowBlocks};
use crate::rsvd::{chunk_rows, CsrRef};
use rayon::prelude::*;
use std::sync::Mutex;

/// Per-cell and per-gene QC statistics (scanpy `pp.calculate_qc_metrics`).
#[derive(Debug, Clone, PartialEq)]
pub struct QcMetrics {
    /// Total counts per cell.
    pub n_counts: Vec<f32>,
    /// Number of genes with a positive value per cell.
    pub n_genes: Vec<u32>,
    /// Counts from mitochondrial genes per cell (`None` if no mask given).
    pub mt_counts: Option<Vec<f32>>,
    /// Number of cells with a positive value per gene.
    pub gene_n_cells: Vec<u32>,
    /// Total counts per gene.
    pub gene_total_counts: Vec<f64>,
}

/// `(block_idx, n_counts, n_genes, mt_counts)` for one block of rows.
type CellBlock = (usize, Vec<f32>, Vec<u32>, Vec<f32>);

/// Compute QC metrics in one parallel pass over the matrix (blocks of
/// [`ROW_BLOCK`] rows via [`RowBlocks`]). Gene accumulators fold in block
/// order — for count data (integers in `f64`) the result is exactly the
/// same as any other summation order; cell outputs are per-row and exact.
///
/// `mito` is an optional per-gene mask of mitochondrial genes. A value counts
/// as "expressed" when it is `> 0`, matching scanpy's `X > 0`.
///
/// # Panics
/// Panics if a gene index in the matrix exceeds `mito.len()` when `mito` is
/// provided (cannot happen for a mask built from `var_names`).
#[must_use]
pub fn qc_metrics(m: &CsrMatrix, mito: Option<&[bool]>) -> QcMetrics {
    qc_metrics_blocks(m, mito)
}

/// [`qc_metrics`] over any [`RowBlocks`] source — in-RAM, compressed,
/// memory-mapped or streamed from disk. Block partials fold in block order,
/// so the result is identical whichever source (and whichever thread count)
/// produced it.
///
/// # Panics
/// Panics if a gene index exceeds `mito.len()` when `mito` is provided.
#[must_use]
pub fn qc_metrics_blocks(src: &dyn RowBlocks, mito: Option<&[bool]>) -> QcMetrics {
    let (nrows, ncols) = (src.n_rows(), src.n_cols());

    // Gene accumulators: (n_cells, total) folded in block order.
    let genes: BlockFold<(Vec<u32>, Vec<f64>)> = BlockFold::new(|(a_n, a_t), (b_n, b_t)| {
        for (a, b) in a_n.iter_mut().zip(&b_n) {
            *a += b;
        }
        for (a, b) in a_t.iter_mut().zip(&b_t) {
            *a += b;
        }
    });
    // Per-cell outputs, collected per block and stitched in block order.
    let cells: Mutex<Vec<CellBlock>> = Mutex::new(Vec::new());

    src.for_each_block(ROW_BLOCK, &|off, blk| {
        let mut n_counts = Vec::with_capacity(blk.n_rows);
        let mut n_genes = Vec::with_capacity(blk.n_rows);
        let mut mt = Vec::with_capacity(if mito.is_some() { blk.n_rows } else { 0 });
        let mut gene_n = vec![0_u32; ncols];
        let mut gene_total = vec![0.0_f64; ncols];
        for i in 0..blk.n_rows {
            let (s, e) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            let mut sum = 0.0_f64;
            let mut expressed = 0_u32;
            let mut mt_sum = 0.0_f64;
            for (&c, &v) in blk.indices[s..e].iter().zip(&blk.data[s..e]) {
                let c = c as usize;
                let v64 = f64::from(v);
                sum += v64;
                gene_total[c] += v64;
                if v > 0.0 {
                    expressed += 1;
                    gene_n[c] += 1;
                }
                if let Some(mask) = mito {
                    if mask[c] {
                        mt_sum += v64;
                    }
                }
            }
            n_counts.push(sum as f32);
            n_genes.push(expressed);
            if mito.is_some() {
                mt.push(mt_sum as f32);
            }
        }
        genes.store(off, (gene_n, gene_total));
        cells
            .lock()
            .expect("qc cell blocks poisoned")
            .push((off, n_counts, n_genes, mt));
    });

    // Stitch per-block cell vectors back into row order.
    let mut blocks = cells.into_inner().expect("qc cell blocks poisoned");
    blocks.sort_unstable_by_key(|b| b.0);
    let mut n_counts = Vec::with_capacity(nrows);
    let mut n_genes = Vec::with_capacity(nrows);
    let mut mt_counts = mito.map(|_| Vec::with_capacity(nrows));
    for (_, nc, ng, mt) in blocks {
        n_counts.extend(nc);
        n_genes.extend(ng);
        if let Some(out) = mt_counts.as_mut() {
            out.extend(mt);
        }
    }
    let (gene_n_cells, gene_total_counts) =
        genes.finish((vec![0_u32; ncols], vec![0.0_f64; ncols]));

    QcMetrics {
        n_counts,
        n_genes,
        mt_counts,
        gene_n_cells,
        gene_total_counts,
    }
}

/// Per-row totals over the columns whose flag is true (every column when
/// `keep_cols` is `None`), summed in `f32` in stored order.
///
/// This is exactly what [`CsrMatrix::row_sums`] returns for the matrix with
/// the other columns removed, so a normalisation scale taken from these
/// totals equals `normalize_total` on a gene-filtered matrix, without
/// writing that matrix anywhere. Each row is summed by one thread, so the
/// result does not depend on the source or the thread count.
///
/// # Panics
/// Panics if a column index exceeds `keep_cols.len()` when it is given.
#[must_use]
pub fn row_totals_blocks(src: &dyn RowBlocks, keep_cols: Option<&[bool]>) -> Vec<f32> {
    let out = Mutex::new(vec![0.0_f32; src.n_rows()]);
    src.for_each_block(ROW_BLOCK, &|off, blk| {
        let mut local = vec![0.0_f32; blk.n_rows];
        for (i, total) in local.iter_mut().enumerate() {
            let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            *total = match keep_cols {
                None => blk.data[a..b].iter().sum(),
                Some(keep) => blk.indices[a..b]
                    .iter()
                    .zip(&blk.data[a..b])
                    .filter(|(&c, _)| keep[c as usize])
                    .map(|(_, &v)| v)
                    .sum(),
            };
        }
        out.lock().expect("row totals poisoned")[off..off + blk.n_rows].copy_from_slice(&local);
    });
    out.into_inner().expect("row totals poisoned")
}

/// Multiply every stored value in row `i` by `scale[i]`, in place.
/// Parallel over row blocks.
///
/// # Panics
/// Panics if `scale.len() != nrows`.
pub fn scale_rows_inplace(m: &mut CsrMatrix, scale: &[f32]) {
    assert_eq!(scale.len(), m.nrows, "one scale factor per row");
    let indptr = &m.indptr;
    let data = &mut m.data;
    split_rows_mut(indptr, data, ROW_BLOCK)
        .into_par_iter()
        .for_each(|(r0, r1, d)| {
            let mut off = 0;
            for r in r0..r1 {
                let n = (indptr[r + 1] - indptr[r]) as usize;
                let s = scale[r];
                for v in &mut d[off..off + n] {
                    *v *= s;
                }
                off += n;
            }
        });
}

/// Scale each cell so its total counts equal `target_sum` (scanpy
/// `pp.normalize_total`). Cells with zero counts are left untouched.
/// Returns the per-cell totals *before* scaling.
#[must_use]
pub fn normalize_total_inplace(m: &mut CsrMatrix, target_sum: f32) -> Vec<f32> {
    let totals = m.row_sums();
    let scale: Vec<f32> = totals
        .iter()
        .map(|&t| if t > 0.0 { target_sum / t } else { 1.0 })
        .collect();
    scale_rows_inplace(m, &scale);
    totals
}

/// Normalize each cell again over the genes the matrix now holds (Scarf's
/// normalization after gene selection), in place, from values that were
/// normalized over all genes: each value is turned back into a count,
/// `x = expm1(v) * back[i]` (or `v * back[i]` without `from_log`), with
/// `back[i]` = the cell's original total / the original target; then each cell
/// is scaled to `target_sum` over its remaining values and, with `log`,
/// log-transformed. No copy of the counts is needed. Cells whose remaining
/// total is zero are left at zero.
///
/// # Panics
/// Panics if `back.len() != nrows`.
pub fn renormalize_inplace(m: &mut CsrMatrix, back: &[f32], from_log: bool, target_sum: f32, log: bool) {
    assert_eq!(back.len(), m.nrows, "one back-scale per row");
    let indptr = &m.indptr;
    let data = &mut m.data;
    split_rows_mut(indptr, data, ROW_BLOCK)
        .into_par_iter()
        .for_each(|(r0, r1, d)| {
            let mut off = 0;
            for r in r0..r1 {
                let n = (indptr[r + 1] - indptr[r]) as usize;
                let row = &mut d[off..off + n];
                let b = f64::from(back[r]);
                let mut total = 0.0_f64;
                for v in row.iter_mut() {
                    let x = if from_log { f64::from(*v).exp_m1() } else { f64::from(*v) } * b;
                    *v = x as f32;
                    total += x;
                }
                let s = if total > 0.0 { f64::from(target_sum) / total } else { 0.0 };
                for v in row.iter_mut() {
                    let y = f64::from(*v) * s;
                    *v = if log { y.ln_1p() as f32 } else { y as f32 };
                }
                off += n;
            }
        });
}

/// Apply `ln(1 + x)` to every stored value in place. Sparsity is preserved
/// because `log1p(0) = 0`.
pub fn log1p_inplace(m: &mut CsrMatrix) {
    m.data.par_iter_mut().for_each(|v| *v = v.ln_1p());
}

/// Per-gene mean and variance over all cells (zeros included).
#[derive(Debug, Clone, PartialEq)]
pub struct GeneStats {
    /// Per-gene mean.
    pub means: Vec<f64>,
    /// Per-gene variance with `ddof = 1` (scanpy convention).
    pub variances: Vec<f64>,
    /// Number of cells the statistics were computed over.
    pub n_obs: usize,
}

/// Per-gene mean and variance in one parallel pass over row blocks, with
/// partials folded in block order (deterministic for any thread count —
/// the previous per-row fold/reduce was only exact for integer data).
/// With `expm1 = true` the statistics are computed on `expm1(x)` — what
/// scanpy's Seurat-flavour HVG does to undo `log1p` before measuring
/// dispersion.
#[must_use]
pub fn gene_mean_var(m: &CsrMatrix, expm1: bool) -> GeneStats {
    gene_mean_var_blocks(m, expm1)
}

/// [`gene_mean_var`] over any [`RowBlocks`] source. See
/// [`qc_metrics_blocks`] for the determinism contract.
#[must_use]
pub fn gene_mean_var_blocks(src: &dyn RowBlocks, expm1: bool) -> GeneStats {
    let (nrows, ncols) = (src.n_rows(), src.n_cols());
    let chunk = chunk_rows(nrows);
    let fold: BlockFold<(Vec<f64>, Vec<f64>)> = BlockFold::new(|(s1, q1), (s2, q2)| {
        for (a, b) in s1.iter_mut().zip(&s2) {
            *a += b;
        }
        for (a, b) in q1.iter_mut().zip(&q2) {
            *a += b;
        }
    });

    src.for_each_block(chunk, &|off, blk| {
        let mut s = vec![0.0_f64; ncols];
        let mut q = vec![0.0_f64; ncols];
        for i in 0..blk.n_rows {
            let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            for (&c, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                let x = if expm1 {
                    f64::from(v).exp_m1()
                } else {
                    f64::from(v)
                };
                s[c as usize] += x;
                q[c as usize] += x * x;
            }
        }
        fold.store(off, (s, q));
    });

    let (sums, sq_sums) = fold.finish((vec![0.0_f64; ncols], vec![0.0_f64; ncols]));
    let n = nrows as f64;
    let means: Vec<f64> = sums.iter().map(|&s| s / n).collect();
    let variances: Vec<f64> = sq_sums
        .iter()
        .zip(&means)
        .map(|(&q, &mean)| {
            if nrows < 2 {
                0.0
            } else {
                ((q / n - mean * mean) * n / (n - 1.0)).max(0.0)
            }
        })
        .collect();
    GeneStats {
        means,
        variances,
        n_obs: nrows,
    }
}

/// A [`RowBlocks`] source that applies `normalize_total` (+ optional `log1p`)
/// to each block **as it is read**, without materialising the result.
///
/// The out-of-core path used to normalise by mutating the memory-mapped
/// spill in place. That dirties every page of the matrix, so the whole thing
/// becomes resident and the "streaming" pass holds as much memory as an
/// in-memory one — measured on heart_100k: 1.14 GB of private memory for an
/// 892 MB matrix, unchanged by the block size. Transforming on the fly keeps
/// the backing pages clean (so the OS can evict them), removes a full
/// rewrite of the file, and is the only option once values are stored
/// narrowed (a `u8` count cannot hold a normalised float).
///
/// Values are `x · scale[row]`, then `ln(1+x)` if `log1p` is set — the same
/// arithmetic, in the same order, as
/// [`normalize_total_inplace`] followed by [`log1p_inplace`].
pub struct NormalizedView<'a> {
    inner: &'a dyn RowBlocks,
    scale: Vec<f32>,
    log1p: bool,
}

impl<'a> NormalizedView<'a> {
    /// Wrap `inner`, scaling row `i` by `scale[i]`.
    ///
    /// # Panics
    /// Panics if `scale` has fewer entries than the source has rows.
    #[must_use]
    pub fn new(inner: &'a dyn RowBlocks, scale: Vec<f32>, log1p: bool) -> Self {
        assert!(
            scale.len() >= inner.n_rows(),
            "one scale factor per row: got {}, need {}",
            scale.len(),
            inner.n_rows()
        );
        Self {
            inner,
            scale,
            log1p,
        }
    }

    /// The scale factors that take each row to `target_sum`, from per-row
    /// totals (rows totalling zero are left alone).
    #[must_use]
    pub fn scale_from_totals(totals: &[f32], target_sum: f32) -> Vec<f32> {
        totals
            .iter()
            .map(|&t| if t > 0.0 { target_sum / t } else { 1.0 })
            .collect()
    }
}

impl RowBlocks for NormalizedView<'_> {
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        self.inner.for_each_block(rows_per_block, &|off, blk| {
            // One transformed copy of *this block*, on this thread. The
            // source's own bound on block size is therefore the bound on
            // this too.
            let mut data = blk.data.to_vec();
            for i in 0..blk.n_rows {
                let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                let s = self.scale[off + i];
                if self.log1p {
                    for v in &mut data[a..b] {
                        *v = (*v * s).ln_1p();
                    }
                } else {
                    for v in &mut data[a..b] {
                        *v *= s;
                    }
                }
            }
            f(
                off,
                CsrRef {
                    data: &data,
                    indices: blk.indices,
                    indptr: blk.indptr,
                    n_rows: blk.n_rows,
                    n_cols: blk.n_cols,
                },
            );
        });
    }
}

/// A [`RowBlocks`] source that drops the rows whose flag is false.
///
/// This is scanpy's `filter_cells` for the out-of-core path: QC decides
/// which rows survive, and everything downstream (per-gene statistics, PCA)
/// must run on the survivors only — without rewriting the spill. Wrapping
/// the source does that: downstream kernels receive compacted blocks whose
/// offsets count *kept* rows, so they compute exactly what they would on a
/// pre-filtered matrix. The disk-backed pipeline without this kept every
/// cell and so answered a different question than the in-RAM pipeline on
/// the same parameters (found by the telemetry harness comparing the two
/// arms' outputs; chapter 09).
///
/// Compose it *outside* [`NormalizedView`], not inside: the scale vector is
/// indexed by original row, so the normaliser must see original offsets
/// while this adapter emits compacted ones.
pub struct FilteredRows<'a> {
    inner: &'a dyn RowBlocks,
    /// Cumulative kept rows before source row `i` (length `n_rows + 1`).
    kept_before: Vec<usize>,
}

impl<'a> FilteredRows<'a> {
    /// Wrap `inner`, keeping row `i` iff `keep[i]`.
    ///
    /// # Panics
    /// Panics if `keep` has fewer entries than the source has rows.
    #[must_use]
    pub fn new(inner: &'a dyn RowBlocks, keep: &[bool]) -> Self {
        assert!(
            keep.len() >= inner.n_rows(),
            "one keep flag per row: got {}, need {}",
            keep.len(),
            inner.n_rows()
        );
        let mut kept_before = Vec::with_capacity(inner.n_rows() + 1);
        kept_before.push(0);
        let mut n = 0;
        for &k in &keep[..inner.n_rows()] {
            n += usize::from(k);
            kept_before.push(n);
        }
        Self { inner, kept_before }
    }

    /// Number of rows kept.
    #[must_use]
    pub fn n_kept(&self) -> usize {
        self.kept_before.last().copied().unwrap_or(0)
    }
}

impl RowBlocks for FilteredRows<'_> {
    fn n_rows(&self) -> usize {
        self.n_kept()
    }

    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        self.inner.for_each_block(rows_per_block, &|off, blk| {
            let kept: Vec<usize> = (0..blk.n_rows)
                .filter(|&i| self.kept_before[off + i + 1] > self.kept_before[off + i])
                .collect();
            if kept.is_empty() {
                return;
            }
            // Compact the kept rows into one block: each row's slice is
            // contiguous within the source arrays, so this is a copy of
            // concatenations plus a rebuilt indptr.
            let nnz: usize = kept
                .iter()
                .map(|&i| (blk.indptr[i + 1] - blk.indptr[i]) as usize)
                .sum();
            let mut indptr = Vec::with_capacity(kept.len() + 1);
            let mut indices = Vec::with_capacity(nnz);
            let mut data = Vec::with_capacity(nnz);
            indptr.push(0);
            for &i in &kept {
                let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                indices.extend_from_slice(&blk.indices[a..b]);
                data.extend_from_slice(&blk.data[a..b]);
                indptr.push(indices.len() as u32);
            }
            f(
                self.kept_before[off],
                CsrRef {
                    data: &data,
                    indices: &indices,
                    indptr: &indptr,
                    n_rows: kept.len(),
                    n_cols: blk.n_cols,
                },
            );
        });
    }
}

/// A [`RowBlocks`] source that drops the columns whose flag is false and
/// renumbers the rest, as `X[:, mask]` does in memory.
///
/// This is `filter_genes` (and a gene subset) for the out-of-core path: the
/// kept columns keep their order and their values, so any kernel run over
/// the view computes exactly what it would on the column-subset matrix,
/// without the subset ever being written. Compose it innermost, beneath
/// [`NormalizedView`] and [`FilteredRows`].
pub struct FilteredCols<'a> {
    inner: &'a dyn RowBlocks,
    /// New index of each source column (`u32::MAX`: dropped).
    map: Vec<u32>,
    n_kept: usize,
}

impl<'a> FilteredCols<'a> {
    /// Wrap `inner`, keeping column `j` iff `keep[j]`.
    ///
    /// # Panics
    /// Panics if `keep` has fewer entries than the source has columns.
    #[must_use]
    pub fn new(inner: &'a dyn RowBlocks, keep: &[bool]) -> Self {
        assert!(
            keep.len() >= inner.n_cols(),
            "one keep flag per column: got {}, need {}",
            keep.len(),
            inner.n_cols()
        );
        let mut map = vec![u32::MAX; inner.n_cols()];
        let mut n = 0_u32;
        for (j, &k) in keep[..inner.n_cols()].iter().enumerate() {
            if k {
                map[j] = n;
                n += 1;
            }
        }
        Self {
            inner,
            map,
            n_kept: n as usize,
        }
    }
}

impl RowBlocks for FilteredCols<'_> {
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    fn n_cols(&self) -> usize {
        self.n_kept
    }

    fn for_each_block(&self, rows_per_block: usize, f: &(dyn Fn(usize, CsrRef<'_>) + Sync)) {
        self.inner.for_each_block(rows_per_block, &|off, blk| {
            let mut indptr = Vec::with_capacity(blk.n_rows + 1);
            let mut indices = Vec::with_capacity(blk.data.len());
            let mut data = Vec::with_capacity(blk.data.len());
            indptr.push(0_u32);
            for i in 0..blk.n_rows {
                let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                for (&c, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                    let m = self.map[c as usize];
                    if m != u32::MAX {
                        indices.push(m);
                        data.push(v);
                    }
                }
                indptr.push(indices.len() as u32);
            }
            f(
                off,
                CsrRef {
                    data: &data,
                    indices: &indices,
                    indptr: &indptr,
                    n_rows: blk.n_rows,
                    n_cols: self.n_kept,
                },
            );
        });
    }
}

/// Per-cluster, per-gene statistics from a single pass.
///
/// Layout is `[gene * n_clusters + cluster]` — genes are the slow axis, which
/// is what marker ranking and annotation scoring iterate over.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterStats {
    /// Summed expression per (gene, cluster).
    pub sums: Vec<f64>,
    /// Summed squares per (gene, cluster) — for within-cluster variance.
    pub sq_sums: Vec<f64>,
    /// Cells with a nonzero value per (gene, cluster).
    pub counts: Vec<u32>,
    /// Cells per cluster.
    pub cluster_sizes: Vec<u32>,
    /// Genes.
    pub n_genes: usize,
    /// Clusters.
    pub n_clusters: usize,
}

impl ClusterStats {
    /// Mean expression per (gene, cluster), zeros included.
    #[must_use]
    pub fn means(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.n_genes * self.n_clusters];
        for g in 0..self.n_genes {
            for c in 0..self.n_clusters {
                let i = g * self.n_clusters + c;
                out[i] = (self.sums[i] / f64::from(self.cluster_sizes[c].max(1))) as f32;
            }
        }
        out
    }
}

/// Per-cluster gene statistics over any [`RowBlocks`] source, in one pass.
///
/// This is **the** cluster-statistics kernel: marker ranking, annotation
/// scoring and per-cluster means all use it, whether the matrix is in RAM,
/// memory-mapped or streamed from disk. Partials fold in block order, so the
/// result does not depend on the thread count or the source.
///
/// `labels` holds one cluster id per row; ids `>= n_clusters` are ignored
/// (that is how a caller excludes cells).
///
/// # Panics
/// Panics if `labels.len()` is smaller than the number of rows.
#[must_use]
pub fn cluster_stats_blocks(
    src: &dyn RowBlocks,
    labels: &[u32],
    n_clusters: usize,
) -> ClusterStats {
    let (nrows, n_genes) = (src.n_rows(), src.n_cols());
    assert!(labels.len() >= nrows, "one cluster label per row");
    let width = n_genes * n_clusters;
    let chunk = chunk_rows(nrows);

    let fold: BlockFold<(Vec<f64>, Vec<f64>, Vec<u32>)> =
        BlockFold::new(|(s1, q1, c1), (s2, q2, c2)| {
            for (a, b) in s1.iter_mut().zip(&s2) {
                *a += b;
            }
            for (a, b) in q1.iter_mut().zip(&q2) {
                *a += b;
            }
            for (a, b) in c1.iter_mut().zip(&c2) {
                *a += b;
            }
        });

    src.for_each_block(chunk, &|off, blk| {
        let mut sums = vec![0.0_f64; width];
        let mut sq = vec![0.0_f64; width];
        let mut counts = vec![0_u32; width];
        for i in 0..blk.n_rows {
            let cl = labels[off + i] as usize;
            if cl >= n_clusters {
                continue;
            }
            let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            for (&c, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                let idx = c as usize * n_clusters + cl;
                let x = f64::from(v);
                sums[idx] += x;
                sq[idx] += x * x;
                counts[idx] += 1;
            }
        }
        fold.store(off, (sums, sq, counts));
    });

    let (sums, sq_sums, counts) = fold.finish((
        vec![0.0_f64; width],
        vec![0.0_f64; width],
        vec![0_u32; width],
    ));

    let mut cluster_sizes = vec![0_u32; n_clusters];
    for &l in labels.iter().take(nrows) {
        if (l as usize) < n_clusters {
            cluster_sizes[l as usize] += 1;
        }
    }

    ClusterStats {
        sums,
        sq_sums,
        counts,
        cluster_sizes,
        n_genes,
        n_clusters,
    }
}

/// Result of highly-variable gene selection.
#[derive(Debug, Clone, PartialEq)]
pub struct HvgResult {
    /// `true` for the selected genes.
    pub mask: Vec<bool>,
    /// `log1p(mean)` per gene — scanpy's `var["means"]` for this flavour.
    pub means: Vec<f32>,
    /// `log(var / mean)` per gene (`NaN` where dispersion is zero).
    pub dispersions: Vec<f32>,
    /// Dispersion z-scored within its mean bin.
    pub dispersions_norm: Vec<f32>,
    /// Number of genes selected.
    pub n_hvg: usize,
}

/// Bin `xs` exactly as `pd.cut(xs, bins=n_bins)` does (right-closed
/// intervals): edges are `np.linspace(min, max, n_bins + 1)` — equal width
/// `(max - min)/n_bins`, the last edge pinned to `max` — with only the
/// *first* edge nudged down by 0.1 % of the range so the minimum is
/// included. A value exactly on an interior edge belongs to the bin below it.
///
/// Dividing `(max - nudged_min)/n_bins` evenly instead — which widens every
/// bin by 0.1 % relative to pandas and drifts interior edges downward —
/// placed 207 of 30,471 heart_10k genes one bin higher than scanpy, perturbing
/// each bin's mean/std enough to reorder the HVG tail (Jaccard 0.984, with
/// genes ranked ~2000 by one implementation and ~100 by the other). With
/// pandas' edge rule the two implementations agree to
/// |Δ dispersions_norm| < 4e-7. See chapter 09.
fn pd_cut_bins(xs: &[f64], n_bins: usize) -> Vec<usize> {
    let (lo, hi) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    if hi <= lo {
        // Degenerate range (all values equal): pandas nudges the point out
        // by 0.1 % of |min| either way and every value lands in one bin; any
        // single bin is equivalent, so keep them together in bin 0.
        return vec![0; xs.len()];
    }
    let step = (hi - lo) / n_bins as f64;
    let mut edges: Vec<f64> = (0..n_bins).map(|k| lo + step * k as f64).collect();
    edges.push(hi); // np.linspace pins the endpoint exactly
    edges[0] -= (hi - lo) * 0.001;
    xs.iter()
        .map(|&x| {
            // first k with edges[k] >= x, minus one for the open left side
            edges
                .partition_point(|&e| e < x)
                .saturating_sub(1)
                .min(n_bins - 1)
        })
        .collect()
}

/// Select highly-variable genes exactly as scanpy's
/// `highly_variable_genes(flavor="seurat", n_top_genes=...)` does:
///
/// 1. `mean`, `var` of `expm1(X)` (pass `gene_mean_var(m, true)`).
/// 2. `dispersion = var / mean`; zero dispersion → `NaN`; take logs.
/// 3. Bin `log1p(mean)` into `n_bins` equal-width bins (`pd.cut`).
/// 4. Within each bin z-score the dispersion (`ddof = 1`); a bin with a
///    single gene gets a normalised dispersion of 1.
/// 5. Keep the `n_top_genes` genes with the largest normalised dispersion.
///
/// One deliberate difference: scanpy keeps *every* gene tied at the cut-off,
/// which can return more than `n_top_genes`; we break ties by gene index so
/// the count is exact.
#[must_use]
pub fn hvg_seurat(stats: &GeneStats, n_top_genes: usize, n_bins: usize) -> HvgResult {
    let n_genes = stats.means.len();
    let n_bins = n_bins.max(1);

    // Steps 1–2.
    let mut mean_log = vec![0.0_f64; n_genes];
    let mut disp_log = vec![f64::NAN; n_genes];
    for g in 0..n_genes {
        let mean = if stats.means[g] == 0.0 {
            1e-12
        } else {
            stats.means[g]
        };
        let disp = stats.variances[g] / mean;
        if disp > 0.0 {
            disp_log[g] = disp.ln();
        }
        mean_log[g] = mean.ln_1p();
    }

    // Step 3: `pd_cut_bins` — pandas' edge rule exactly (see its comment for
    // why anything else reorders the HVG tail against scanpy).
    let bins = pd_cut_bins(&mean_log, n_bins);

    // Step 4: per-bin mean and std of the (finite) log dispersions.
    let mut bin_sum = vec![0.0_f64; n_bins];
    let mut bin_sq = vec![0.0_f64; n_bins];
    let mut bin_n = vec![0_usize; n_bins];
    for g in 0..n_genes {
        let d = disp_log[g];
        if d.is_finite() {
            let b = bins[g];
            bin_sum[b] += d;
            bin_sq[b] += d * d;
            bin_n[b] += 1;
        }
    }
    let mut dnorm = vec![f32::NAN; n_genes];
    for g in 0..n_genes {
        let d = disp_log[g];
        if !d.is_finite() {
            continue;
        }
        let b = bins[g];
        let n = bin_n[b] as f64;
        dnorm[g] = if bin_n[b] < 2 {
            1.0
        } else {
            let mean = bin_sum[b] / n;
            let var = ((bin_sq[b] / n - mean * mean) * n / (n - 1.0)).max(0.0);
            let std = var.sqrt();
            if std < 1e-12 {
                0.0
            } else {
                ((d - mean) / std) as f32
            }
        };
    }

    // Step 5: top-n by normalised dispersion, NaN last, ties by gene index.
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| {
        let (da, db) = (dnorm[a], dnorm[b]);
        match (da.is_nan(), db.is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => db
                .partial_cmp(&da)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b)),
        }
    });
    let n_top = n_top_genes.min(n_genes);
    let mut mask = vec![false; n_genes];
    let mut n_hvg = 0;
    for &g in order.iter().take(n_top) {
        if dnorm[g].is_nan() {
            break; // never select a gene without a dispersion
        }
        mask[g] = true;
        n_hvg += 1;
    }

    HvgResult {
        mask,
        means: mean_log.iter().map(|&x| x as f32).collect(),
        dispersions: disp_log.iter().map(|&x| x as f32).collect(),
        dispersions_norm: dnorm,
        n_hvg,
    }
}

/// Per-gene mean, variance and the number of cells with a positive value, in
/// one pass (the statistics of [`gene_mean_var_blocks`] plus detection, which
/// [`hvg_scarf`] needs). Folded in block order: identical for any thread count.
#[must_use]
pub fn gene_stats_detected_blocks(src: &dyn RowBlocks, expm1: bool) -> (GeneStats, Vec<u64>) {
    type Part = (Vec<f64>, Vec<f64>, Vec<u64>);
    let (nrows, ncols) = (src.n_rows(), src.n_cols());
    let chunk = chunk_rows(nrows);
    let fold: BlockFold<Part> = BlockFold::new(|(s1, q1, c1), (s2, q2, c2)| {
        for (a, b) in s1.iter_mut().zip(&s2) {
            *a += b;
        }
        for (a, b) in q1.iter_mut().zip(&q2) {
            *a += b;
        }
        for (a, b) in c1.iter_mut().zip(&c2) {
            *a += b;
        }
    });
    src.for_each_block(chunk, &|off, blk| {
        let mut s = vec![0.0_f64; ncols];
        let mut q = vec![0.0_f64; ncols];
        let mut c = vec![0_u64; ncols];
        for i in 0..blk.n_rows {
            let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            for (&col, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                let x = if expm1 { f64::from(v).exp_m1() } else { f64::from(v) };
                let g = col as usize;
                s[g] += x;
                q[g] += x * x;
                if x > 0.0 {
                    c[g] += 1;
                }
            }
        }
        fold.store(off, (s, q, c));
    });
    let (sums, sq_sums, detected) =
        fold.finish((vec![0.0_f64; ncols], vec![0.0_f64; ncols], vec![0_u64; ncols]));
    let n = nrows as f64;
    let means: Vec<f64> = sums.iter().map(|&s| s / n).collect();
    let variances: Vec<f64> = sq_sums
        .iter()
        .zip(&means)
        .map(|(&q, &mean)| if nrows < 2 { 0.0 } else { ((q / n - mean * mean) * n / (n - 1.0)).max(0.0) })
        .collect();
    (GeneStats { means, variances, n_obs: nrows }, detected)
}

/// [`hvg_seurat`] among the genes detected in more than `min_cells` cells: the
/// genes below the floor are left out before the mean bins and the dispersion
/// z-scores are computed, as if they were not in the data, and are never
/// selected (their outputs are `NaN`). With `min_cells = 0` this is
/// [`hvg_seurat`] itself.
///
/// # Panics
/// Panics if `detected` does not hold one count per gene.
#[must_use]
pub fn hvg_seurat_floor(
    stats: &GeneStats,
    detected: &[u64],
    n_top_genes: usize,
    n_bins: usize,
    min_cells: u64,
) -> HvgResult {
    let g = stats.means.len();
    assert_eq!(detected.len(), g, "one detection count per gene");
    let keep: Vec<usize> = (0..g).filter(|&i| detected[i] > min_cells).collect();
    let sub = GeneStats {
        means: keep.iter().map(|&i| stats.means[i]).collect(),
        variances: keep.iter().map(|&i| stats.variances[i]).collect(),
        n_obs: stats.n_obs,
    };
    let r = hvg_seurat(&sub, n_top_genes, n_bins);
    let mut out = HvgResult {
        mask: vec![false; g],
        means: vec![f32::NAN; g],
        dispersions: vec![f32::NAN; g],
        dispersions_norm: vec![f32::NAN; g],
        n_hvg: r.n_hvg,
    };
    for (j, &i) in keep.iter().enumerate() {
        out.mask[i] = r.mask[j];
        out.means[i] = r.means[j];
        out.dispersions[i] = r.dispersions[j];
        out.dispersions_norm[i] = r.dispersions_norm[j];
    }
    out
}

/// Result of [`hvg_scarf`].
#[derive(Debug, Clone, PartialEq)]
pub struct HvgScarf {
    /// `true` for the selected genes.
    pub mask: Vec<bool>,
    /// Natural log of each gene's mean (`NaN` where the mean is zero).
    pub log_mean: Vec<f32>,
    /// Natural log of each gene's variance (`NaN` where it is zero).
    pub log_var: Vec<f32>,
    /// Variance over the fitted minimum-variance trend (0 where undefined).
    pub corrected_var: Vec<f32>,
    /// Cells in which each gene is detected.
    pub detected: Vec<u64>,
    /// The detection floor applied (a gene must be detected in more cells).
    pub min_cells: u64,
    /// Number of genes selected.
    pub n_hvg: usize,
}

/// Highly variable genes as Scarf's `mark_hvgs` selects them (Dhapola et al.,
/// Nat. Commun. 2022; Scarf is BSD-3-Clause, see THIRD_PARTY_NOTICES.md):
///
/// 1. Per gene, the mean and variance of library-size-normalized counts
///    (pass [`gene_stats_detected_blocks`] on the normalized matrix, with
///    `expm1` when it is log-transformed). The scale of the normalization does
///    not matter: it shifts every log mean and log variance alike, and the
///    steps below are unchanged by such a shift.
/// 2. Genes with a positive mean are binned by log mean into `n_bins` equal
///    bins (left-closed; the top edge widened by 0.1 so the largest is in).
/// 3. In each bin the gene with the smallest log variance is taken, and a
///    LOWESS curve (`frac`, 100 robust passes) is fitted to these floor genes.
/// 4. A gene's corrected variance is its variance over the curve's value for
///    its bin.
/// 5. Among genes detected in more than `min_cells` cells, the `n_top_genes`
///    with the largest corrected variance are kept (ties by gene index, so the
///    count is exact; Scarf keeps those strictly above the `n_top_genes`-th
///    value). Scarf's default floor is 1% of the cells.
///
/// # Panics
/// Panics if `detected` does not hold one count per gene.
#[must_use]
pub fn hvg_scarf(
    stats: &GeneStats,
    detected: &[u64],
    n_top_genes: usize,
    n_bins: usize,
    frac: f64,
    min_cells: u64,
) -> HvgScarf {
    let g = stats.means.len();
    assert_eq!(detected.len(), g, "one detection count per gene");
    let n_bins = n_bins.max(1);
    let ok: Vec<usize> = (0..g)
        .filter(|&i| stats.means[i] > 0.0 && stats.variances[i] > 0.0)
        .collect();
    let la: Vec<f64> = ok.iter().map(|&i| stats.means[i].ln()).collect();
    let lb: Vec<f64> = ok.iter().map(|&i| stats.variances[i].ln()).collect();
    let mut corrected = vec![0.0_f64; g];
    if !ok.is_empty() {
        // numpy.histogram edges, then Scarf's widening of the top edge
        let lo = la.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = la.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (lo, hi) = if hi > lo { (lo, hi) } else { (lo - 0.5, hi + 0.5) };
        let step = (hi - lo) / n_bins as f64;
        let mut edges: Vec<f64> = (0..=n_bins).map(|k| lo + step * k as f64).collect();
        edges[n_bins] = hi + 0.1;
        let bin_of: Vec<usize> = la
            .iter()
            .map(|&v| edges.partition_point(|&e| e <= v).saturating_sub(1).min(n_bins - 1))
            .collect();
        // the least variable gene of each non-empty bin (first one on ties)
        let mut floor: Vec<Option<usize>> = vec![None; n_bins];
        for (j, &b) in bin_of.iter().enumerate() {
            match floor[b] {
                Some(f) if lb[f] <= lb[j] => {}
                _ => floor[b] = Some(j),
            }
        }
        let bins: Vec<usize> = (0..n_bins).filter(|&b| floor[b].is_some()).collect();
        let fx: Vec<f64> = bins.iter().map(|&b| la[floor[b].unwrap()]).collect();
        let fy: Vec<f64> = bins.iter().map(|&b| lb[floor[b].unwrap()]).collect();
        let fit = crate::lowess::lowess(&fx, &fy, frac, 100);
        let mut trend = vec![f64::NAN; n_bins];
        for (k, &b) in bins.iter().enumerate() {
            trend[b] = fit[k];
        }
        for (j, &i) in ok.iter().enumerate() {
            corrected[i] = (lb[j] - trend[bin_of[j]]).exp();
        }
    }
    let mut order: Vec<usize> = (0..g).filter(|&i| detected[i] > min_cells && corrected[i] > 0.0).collect();
    order.sort_by(|&a, &b| {
        corrected[b]
            .partial_cmp(&corrected[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut mask = vec![false; g];
    let n_hvg = n_top_genes.min(order.len());
    for &i in order.iter().take(n_hvg) {
        mask[i] = true;
    }
    let ln_or_nan = |v: f64| if v > 0.0 { v.ln() as f32 } else { f32::NAN };
    HvgScarf {
        mask,
        log_mean: stats.means.iter().map(|&v| ln_or_nan(v)).collect(),
        log_var: stats.variances.iter().map(|&v| ln_or_nan(v)).collect(),
        corrected_var: corrected.iter().map(|&v| v as f32).collect(),
        detected: detected.to_vec(),
        min_cells,
        n_hvg,
    }
}

// ── legacy oracles: the pre-RowBlocks kernel bodies, verbatim.
#[cfg(test)]
mod legacy {
    use super::*;
    use crate::matrix::{CsrMatrix, ROW_BLOCK};

    pub fn qc_metrics(m: &CsrMatrix, mito: Option<&[bool]>) -> QcMetrics {
        let (nrows, ncols) = m.shape();
        let n_blocks = nrows.div_ceil(ROW_BLOCK);
        let acc = (0..n_blocks)
            .into_par_iter()
            .fold(
                || (vec![0_u32; ncols], vec![0.0_f64; ncols], Vec::new()),
                |(mut gene_n, mut gene_total, mut blocks), b| {
                    let (r0, r1) = (b * ROW_BLOCK, ((b + 1) * ROW_BLOCK).min(nrows));
                    let mut n_counts = Vec::with_capacity(r1 - r0);
                    let mut n_genes = Vec::with_capacity(r1 - r0);
                    let mut mt = Vec::with_capacity(if mito.is_some() { r1 - r0 } else { 0 });
                    for i in r0..r1 {
                        let (cols, vals) = m.row(i);
                        let mut sum = 0.0_f64;
                        let mut expressed = 0_u32;
                        let mut mt_sum = 0.0_f64;
                        for (&c, &v) in cols.iter().zip(vals) {
                            let c = c as usize;
                            let v64 = f64::from(v);
                            sum += v64;
                            gene_total[c] += v64;
                            if v > 0.0 {
                                expressed += 1;
                                gene_n[c] += 1;
                            }
                            if let Some(mask) = mito {
                                if mask[c] {
                                    mt_sum += v64;
                                }
                            }
                        }
                        n_counts.push(sum as f32);
                        n_genes.push(expressed);
                        if mito.is_some() {
                            mt.push(mt_sum as f32);
                        }
                    }
                    blocks.push((b, n_counts, n_genes, mt));
                    (gene_n, gene_total, blocks)
                },
            )
            .reduce(
                || (vec![0_u32; ncols], vec![0.0_f64; ncols], Vec::new()),
                |(mut an, mut at, mut blocks), (bn, bt, mut bblocks)| {
                    for (x, y) in an.iter_mut().zip(&bn) {
                        *x += y;
                    }
                    for (x, y) in at.iter_mut().zip(&bt) {
                        *x += y;
                    }
                    blocks.append(&mut bblocks);
                    (an, at, blocks)
                },
            );
        let (gene_n_cells, gene_total_counts, mut blocks) = acc;
        blocks.sort_unstable_by_key(|b| b.0);
        let mut n_counts = Vec::with_capacity(nrows);
        let mut n_genes = Vec::with_capacity(nrows);
        let mut mt_counts = mito.map(|_| Vec::with_capacity(nrows));
        for (_, nc, ng, mt) in blocks {
            n_counts.extend(nc);
            n_genes.extend(ng);
            if let Some(out) = mt_counts.as_mut() {
                out.extend(mt);
            }
        }
        QcMetrics {
            n_counts,
            n_genes,
            mt_counts,
            gene_n_cells,
            gene_total_counts,
        }
    }

    pub fn gene_mean_var(m: &CsrMatrix, expm1: bool) -> GeneStats {
        let (nrows, ncols) = m.shape();
        let (sums, sq_sums) = (0..nrows)
            .into_par_iter()
            .fold(
                || (vec![0.0_f64; ncols], vec![0.0_f64; ncols]),
                |(mut s, mut q), i| {
                    let (cols, vals) = m.row(i);
                    for (&c, &v) in cols.iter().zip(vals) {
                        let x = if expm1 {
                            f64::from(v).exp_m1()
                        } else {
                            f64::from(v)
                        };
                        s[c as usize] += x;
                        q[c as usize] += x * x;
                    }
                    (s, q)
                },
            )
            .reduce(
                || (vec![0.0_f64; ncols], vec![0.0_f64; ncols]),
                |(mut s1, mut q1), (s2, q2)| {
                    for (a, b) in s1.iter_mut().zip(&s2) {
                        *a += b;
                    }
                    for (a, b) in q1.iter_mut().zip(&q2) {
                        *a += b;
                    }
                    (s1, q1)
                },
            );
        let n = nrows as f64;
        let means: Vec<f64> = sums.iter().map(|&s| s / n).collect();
        let variances: Vec<f64> = sq_sums
            .iter()
            .zip(&means)
            .map(|(&q, &mean)| {
                if nrows < 2 {
                    0.0
                } else {
                    ((q / n - mean * mean) * n / (n - 1.0)).max(0.0)
                }
            })
            .collect();
        GeneStats {
            means,
            variances,
            n_obs: nrows,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::legacy;

    /// One block's three arrays (data, indices, indptr), held keyed by
    /// offset until every block has arrived.
    type Piece = (Vec<f32>, Vec<u32>, Vec<u32>);

    fn counts_matrix(nrows: usize, ncols: usize, seed: u64) -> crate::matrix::CsrMatrix {
        let mut x = seed;
        let mut dense = vec![0.0_f32; nrows * ncols];
        for i in 0..nrows {
            for _ in 0..=(x % 5) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let j = (x as usize) % ncols;
                dense[i * ncols + j] = ((x % 7) + 1) as f32;
            }
        }
        crate::matrix::CsrMatrix::from_dense_row_major(&dense, nrows, ncols)
    }

    /// Row totals over a gene subset are `row_sums` of the matrix with the
    /// other genes removed, bit for bit, on non-integer values where the
    /// summation order shows: the scale an out-of-core run takes from them is
    /// the one `normalize_total` takes after `filter_genes`.
    #[test]
    fn row_totals_over_kept_genes_equal_row_sums_of_the_subset() {
        let mut m = counts_matrix(5_000, 37, 0x5eed_0001);
        for v in &mut m.data {
            *v *= 0.37;
        }
        let keep: Vec<bool> = (0..37).map(|j| j % 4 != 1).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let sub = m.select_cols_mask(&keep).unwrap();
        assert_eq!(
            bits(&row_totals_blocks(&m, Some(&keep))),
            bits(&sub.row_sums())
        );
        assert_eq!(bits(&row_totals_blocks(&m, None)), bits(&m.row_sums()));
    }

    /// A column-filter view is the column subset: every kernel run over it
    /// returns what it returns on `select_cols_mask`, bit for bit.
    #[test]
    fn filtered_cols_equals_the_column_subset() {
        let mut m = counts_matrix(4_000, 29, 0x5eed_0002);
        for v in &mut m.data {
            *v *= 0.61;
        }
        let keep: Vec<bool> = (0..29).map(|j| j % 3 != 0).collect();
        let sub = m.select_cols_mask(&keep).unwrap();
        let view = FilteredCols::new(&m, &keep);
        assert_eq!((view.n_rows(), view.n_cols()), (sub.nrows, sub.ncols));
        assert_eq!(
            qc_metrics_blocks(&view, None),
            qc_metrics_blocks(&sub, None)
        );
        assert_eq!(
            gene_mean_var_blocks(&view, false),
            gene_mean_var_blocks(&sub, false)
        );
        let labels: Vec<u32> = (0..4_000).map(|i| (i % 5) as u32).collect();
        assert_eq!(
            cluster_stats_blocks(&view, &labels, 5),
            cluster_stats_blocks(&sub, &labels, 5)
        );
    }

    #[test]
    fn qc_and_gene_stats_match_legacy_exactly_on_counts() {
        let m = counts_matrix(9_000, 41, 0xfeed_face);
        let mito: Vec<bool> = (0..41).map(|j| j % 3 == 0).collect();

        let new_qc = super::qc_metrics(&m, Some(&mito));
        let old_qc = legacy::qc_metrics(&m, Some(&mito));
        for (x, y) in new_qc.n_counts.iter().zip(&old_qc.n_counts) {
            assert_eq!(x.to_bits(), y.to_bits(), "n_counts");
        }
        assert_eq!(new_qc.n_genes, old_qc.n_genes);
        let (mt_n, mt_o) = (new_qc.mt_counts.unwrap(), old_qc.mt_counts.unwrap());
        for (x, y) in mt_n.iter().zip(&mt_o) {
            assert_eq!(x.to_bits(), y.to_bits(), "mt_counts");
        }
        assert_eq!(new_qc.gene_n_cells, old_qc.gene_n_cells);
        for (x, y) in new_qc
            .gene_total_counts
            .iter()
            .zip(&old_qc.gene_total_counts)
        {
            assert_eq!(x.to_bits(), y.to_bits(), "gene_total");
        }

        // expm1 = false on integer counts: exact in any order.
        let new_stats = super::gene_mean_var(&m, false);
        let old_stats = legacy::gene_mean_var(&m, false);
        for (x, y) in new_stats.means.iter().zip(&old_stats.means) {
            assert_eq!(x.to_bits(), y.to_bits(), "means");
        }
        for (x, y) in new_stats.variances.iter().zip(&old_stats.variances) {
            assert_eq!(x.to_bits(), y.to_bits(), "variances");
        }
    }

    use super::*;

    fn sample_csr() -> CsrMatrix {
        // [[1, 0, 2],
        //  [0, 3, 0],
        //  [4, 0, 6]]
        CsrMatrix::new(
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 6.0],
            3,
            3,
        )
        .unwrap()
    }

    /// A matrix large enough to span several row blocks, with an empty row
    /// and a row of explicit zeros to exercise the edge cases.
    fn big_csr() -> CsrMatrix {
        let nrows = ROW_BLOCK * 2 + 7;
        let ncols = 5;
        let mut indptr = vec![0_u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..nrows {
            match i % 4 {
                0 => {}
                1 => {
                    indices.extend([0, 2]);
                    data.extend([1.0, 0.0]); // explicit zero
                }
                _ => {
                    indices.extend([1, 3, 4]);
                    data.extend([2.0, (i % 7) as f32, 1.0]);
                }
            }
            indptr.push(data.len() as u32);
        }
        CsrMatrix::new(indptr, indices, data, nrows, ncols).unwrap()
    }

    #[test]
    fn qc_matches_reference_on_dense() {
        let m = big_csr();
        let dense = m.to_dense();
        let mito = vec![false, true, false, false, true];
        let qc = qc_metrics(&m, Some(&mito));
        for i in 0..m.nrows {
            let row = dense.row(i);
            let sum: f32 = row.iter().sum();
            let expressed = row.iter().filter(|&&v| v > 0.0).count() as u32;
            let mt: f32 = row
                .iter()
                .zip(&mito)
                .filter(|(_, &m)| m)
                .map(|(v, _)| v)
                .sum();
            assert!((qc.n_counts[i] - sum).abs() < 1e-5, "row {i}");
            assert_eq!(qc.n_genes[i], expressed, "row {i}");
            assert!((qc.mt_counts.as_ref().unwrap()[i] - mt).abs() < 1e-5);
        }
        for j in 0..m.ncols {
            let col: Vec<f32> = (0..m.nrows).map(|i| dense.row(i)[j]).collect();
            let total: f64 = col.iter().map(|&v| f64::from(v)).sum();
            let n_cells = col.iter().filter(|&&v| v > 0.0).count() as u32;
            assert!((qc.gene_total_counts[j] - total).abs() < 1e-6, "col {j}");
            assert_eq!(qc.gene_n_cells[j], n_cells, "col {j}");
        }
        assert!(qc_metrics(&m, None).mt_counts.is_none());
    }

    /// The streaming view must be arithmetically identical to materialising
    /// the transform — that equivalence is the whole reason it is safe to
    /// stop rewriting the matrix.
    #[test]
    fn normalized_view_matches_materialising() {
        let m = big_csr();
        let totals = m.row_sums();
        let scale = NormalizedView::scale_from_totals(&totals, 1e4);

        for log1p in [false, true] {
            let view = NormalizedView::new(&m, scale.clone(), log1p);

            let mut materialised = m.clone();
            let _ = normalize_total_inplace(&mut materialised, 1e4);
            if log1p {
                log1p_inplace(&mut materialised);
            }

            // Same gene statistics, bit for bit.
            assert_eq!(
                gene_mean_var_blocks(&view, false),
                gene_mean_var_blocks(&materialised, false),
                "gene stats (log1p={log1p})"
            );
            // And the same values, row by row.
            let (a, b) = (view_to_dense(&view), materialised.to_dense());
            assert_eq!(a, b.data, "values (log1p={log1p})");
        }
    }

    /// Read every block of a source back into a dense row-major buffer.
    fn view_to_dense(src: &dyn RowBlocks) -> Vec<f32> {
        let (n, m) = (src.n_rows(), src.n_cols());
        let out = std::sync::Mutex::new(vec![0.0_f32; n * m]);
        src.for_each_block(64, &|off, blk| {
            let mut buf = out.lock().unwrap();
            for i in 0..blk.n_rows {
                let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                for (&c, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                    buf[(off + i) * m + c as usize] = v;
                }
            }
        });
        out.into_inner().unwrap()
    }

    #[test]
    fn normalized_view_rejects_a_short_scale_vector() {
        let m = big_csr();
        let r = std::panic::catch_unwind(|| {
            let _ = NormalizedView::new(&m, vec![1.0; 3], true);
        });
        assert!(
            r.is_err(),
            "a scale vector shorter than the matrix must panic"
        );
    }

    #[test]
    fn normalize_total_scales_rows_to_target() {
        let mut m = big_csr();
        let before = normalize_total_inplace(&mut m, 100.0);
        assert_eq!(before.len(), m.nrows);
        for (i, s) in m.row_sums().iter().enumerate() {
            if before[i] > 0.0 {
                assert!((*s - 100.0).abs() < 1e-3, "row {i} sums to {s}");
            } else {
                assert!(s.abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn log1p_preserves_pattern() {
        let mut m = sample_csr();
        log1p_inplace(&mut m);
        assert_eq!(m.indptr, sample_csr().indptr);
        assert_eq!(m.indices, sample_csr().indices);
        let expected: Vec<f32> = [1.0_f32, 2.0, 3.0, 4.0, 6.0]
            .iter()
            .map(|v| v.ln_1p())
            .collect();
        assert_eq!(m.data, expected);
    }

    #[test]
    fn gene_mean_var_matches_two_pass_reference() {
        let m = big_csr();
        let dense = m.to_dense();
        for expm1 in [false, true] {
            let stats = gene_mean_var(&m, expm1);
            let n = m.nrows as f64;
            for j in 0..m.ncols {
                let col: Vec<f64> = (0..m.nrows)
                    .map(|i| {
                        let v = f64::from(dense.row(i)[j]);
                        if expm1 {
                            v.exp_m1()
                        } else {
                            v
                        }
                    })
                    .collect();
                let mean = col.iter().sum::<f64>() / n;
                let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                assert!((stats.means[j] - mean).abs() < 1e-9);
                assert!(
                    (stats.variances[j] - var).abs() < 1e-6,
                    "{} vs {var}",
                    stats.variances[j]
                );
            }
        }
    }

    #[test]
    fn hvg_selects_exact_count_and_prefers_variable_genes() {
        // 200 cells × 40 genes: genes 0..20 vary with cell, 20..40 constant.
        let (nrows, ncols) = (200, 40);
        let mut dense = vec![0.0_f32; nrows * ncols];
        for i in 0..nrows {
            for j in 0..20 {
                // 23 is prime and > 20, so no gene is identically zero.
                dense[i * ncols + j] = ((i * (j + 1)) % 23) as f32;
            }
            for j in 20..40 {
                dense[i * ncols + j] = 1.0;
            }
        }
        let m = CsrMatrix::from_dense_row_major(&dense, nrows, ncols);
        let stats = gene_mean_var(&m, false);
        let hvg = hvg_seurat(&stats, 10, 20);
        assert_eq!(hvg.n_hvg, 10);
        assert_eq!(hvg.mask.iter().filter(|&&b| b).count(), 10);
        assert!(hvg.mask.iter().take(20).any(|&b| b));
        assert!(
            hvg.mask.iter().skip(20).all(|&b| !b),
            "constant genes must not be HVG"
        );
        // Constant genes have zero dispersion → NaN, never selected even if asked.
        let all = hvg_seurat(&stats, 40, 20);
        assert_eq!(all.n_hvg, 20);
        assert!(all.dispersions[25].is_nan());
    }

    /// Bin edges must match `pd.cut(n_bins)`: linspace over [min, max] with
    /// only the first edge nudged down by 0.1 % of the range, and
    /// left-open/right-closed intervals. Widening every bin instead (the old
    /// rule) shifts boundary-adjacent genes one bin up and reorders the HVG
    /// tail against scanpy (Jaccard 0.984 on heart_10k; see chapter 09).
    #[test]
    fn hvg_bin_edges_match_pandas_cut() {
        // Data spanning [0, 1]: edges are 0.05·k (k = 0..=20 in f64) with
        // edges[0] = -0.001. Expected bins follow pandas' right-closed rule.
        // 0.0499 and 0.502 are the discriminators: both sit just *below* a
        // pandas interior edge but above where the old widened-edge rule
        // would place it, so the old rule bins them one higher.
        let xs = [0.0, 1.0, 0.05, 0.0499, 0.0501, 0.5, 0.502, 0.749, 0.9999];
        let expected = [0, 19, 0, 0, 1, 9, 10, 14, 19];
        // (0.05 and 0.5 land exactly on interior edges 1 and 10 → the bin
        // *below*, i.e. 0 and 9, because intervals are (a, b].)
        let bins = pd_cut_bins(&xs, 20);
        assert_eq!(bins, expected.to_vec());

        // Degenerate range: all values equal → one bin, no panic.
        assert_eq!(pd_cut_bins(&[2.5; 7], 20), vec![0; 7]);
    }

    /// `FilteredRows` must present exactly the matrix `select_rows` would
    /// build: same values, same offsets, and per-gene statistics that agree
    /// with the pre-selected matrix. This is the out-of-core path's
    /// `filter_cells` — before it existed, the disk pipeline kept every cell
    /// and answered a different question than the in-RAM one.
    #[test]
    fn filtered_rows_match_a_preselected_matrix() {
        let (nrows, ncols) = (60, 8);
        let mut dense = vec![0.0_f32; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                dense[i * ncols + j] = ((i * 13 + j * 7) % 11) as f32;
            }
        }
        let m = CsrMatrix::from_dense_row_major(&dense, nrows, ncols);

        let keep: Vec<bool> = (0..nrows).map(|i| i % 3 != 0).collect();
        let idx: Vec<usize> = (0..nrows).filter(|&i| keep[i]).collect();
        let want = m.select_rows(&idx).unwrap();

        let filtered = FilteredRows::new(&m, &keep);
        assert_eq!(filtered.n_rows(), idx.len());
        assert_eq!(filtered.n_cols(), ncols);

        // Blocks concatenate, exactly, to the selected matrix; offsets count
        // kept rows.
        //
        // Blocks are collected *keyed by offset* and assembled afterwards,
        // never appended in arrival order. `for_each_block` makes no
        // ordering promise — the source runs blocks in parallel batches —
        // and an earlier version of this test appended as they arrived and
        // asserted `off == next_off`. That held when the whole suite ran
        // (contention happened to serialise the batch) and failed when the
        // test was run by name, which is the worst way for a test to be
        // wrong: green in CI, red for whoever reruns it alone. Keying by
        // offset is the discipline `BlockFold` uses, for the same reason.
        let collected: std::sync::Mutex<BTreeMap<usize, Piece>> =
            std::sync::Mutex::new(BTreeMap::new());
        filtered.for_each_block(7, &|off, blk| {
            let piece = (
                blk.data.to_vec(),
                blk.indices.to_vec(),
                blk.indptr[..=blk.n_rows].to_vec(),
            );
            let prev = collected.lock().unwrap().insert(off, piece);
            assert!(prev.is_none(), "offset {off} emitted twice");
        });

        let (mut data, mut indices, mut indptr) = (Vec::new(), Vec::new(), vec![0_u32]);
        let mut next_off = 0_usize;
        for (off, (d, i, ptr)) in collected.into_inner().unwrap() {
            assert_eq!(off, next_off, "a block offset is missing from the stream");
            next_off += ptr.len() - 1;
            data.extend_from_slice(&d);
            indices.extend_from_slice(&i);
            let base = *indptr.last().unwrap();
            indptr.extend(ptr[1..].iter().map(|&v| base + v));
        }
        assert_eq!(next_off, idx.len());
        assert_eq!(data, want.data);
        assert_eq!(indices, want.indices);
        assert_eq!(indptr, want.indptr);

        // Per-gene statistics over the filtered stream agree with the ones
        // computed on the pre-selected matrix (block folds differ in
        // summation order, so this is approximate, not bitwise).
        let stats = gene_mean_var_blocks(&filtered, false);
        let want_stats = gene_mean_var(&want, false);
        for g in 0..ncols {
            assert!((stats.means[g] - want_stats.means[g]).abs() < 1e-12);
            assert!((stats.variances[g] - want_stats.variances[g]).abs() < 1e-9);
        }

        // An empty filter emits no blocks at all.
        let none = FilteredRows::new(&m, &vec![false; nrows]);
        assert_eq!(none.n_rows(), 0);
        none.for_each_block(7, &|_, _| panic!("no blocks expected"));
    }
}
