//! Doublet detection: Scrublet (Wolock, Lopez & Klein, *Cell Systems* 8,
//! 281-291, 2019), ported from its reference implementation,
//! <https://github.com/swolock/scrublet>, MIT License, Copyright (c) 2018
//! Samuel Wolock.
//!
//! Each batch (one capture, such as a 10x channel: doublets form within one)
//! is scored on its own, as the reference scores one sample:
//!
//! 1. Cells are scaled to the batch's mean total count, and genes are kept
//!    when their variability exceeds what counting noise explains (the
//!    v-score of Klein et al. 2015, top 15% by default) and they reach 3
//!    normalized counts in at least 3 cells.
//! 2. Doublets are simulated by adding the counts of random pairs of cells,
//!    twice as many as there are cells.
//! 3. Cells and doublets are scaled to 10^6 counts, standardized with the
//!    cells' gene means and deviations, and projected on the cells' first 30
//!    principal components.
//! 4. Each cell's score is the Bayesian estimate, from the share of simulated
//!    doublets among its `k` nearest neighbours, of the probability that it
//!    is a doublet. The threshold is the minimum between the two modes of the
//!    simulated doublets' scores.
//!
//! Two departures from the reference, neither changing what is computed:
//!
//! - **Simulated doublets are never built.** After the per-cell scaling of
//!   step 3 every operation is linear, so a doublet's coordinates are its two
//!   parents' projections summed and rescaled:
//!   `z(a + b) = 10^6 (u_a + u_b) / (t_a + t_b) − m`, where `u` is a cell's
//!   raw counts over the genes' deviations projected on the components. The
//!   memory is `30` numbers per cell, not a second count matrix twice the
//!   size of the first.
//! - **Neighbours are exact** up to [`ScrubletOptions::exact_knn_max`]
//!   points; the reference uses annoy's approximate search. Above that,
//!   NN-descent.
//!
//! The reference's licence, which covers the parts of this module that follow
//! its code:
//!
//! > The MIT License (MIT). Copyright (c) 2018 Samuel Wolock.
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a
//! > copy of this software and associated documentation files (the
//! > "Software"), to deal in the Software without restriction, including
//! > without limitation the rights to use, copy, modify, merge, publish,
//! > distribute, sublicense, and/or sell copies of the Software, and to permit
//! > persons to whom the Software is furnished to do so, subject to the
//! > following conditions: The above copyright notice and this permission
//! > notice shall be included in all copies or substantial portions of the
//! > Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//! > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//! > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//! > NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! > DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! > OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//! > USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! The numerical helpers below reproduce, to match the reference, NumPy's
//! `percentile`, `linspace`, `histogram` and `round` (BSD-3-Clause, Copyright
//! (c) 2005-2025 NumPy Developers), SciPy's `optimize.fmin` (BSD-3-Clause,
//! Copyright (c) 2001-2002 Enthought, Inc. and 2003 onwards SciPy Developers)
//! and scikit-image's `filters.threshold_minimum` (in
//! `skimage/filters/thresholding.py`, BSD-2-Clause, Copyright 2009-2015 Board
//! of Regents of the University of Wisconsin-Madison, Broad Institute of MIT
//! and Harvard, and Max Planck Institute of Molecular Cell Biology and
//! Genetics, 2009 Zachary Pincus, 2009 Almar Klein, and 2009-2022 the
//! scikit-image team). Their licences permit redistribution in source and
//! binary form provided the copyright notice, their conditions and disclaimer
//! are retained, which `THIRD_PARTY_NOTICES.md` does.
//!
//! The counts are read in three streamed passes (totals, gene statistics,
//! and the selected genes of each batch), so the source may be the in-memory
//! matrix or the on-disk store alike; only the selected genes of the batches
//! being scored are held, in groups bounded by
//! [`ScrubletOptions::max_batch_bytes`].

// Numerical code written after its references: single letters follow the
// formulas (n cells, h genes, p components, k neighbours, w scale factors,
// u projections, m means), loops index several arrays at once, the exact
// float comparisons are the reference's, and project names (NumPy, SciPy)
// appear in prose.
#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::sync::Mutex;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::matrix::CsrMatrix;
use crate::preprocess::row_totals_blocks;
use crate::rowblocks::{BlockFold, RowBlocks};

/// A batch code marking a row that is not scored.
pub const UNSCORED: u32 = u32::MAX;

/// Rows per streamed block.
const BLOCK: usize = 8_192;

/// Options for [`scrublet_blocks`]; the defaults are Scanpy's for
/// `pp.scrublet`, which follow the reference except for the expected rate.
#[derive(Debug, Clone)]
pub struct ScrubletOptions {
    /// Doublets simulated per cell.
    pub sim_doublet_ratio: f64,
    /// Expected share of doublets.
    pub expected_doublet_rate: f64,
    /// Uncertainty of the expected share.
    pub stdev_doublet_rate: f64,
    /// Neighbours per cell (`None`: `round(0.5 √n)`), before scaling by the
    /// share of simulated points.
    pub n_neighbors: Option<usize>,
    /// Principal components.
    pub n_prin_comps: usize,
    /// Normalized count a gene must reach in `min_cells` cells.
    pub min_counts: f64,
    /// See `min_counts`.
    pub min_cells: usize,
    /// Genes kept above this percentile of the v-score.
    pub min_gene_variability_pctl: f64,
    /// Score threshold (`None`: found from the simulated doublets).
    pub threshold: Option<f64>,
    /// Random seed of the simulation (batch `b` uses `seed + b`).
    pub seed: u64,
    /// Points (cells and simulated doublets) up to which the neighbour
    /// search is exact.
    pub exact_knn_max: usize,
    /// Bytes the batches' selected genes may occupy at once.
    pub max_batch_bytes: usize,
}

impl Default for ScrubletOptions {
    fn default() -> Self {
        Self {
            sim_doublet_ratio: 2.0,
            expected_doublet_rate: 0.05,
            stdev_doublet_rate: 0.02,
            n_neighbors: None,
            n_prin_comps: 30,
            min_counts: 3.0,
            min_cells: 3,
            min_gene_variability_pctl: 85.0,
            threshold: None,
            seed: 0,
            exact_knn_max: 60_000,
            max_batch_bytes: 1 << 30,
        }
    }
}

/// What scoring one batch found.
#[derive(Debug, Clone, Default)]
pub struct BatchResult {
    /// The batch code.
    pub batch: u32,
    /// Cells scored.
    pub n_cells: usize,
    /// Genes selected.
    pub n_genes: usize,
    /// Doublets simulated.
    pub n_sim: usize,
    /// Neighbours per point, after scaling.
    pub n_neighbors: usize,
    /// The threshold used (`None` when it could not be found; no cell is
    /// then called a doublet).
    pub threshold: Option<f64>,
    /// Scores of the simulated doublets.
    pub scores_sim: Vec<f32>,
    /// Share of cells called doublets.
    pub detected_doublet_rate: f64,
    /// Share of simulated doublets above the threshold.
    pub detectable_doublet_fraction: f64,
    /// `detected / detectable`.
    pub overall_doublet_rate: f64,
    /// Why the batch was not scored or has no threshold, if so.
    pub note: Option<String>,
}

// ── numerical helpers that reproduce numpy / scipy / scikit-image ────────

/// numpy's default (linear) percentile of `v`, `q` in `[0, 100]`.
fn percentile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let pos = (n - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    v[lo] + (v[hi] - v[lo]) * frac
}

/// numpy's `linspace(start, stop, num)`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    out[num - 1] = stop;
    out
}

/// numpy's `histogram(v, bins)` over `[min, max]`: counts and bin centers.
fn histogram(v: &[f64], bins: usize) -> (Vec<u64>, Vec<f64>) {
    let (lo, hi) = v
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let (lo, hi) = if lo == hi {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    };
    let edges = linspace(lo, hi, bins + 1);
    let norm = bins as f64 / (hi - lo);
    let mut counts = vec![0_u64; bins];
    for &x in v {
        let mut i = ((x - lo) * norm) as usize;
        if i >= bins {
            i = bins - 1;
        }
        if x < edges[i] && i > 0 {
            i -= 1;
        } else if i + 1 < bins && x >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1;
    }
    let centers = edges.windows(2).map(|e| (e[0] + e[1]) / 2.0).collect();
    (counts, centers)
}

/// scipy's `optimize.fmin` (Nelder-Mead) in one dimension, with its defaults.
fn fmin_1d(f: impl Fn(f64) -> f64, x0: f64) -> f64 {
    let (xatol, fatol, maxiter) = (1e-4, 1e-4, 200);
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);
    let x1 = if x0 == 0.0 { 0.000_25 } else { x0 * 1.05 };
    let mut sim = [x0, x1];
    let mut fsim = [f(x0), f(x1)];
    let mut calls = 2;
    let sort = |sim: &mut [f64; 2], fsim: &mut [f64; 2]| {
        // A stable sort on the values, as numpy's argsort of two entries.
        if fsim[1] < fsim[0] {
            sim.swap(0, 1);
            fsim.swap(0, 1);
        }
    };
    sort(&mut sim, &mut fsim);
    for _ in 0..maxiter {
        if calls >= maxiter {
            break;
        }
        if (sim[1] - sim[0]).abs() <= xatol && (fsim[0] - fsim[1]).abs() <= fatol {
            break;
        }
        let xbar = sim[0];
        let xr = (1.0 + rho) * xbar - rho * sim[1];
        let fxr = f(xr);
        calls += 1;
        if fxr < fsim[0] {
            let xe = (1.0 + rho * chi) * xbar - rho * chi * sim[1];
            let fxe = f(xe);
            calls += 1;
            if fxe < fxr {
                (sim[1], fsim[1]) = (xe, fxe);
            } else {
                (sim[1], fsim[1]) = (xr, fxr);
            }
        } else {
            // With one dimension scipy's "accept the reflection" case, which
            // compares with the second-worst point, cannot occur.
            let mut shrink = false;
            if fxr < fsim[1] {
                let xc = (1.0 + psi * rho) * xbar - psi * rho * sim[1];
                let fxc = f(xc);
                calls += 1;
                if fxc <= fxr {
                    (sim[1], fsim[1]) = (xc, fxc);
                } else {
                    shrink = true;
                }
            } else {
                let xcc = (1.0 - psi) * xbar + psi * sim[1];
                let fxcc = f(xcc);
                calls += 1;
                if fxcc < fsim[1] {
                    (sim[1], fsim[1]) = (xcc, fxcc);
                } else {
                    shrink = true;
                }
            }
            if shrink {
                sim[1] = sim[0] + sigma * (sim[1] - sim[0]);
                fsim[1] = f(sim[1]);
                calls += 1;
            }
        }
        sort(&mut sim, &mut fsim);
    }
    sim[0]
}

/// Scrublet's `runningquantile`: the `p`-th percentile of `y` in `n_bins`
/// equal bins of `x`.
fn running_quantile(x: &[f64], y: &[f64], p: f64, n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let (first, last) = (xs[0], xs[xs.len() - 1]);
    let dx = (last - first) / n_bins as f64;
    let x_out = linspace(first + dx / 2.0, last - dx / 2.0, n_bins);
    let mut y_out = vec![f64::NAN; n_bins];
    for (i, &c) in x_out.iter().enumerate() {
        let mut inside: Vec<f64> = xs
            .iter()
            .zip(&ys)
            .filter(|(&xv, _)| xv >= c - dx / 2.0 && xv < c + dx / 2.0)
            .map(|(_, &yv)| yv)
            .collect();
        y_out[i] = if inside.is_empty() {
            if i > 0 {
                y_out[i - 1]
            } else {
                f64::NAN
            }
        } else {
            percentile(&mut inside, p)
        };
    }
    (x_out, y_out)
}

/// Scrublet's v-scores (`get_vscores`, defaults `nBins=50`,
/// `fit_percentile=0.1`, `error_wt=1`) of genes with the given means and
/// population variances; `NaN` where the mean is zero. Genes whose
/// Fano factor is zero take no part in the fit.
fn vscores(mu: &[f64], var: &[f64]) -> Vec<f64> {
    let genes: Vec<usize> = (0..mu.len()).filter(|&g| mu[g] > 0.0).collect();
    let mut out = vec![f64::NAN; mu.len()];
    if genes.len() < 3 {
        return out;
    }
    let ff: Vec<f64> = genes.iter().map(|&g| var[g] / mu[g]).collect();
    let fit: Vec<usize> = (0..genes.len()).filter(|&i| ff[i] > 0.0).collect();
    if fit.len() < 3 {
        return out;
    }
    let x: Vec<f64> = fit.iter().map(|&i| mu[genes[i]].ln()).collect();
    let y: Vec<f64> = fit.iter().map(|&i| (ff[i] / mu[genes[i]]).ln()).collect();
    let (xq, yq) = running_quantile(&x, &y, 0.1, 50);
    let (xq, yq): (Vec<f64>, Vec<f64>) = xq
        .into_iter()
        .zip(yq)
        .filter(|(_, yv)| !yv.is_nan())
        .unzip();
    let log_ff: Vec<f64> = fit.iter().map(|&i| ff[i].ln()).collect();
    let (h, centers) = histogram(&log_ff, 200);
    let top = h
        .iter()
        .enumerate()
        .fold(
            (0, 0),
            |(bi, bv), (i, &c)| if c > bv { (i, c) } else { (bi, bv) },
        )
        .0;
    let c = centers[top].exp().max(1.0);
    let err = |b: f64| -> f64 {
        xq.iter()
            .zip(&yq)
            .map(|(&xv, &yv)| ((c * (-xv).exp() + b).ln() - yv).abs())
            .sum()
    };
    let b = fmin_1d(err, 0.1);
    let a = c / (1.0 + b) - 1.0;
    for (i, &g) in genes.iter().enumerate() {
        out[g] = ff[i] / ((1.0 + a) * (1.0 + b) + b * mu[g]);
    }
    out
}

/// scikit-image's `threshold_minimum` (256 bins): the minimum between the
/// two modes of the histogram, smoothed until it has two.
fn threshold_minimum(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let (counts, centers) = histogram(v, 256);
    let mut smooth: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
    let maxima = |h: &[f32]| -> Vec<usize> {
        let mut out = Vec::new();
        let mut up = true;
        for i in 0..h.len() - 1 {
            if up {
                if h[i + 1] < h[i] {
                    up = false;
                    out.push(i);
                }
            } else if h[i + 1] > h[i] {
                up = true;
            }
        }
        out
    };
    let max_iter = 10_000;
    let mut peaks = Vec::new();
    let mut done = false;
    for _ in 0..max_iter {
        // scipy's uniform_filter1d(size=3), mode 'reflect'.
        let n = smooth.len();
        let prev = smooth.clone();
        for i in 0..n {
            let l = if i == 0 { prev[0] } else { prev[i - 1] };
            let r = if i + 1 == n { prev[n - 1] } else { prev[i + 1] };
            smooth[i] = ((f64::from(l) + f64::from(prev[i]) + f64::from(r)) / 3.0) as f32;
        }
        peaks = maxima(&smooth);
        if peaks.len() < 3 {
            done = true;
            break;
        }
    }
    if !done || peaks.len() != 2 {
        return None;
    }
    let (a, b) = (peaks[0], peaks[1]);
    let low = (a..=b).fold(a, |best, i| if smooth[i] < smooth[best] { i } else { best });
    Some(centers[low])
}

/// The threshold between the two modes of the simulated doublets' scores:
/// the reference's rule ([`threshold_minimum`]), except that a split leaving
/// fewer than 5% of the simulated doublets above it is not accepted. Such a
/// split separates a handful of isolated extreme scores (a second "mode" of a
/// few points that no amount of smoothing merges) rather than the two
/// populations; those scores are set aside and the valley found again. On
/// scores where the rule already works this changes nothing.
fn threshold_from_simulated(sim: &[f64]) -> Option<f64> {
    let mut s: Vec<f64> = sim.to_vec();
    for _ in 0..20 {
        let t = threshold_minimum(&s)?;
        let above = s.iter().filter(|&&x| x > t).count();
        if above as f64 >= 0.05 * s.len() as f64 {
            return Some(t);
        }
        s.retain(|&x| x <= t);
    }
    None
}

/// numpy's `round` (half to even).
fn round_half_even(x: f64) -> usize {
    let r = x.round();
    let r = if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    };
    r.max(0.0) as usize
}

// ── the pipeline ──────────────────────────────────────────────────────────

/// Per-batch gene statistics at the batch's mean total.
#[derive(Clone)]
struct GeneAcc {
    sum: Vec<f64>,
    sq: Vec<f64>,
    over: Vec<u32>,
    nnz: Vec<u64>,
}

impl GeneAcc {
    fn new(g: usize) -> Self {
        Self {
            sum: vec![0.0; g],
            sq: vec![0.0; g],
            over: vec![0; g],
            nnz: vec![0; g],
        }
    }

    fn add(&mut self, o: &Self) {
        for (a, b) in self.sum.iter_mut().zip(&o.sum) {
            *a += b;
        }
        for (a, b) in self.sq.iter_mut().zip(&o.sq) {
            *a += b;
        }
        for (a, b) in self.over.iter_mut().zip(&o.over) {
            *a += b;
        }
        for (a, b) in self.nnz.iter_mut().zip(&o.nnz) {
            *a += b;
        }
    }
}

/// One batch's cells over its selected genes: raw counts, their totals over
/// every gene, and their rows in the source.
struct BatchCounts {
    counts: CsrMatrix,
    totals: Vec<f64>,
    rows: Vec<usize>,
}

/// One block's rows of one batch, over the batch's selected genes.
#[derive(Default)]
struct Part {
    rows: Vec<usize>,
    lens: Vec<u32>,
    ix: Vec<u32>,
    vals: Vec<f32>,
}

/// Order of candidate neighbours: by distance, then by index. (A comparator,
/// so it takes references.)
#[allow(clippy::trivially_copy_pass_by_ref)]
fn by_distance(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

fn sift_up(h: &mut [(f32, u32)], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if by_distance(&h[i], &h[parent]).is_gt() {
            h.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

fn sift_down(h: &mut [(f32, u32)], mut i: usize) {
    let n = h.len();
    loop {
        let (l, r) = (2 * i + 1, 2 * i + 2);
        let mut top = i;
        if l < n && by_distance(&h[l], &h[top]).is_gt() {
            top = l;
        }
        if r < n && by_distance(&h[r], &h[top]).is_gt() {
            top = r;
        }
        if top == i {
            break;
        }
        h.swap(i, top);
        i = top;
    }
}

/// Exact `k` nearest neighbours of every point, itself excluded, ties broken
/// by index: `n × k` indices. Tiles of queries run in parallel, one per
/// thread; each takes its squared distances to every point from one matrix
/// product, `|q|² + |p|² − 2 p·q` (at most 8 MB per tile), and keeps the
/// nearest in a bounded max-heap.
fn knn_exact(p: &[f32], n: usize, d: usize, k: usize) -> Vec<u32> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat, MatRef, Par};

    let norms: Vec<f32> = p
        .chunks_exact(d)
        .map(|r| r.iter().map(|x| x * x).sum())
        .collect();
    let all = MatRef::from_row_major_slice(p, n, d);
    let tile = ((8usize << 20) / (4 * n).max(1)).clamp(8, 1024).min(n);
    let mut out = vec![0_u32; n * k];
    out.par_chunks_mut(tile * k)
        .enumerate()
        .for_each(|(ti, block)| {
            let q0 = ti * tile;
            let rows = block.len() / k;
            let q = MatRef::from_row_major_slice(&p[q0 * d..(q0 + rows) * d], rows, d);
            // (n × rows), column-major: each query's dot products are contiguous.
            let mut dots = Mat::<f32>::zeros(n, rows);
            matmul(
                dots.as_mut(),
                Accum::Replace,
                all,
                q.transpose(),
                1.0,
                Par::Seq,
            );
            let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k);
            for (t, row) in block.chunks_mut(k).enumerate() {
                let i = q0 + t;
                heap.clear();
                // Most candidates are turned away by one comparison with the
                // farthest of the nearest so far.
                for (j, &dot) in dots.col_as_slice(t).iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let c = (norms[i] + norms[j] - 2.0 * dot, j as u32);
                    if heap.len() < k {
                        heap.push(c);
                        let last = heap.len() - 1;
                        sift_up(&mut heap, last);
                    } else if by_distance(&c, &heap[0]).is_lt() {
                        heap[0] = c;
                        sift_down(&mut heap, 0);
                    }
                }
                heap.sort_unstable_by(by_distance);
                for (o, &(_, j)) in row.iter_mut().zip(heap.iter()) {
                    *o = j;
                }
            }
        });
    out
}

/// Score one batch; writes its cells' scores and calls.
fn score_batch(
    b: u32,
    bc: &BatchCounts,
    opts: &ScrubletOptions,
    scores: &Mutex<&mut [f32]>,
    calls: &Mutex<&mut [bool]>,
) -> BatchResult {
    let (n, h) = (bc.counts.nrows, bc.counts.ncols);
    let trace = std::env::var_os("POLARISEQ_TRACE").is_some();
    let t0 = std::time::Instant::now();
    let phase = |name: &str| {
        if trace {
            eprintln!(
                "[scrublet {b}] {name:>12} done at {:6.2}s",
                t0.elapsed().as_secs_f64()
            );
        }
    };
    let mut res = BatchResult {
        batch: b,
        n_cells: n,
        n_genes: h,
        ..BatchResult::default()
    };
    let p = opts.n_prin_comps.min(h).min(n.saturating_sub(1));
    if n < 10 || p == 0 {
        res.note = Some(format!(
            "{n} cells and {h} selected genes are too few to score"
        ));
        return res;
    }

    // Cells scaled to 10^6 counts; gene means and population deviations.
    let w: Vec<f64> = bc.totals.iter().map(|&t| 1e6 / t).collect();
    let mut s1 = vec![0.0_f64; h];
    let mut s2 = vec![0.0_f64; h];
    for i in 0..n {
        let (cols, vals) = bc.counts.row(i);
        for (&c, &x) in cols.iter().zip(vals) {
            let y = f64::from(x) * w[i];
            s1[c as usize] += y;
            s2[c as usize] += y * y;
        }
    }
    let nf = n as f64;
    let mu: Vec<f64> = s1.iter().map(|s| s / nf).collect();
    let sd: Vec<f64> = s2
        .iter()
        .zip(&mu)
        .map(|(s, m)| {
            let v = (s / nf - m * m).max(0.0).sqrt();
            if v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();
    let m: Vec<f64> = mu.iter().zip(&sd).map(|(a, s)| a / s).collect();

    // Principal components of the standardized cells: the centred Gram
    // matrix of y/σ, whose mean is m.
    let mut a = bc.counts.clone();
    for i in 0..n {
        let (s, e) = (a.indptr[i] as usize, a.indptr[i + 1] as usize);
        for t in s..e {
            let c = a.indices[t] as usize;
            a.data[t] = (f64::from(a.data[t]) * w[i] / sd[c]) as f32;
        }
    }
    let mut gram = crate::rsvd::gram_matrix(crate::rsvd::CsrRef::from(&a));
    drop(a);
    gram.par_chunks_mut(h).enumerate().for_each(|(i, row)| {
        for (j, x) in row.iter_mut().enumerate() {
            *x -= nf * m[i] * m[j];
        }
    });
    let eig = match faer::MatRef::from_row_major_slice(&gram, h, h)
        .self_adjoint_eigen(faer::Side::Lower)
    {
        Ok(e) => e,
        Err(e) => {
            res.note = Some(format!("PCA failed: {e:?}"));
            return res;
        }
    };
    phase("pca");
    let vecs = eig.U();
    // v[g * p + j]: gene g's weight in component j, largest first.
    let mut v = vec![0.0_f64; h * p];
    for j in 0..p {
        for g in 0..h {
            v[g * p + j] = vecs[(g, h - 1 - j)];
        }
    }
    drop(gram);

    // u_i: raw counts over σ, projected; mv: the projected mean.
    let mut mv = vec![0.0_f64; p];
    for g in 0..h {
        for j in 0..p {
            mv[j] += m[g] * v[g * p + j];
        }
    }
    let mut u = vec![0.0_f64; n * p];
    u.par_chunks_mut(p).enumerate().for_each(|(i, ui)| {
        let (cols, vals) = bc.counts.row(i);
        for (&c, &x) in cols.iter().zip(vals) {
            let c = c as usize;
            let y = f64::from(x) / sd[c];
            for j in 0..p {
                ui[j] += y * v[c * p + j];
            }
        }
    });

    // Cells, then simulated doublets, in one point set.
    let n_sim = (nf * opts.sim_doublet_ratio) as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed.wrapping_add(u64::from(b)));
    let pairs: Vec<(usize, usize)> = (0..n_sim)
        .map(|_| (rng.random_range(0..n), rng.random_range(0..n)))
        .collect();
    let total = n + n_sim;
    let mut pts = vec![0.0_f32; total * p];
    pts.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
        if i < n {
            for j in 0..p {
                row[j] = (w[i] * u[i * p + j] - mv[j]) as f32;
            }
        } else {
            let (x, y) = pairs[i - n];
            let s = 1e6 / (bc.totals[x] + bc.totals[y]);
            for j in 0..p {
                row[j] = (s * (u[x * p + j] + u[y * p + j]) - mv[j]) as f32;
            }
        }
    });
    drop(u);
    phase("projection");

    let k = opts
        .n_neighbors
        .unwrap_or_else(|| round_half_even(0.5 * nf.sqrt()))
        .max(1);
    let r = n_sim as f64 / nf;
    let k_adj = round_half_even(k as f64 * (1.0 + r)).clamp(1, total - 1);
    let knn: Vec<u32> = if total <= opts.exact_knn_max {
        knn_exact(&pts, total, p, k_adj)
    } else {
        let g = crate::nndescent::nndescent(
            &pts,
            total,
            p,
            &crate::nndescent::NndescentOptions {
                k: k_adj,
                seed: opts.seed,
                ..Default::default()
            },
        );
        let mut flat = vec![0_u32; total * k_adj];
        for i in 0..total {
            let (s, e) = (g.indptr[i] as usize, g.indptr[i + 1] as usize);
            let row: Vec<u32> = g.indices[s..e]
                .iter()
                .copied()
                .filter(|&j| j as usize != i)
                .collect();
            for (t, o) in flat[i * k_adj..(i + 1) * k_adj].iter_mut().enumerate() {
                *o = row.get(t).copied().unwrap_or(i as u32);
            }
        }
        flat
    };
    drop(pts);
    phase("neighbours");

    // The Bayesian score of the reference.
    let rho = opts.expected_doublet_rate;
    let big_n = k_adj as f64;
    let score = |i: usize| -> f64 {
        let nd = knn[i * k_adj..(i + 1) * k_adj]
            .iter()
            .filter(|&&j| j as usize >= n)
            .count() as f64;
        let q = (nd + 1.0) / (big_n + 2.0);
        q * rho / r / (1.0 - rho - q * (1.0 - rho - rho / r))
    };
    let obs: Vec<f64> = (0..n).into_par_iter().map(score).collect();
    let sim: Vec<f64> = (n..total).into_par_iter().map(score).collect();

    let threshold = opts.threshold.or_else(|| threshold_from_simulated(&sim));
    res.n_sim = n_sim;
    res.n_neighbors = k_adj;
    res.threshold = threshold;
    res.scores_sim = sim.iter().map(|&s| s as f32).collect();
    match threshold {
        Some(t) => {
            let called = obs.iter().filter(|&&s| s > t).count();
            let detectable = sim.iter().filter(|&&s| s > t).count();
            res.detected_doublet_rate = called as f64 / nf;
            res.detectable_doublet_fraction = detectable as f64 / n_sim.max(1) as f64;
            res.overall_doublet_rate = if detectable > 0 {
                res.detected_doublet_rate / res.detectable_doublet_fraction
            } else {
                f64::NAN
            };
        }
        None => {
            res.note = Some(
                "no threshold: the simulated doublets' scores do not have two modes; pass one"
                    .into(),
            );
        }
    }
    let mut sc = scores.lock().expect("scores poisoned");
    let mut ca = calls.lock().expect("calls poisoned");
    for (i, &row) in bc.rows.iter().enumerate() {
        sc[row] = obs[i] as f32;
        ca[row] = threshold.is_some_and(|t| obs[i] > t);
    }
    res
}

/// Score the doublets of every batch of `src` (raw counts): `batch[i]` is
/// row `i`'s batch, below `n_batches`, or [`UNSCORED`]. Writes each scored
/// row's score and call into `scores` and `calls` (rows not scored get
/// `NaN` and `false`), and returns what each batch found.
///
/// # Panics
/// Panics if `batch`, `scores` or `calls` does not have one entry per row.
pub fn scrublet_blocks(
    src: &dyn RowBlocks,
    batch: &[u32],
    n_batches: usize,
    opts: &ScrubletOptions,
    scores: &mut [f32],
    calls: &mut [bool],
) -> Vec<BatchResult> {
    let (n_rows, g) = (src.n_rows(), src.n_cols());
    assert_eq!(batch.len(), n_rows, "one batch code per row");
    assert_eq!(scores.len(), n_rows, "one score per row");
    assert_eq!(calls.len(), n_rows, "one call per row");
    scores.fill(f32::NAN);
    calls.fill(false);

    // Pass 1: totals; cells with no counts are not scored.
    let totals = row_totals_blocks(src, None);
    let mut size = vec![0_usize; n_batches];
    let mut mean = vec![0.0_f64; n_batches];
    for (i, &b) in batch.iter().enumerate() {
        if (b as usize) < n_batches && totals[i] > 0.0 {
            size[b as usize] += 1;
            mean[b as usize] += f64::from(totals[i]);
        }
    }
    for (m, &s) in mean.iter_mut().zip(&size) {
        *m /= s.max(1) as f64;
    }
    let scored = |i: usize| (batch[i] as usize) < n_batches && totals[i] > 0.0;

    // Pass 2: each batch's gene statistics at its mean total.
    let fold: BlockFold<Vec<Option<GeneAcc>>> = BlockFold::new(|acc, part| {
        for (a, p) in acc.iter_mut().zip(part) {
            match (a.as_mut(), p) {
                (Some(a), Some(p)) => a.add(&p),
                (None, Some(p)) => *a = Some(p),
                _ => {}
            }
        }
    });
    let min_counts = opts.min_counts;
    src.for_each_block(BLOCK, &|off, blk| {
        let mut part: Vec<Option<GeneAcc>> = vec![None; n_batches];
        for i in 0..blk.n_rows {
            let row = off + i;
            if !scored(row) {
                continue;
            }
            let b = batch[row] as usize;
            let acc = part[b].get_or_insert_with(|| GeneAcc::new(g));
            let s = mean[b] / f64::from(totals[row]);
            let (x0, x1) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            for (&c, &x) in blk.indices[x0..x1].iter().zip(&blk.data[x0..x1]) {
                let y = f64::from(x) * s;
                let c = c as usize;
                acc.sum[c] += y;
                acc.sq[c] += y * y;
                acc.over[c] += u32::from(y >= min_counts);
                acc.nnz[c] += u64::from(x != 0.0);
            }
        }
        fold.store(off, part);
    });
    let stats = fold.finish(vec![None; n_batches]);

    // Each batch's genes: variable beyond counting noise, and expressed.
    let genes: Vec<Vec<u32>> = stats
        .par_iter()
        .enumerate()
        .map(|(b, acc)| {
            let Some(acc) = acc else { return Vec::new() };
            let nb = size[b] as f64;
            let mu: Vec<f64> = acc.sum.iter().map(|s| s / nb).collect();
            let var: Vec<f64> = acc
                .sq
                .iter()
                .zip(&mu)
                .map(|(q, m)| q / nb - m * m)
                .collect();
            let v = vscores(&mu, &var);
            let mut positive: Vec<f64> = v.iter().copied().filter(|x| *x > 0.0).collect();
            if positive.is_empty() {
                return Vec::new();
            }
            let cut = percentile(&mut positive, opts.min_gene_variability_pctl);
            (0..g)
                .filter(|&j| v[j] > 0.0 && v[j] >= cut && acc.over[j] as usize >= opts.min_cells)
                .map(|j| j as u32)
                .collect()
        })
        .collect();

    // Pass 3, in groups of batches that fit: their cells over their genes.
    let bytes = |b: usize| -> usize {
        stats[b].as_ref().map_or(0, |acc| {
            genes[b]
                .iter()
                .map(|&j| acc.nnz[j as usize] as usize)
                .sum::<usize>()
                * 8
                + size[b] * 24
        })
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut used = 0;
    for b in (0..n_batches).filter(|&b| size[b] > 0) {
        let need = bytes(b);
        if groups.is_empty() || used + need > opts.max_batch_bytes {
            groups.push(Vec::new());
            used = 0;
        }
        groups.last_mut().expect("a group").push(b);
        used += need;
    }

    let scores_m = Mutex::new(scores);
    let calls_m = Mutex::new(calls);
    let mut results: Vec<BatchResult> = Vec::new();
    for group in groups {
        let mut col_map: Vec<Option<Vec<u32>>> = vec![None; n_batches];
        for &b in &group {
            let mut map = vec![u32::MAX; g];
            for (k, &j) in genes[b].iter().enumerate() {
                map[j as usize] = k as u32;
            }
            col_map[b] = Some(map);
        }
        // Per block, for each batch of the group, its rows' selected genes.
        let parts: Mutex<BTreeMap<usize, Vec<(usize, Part)>>> = Mutex::new(BTreeMap::new());
        src.for_each_block(BLOCK, &|off, blk| {
            let mut local: Vec<(usize, Part)> = Vec::new();
            for i in 0..blk.n_rows {
                let row = off + i;
                if !scored(row) {
                    continue;
                }
                let b = batch[row] as usize;
                let Some(map) = col_map[b].as_ref() else {
                    continue;
                };
                let k = if let Some(k) = local.iter().position(|(pb, _)| *pb == b) {
                    k
                } else {
                    local.push((b, Part::default()));
                    local.len() - 1
                };
                let part = &mut local[k].1;
                let before = part.ix.len();
                let (x0, x1) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
                for (&c, &x) in blk.indices[x0..x1].iter().zip(&blk.data[x0..x1]) {
                    let j = map[c as usize];
                    if j != u32::MAX && x != 0.0 {
                        part.ix.push(j);
                        part.vals.push(x);
                    }
                }
                part.lens.push((part.ix.len() - before) as u32);
                part.rows.push(row);
            }
            parts.lock().expect("parts poisoned").insert(off, local);
        });
        let mut per: BTreeMap<usize, Part> = group.iter().map(|&b| (b, Part::default())).collect();
        for (_, local) in parts.into_inner().expect("parts poisoned") {
            for (b, part) in local {
                let e = per.get_mut(&b).expect("a batch of the group");
                e.rows.extend(part.rows);
                e.lens.extend(part.lens);
                e.ix.extend(part.ix);
                e.vals.extend(part.vals);
            }
        }
        for (b, part) in per {
            let n = part.rows.len();
            let mut indptr = Vec::with_capacity(n + 1);
            indptr.push(0_u32);
            let mut acc = 0_u64;
            for &l in &part.lens {
                acc += u64::from(l);
                indptr.push(u32::try_from(acc).expect("a batch holds fewer than 2^32 values"));
            }
            let counts = CsrMatrix::new(indptr, part.ix, part.vals, n, genes[b].len())
                .expect("a valid batch matrix");
            let bc = BatchCounts {
                totals: part.rows.iter().map(|&r| f64::from(totals[r])).collect(),
                counts,
                rows: part.rows,
            };
            results.push(score_batch(b as u32, &bc, opts, &scores_m, &calls_m));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_and_scipy_helpers() {
        let mut v = vec![3.0, 1.0, 2.0, 10.0];
        assert!((percentile(&mut v, 85.0) - 6.85).abs() < 1e-12);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let (h, c) = histogram(&[0.0, 0.1, 0.5, 1.0, 1.0], 2);
        assert_eq!(h, vec![2, 3]);
        assert_eq!(c, vec![0.25, 0.75]);
        let x = fmin_1d(|b| (b - 2.5).abs(), 0.1);
        assert!((x - 2.5).abs() < 1e-3, "{x}");
        assert_eq!(
            (
                round_half_even(2.5),
                round_half_even(3.5),
                round_half_even(2.4)
            ),
            (2, 4, 2)
        );
    }

    #[test]
    fn threshold_minimum_finds_the_valley_between_two_modes() {
        let mut v = Vec::new();
        for i in 0..1000 {
            v.push(0.1 + 0.05 * (f64::from(i % 50) / 50.0));
            if i % 3 == 0 {
                v.push(0.7 + 0.1 * (f64::from(i % 40) / 40.0));
            }
        }
        let t = threshold_minimum(&v).unwrap();
        assert!(t > 0.15 && t < 0.7, "{t}");
        assert!(threshold_minimum(&[0.5; 100]).is_none());
    }

    /// The simulated doublets of Kang et al.'s control capture (29,238 of
    /// them, as counts per neighbour count, k = 180): the reference's rule
    /// splits off the two highest scores at 0.68, leaving 0.003% above it;
    /// the peeled rule finds the valley the reference Scrublet found on its
    /// own simulation of the same data (0.16).
    #[test]
    fn a_split_leaving_a_handful_above_is_peeled() {
        const LEVELS: &[(u32, usize)] = &[
            (27, 1),
            (28, 1),
            (29, 1),
            (32, 1),
            (34, 1),
            (35, 2),
            (36, 2),
            (38, 1),
            (39, 2),
            (41, 3),
            (42, 3),
            (43, 1),
            (44, 1),
            (45, 1),
            (46, 6),
            (47, 1),
            (48, 2),
            (50, 7),
            (51, 3),
            (52, 3),
            (53, 7),
            (54, 7),
            (55, 4),
            (56, 6),
            (57, 8),
            (58, 5),
            (59, 11),
            (60, 10),
            (61, 10),
            (62, 4),
            (63, 9),
            (64, 10),
            (65, 11),
            (66, 13),
            (67, 12),
            (68, 18),
            (69, 12),
            (70, 12),
            (71, 17),
            (72, 24),
            (73, 24),
            (74, 23),
            (75, 9),
            (76, 13),
            (77, 27),
            (78, 23),
            (79, 21),
            (80, 16),
            (81, 27),
            (82, 27),
            (83, 21),
            (84, 29),
            (85, 27),
            (86, 34),
            (87, 34),
            (88, 30),
            (89, 43),
            (90, 41),
            (91, 41),
            (92, 36),
            (93, 50),
            (94, 44),
            (95, 49),
            (96, 60),
            (97, 56),
            (98, 49),
            (99, 75),
            (100, 70),
            (101, 62),
            (102, 73),
            (103, 72),
            (104, 84),
            (105, 76),
            (106, 85),
            (107, 76),
            (108, 74),
            (109, 83),
            (110, 83),
            (111, 93),
            (112, 98),
            (113, 108),
            (114, 99),
            (115, 117),
            (116, 144),
            (117, 129),
            (118, 144),
            (119, 123),
            (120, 132),
            (121, 152),
            (122, 148),
            (123, 143),
            (124, 144),
            (125, 153),
            (126, 169),
            (127, 162),
            (128, 178),
            (129, 191),
            (130, 176),
            (131, 184),
            (132, 213),
            (133, 192),
            (134, 198),
            (135, 205),
            (136, 207),
            (137, 232),
            (138, 217),
            (139, 248),
            (140, 259),
            (141, 249),
            (142, 284),
            (143, 291),
            (144, 304),
            (145, 314),
            (146, 331),
            (147, 328),
            (148, 354),
            (149, 360),
            (150, 388),
            (151, 378),
            (152, 395),
            (153, 451),
            (154, 439),
            (155, 456),
            (156, 543),
            (157, 547),
            (158, 586),
            (159, 582),
            (160, 631),
            (161, 681),
            (162, 814),
            (163, 879),
            (164, 978),
            (165, 1113),
            (166, 1133),
            (167, 1193),
            (168, 1202),
            (169, 1235),
            (170, 1166),
            (171, 1076),
            (172, 931),
            (173, 703),
            (174, 467),
            (175, 280),
            (176, 130),
            (177, 52),
            (178, 14),
            (179, 1),
            (180, 1),
        ];
        let (n, rho, r) = (180.0_f64, 0.05, 2.0);
        let score = |nd: u32| {
            let q = (f64::from(nd) + 1.0) / (n + 2.0);
            q * rho / r / (1.0 - rho - q * (1.0 - rho - rho / r))
        };
        let v: Vec<f64> = LEVELS
            .iter()
            .flat_map(|&(nd, c)| std::iter::repeat_n(score(nd), c))
            .collect();
        let plain = threshold_minimum(&v).unwrap();
        assert!(
            (plain - 0.6836).abs() < 1e-3,
            "the reference's rule on these scores: {plain}"
        );
        let peeled = threshold_from_simulated(&v).unwrap();
        assert!(peeled > 0.14 && peeled < 0.2, "{peeled}");
    }

    #[test]
    fn exact_neighbours_exclude_self_and_break_ties_by_index() {
        // Points on a line: 0, 1, 2, 3, 10.
        let p = [0.0_f32, 1.0, 2.0, 3.0, 10.0];
        let knn = knn_exact(&p, 5, 1, 2);
        assert_eq!(&knn[0..2], &[1, 2]);
        assert_eq!(&knn[2..4], &[0, 2]); // 1: 0 and 2 tie, lower index first
        assert_eq!(&knn[8..10], &[3, 2]);
    }

    /// Two cell types; doublets of the two are simulated by adding a cell of
    /// each, and must score above both types' singlets.
    #[test]
    fn heterotypic_doublets_score_high() {
        let (n_a, n_b, n_d, genes) = (300, 300, 30, 60);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut dense: Vec<Vec<f32>> = Vec::new();
        let cell = |rng: &mut ChaCha8Rng, hi: std::ops::Range<usize>| -> Vec<f32> {
            (0..genes)
                .map(|j| {
                    let lam = if hi.contains(&j) { 8.0_f64 } else { 0.3 };
                    // Poisson by inversion.
                    let (mut k, mut p, l) = (0.0_f32, 1.0_f64, (-lam).exp());
                    loop {
                        p *= rng.random::<f64>();
                        if p <= l {
                            break;
                        }
                        k += 1.0;
                    }
                    k
                })
                .collect()
        };
        for _ in 0..n_a {
            dense.push(cell(&mut rng, 0..20));
        }
        for _ in 0..n_b {
            dense.push(cell(&mut rng, 20..40));
        }
        for _ in 0..n_d {
            let a = cell(&mut rng, 0..20);
            let b = cell(&mut rng, 20..40);
            dense.push(a.iter().zip(&b).map(|(x, y)| x + y).collect());
        }
        let n = dense.len();
        let flat: Vec<f32> = dense.into_iter().flatten().collect();
        let m = CsrMatrix::from_dense_row_major(&flat, n, genes);
        let batch = vec![0_u32; n];
        let (mut scores, mut calls) = (vec![0.0_f32; n], vec![false; n]);
        let opts = ScrubletOptions {
            n_prin_comps: 10,
            min_gene_variability_pctl: 0.0,
            ..Default::default()
        };
        let res = scrublet_blocks(&m, &batch, 1, &opts, &mut scores, &mut calls);
        assert_eq!(res.len(), 1);
        let singlets: f32 = scores[..n_a + n_b].iter().sum::<f32>() / (n_a + n_b) as f32;
        let doublets: f32 = scores[n_a + n_b..].iter().sum::<f32>() / n_d as f32;
        assert!(
            doublets > 3.0 * singlets,
            "doublets {doublets} vs singlets {singlets}"
        );
        assert!(res[0].n_sim == 2 * n);
    }
}
