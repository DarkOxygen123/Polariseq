#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
// Single letters follow the paper's notation (n cells, d dimensions, k
// clusters, b batches, r assignments, e expected and o observed tallies, y
// centroids, w ridge coefficients); exact float comparisons (zero weights)
// are deliberate.

//! Harmony batch correction (Korsunsky et al., "Fast, sensitive and accurate
//! integration of single-cell data with Harmony", Nature Methods 16, 1289,
//! 2019).
//!
//! Cells are assigned softly to `K` clusters on the unit sphere. The
//! assignment favours clusters that hold every batch in its overall
//! proportion (the diversity penalty, `theta`). Within each cluster a ridge
//! regression on the batch indicators estimates how far each batch sits from
//! the cluster's centre, and every cell is moved back by those offsets,
//! weighted by its assignments. Clustering and correction alternate until the
//! objective stops falling.
//!
//! The algorithm follows the paper (Methods and Supplementary Note) and
//! harmony-pytorch (https://github.com/lilab-bcb/harmony-pytorch, BSD-3-Clause,
//! Copyright (c) 2019-2020 The Broad Institute, Inc. and The General Hospital
//! Corporation; see `THIRD_PARTY_NOTICES.md`): k-means for the starting
//! centroids, assignment rounds over shuffled blocks of cells with the
//! tallies of the other cells fixed, the objective and its convergence
//! windows, and the mixture-of-experts ridge correction. The options R
//! harmony documents (`harmony_options`, `RunHarmony`) are supported as
//! documented: a diversity penalty and a ridge penalty per batch variable,
//! the ridge penalty estimated as `alpha` times a batch's expected number of
//! cells in a cluster when none is given, and `batch_prop_cutoff`.
//!
//! The data layout is our own. The embedding is read where it lies (often a
//! memory-mapped result file) and the corrected one is written into the
//! caller's buffer; the soft assignments (`n × K` `f32`) are the only
//! working array the size of the data, distances to the centroids are
//! computed a tile of cells at a time, and the shuffled order is an index
//! list. Tallies, ridge systems and the objective are kept in `f64`. One
//! `ChaCha8` stream draws every random choice, and sums over cells run in
//! fixed chunks merged in order, so the result does not depend on the number
//! of threads.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Cells per tile of centroid dot products (a `K × TILE` block).
const TILE: usize = 1024;
/// At most this many chunks per pass over the cells: their partial sums
/// merge in order, and their number is fixed by the cell count alone.
const MAX_PARTS: usize = 64;
/// Cells per task when summing the embedding by cluster and batch.
const PIECE: usize = 4096;
/// Width of the convergence window of the assignment rounds.
const WINDOW: usize = 3;
/// Most Lloyd iterations refining the starting centroids.
const INIT_LLOYD: usize = 25;

/// One batch variable: a level code per cell, each below `n_levels`.
#[derive(Debug, Clone, Copy)]
pub struct Covariate<'a> {
    /// The level of every cell.
    pub codes: &'a [u32],
    /// Number of levels.
    pub n_levels: usize,
}

/// Options for Harmony, with harmonypy's defaults.
#[derive(Debug, Clone)]
pub struct HarmonyOptions {
    /// Number of clusters; `None` is `min(round(n / 30), 100)`.
    pub n_clusters: Option<usize>,
    /// Diversity penalty, one per batch variable (or one for all). Larger
    /// values push harder for clusters that mix the batches.
    pub theta: Vec<f32>,
    /// Width of the soft assignment: smaller is closer to hard k-means.
    pub sigma: f32,
    /// Ridge penalty, one per batch variable (or one for all); `None`
    /// estimates it per cluster and batch as `alpha` times the expected
    /// number of cells.
    pub lambda: Option<Vec<f32>>,
    /// Scale of the estimated ridge penalty.
    pub alpha: f32,
    /// Protection of small batches against overcorrection (0 = off).
    pub tau: f32,
    /// Fraction of the cells reassigned at a time.
    pub block_size: f32,
    /// Maximum rounds of clustering and correction.
    pub max_iter_harmony: usize,
    /// Assignment rounds per clustering.
    pub max_iter_kmeans: usize,
    /// Convergence threshold of the assignment rounds.
    pub epsilon_cluster: f32,
    /// Convergence threshold of the rounds of clustering and correction.
    pub epsilon_harmony: f32,
    /// A batch whose share of a cluster is at most this is left alone there.
    pub batch_prop_cutoff: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for HarmonyOptions {
    fn default() -> Self {
        Self {
            n_clusters: None,
            theta: vec![2.0],
            sigma: 0.1,
            lambda: None,
            alpha: 0.2,
            tau: 0.0,
            block_size: 0.05,
            max_iter_harmony: 10,
            max_iter_kmeans: 4,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            batch_prop_cutoff: 1e-5,
            seed: 0,
        }
    }
}

/// What a run did.
#[derive(Debug, Clone)]
pub struct HarmonyReport {
    /// Number of clusters.
    pub n_clusters: usize,
    /// The objective after initialisation and after each clustering.
    pub objective: Vec<f64>,
    /// Assignment rounds of each clustering.
    pub kmeans_rounds: Vec<usize>,
    /// Whether the objective stopped falling before the iteration limit.
    pub converged: bool,
}

/// Errors raised by Harmony.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum HarmonyError {
    /// The embedding has zero rows or dimensions.
    #[error("empty embedding: {nrows} rows, {ndims} dims")]
    EmptyInput {
        /// Row count.
        nrows: usize,
        /// Dimension count.
        ndims: usize,
    },
    /// The number of batch labels doesn't match the embedding.
    #[error("batch label count ({n_labels}) != n_obs ({n_obs})")]
    LabelMismatch {
        /// Number of labels.
        n_labels: usize,
        /// Number of observations.
        n_obs: usize,
    },
    /// No batch variable has two levels with cells.
    #[error("only one batch present — batch correction is a no-op")]
    SingleBatch,
    /// A level code at or above its variable's number of levels.
    #[error("batch variable {covariate}: level {code} with {n_levels} levels")]
    BadLevel {
        /// The variable.
        covariate: usize,
        /// The offending code.
        code: u32,
        /// Its number of levels.
        n_levels: usize,
    },
    /// An option or an input outside its range.
    #[error("{0}")]
    BadOption(String),
    /// The optimisation produced a non-finite or non-positive quantity.
    #[error("Harmony numerical error during {0}")]
    Numerical(String),
}

/// harmonypy's default number of clusters: `min(round(n / 30), 100)`, with
/// Python's rounding (half to even), and at least one.
#[must_use]
pub fn default_n_clusters(n_obs: usize) -> usize {
    ((n_obs as f64 / 30.0).round_ties_even() as usize).clamp(1, 100)
}

/// Run Harmony on a single batch variable; returns the corrected embedding.
///
/// `embedding` is `(n_obs × n_pcs)` row-major, `batch_labels` a batch code
/// per cell (codes `0..=max`).
///
/// # Errors
/// Returns [`HarmonyError`] on invalid inputs or a numerical failure.
pub fn harmony(
    embedding: &[f32],
    batch_labels: &[u32],
    n_obs: usize,
    n_pcs: usize,
    opts: &HarmonyOptions,
) -> Result<Vec<f32>, HarmonyError> {
    let n_levels = batch_labels.iter().max().map_or(0, |&m| m as usize + 1);
    let mut out = vec![0.0_f32; n_obs * n_pcs];
    harmony_into(
        embedding,
        n_obs,
        n_pcs,
        &[Covariate {
            codes: batch_labels,
            n_levels,
        }],
        opts,
        &mut out,
    )?;
    Ok(out)
}

/// Run Harmony on `z` (`n_obs × n_dims`, row-major), correcting for every
/// variable in `covariates` at once, and write the corrected embedding into
/// `out` (same shape).
///
/// # Errors
/// Returns [`HarmonyError`] on invalid inputs or a numerical failure.
pub fn harmony_into(
    z: &[f32],
    n_obs: usize,
    n_dims: usize,
    covariates: &[Covariate<'_>],
    opts: &HarmonyOptions,
    out: &mut [f32],
) -> Result<HarmonyReport, HarmonyError> {
    if out.len() != n_obs * n_dims {
        return Err(HarmonyError::BadOption(format!(
            "the output holds {} values for {n_obs} × {n_dims}",
            out.len()
        )));
    }
    Harmony::new(z, n_obs, n_dims, covariates, opts)?.run(out)
}

/// Assignment sums over a set of cells: `Σ_j R_jk`, the observed tallies
/// `O_bk = Σ_{j ∈ b} R_jk`, and, when the cells were just assigned, the
/// objective's distance and entropy terms.
struct Tally {
    rsum: Vec<f64>,
    o: Vec<f64>,
    kerr: f64,
    ent: f64,
}

impl Tally {
    fn new(k: usize, b: usize) -> Self {
        Self {
            rsum: vec![0.0; k],
            o: vec![0.0; b * k],
            kerr: 0.0,
            ent: 0.0,
        }
    }

    fn add(&mut self, row: &[f32], batches: &[u32]) {
        let k = row.len();
        for (s, &r) in self.rsum.iter_mut().zip(row) {
            *s += f64::from(r);
        }
        for &b in batches {
            let o = &mut self.o[b as usize * k..(b as usize + 1) * k];
            for (s, &r) in o.iter_mut().zip(row) {
                *s += f64::from(r);
            }
        }
    }

    fn merge(&mut self, other: &Tally) {
        for (s, x) in self.rsum.iter_mut().zip(&other.rsum) {
            *s += x;
        }
        for (s, x) in self.o.iter_mut().zip(&other.o) {
            *s += x;
        }
        self.kerr += other.kerr;
        self.ent += other.ent;
    }
}

/// Cells grouped by their levels across all batch variables.
struct Combos {
    n_cov: usize,
    /// Batches of each group, `m × n_cov`.
    levels: Vec<u32>,
    /// Cells sorted by group, ascending within one.
    cells: Vec<u32>,
    /// Where each group's cells start in `cells` (`m + 1` entries).
    starts: Vec<usize>,
}

impl Combos {
    fn new(ids: &[u32], n: usize, n_cov: usize, n_batches: usize) -> Self {
        let mut cells: Vec<u32> = (0..n as u32).collect();
        if n_cov == 1 {
            let mut start = vec![0_usize; n_batches + 1];
            for &b in ids {
                start[b as usize + 1] += 1;
            }
            for b in 0..n_batches {
                start[b + 1] += start[b];
            }
            for (j, &b) in ids.iter().enumerate() {
                cells[start[b as usize]] = j as u32;
                start[b as usize] += 1;
            }
        } else {
            let key = |j: u32| &ids[j as usize * n_cov..(j as usize + 1) * n_cov];
            cells.par_sort_unstable_by(|&a, &b| key(a).cmp(key(b)).then(a.cmp(&b)));
        }
        let mut levels = Vec::new();
        let mut starts = Vec::new();
        for (i, &j) in cells.iter().enumerate() {
            let key = &ids[j as usize * n_cov..(j as usize + 1) * n_cov];
            let prev = (i > 0)
                .then(|| &ids[cells[i - 1] as usize * n_cov..(cells[i - 1] as usize + 1) * n_cov]);
            if prev != Some(key) {
                starts.push(i);
                levels.extend_from_slice(key);
            }
        }
        starts.push(n);
        Self {
            n_cov,
            levels,
            cells,
            starts,
        }
    }

    fn len(&self) -> usize {
        self.starts.len() - 1
    }

    fn levels(&self, m: usize) -> &[u32] {
        &self.levels[m * self.n_cov..(m + 1) * self.n_cov]
    }
}

/// Sums of the original embedding weighted by the assignments.
struct Sums {
    /// `Σ_{j ∈ b} R_jk z_j`, `B × K × d`.
    s: Vec<f64>,
    /// `Σ_{j ∈ m} R_jk` per group of cells, `m × K` (several variables only).
    f: Vec<f64>,
    /// `Σ_j R_jk z_j` over the cells with a batch the cluster corrects,
    /// `K × d` (several variables only).
    z_all: Vec<f64>,
}

/// The ridge solution of one cluster: its intercept and the offset of each
/// batch it corrects, in the order of those batches.
struct Ridge {
    intercept: Vec<f64>,
    offsets: Vec<f64>,
}

struct Harmony<'a> {
    z: &'a [f32],
    n: usize,
    d: usize,
    k: usize,
    b: usize,
    n_cov: usize,
    /// Batch of every cell for every variable, `n × n_cov`, numbered across
    /// all variables.
    ids: Vec<u32>,
    /// The variable each batch belongs to.
    cov_of: Vec<usize>,
    /// Cells per batch, and their share of all cells.
    count: Vec<f64>,
    pr: Vec<f64>,
    theta: Vec<f64>,
    lambda: Option<Vec<f64>>,
    alpha: f64,
    sigma: f64,
    cutoff: f64,
    block_size: f32,
    max_rounds: usize,
    max_iter_kmeans: usize,
    eps_cluster: f64,
    eps_rounds: f64,
    /// Centroids on the unit sphere, `K × d`.
    y: Vec<f32>,
    /// Soft assignments, `n × K`.
    r: Vec<f32>,
    /// `Σ_j R_jk`; the expected tallies are `E_bk = rsum_k · pr_b`.
    rsum: Vec<f64>,
    /// Observed tallies `O_bk = Σ_{j ∈ b} R_jk`, `B × K`.
    o: Vec<f64>,
    combos: Combos,
    rng: ChaCha8Rng,
    obj_kmeans: Vec<f64>,
    obj_rounds: Vec<f64>,
    rounds: Vec<usize>,
    trace: bool,
}

impl<'a> Harmony<'a> {
    fn new(
        z: &'a [f32],
        n: usize,
        d: usize,
        covariates: &[Covariate<'_>],
        opts: &HarmonyOptions,
    ) -> Result<Self, HarmonyError> {
        let bad = |s: String| Err(HarmonyError::BadOption(s));
        if n == 0 || d == 0 {
            return Err(HarmonyError::EmptyInput { nrows: n, ndims: d });
        }
        if z.len() != n * d {
            return bad(format!(
                "the embedding holds {} values for {n} × {d}",
                z.len()
            ));
        }
        if covariates.is_empty() {
            return bad("no batch variable".into());
        }
        let n_cov = covariates.len();
        for (c, cov) in covariates.iter().enumerate() {
            if cov.codes.len() != n {
                return Err(HarmonyError::LabelMismatch {
                    n_labels: cov.codes.len(),
                    n_obs: n,
                });
            }
            if let Some(&code) = cov.codes.iter().find(|&&x| x as usize >= cov.n_levels) {
                return Err(HarmonyError::BadLevel {
                    covariate: c,
                    code,
                    n_levels: cov.n_levels,
                });
            }
        }
        if !z.par_iter().all(|x| x.is_finite()) {
            return bad("the embedding has non-finite values".into());
        }
        let per_cov = |v: &[f32], what: &str| -> Result<Vec<f64>, HarmonyError> {
            match v.len() {
                1 => Ok(vec![f64::from(v[0]); n_cov]),
                l if l == n_cov => Ok(v.iter().map(|&x| f64::from(x)).collect()),
                l => Err(HarmonyError::BadOption(format!(
                    "{l} values of {what} for {n_cov} batch variables"
                ))),
            }
        };
        let theta_cov = per_cov(&opts.theta, "theta")?;
        let lambda_cov = opts
            .lambda
            .as_ref()
            .map(|l| per_cov(l, "lambda"))
            .transpose()?;
        let finite_nonneg = |x: f64| x.is_finite() && x >= 0.0;
        if !theta_cov.iter().all(|&x| finite_nonneg(x))
            || !lambda_cov
                .as_ref()
                .is_none_or(|l| l.iter().all(|&x| finite_nonneg(x)))
            || !finite_nonneg(f64::from(opts.alpha))
            || !finite_nonneg(f64::from(opts.tau))
        {
            return bad("theta, lambda, alpha and tau must be finite and non-negative".into());
        }
        if !(opts.sigma.is_finite() && opts.sigma > 0.0) {
            return bad(format!("sigma must be positive, got {}", opts.sigma));
        }
        if !(opts.block_size > 0.0 && opts.block_size <= 1.0) {
            return bad(format!(
                "block_size must be in (0, 1], got {}",
                opts.block_size
            ));
        }
        if opts.max_iter_harmony == 0 {
            return bad("max_iter_harmony must be at least 1".into());
        }
        let k = opts.n_clusters.unwrap_or_else(|| default_n_clusters(n));
        if k == 0 || k > n {
            return bad(format!("{k} clusters for {n} cells"));
        }

        // Batches numbered across the variables.
        let mut offset = Vec::with_capacity(n_cov);
        let mut b = 0;
        let mut cov_of = Vec::new();
        for (c, cov) in covariates.iter().enumerate() {
            offset.push(b as u32);
            b += cov.n_levels;
            cov_of.extend(std::iter::repeat_n(c, cov.n_levels));
        }
        let mut ids = vec![0_u32; n * n_cov];
        for (c, cov) in covariates.iter().enumerate() {
            for (j, &code) in cov.codes.iter().enumerate() {
                ids[j * n_cov + c] = offset[c] + code;
            }
        }
        let mut count = vec![0.0_f64; b];
        for &x in &ids {
            count[x as usize] += 1.0;
        }
        let two_levels = covariates.iter().enumerate().any(|(c, cov)| {
            let o = offset[c] as usize;
            count[o..o + cov.n_levels]
                .iter()
                .filter(|&&x| x > 0.0)
                .count()
                >= 2
        });
        if !two_levels {
            return Err(HarmonyError::SingleBatch);
        }
        let pr: Vec<f64> = count.iter().map(|&x| x / n as f64).collect();
        // tau discounts the penalty of batches with few cells per cluster.
        let tau = f64::from(opts.tau);
        let theta: Vec<f64> = (0..b)
            .map(|bb| {
                let t = theta_cov[cov_of[bb]];
                if tau > 0.0 {
                    t * (1.0 - (-(count[bb] / (k as f64 * tau)).powi(2)).exp())
                } else {
                    t
                }
            })
            .collect();
        let lambda = lambda_cov.map(|l| (0..b).map(|bb| l[cov_of[bb]]).collect());
        let combos = Combos::new(&ids, n, n_cov, b);
        Ok(Self {
            z,
            n,
            d,
            k,
            b,
            n_cov,
            ids,
            cov_of,
            count,
            pr,
            theta,
            lambda,
            alpha: f64::from(opts.alpha),
            sigma: f64::from(opts.sigma),
            cutoff: f64::from(opts.batch_prop_cutoff),
            block_size: opts.block_size,
            max_rounds: opts.max_iter_harmony,
            max_iter_kmeans: opts.max_iter_kmeans,
            eps_cluster: f64::from(opts.epsilon_cluster),
            eps_rounds: f64::from(opts.epsilon_harmony),
            y: Vec::new(),
            r: vec![0.0; n * k],
            rsum: vec![0.0; k],
            o: vec![0.0; b * k],
            combos,
            rng: ChaCha8Rng::seed_from_u64(opts.seed),
            obj_kmeans: Vec::new(),
            obj_rounds: Vec::new(),
            rounds: Vec::new(),
            trace: std::env::var_os("POLARISEQ_TRACE").is_some(),
        })
    }

    fn numerical(&self, stage: &str, what: &str) -> HarmonyError {
        HarmonyError::Numerical(format!(
            "{stage}: {what} (sigma={}, theta={:?}, block_size={}, K={}, N={})",
            self.sigma, self.theta, self.block_size, self.k, self.n
        ))
    }

    /// The algorithm of Korsunsky et al. (2019): cells on the unit sphere;
    /// starting centroids by k-means; then rounds of clustering (soft
    /// assignments with the diversity penalty) and correction (a ridge
    /// regression per cluster), until the objective stops falling. After a
    /// correction the centroids are the clusters' ridge intercepts, and each
    /// round starts from assignments by distance alone; the current versions
    /// of Harmony (R harmony 1.x, harmonypy 2) behave so, and with it the
    /// corrections agree with harmonypy's as its own seeds agree
    /// (`benchmarks/relicence/harmony_compare.py` on the reproduce branch).
    fn run(mut self, zc: &mut [f32]) -> Result<HarmonyReport, HarmonyError> {
        normalize_rows_into(self.z, zc, self.d);
        self.y = kmeans_pp(zc, self.n, self.d, self.k, &mut self.rng);
        normalize_rows(&mut self.y, self.d);
        let (kerr, ent) = self.assign_all(zc, false, "initialization")?;
        let obj = self.objective(kerr, ent, "initialization")?;
        self.obj_kmeans.push(obj);
        self.obj_rounds.push(obj);
        let mut converged = false;
        for it in 1..=self.max_rounds {
            if it > 1 {
                normalize_rows(zc, self.d);
                self.assign_all(zc, false, "cluster initialization")?;
            }
            self.cluster(zc)?;
            self.correct(zc)?;
            let h = &self.obj_rounds;
            if self.trace {
                eprintln!(
                    "[harmony] round {it}: objective {:.6e} after {} assignment rounds",
                    h[h.len() - 1],
                    self.rounds[self.rounds.len() - 1]
                );
            }
            if relative_drop(h[h.len() - 2], h[h.len() - 1]) < self.eps_rounds {
                converged = true;
                break;
            }
        }
        Ok(HarmonyReport {
            n_clusters: self.k,
            objective: self.obj_rounds,
            kmeans_rounds: self.rounds,
            converged,
        })
    }

    /// The objective (the paper's eq. 1, cosine distances): `Σ R_jk ‖z_j − y_k‖²
    /// + σ Σ R_jk ln R_jk + σ Σ θ_b O_bk ln((O + E + 1)/(2E + 1))`, the last
    /// the diversity term of the weight in [`Self::penalty`].
    fn objective(&self, kerr: f64, ent: f64, stage: &str) -> Result<f64, HarmonyError> {
        let mut cross = 0.0;
        for bb in 0..self.b {
            for kk in 0..self.k {
                let e = self.rsum[kk] * self.pr[bb];
                let o = self.o[bb * self.k + kk];
                cross += self.theta[bb] * o * ((o + e + 1.0) / (2.0 * e + 1.0)).ln();
            }
        }
        let obj = kerr + self.sigma * ent + self.sigma * cross;
        if obj.is_finite() {
            Ok(obj)
        } else {
            Err(self.numerical(stage, "objectives must be finite"))
        }
    }

    /// `ln` of each batch's diversity weight in each cluster, `B × K`, from
    /// the current tallies: `θ_b ln((2E + 1)/(O + E + 1))`. The paper's
    /// `((E + 1)/(O + 1))^θ` gives corrections further from those of the
    /// current versions of Harmony than their own seed-to-seed spread (fetal
    /// atlas, 28 donors: 0.45 against 0.31), this form within it (0.28).
    fn penalty(&self) -> Vec<f32> {
        let k = self.k;
        let mut p = vec![0.0_f32; self.b * k];
        for bb in 0..self.b {
            for kk in 0..k {
                let e = self.rsum[kk] * self.pr[bb];
                let o = self.o[bb * k + kk];
                p[bb * k + kk] =
                    (self.theta[bb] * ((2.0 * e + 1.0).ln() - (o + e + 1.0).ln())) as f32;
            }
        }
        p
    }

    /// Every cell's assignments, with (`penalized`) or without the diversity
    /// penalty of the current tallies, and the tallies recomputed; returns the
    /// objective's distance and entropy terms.
    fn assign_all(
        &mut self,
        zc: &[f32],
        penalized: bool,
        stage: &str,
    ) -> Result<(f64, f64), HarmonyError> {
        let (k, d, b, nc) = (self.k, self.d, self.b, self.n_cov);
        let chunk = chunk_len(self.n);
        let pen = penalized.then(|| self.penalty());
        let y = MatRef::from_row_major_slice(&self.y, k, d);
        let sigma = self.sigma as f32;
        let ids = &self.ids;
        let parts: Vec<Option<Tally>> = self
            .r
            .par_chunks_mut(chunk * k)
            .zip(zc.par_chunks(chunk * d))
            .enumerate()
            .map(|(ci, (rc, xc))| {
                let m = rc.len() / k;
                let mut t = Tally::new(k, b);
                let mut pj = vec![0.0_f32; k];
                for t0 in (0..m).step_by(TILE) {
                    let mt = TILE.min(m - t0);
                    let dots = centroid_dots(y, &xc[t0 * d..(t0 + mt) * d], mt, d);
                    for tt in 0..mt {
                        let j = ci * chunk + t0 + tt;
                        let batches = &ids[j * nc..(j + 1) * nc];
                        let p = pen
                            .as_ref()
                            .map(|pen| cell_penalty(pen, batches, k, &mut pj));
                        let row = &mut rc[(t0 + tt) * k..(t0 + tt + 1) * k];
                        let (ke, en) = assign_cell(dots.col_as_slice(tt), p, sigma, row)?;
                        t.kerr += ke;
                        t.ent += en;
                        t.add(row, batches);
                    }
                }
                Some(t)
            })
            .collect();
        let mut total = Tally::new(k, b);
        for p in &parts {
            match p {
                Some(p) => total.merge(p),
                None => {
                    return Err(
                        self.numerical(stage, "assignment normalizers must be finite and positive")
                    )
                }
            }
        }
        self.rsum = total.rsum;
        self.o = total.o;
        Ok((total.kerr, total.ent))
    }

    /// The clustering of one round: assignment rounds, each over the cells in
    /// a fresh random order a block at a time, until the objective, summed
    /// over a window of three rounds, falls by less than `epsilon_cluster`.
    fn cluster(&mut self, zc: &[f32]) -> Result<(), HarmonyError> {
        let mut rounds = 0;
        let mut here: Vec<f64> = Vec::new();
        for i in 0..self.max_iter_kmeans {
            let (kerr, ent) = self.update_blocks(zc)?;
            let obj = self.objective(kerr, ent, "assignment update")?;
            self.obj_kmeans.push(obj);
            here.push(obj);
            rounds = i + 1;
            let o = &here;
            if o.len() > WINDOW {
                let old: f64 = o[o.len() - 1 - WINDOW..o.len() - 1].iter().sum();
                let new: f64 = o[o.len() - WINDOW..].iter().sum();
                if relative_drop(old, new) < self.eps_cluster {
                    break;
                }
            }
        }
        self.rounds.push(rounds);
        self.obj_rounds
            .push(self.obj_kmeans[self.obj_kmeans.len() - 1]);
        Ok(())
    }

    /// One assignment round. The cells go in a random order, a block of
    /// `block_size · n` at a time; a block's own assignments are taken out of
    /// the tallies, its cells reassigned against the tallies of all the other
    /// cells, and the new assignments put back. Returns the distance and
    /// entropy terms over all cells.
    fn update_blocks(&mut self, zc: &[f32]) -> Result<(f64, f64), HarmonyError> {
        let (n, k, d, b, nc) = (self.n, self.k, self.d, self.b, self.n_cov);
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.shuffle(&mut self.rng);
        let per_block = ((n as f64 * f64::from(self.block_size)) as usize).max(1);
        let sigma = self.sigma as f32;
        let mut r = std::mem::take(&mut self.r);
        let mut rows: Vec<&mut [f32]> = r.chunks_mut(k).collect();
        let (mut kerr, mut ent) = (0.0, 0.0);
        let mut failed = false;
        for cells in order.chunks(per_block) {
            let mut taken: Vec<&mut [f32]> = cells
                .iter()
                .map(|&j| std::mem::take(&mut rows[j as usize]))
                .collect();
            let sub = chunk_len(cells.len());
            let ids = &self.ids;
            let old: Vec<Tally> = taken
                .par_chunks(sub)
                .zip(cells.par_chunks(sub))
                .map(|(rc, cc)| {
                    let mut t = Tally::new(k, b);
                    for (row, &j) in rc.iter().zip(cc) {
                        t.add(row, &ids[j as usize * nc..(j as usize + 1) * nc]);
                    }
                    t
                })
                .collect();
            for t in &old {
                for (s, x) in self.rsum.iter_mut().zip(&t.rsum) {
                    *s -= x;
                }
                for (s, x) in self.o.iter_mut().zip(&t.o) {
                    *s -= x;
                }
            }
            let pen = self.penalty();
            let y = MatRef::from_row_major_slice(&self.y, k, d);
            let new: Vec<Option<Tally>> = taken
                .par_chunks_mut(sub)
                .zip(cells.par_chunks(sub))
                .map(|(rc, cc)| {
                    let m = cc.len();
                    let mut t = Tally::new(k, b);
                    let mut x = vec![0.0_f32; TILE.min(m) * d];
                    let mut pj = vec![0.0_f32; k];
                    for t0 in (0..m).step_by(TILE) {
                        let mt = TILE.min(m - t0);
                        for (tt, &j) in cc[t0..t0 + mt].iter().enumerate() {
                            let j = j as usize;
                            x[tt * d..(tt + 1) * d].copy_from_slice(&zc[j * d..(j + 1) * d]);
                        }
                        let dots = centroid_dots(y, &x[..mt * d], mt, d);
                        for tt in 0..mt {
                            let j = cc[t0 + tt] as usize;
                            let batches = &ids[j * nc..(j + 1) * nc];
                            let p = cell_penalty(&pen, batches, k, &mut pj);
                            let row = &mut *rc[t0 + tt];
                            let (ke, en) = assign_cell(dots.col_as_slice(tt), Some(p), sigma, row)?;
                            t.kerr += ke;
                            t.ent += en;
                            t.add(row, batches);
                        }
                    }
                    Some(t)
                })
                .collect();
            for t in &new {
                match t {
                    Some(t) => {
                        for (s, x) in self.rsum.iter_mut().zip(&t.rsum) {
                            *s += x;
                        }
                        for (s, x) in self.o.iter_mut().zip(&t.o) {
                            *s += x;
                        }
                        kerr += t.kerr;
                        ent += t.ent;
                    }
                    None => failed = true,
                }
            }
            for (&j, row) in cells.iter().zip(taken) {
                rows[j as usize] = row;
            }
            if failed {
                break;
            }
        }
        drop(rows);
        self.r = r;
        if failed {
            return Err(self.numerical(
                "assignment update",
                "assignment normalizers must be finite and positive",
            ));
        }
        Ok((kerr, ent))
    }

    /// The batches cluster `kk` corrects: those with more than
    /// `batch_prop_cutoff` of their cells in it (R harmony's documented
    /// `batch.prop.cutoff`), within variables keeping at least two batches
    /// (a variable left with one has no difference to remove). Empty when
    /// no variable qualifies.
    fn corrected_batches(&self, kk: usize) -> Vec<usize> {
        let present: Vec<bool> = (0..self.b)
            .map(|bb| {
                self.count[bb] > 0.0 && self.o[bb * self.k + kk] / self.count[bb] > self.cutoff
            })
            .collect();
        let mut per_cov = vec![0_usize; self.n_cov];
        for bb in 0..self.b {
            per_cov[self.cov_of[bb]] += usize::from(present[bb]);
        }
        (0..self.b)
            .filter(|&bb| present[bb] && per_cov[self.cov_of[bb]] >= 2)
            .collect()
    }

    /// The correction (the paper's mixture of experts): per cluster, a
    /// ridge regression of the original embedding on an intercept and the
    /// indicators of the batches it corrects, each cell weighted by its
    /// assignment; every cell then moves back by its batches' coefficients,
    /// weighted by its assignments.
    fn correct(&mut self, zc: &mut [f32]) -> Result<(), HarmonyError> {
        let (k, d, b) = (self.k, self.d, self.b);
        let keep: Vec<Vec<usize>> = (0..k).map(|kk| self.corrected_batches(kk)).collect();
        let sums = self.cell_sums(&keep);
        let solved: Vec<Option<Ridge>> = (0..k)
            .into_par_iter()
            .map(|kk| self.ridge(kk, &keep[kk], &sums))
            .collect();
        let mut w = vec![0.0_f32; b * k * d];
        let mut corrects = vec![false; b * k];
        for (kk, ridge) in solved.into_iter().enumerate() {
            if keep[kk].is_empty() {
                continue;
            }
            let Some(ridge) = ridge else {
                return Err(self.numerical("correction", "the ridge system is singular"));
            };
            for (t, &v) in ridge.intercept.iter().enumerate() {
                self.y[kk * d + t] = v as f32;
            }
            for (i, &bb) in keep[kk].iter().enumerate() {
                corrects[bb * k + kk] = true;
                for t in 0..d {
                    w[(bb * k + kk) * d + t] = ridge.offsets[i * d + t] as f32;
                }
            }
        }
        self.apply(&w, &corrects, zc)?;
        normalize_rows(&mut self.y, d);
        Ok(())
    }

    /// Cluster `kk`'s ridge regression. With `X` the design (an intercept
    /// and one indicator per corrected batch, over the cells holding at least
    /// one of those batches) and `R` the cluster's assignments, the
    /// coefficients solve `(Xᵀ R X + Λ) W = Xᵀ R Z`, `Λ` diagonal with no
    /// penalty on the intercept and `λ_b` on each batch: the option's value,
    /// or `alpha` times the batch's expected count in the cluster (R
    /// harmony's documented estimate). `Xᵀ R X` has the assignment sums of
    /// the cells in the intercept corner, `O_bk` on the diagonal and its
    /// first row and column, and between batches of two variables the
    /// assignment sum of the cells in both. One variable gives an arrow
    /// matrix, solved through its Schur complement; several, a dense solve.
    fn ridge(&self, kk: usize, keep: &[usize], sums: &Sums) -> Option<Ridge> {
        if keep.is_empty() {
            return None;
        }
        let (k, d, b) = (self.k, self.d, self.b);
        let sz = keep.len() + 1;
        let pen = |bb: usize| match &self.lambda {
            None => self.alpha * self.rsum[kk] * self.pr[bb],
            Some(l) => l[bb],
        };
        let mut rhs = vec![0.0_f64; sz * d];
        for (i, &bb) in keep.iter().enumerate() {
            rhs[(i + 1) * d..(i + 2) * d]
                .copy_from_slice(&sums.s[(bb * k + kk) * d..(bb * k + kk + 1) * d]);
        }
        let w = if self.n_cov == 1 {
            // Arrow matrix [[a, oᵀ], [o, D]]: W_0 = (r_0 − Σ o_b r_b / D_b) / s
            // with s = a − Σ o_b² / D_b, then W_b = (r_b − o_b W_0) / D_b.
            let o: Vec<f64> = keep.iter().map(|&bb| self.o[bb * k + kk]).collect();
            let dg: Vec<f64> = keep.iter().zip(&o).map(|(&bb, &ob)| ob + pen(bb)).collect();
            let a: f64 = o.iter().sum();
            for i in 1..sz {
                for t in 0..d {
                    rhs[t] += rhs[i * d + t];
                }
            }
            let s = a - o.iter().zip(&dg).map(|(ob, db)| ob * ob / db).sum::<f64>();
            let mut w = vec![0.0_f64; sz * d];
            for t in 0..d {
                let mut num = rhs[t];
                for i in 0..sz - 1 {
                    num -= o[i] / dg[i] * rhs[(i + 1) * d + t];
                }
                w[t] = num / s;
            }
            for i in 0..sz - 1 {
                for t in 0..d {
                    w[(i + 1) * d + t] = (rhs[(i + 1) * d + t] - o[i] * w[t]) / dg[i];
                }
            }
            w
        } else {
            let mut row_of = vec![0_usize; b];
            for (i, &bb) in keep.iter().enumerate() {
                row_of[bb] = i + 1;
            }
            let mut g = vec![0.0_f64; sz * sz];
            for m in 0..self.combos.len() {
                let wsum = sums.f[m * k + kk];
                let rows: Vec<usize> = self
                    .combos
                    .levels(m)
                    .iter()
                    .map(|&bb| row_of[bb as usize])
                    .filter(|&r| r != 0)
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                g[0] += wsum;
                for &r1 in &rows {
                    g[r1] += wsum;
                    g[r1 * sz] += wsum;
                    for &r2 in &rows {
                        g[r1 * sz + r2] += wsum;
                    }
                }
            }
            for (i, &bb) in keep.iter().enumerate() {
                g[(i + 1) * sz + i + 1] += pen(bb);
            }
            rhs[..d].copy_from_slice(&sums.z_all[kk * d..(kk + 1) * d]);
            solve_dense(&g, sz, &rhs, d)?
        };
        if !w.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut w = w;
        let offsets = w.split_off(d);
        Some(Ridge {
            intercept: w,
            offsets,
        })
    }

    /// `Σ_{j ∈ b} R_jk z_j` for every batch and cluster, from tasks over the
    /// groups of cells sharing their batches, merged in order; with several
    /// variables also the per-group assignment sums and, per cluster, the
    /// sum over the cells it corrects.
    fn cell_sums(&self, keep: &[Vec<usize>]) -> Sums {
        let (k, d, b, nc) = (self.k, self.d, self.b, self.n_cov);
        let combos = &self.combos;
        let mut tasks = Vec::new();
        for m in 0..combos.len() {
            let (mut lo, hi) = (combos.starts[m], combos.starts[m + 1]);
            while lo < hi {
                let e = (lo + PIECE).min(hi);
                tasks.push((m, lo, e));
                lo = e;
            }
        }
        let parts: Vec<(usize, Vec<f32>, Vec<f64>)> = tasks
            .par_iter()
            .map(|&(m, lo, hi)| {
                let cells = &combos.cells[lo..hi];
                let mt = cells.len();
                let mut rb = vec![0.0_f32; mt * k];
                let mut zb = vec![0.0_f32; mt * d];
                let mut f = vec![0.0_f64; k];
                for (t, &j) in cells.iter().enumerate() {
                    let j = j as usize;
                    let row = &self.r[j * k..(j + 1) * k];
                    rb[t * k..(t + 1) * k].copy_from_slice(row);
                    zb[t * d..(t + 1) * d].copy_from_slice(&self.z[j * d..(j + 1) * d]);
                    for (s, &x) in f.iter_mut().zip(row) {
                        *s += f64::from(x);
                    }
                }
                let mut g = vec![0.0_f32; k * d];
                matmul(
                    MatMut::from_row_major_slice_mut(&mut g, k, d),
                    Accum::Replace,
                    MatRef::from_row_major_slice(&rb, mt, k).transpose(),
                    MatRef::from_row_major_slice(&zb, mt, d),
                    1.0,
                    Par::Seq,
                );
                (m, g, f)
            })
            .collect();

        let multi = nc > 1;
        let mut kept = vec![false; k * b];
        for (kk, ks) in keep.iter().enumerate() {
            for &bb in ks {
                kept[kk * b + bb] = true;
            }
        }
        let mut s = vec![0.0_f64; b * k * d];
        let mut f_all = vec![0.0_f64; if multi { combos.len() * k } else { 0 }];
        let mut z_all = vec![0.0_f64; if multi { k * d } else { 0 }];
        for (m, g, f) in &parts {
            let lv = combos.levels(*m);
            for &bb in lv {
                let dst = &mut s[bb as usize * k * d..(bb as usize + 1) * k * d];
                for (x, &v) in dst.iter_mut().zip(g) {
                    *x += f64::from(v);
                }
            }
            if multi {
                for (x, &v) in f_all[m * k..(m + 1) * k].iter_mut().zip(f) {
                    *x += v;
                }
                for kk in 0..k {
                    if lv.iter().any(|&bb| kept[kk * b + bb as usize]) {
                        for t in 0..d {
                            z_all[kk * d + t] += f64::from(g[kk * d + t]);
                        }
                    }
                }
            }
        }
        Sums { s, f: f_all, z_all }
    }

    /// `zc_j = z_j − Σ_variables Σ_k R_jk W_{b(j), k}`.
    fn apply(&self, w: &[f32], corrects: &[bool], zc: &mut [f32]) -> Result<(), HarmonyError> {
        let (k, d, nc) = (self.k, self.d, self.n_cov);
        let chunk = chunk_len(self.n);
        let ids = &self.ids;
        let ok = zc
            .par_chunks_mut(chunk * d)
            .zip(self.z.par_chunks(chunk * d))
            .zip(self.r.par_chunks(chunk * k))
            .enumerate()
            .all(|(ci, ((out, z), r))| {
                for (t, (o, zj)) in out.chunks_mut(d).zip(z.chunks(d)).enumerate() {
                    let j = ci * chunk + t;
                    let row = &r[t * k..(t + 1) * k];
                    o.copy_from_slice(zj);
                    for &bb in &ids[j * nc..(j + 1) * nc] {
                        let bb = bb as usize;
                        for (kk, &rv) in row.iter().enumerate() {
                            if !corrects[bb * k + kk] {
                                continue;
                            }
                            let wk = &w[(bb * k + kk) * d..(bb * k + kk + 1) * d];
                            for (x, &v) in o.iter_mut().zip(wk) {
                                *x -= rv * v;
                            }
                        }
                    }
                    if !o.iter().all(|x| x.is_finite()) {
                        return false;
                    }
                }
                true
            });
        if ok {
            Ok(())
        } else {
            Err(self.numerical("correction", "corrected coordinates must be finite"))
        }
    }
}

/// Cells per parallel chunk of a pass over `n` cells.
fn chunk_len(n: usize) -> usize {
    n.div_ceil(MAX_PARTS).max(TILE)
}

/// Dot products of `mt` cells (the rows of `x`) with every centroid: a
/// `K × mt` column-major matrix, each cell's products contiguous.
fn centroid_dots(y: MatRef<'_, f32>, x: &[f32], mt: usize, d: usize) -> Mat<f32> {
    let mut dots = Mat::<f32>::zeros(y.nrows(), mt);
    matmul(
        dots.as_mut(),
        Accum::Replace,
        y,
        MatRef::from_row_major_slice(x, mt, d).transpose(),
        1.0,
        Par::Seq,
    );
    dots
}

/// One cell's assignments from its centroid dot products: the logits
/// `−2(1 − y·z)/σ` plus the batch penalties, shifted by their maximum,
/// exponentiated and normalised. Returns the cell's `Σ R·dist` and
/// `Σ R ln R`, or `None` when the normaliser is not finite and positive.
#[inline]
fn assign_cell(
    dots: &[f32],
    pen: Option<&[f32]>,
    sigma: f32,
    row: &mut [f32],
) -> Option<(f64, f64)> {
    let mut top = f32::NEG_INFINITY;
    for (kk, (r, &dot)) in row.iter_mut().zip(dots).enumerate() {
        let mut l = -(2.0 * (1.0 - dot)) / sigma;
        if let Some(p) = pen {
            l += p[kk];
        }
        *r = l;
        top = top.max(l);
    }
    let mut total = 0.0_f32;
    for r in row.iter_mut() {
        *r = (*r - top).exp();
        total += *r;
    }
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let (mut kerr, mut ent) = (0.0_f64, 0.0_f64);
    for (r, &dot) in row.iter_mut().zip(dots) {
        *r /= total;
        let v = f64::from(*r);
        kerr += v * f64::from(2.0 * (1.0 - dot));
        if *r > 0.0 {
            ent += v * v.ln();
        }
    }
    Some((kerr, ent))
}

/// Rows of `src` scaled to unit length into `dst` (a zero row stays zero).
fn normalize_rows_into(src: &[f32], dst: &mut [f32], d: usize) {
    dst.par_chunks_mut(d * TILE)
        .zip(src.par_chunks(d * TILE))
        .for_each(|(o, s)| {
            for (orow, srow) in o.chunks_mut(d).zip(s.chunks(d)) {
                let norm = srow
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                for (a, &x) in orow.iter_mut().zip(srow) {
                    *a = if norm > 0.0 {
                        (f64::from(x) / norm) as f32
                    } else {
                        x
                    };
                }
            }
        });
}

/// Rows of `x` scaled to unit length in place.
fn normalize_rows(x: &mut [f32], d: usize) {
    x.par_chunks_mut(d * TILE).for_each(|c| {
        for row in c.chunks_mut(d) {
            let norm = row
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v = (f64::from(*v) / norm) as f32;
                }
            }
        }
    });
}

/// `(old − new) / |old|`: how much the objective fell, relative to its size
/// (negative when it rose, which counts as converged, as in the paper's
/// procedure and harmony-pytorch).
fn relative_drop(old: f64, new: f64) -> f64 {
    if old == 0.0 {
        return if new == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (old - new) / old.abs()
}

/// A cell's log diversity weights: the sum, over its batch variables, of its
/// batches' rows of `pen` (`B × K`).
#[inline]
fn cell_penalty<'p>(pen: &'p [f32], batches: &[u32], k: usize, buf: &'p mut [f32]) -> &'p [f32] {
    if let [bb] = batches {
        let bb = *bb as usize;
        return &pen[bb * k..(bb + 1) * k];
    }
    buf.fill(0.0);
    for &bb in batches {
        let bb = bb as usize;
        for (s, &v) in buf.iter_mut().zip(&pen[bb * k..(bb + 1) * k]) {
            *s += v;
        }
    }
    buf
}

/// Starting centroids: k-means++ seeding (Arthur & Vassilvitskii 2007) on the
/// normalized cells, then Lloyd iterations until no cell changes cluster or
/// [`INIT_LLOYD`] passes. Seeding draws each next centre with probability
/// proportional to the squared distance to the nearest centre so far; the
/// draw walks chunk sums in a fixed order, so the result does not depend on
/// the number of threads.
fn kmeans_pp(x: &[f32], n: usize, d: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut y = vec![0.0_f32; k * d];
    let first = rng.random_range(0..n);
    y[..d].copy_from_slice(&x[first * d..(first + 1) * d]);
    let chunk = chunk_len(n);
    let mut nearest = vec![f32::INFINITY; n];
    for c in 1..k {
        let prev = &y[(c - 1) * d..c * d];
        let sums: Vec<f64> = nearest
            .par_chunks_mut(chunk)
            .zip(x.par_chunks(chunk * d))
            .map(|(nc, xc)| {
                let mut s = 0.0_f64;
                for (best, row) in nc.iter_mut().zip(xc.chunks(d)) {
                    let dist: f32 = row.iter().zip(prev).map(|(a, b)| (a - b) * (a - b)).sum();
                    if dist < *best {
                        *best = dist;
                    }
                    s += f64::from(*best);
                }
                s
            })
            .collect();
        let total: f64 = sums.iter().sum();
        let mut pick = rng.random::<f64>() * total;
        let mut j = n - 1;
        'walk: for (ci, &s) in sums.iter().enumerate() {
            if pick < s {
                for (t, &v) in nearest[ci * chunk..(ci * chunk + chunk).min(n)]
                    .iter()
                    .enumerate()
                {
                    pick -= f64::from(v);
                    if pick < 0.0 {
                        j = ci * chunk + t;
                        break 'walk;
                    }
                }
            } else {
                pick -= s;
            }
        }
        y[c * d..(c + 1) * d].copy_from_slice(&x[j * d..(j + 1) * d]);
    }
    for _ in 0..INIT_LLOYD {
        if !lloyd_step(x, n, d, k, &mut y) {
            break;
        }
    }
    y
}

/// One Lloyd iteration: each cell to its nearest centroid, each centroid to
/// the mean of its cells; an empty cluster is moved to the cell farthest from
/// its centroid. Returns whether any cell changed cluster since the centroids
/// were last computed (always true the first time).
fn lloyd_step(x: &[f32], n: usize, d: usize, k: usize, y: &mut [f32]) -> bool {
    type Part = (Vec<f64>, Vec<u64>, (f32, usize));
    let norms: Vec<f32> = y.chunks(d).map(|c| c.iter().map(|v| v * v).sum()).collect();
    let chunk = chunk_len(n);
    let parts: Vec<Part> = {
        let yr = MatRef::from_row_major_slice(y, k, d);
        x.par_chunks(chunk * d)
            .enumerate()
            .map(|(ci, xc)| {
                let m = xc.len() / d;
                let mut sums = vec![0.0_f64; k * d];
                let mut counts = vec![0_u64; k];
                let mut far = (f32::NEG_INFINITY, 0_usize);
                for t0 in (0..m).step_by(TILE) {
                    let mt = TILE.min(m - t0);
                    let dots = centroid_dots(yr, &xc[t0 * d..(t0 + mt) * d], mt, d);
                    for tt in 0..mt {
                        let mut best = (f32::INFINITY, 0_usize);
                        for (c, &dot) in dots.col_as_slice(tt).iter().enumerate() {
                            let score = norms[c] - 2.0 * dot;
                            if score < best.0 {
                                best = (score, c);
                            }
                        }
                        let row = &xc[(t0 + tt) * d..(t0 + tt + 1) * d];
                        let self_sq: f32 = row.iter().map(|v| v * v).sum();
                        let dist = (self_sq + best.0).max(0.0);
                        if dist > far.0 {
                            far = (dist, ci * chunk + t0 + tt);
                        }
                        let c = best.1;
                        for (s, &v) in sums[c * d..(c + 1) * d].iter_mut().zip(row) {
                            *s += f64::from(v);
                        }
                        counts[c] += 1;
                    }
                }
                (sums, counts, far)
            })
            .collect()
    };
    let mut sums = vec![0.0_f64; k * d];
    let mut counts = vec![0_u64; k];
    let mut far = (f32::NEG_INFINITY, 0_usize);
    for (s, c, f) in &parts {
        for (a, b) in sums.iter_mut().zip(s) {
            *a += b;
        }
        for (a, b) in counts.iter_mut().zip(c) {
            *a += b;
        }
        if f.0 > far.0 {
            far = *f;
        }
    }
    let before: Vec<f32> = y.to_vec();
    for c in 0..k {
        if counts[c] > 0 {
            for t in 0..d {
                y[c * d + t] = (sums[c * d + t] / counts[c] as f64) as f32;
            }
        } else {
            y[c * d..(c + 1) * d].copy_from_slice(&x[far.1 * d..(far.1 + 1) * d]);
        }
    }
    y != before.as_slice()
}

/// Solve `A X = B` (`A` `n × n`, `B` `n × m`, row-major) by Gaussian
/// elimination with partial pivoting; `None` if `A` is singular.
fn solve_dense(a: &[f64], n: usize, b: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut a = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let piv =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        let p = a[piv * n + col];
        if p == 0.0 || !p.is_finite() {
            return None;
        }
        if piv != col {
            for t in 0..n {
                a.swap(piv * n + t, col * n + t);
            }
            for t in 0..m {
                x.swap(piv * m + t, col * m + t);
            }
        }
        for r in col + 1..n {
            let f = a[r * n + col] / p;
            if f == 0.0 {
                continue;
            }
            for t in col..n {
                a[r * n + t] -= f * a[col * n + t];
            }
            for t in 0..m {
                x[r * m + t] -= f * x[col * m + t];
            }
        }
    }
    for col in (0..n).rev() {
        for t in 0..m {
            let mut v = x[col * m + t];
            for c in col + 1..n {
                v -= a[col * n + c] * x[c * m + t];
            }
            x[col * m + t] = v / a[col * n + col];
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> HarmonyOptions {
        HarmonyOptions::default()
    }

    /// Two batches of three cell types in `d` dimensions; the second batch
    /// is shifted by `shift` along every axis.
    fn batches(n_per: usize, d: usize, shift: f32, seed: u64) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centres: Vec<Vec<f32>> = (0..3)
            .map(|c| (0..d).map(|t| if t % 3 == c { 6.0 } else { 0.0 }).collect())
            .collect();
        let (mut z, mut batch, mut kind) = (Vec::new(), Vec::new(), Vec::new());
        for b in 0..2_u32 {
            for c in 0..3_u32 {
                for _ in 0..n_per {
                    for t in 0..d {
                        let noise: f32 = rng.random::<f32>() - 0.5 + rng.random::<f32>() - 0.5;
                        let s = if b == 1 { shift } else { 0.0 };
                        z.push(centres[c as usize][t] + noise + s);
                    }
                    batch.push(b);
                    kind.push(c);
                }
            }
        }
        (z, batch, kind)
    }

    /// Mean distance between the two batches' centres within each cell type.
    fn batch_gap(z: &[f32], d: usize, batch: &[u32], kind: &[u32]) -> f32 {
        let mut gap = 0.0;
        for c in 0..3 {
            let mut m = [vec![0.0_f32; d], vec![0.0_f32; d]];
            let mut cnt = [0.0_f32; 2];
            for (j, (&b, &k)) in batch.iter().zip(kind).enumerate() {
                if k == c {
                    for t in 0..d {
                        m[b as usize][t] += z[j * d + t];
                    }
                    cnt[b as usize] += 1.0;
                }
            }
            let dist: f32 = (0..d)
                .map(|t| (m[0][t] / cnt[0] - m[1][t] / cnt[1]).powi(2))
                .sum::<f32>()
                .sqrt();
            gap += dist / 3.0;
        }
        gap
    }

    /// Mean distance between the cell types' centres.
    fn type_gap(z: &[f32], d: usize, kind: &[u32]) -> f32 {
        let mut m = vec![vec![0.0_f32; d]; 3];
        let mut cnt = [0.0_f32; 3];
        for (j, &k) in kind.iter().enumerate() {
            for t in 0..d {
                m[k as usize][t] += z[j * d + t];
            }
            cnt[k as usize] += 1.0;
        }
        let dist = |a: usize, b: usize| {
            (0..d)
                .map(|t| (m[a][t] / cnt[a] - m[b][t] / cnt[b]).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        (dist(0, 1) + dist(0, 2) + dist(1, 2)) / 3.0
    }

    #[test]
    fn a_batch_shift_is_removed_and_the_cell_types_stay_apart() {
        let d = 6;
        let (z, batch, kind) = batches(200, d, 2.0, 1);
        let out = harmony(&z, &batch, z.len() / d, d, &opts()).unwrap();
        let before = batch_gap(&z, d, &batch, &kind);
        let after = batch_gap(&out, d, &batch, &kind);
        assert!(after < 0.2 * before, "batch gap {before} -> {after}");
        assert!(type_gap(&out, d, &kind) > 0.8 * type_gap(&z, d, &kind));
    }

    #[test]
    fn well_separated_groups_with_a_large_shift_are_still_aligned() {
        // Hard k-means gave each batch its own clusters here, and corrected
        // nothing; the soft, diversity-penalised assignment does not.
        let d = 6;
        let (z, batch, kind) = batches(150, d, 4.0, 2);
        let out = harmony(&z, &batch, z.len() / d, d, &opts()).unwrap();
        let before = batch_gap(&z, d, &batch, &kind);
        let after = batch_gap(&out, d, &batch, &kind);
        assert!(after < 0.3 * before, "batch gap {before} -> {after}");
    }

    #[test]
    fn two_cells_in_two_variables_move_closer_without_crossing() {
        // The ridge regression of two cells by hand: one cluster, lambda 0.5,
        // the cells in labs A, B and on days 1, 2.
        let z = [1.0_f32, 2.0];
        let lab = [0_u32, 1];
        let day = [0_u32, 1];
        let o = HarmonyOptions {
            n_clusters: Some(1),
            lambda: Some(vec![0.5]),
            sigma: 1.0,
            max_iter_harmony: 1,
            max_iter_kmeans: 0,
            ..opts()
        };
        let mut out = [0.0_f32; 2];
        let cov = [
            Covariate {
                codes: &lab,
                n_levels: 2,
            },
            Covariate {
                codes: &day,
                n_levels: 2,
            },
        ];
        harmony_into(&z, 2, 1, &cov, &o, &mut out).unwrap();
        assert!(
            (out[0] - 1.4).abs() < 1e-6 && (out[1] - 1.6).abs() < 1e-6,
            "{out:?}"
        );

        // One variable: the same ridge by hand gives 4/3 and 5/3.
        harmony_into(&z, 2, 1, &cov[..1], &o, &mut out).unwrap();
        assert!(
            (out[0] - 4.0 / 3.0).abs() < 1e-6 && (out[1] - 5.0 / 3.0).abs() < 1e-6,
            "{out:?}"
        );
    }

    #[test]
    fn the_ridge_solvers_agree_and_a_rise_counts_as_converged() {
        let a = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let x = solve_dense(&a, 3, &[1.0, 2.0, 3.0], 1).unwrap();
        let back: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum())
            .collect();
        for (u, v) in back.iter().zip([1.0, 2.0, 3.0]) {
            assert!((u - v).abs() < 1e-12, "{back:?}");
        }
        assert!(solve_dense(&[1.0, 2.0, 2.0, 4.0], 2, &[1.0, 1.0], 1).is_none());
        assert!(relative_drop(100.0, 101.0) < 0.0);
        assert!((relative_drop(100.0, 99.5) - 0.005).abs() < 1e-12);
        assert!((relative_drop(-100.0, -100.5) - 0.005).abs() < 1e-12);
    }

    #[test]
    fn log_space_assignments_equal_the_weighted_softmax() {
        // exp(-dist/σ) times the diversity weight to the power θ, normalised.
        let dist = [0.2_f32, 1.2, 0.8];
        let dots: Vec<f32> = dist.iter().map(|&x| 1.0 - x / 2.0).collect();
        let ratio = [1.3_f32, 0.6, 2.1];
        let theta = 1.5_f32;
        let pen: Vec<f32> = ratio.iter().map(|r| theta * r.ln()).collect();
        let sigma = 0.8;
        let mut row = [0.0_f32; 3];
        assign_cell(&dots, Some(&pen), sigma, &mut row).unwrap();
        let w: Vec<f32> = dist
            .iter()
            .zip(&ratio)
            .map(|(&x, &r)| (-x / sigma).exp() * r.powf(theta))
            .collect();
        let s: f32 = w.iter().sum();
        for (a, b) in row.iter().zip(&w) {
            assert!((a - b / s).abs() < 1e-6, "{row:?}");
        }
        // Large logits stay finite.
        let mut row = [0.0_f32; 3];
        assign_cell(&dots, None, 1e-4, &mut row).unwrap();
        assert!(row.iter().all(|v| v.is_finite()) && (row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let d = 5;
        let (z, batch, _) = batches(400, d, 1.5, 3);
        let n = z.len() / d;
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| harmony(&z, &batch, n, d, &opts()).unwrap())
        };
        let (a, b) = (run(1), run(4));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(a, harmony(&z, &batch, n, d, &opts()).unwrap());
    }

    #[test]
    fn several_variables_give_finite_coordinates() {
        // A lab-and-day design: 120 cells, two labs, three days.
        let d = 5;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let z: Vec<f32> = (0..120 * d)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let lab: Vec<u32> = (0..120).map(|j| (j % 2) as u32).collect();
        let day: Vec<u32> = (0..120).map(|j| ((j / 2) % 3) as u32).collect();
        let cov = [
            Covariate {
                codes: &lab,
                n_levels: 2,
            },
            Covariate {
                codes: &day,
                n_levels: 3,
            },
        ];
        let o = HarmonyOptions {
            max_iter_harmony: 2,
            max_iter_kmeans: 2,
            ..opts()
        };
        let mut out = vec![0.0_f32; 120 * d];
        let report = harmony_into(&z, 120, d, &cov, &o, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(report.n_clusters, 4);
        assert!(report.objective.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bad_inputs_are_refused() {
        let z = vec![0.5_f32; 20];
        assert_eq!(
            harmony(&z, &[0; 10], 10, 2, &opts()),
            Err(HarmonyError::SingleBatch)
        );
        assert!(matches!(
            harmony(&z, &[0, 1, 0], 10, 2, &opts()),
            Err(HarmonyError::LabelMismatch { .. })
        ));
        assert!(matches!(
            harmony(&[], &[], 0, 2, &opts()),
            Err(HarmonyError::EmptyInput { .. })
        ));
        let codes = [0_u32, 1, 0, 1, 0, 1, 0, 1, 0, 5];
        let mut out = vec![0.0; 20];
        let cov = [Covariate {
            codes: &codes,
            n_levels: 2,
        }];
        assert!(matches!(
            harmony_into(&z, 10, 2, &cov, &opts(), &mut out),
            Err(HarmonyError::BadLevel { code: 5, .. })
        ));
        let labels: Vec<u32> = (0..10).map(|j| j % 2).collect();
        let bad = HarmonyOptions {
            sigma: 0.0,
            ..opts()
        };
        assert!(matches!(
            harmony(&z, &labels, 10, 2, &bad),
            Err(HarmonyError::BadOption(_))
        ));
        let mut nan = z.clone();
        nan[3] = f32::NAN;
        assert!(matches!(
            harmony(&nan, &labels, 10, 2, &opts()),
            Err(HarmonyError::BadOption(_))
        ));
    }

    #[test]
    fn default_cluster_count_follows_python_rounding() {
        assert_eq!(default_n_clusters(3500), 100); // 117 capped
        assert_eq!(default_n_clusters(45), 2); // 1.5 rounds to 2
        assert_eq!(default_n_clusters(75), 2); // 2.5 rounds to 2
        assert_eq!(default_n_clusters(10), 1);
    }
}
