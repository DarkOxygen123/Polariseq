//! Comparing conditions: pseudobulk sums and limma-trend.
//!
//! A difference between conditions is a difference between biological
//! replicates. Cells of one donor are not independent observations, and tests
//! that treat them as independent overstate the significance. The standard
//! remedy is the pseudobulk: add up each replicate's counts within a cell type,
//! then test the sums as bulk RNA-seq samples. This module does both steps.
//!
//! [`unit_sums_blocks`] reads the raw counts once and sums them per unit (a
//! replicate, condition and cell type), from any source of row blocks: the
//! matrix in memory or the store on disk.
//!
//! [`limma_trend`] tests one cell type's sums by the pseudobulk workflow edgeR
//! and limma recommend:
//!
//! 1. the expression filter (Chen, Lun and Smyth 2016),
//! 2. library sizes over the genes kept, and TMM normalization (Robinson and
//!    Oshlack 2010),
//! 3. log2 counts per million, with a prior count of 3 scaled to the library
//!    size,
//! 4. a linear model for each gene,
//! 5. empirical Bayes moderation of the variances with an intensity trend
//!    (Smyth 2004, Phipson et al. 2016, limma-trend as in Law et al. 2014),
//! 6. Benjamini-Hochberg adjusted p-values.
//!
//! Replicates measured under several conditions, such as one donor before and
//! after treatment, are paired: the design blocks on them,
//! `~ replicate + condition`. Otherwise the design is `~ condition`.
//!
//! # Agreement with edgeR and limma
//!
//! The methods are implemented from the papers and from the edgeR and limma
//! documentation. Where the documentation leaves a detail open (the trimming
//! ranks of TMM, the knots of the trend, the treatment of zero variances, the
//! order of operations at the filter's cutoff), it was fixed by comparing
//! outputs with edgeR 4.10.5 and limma 3.68.5 on generated data, with
//! `benchmarks/relicence/pseudobulk_reference.R` on the `reproduce` branch. On
//! 350 datasets the genes kept are the same and the p-values agree to within
//! 4e-12.

// Formula-style names (n, g, p, x, y, v, z) follow the references; exact
// float comparisons (ties, zeros) are deliberate.
#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

use std::collections::BTreeMap;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::rowblocks::RowBlocks;

// ── pseudobulk sums ──────────────────────────────────────────────────────

/// Raw counts summed per unit.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitSums {
    /// Units.
    pub n_units: usize,
    /// Genes.
    pub n_genes: usize,
    /// `sums[u * n_genes + g]`: unit `u`'s total count of gene `g`.
    pub sums: Vec<f64>,
    /// `expressing[u * n_genes + g]`: cells of unit `u` with gene `g` > 0.
    pub expressing: Vec<u32>,
    /// Cells per unit.
    pub cells: Vec<u32>,
}

/// One block's sums, for the units its rows belong to.
type BlockPart = Vec<(u32, Vec<f64>, Vec<u32>)>;

/// Sum the rows of `src` per unit: `unit[i]` is row `i`'s unit, below
/// `n_units`, or `u32::MAX` to leave it out.
///
/// One streamed pass. Blocks are merged in row order as they complete, so the
/// sums are the same for any thread schedule, and at most a few blocks' sums
/// wait at a time (each holds only the units its rows belong to).
///
/// # Panics
/// Panics if `unit` does not have one entry per row.
#[must_use]
pub fn unit_sums_blocks(src: &dyn RowBlocks, unit: &[u32], n_units: usize) -> UnitSums {
    struct State {
        next: usize,
        waiting: BTreeMap<usize, (usize, BlockPart)>,
        sums: Vec<f64>,
        expressing: Vec<u32>,
        cells: Vec<u32>,
    }
    let (n, g) = (src.n_rows(), src.n_cols());
    assert_eq!(unit.len(), n, "one unit per row");
    let state = Mutex::new(State {
        next: 0,
        waiting: BTreeMap::new(),
        sums: vec![0.0; n_units * g],
        expressing: vec![0; n_units * g],
        cells: vec![0; n_units],
    });
    src.for_each_block(8_192, &|off, blk| {
        let mut part: BlockPart = Vec::new();
        let mut cells: Vec<(u32, u32)> = Vec::new();
        for i in 0..blk.n_rows {
            let u = unit[off + i];
            if (u as usize) >= n_units {
                continue;
            }
            let k = if let Some(k) = part.iter().position(|(pu, _, _)| *pu == u) {
                k
            } else {
                part.push((u, vec![0.0; g], vec![0; g]));
                cells.push((u, 0));
                part.len() - 1
            };
            cells[k].1 += 1;
            let (_, s, e) = &mut part[k];
            let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            for (&c, &x) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                s[c as usize] += f64::from(x);
                e[c as usize] += u32::from(x > 0.0);
            }
        }
        let mut st = state.lock().expect("sums poisoned");
        for (u, c) in cells {
            st.cells[u as usize] += c;
        }
        st.waiting.insert(off, (blk.n_rows, part));
        loop {
            let next = st.next;
            let Some((rows, part)) = st.waiting.remove(&next) else {
                break;
            };
            for (u, s, e) in part {
                let base = u as usize * g;
                for (acc, v) in st.sums[base..base + g].iter_mut().zip(&s) {
                    *acc += v;
                }
                for (acc, v) in st.expressing[base..base + g].iter_mut().zip(&e) {
                    *acc += v;
                }
            }
            st.next += rows;
        }
    });
    let st = state.into_inner().expect("sums poisoned");
    assert!(
        st.waiting.is_empty(),
        "row blocks did not tile the rows ({} still waiting)",
        st.waiting.len()
    );
    UnitSums {
        n_units,
        n_genes: g,
        sums: st.sums,
        expressing: st.expressing,
        cells: st.cells,
    }
}

// ── special functions ─────────────────────────────────────────────────────
//
// From the NIST Digital Library of Mathematical Functions (DLMF,
// https://dlmf.nist.gov): the polygamma functions by their recurrences up to
// `SERIES_FROM`, then their asymptotic series in Bernoulli numbers; the
// regularized incomplete beta function by its continued fraction (DLMF
// 8.17.22), evaluated with the modified Lentz method (Thompson & Barnett
// 1986). `lgamma` and `erfc` are libm's (MIT or Apache-2.0, a port of musl).

/// `B_2, B_4, …, B_16` (DLMF 24.2.1).
const BERNOULLI: [f64; 8] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
];

/// Arguments from which the asymptotic series are used: their first omitted
/// term is below 1e-19 relative there.
const SERIES_FROM: f64 = 12.0;

/// `log(x) − ψ(x)` for `x > 0`, without the cancellation of computing the two
/// apart: `ψ(x) = ψ(x + n) − Σ_{j<n} 1/(x + j)` (DLMF 5.5.2) and
/// `ln z − ψ(z) ~ 1/(2z) + Σ_k B_2k / (2k z^2k)` (DLMF 5.11.2).
fn log_minus_digamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let mut z = x;
    let mut shift = 0.0;
    while z < SERIES_FROM {
        shift += 1.0 / z;
        z += 1.0;
    }
    let w = 1.0 / (z * z);
    let mut series = 0.0;
    let mut wk = w;
    for (k, b) in BERNOULLI.iter().enumerate() {
        series += b / (2.0 * (k + 1) as f64) * wk;
        wk *= w;
    }
    1.0 / (2.0 * z) + series + (x / z).ln() + shift
}

/// The trigamma function `ψ'(x)`, `x > 0`: `ψ'(x) = ψ'(x + 1) + 1/x²` and
/// `ψ'(z) ~ 1/z + 1/(2z²) + Σ_k B_2k / z^(2k+1)` (DLMF 5.15.8).
fn trigamma(x: f64) -> f64 {
    let mut z = x;
    let mut shift = 0.0;
    while z < SERIES_FROM {
        shift += 1.0 / (z * z);
        z += 1.0;
    }
    let w = 1.0 / (z * z);
    let mut series = 0.0;
    let mut zk = w / z;
    for b in BERNOULLI {
        series += b * zk;
        zk *= w;
    }
    shift + 1.0 / z + 0.5 * w + series
}

/// The tetragamma function `ψ''(x)`, `x > 0`: the derivative of the above,
/// `ψ''(x) = ψ''(x + 1) − 2/x³`.
fn tetragamma(x: f64) -> f64 {
    let mut z = x;
    let mut shift = 0.0;
    while z < SERIES_FROM {
        shift -= 2.0 / (z * z * z);
        z += 1.0;
    }
    let w = 1.0 / (z * z);
    let mut series = 0.0;
    let mut zk = w * w;
    for (k, b) in BERNOULLI.iter().enumerate() {
        series -= (2.0 * (k + 1) as f64 + 1.0) * b * zk;
        zk *= w;
    }
    shift - w - w / z + series
}

/// The `y > 0` with `ψ'(y) = x`: Newton's method on `u = ln y` for
/// `ln ψ'(e^u) = ln x` (`ψ'` falls monotonically from ∞ to 0), started from
/// its asymptotes `ψ'(y) ≈ 1/y + 1/(2y²)` (large `y`) and `≈ 1/y²` (small).
fn trigamma_inverse(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let target = x.ln();
    let mut u = if x < 1.0 {
        (1.0 / x + 0.5).ln()
    } else {
        -0.5 * target
    };
    for _ in 0..100 {
        let y = u.exp();
        let (t1, t2) = (trigamma(y), tetragamma(y));
        let step = (t1.ln() - target) / (y * t2 / t1);
        u -= step;
        if step.abs() < 1e-15 {
            break;
        }
    }
    u.exp()
}

/// The Stirling remainder `ln Γ(x) − [(x − ½) ln x − x + ½ ln 2π]` for
/// `x ≥ SERIES_FROM` (DLMF 5.11.1).
fn stirling_remainder(x: f64) -> f64 {
    let w = 1.0 / (x * x);
    let mut s = 0.0;
    let mut xk = 1.0 / x;
    for (k, b) in BERNOULLI.iter().enumerate() {
        let k2 = 2.0 * (k + 1) as f64;
        s += b / (k2 * (k2 - 1.0)) * xk;
        xk *= w;
    }
    s
}

/// `ln B(a, b)`. When the larger argument is large, `ln Γ(a) − ln Γ(a + b)`
/// is formed from Stirling's series with `ln(a/(a + b))` as a `ln_1p`, so it
/// keeps its relative accuracy where the difference of two `lgamma`s would
/// cancel (degrees of freedom in the thousands and more).
fn ln_beta(a: f64, b: f64) -> f64 {
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    if big < SERIES_FROM {
        return libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    }
    let s = big + small;
    let ratio =
        -(big - 0.5) * (small / big).ln_1p() - small * s.ln() + small + stirling_remainder(big)
            - stirling_remainder(s);
    libm::lgamma(small) + ratio
}

/// The regularized incomplete beta function `I_x(a, b)`, given `x` and
/// `y = 1 − x` separately so the caller can form the smaller one without
/// cancellation. Below the mean, `x < (a + 1)/(a + b + 2)`, the continued
/// fraction converges quickly; above it, `I_x(a, b) = 1 − I_y(b, a)` (DLMF
/// 8.17.4).
fn inc_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - inc_beta(b, a, y, x);
    }
    let ln_x = if x > 0.5 { (-y).ln_1p() } else { x.ln() };
    let ln_y = if y > 0.5 { (-x).ln_1p() } else { y.ln() };
    let front = (a * ln_x + b * ln_y - ln_beta(a, b)).exp() / a;
    // 1 / (1 + d1 / (1 + d2 / (1 + …))), d_2m+1 and d_2m of DLMF 8.17.22.
    let tiny = 1e-300;
    let (mut f, mut c, mut d) = (1.0_f64, 1.0_f64, 0.0_f64);
    for j in 1..=20_000_u32 {
        let m = f64::from(j / 2);
        let dj = if j % 2 == 1 {
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        } else {
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        };
        d = 1.0 + dj * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + dj / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (front / f).clamp(0.0, 1.0)
}

/// Two-sided p-value of Student's `t` on `df` (real) degrees of freedom:
/// `I_{df/(df+t²)}(df/2, 1/2)`; the normal distribution when `df` is infinite.
fn t_two_sided(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    if df.is_infinite() {
        return libm::erfc(t.abs() / std::f64::consts::SQRT_2);
    }
    let t2 = t * t;
    inc_beta(df / 2.0, 0.5, df / (df + t2), t2 / (df + t2))
}

/// R's `quantile(x, p)` (type 7).
fn quantile7(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    s[lo] + (h - lo as f64) * (s[hi] - s[lo])
}

fn median(v: &[f64]) -> f64 {
    quantile7(v, 0.5)
}

/// R's `rank` with ties averaged (1-based).
fn rank_average(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[order[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Benjamini-Hochberg adjusted p-values, as R's `p.adjust(p, "BH")`.
fn p_adjust_bh(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Decreasing p, stable, as R's order(p, decreasing = TRUE).
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let mut out = vec![0.0; n];
    let mut running = f64::INFINITY;
    for (k, &i) in order.iter().enumerate() {
        let rank = (n - k) as f64;
        running = running.min(n as f64 / rank * p[i]);
        out[i] = running.min(1.0);
    }
    out
}

// ── edgeR ─────────────────────────────────────────────────────────────────

/// Options of the pseudobulk test, edgeR's and limma's defaults.
#[derive(Debug, Clone)]
pub struct PseudobulkOptions {
    /// `filterByExpr(min.count)`.
    pub min_count: f64,
    /// `filterByExpr(min.total.count)`.
    pub min_total_count: f64,
    /// `filterByExpr(large.n)`.
    pub large_n: f64,
    /// `filterByExpr(min.prop)`.
    pub min_prop: f64,
    /// `cpm(log = TRUE, prior.count)`.
    pub prior_count: f64,
}

impl Default for PseudobulkOptions {
    fn default() -> Self {
        Self {
            min_count: 10.0,
            min_total_count: 15.0,
            large_n: 10.0,
            min_prop: 0.7,
            prior_count: 3.0,
        }
    }
}

/// The expression filter (Chen, Lun & Smyth 2016; edgeR's `filterByExpr`,
/// as its documentation states it): a gene is kept when its count per
/// million reaches `min_count / median(library size) × 10⁶` in at least `k`
/// samples, `k` the smallest condition's sample count (above `large_n`
/// samples, `large_n + (k − large_n) × min_prop`), and its counts total at
/// least `min_total_count`. CPM is `count × 10⁶ / library size`, the order of
/// operations that agrees with edgeR where a count sits exactly on the cutoff.
fn filter_by_expr(
    y: &[f64],
    n: usize,
    g: usize,
    group: &[u32],
    opts: &PseudobulkOptions,
) -> Vec<bool> {
    let lib: Vec<f64> = (0..n).map(|i| y[i * g..(i + 1) * g].iter().sum()).collect();
    let cutoff = opts.min_count / median(&lib) * 1e6;
    let mut sizes: BTreeMap<u32, usize> = BTreeMap::new();
    for &c in group {
        *sizes.entry(c).or_default() += 1;
    }
    let smallest = sizes.values().copied().min().unwrap_or(0) as f64;
    let k = if smallest > opts.large_n {
        opts.large_n + (smallest - opts.large_n) * opts.min_prop
    } else {
        smallest
    };
    (0..g)
        .map(|j| {
            let above = (0..n)
                .filter(|&i| y[i * g + j] * 1e6 / lib[i] >= cutoff)
                .count() as f64;
            let total: f64 = (0..n).map(|i| y[i * g + j]).sum();
            above >= k - 1e-14 && total >= opts.min_total_count - 1e-14
        })
        .collect()
}

/// TMM factor of one sample against the reference (Robinson & Oshlack 2010):
/// over genes counted in both, the log ratios `M = log2((o/N_o)/(r/N_r))` and
/// mean log expressions `A = (log2(o/N_o) + log2(r/N_r))/2`; the genes whose
/// rank (ties averaged) is within the middle of both, 30 % trimmed from each
/// end of M and 5 % of A (ranks `⌊0.3 G⌋ + 1 … G − ⌊0.3 G⌋`); their M
/// averaged with the inverse of the delta-method variance
/// `(N_o − o)/(N_o o) + (N_r − r)/(N_r r)` as weights; the factor is 2 to that.
fn tmm_pair(obs: &[f64], refc: &[f64], n_o: f64, n_r: f64) -> f64 {
    let (mut m, mut a, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for (&o, &r) in obs.iter().zip(refc) {
        if o > 0.0 && r > 0.0 {
            let (po, pr) = (o / n_o, r / n_r);
            let mi = (po / pr).log2();
            let ai = 0.5 * (po.log2() + pr.log2());
            if mi.is_finite() && ai.is_finite() {
                m.push(mi);
                a.push(ai);
                v.push((n_o - o) / n_o / o + (n_r - r) / n_r / r);
            }
        }
    }
    let count = m.len();
    if count == 0 {
        return 1.0;
    }
    let bounds = |trim: f64| {
        let lo = (count as f64 * trim).floor() + 1.0;
        (lo, count as f64 + 1.0 - lo)
    };
    let ((lo_m, hi_m), (lo_a, hi_a)) = (bounds(0.3), bounds(0.05));
    let (rm, ra) = (rank_average(&m), rank_average(&a));
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..count {
        if rm[i] >= lo_m && rm[i] <= hi_m && ra[i] >= lo_a && ra[i] <= hi_a {
            num += m[i] / v[i];
            den += 1.0 / v[i];
        }
    }
    if den > 0.0 {
        (num / den).exp2()
    } else {
        1.0
    }
}

/// TMM normalization factors (edgeR's documented defaults): genes counted in
/// no sample left out; the reference is the sample whose upper quartile of
/// count proportions is closest to the mean of those quartiles; the factors
/// scaled to a geometric mean of 1.
fn tmm_factors(y: &[f64], n: usize, g: usize, lib: &[f64]) -> Vec<f64> {
    let used: Vec<usize> = (0..g)
        .filter(|&j| (0..n).any(|i| y[i * g + j] > 0.0))
        .collect();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| used.iter().map(|&j| y[i * g + j]).collect())
        .collect();
    let upper: Vec<f64> = (0..n)
        .map(|i| {
            let p: Vec<f64> = rows[i].iter().map(|c| c / lib[i]).collect();
            quantile7(&p, 0.75)
        })
        .collect();
    let mean_upper = upper.iter().sum::<f64>() / n as f64;
    let mut reference = 0;
    for i in 1..n {
        if (upper[i] - mean_upper).abs() < (upper[reference] - mean_upper).abs() {
            reference = i;
        }
    }
    let f: Vec<f64> = (0..n)
        .map(|i| tmm_pair(&rows[i], &rows[reference], lib[i], lib[reference]))
        .collect();
    let geo = (f.iter().map(|x| x.ln()).sum::<f64>() / n as f64).exp();
    f.iter().map(|x| x / geo).collect()
}

// ── limma ─────────────────────────────────────────────────────────────────

/// A natural cubic spline basis (intercept, linear term, and one column per
/// interior knot) in the truncated-power form of Hastie, Tibshirani &
/// Friedman (2009), eq. 5.4-5.5: boundary knots at the range of `x`,
/// `inner` interior knots at its quantiles `k/(inner + 1)` (type 7). `x` is
/// mapped to [0, 1] first, which leaves the space unchanged and keeps the
/// normal equations well conditioned.
fn natural_spline_basis(x: &[f64], inner: usize) -> Vec<Vec<f64>> {
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if hi > lo { hi - lo } else { 1.0 };
    let s: Vec<f64> = x.iter().map(|v| (v - lo) / span).collect();
    let mut knots = vec![0.0];
    for k in 1..=inner {
        knots.push((quantile7(x, k as f64 / (inner + 1) as f64) - lo) / span);
    }
    knots.push(1.0);
    let last = knots.len() - 1;
    let d = |k: usize, v: f64| {
        let cube = |t: f64| (v - t).max(0.0).powi(3);
        (cube(knots[k]) - cube(knots[last])) / (knots[last] - knots[k])
    };
    let mut basis = vec![vec![1.0; x.len()], s.clone()];
    for k in 0..last.saturating_sub(1) {
        basis.push(s.iter().map(|&v| d(k, v) - d(last - 1, v)).collect());
    }
    basis
}

/// Least squares of `y` on the columns of `x` (small `p`): fitted values and
/// residual sum of squares, through the normal equations in f64 with
/// Cholesky. Returns `None` when `x` is rank-deficient.
fn least_squares(x: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, f64)> {
    let p = x.len();
    let n = y.len();
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    for i in 0..p {
        for j in 0..=i {
            let v: f64 = (0..n).map(|k| x[i][k] * x[j][k]).sum();
            a[i * p + j] = v;
            a[j * p + i] = v;
        }
        b[i] = (0..n).map(|k| x[i][k] * y[k]).sum();
    }
    let l = cholesky(&a, p)?;
    let beta = chol_solve(&l, p, &b);
    let fitted: Vec<f64> = (0..n)
        .map(|k| (0..p).map(|i| x[i][k] * beta[i]).sum())
        .collect();
    let rss = fitted.iter().zip(y).map(|(f, v)| (v - f) * (v - f)).sum();
    Some((fitted, rss))
}

fn cholesky(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s <= 1e-12 * a[i * p + i].abs().max(1e-300) {
                    return None;
                }
                l[i * p + i] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    Some(l)
}

fn chol_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; p];
    for i in 0..p {
        let s: f64 = (0..i).map(|k| l[i * p + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * p + i];
    }
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let s: f64 = (i + 1..p).map(|k| l[k * p + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * p + i];
    }
    x
}

fn chol_inverse_diag(l: &[f64], p: usize) -> Vec<f64> {
    (0..p)
        .map(|j| {
            let mut e = vec![0.0; p];
            e[j] = 1.0;
            chol_solve(l, p, &e)[j]
        })
        .collect()
}

/// The prior of the residual variances (Smyth 2004, section 6.2, with the
/// trend of Phipson et al. 2016): `s²` given `d` residual degrees of freedom
/// is the prior variance `s0²` times an F on `d` and `d0` degrees of freedom.
/// By the method of moments on `e = ln s² + ln(d/2) − ψ(d/2)`: a natural
/// spline in mean expression gives the trend of `E e` (interior knots: none
/// below 6 genes, one below 30, else two), the residual variance of `e` less
/// `ψ'(d/2)` is `ψ'(d0/2)`, and `s0² = exp(E e − ln(d0/2) + ψ(d0/2))`. When
/// that variance is not positive there is no extra variation: `d0 = ∞` and
/// `s0² = exp(E e)`. Variances below 10⁻⁵ of their median are raised to it
/// first, so that a gene of zero variance cannot dominate.
/// Returns `(s0²` per gene`, d0)`.
fn fit_prior(s2: &[f64], d: f64, covariate: &[f64]) -> (Vec<f64>, f64) {
    let g = s2.len();
    let floor = (1e-5 * median(s2)).max(f64::MIN_POSITIVE);
    let e: Vec<f64> = s2
        .iter()
        .map(|&v| v.max(floor).ln() + log_minus_digamma(d / 2.0))
        .collect();
    let inner = if g < 6 {
        0
    } else if g < 30 {
        1
    } else {
        2
    };
    let mean_e = e.iter().sum::<f64>() / g as f64;
    // A trend the covariate cannot support (all genes at one mean) falls
    // back to a constant.
    let (fitted, rss, p) = least_squares(&natural_spline_basis(covariate, inner), &e).map_or_else(
        || {
            let rss = e.iter().map(|v| (v - mean_e) * (v - mean_e)).sum();
            (vec![mean_e; g], rss, 1)
        },
        |(f, rss)| (f, rss, inner + 2),
    );
    let extra = rss / (g as f64 - p as f64) - trigamma(d / 2.0);
    if extra > 0.0 {
        let d0 = 2.0 * trigamma_inverse(extra);
        let shift = log_minus_digamma(d0 / 2.0);
        (fitted.iter().map(|f| (f - shift).exp()).collect(), d0)
    } else {
        (fitted.iter().map(|f| f.exp()).collect(), f64::INFINITY)
    }
}

/// What one condition's comparison with the reference found, gene by gene
/// over the genes tested.
#[derive(Debug, Clone, Default)]
pub struct Comparison {
    /// The condition compared with the reference.
    pub condition: u32,
    /// Genes tested (indices of the input's columns), in input order.
    pub genes: Vec<u32>,
    /// log2 fold change (the design's coefficient).
    pub log_fc: Vec<f64>,
    /// Mean log2 CPM over the samples (limma's `AveExpr`).
    pub ave_expr: Vec<f64>,
    /// Mean log2 CPM of the reference's samples.
    pub mean_reference: Vec<f64>,
    /// Mean log2 CPM of the condition's samples.
    pub mean_condition: Vec<f64>,
    /// Moderated t.
    pub t: Vec<f64>,
    /// Two-sided p-value.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    pub adj_p_value: Vec<f64>,
}

/// Summary of one limma-trend fit.
#[derive(Debug, Clone, Default)]
pub struct LimmaFit {
    /// One comparison per condition other than the reference present.
    pub comparisons: Vec<Comparison>,
    /// Whether the design blocked on the replicate.
    pub paired: bool,
    /// Residual degrees of freedom of the linear model.
    pub df_residual: f64,
    /// The prior's degrees of freedom (`eBayes`'s `df.prior`).
    pub df_prior: f64,
    /// TMM normalization factors, one per sample.
    pub norm_factors: Vec<f64>,
}

/// Test one cell type's pseudobulk samples: `y` is samples × genes
/// (row-major) raw count sums, `condition[s]` and `replicate[s]` each
/// sample's codes. See the module documentation for the steps.
///
/// # Errors
/// Returns why a fit is impossible: a condition or the reference without
/// samples, no residual degrees of freedom, or no gene passing the filter.
///
/// # Panics
/// Panics if `y` is not `n × g`.
pub fn limma_trend(
    y: &[f64],
    n: usize,
    g: usize,
    condition: &[u32],
    reference: u32,
    replicate: &[u32],
    opts: &PseudobulkOptions,
) -> Result<LimmaFit, String> {
    assert_eq!(y.len(), n * g, "samples × genes");
    if !condition.contains(&reference) {
        return Err("no sample of the reference condition".into());
    }
    let mut conds: Vec<u32> = condition
        .iter()
        .copied()
        .filter(|&c| c != reference)
        .collect();
    conds.sort_unstable();
    conds.dedup();
    if conds.is_empty() {
        return Err("no sample of a condition other than the reference".into());
    }

    // The expression filter on the condition groups, then library sizes of
    // the genes kept and TMM.
    let keep = filter_by_expr(y, n, g, condition, opts);
    let genes: Vec<usize> = (0..g).filter(|&j| keep[j]).collect();
    let h = genes.len();
    if h < 3 {
        return Err(format!("{h} genes pass the expression filter"));
    }
    let yk: Vec<f64> = (0..n)
        .flat_map(|i| genes.iter().map(move |&j| y[i * g + j]))
        .collect();
    let lib: Vec<f64> = (0..n)
        .map(|i| yk[i * h..(i + 1) * h].iter().sum())
        .collect();
    let nf = tmm_factors(&yk, n, h, &lib);
    let eff: Vec<f64> = lib.iter().zip(&nf).map(|(l, f)| l * f).collect();
    let mean_eff = eff.iter().sum::<f64>() / n as f64;
    // log2 CPM, the prior count scaled to the library size and twice it added
    // to the library (edgeR's documented `cpm(log = TRUE)`).
    let logcpm: Vec<f64> = (0..n)
        .flat_map(|i| {
            let pc = opts.prior_count * eff[i] / mean_eff;
            let denom = eff[i] + 2.0 * pc;
            let yk = &yk;
            (0..h).map(move |j| ((yk[i * h + j] + pc) / denom * 1e6).log2())
        })
        .collect();

    // Design: intercept, replicates (when some replicate is measured under
    // two conditions), conditions other than the reference.
    let mut cond_of_rep: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&r, &c) in replicate.iter().zip(condition) {
        cond_of_rep.entry(r).or_default().push(c);
    }
    let paired = cond_of_rep.values().any(|cs| {
        let mut cs = cs.clone();
        cs.sort_unstable();
        cs.dedup();
        cs.len() > 1
    });
    let mut cols: Vec<Vec<f64>> = vec![vec![1.0; n]];
    if paired {
        for &r in cond_of_rep.keys().skip(1) {
            cols.push(
                replicate
                    .iter()
                    .map(|&x| f64::from(u8::from(x == r)))
                    .collect(),
            );
        }
    }
    let first_cond = cols.len();
    for &c in &conds {
        cols.push(
            condition
                .iter()
                .map(|&x| f64::from(u8::from(x == c)))
                .collect(),
        );
    }
    let p = cols.len();
    if n <= p {
        return Err(format!(
            "{n} samples leave no residual degrees of freedom for {p} coefficients"
        ));
    }
    let mut xtx = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            xtx[i * p + j] = (0..n).map(|k| cols[i][k] * cols[j][k]).sum();
        }
    }
    let l = cholesky(&xtx, p).ok_or("the design is not of full rank")?;
    let unscaled: Vec<f64> = chol_inverse_diag(&l, p).iter().map(|v| v.sqrt()).collect();
    let df = (n - p) as f64;

    // The linear model: coefficients and residual variance per gene.
    let fits: Vec<(Vec<f64>, f64)> = (0..h)
        .into_par_iter()
        .map(|j| {
            let yj: Vec<f64> = (0..n).map(|i| logcpm[i * h + j]).collect();
            let xty: Vec<f64> = (0..p)
                .map(|c| (0..n).map(|k| cols[c][k] * yj[k]).sum())
                .collect();
            let beta = chol_solve(&l, p, &xty);
            let rss: f64 = (0..n)
                .map(|k| {
                    let fit: f64 = (0..p).map(|c| cols[c][k] * beta[c]).sum();
                    (yj[k] - fit) * (yj[k] - fit)
                })
                .sum();
            (beta, rss / df)
        })
        .collect();
    let s2: Vec<f64> = fits.iter().map(|f| f.1).collect();
    let amean: Vec<f64> = (0..h)
        .map(|j| (0..n).map(|i| logcpm[i * h + j]).sum::<f64>() / n as f64)
        .collect();

    // Empirical Bayes: the prior variance follows mean expression; posterior
    // variances, and the moderated t on d + d0 degrees of freedom (at most the
    // pooled residual degrees of freedom), Smyth (2004).
    let (s2_prior, df_prior) = fit_prior(&s2, df, &amean);
    let s2_post: Vec<f64> = if df_prior.is_finite() {
        s2.iter()
            .zip(&s2_prior)
            .map(|(&v, &pr)| (df * v + df_prior * pr) / (df + df_prior))
            .collect()
    } else {
        s2_prior.clone()
    };
    let df_total = (df + df_prior).min(df * h as f64);

    let comparisons = conds
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            let col = first_cond + k;
            let log_fc: Vec<f64> = fits.iter().map(|f| f.0[col]).collect();
            let t: Vec<f64> = log_fc
                .iter()
                .zip(&s2_post)
                .map(|(b, v)| b / unscaled[col] / v.sqrt())
                .collect();
            let p_value: Vec<f64> = t.iter().map(|&tv| t_two_sided(tv, df_total)).collect();
            let adj_p_value = p_adjust_bh(&p_value);
            let mean_of = |code: u32| -> Vec<f64> {
                let rows: Vec<usize> = (0..n).filter(|&i| condition[i] == code).collect();
                (0..h)
                    .map(|j| {
                        rows.iter().map(|&i| logcpm[i * h + j]).sum::<f64>() / rows.len() as f64
                    })
                    .collect()
            };
            Comparison {
                condition: c,
                genes: genes.iter().map(|&j| j as u32).collect(),
                log_fc,
                ave_expr: amean.clone(),
                mean_reference: mean_of(reference),
                mean_condition: mean_of(c),
                t,
                p_value,
                adj_p_value,
            }
        })
        .collect();
    Ok(LimmaFit {
        comparisons,
        paired,
        df_residual: df,
        df_prior,
        norm_factors: nf,
    })
}

/// Per-gene statistics of one set of cells: sums, sums of squares and
/// counts of nonzero values, over `n` cells.
#[derive(Debug, Clone)]
pub struct CellStats<'a> {
    /// Cells.
    pub n: f64,
    /// Sum per gene.
    pub sum: &'a [f64],
    /// Sum of squares per gene.
    pub sq: &'a [f64],
    /// Cells with a nonzero value, per gene.
    pub nonzero: &'a [u32],
}

/// A comparison of cells: Welch's t-test per gene between a condition's cells
/// and the reference's, as Scanpy's `rank_genes_groups(method = "t-test")`
/// computes it (sample variances, Welch-Satterthwaite degrees of freedom,
/// two-sided p from Student's t). Values are the matrix's (log-normalized),
/// so the fold change is Scanpy's, `log2((expm1(m₁) + 1e-9) / (expm1(m₀) + 1e-9))`.
#[derive(Debug, Clone, Default)]
pub struct CellComparison {
    /// log2 fold change.
    pub log_fc: Vec<f64>,
    /// Mean of the reference's cells.
    pub mean_reference: Vec<f64>,
    /// Mean of the condition's cells.
    pub mean_condition: Vec<f64>,
    /// Share of the reference's cells expressing the gene.
    pub pct_reference: Vec<f64>,
    /// Share of the condition's cells expressing the gene.
    pub pct_condition: Vec<f64>,
    /// Welch's t.
    pub t: Vec<f64>,
    /// Two-sided p-value.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    pub adj_p_value: Vec<f64>,
}

/// Welch's t-test of `cond` against `reference`, gene by gene.
#[must_use]
pub fn welch_compare(cond: &CellStats<'_>, reference: &CellStats<'_>) -> CellComparison {
    let g = cond.sum.len();
    let moments = |s: &CellStats<'_>, j: usize| -> (f64, f64) {
        let m = s.sum[j] / s.n;
        let v = if s.n > 1.0 {
            ((s.sq[j] - s.n * m * m) / (s.n - 1.0)).max(0.0)
        } else {
            0.0
        };
        (m, v)
    };
    let mut out = CellComparison::default();
    for j in 0..g {
        let (m1, v1) = moments(cond, j);
        let (m0, v0) = moments(reference, j);
        let (a, b) = (v1 / cond.n, v0 / reference.n);
        let se = (a + b).sqrt();
        let t = if se > 0.0 { (m1 - m0) / se } else { f64::NAN };
        let df = (a + b) * (a + b)
            / (a * a / (cond.n - 1.0).max(1.0) + b * b / (reference.n - 1.0).max(1.0));
        out.t.push(t);
        out.p_value
            .push(if t.is_nan() { 1.0 } else { t_two_sided(t, df) });
        out.log_fc
            .push(((m1.exp_m1() + 1e-9) / (m0.exp_m1() + 1e-9)).log2());
        out.mean_condition.push(m1);
        out.mean_reference.push(m0);
        out.pct_condition.push(f64::from(cond.nonzero[j]) / cond.n);
        out.pct_reference
            .push(f64::from(reference.nonzero[j]) / reference.n);
    }
    out.adj_p_value = p_adjust_bh(&out.p_value);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CsrMatrix;

    #[test]
    #[allow(clippy::excessive_precision)] // R's values, printed to 17 digits
    fn special_functions_match_r() {
        // R 4.6.1: trigamma, psigamma(deriv = 2), lbeta, 2 * pt(-|t|, df),
        // log(x) - digamma(x).
        let close =
            |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol * b.abs(), "{a} vs {b}");
        close(trigamma(0.7), 2.834_049_156_694_610_4, 1e-14);
        close(trigamma(12.3), 0.084_695_170_245_916_385, 1e-14);
        close(trigamma(1e-4), 100_000_001.644_693_67, 1e-14);
        close(tetragamma(1.3), -1.198_462_514_651_956_5, 1e-13);
        close(tetragamma(40.0), -0.000_640_820_271_835_298_58, 1e-13);
        close(ln_beta(3.2, 0.5), 0.029_697_443_414_221_078, 1e-13);
        close(ln_beta(5000.0, 0.5), -3.686_206_652_783_460_5, 1e-14);
        close(ln_beta(2e6, 0.5), -6.681_963_863_837_409_3, 1e-14);
        close(t_two_sided(2.1, 7.5), 0.071_237_071_458_078_047, 1e-13);
        close(t_two_sided(-40.0, 12.0), 3.851_229_580_772_036_9e-14, 1e-12);
        close(t_two_sided(0.01, 3.0), 0.992_649_111_412_851_95, 1e-14);
        close(
            t_two_sided(3.3, 150_000.0),
            0.000_967_073_595_280_029_2,
            1e-11,
        );
        close(t_two_sided(9.0, 4e6), 2.258_125_307_522_478_7e-19, 1e-10);
        close(t_two_sided(1.5, 0.7), 0.440_365_097_019_240_15, 1e-13);
        close(log_minus_digamma(2.2), 0.244_163_923_623_125_33, 1e-14);
        close(log_minus_digamma(0.03), 30.355_696_523_298_786, 1e-14);
        close(log_minus_digamma(250.0), 0.002_001_333_331_199_894_9, 1e-13);
        for x in [1e-9, 1e-3, 0.37, 2.0, 50.0, 1e6] {
            close(trigamma(trigamma_inverse(x)), x, 1e-13);
        }
    }

    #[test]
    fn tmm_of_a_sample_against_itself_is_one() {
        let row = [3.0, 0.0, 17.0, 250.0, 9.0, 1.0];
        assert_eq!(tmm_pair(&row, &row, 280.0, 280.0), 1.0);
    }

    #[test]
    fn bh_and_ranks_match_r() {
        let p = [0.01, 0.04, 0.03, 0.5, 0.2];
        let want = [
            0.05,
            0.066_666_666_666_666_67,
            0.066_666_666_666_666_67,
            0.5,
            0.25,
        ];
        for (a, b) in p_adjust_bh(&p).iter().zip(want) {
            assert!((a - b).abs() < 1e-15, "{a} vs {b}");
        }
        assert_eq!(
            rank_average(&[3.0, 1.0, 3.0, 2.0]),
            vec![3.5, 1.0, 3.5, 2.0]
        );
    }

    #[test]
    fn sums_are_per_unit_and_leave_out_unassigned_rows() {
        let dense = [
            1.0_f32, 0.0, 2.0, 3.0, 1.0, 0.0, 0.0, 5.0, 1.0, 2.0, 2.0, 2.0,
        ];
        let m = CsrMatrix::from_dense_row_major(&dense, 4, 3);
        let s = unit_sums_blocks(&m, &[1, 0, u32::MAX, 1], 2);
        assert_eq!(s.cells, vec![1, 2]);
        assert_eq!(&s.sums[0..3], &[3.0, 1.0, 0.0]);
        assert_eq!(&s.sums[3..6], &[3.0, 2.0, 4.0]);
        assert_eq!(&s.expressing[3..6], &[2, 1, 2]);
    }
}
