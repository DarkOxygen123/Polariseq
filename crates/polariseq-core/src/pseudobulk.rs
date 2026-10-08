//! Comparing conditions: pseudobulk sums and limma-trend.
//!
//! A difference between conditions is a difference between biological
//! replicates, not between cells: cells of one donor are not independent
//! observations, and tests that treat them so overstate significance. The
//! standard remedy is the pseudobulk: sum each replicate's counts within a
//! cell type, then test the sums as bulk RNA-seq samples. This module does
//! both steps.
//!
//! [`unit_sums_blocks`] streams the raw counts once and sums them per unit
//! (a replicate, condition and cell type), in any row-block source (the
//! in-memory matrix, the on-disk store).
//!
//! [`limma_trend`] tests one cell type's sums as edgeR and limma do in their
//! recommended pseudobulk workflow (`filterByExpr(y, group = condition)`,
//! `y[keep, , keep.lib.sizes = FALSE]`, TMM `normLibSizes`,
//! `cpm(y, log = TRUE, prior.count = 3)`, `lmFit`, `eBayes(trend = TRUE)`,
//! BH-adjusted p-values), re-implemented from edgeR 4.10 and limma 3.68 and
//! checked against them. Replicates measured under several conditions (the
//! same donor before and after treatment) are paired: the design blocks on
//! them, `~ replicate + condition`; otherwise it is `~ condition`.
//!
//! Licences of the code this module follows. `filter_by_expr`, `tmm_factor`
//! and `tmm_factors` follow edgeR's `filterByExpr`, `.calcFactorTMM` and
//! `normLibSizes` (edgeR, GPL (>= 2), Copyright Gordon Smyth, Yunshun Chen,
//! Aaron Lun, Davis McCarthy, Mark Robinson and contributors); `fit_f_dist`,
//! `trigamma_inverse` and the empirical Bayes steps of `limma_trend` follow
//! limma's `fitFDist`, `trigammaInverse`, `squeezeVar` and `eBayes` (limma,
//! GPL (>= 2), Copyright Gordon Smyth and contributors); `logmdigamma`
//! follows statmod's `logmdigamma` (statmod, GPL-2 | GPL-3, Copyright Gordon
//! Smyth and contributors). Polariseq is therefore to be distributed under a
//! licence compatible with the GPL, version 3.

// Formula-style names (n, g, p, x, y, v, z) follow the references; exact
// float comparisons (ties, zeros) are the references'; authors' names
// (McCarthy) appear in prose.
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

/// `log(x) - digamma(x)`, as statmod's `logmdigamma` computes it (limma's
/// `fitFDist` uses it).
fn logmdigamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    if x < 5.0 {
        return (x / (x + 5.0)).ln()
            + logmdigamma(x + 5.0)
            + 1.0 / x
            + 1.0 / (x + 1.0)
            + 1.0 / (x + 2.0)
            + 1.0 / (x + 3.0)
            + 1.0 / (x + 4.0);
    }
    let y = 1.0 / (x * x);
    let tail = y
        * (-1.0 / 12.0
            + y * (1.0 / 120.0
                + y * (-1.0 / 252.0
                    + y * (1.0 / 240.0
                        + y * (-1.0 / 132.0
                            + y * (691.0 / 32_760.0 + y * (-1.0 / 12.0 + 3617.0 * y / 8160.0)))))));
    1.0 / (2.0 * x) - tail
}

/// The trigamma function, `ψ'(x)`, for `x > 0`.
fn trigamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 10.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let y = 1.0 / (x * x);
    acc + 1.0 / x
        + y / 2.0
        + (1.0 / x)
            * y
            * (1.0 / 6.0
                + y * (-1.0 / 30.0
                    + y * (1.0 / 42.0
                        + y * (-1.0 / 30.0
                            + y * (5.0 / 66.0 + y * (-691.0 / 2730.0 + y * 7.0 / 6.0))))))
}

/// The tetragamma function, `ψ''(x)`, for `x > 0`.
fn tetragamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let y = 1.0 / (x * x);
    acc - 1.0 / (x * x)
        - 1.0 / (x * x * x)
        - (y * y)
            * (1.0 / 2.0
                + y * (-1.0 / 6.0
                    + y * (1.0 / 6.0
                        + y * (-3.0 / 10.0
                            + y * (5.0 / 6.0 + y * (-691.0 / 210.0 + y * 35.0 / 2.0))))))
}

/// limma's `trigammaInverse`: `y` with `trigamma(y) = x`, by Newton's method.
fn trigamma_inverse(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x > 1e7 {
        return 1.0 / x.sqrt();
    }
    if x < 1e-6 {
        return 1.0 / x;
    }
    let mut y = 0.5 + 1.0 / x;
    for _ in 0..=50 {
        let tri = trigamma(y);
        let dif = tri * (1.0 - tri / x) / tetragamma(y);
        y += dif;
        if -dif / y < 1e-8 {
            break;
        }
    }
    y
}

/// `ln Γ(x)`, `x > 0` (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction of the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=10_000 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// The regularized incomplete beta function `I_x(a, b)`.
fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        (ln_front.exp() * beta_cf(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - ln_front.exp() * beta_cf(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

/// Two-sided p-value of Student's `t` with `df` degrees of freedom (real
/// `df`; infinite means the normal distribution).
fn t_two_sided(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    if df.is_infinite() || df > 1e7 {
        return 2.0 * normal_sf(t.abs());
    }
    inc_beta(df / 2.0, 0.5, df / (df + t * t))
}

/// Upper tail of the standard normal, accurate far into the tail.
fn normal_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical Recipes' Chebyshev fit, relative
/// error below 1.2e-7 would not do; this is W. J. Cody's rational form).
fn erfc(x: f64) -> f64 {
    // erfc through the incomplete gamma: erfc(x) = Q(1/2, x²) for x ≥ 0.
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    upper_gamma_q(0.5, x * x)
}

/// The regularized upper incomplete gamma function `Q(a, x)`.
fn upper_gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_front = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        // Series for P, then Q = 1 - P.
        let (mut sum, mut del, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..100_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        (1.0 - sum * ln_front.exp()).max(0.0)
    } else {
        // Continued fraction for Q (Lentz).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        ln_front.exp() * h
    }
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

/// edgeR's `filterByExpr(y, group = group)`: genes (columns of the
/// samples × genes `y`) with enough counts to test.
fn filter_by_expr(
    y: &[f64],
    n: usize,
    g: usize,
    group: &[u32],
    o: &PseudobulkOptions,
) -> Vec<bool> {
    let lib: Vec<f64> = (0..n).map(|i| y[i * g..(i + 1) * g].iter().sum()).collect();
    let mut sizes: BTreeMap<u32, usize> = BTreeMap::new();
    for &c in group {
        *sizes.entry(c).or_insert(0) += 1;
    }
    let mut min_size = sizes.values().copied().min().unwrap_or(0) as f64;
    if min_size > o.large_n {
        min_size = o.large_n + (min_size - o.large_n) * o.min_prop;
    }
    let cutoff = o.min_count / median(&lib) * 1e6;
    let tol = 1e-14;
    (0..g)
        .map(|j| {
            let mut above = 0.0;
            let mut total = 0.0;
            for i in 0..n {
                let v = y[i * g + j];
                total += v;
                if v / lib[i] * 1e6 >= cutoff {
                    above += 1.0;
                }
            }
            above >= min_size - tol && total >= o.min_total_count - tol
        })
        .collect()
}

/// edgeR's TMM `.calcFactorTMM(obs, ref, ...)` with its defaults.
fn tmm_factor(obs: &[f64], refc: &[f64], n_o: f64, n_r: f64) -> f64 {
    let (log_ratio_trim, sum_trim) = (0.3, 0.05);
    let mut lr = Vec::new();
    let mut ae = Vec::new();
    let mut v = Vec::new();
    for (&o, &r) in obs.iter().zip(refc) {
        let l = ((o / n_o) / (r / n_r)).log2();
        let a = ((o / n_o).log2() + (r / n_r).log2()) / 2.0;
        if l.is_finite() && a.is_finite() && a > -1e10 {
            lr.push(l);
            ae.push(a);
            v.push((n_o - o) / n_o / o + (n_r - r) / n_r / r);
        }
    }
    if lr.iter().fold(0.0_f64, |m, x| m.max(x.abs())) < 1e-6 {
        return 1.0;
    }
    let n = lr.len() as f64;
    let lo_l = (n * log_ratio_trim).floor() + 1.0;
    let hi_l = n + 1.0 - lo_l;
    let lo_s = (n * sum_trim).floor() + 1.0;
    let hi_s = n + 1.0 - lo_s;
    let (rl, ra) = (rank_average(&lr), rank_average(&ae));
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..lr.len() {
        if rl[k] >= lo_l && rl[k] <= hi_l && ra[k] >= lo_s && ra[k] <= hi_s {
            num += lr[k] / v[k];
            den += 1.0 / v[k];
        }
    }
    let f = num / den;
    if f.is_nan() {
        1.0
    } else {
        f.exp2()
    }
}

/// edgeR's TMM normalization factors (`normLibSizes`) of the samples ×
/// genes `y` with library sizes `lib`.
fn tmm_factors(y: &[f64], n: usize, g: usize, lib: &[f64]) -> Vec<f64> {
    let genes: Vec<usize> = (0..g)
        .filter(|&j| (0..n).any(|i| y[i * g + j] > 0.0))
        .collect();
    if genes.is_empty() || n == 1 {
        return vec![1.0; n];
    }
    let col = |i: usize| -> Vec<f64> { genes.iter().map(|&j| y[i * g + j]).collect() };
    let f75: Vec<f64> = (0..n).map(|i| quantile7(&col(i), 0.75) / lib[i]).collect();
    let refi = if median(&f75) < 1e-20 {
        (0..n)
            .max_by(|&a, &b| {
                let s = |i: usize| col(i).iter().map(|x| x.sqrt()).sum::<f64>();
                s(a).total_cmp(&s(b)).then(b.cmp(&a))
            })
            .unwrap_or(0)
    } else {
        let mean = f75.iter().sum::<f64>() / n as f64;
        (0..n)
            .min_by(|&a, &b| {
                (f75[a] - mean)
                    .abs()
                    .total_cmp(&(f75[b] - mean).abs())
                    .then(a.cmp(&b))
            })
            .unwrap_or(0)
    };
    let rcol = col(refi);
    let f: Vec<f64> = (0..n)
        .map(|i| tmm_factor(&col(i), &rcol, lib[i], lib[refi]))
        .collect();
    let geo = (f.iter().map(|x| x.ln()).sum::<f64>() / n as f64).exp();
    f.iter().map(|x| x / geo).collect()
}

// ── limma ─────────────────────────────────────────────────────────────────

/// A natural cubic spline basis with R's `ns(x, df, intercept = TRUE)`
/// knots (boundary at the range, interior at quantiles). Any basis of that
/// space gives `lm.fit` the same fitted values and residuals.
fn natural_spline_basis(x: &[f64], df: usize) -> Vec<Vec<f64>> {
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = if hi > lo { hi - lo } else { 1.0 };
    let s: Vec<f64> = x.iter().map(|v| (v - lo) / scale).collect();
    let n_inner = df.saturating_sub(2);
    let mut knots = vec![0.0];
    for k in 1..=n_inner {
        knots.push((quantile7(x, k as f64 / (n_inner + 1) as f64) - lo) / scale);
    }
    knots.push(1.0);
    let kk = knots.len();
    let d = |k: usize, v: f64| -> f64 {
        let c = |t: f64| (v - t).max(0.0).powi(3);
        (c(knots[k]) - c(knots[kk - 1])) / (knots[kk - 1] - knots[k])
    };
    let mut basis = vec![vec![1.0; x.len()]];
    if df >= 2 {
        basis.push(s.clone());
    }
    for k in 0..kk.saturating_sub(2) {
        basis.push(s.iter().map(|&v| d(k, v) - d(kk - 2, v)).collect());
    }
    basis.truncate(df.max(1));
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

/// limma's `fitFDist(x, df1, covariate)` for one common `df1`: the prior's
/// scale for each gene and its degrees of freedom.
fn fit_f_dist(x: &[f64], df1: f64, covariate: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let ok: Vec<bool> = x.iter().map(|&v| v.is_finite() && v > -1e-15).collect();
    let idx: Vec<usize> = (0..n).filter(|&i| ok[i]).collect();
    let nok = idx.len();
    if nok == 1 {
        return (vec![x[idx[0]]; n], 0.0);
    }
    let cov_ok: Vec<f64> = idx.iter().map(|&i| covariate[i]).collect();
    let mut uniq = cov_ok.clone();
    uniq.sort_by(f64::total_cmp);
    uniq.dedup();
    let mut splinedf = 1 + usize::from(nok >= 3) + usize::from(nok >= 6) + usize::from(nok >= 30);
    splinedf = splinedf.min(uniq.len());
    let xs: Vec<f64> = idx.iter().map(|&i| x[i].max(0.0)).collect();
    let mut m = median(&xs);
    if m == 0.0 {
        m = 1.0;
    }
    let xs: Vec<f64> = xs.iter().map(|&v| v.max(1e-5 * m)).collect();
    let lmd = logmdigamma(df1 / 2.0);
    let e: Vec<f64> = xs.iter().map(|v| v.ln() + lmd).collect();
    let (emean_ok, evar): (Vec<f64>, f64) = if splinedf < 2 {
        let mean = e.iter().sum::<f64>() / nok as f64;
        let var = e.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nok as f64 - 1.0);
        (vec![mean; nok], var)
    } else {
        let basis = natural_spline_basis(&cov_ok, splinedf);
        let rank = basis.len();
        let (fitted, rss) = least_squares(&basis, &e).unwrap_or_else(|| {
            let mean = e.iter().sum::<f64>() / nok as f64;
            let rss = e.iter().map(|v| (v - mean) * (v - mean)).sum();
            (vec![mean; nok], rss)
        });
        (fitted, rss / (nok - rank) as f64)
    };
    // Genes left out of the fit take the trend's value at their covariate;
    // with every variance finite (the pseudobulk case) there are none.
    let mut emean = vec![f64::NAN; n];
    for (k, &i) in idx.iter().enumerate() {
        emean[i] = emean_ok[k];
    }
    let evar = evar - trigamma(df1 / 2.0);
    if evar > 0.0 {
        let df2 = 2.0 * trigamma_inverse(evar);
        let lmd2 = logmdigamma(df2 / 2.0);
        (emean.iter().map(|&em| (em - lmd2).exp()).collect(), df2)
    } else {
        (emean.iter().map(|&em| em.exp()).collect(), f64::INFINITY)
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

    // filterByExpr on the condition groups, then library sizes of the genes
    // kept (keep.lib.sizes = FALSE) and TMM.
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
    // log2 CPM with the prior count scaled by library size, as edgeR's cpm.
    let logcpm: Vec<f64> = (0..n)
        .flat_map(|i| {
            let pc = opts.prior_count * eff[i] / mean_eff;
            let denom = eff[i] + 2.0 * pc;
            let yk = &yk;
            (0..h).map(move |j| ((yk[i * h + j] + pc) / denom).log2() + 1e6_f64.log2())
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

    // lmFit: coefficients and residual variance per gene.
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

    // eBayes(trend = TRUE): the prior variance follows mean expression.
    let (s2_prior, df_prior) = fit_f_dist(&s2, df, &amean);
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
    fn special_functions_match_r() {
        // R 4.5: trigamma, psigamma(deriv = 2), lgamma, 2 * pt(-|t|, df),
        // log(x) - digamma(x).
        let close =
            |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol * b.abs(), "{a} vs {b}");
        close(trigamma(0.7), 2.834_049_156_694_610_4, 1e-13);
        close(trigamma(12.3), 0.084_695_170_245_916_38, 1e-13);
        close(tetragamma(1.3), -1.198_462_514_651_956_5, 1e-12);
        close(ln_gamma(4.5), 2.453_736_570_842_442_3, 1e-14);
        close(trigamma(trigamma_inverse(0.37)), 0.37, 1e-10);
        close(t_two_sided(2.1, 7.5), 0.071_237_071_458_078_05, 1e-12);
        close(t_two_sided(-40.0, 12.0), 3.851_229_580_772_037e-14, 1e-10);
        close(logmdigamma(2.2), 0.244_163_923_623_125_33, 1e-14);
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
