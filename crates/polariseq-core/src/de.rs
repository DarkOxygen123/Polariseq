//! Differential expression analysis: per-cluster marker gene identification.
//!
//! For each cluster, finds genes that are significantly upregulated compared
//! to all other cells. Uses Welch's t-test (unequal variance) which is fast
//! and appropriate for single-cell expression data.
//!
//! The sparse implementation makes a single pass over the CSR matrix to
//! accumulate per-gene per-cluster statistics (sum, sum-of-squares, count),
//! then computes t-statistics in closed form. This is `O(nnz)` for the data
//! pass + `O(n_genes × n_groups)` for the statistics.

use crate::matrix::CsrMatrix;
use crate::preprocess::{cluster_stats_blocks, ClusterStats};

/// Errors from differential expression.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DeError {
    /// Empty matrix or labels.
    #[error("empty input: {n_cells} cells, {n_genes} genes, {n_groups} groups")]
    EmptyInput {
        /// Cell count.
        n_cells: usize,
        /// Gene count.
        n_genes: usize,
        /// Group count.
        n_groups: usize,
    },
    /// Label array length mismatch.
    #[error("labels length ({n_labels}) != n_cells ({n_cells})")]
    LabelMismatch {
        /// Label array length.
        n_labels: usize,
        /// Expected cell count.
        n_cells: usize,
    },
}

/// Per-group DE results, sorted by descending |z-score|.
#[derive(Debug, Clone)]
pub struct GroupDeResult {
    /// Group (cluster) ID.
    pub group_id: u32,
    /// Gene indices sorted by descending |score|. Into `var_names`.
    pub gene_indices: Vec<usize>,
    /// Z-scores from Welch's t-test (positive = upregulated).
    pub scores: Vec<f32>,
    /// log2 fold change: `log2(mean_group / mean_rest)`.
    pub logfoldchanges: Vec<f32>,
    /// Raw p-values (two-sided t-test). `f64`, not `f32`, deliberately: a
    /// strong marker's p-value is routinely below f32's smallest normal
    /// (~1.2e-38) — pbmc3k's top gene is 1.9e-187 — so storing these as
    /// `f32` flushes every significant gene to exactly 0 and throws away the
    /// ordering that makes the column worth reporting.
    pub pvals: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-values. `f64` for the same reason.
    pub pvals_adj: Vec<f64>,
    /// Fraction of cells in this group expressing each gene.
    pub pts: Vec<f32>,
}

impl GroupDeResult {
    /// An empty result for a group too small to test.
    fn empty(group_id: u32) -> Self {
        Self {
            group_id,
            gene_indices: Vec::new(),
            scores: Vec::new(),
            logfoldchanges: Vec::new(),
            pvals: Vec::new(),
            pvals_adj: Vec::new(),
            pts: Vec::new(),
        }
    }
}

/// Complete DE result for all groups.
#[derive(Debug, Clone)]
pub struct DeResult {
    /// Results per group.
    pub groups: Vec<GroupDeResult>,
    /// Number of genes tested.
    pub n_genes: usize,
    /// Number of cells.
    pub n_cells: usize,
}

/// Run differential expression: each cluster vs all others.
///
/// For each gene g and cluster c, computes Welch's t-test comparing
/// expression of g in cells of cluster c vs all other cells.
///
/// # Errors
/// Returns [`DeError`] on invalid input.
pub fn rank_genes_groups(
    data: &CsrMatrix,
    labels: &[u32],
    n_groups: usize,
) -> Result<DeResult, DeError> {
    let (n_cells, n_genes) = data.shape();
    if n_cells == 0 || n_genes == 0 || n_groups == 0 {
        return Err(DeError::EmptyInput {
            n_cells,
            n_genes,
            n_groups,
        });
    }
    if labels.len() != n_cells {
        return Err(DeError::LabelMismatch {
            n_labels: labels.len(),
            n_cells,
        });
    }

    // ── Phase 1: accumulate per-gene per-cluster statistics ───────────
    // For each gene g and cluster c:
    //   sum[g][c]     = sum of expression values
    //   sq_sum[g][c]  = sum of *squared* expression values
    //   count[g][c]   = number of cells with nonzero expression
    // Also track per-cluster cell counts.
    //
    // The sum of squares is what makes the variance a variance. Without it
    // there is no second moment to compute one from, and an earlier version
    // of this function tried to fake one from `sum^2 / count` — which is not
    // a second moment at all, and inflated the standard error enough to drive
    // every p-value to ~0.5 (measured on pbmc3k: top marker t = 0.465,
    // p = 0.64, where scanpy reports t = 33.4, p = 1.9e-187).

    //
    // The statistics come from `cluster_stats_blocks`, the kernel the
    // out-of-core path streams from a file, with partials folded in block
    // order: the result no longer depends on how rayon split the rows, and
    // the two paths share every step after the matrix is read.
    let stats = cluster_stats_blocks(data, labels, n_groups);

    // ── Phase 2: compute t-statistics per gene per group ───────────────
    rank_genes_groups_from_stats(&stats, None)
}

/// Welch's t-test for every group from per-cluster statistics: phase 2 of
/// [`rank_genes_groups`], shared with the out-of-core path, which streams
/// the same statistics from a file.
///
/// `gene_keep` restricts the test to a subset of genes, as if the others had
/// been removed from the matrix (`filter_genes`): the Benjamini-Hochberg
/// adjustment then counts only the genes tested. Gene indices in the result
/// refer to the genes of `stats`. Cells are those with a label below
/// `stats.n_clusters`; the rest were excluded when the statistics were made.
///
/// # Errors
/// Returns [`DeError::EmptyInput`] when no cell, gene or group is left.
pub fn rank_genes_groups_from_stats(
    stats: &ClusterStats,
    gene_keep: Option<&[bool]>,
) -> Result<DeResult, DeError> {
    let n_groups = stats.n_clusters;
    let n_cells: usize = stats.cluster_sizes.iter().map(|&s| s as usize).sum();
    let kept: Vec<usize> = match gene_keep {
        Some(keep) => (0..stats.n_genes).filter(|&g| keep[g]).collect(),
        None => (0..stats.n_genes).collect(),
    };
    let n_genes = kept.len();
    if n_cells == 0 || n_genes == 0 || n_groups == 0 {
        return Err(DeError::EmptyInput {
            n_cells,
            n_genes,
            n_groups,
        });
    }
    let restrict = |v: &[f64]| -> Vec<f64> {
        kept.iter()
            .flat_map(|&g| v[g * n_groups..(g + 1) * n_groups].iter().copied())
            .collect()
    };
    let (sums, sq_sums) = (restrict(&stats.sums), restrict(&stats.sq_sums));
    let counts: Vec<u32> = kept
        .iter()
        .flat_map(|&g| {
            stats.counts[g * n_groups..(g + 1) * n_groups]
                .iter()
                .copied()
        })
        .collect();
    let acc = Accumulators {
        sums: &sums,
        sq_sums: &sq_sums,
        counts: &counts,
        n_cells,
        n_genes,
        n_groups,
    };
    let groups = stats
        .cluster_sizes
        .iter()
        .enumerate()
        .map(|(g, &size)| {
            let mut r = de_for_group(g, size, &acc);
            for i in &mut r.gene_indices {
                *i = kept[*i];
            }
            r
        })
        .collect();

    Ok(DeResult {
        groups,
        n_genes,
        n_cells,
    })
}

/// Phase-1 accumulators, flattened as `(gene * n_groups + group)`.
struct Accumulators<'a> {
    /// Σx per gene per cluster.
    sums: &'a [f64],
    /// Σx² per gene per cluster — the second moment the variance needs.
    sq_sums: &'a [f64],
    /// Cells with a nonzero value, per gene per cluster.
    counts: &'a [u32],
    n_cells: usize,
    n_genes: usize,
    n_groups: usize,
}

/// Welch's t-test of group `g` against all other cells, for every gene.
fn de_for_group(g: usize, group_size: u32, acc: &Accumulators<'_>) -> GroupDeResult {
    let Accumulators {
        sums,
        sq_sums,
        counts,
        n_cells,
        n_genes,
        n_groups,
    } = *acc;
    let group_id = g as u32;
    let group_size = f64::from(group_size);
    let rest_size = (n_cells as f64) - group_size;

    if group_size < 2.0 || rest_size < 2.0 {
        // Degenerate group: nothing to test.
        return GroupDeResult::empty(group_id);
    }

    // (gene, z-score, log2 fold change, p-value, fraction expressing)
    let mut gene_results: Vec<(usize, f64, f64, f64, f64)> = Vec::with_capacity(n_genes);

    for gene in 0..n_genes {
        let sum_row = &sums[gene * n_groups..(gene + 1) * n_groups];
        let sq_row = &sq_sums[gene * n_groups..(gene + 1) * n_groups];
        let count_row = &counts[gene * n_groups..(gene + 1) * n_groups];
        let sum_in = sum_row[g];
        let sq_in = sq_row[g];
        let count_in = f64::from(count_row[g]);

        // Total sum/count for this gene across all clusters.
        let sum_total: f64 = sum_row.iter().sum();
        let sq_total: f64 = sq_row.iter().sum();
        let count_total: f64 = count_row.iter().map(|&c| f64::from(c)).sum();
        let sum_out = sum_total - sum_in;
        let sq_out = sq_total - sq_in;
        let count_out = count_total - count_in;
        debug_assert!(count_out >= 0.0);

        // Means (sparse-aware: include zeros).
        let mean_in = sum_in / group_size;
        let mean_out = if rest_size > 0.0 {
            sum_out / rest_size
        } else {
            0.0
        };

        // Sample variance over *all* cells in the group, zeros included.
        // Sparse storage omits the zeros, but they are real observations, and
        // both moments are already correct for them: a zero contributes
        // nothing to `sum` or `sq_sum` while still counting towards `n`. So
        //
        //     var = (Σx² − n·mean²) / (n − 1)
        //
        // needs no sparse-specific correction. `max(0.0)` guards the
        // cancellation when every value in the group is identical.
        let var_in = ((sq_in - group_size * mean_in * mean_in) / (group_size - 1.0)).max(0.0);
        let var_out = ((sq_out - rest_size * mean_out * mean_out) / (rest_size - 1.0)).max(0.0);

        // Welch's t-statistic.
        let se = (var_in / group_size + var_out / rest_size).sqrt();
        if se < 1e-12 {
            continue; // skip genes with no variance
        }
        let t_stat = (mean_in - mean_out) / se;

        // Approximate z-score (for large n, t ≈ z).
        let z_score = t_stat;

        // log2 fold change.
        let lfc = if mean_in > 1e-10 && mean_out > 1e-10 {
            (mean_in / mean_out).log2()
        } else if mean_in > 1e-10 {
            10.0 // cap at large value
        } else {
            -10.0
        };

        // P-value from normal approximation (two-sided).
        // Two-sided, from the tail directly — see `normal_sf`. Going through
        // `1 - cdf` collapses every strong marker to p = 0.
        let pval = (2.0 * normal_sf(z_score.abs())).min(1.0);

        // Fraction expressing in this group.
        let pts = count_in / group_size;

        gene_results.push((gene, z_score, lfc, pval, pts));
    }

    // Sort by descending z-score (upregulated first).
    gene_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let pvals: Vec<f64> = gene_results.iter().map(|r| r.3).collect();
    let pvals_adj = benjamini_hochberg(&pvals);

    GroupDeResult {
        group_id,
        gene_indices: gene_results.iter().map(|r| r.0).collect(),
        scores: gene_results.iter().map(|r| r.1 as f32).collect(),
        logfoldchanges: gene_results.iter().map(|r| r.2 as f32).collect(),
        pvals,
        pvals_adj,
        pts: gene_results.iter().map(|r| r.4 as f32).collect(),
    }
}

/// Benjamini–Hochberg adjusted p-values, returned in the same order as `pvals`.
fn benjamini_hochberg(pvals: &[f64]) -> Vec<f64> {
    let n_tests = pvals.len();
    let mut order: Vec<usize> = (0..n_tests).collect();
    order.sort_by(|&a, &b| {
        pvals[a]
            .partial_cmp(&pvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step-up from the largest p-value, enforcing monotonicity.
    let mut bh = vec![1.0_f64; n_tests];
    for i in (0..n_tests).rev() {
        let rank = (i + 1) as f64;
        bh[i] = (pvals[order[i]] * n_tests as f64 / rank).min(1.0);
        if i + 1 < n_tests {
            bh[i] = bh[i].min(bh[i + 1]);
        }
    }

    // Map back from rank order to input order.
    let mut adjusted = vec![1.0_f64; n_tests];
    for (rank, &orig) in order.iter().enumerate() {
        adjusted[orig] = bh[rank];
    }
    adjusted
}

/// Standard normal CDF approximation (Abramowitz & Stegun 7.1.26).
/// Upper-tail probability `P(Z > x)`, computed directly rather than as
/// `1 - cdf(x)`.
///
/// Computing the tail by subtraction is what makes a differential-expression
/// table saturate: a marker with `z = 33` has `cdf(z) = 1.0` in f64, so
/// `1 - cdf` is exactly 0 and every strong gene reports `p = 0`,
/// indistinguishable from every other strong gene.
///
/// Two regimes, because no single cheap formula covers both:
///
/// - `x < 2`: the Abramowitz-Stegun 26.2.17 rational approximation, accurate
///   to ~7.5e-8 *absolute* — negligible against values of 0.023 and up.
/// - `x >= 2`: the Mills-ratio continued fraction
///   `phi(x) / (x + 1/(x + 2/(x + 3/(x + ...))))`, which is accurate in
///   *relative* terms all the way down. Verified against scipy: relative
///   error below 1.2e-13 from x = 2 to x = 33.37, where it returns
///   1.868e-244 rather than underflowing.
///
/// The rational approximation cannot be used in the tail at any depth: its
/// error floor of 7.5e-8 already dwarfs the true value by x = 5.3.
fn normal_sf(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - normal_sf(-x);
    }
    let phi = 0.398_942_280_401_432_7 * (-x * x / 2.0).exp();
    if x >= 2.0 {
        // Evaluated backwards from the deepest term; 60 levels is far past
        // convergence for x >= 2 and costs nothing at this call frequency.
        let mut cf = 0.0_f64;
        for k in (1..=60).rev() {
            cf = f64::from(k) / (x + cf);
        }
        return phi / (x + cf);
    }
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    phi * t
        * (0.319_381_530
            + t * (-0.356_563_782 + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranking from statistics restricted to a gene subset equals ranking
    /// the matrix with the other genes removed. The out-of-core path streams
    /// statistics for every gene of the file and drops the genes quality
    /// control removed; it must give the in-memory answer exactly.
    #[test]
    fn ranking_from_stats_with_a_gene_mask_equals_ranking_the_subset() {
        let (nrows, ncols, k) = (600, 23, 3_usize);
        let labels: Vec<u32> = (0..nrows).map(|i| (i % k) as u32).collect();
        let mut dense = vec![0.0_f32; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                if (i + j) % 3 != 0 {
                    let marker = if j % k == labels[i] as usize {
                        2.0
                    } else {
                        0.0
                    };
                    dense[i * ncols + j] = ((i * 7 + j * 13) % 11) as f32 * 0.3 + marker;
                }
            }
        }
        let m = CsrMatrix::from_dense_row_major(&dense, nrows, ncols);
        let keep: Vec<bool> = (0..ncols).map(|j| j % 5 != 2).collect();
        let kept: Vec<usize> = (0..ncols).filter(|&j| keep[j]).collect();
        let want = rank_genes_groups(&m.select_cols_mask(&keep).unwrap(), &labels, k).unwrap();
        let stats = cluster_stats_blocks(&m, &labels, k);
        let got = rank_genes_groups_from_stats(&stats, Some(&keep)).unwrap();
        assert_eq!((got.n_genes, got.n_cells), (want.n_genes, want.n_cells));
        for (g, w) in got.groups.iter().zip(&want.groups) {
            let mapped: Vec<usize> = w.gene_indices.iter().map(|&i| kept[i]).collect();
            assert_eq!(g.gene_indices, mapped);
            assert_eq!(g.scores, w.scores);
            assert_eq!(g.pvals_adj, w.pvals_adj);
        }
    }

    /// The t-statistic must match the analytic Welch value on data whose
    /// moments are known by hand.
    ///
    /// This is the test that would have caught the original defect. The
    /// variance was being faked from `sum^2 / count` — no sum of squares was
    /// ever accumulated — which inflated the standard error by roughly
    /// sqrt(n) and drove every p-value to ~0.5. On pbmc3k the top marker
    /// scored t = 0.465 (p = 0.64) where scanpy reported t = 33.4
    /// (p = 1.9e-187). Nothing in the suite noticed, because every assertion
    /// only checked the *sign* and the *ordering* of the scores, both of
    /// which survive an arbitrary positive rescaling of the denominator.
    #[test]
    fn welch_t_matches_the_analytic_value() {
        // Gene 0: group A = [4, 4, 4, 4], group B = [1, 1, 1, 1].
        // Both variances are 0, so add one different value per group to make
        // them nonzero and hand-computable:
        //   A = [4, 4, 4, 2]  mean 3.5  var = ((3*0.25)+2.25)/3 = 1.0
        //   B = [1, 1, 1, 3]  mean 1.5  var = 1.0
        //   se = sqrt(1/4 + 1/4) = 0.7071..., t = (3.5 - 1.5) / se = 2.828...
        let values: [f32; 8] = [4.0, 4.0, 4.0, 2.0, 1.0, 1.0, 1.0, 3.0];
        let mut indptr = vec![0_u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for &v in &values {
            indices.push(0_u32);
            data.push(v);
            indptr.push(data.len() as u32);
        }
        let m = CsrMatrix::new(indptr, indices, data, 8, 1).unwrap();
        let labels = [0_u32, 0, 0, 0, 1, 1, 1, 1];

        let res = rank_genes_groups(&m, &labels, 2).unwrap();
        let g0 = &res.groups[0];
        assert_eq!(g0.gene_indices[0], 0);

        let expected = 2.0 / (0.5_f64).sqrt(); // 2.8284...
        assert!(
            (f64::from(g0.scores[0]) - expected).abs() < 1e-4,
            "t = {} but Welch says {expected}",
            g0.scores[0]
        );
    }

    /// A strong effect must not report `p = 0`.
    ///
    /// Two ways this used to happen and both are guarded here: computing the
    /// tail as `1 - cdf` (which saturates once cdf rounds to 1.0), and
    /// storing the result in `f32` (whose smallest normal is ~1.2e-38, while
    /// a real marker reaches 1e-187). Either one collapses every significant
    /// gene to the same value and destroys the ordering.
    #[test]
    fn extreme_significance_does_not_flush_to_zero() {
        // z = 30 is the magnitude a real marker reaches on pbmc3k.
        let p = 2.0 * normal_sf(30.0);
        assert!(p > 0.0, "tail underflowed to zero");
        assert!(p < 1e-190, "tail is implausibly large: {p}");
        assert!(
            p < f64::from(f32::MIN_POSITIVE),
            "this value is below f32's range ({p:e}); storing it as f32 \
             would flush it to zero"
        );
    }

    /// The two regimes must agree where they meet, and both must match
    /// known values. Reference figures are scipy's `norm.sf`.
    #[test]
    fn survival_function_matches_known_values() {
        // Straddle the crossover and check the step is no larger than the
        // function's own slope over that interval. A fixed tolerance would
        // be arbitrary: sf falls by phi(2) * dx = 1.08e-4 across [1.999,
        // 2.001] on its own, so any honest test has to allow that much.
        let (below, above) = (normal_sf(1.999), normal_sf(2.001));
        let phi_2 = 0.398_942_280_401_432_7 * (-2.0_f64).exp();
        let slope_change = phi_2 * 0.002;
        assert!(below > above, "sf must decrease: {below:e} then {above:e}");
        assert!(
            (below - above) < 1.5 * slope_change,
            "step at the x=2 crossover ({:e}) exceeds the function's own \
             change over the interval ({slope_change:e})",
            below - above
        );
        for &(x, want) in &[
            (1.0_f64, 1.586_552_5e-1),
            (3.0, 1.349_898_0e-3),
            (5.0, 2.866_515_7e-7),
            (8.0, 6.220_960_6e-16),
            (30.0, 4.906_714e-198),
        ] {
            let got = normal_sf(x);
            assert!(
                (got - want).abs() / want < 1e-6,
                "sf({x}) = {got:e}, want {want:e}"
            );
        }
    }

    fn make_matrix() -> CsrMatrix {
        // 6 cells × 4 genes, 2 clusters.
        // Cluster 0 (cells 0-2): high expression of gene 0.
        // Cluster 1 (cells 3-5): high expression of gene 1.
        // Genes 2,3: uniform (no DE).
        let data = vec![
            5.0_f32, 1.0, 1.0, 1.0, // cell 0 (cluster 0)
            6.0, 1.0, 1.0, 0.0, // cell 1 (cluster 0)
            4.0, 2.0, 0.0, 1.0, // cell 2 (cluster 0)
            1.0, 5.0, 1.0, 1.0, // cell 3 (cluster 1)
            1.0, 6.0, 1.0, 0.0, // cell 4 (cluster 1)
            2.0, 4.0, 0.0, 1.0, // cell 5 (cluster 1)
        ];
        let indptr = vec![0, 4, 8, 12, 16, 20, 24];
        let indices: Vec<u32> = (0..24).map(|i| (i % 4) as u32).collect();
        CsrMatrix::new(indptr, indices, data, 6, 4).unwrap()
    }

    #[test]
    fn finds_marker_genes() {
        let data = make_matrix();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let result = rank_genes_groups(&data, &labels, 2).unwrap();

        assert_eq!(result.groups.len(), 2);

        // Cluster 0: gene 0 should be top marker (highest in cluster 0).
        let g0 = &result.groups[0];
        assert_eq!(g0.gene_indices[0], 0, "gene 0 should be top for cluster 0");
        assert!(g0.scores[0] > 0.0, "upregulated");

        // Cluster 1: gene 1 should be top marker.
        let g1 = &result.groups[1];
        assert_eq!(g1.gene_indices[0], 1, "gene 1 should be top for cluster 1");
        assert!(g1.scores[0] > 0.0, "upregulated");
    }

    #[test]
    fn logfoldchange_direction() {
        let data = make_matrix();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let result = rank_genes_groups(&data, &labels, 2).unwrap();

        let g0 = &result.groups[0];
        // Gene 0 is upregulated in cluster 0 → positive LFC.
        let gene0_pos = g0.gene_indices.iter().position(|&g| g == 0).unwrap();
        assert!(g0.logfoldchanges[gene0_pos] > 0.0);

        // Gene 1 is downregulated in cluster 0 → negative LFC.
        let gene1_pos = g0.gene_indices.iter().position(|&g| g == 1).unwrap();
        assert!(g0.logfoldchanges[gene1_pos] < 0.0);
    }

    #[test]
    fn pvals_in_range() {
        let data = make_matrix();
        let labels = vec![0, 0, 0, 1, 1, 1];
        let result = rank_genes_groups(&data, &labels, 2).unwrap();
        for g in &result.groups {
            for &p in &g.pvals {
                assert!((0.0..=1.0).contains(&p), "pval {p} out of range");
            }
        }
    }

    #[test]
    fn bh_is_monotone_and_bounded() {
        let p = vec![0.01, 0.04, 0.03, 0.5, 0.001];
        let adj = benjamini_hochberg(&p);
        assert_eq!(adj.len(), 5);
        assert!(adj.iter().all(|&a| (0.0..=1.0).contains(&a)));
        // Adjusted values never fall below raw values.
        for (raw, a) in p.iter().zip(&adj) {
            assert!(a >= raw, "adjusted {a} < raw {raw}");
        }
        // Smallest raw p keeps the smallest adjusted p.
        assert!(adj[4] <= adj[0] && adj[0] <= adj[2] && adj[2] <= adj[1]);
    }

    #[test]
    fn rejects_mismatched_labels() {
        let data = make_matrix();
        let labels = vec![0, 0, 1]; // wrong length
        assert!(rank_genes_groups(&data, &labels, 2).is_err());
    }
}
