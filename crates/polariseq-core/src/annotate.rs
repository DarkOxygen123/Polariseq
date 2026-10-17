//! Marker-based cell type scores, streamed over row blocks.
//!
//! Each marker gene is standardized over all cells (its mean and standard
//! deviation from a first pass), and each cell's score for a cell type is the
//! weighted sum of its standardized marker values, `W` holding positive
//! weights for a type's markers and negative ones for genes the type should
//! not express (Sc-Type, Ianevski et al. 2022; Cell Marker Accordion,
//! Busarello et al. 2025). A zero entry of the sparse matrix is a value too:
//! its standardized value enters through a per-type offset, so the score is
//! exact without densifying anything. The weights come from Python
//! (`python/polariseq/_annotate.py`), following the Cell Marker Accordion's
//! MIT-licensed code; see `THIRD_PARTY_NOTICES.md`.
//!
//! Memory stays at the per-cluster sums and three numbers per cell (the best
//! type, its score, and the runner-up's score): no cells-by-types matrix is
//! held, so an atlas is scored within the budget of one block.

use crate::rowblocks::{BlockFold, RowBlocks};

/// One block's share of pass 2: cluster sums, cluster counts, cells scoring
/// above zero per cluster and type, and each cell's best type, best score and
/// runner-up score.
type Part = (Vec<f64>, Vec<u64>, Vec<u64>, Vec<u32>, Vec<f32>, Vec<f32>);

/// What [`marker_scores`] returns.
#[derive(Debug, Clone)]
pub struct MarkerScores {
    /// Sum of each cluster's cells' scores, `n_groups x n_types`, row-major.
    pub group_sums: Vec<f64>,
    /// Cells per cluster.
    pub group_counts: Vec<u64>,
    /// Cells of each cluster scoring above zero for each type, `n_groups x
    /// n_types`, row-major: at least a quarter of them is the Cell Marker
    /// Accordion's test (the cluster's 75th percentile score above zero).
    pub group_positive: Vec<u64>,
    /// Each cell's best-scoring type.
    pub best: Vec<u32>,
    /// Each cell's best score.
    pub best_score: Vec<f32>,
    /// Each cell's second-best score.
    pub second_score: Vec<f32>,
    /// Each marker gene's mean over the cells.
    pub mean: Vec<f64>,
    /// Each marker gene's standard deviation over the cells.
    pub sd: Vec<f64>,
}

/// Score every cell for every cell type.
///
/// `cols[k]` is the matrix column of marker gene `k`; `weights` is
/// `k x n_types`, row-major. `labels[i]` is cell `i`'s cluster; a label
/// `>= n_groups` leaves the cell out of the cluster sums (its own scores are
/// still returned).
///
/// # Panics
/// Panics if `weights.len() != cols.len() * n_types` or there are fewer
/// labels than rows.
#[must_use]
pub fn marker_scores(
    src: &dyn RowBlocks,
    cols: &[u32],
    weights: &[f32],
    n_types: usize,
    labels: &[u32],
    n_groups: usize,
) -> MarkerScores {
    let (n_rows, n_cols) = (src.n_rows(), src.n_cols());
    let n_markers = cols.len();
    assert_eq!(
        weights.len(),
        n_markers * n_types,
        "weights must be markers x types"
    );
    assert!(labels.len() >= n_rows, "one label per row");
    let mut slot = vec![u32::MAX; n_cols];
    for (i, &c) in cols.iter().enumerate() {
        slot[c as usize] = i as u32;
    }
    let chunk = crate::rsvd::chunk_rows(n_rows);

    let (mean, sd) = marker_moments(src, chunk, &slot, n_markers);
    let (scaled, offset) = standardized_weights(weights, &mean, &sd, n_types);

    // pass 2: per-cell scores, folded into cluster sums and each cell's two best
    let fold2: BlockFold<Part> = BlockFold::new(|a, b| {
        for (x, y) in a.0.iter_mut().zip(&b.0) {
            *x += y;
        }
        for (x, y) in a.1.iter_mut().zip(&b.1) {
            *x += y;
        }
        for (x, y) in a.2.iter_mut().zip(&b.2) {
            *x += y;
        }
        a.3.extend(b.3);
        a.4.extend(b.4);
        a.5.extend(b.5);
    });
    src.for_each_block(chunk, &|off, blk| {
        let mut gs = vec![0.0_f64; n_groups * n_types];
        let mut gc = vec![0_u64; n_groups];
        let mut gp = vec![0_u64; n_groups * n_types];
        let (mut best, mut bs, mut ss) = (
            Vec::with_capacity(blk.n_rows),
            Vec::with_capacity(blk.n_rows),
            Vec::with_capacity(blk.n_rows),
        );
        let mut acc = vec![0.0_f32; n_types];
        for i in 0..blk.n_rows {
            acc.copy_from_slice(&offset);
            let (a, b) = (blk.indptr[i] as usize, blk.indptr[i + 1] as usize);
            for (&c, &v) in blk.indices[a..b].iter().zip(&blk.data[a..b]) {
                let m = slot[c as usize];
                if m != u32::MAX {
                    let w = &scaled[m as usize * n_types..(m as usize + 1) * n_types];
                    for (x, &wt) in acc.iter_mut().zip(w) {
                        *x += v * wt;
                    }
                }
            }
            let (b1, s1, s2) = two_best(&acc);
            best.push(b1);
            bs.push(s1);
            ss.push(s2);
            let g = labels[off + i] as usize;
            if g < n_groups {
                gc[g] += 1;
                let span = g * n_types..(g + 1) * n_types;
                for ((x, p), &v) in gs[span.clone()].iter_mut().zip(&mut gp[span]).zip(&acc) {
                    *x += f64::from(v);
                    *p += u64::from(v > 0.0);
                }
            }
        }
        fold2.store(off, (gs, gc, gp, best, bs, ss));
    });
    let (group_sums, group_counts, group_positive, best, best_score, second_score) =
        fold2.finish((
            vec![0.0; n_groups * n_types],
            vec![0; n_groups],
            vec![0; n_groups * n_types],
            Vec::with_capacity(n_rows),
            Vec::with_capacity(n_rows),
            Vec::with_capacity(n_rows),
        ));
    MarkerScores {
        group_sums,
        group_counts,
        group_positive,
        best,
        best_score,
        second_score,
        mean,
        sd,
    }
}

/// Weights divided by each marker's standard deviation (a marker that never
/// varies says nothing: its weight is dropped), and per type the score of a
/// cell with no counts in any marker: the sum of `-mean / sd` times the weight.
fn standardized_weights(
    weights: &[f32],
    mean: &[f64],
    sd: &[f64],
    n_types: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_markers = mean.len();
    let scaled: Vec<f32> = (0..n_markers)
        .flat_map(|g| {
            let inv = if sd[g] > 1e-9 { 1.0 / sd[g] } else { 0.0 };
            (0..n_types).map(move |t| (f64::from(weights[g * n_types + t]) * inv) as f32)
        })
        .collect();
    // the standardized value of a zero: -mean/sd, summed into one offset per type
    let mut offset = vec![0.0_f64; n_types];
    for g in 0..n_markers {
        for (t, o) in offset.iter_mut().enumerate() {
            *o -= mean[g] * f64::from(scaled[g * n_types + t]);
        }
    }
    let offset: Vec<f32> = offset.iter().map(|&o| o as f32).collect();
    (scaled, offset)
}

/// The best type, its score and the runner-up's score (the best's when there
/// is one type).
fn two_best(acc: &[f32]) -> (u32, f32, f32) {
    let (mut b1, mut s1, mut s2) = (0_u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (t, &x) in acc.iter().enumerate() {
        if x > s1 {
            s2 = s1;
            s1 = x;
            b1 = t as u32;
        } else if x > s2 {
            s2 = x;
        }
    }
    (b1, s1, if acc.len() > 1 { s2 } else { s1 })
}

/// Pass 1: each marker gene's mean and standard deviation over the cells.
fn marker_moments(
    src: &dyn RowBlocks,
    chunk: usize,
    slot: &[u32],
    n_markers: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_rows = src.n_rows();
    let fold: BlockFold<(Vec<f64>, Vec<f64>)> = BlockFold::new(|(s1, q1), (s2, q2)| {
        for (a, b) in s1.iter_mut().zip(&s2) {
            *a += b;
        }
        for (a, b) in q1.iter_mut().zip(&q2) {
            *a += b;
        }
    });
    src.for_each_block(chunk, &|off, blk| {
        let mut sum = vec![0.0_f64; n_markers];
        let mut sum_sq = vec![0.0_f64; n_markers];
        for (&col, &value) in blk.indices.iter().zip(blk.data) {
            let marker = slot[col as usize];
            if marker != u32::MAX {
                let x = f64::from(value);
                sum[marker as usize] += x;
                sum_sq[marker as usize] += x * x;
            }
        }
        fold.store(off, (sum, sum_sq));
    });
    let (sums, sq) = fold.finish((vec![0.0; n_markers], vec![0.0; n_markers]));
    let nf = n_rows.max(1) as f64;
    let mean: Vec<f64> = sums.iter().map(|s| s / nf).collect();
    let sd: Vec<f64> = sq
        .iter()
        .zip(&mean)
        .map(|(q, m)| (q / nf - m * m).max(0.0).sqrt())
        .collect();
    (mean, sd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::CsrMatrix;

    #[test]
    fn scores_equal_the_dense_standardized_sum() {
        // 4 cells x 3 genes; type 0 = +gene0, type 1 = +gene1 -gene2
        let dense = [
            1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let matrix = CsrMatrix::from_dense_row_major(&dense, 4, 3);
        let weights = [1.0_f32, 0.0, 0.0, 1.0, 0.0, -1.0];
        let got = marker_scores(&matrix, &[0, 1, 2], &weights, 2, &[0, 0, 1, 1], 2);
        let zscores = |gene: usize| -> Vec<f64> {
            let col: Vec<f64> = (0..4).map(|i| f64::from(dense[i * 3 + gene])).collect();
            let mu = col.iter().sum::<f64>() / 4.0;
            let sd = (col.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / 4.0).sqrt();
            col.iter().map(|x| (x - mu) / sd).collect()
        };
        let (z0, z1, z2) = (zscores(0), zscores(1), zscores(2));
        let want: Vec<[f64; 2]> = (0..4).map(|i| [z0[i], z1[i] - z2[i]]).collect();
        for (cell, expected) in want.iter().enumerate() {
            let top = usize::from(expected[0] < expected[1]);
            assert_eq!(got.best[cell] as usize, top);
            assert!((f64::from(got.best_score[cell]) - expected[top]).abs() < 1e-5);
        }
        let group1: f64 = want[0][1] + want[1][1];
        assert!((got.group_sums[1] - group1).abs() < 1e-5);
        assert_eq!(got.group_counts, vec![2, 2]);
        let positive =
            |g: usize, t: usize| (0..4).filter(|&c| c / 2 == g && want[c][t] > 0.0).count();
        for (g, t) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert_eq!(got.group_positive[g * 2 + t] as usize, positive(g, t));
        }
    }
}
