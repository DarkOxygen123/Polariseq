#![allow(
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::unused_enumerate_index,
    clippy::too_many_arguments,
    clippy::redundant_clone,
    clippy::explicit_iter_loop,
    clippy::manual_range_contains,
    clippy::many_single_char_names
)]

//! UMAP (Uniform Manifold Approximation and Projection) — McInnes et al. 2018.
//!
//! Projects high-dimensional data to 2D for visualization. Operates on the kNN
//! graph we already compute (exact or NN-Descent).
//!
//! Algorithm:
//! 1. **Fuzzy simplicial set**: binary-search sigma per point so the
//!    k-neighborhood has fixed fuzzy mass. Symmetrize via fuzzy union — built
//!    as a symmetric CSR (transpose by counting sort, row-wise merge), no
//!    hash map, and built exactly once for both the init and the SGD.
//! 2. **Initialization**: spectral (2nd/3rd eigenvectors of the normalized
//!    graph Laplacian, subspace iteration with early stop) or random.
//! 3. **SGD**: cross-entropy loss. Attractive on graph edges, repulsive on
//!    negative samples. Weight is encoded in the edge sampling schedule —
//!    the gradient itself is unweighted (matching umap-learn exactly).
//!    Runs Hogwild-parallel over edge chunks (umap-learn's `parallel=True`
//!    model): shared embedding in atomic f32 storage, benign races, per-chunk
//!    RNG streams — thread-count independent but not run-to-run identical.
//!    `deterministic = true` selects the sequential loop, bit-reproducible.
//!
//! The fuzzy simplicial set, the curve parameters and the layout
//! optimisation follow umap-learn (BSD-3-Clause, Copyright (c) 2017 Leland
//! McInnes), whose notice `THIRD_PARTY_NOTICES.md` keeps.

use crate::backend::NeighborGraph;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Options for UMAP.
#[derive(Debug, Clone)]
pub struct UmapOptions {
    /// Output dimensionality (2 for standard UMAP).
    pub n_components: usize,
    /// Minimum distance between embedded points.
    pub min_dist: f32,
    /// Scale of embedded point spread.
    pub spread: f32,
    /// Number of SGD epochs. None = auto (500 if n ≤ 10K, else 200).
    pub n_epochs: Option<usize>,
    /// SGD learning rate.
    pub learning_rate: f32,
    /// Negative samples per positive edge.
    pub negative_sample_rate: usize,
    /// Repulsion strength multiplier.
    pub repulsion_strength: f32,
    /// Initialization method.
    pub init: UmapInit,
    /// Random seed.
    pub seed: u64,
    /// Run the sequential, bit-reproducible SGD instead of the parallel one.
    /// The parallel layout differs run to run by construction (Hogwild
    /// races), exactly as umap-learn's parallel mode does.
    pub deterministic: bool,
}

/// Initialization strategy for the low-dimensional embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UmapInit {
    /// Random uniform in [-10, 10].
    Random,
    /// Spectral: eigenvectors of the normalized graph Laplacian.
    Spectral,
    /// The coordinates already in the output buffer (the caller's start, for
    /// example principal components or a converged spectral layout).
    Given,
}

impl Default for UmapOptions {
    fn default() -> Self {
        Self {
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            init: UmapInit::Spectral,
            seed: 42,
            deterministic: false,
        }
    }
}

/// Symmetrised fuzzy simplicial set.
///
/// `indptr`/`indices`/`weights` are a symmetric CSR (both directions of every
/// edge stored, rows sorted by column). The SGD's edge set is its upper
/// triangle, read in row order.
pub(crate) struct FuzzyGraph {
    pub(crate) indptr: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    pub(crate) weights: Vec<f32>,
}

/// The symmetrised fuzzy-simplicial-set connectivities of a kNN distance
/// graph, scanpy's `obsp['connectivities']` analogue: a symmetric CSR with
/// both directions stored, rows sorted by column, no self-loops. Clustering
/// (Leiden / fast community / igraph comparisons) runs on this graph so
/// every method sees identical weights.
#[must_use]
pub fn fuzzy_connectivities(graph: &NeighborGraph) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    fuzzy_connectivities_from(&graph.indptr, &graph.indices, &graph.data, graph.n_obs)
}

/// [`fuzzy_connectivities`] from borrowed CSR arrays, so a caller holding
/// the distance graph elsewhere (a NumPy array) need not copy it. Builds only
/// the CSR, not the edge list UMAP's optimiser needs.
#[must_use]
pub fn fuzzy_connectivities_from(
    indptr: &[u32],
    indices: &[u32],
    data: &[f32],
    n: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let fuzzy = build_fuzzy(indptr, indices, data, n);
    (fuzzy.indptr, fuzzy.indices, fuzzy.weights)
}

/// Run UMAP on a kNN graph. Returns the embedding (n × n_components, row-major).
///
/// # Panics
/// Panics if the graph is empty.
#[must_use]
pub fn umap(graph: &NeighborGraph, opts: &UmapOptions) -> Vec<f32> {
    let mut out = vec![0.0_f32; graph.n_obs * opts.n_components];
    umap_into(
        &graph.indptr,
        &graph.indices,
        &graph.data,
        graph.n_obs,
        opts,
        &mut out,
    );
    out
}

/// [`umap`] on a kNN distance graph given as CSR slices (nothing is copied),
/// with the embedding written into `out` (`n × n_components`, row-major),
/// which may be a memory-mapped file.
///
/// # Panics
/// Panics if the graph is empty or `out` has the wrong length.
pub fn umap_into(
    indptr: &[u32],
    indices: &[u32],
    data: &[f32],
    n: usize,
    opts: &UmapOptions,
    out: &mut [f32],
) {
    assert!(n > 0, "empty graph");
    let trace = std::env::var_os("POLARISEQ_TRACE").is_some();
    let t0 = std::time::Instant::now();
    // Step 1: the fuzzy simplicial set, shared by init and SGD.
    let fuzzy = build_fuzzy(indptr, indices, data, n);
    if trace {
        eprintln!("[umap] fuzzy set: {:.2}s", t0.elapsed().as_secs_f64());
    }
    optimize(
        &fuzzy.indptr,
        &fuzzy.indices,
        &fuzzy.weights,
        n,
        opts,
        out,
        trace,
    );
}

/// [`umap_into`] on the fuzzy connectivities themselves (a symmetric CSR with
/// both directions stored, rows sorted by column, as `pp.neighbors` stores
/// them and [`fuzzy_connectivities_from`] builds them), so the graph the
/// neighbour step already wrote is used as it is: nothing is rebuilt, and no
/// copy of it is made. The result equals [`umap_into`] on the distances the
/// connectivities came from.
///
/// # Panics
/// Panics if the graph is empty or `out` has the wrong length.
pub fn umap_from_connectivities_into(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    n: usize,
    opts: &UmapOptions,
    out: &mut [f32],
) {
    assert!(n > 0, "empty graph");
    let trace = std::env::var_os("POLARISEQ_TRACE").is_some();
    optimize(indptr, indices, weights, n, opts, out, trace);
}

/// Steps 2 to 5 of UMAP on a fuzzy simplicial set given as a symmetric CSR.
fn optimize(
    f_indptr: &[u32],
    f_indices: &[u32],
    f_weights: &[f32],
    n: usize,
    opts: &UmapOptions,
    out: &mut [f32],
    trace: bool,
) {
    let d = opts.n_components;
    assert_eq!(out.len(), n * d, "the embedding must be n × n_components");

    // Step 2: Initialize the embedding.
    let t1 = std::time::Instant::now();
    let embedding = match opts.init {
        UmapInit::Random => random_init(n, d, opts.seed),
        UmapInit::Spectral => spectral_init(f_indptr, f_indices, f_weights, n, d, opts.seed),
        UmapInit::Given => out.to_vec(),
    };
    if trace {
        eprintln!(
            "[umap] init ({:?}): {:.2}s",
            opts.init,
            t1.elapsed().as_secs_f64()
        );
    }

    // Step 3: Compute the a, b curve parameters.
    let (a, b) = find_ab_params(opts.spread, opts.min_dist);

    // Step 4: Per-edge sampling schedules (the edge weight enters here, NOT
    // in the gradients).
    let n_epochs = opts.n_epochs.unwrap_or(if n <= 10_000 { 500 } else { 200 });
    let mut states = edge_states(f_indptr, f_indices, f_weights, opts.negative_sample_rate);
    let emb = Hogwild::from_f32(&embedding);
    drop(embedding);

    // Step 5: SGD optimization.
    let t2 = std::time::Instant::now();
    if opts.deterministic {
        sgd_sequential(&emb, &mut states, opts, n, d, a, b, n_epochs);
    } else {
        shuffle_edges(&mut states, opts.seed);
        sgd_parallel(&emb, &mut states, opts, n, d, a, b, n_epochs);
    }
    if trace {
        eprintln!(
            "[umap] sgd ({} epochs, {}): {:.2}s",
            n_epochs,
            if opts.deterministic {
                "sequential"
            } else {
                "parallel"
            },
            t2.elapsed().as_secs_f64()
        );
    }
    emb.write_f32(out);
}

// ── fuzzy simplicial set ──────────────────────────────────────────────────

/// Build the symmetrised fuzzy simplicial set from a kNN distance graph.
///
/// Membership strengths per directed edge come from [`smooth_knn_dist`]
/// (parallel over nodes). Symmetrisation is a counting-sort transpose plus a
/// row-wise sorted merge with the fuzzy union `w = a ⊕ b = a + b − ab`,
/// folded in the same encounter order the historical HashMap implementation
/// used, so weights match it bit for bit. Weights ≤ 1e-5 are dropped.
pub(crate) fn build_fuzzy_graph(graph: &NeighborGraph) -> FuzzyGraph {
    build_fuzzy(&graph.indptr, &graph.indices, &graph.data, graph.n_obs)
}

/// Split `out` into blocks of `rows_per_block` rows of the CSR `indptr`
/// (`(first row, last row + 1, the block's entries)`), for writing rows in
/// parallel into disjoint ranges.
fn split_rows_mut<'s, T>(
    indptr: &[u32],
    mut out: &'s mut [T],
    rows_per_block: usize,
) -> Vec<(usize, usize, &'s mut [T])> {
    let n = indptr.len() - 1;
    let mut blocks = Vec::with_capacity(n.div_ceil(rows_per_block));
    let mut r0 = 0;
    while r0 < n {
        let r1 = (r0 + rows_per_block).min(n);
        let len = (indptr[r1] - indptr[r0]) as usize;
        let (head, tail) = std::mem::take(&mut out).split_at_mut(len);
        blocks.push((r0, r1, head));
        out = tail;
        r0 = r1;
    }
    blocks
}

/// Rows per parallel block when writing the symmetrised graph.
const FUZZY_BLOCK_ROWS: usize = 4_096;

/// Reusable buffers of [`merge_fuzzy_row`], one set per worker thread.
#[derive(Default)]
struct MergeScratch {
    own: Vec<(u32, f32)>,
    transposed: Vec<(u32, f32)>,
    out: Vec<(u32, f32)>,
}

/// The first two phases of the fuzzy simplicial set of a kNN distance graph
/// (given as CSR slices): each edge's membership strength, and the transpose
/// of the strengths. The symmetrised graph is then merged row by row twice,
/// once to count each row ([`FuzzyPlan::row_lens`]) and once to write it into
/// the caller's arrays ([`FuzzyPlan::fill`]), so the graph, which can be
/// gigabytes, is never built anywhere but where it is kept (a memory-mapped
/// file, say) and no intermediate copy of it is held.
pub struct FuzzyPlan<'a> {
    g_indptr: &'a [u32],
    g_indices: &'a [u32],
    n: usize,
    /// Membership strengths, aligned with `g_indices`.
    strengths: Vec<f32>,
    t_indptr: Vec<u32>,
    t_cols: Vec<u32>,
    t_vals: Vec<f32>,
}

impl<'a> FuzzyPlan<'a> {
    /// Strengths and transpose of the distance graph `(indptr, indices,
    /// data)` of `n` points.
    #[must_use]
    pub fn new(g_indptr: &'a [u32], g_indices: &'a [u32], g_data: &[f32], n: usize) -> Self {
        // Phase 1: per-point membership strengths (parallel, order-preserving).
        let mut strengths = vec![0.0_f32; g_indices.len()];
        split_rows_mut(g_indptr, &mut strengths, FUZZY_BLOCK_ROWS)
            .into_par_iter()
            .for_each(|(r0, r1, block)| {
                let base = g_indptr[r0] as usize;
                for i in r0..r1 {
                    let (s, e) = (g_indptr[i] as usize, g_indptr[i + 1] as usize);
                    let row = smooth_knn_dist(&g_data[s..e]);
                    block[s - base..e - base].copy_from_slice(&row);
                }
            });

        // Phase 2: transpose Wᵀ by counting sort — row j of Wᵀ holds (i, w_ij)
        // for every directed edge i → j (same technique as CsrMatrix::from_csc).
        let nnz_dir = g_indices.len();
        let mut t_indptr = vec![0_u32; n + 1];
        for &j in g_indices {
            t_indptr[j as usize + 1] += 1;
        }
        for j in 0..n {
            t_indptr[j + 1] += t_indptr[j];
        }
        let mut t_cols = vec![0_u32; nnz_dir];
        let mut t_vals = vec![0.0_f32; nnz_dir];
        let mut cursor = t_indptr.clone();
        for i in 0..n {
            let (s, e) = (g_indptr[i] as usize, g_indptr[i + 1] as usize);
            for k in s..e {
                let j = g_indices[k] as usize;
                let c = cursor[j] as usize;
                t_cols[c] = i as u32;
                t_vals[c] = strengths[k];
                cursor[j] += 1;
            }
        }
        Self {
            g_indptr,
            g_indices,
            n,
            strengths,
            t_indptr,
            t_cols,
            t_vals,
        }
    }

    /// Merge row `i` of the symmetrised graph into `sc.out`.
    fn merge(&self, i: usize, sc: &mut MergeScratch) {
        merge_fuzzy_row(
            self.g_indptr,
            self.g_indices,
            &self.strengths,
            &self.t_indptr,
            &self.t_cols,
            &self.t_vals,
            i,
            sc,
        );
    }

    /// The number of entries in each row of the symmetrised graph (pruned
    /// weights excluded): `n` counts, whose running sum is the row pointers.
    #[must_use]
    pub fn row_lens(&self) -> Vec<u32> {
        (0..self.n)
            .into_par_iter()
            .map_init(MergeScratch::default, |sc, i| {
                self.merge(i, sc);
                sc.out.len() as u32
            })
            .collect()
    }

    /// Write the symmetrised graph into `indices_out` and `weights_out`, laid
    /// out by `indptr` (the running sum of [`FuzzyPlan::row_lens`]); rows are
    /// written in parallel into their own ranges.
    ///
    /// # Panics
    /// Panics if `indptr` is not `n + 1` long or the arrays are not
    /// `indptr[n]` long.
    pub fn fill(&self, indptr: &[u32], indices_out: &mut [u32], weights_out: &mut [f32]) {
        self.fill_by(indptr, indices_out, weights_out, FUZZY_BLOCK_ROWS);
    }

    /// [`FuzzyPlan::fill`] with blocks of `rows_per_block` rows.
    fn fill_by(
        &self,
        indptr: &[u32],
        indices_out: &mut [u32],
        weights_out: &mut [f32],
        rows_per_block: usize,
    ) {
        assert_eq!(indptr.len(), self.n + 1, "indptr must have n + 1 entries");
        let nnz = indptr[self.n] as usize;
        assert!(
            indices_out.len() == nnz && weights_out.len() == nnz,
            "outputs must hold indptr[n] entries"
        );
        let ix_blocks = split_rows_mut(indptr, indices_out, rows_per_block);
        let w_blocks = split_rows_mut(indptr, weights_out, rows_per_block);
        ix_blocks.into_par_iter().zip(w_blocks).for_each_init(
            MergeScratch::default,
            |sc, ((r0, r1, ix), (_, _, wt))| {
                let base = indptr[r0] as usize;
                for i in r0..r1 {
                    self.merge(i, sc);
                    let s = indptr[i] as usize - base;
                    debug_assert_eq!(sc.out.len(), (indptr[i + 1] - indptr[i]) as usize);
                    for (t, &(col, w)) in sc.out.iter().enumerate() {
                        ix[s + t] = col;
                        wt[s + t] = w;
                    }
                }
            },
        );
    }
}

/// The fuzzy simplicial set of a kNN distance graph given as CSR slices.
fn build_fuzzy(g_indptr: &[u32], g_indices: &[u32], g_data: &[f32], n: usize) -> FuzzyGraph {
    let plan = FuzzyPlan::new(g_indptr, g_indices, g_data, n);
    let lens = plan.row_lens();
    let mut indptr = vec![0_u32; n + 1];
    for (i, &l) in lens.iter().enumerate() {
        indptr[i + 1] = indptr[i] + l;
    }
    drop(lens);
    let nnz = indptr[n] as usize;
    let mut indices = vec![0_u32; nnz];
    let mut weights = vec![0.0_f32; nnz];
    plan.fill(&indptr, &mut indices, &mut weights);
    drop(plan);
    FuzzyGraph {
        indptr,
        indices,
        weights,
    }
}

/// Symmetrise one row into `sc.out`: merge the node's own neighborhood list
/// with the transposed entries other nodes point at it, folding equal columns
/// with the fuzzy union. Both lists are short (≤ k + duplicates), so sorting
/// them per row is cheap and the merge is linear.
#[allow(clippy::too_many_arguments)]
fn merge_fuzzy_row(
    g_indptr: &[u32],
    g_indices: &[u32],
    strengths: &[f32],
    t_indptr: &[u32],
    t_cols: &[u32],
    t_vals: &[f32],
    i: usize,
    sc: &mut MergeScratch,
) {
    let (s, e) = (g_indptr[i] as usize, g_indptr[i + 1] as usize);
    let w = &mut sc.own;
    w.clear();
    w.extend((s..e).map(|k| (g_indices[k], strengths[k])));
    w.sort_unstable_by_key(|&(c, _)| c);
    let (ts, te) = (t_indptr[i] as usize, t_indptr[i + 1] as usize);
    let t = &mut sc.transposed;
    t.clear();
    t.extend((ts..te).map(|k| (t_cols[k], t_vals[k])));
    t.sort_unstable_by_key(|&(c, _)| c);

    let out = &mut sc.out;
    out.clear();
    let (mut a, mut b) = (0, 0);
    while a < w.len() || b < t.len() {
        let col = if a < w.len() && (b >= t.len() || w[a].0 <= t[b].0) {
            w[a].0
        } else {
            t[b].0
        };
        // Fold every occurrence of `col` in both lists (own list first —
        // the encounter order of the HashMap version this replaced).
        let mut acc = 0.0_f32;
        while a < w.len() && w[a].0 == col {
            acc = acc + w[a].1 - acc * w[a].1;
            a += 1;
        }
        while b < t.len() && t[b].0 == col {
            acc = acc + t[b].1 - acc * t[b].1;
            b += 1;
        }
        if acc > 1e-5 {
            out.push((col, acc));
        }
    }
}

/// Membership strengths of one point's neighbours: umap-learn's
/// `smooth_knn_dist` + `compute_membership_strengths`, which scanpy's
/// `obsp['connectivities']` come from.
///
/// `dists` are the distances to the point's neighbours in ascending order,
/// self excluded (scanpy's `n_neighbors` counts the point itself, so
/// `k = dists.len() + 1`). With `rho` the nearest non-zero distance, the
/// bandwidth `sigma` solves `sum_j exp(-max(0, d_j - rho) / sigma) = log2(k)`
/// by umap-learn's bisection (64 steps, tolerance 1e-5), floored at 1e-3 of
/// the mean neighbour distance (umap-learn's `MIN_K_DIST_SCALE`, a mean that
/// includes the self distance of zero).
///
/// Until 2026-09-20 this searched in the wrong direction (a sum below the
/// target shrank sigma instead of growing it), so sigma ran to its upper
/// bound and every weight was 1: clustering and UMAP saw an unweighted kNN
/// graph. The target was also ln(k), not log2(k).
fn smooth_knn_dist(dists: &[f32]) -> Vec<f32> {
    const N_ITER: usize = 64;
    const TOLERANCE: f64 = 1e-5;
    const MIN_K_DIST_SCALE: f64 = 1e-3;

    if dists.is_empty() {
        return Vec::new();
    }
    let k_with_self = dists.len() + 1;
    let target = (k_with_self as f64).log2();
    let d64: Vec<f64> = dists.iter().map(|&d| f64::from(d)).collect();
    // local_connectivity = 1: rho is the nearest non-zero distance.
    let rho = d64.iter().copied().find(|&d| d > 0.0).unwrap_or(0.0);

    let (mut lo, mut hi, mut mid) = (0.0_f64, f64::INFINITY, 1.0_f64);
    for _ in 0..N_ITER {
        let psum: f64 = d64
            .iter()
            .map(|&d| {
                let x = d - rho;
                if x > 0.0 {
                    (-(x / mid)).exp()
                } else {
                    1.0
                }
            })
            .sum();
        if (psum - target).abs() < TOLERANCE {
            break;
        }
        if psum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            mid = if hi.is_infinite() {
                mid * 2.0
            } else {
                (lo + hi) / 2.0
            };
        }
    }
    let mean = d64.iter().sum::<f64>() / k_with_self as f64;
    let sigma = if rho > 0.0 {
        mid.max(MIN_K_DIST_SCALE * mean)
    } else {
        mid
    };

    d64.iter()
        .map(|&d| {
            let x = d - rho;
            if x <= 0.0 || sigma == 0.0 {
                1.0
            } else {
                (-(x / sigma)).exp() as f32
            }
        })
        .collect()
}

// ── initialization ────────────────────────────────────────────────────────

/// Spectral initialization: leading non-trivial eigenvectors of the normalized
/// adjacency `S = D^(-1/2) W D^(-1/2)` (equivalently the 2nd/3rd smallest
/// eigenvectors of the normalized Laplacian `I - S`).
#[allow(clippy::too_many_lines)]
///
/// Computed by randomized subspace iteration on the SPARSE S (CSR mat-vec
/// only, never forming an n×n matrix), followed by a q×q Rayleigh-Ritz
/// extraction. Cost is O(iterations × nnz × q) time and O(n × q) memory, so
/// the same code path serves every scale; umap-learn reaches the same
/// eigenvectors through scipy's sparse `eigsh`. Convergence is checked with
/// Ritz-value changes every few iterations instead of always running all 128.
fn spectral_init(
    f_indptr: &[u32],
    f_indices: &[u32],
    f_weights: &[f32],
    n: usize,
    d: usize,
    seed: u64,
) -> Vec<f32> {
    // Degrees of the symmetrized weight matrix W, then S = D^(-1/2) W D^(-1/2)
    // reusing the fuzzy CSR structure with scaled values.
    let nnz = f_indptr[n] as usize;
    let mut degrees = vec![0.0_f64; n];
    for i in 0..n {
        let (s, e) = (f_indptr[i] as usize, f_indptr[i + 1] as usize);
        for k in s..e {
            degrees[i] += f64::from(f_weights[k]);
        }
    }
    let mut s_vals = vec![0.0_f64; nnz];
    for i in 0..n {
        let (s, e) = (f_indptr[i] as usize, f_indptr[i + 1] as usize);
        let di = degrees[i].max(1e-12).sqrt();
        for k in s..e {
            let j = f_indices[k] as usize;
            let dj = degrees[j].max(1e-12).sqrt();
            s_vals[k] = f64::from(f_weights[k]) / (di * dj);
        }
    }

    let q = n.min(10); // subspace size (must exceed d + a few guard vectors)

    // Seeded random start block X (n × q), row-major flat.
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xbeef_cafeu64);
    let mut x: Vec<f64> = (0..n * q)
        .map(|_| rng.random::<f64>() * 2.0 - 1.0)
        .collect();
    let mut y = vec![0.0_f64; n * q];
    let n_chunks = n.div_ceil(GS_ROW_CHUNK);
    let mut partials = vec![0.0_f64; n_chunks * (q - 1).max(1)];

    // Subspace iteration: y = S·x, orthonormalize y into x. On real kNN
    // graphs the leading Ritz eigenvalues are nearly degenerate (measured on
    // heart_100k: λ2/λ3 gap ~5e-4 after 128 iterations), so eigenvalue
    // change never triggers quickly; convergence is judged on the quantity
    // actually used — the principal angle of the span of the two embedding
    // eigenvectors, which is init-equivalent up to a rotation.
    let check_every = 4;
    let min_checks = 2;
    let mut prev_cols: Option<Vec<f64>> = None;
    let mut checks = 0;
    let mut iters_done = 0;
    for iter in 0..128 {
        iters_done = iter + 1;
        spectral_matvec(f_indptr, f_indices, &s_vals, &x, &mut y, q);
        orthonormalize_columns(&mut y, &mut x, n, q, seed, &mut partials, n_chunks);
        checks += 1;
        if checks > min_checks && checks % check_every == 0 {
            spectral_matvec(f_indptr, f_indices, &s_vals, &x, &mut y, q);
            let Some(cols) = ritz_embedding_cols(&x, &y, n, q, d) else {
                continue;
            };
            if let Some(prev) = &prev_cols {
                // Cross-Gram of the two successive 2-D spans; smallest
                // singular value = cosine of the largest principal angle.
                let (mut g11, mut g12, mut g21, mut g22) = (0.0_f64, 0.0, 0.0, 0.0);
                for i in 0..n {
                    g11 += cols[i] * prev[i];
                    g12 += cols[i] * prev[n + i];
                    g21 += cols[n + i] * prev[i];
                    g22 += cols[n + i] * prev[n + i];
                }
                // 2×2 SVD closed form.
                let (s, dd, t, u) = (g11 + g22, g11 - g22, g12 + g21, g12 - g21);
                let (l1, l2) = ((s * s + u * u).sqrt(), (dd * dd + t * t).sqrt());
                let cos_max_angle = (l1 - l2).abs() * 0.5;
                if cos_max_angle > 0.999 {
                    break;
                }
            }
            prev_cols = Some(cols);
        }
    }
    if std::env::var_os("POLARISEQ_TRACE").is_some() {
        eprintln!("[umap] spectral: {iters_done} subspace iterations");
    }

    // Rayleigh-Ritz: B = X^T S X (q × q), eigendecompose, map back.
    spectral_matvec(f_indptr, f_indices, &s_vals, &x, &mut y, q);
    let mut bm = vec![0.0_f64; q * q];
    for a in 0..q {
        for c in 0..q {
            let mut dot = 0.0;
            for i in 0..n {
                dot += x[i * q + a] * y[i * q + c];
            }
            bm[a * q + c] = dot;
        }
    }
    let b_ref = faer::MatRef::from_row_major_slice(&bm, q, q);
    let ritz = b_ref
        .self_adjoint_eigen(faer::Side::Lower)
        .unwrap_or_else(|e| panic!("Rayleigh-Ritz eigendecomposition failed: {e:?}"));
    let u_ritz = ritz.U();
    // faer sorts ascending; the trivial (largest of S) eigenvector is column
    // q-1. Embedding dims 0 and 1 take columns q-2 and q-3.
    let mut embedding = vec![0.0_f32; n * d];
    let mut ok = true;
    for dim in 0..d.min(2) {
        let idx = q as isize - 2 - dim as isize;
        if idx < 0 {
            ok = false;
            break;
        }
        let mut comp = vec![0.0_f64; n];
        for i in 0..n {
            let mut v = 0.0;
            for m in 0..q {
                v += x[i * q + m] * u_ritz[(m, idx as usize)];
            }
            comp[i] = v;
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &comp {
            lo = lo.min(v as f32);
            hi = hi.max(v as f32);
        }
        let range = (hi - lo).max(1e-8);
        if !range.is_finite() || range > 1e30 {
            ok = false;
            break;
        }
        for i in 0..n {
            embedding[i * d + dim] = ((comp[i] as f32 - lo) / range) * 20.0 - 10.0;
        }
    }
    if !ok {
        return random_init(n, d, seed);
    }
    embedding
}

/// Sparse `out = S · x` for the CSR `(indptr, indices, vals)` and a row-major
/// n × q block `x`; parallel over rows (each row written by exactly one task,
/// so the result is deterministic for any thread count).
fn spectral_matvec(
    indptr: &[u32],
    indices: &[u32],
    vals: &[f64],
    x: &[f64],
    out: &mut [f64],
    q: usize,
) {
    out.par_chunks_mut(q).enumerate().for_each(|(i, row)| {
        row.fill(0.0);
        let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
        for k in s..e {
            let j = indices[k] as usize;
            let w = vals[k];
            let src = &x[j * q..j * q + q];
            for c in 0..q {
                row[c] += w * src[c];
            }
        }
    });
}

/// Orthonormalize the columns of `y` into `x` (both row-major n × q) with
/// classic Gram-Schmidt, parallel over fixed row chunks whose partials
/// merge in chunk order — deterministic for any thread count, the same
/// discipline as `rsvd::chunk_rows`. Degenerate columns (norm < 1e-12) are
/// re-randomized from a coordinate hash, deterministic as well.
const GS_ROW_CHUNK: usize = 8192;

fn orthonormalize_columns(
    y: &mut [f64],
    x: &mut [f64],
    n: usize,
    q: usize,
    seed: u64,
    partials: &mut [f64],
    n_chunks: usize,
) {
    let row_range = |ch: usize| (ch * GS_ROW_CHUNK)..((ch + 1) * GS_ROW_CHUNK).min(n);
    for c in 0..q {
        // dots[pc] = <y_c, x_pc> against the finalized columns pc < c.
        let mut dots = [0.0_f64; 10];
        if c > 0 {
            partials[..n_chunks * c]
                .par_chunks_mut(c)
                .enumerate()
                .for_each(|(ch, part)| {
                    part.fill(0.0);
                    for i in row_range(ch) {
                        let yc = y[i * q + c];
                        for (pc, p) in part.iter_mut().enumerate() {
                            *p += yc * x[i * q + pc];
                        }
                    }
                });
            for ch in 0..n_chunks {
                for pc in 0..c {
                    dots[pc] += partials[ch * c + pc];
                }
            }
            // Subtract the projections.
            y.par_chunks_mut(q).enumerate().for_each(|(i, row)| {
                let mut acc = row[c];
                for pc in 0..c {
                    acc -= dots[pc] * row_of(x, i, q)[pc];
                }
                row[c] = acc;
            });
        }

        // Norm via fixed-chunk partials merged in order.
        partials[..n_chunks]
            .par_chunks_mut(1)
            .enumerate()
            .for_each(|(ch, part)| {
                let mut s = 0.0;
                for i in row_range(ch) {
                    s += y[i * q + c] * y[i * q + c];
                }
                part[0] = s;
            });
        let norm = partials[..n_chunks].iter().sum::<f64>().sqrt();
        if norm < 1e-12 {
            // Degenerate direction: re-randomize (hash-keyed, deterministic).
            x.par_chunks_mut(q).enumerate().for_each(|(i, row)| {
                let h = mix64(seed ^ mix64(i as u64) ^ mix64(c as u64));
                row[c] = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
            });
        } else {
            x.par_chunks_mut(q).enumerate().for_each(|(i, row)| {
                row[c] = row_of(y, i, q)[c] / norm;
            });
        }
    }
}

#[inline]
fn row_of(v: &[f64], i: usize, q: usize) -> &[f64] {
    &v[i * q..(i + 1) * q]
}

/// The Ritz vectors the spectral init embeds with: the eigenvectors of the
/// q × q Rayleigh quotient B = Xᵀ(S·X) for the leading non-trivial
/// eigenvalues (faer ascending → columns q-2 and q-3 of U), mapped back to
/// n-vectors. Returned as a flat 2n buffer (unit norm). `None` if the tiny
/// eigendecomposition fails.
fn ritz_embedding_cols(x: &[f64], y: &[f64], n: usize, q: usize, d: usize) -> Option<Vec<f64>> {
    let mut bm = vec![0.0_f64; q * q];
    for a in 0..q {
        for c in 0..q {
            let mut dot = 0.0;
            let mut dot_t = 0.0;
            for i in 0..n {
                dot += x[i * q + a] * y[i * q + c];
                dot_t += x[i * q + c] * y[i * q + a];
            }
            bm[a * q + c] = 0.5 * (dot + dot_t);
        }
    }
    let b_ref = faer::MatRef::from_row_major_slice(&bm, q, q);
    let evd = b_ref.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let u = evd.U();
    let n_dims = d.min(2);
    let mut cols = vec![0.0_f64; 2 * n];
    for dim in 0..n_dims {
        let idx = q as isize - 2 - dim as isize;
        if idx < 0 {
            return None;
        }
        let mut norm = 0.0_f64;
        for i in 0..n {
            let mut v = 0.0;
            for m in 0..q {
                v += x[i * q + m] * u[(m, idx as usize)];
            }
            cols[dim * n + i] = v;
            norm += v * v;
        }
        // Normalize so the cross-Gram singular values are principal-angle
        // cosines regardless of eigenvalue magnitude.
        let norm = norm.max(1e-300).sqrt();
        for i in 0..n {
            cols[dim * n + i] /= norm;
        }
    }
    Some(cols)
}

/// The spectral start alone (2-D, each axis scaled to [-10, 10]) for a fuzzy
/// simplicial set given as a symmetric CSR, so that it can be inspected or
/// rescaled before the optimization.
#[must_use]
pub fn spectral_layout(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    n: usize,
    seed: u64,
) -> Vec<f32> {
    spectral_init(indptr, indices, weights, n, 2, seed)
}

/// k-means of the cells' coordinates (`data`, n x d, typically principal
/// components) into `k` groups, as Scarf's UMAP start uses it (Dhapola et al.
/// 2022; Scarf is BSD-3-Clause, see THIRD_PARTY_NOTICES.md; `_get_ini_embed` in
/// `scarf/datastore/graph_datastore.py`). Returns each cell's group and the
/// centroids (k x d, row-major).
///
/// The centroids are found on a sample of at most `SAMPLE` cells (k-means++
/// seeding, then Lloyd iterations), as Scarf's mini-batch k-means does, and every
/// cell is then assigned to its nearest centroid, in parallel. Deterministic for a
/// given seed and any thread count.
///
/// # Panics
/// Panics if `data` is not `n x d`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn kmeans_labels(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    seed: u64,
) -> (Vec<u32>, Vec<f32>) {
    const SAMPLE: usize = 100_000;
    const LLOYD: usize = 10;
    assert_eq!(data.len(), n * d, "data must be n x d");
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let k = k.clamp(1, n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5ca7_f00d);
    // the sample: every cell when there are few, else a seeded draw without replacement
    let sample: Vec<usize> = if n <= SAMPLE {
        (0..n).collect()
    } else {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..SAMPLE {
            let j = rng.random_range(i..n);
            idx.swap(i, j);
        }
        let mut s = idx[..SAMPLE].to_vec();
        s.sort_unstable();
        s
    };
    let m = sample.len();
    let row = |i: usize| &data[i * d..(i + 1) * d];
    let sq =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    // k-means++ seeding on the sample
    let mut centroids: Vec<f32> = Vec::with_capacity(k * d);
    centroids.extend_from_slice(row(sample[rng.random_range(0..m)]));
    let mut nearest: Vec<f32> = sample
        .par_iter()
        .map(|&i| sq(row(i), &centroids[0..d]))
        .collect();
    for c in 1..k {
        let total: f64 = nearest.iter().map(|&v| f64::from(v)).sum();
        let pick = if total > 0.0 {
            let r = rng.random::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = m - 1;
            for (j, &v) in nearest.iter().enumerate() {
                cum += f64::from(v);
                if cum >= r {
                    chosen = j;
                    break;
                }
            }
            chosen
        } else {
            rng.random_range(0..m)
        };
        centroids.extend_from_slice(row(sample[pick]));
        let new_c = &centroids[c * d..(c + 1) * d];
        nearest
            .par_iter_mut()
            .zip(sample.par_iter())
            .for_each(|(v, &i)| {
                let dd = sq(row(i), new_c);
                if dd < *v {
                    *v = dd;
                }
            });
    }

    // nearest centroid by ||x||^2 - 2 x.c + ||c||^2 (the ||x||^2 term is common)
    let assign = |cells: &[usize], cent: &[f32]| -> Vec<u32> {
        let cnorm: Vec<f32> = cent
            .chunks_exact(d)
            .map(|c| c.iter().map(|v| v * v).sum())
            .collect();
        cells
            .par_iter()
            .map(|&i| {
                let x = row(i);
                let mut best = (f32::INFINITY, 0_u32);
                for (c, cv) in cent.chunks_exact(d).enumerate() {
                    let dot: f32 = x.iter().zip(cv).map(|(a, b)| a * b).sum();
                    let v = cnorm[c] - 2.0 * dot;
                    if v < best.0 {
                        best = (v, c as u32);
                    }
                }
                best.1
            })
            .collect()
    };
    for _ in 0..LLOYD {
        let lab = assign(&sample, &centroids);
        let mut sums = vec![0.0_f64; k * d];
        let mut counts = vec![0_usize; k];
        for (&i, &c) in sample.iter().zip(&lab) {
            let c = c as usize;
            counts[c] += 1;
            for (s, &v) in sums[c * d..(c + 1) * d].iter_mut().zip(row(i)) {
                *s += f64::from(v);
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c * d + j] = (sums[c * d + j] / counts[c] as f64) as f32;
                }
            }
        }
    }
    let all: Vec<usize> = (0..n).collect();
    let labels = assign(&all, &centroids);
    (labels, centroids)
}

/// Scarf's UMAP start from cell coordinates; see [`kmeans_labels`] for the
/// clustering. The centroids are laid out by their own PCA, each axis clipped
/// to the 10th-90th percentiles of a normal centred on its median (Scarf's
/// `rescale_array`), and every cell is placed at its centroid. Returns n x 2.
///
/// # Panics
/// Panics if `data` is not `n x d`, or if the centroids' eigendecomposition fails.
#[must_use]
pub fn kmeans_start(data: &[f32], n: usize, d: usize, k: usize, seed: u64) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let (labels, centroids) = kmeans_labels(data, n, d, k, seed);
    let k = centroids.len() / d;

    // PCA of the centroids to two axes
    let mean: Vec<f64> = (0..d)
        .map(|j| (0..k).map(|c| f64::from(centroids[c * d + j])).sum::<f64>() / k as f64)
        .collect();
    let centred: Vec<f64> = (0..k * d)
        .map(|t| f64::from(centroids[t]) - mean[t % d])
        .collect();
    let mut cov = vec![0.0_f64; d * d];
    for c in 0..k {
        let r = &centred[c * d..(c + 1) * d];
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] += r[a] * r[b];
            }
        }
    }
    let eig = faer::MatRef::from_row_major_slice(&cov, d, d)
        .self_adjoint_eigen(faer::Side::Lower)
        .unwrap_or_else(|e| panic!("eigendecomposition of the centroid covariance failed: {e:?}"));
    let u = eig.U();
    let mut pos = vec![0.0_f64; k * 2];
    for axis in 0..2.min(d) {
        let col = d - 1 - axis; // faer sorts ascending
        for c in 0..k {
            pos[c * 2 + axis] = (0..d).map(|j| centred[c * d + j] * u[(j, col)]).sum();
        }
        // Scarf's rescale_array: clip to loc +- z(0.9) sd, loc the median
        let mut v: Vec<f64> = (0..k).map(|c| pos[c * 2 + axis]).collect();
        let sd = {
            let mu = v.iter().sum::<f64>() / k as f64;
            (v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / k as f64).sqrt()
        };
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let loc = if k % 2 == 1 {
            v[k / 2]
        } else {
            0.5 * (v[k / 2 - 1] + v[k / 2])
        };
        let z = 1.281_551_565_544_600_5; // standard normal 0.9 quantile
        for c in 0..k {
            let x = &mut pos[c * 2 + axis];
            *x = x.clamp(loc - z * sd, loc + z * sd);
        }
    }
    let mut out = vec![0.0_f32; n * 2];
    for (i, &c) in labels.iter().enumerate() {
        out[i * 2] = pos[c as usize * 2] as f32;
        out[i * 2 + 1] = pos[c as usize * 2 + 1] as f32;
    }
    out
}

/// Multilevel spectral start (Polariseq's own): the spectral layout of a graph
/// of millions of cells computed through a graph of their groups.
///
/// 1. The fuzzy graph's weights are summed between the `k` groups given by
///    `labels` (k-means groups or clusters): a k x k graph.
/// 2. Its normalized adjacency `D^-1/2 W D^-1/2` is decomposed exactly; the two
///    leading non-trivial eigenvectors, mapped to random-walk form, give each
///    group a position.
/// 3. Every cell takes its group's position, scaled by the square root of its
///    own degree (the change of variables between the two forms), which is a
///    start for the cell graph's own eigenvectors, and `smooth` iterations of
///    `(I + S) / 2` on the cell graph refine it (the shift keeps every
///    eigenvalue non-negative, so no spurious negative mode is amplified).
///
/// Subspace iteration on the cell graph from a random start (`spectral_init`)
/// needs many iterations because its leading eigenvalues nearly coincide; from
/// the coarse solution, which already has the right large-scale shape, a few
/// suffice. Returns n x 2, each axis scaled to [-10, 10] as `spectral_init`.
///
/// # Panics
/// Panics if `labels` does not hold one group in `0..k` per cell.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn multilevel_layout(
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
    n: usize,
    labels: &[u32],
    k: usize,
    smooth: usize,
) -> Vec<f32> {
    assert_eq!(labels.len(), n, "one group per cell");
    assert!(
        labels.iter().all(|&g| (g as usize) < k),
        "groups must be in 0..k"
    );
    if n == 0 {
        return Vec::new();
    }
    // 1. the group graph, folded over row chunks in order (deterministic)
    let chunk = n.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    let parts: Vec<Vec<f64>> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|c| {
            let mut w = vec![0.0_f64; k * k];
            for i in c * chunk..((c + 1) * chunk).min(n) {
                let gi = labels[i] as usize;
                for e in indptr[i] as usize..indptr[i + 1] as usize {
                    let gj = labels[indices[e] as usize] as usize;
                    w[gi * k + gj] += f64::from(weights[e]);
                }
            }
            w
        })
        .collect();
    let mut wc = vec![0.0_f64; k * k];
    for p in &parts {
        for (a, b) in wc.iter_mut().zip(p) {
            *a += b;
        }
    }
    drop(parts);
    // 2. exact eigenvectors of the group graph's normalized adjacency
    let dc: Vec<f64> = (0..k)
        .map(|a| wc[a * k..(a + 1) * k].iter().sum::<f64>())
        .collect();
    let mut sc = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let den = (dc[a] * dc[b]).sqrt();
            sc[a * k + b] = if den > 0.0 { wc[a * k + b] / den } else { 0.0 };
        }
    }
    let eig = faer::MatRef::from_row_major_slice(&sc, k, k)
        .self_adjoint_eigen(faer::Side::Lower)
        .unwrap_or_else(|e| panic!("eigendecomposition of the group graph failed: {e:?}"));
    let u = eig.U();
    let mut group_pos = vec![[0.0_f64; 2]; k];
    for (axis, col) in [(0_usize, k.saturating_sub(2)), (1, k.saturating_sub(3))] {
        if k < 3 {
            break;
        }
        for a in 0..k {
            group_pos[a][axis] = if dc[a] > 0.0 {
                u[(a, col)] / dc[a].sqrt()
            } else {
                0.0
            };
        }
    }
    // 3. lift to the cells and smooth with (I + S) / 2 on the cell graph
    let deg: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            (indptr[i] as usize..indptr[i + 1] as usize)
                .map(|e| f64::from(weights[e]))
                .sum()
        })
        .collect();
    let sq: Vec<f64> = deg.iter().map(|&d| d.max(1e-12).sqrt()).collect();
    let tnorm = sq.iter().map(|v| v * v).sum::<f64>().sqrt();
    let triv: Vec<f64> = sq.iter().map(|v| v / tnorm).collect();
    let mut x: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let g = group_pos[labels[i] as usize];
            [g[0] * sq[i], g[1] * sq[i]]
        })
        .collect();
    let orth = |x: &mut Vec<[f64; 2]>| {
        // remove the trivial direction, then Gram-Schmidt the two axes
        for axis in 0..2 {
            let p: f64 = x.par_iter().zip(&triv).map(|(v, t)| v[axis] * t).sum();
            x.par_iter_mut()
                .zip(&triv)
                .for_each(|(v, t)| v[axis] -= p * t);
        }
        let n0: f64 = x
            .par_iter()
            .map(|v| v[0] * v[0])
            .sum::<f64>()
            .sqrt()
            .max(1e-300);
        x.par_iter_mut().for_each(|v| v[0] /= n0);
        let p: f64 = x.par_iter().map(|v| v[0] * v[1]).sum();
        x.par_iter_mut().for_each(|v| v[1] -= p * v[0]);
        let n1: f64 = x
            .par_iter()
            .map(|v| v[1] * v[1])
            .sum::<f64>()
            .sqrt()
            .max(1e-300);
        x.par_iter_mut().for_each(|v| v[1] /= n1);
    };
    orth(&mut x);
    for _ in 0..smooth {
        let y: Vec<[f64; 2]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut acc = [0.0_f64; 2];
                for e in indptr[i] as usize..indptr[i + 1] as usize {
                    let j = indices[e] as usize;
                    let s = f64::from(weights[e]) / (sq[i] * sq[j]);
                    acc[0] += s * x[j][0];
                    acc[1] += s * x[j][1];
                }
                [0.5 * (x[i][0] + acc[0]), 0.5 * (x[i][1] + acc[1])]
            })
            .collect();
        x = y;
        orth(&mut x);
    }
    // each axis to [-10, 10], as spectral_init scales its output
    let mut out = vec![0.0_f32; n * 2];
    for axis in 0..2 {
        let (lo, hi) = x
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
                (l.min(v[axis]), h.max(v[axis]))
            });
        let range = (hi - lo).max(1e-12);
        for i in 0..n {
            out[i * 2 + axis] = (((x[i][axis] - lo) / range) * 20.0 - 10.0) as f32;
        }
    }
    out
}

/// Random initialization: uniform in [-10, 10].
fn random_init(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * d)
        .map(|_| rng.random::<f32>() * 20.0 - 10.0)
        .collect()
}

// ── SGD ───────────────────────────────────────────────────────────────────

/// Per-edge SGD state: endpoints, sampling schedules, and the running count
/// of negative-sample draws (the draw index keys the per-edge hash stream).
/// Every edge lives in exactly one parallel chunk, so this state never
/// races; only the embedding coordinates are shared (Hogwild).
#[derive(Debug, Clone, Copy)]
struct EdgeState {
    src: u32,
    dst: u32,
    epochs_per_sample: f32,
    epochs_per_negative: f32,
    next_sample: f32,
    next_negative: f32,
    draws_done: u64,
}

/// Build per-edge states from the fuzzy edge list. Higher-weight edges are
/// sampled more often (umap-learn's `compute_epochs_per_sample`); this is
/// where the edge weight enters the algorithm.
fn edge_states(
    f_indptr: &[u32],
    f_indices: &[u32],
    f_weights: &[f32],
    negative_sample_rate: usize,
) -> Vec<EdgeState> {
    // The canonical undirected edge list: each edge once, from its lower
    // endpoint, in row order.
    let upper = |i: usize| {
        (f_indptr[i] as usize..f_indptr[i + 1] as usize).filter(move |&k| f_indices[k] as usize > i)
    };
    let n = f_indptr.len() - 1;
    let max_weight = (0..n)
        .flat_map(|i| upper(i).map(|k| f_weights[k]))
        .fold(0.0_f32, f32::max);
    let n_edges: usize = (0..n).map(|i| upper(i).count()).sum();
    let mut states = Vec::with_capacity(n_edges);
    for i in 0..n {
        for k in upper(i) {
            let w = f_weights[k];
            let eps = if max_weight == 0.0 {
                1.0
            } else {
                max_weight / w.max(1e-8)
            };
            let epn = eps / negative_sample_rate as f32;
            states.push(EdgeState {
                src: i as u32,
                dst: f_indices[k],
                epochs_per_sample: eps,
                epochs_per_negative: epn,
                next_sample: eps,
                next_negative: epn,
                draws_done: 0,
            });
        }
    }
    states
}

/// Hogwild embedding storage: f32 coordinates packed two per relaxed
/// `AtomicU64` word, so parallel edge tasks update the shared layout
/// without locks. For d = 2 (the standard case) a point's coordinates load
/// and store in a single atomic access — and a reader always sees a
/// consistent (x, y) pair, never a half-updated point. Concurrent lost
/// updates are accepted SGD noise (umap-learn's `parallel=True` does the
/// same); the sequential driver runs identical arithmetic single-threaded,
/// so `deterministic = true` is bit-reproducible.
struct Hogwild {
    words: Vec<AtomicU64>,
}

impl Hogwild {
    fn from_f32(v: &[f32]) -> Self {
        let mut words = Vec::with_capacity(v.len().div_ceil(2));
        for pair in v.chunks(2) {
            let lo = u64::from(pair[0].to_bits());
            let hi = pair.get(1).map_or(0, |&f| u64::from(f.to_bits()));
            words.push(AtomicU64::new(lo | (hi << 32)));
        }
        Self { words }
    }

    /// Write the coordinates into `out`, which holds exactly the `len`
    /// values the embedding has.
    fn write_f32(&self, out: &mut [f32]) {
        let len = out.len();
        let mut at = 0;
        for w in &self.words {
            let bits = w.load(Ordering::Relaxed);
            out[at] = f32::from_bits(bits as u32);
            at += 1;
            if at < len {
                out[at] = f32::from_bits((bits >> 32) as u32);
                at += 1;
            }
        }
    }

    /// Per-coordinate read (generic-d path).
    #[inline]
    fn get(&self, i: usize) -> f32 {
        let w = self.words[i / 2].load(Ordering::Relaxed);
        let bits = if i % 2 == 0 {
            w as u32
        } else {
            (w >> 32) as u32
        };
        f32::from_bits(bits)
    }

    /// Per-coordinate read-modify-write (generic-d path).
    #[inline]
    fn set(&self, i: usize, v: f32) {
        let w = &self.words[i / 2];
        let mut cur = w.load(Ordering::Relaxed);
        loop {
            let nb = if i % 2 == 0 {
                (cur & 0xffff_ffff_0000_0000) | u64::from(v.to_bits())
            } else {
                (cur & 0x0000_0000_ffff_ffff) | (u64::from(v.to_bits()) << 32)
            };
            match w.compare_exchange_weak(cur, nb, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Whole-point read for d = 2.
    #[inline]
    fn get2(&self, point: usize) -> (f32, f32) {
        let bits = self.words[point].load(Ordering::Relaxed);
        (
            f32::from_bits(bits as u32),
            f32::from_bits((bits >> 32) as u32),
        )
    }

    /// Whole-point write for d = 2.
    #[inline]
    fn set2(&self, point: usize, x: f32, y: f32) {
        let packed = u64::from(x.to_bits()) | (u64::from(y.to_bits()) << 32);
        self.words[point].store(packed, Ordering::Relaxed);
    }
}

/// Fit the a, b parameters for the low-dim similarity curve.
///
/// Target: y = 1/(1 + a·x^(2b)) fit to the piecewise curve
///   y = 1 for x < min_dist, exp(-(x-min_dist)/spread) for x ≥ min_dist.
fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    let xv: Vec<f32> = (0..300)
        .map(|i| i as f32 * 3.0 * spread / 299.0 + 1e-4)
        .collect();
    let yv: Vec<f32> = xv
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    let mut sum_xx = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut count = 0.0_f64;

    for (&x, &y) in xv.iter().zip(&yv) {
        let y_clamped = y.clamp(1e-8, 1.0 - 1e-8);
        let logit_y = f64::from(((1.0 / y_clamped) - 1.0).ln());
        let log_x = f64::from(x.ln());
        if y < 0.999 {
            sum_xx += log_x * log_x;
            sum_xy += log_x * logit_y;
            sum_x += log_x;
            sum_y += logit_y;
            count += 1.0;
        }
    }

    let denom = count * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return (1.577, 0.895); // defaults for min_dist=0.1, spread=1.0
    }
    let slope = (count * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / count;

    let b = (slope / 2.0) as f32;
    let a = intercept.exp() as f32;
    (a.max(1e-4), b.max(0.1))
}

/// Clip gradients to prevent explosion (umap-learn uses ±4).
#[inline]
fn clip_grad(g: f32) -> f32 {
    g.clamp(-4.0, 4.0)
}

/// Fast approximate `powf` for the SGD gradients: `exp2(e·log2(x))` with
/// degree-6 minimax-style polynomials (least-squares fits; max relative
/// error ~4e-6 over the gradient argument range). The error is three
/// orders of magnitude below the negative-sampling noise, and the gradient
/// clips at ±4 anyway; libm `powf` was ~40% of the SGD inner loop.
#[inline]
fn fast_pow(x: f32, e: f32) -> f32 {
    // log2 via exponent/mantissa split: x = m·2^E, m ∈ [1,2).
    // Clamp keeps the bit tricks in the normal range; dist_sq values below
    // 1e-30 cannot arise from f32 coordinates and would clip to zero force.
    let x = x.clamp(1e-30, f32::MAX);
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let s = m - 1.0;
    // log2(1+s), Horner: s·(s·(s·(s·(s·(s·c6 + c5) + c4) + c3) + c2) + c1) + c0
    #[allow(clippy::excessive_precision)]
    let log2_m = (((((-0.026_135_083_f32 * s + 0.121_804_96_f32) * s - 0.276_745_64_f32) * s
        + 0.456_173_43_f32)
        * s
        - 0.717_558_2_f32)
        * s
        + 1.442_451_6_f32)
        * s
        + 0.000_003_762_f32;
    let log2_x = log2_m + exp as f32;

    let t = e * log2_x;
    if t < -126.0 {
        return 0.0;
    }
    if t > 126.0 {
        return f32::INFINITY;
    }
    // exp2: split t = i + f; 2^t = 2^f · 2^i, with 2^i from the exponent bits.
    let fi = t.floor();
    let f = t - fi;
    #[allow(clippy::excessive_precision)]
    let p = ((((((0.000_216_834_4_f32 * f + 0.001_243_793_6_f32) * f + 0.009_679_988_f32) * f
        + 0.055_482_148_f32)
        * f
        + 0.240_230_31_f32)
        * f
        + 0.693_146_9_f32)
        * f
        + 1.0)
        * f32::from_bits(((fi as i32 + 127) as u32) << 23);
    p
}

/// Deterministic negative-sample draw. The per-edge key is precomputed once
/// per firing ([`edge_key`]); each draw costs one mix64 and a Lemire
/// multiply-shift range reduction. Reproducible per (seed, edge, draw) —
/// identical in the sequential and parallel drivers and independent of
/// thread count — and ~5× cheaper per draw than a ChaCha8 stream, which
/// mattered at 1.2 G draws per 100k-cell run. Same technique as nndescent's
/// candidate sampling.
#[inline]
fn edge_key(seed: u64, src: u32, dst: u32) -> u64 {
    let pair = (u64::from(src) << 32) ^ u64::from(dst);
    mix64(seed ^ mix64(pair))
}

#[inline]
fn negative_sample(key: u64, draw: u64, n: usize) -> usize {
    let h = mix64(key ^ draw.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ((u128::from(h) * u128::from(n as u64)) >> 64) as usize
}

/// One SGD edge update — the single implementation of the UMAP gradient
/// maths, shared by the sequential and parallel drivers.
///
/// Matches umap-learn's `optimize_layout_euclidean`: the attractive gradient
/// carries no edge weight (the sampling schedule encodes it), repulsion is
/// unconditionally `4·γ`-scaled and moves only the source, gradients clip to
/// ±4, and negative samples never pick the source itself.
#[allow(clippy::too_many_arguments)]
fn apply_edge(
    emb: &Hogwild,
    st: &mut EdgeState,
    epoch: f32,
    n: usize,
    d: usize,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    seed: u64,
) {
    if epoch < st.next_sample {
        return;
    }
    if d == 2 {
        apply_edge_2(emb, st, epoch, n, a, b, gamma, alpha, seed);
    } else {
        apply_edge_n(emb, st, epoch, n, d, a, b, gamma, alpha, seed);
    }
}

/// d = 2 fast path: whole points in single atomic accesses, source
/// coordinates carried in registers across the negative samples,
/// `fast_pow` for both curve evaluations, one hash key per edge firing.
/// The sequential driver runs the same arithmetic as the generic path
/// (single thread ⇒ register and memory values coincide).
#[allow(clippy::too_many_arguments)]
fn apply_edge_2(
    emb: &Hogwild,
    st: &mut EdgeState,
    epoch: f32,
    n: usize,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    seed: u64,
) {
    let si = st.src as usize;
    let ti = st.dst as usize;

    // --- Attractive force (positive edge); weight is in the schedule. ---
    let (mut sx, mut sy) = emb.get2(si);
    let (tx, ty) = emb.get2(ti);
    let (dx, dy) = (sx - tx, sy - ty);
    let dist_sq = dx * dx + dy * dy;
    if dist_sq > 0.0 {
        // grad_coeff = -2ab·(d²)^(b-1) / (a·(d²)^b + 1); one pow, the
        // second power derived from the first.
        let p = fast_pow(dist_sq, b - 1.0);
        let q = p * dist_sq;
        let grad_coeff = -2.0 * a * b * p / (a * q + 1.0);
        let gx = clip_grad(grad_coeff * dx);
        let gy = clip_grad(grad_coeff * dy);
        sx += alpha * gx;
        sy += alpha * gy;
        emb.set2(si, sx, sy);
        let (tx, ty) = emb.get2(ti);
        emb.set2(ti, tx - alpha * gx, ty - alpha * gy);
    }

    // --- Repulsive forces (negative samples). ---
    let key = edge_key(seed, st.src, st.dst);
    let epn = st.epochs_per_negative;
    while epoch >= st.next_negative {
        let neg = negative_sample(key, st.draws_done, n);
        st.draws_done += 1;
        if neg == si {
            st.next_negative += epn;
            continue;
        }
        let (nx, ny) = emb.get2(neg);
        let (dx, dy) = (sx - nx, sy - ny);
        let neg_dist_sq = dx * dx + dy * dy;
        if neg_dist_sq > 0.0 {
            // grad_coeff = 2bγ / ((0.001 + d²)(a·(d²)^b + 1))
            let q = fast_pow(neg_dist_sq, b);
            let grad_coeff = 2.0 * b * gamma / ((0.001 + neg_dist_sq) * (a * q + 1.0));
            // Only the SOURCE moves (umap-learn behavior).
            sx += alpha * clip_grad(4.0 * grad_coeff * dx);
            sy += alpha * clip_grad(4.0 * grad_coeff * dy);
        }
        st.next_negative += epn;
    }
    emb.set2(si, sx, sy);

    st.next_sample += st.epochs_per_sample;
}

/// Generic-dimension path (n_components > 2).
#[allow(clippy::too_many_arguments)]
fn apply_edge_n(
    emb: &Hogwild,
    st: &mut EdgeState,
    epoch: f32,
    n: usize,
    d: usize,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    seed: u64,
) {
    let si = st.src as usize;
    let ti = st.dst as usize;

    // --- Attractive force (positive edge) ---
    let mut dist_sq = 0.0_f32;
    for k in 0..d {
        let diff = emb.get(si * d + k) - emb.get(ti * d + k);
        dist_sq += diff * diff;
    }

    if dist_sq > 0.0 {
        let p = fast_pow(dist_sq, b - 1.0);
        let q = p * dist_sq;
        let grad_coeff = -2.0 * a * b * p / (a * q + 1.0);

        for k in 0..d {
            let diff = emb.get(si * d + k) - emb.get(ti * d + k);
            let grad = clip_grad(grad_coeff * diff);
            emb.set(si * d + k, emb.get(si * d + k) + alpha * grad);
            emb.set(ti * d + k, emb.get(ti * d + k) - alpha * grad);
        }
    }

    // --- Repulsive forces (negative samples) ---
    let key = edge_key(seed, st.src, st.dst);
    let epn = st.epochs_per_negative;
    while epoch >= st.next_negative {
        let neg_target = negative_sample(key, st.draws_done, n);
        st.draws_done += 1;
        if neg_target == si {
            st.next_negative += epn;
            continue;
        }

        let mut neg_dist_sq = 0.0_f32;
        for k in 0..d {
            let diff = emb.get(si * d + k) - emb.get(neg_target * d + k);
            neg_dist_sq += diff * diff;
        }

        if neg_dist_sq > 0.0 {
            let grad_coeff =
                2.0 * b * gamma / ((0.001 + neg_dist_sq) * (a * fast_pow(neg_dist_sq, b) + 1.0));

            for k in 0..d {
                // Only the SOURCE moves (umap-learn behavior).
                let diff = emb.get(si * d + k) - emb.get(neg_target * d + k);
                let grad = clip_grad(4.0 * grad_coeff * diff);
                emb.set(si * d + k, emb.get(si * d + k) + alpha * grad);
            }
        }

        st.next_negative += epn;
    }

    st.next_sample += st.epochs_per_sample;
}

/// Learning-rate schedule: linear decay with umap-learn's floor.
#[inline]
fn alpha_for_epoch(learning_rate: f32, epoch: usize, n_epochs: usize) -> f32 {
    learning_rate * (1.0 - epoch as f32 / n_epochs as f32).max(0.01)
}

/// Sequential (single-threaded) SGD — used when `deterministic` is set.
/// Bit-reproducible: fixed edge order, hash-keyed negative samples.
fn sgd_sequential(
    emb: &Hogwild,
    states: &mut [EdgeState],
    opts: &UmapOptions,
    n: usize,
    d: usize,
    a: f32,
    b: f32,
    n_epochs: usize,
) {
    let gamma = opts.repulsion_strength;
    for epoch in 1..=n_epochs {
        let alpha = alpha_for_epoch(opts.learning_rate, epoch, n_epochs);
        for st in states.iter_mut() {
            apply_edge(emb, st, epoch as f32, n, d, a, b, gamma, alpha, opts.seed);
        }
    }
}

/// Edge-block size for the parallel SGD. Small enough that rayon's work
/// stealing keeps the fast cores busy while the efficiency cores grind
/// (with 4096-edge chunks every epoch ended up gated on the slowest core's
/// last chunk — measured 8 threads = 4 threads); large enough that per-task
/// overhead is noise.
const SGD_CHUNK: usize = 1024;

/// splitmix64 finalizer (same mixer nndescent.rs uses, kept local — the
/// modules share no RNG plumbing on purpose).
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Hogwild-parallel SGD: edge chunks across threads, shared embedding in
/// atomic storage. Benign races on embedding coordinates; per-edge state
/// (schedules, draw counters) is chunk-owned and race-free.
fn sgd_parallel(
    emb: &Hogwild,
    states: &mut [EdgeState],
    opts: &UmapOptions,
    n: usize,
    d: usize,
    a: f32,
    b: f32,
    n_epochs: usize,
) {
    let gamma = opts.repulsion_strength;
    for epoch in 1..=n_epochs {
        let alpha = alpha_for_epoch(opts.learning_rate, epoch, n_epochs);
        let epoch_f = epoch as f32;
        states.par_chunks_mut(SGD_CHUNK).for_each(|chunk| {
            for st in chunk {
                apply_edge(emb, st, epoch_f, n, d, a, b, gamma, alpha, opts.seed);
            }
        });
    }
}

/// Shuffle the edge states with a seeded Fisher–Yates permutation so parallel
/// chunks mix nodes (neighborhoods are contiguous in CSR order, which would
/// make concurrent chunks touch nearly disjoint node sets and serialize the
/// Hogwild updates).
fn shuffle_edges(states: &mut Vec<EdgeState>, seed: u64) {
    let m = states.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5EED_C0DE);
    let mut perm: Vec<u32> = (0..m as u32).collect();
    for i in (1..m).rev() {
        let j = rng.random_range(0..=i);
        perm.swap(i, j);
    }
    let shuffled: Vec<EdgeState> = perm.iter().map(|&p| states[p as usize]).collect();
    *states = shuffled;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_knn_graph(n: usize, k: usize, seed: u64) -> NeighborGraph {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_clusters = 3;
        let mut data = vec![0.0_f32; n * 10];
        for i in 0..n {
            let cluster = i % n_clusters;
            for j in 0..10 {
                data[i * 10 + j] = cluster as f32 * 10.0 + rng.random::<f32>() * 2.0;
            }
        }

        let mut indptr = vec![0_u32; n + 1];
        let mut indices = Vec::new();
        let mut dists = Vec::new();
        for i in 0..n {
            let mut candidates: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut d = 0.0_f32;
                    for k in 0..10 {
                        let diff = data[i * 10 + k] - data[j * 10 + k];
                        d += diff * diff;
                    }
                    (j, d.sqrt())
                })
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for &(j, d) in candidates.iter().take(k) {
                indices.push(j as u32);
                dists.push(d);
            }
            indptr[i + 1] = indices.len() as u32;
        }
        NeighborGraph {
            indptr,
            indices,
            data: dists,
            n_obs: n,
        }
    }

    /// The pre-phase-3 reference implementation: HashMap symmetrisation.
    /// Kept as the accuracy oracle for the CSR construction.
    fn fuzzy_simplicial_set_hashmap(graph: &NeighborGraph) -> Vec<(u32, u32, f32)> {
        let n = graph.n_obs;
        let all_weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let (s, e) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
                smooth_knn_dist(&graph.data[s..e])
            })
            .collect();

        let mut edge_map: HashMap<(u32, u32), f32> = HashMap::new();
        for i in 0..n {
            let (s, e) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
            for k in s..e {
                let j = graph.indices[k];
                let w_ij = all_weights[i][k - s];
                let key = (i.min(j as usize) as u32, i.max(j as usize) as u32);
                let old = *edge_map.get(&key).unwrap_or(&0.0);
                let combined = old + w_ij - old * w_ij;
                edge_map.insert(key, combined);
            }
        }
        let mut edges: Vec<(u32, u32, f32)> = edge_map
            .into_iter()
            .filter(|&(_, w)| w > 1e-5)
            .map(|((s, t), w)| (s, t, w))
            .collect();
        edges.sort_unstable_by_key(|a| (a.0, a.1));
        edges
    }

    /// The undirected edges (source < target), each once, from the CSR's
    /// upper triangle.
    fn fuzzy_edge_list(fuzzy: &FuzzyGraph) -> Vec<(u32, u32, f32)> {
        let n = fuzzy.indptr.len() - 1;
        let mut edges = Vec::new();
        for i in 0..n {
            for k in fuzzy.indptr[i] as usize..fuzzy.indptr[i + 1] as usize {
                if fuzzy.indices[k] as usize > i {
                    edges.push((i as u32, fuzzy.indices[k], fuzzy.weights[k]));
                }
            }
        }
        edges
    }

    #[test]
    fn fuzzy_graph_matches_hashmap_oracle() {
        let graph = make_knn_graph(80, 5, 7);
        let fuzzy = build_fuzzy_graph(&graph);
        let got = fuzzy_edge_list(&fuzzy);
        let want = fuzzy_simplicial_set_hashmap(&graph);
        assert_eq!(got.len(), want.len(), "edge count differs");
        for ((gs, gt, gw), (ws, wt, ww)) in got.iter().zip(&want) {
            assert_eq!(gs, ws, "source mismatch");
            assert_eq!(gt, wt, "target mismatch");
            assert!(
                gw.to_bits() == ww.to_bits(),
                "weight bits differ: {gw} vs {ww}"
            );
        }
    }

    #[test]
    fn fuzzy_graph_is_symmetric_sorted_and_canonical() {
        let graph = make_knn_graph(120, 8, 3);
        let n = graph.n_obs;
        let fuzzy = build_fuzzy_graph(&graph);

        // CSR shape: prefix sums, sorted unique columns, no self-loops.
        assert_eq!(fuzzy.indptr.len(), n + 1);
        assert_eq!(fuzzy.indptr[0], 0);
        for i in 0..n {
            let (s, e) = (fuzzy.indptr[i] as usize, fuzzy.indptr[i + 1] as usize);
            assert!(s <= e);
            for k in s..e {
                assert_ne!(fuzzy.indices[k] as usize, i, "self-loop at {i}");
                if k > s {
                    assert!(
                        fuzzy.indices[k - 1] < fuzzy.indices[k],
                        "row {i} not sorted/deduped"
                    );
                }
            }
        }

        // Both directions stored with bitwise-equal weights.
        let weight = |i: usize, j: usize| -> Option<u32> {
            let (s, e) = (fuzzy.indptr[i] as usize, fuzzy.indptr[i + 1] as usize);
            fuzzy.indices[s..e]
                .iter()
                .position(|&c| c as usize == j)
                .map(|p| fuzzy.weights[s + p].to_bits())
        };
        for i in 0..n {
            let (s, e) = (fuzzy.indptr[i] as usize, fuzzy.indptr[i + 1] as usize);
            for k in s..e {
                let j = fuzzy.indices[k] as usize;
                let wij = fuzzy.weights[k].to_bits();
                let wji = weight(j, i).expect("missing reverse entry");
                assert_eq!(wij, wji, "asymmetric weight {i}<->{j}");
            }
        }

        // Canonical edge list: strictly ordered pairs, each once, weights
        // equal to the CSR entry.
        for (s, t, w) in fuzzy_edge_list(&fuzzy) {
            assert!(s < t, "non-canonical edge {s}->{t}");
            let wij = weight(s as usize, t as usize).expect("edge missing from CSR");
            assert_eq!(wij, w.to_bits());
        }
    }

    #[test]
    fn umap_produces_2d_embedding() {
        let graph = make_knn_graph(300, 15, 42);
        let result = umap(
            &graph,
            &UmapOptions {
                n_epochs: Some(50),
                ..Default::default()
            },
        );
        assert_eq!(result.len(), 300 * 2);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn umap_separates_clusters() {
        let graph = make_knn_graph(450, 15, 42);
        for deterministic in [false, true] {
            let emb = umap(
                &graph,
                &UmapOptions {
                    n_epochs: Some(100),
                    deterministic,
                    ..Default::default()
                },
            );

            let mut centroids = [(0.0_f32, 0.0_f32, 0_usize); 3];
            for i in 0..450 {
                let c = i % 3;
                centroids[c].0 += emb[i * 2];
                centroids[c].1 += emb[i * 2 + 1];
                centroids[c].2 += 1;
            }
            for c in centroids.iter_mut() {
                c.0 /= c.2 as f32;
                c.1 /= c.2 as f32;
            }

            let d01 = ((centroids[0].0 - centroids[1].0).powi(2)
                + (centroids[0].1 - centroids[1].1).powi(2))
            .sqrt();
            let d02 = ((centroids[0].0 - centroids[2].0).powi(2)
                + (centroids[0].1 - centroids[2].1).powi(2))
            .sqrt();
            let d12 = ((centroids[1].0 - centroids[2].0).powi(2)
                + (centroids[1].1 - centroids[2].1).powi(2))
            .sqrt();
            let mode = if deterministic {
                "sequential"
            } else {
                "parallel"
            };
            assert!(d01 > 0.5, "{mode}: clusters 0-1 too close: {d01}");
            assert!(d02 > 0.5, "{mode}: clusters 0-2 too close: {d02}");
            assert!(d12 > 0.5, "{mode}: clusters 1-2 too close: {d12}");
        }
    }

    #[test]
    fn umap_spread_is_reasonable() {
        // The embedding should spread out, not collapse to a point.
        let graph = make_knn_graph(300, 15, 42);
        let emb = umap(
            &graph,
            &UmapOptions {
                n_epochs: Some(100),
                ..Default::default()
            },
        );
        let xs: Vec<f32> = (0..300).map(|i| emb[i * 2]).collect();
        let ys: Vec<f32> = (0..300).map(|i| emb[i * 2 + 1]).collect();
        let x_range = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - xs.iter().copied().fold(f32::INFINITY, f32::min);
        let y_range = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - ys.iter().copied().fold(f32::INFINITY, f32::min);
        // Should not collapse (range > 1) and not explode (range < 1000).
        assert!(x_range > 1.0, "x collapsed: range={x_range}");
        assert!(y_range > 1.0, "y collapsed: range={y_range}");
        assert!(
            x_range < 1000.0 && y_range < 1000.0,
            "exploded: {x_range} x {y_range}"
        );
    }

    #[test]
    fn spectral_init_produces_valid_embedding() {
        let graph = make_knn_graph(300, 15, 42);
        let fuzzy = build_fuzzy_graph(&graph);
        let emb = spectral_init(&fuzzy.indptr, &fuzzy.indices, &fuzzy.weights, 300, 2, 42);
        assert_eq!(emb.len(), 300 * 2);
        assert!(emb.iter().all(|x| x.is_finite()));
        // At least one pair of clusters should be separated (2 eigenvectors
        // can distinguish 2 of 3 clusters; the SGD refines the rest).
        let mut centroids = [(0.0_f32, 0.0_f32, 0_usize); 3];
        for i in 0..300 {
            let c = i % 3;
            centroids[c].0 += emb[i * 2];
            centroids[c].1 += emb[i * 2 + 1];
            centroids[c].2 += 1;
        }
        for c in centroids.iter_mut() {
            c.0 /= c.2 as f32;
            c.1 /= c.2 as f32;
        }
        let dists = [
            ((centroids[0].0 - centroids[1].0).powi(2) + (centroids[0].1 - centroids[1].1).powi(2))
                .sqrt(),
            ((centroids[0].0 - centroids[2].0).powi(2) + (centroids[0].1 - centroids[2].1).powi(2))
                .sqrt(),
            ((centroids[1].0 - centroids[2].0).powi(2) + (centroids[1].1 - centroids[2].1).powi(2))
                .sqrt(),
        ];
        let max_pair_dist = dists[0].max(dists[1]).max(dists[2]);
        assert!(
            max_pair_dist > 1.0,
            "no cluster separation in spectral init: {dists:?}"
        );
    }

    /// Leave-self-out k-NN overlap between two 2-D layouts: the fraction of
    /// shared neighbors, averaged over points. Rotation- and reflection-
    /// invariant, so it measures local-structure agreement rather than
    /// point identity.
    fn knn_overlap_2d(a: &[f32], b: &[f32], n: usize, k: usize) -> f64 {
        let neighbors = |v: &[f32], i: usize| -> Vec<usize> {
            let mut d: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dx = f64::from(v[i * 2] - v[j * 2]);
                    let dy = f64::from(v[i * 2 + 1] - v[j * 2 + 1]);
                    (dx * dx + dy * dy, j)
                })
                .collect();
            d.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            d.truncate(k);
            d.into_iter().map(|(_, j)| j).collect()
        };
        let mut total = 0.0_f64;
        for i in 0..n {
            let na = neighbors(a, i);
            let nb = neighbors(b, i);
            total += na.iter().filter(|j| nb.contains(j)).count() as f64 / k as f64;
        }
        total / n as f64
    }

    #[test]
    fn parallel_sgd_matches_sequential_within_seed_noise() {
        // Point-identical layouts are not a property UMAP has across
        // independent SGD runs: on this graph the inter-cluster angles are
        // driven by repulsion noise (measured Procrustes distance ~1.2 RMS
        // between two sequential runs with different seeds — worse than
        // sequential vs parallel at ~0.38). The fair statement is that the
        // parallel driver agrees with the sequential one at least as well
        // as two sequential runs agree with each other.
        let graph = make_knn_graph(450, 15, 42);
        let opts = |det: bool, seed: u64| UmapOptions {
            n_epochs: Some(150),
            seed,
            deterministic: det,
            ..Default::default()
        };
        let seq = umap(&graph, &opts(true, 3));
        let par = umap(&graph, &opts(false, 3));
        let seq_other_seed = umap(&graph, &opts(true, 4));
        let agree_sp = knn_overlap_2d(&seq, &par, 450, 15);
        let agree_ss = knn_overlap_2d(&seq, &seq_other_seed, 450, 15);
        assert!(
            agree_sp >= 0.9 * agree_ss,
            "parallel/sequential kNN agreement {agree_sp:.3} fell below the seed-noise floor {agree_ss:.3}"
        );
    }

    #[test]
    fn deterministic_mode_is_bit_identical() {
        let graph = make_knn_graph(300, 15, 42);
        let opts = UmapOptions {
            n_epochs: Some(60),
            deterministic: true,
            ..Default::default()
        };
        let a = umap(&graph, &opts);
        let b = umap(&graph, &opts);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), y.to_bits(), "deterministic run differs");
        }
    }

    /// The fuzzy graph counted, then written by row blocks into the caller's
    /// arrays, is the graph `build_fuzzy` returns, and UMAP on those
    /// connectivities equals UMAP on the distances they came from, bit for bit.
    #[test]
    fn umap_on_the_connectivities_equals_umap_on_the_distances() {
        let n = 800;
        let graph = make_knn_graph(n, 12, 7);
        let plan = FuzzyPlan::new(&graph.indptr, &graph.indices, &graph.data, n);
        let lens = plan.row_lens();
        let mut indptr = vec![0_u32; n + 1];
        for (i, &l) in lens.iter().enumerate() {
            indptr[i + 1] = indptr[i] + l;
        }
        let nnz = indptr[n] as usize;
        let (mut indices, mut weights) = (vec![0_u32; nnz], vec![0.0_f32; nnz]);
        // Blocks of 37 rows: many splits, a short last block.
        plan.fill_by(&indptr, &mut indices, &mut weights, 37);
        let (ip, ix, w) = fuzzy_connectivities_from(&graph.indptr, &graph.indices, &graph.data, n);
        assert_eq!((&indptr, &indices), (&ip, &ix));
        assert!(weights
            .iter()
            .zip(&w)
            .all(|(a, b)| a.to_bits() == b.to_bits()));

        let opts = UmapOptions {
            n_epochs: Some(30),
            deterministic: true,
            ..Default::default()
        };
        let from_distances = umap(&graph, &opts);
        let mut from_conn = vec![0.0_f32; n * 2];
        umap_from_connectivities_into(&indptr, &indices, &weights, n, &opts, &mut from_conn);
        assert!(from_distances
            .iter()
            .zip(&from_conn)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn find_ab_params_default() {
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!(a > 1.0 && a < 2.0, "a={a}");
        assert!(b > 0.7 && b < 1.1, "b={b}");
    }

    #[test]
    fn fast_pow_matches_libm_within_sgd_tolerance() {
        // Over the argument ranges the SGD gradients actually use —
        // d² from ~1e-6 to ~1e6, exponents in [b-1, b] for b ∈ [0.1, 1.1] —
        // the polynomial approximation must stay within 2e-5 relative of
        // libm powf (it is ~3 orders below the SGD noise floor).
        let mut worst = 0.0_f64;
        let mut xr = 1e-6_f32;
        while xr < 1e6 {
            for e in [0.1_f32, 0.5, 0.895, 1.0, 1.1, -0.105, -0.9] {
                let approx = fast_pow(xr, e);
                let exact = xr.powf(e);
                let rel = f64::from(((approx - exact) / exact).abs());
                if rel > worst {
                    worst = rel;
                }
                assert!(rel < 2e-5, "fast_pow({xr}, {e}) off by {rel}");
            }
            xr *= 1.3;
        }
        assert!(worst < 2e-5, "worst {worst}");
    }

    /// Values from umap-learn 0.5.12's `smooth_knn_dist` +
    /// `compute_membership_strengths` on the row [0, 0.5, 1, 2, 4] (self
    /// first). All-ones weights, the pre-2026-09-20 behaviour, fail this.
    const UMAP_LEARN_REFERENCE: [f32; 4] = [1.0, 0.754_181, 0.428_969_95, 0.138_780_77];

    #[test]
    fn smooth_knn_dist_matches_umap_learn() {
        let w = smooth_knn_dist(&[0.5_f32, 1.0, 2.0, 4.0]);
        let want = UMAP_LEARN_REFERENCE;
        for (a, b) in w.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{w:?} != {want:?}");
        }
        let target = 5_f32.log2();
        assert!((w.iter().sum::<f32>() - target).abs() < 1e-4);
    }

    #[test]
    fn smooth_knn_dist_produces_valid_weights() {
        let dists = vec![0.5_f32, 1.0, 2.0, 4.0];
        let w = smooth_knn_dist(&dists);
        assert_eq!(w.len(), 4);
        assert!(w.iter().all(|&x| x >= 0.0 && x <= 1.0));
        assert!(w[0] >= w[1] && w[1] >= w[2]);
    }
}
