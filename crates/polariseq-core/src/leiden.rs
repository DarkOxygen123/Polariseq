#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
// Single-character names (i, j, k, c, r, s, t, u, v, w) follow the notation of
// Traag, Waltman & van Eck (2019).

//! Leiden community detection (Traag, Waltman & van Eck 2019, "From Louvain
//! to Leiden: guaranteeing well-connected communities", Sci. Rep. 9, 5233)
//! on CSR.
//!
//! The algorithm follows the paper's Algorithms A.2 and A.3 and the
//! reference implementation by the same authors, CWTS `networkanalysis`
//! (https://github.com/CWTSLeiden/networkanalysis, MIT licence, Copyright (c)
//! 2018 Centre for Science and Technology Studies (CWTS), Leiden University;
//! see `THIRD_PARTY_NOTICES.md`): fast local moving, refinement into
//! well-connected sub-clusters, reduction of the graph onto the refined
//! partition, repeated on the reduced graph until nothing moves.
//!
//! The quality function is modularity with resolution γ, written as the
//! constant Potts model with node weights `k_i` (the strengths) and
//! resolution `γ / 2m`, as in the CWTS code. Two iterations from the previous
//! partition are run, the number Scanpy's documentation recommends.
//!
//! The data layout is our own: the graph is borrowed in place (it can be a
//! memory-mapped file), the sums over neighbouring clusters use a dense
//! scratch vector and the list of clusters touched, reductions are a
//! counting-sort CSR build with the rows merged in parallel, and one
//! `ChaCha8` stream drives every random choice, so the same graph and seed
//! give the same output.

use std::borrow::Cow;

use crate::backend::NeighborGraph;
use crate::umap;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Safety cap on levels (reduced graphs) per iteration; convergence usually
/// takes 3 to 6.
const MAX_LEVELS: usize = 20;

/// Leiden iterations, each continuing from the partition the previous one
/// left (the CWTS code's `nIterations`; Scanpy's documentation recommends 2).
const OUTER_ITERATIONS: usize = 2;

/// Randomness θ of the refinement (the CWTS default, 0.01).
const THETA: f64 = 0.01;

/// Weighted undirected graph in CSR form: both directions stored, rows
/// sorted by column, self-loops (aggregate graphs) as a single entry.
/// Sums are f64. `strength[i]` is the node's weight in the quality function:
/// its row sum (a self-loop counted twice) for a graph given by the caller,
/// the sum of its members' strengths for a reduced graph. `two_m` is
/// Σ strength.
///
/// The arrays are borrowed when the graph is the caller's (the
/// connectivities `pp.neighbors` stored, possibly a memory-mapped file), so
/// clustering reads them in place, and owned for the aggregate graphs the
/// algorithm builds.
#[derive(Clone)]
pub(crate) struct WeightedGraph<'a> {
    pub(crate) n: usize,
    pub(crate) indptr: Cow<'a, [u32]>,
    pub(crate) adj: Cow<'a, [u32]>,
    pub(crate) w: Cow<'a, [f32]>,
    pub(crate) strength: Vec<f64>,
    pub(crate) two_m: f64,
    /// Undirected edge count (self-loops once) — for reporting.
    pub(crate) n_undirected: usize,
}

impl WeightedGraph<'static> {
    /// Build from a symmetric CSR (both directions; rows sorted by column).
    pub(crate) fn from_symmetric_csr(
        indptr: Vec<u32>,
        adj: Vec<u32>,
        w: Vec<f32>,
        n: usize,
    ) -> Self {
        WeightedGraph::from_parts(Cow::Owned(indptr), Cow::Owned(adj), Cow::Owned(w), n)
    }
}

impl WeightedGraph<'static> {
    /// A reduced graph: no self-loops, strengths given (its members' sums).
    fn from_csr_with_strength(
        indptr: Vec<u32>,
        adj: Vec<u32>,
        w: Vec<f32>,
        n: usize,
        strength: Vec<f64>,
    ) -> Self {
        let n_undirected = (0..n)
            .map(|i| {
                (indptr[i] as usize..indptr[i + 1] as usize)
                    .filter(|&t| adj[t] as usize >= i)
                    .count()
            })
            .sum();
        let two_m = strength.iter().sum();
        WeightedGraph {
            n,
            indptr: Cow::Owned(indptr),
            adj: Cow::Owned(adj),
            w: Cow::Owned(w),
            strength,
            two_m,
            n_undirected,
        }
    }
}

impl<'a> WeightedGraph<'a> {
    /// Build over a symmetric CSR the caller holds, without copying it.
    pub(crate) fn from_symmetric_csr_ref(
        indptr: &'a [u32],
        adj: &'a [u32],
        w: &'a [f32],
        n: usize,
    ) -> Self {
        Self::from_parts(
            Cow::Borrowed(indptr),
            Cow::Borrowed(adj),
            Cow::Borrowed(w),
            n,
        )
    }

    fn from_parts(
        indptr: Cow<'a, [u32]>,
        adj: Cow<'a, [u32]>,
        w: Cow<'a, [f32]>,
        n: usize,
    ) -> Self {
        debug_assert_eq!(indptr.len(), n + 1);
        debug_assert_eq!(adj.len(), w.len());
        let mut strength = vec![0.0_f64; n];
        let mut n_undirected = 0_usize;
        for i in 0..n {
            let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
            let mut k = 0.0_f64;
            for t in s..e {
                k += f64::from(w[t]);
                let j = adj[t];
                if j as usize >= i {
                    // each undirected edge once: its entry in the lower row
                    n_undirected += 1;
                    if j as usize == i {
                        k += f64::from(w[t]); // self-loop counts twice
                    }
                }
            }
            strength[i] = k;
        }
        let two_m = strength.iter().sum();
        Self {
            n,
            indptr,
            adj,
            w,
            strength,
            two_m,
            n_undirected,
        }
    }

    #[inline]
    fn row(&self, i: usize) -> (usize, usize) {
        (self.indptr[i] as usize, self.indptr[i + 1] as usize)
    }
}

/// The weighted graph every clustering method runs on: the UMAP fuzzy
/// connectivities of the kNN distance graph (scanpy semantics — Leiden
/// clusters `connectivities`, not raw distances).
pub(crate) fn weighted_from_distances(graph: &NeighborGraph) -> WeightedGraph<'static> {
    let fuzzy = umap::build_fuzzy_graph(graph);
    let n = graph.n_obs;
    WeightedGraph::from_symmetric_csr(fuzzy.indptr, fuzzy.indices, fuzzy.weights, n)
}

/// The weighted graph given directly: a symmetric connectivity CSR (both
/// directions, rows sorted by column), such as `pp.neighbors`'
/// `obsp['connectivities']`. Taken by value, so no second graph is built.
///
/// This is the entry point for a caller that already holds the
/// connectivities. Passing them to [`weighted_from_distances`] instead
/// treats similarities as distances and builds the fuzzy set a second time,
/// which inverts which neighbours are close; `tl.leiden` did exactly that
/// from 2026-09-06 until this function existed.
pub(crate) fn weighted_from_connectivities(graph: NeighborGraph) -> WeightedGraph<'static> {
    let n = graph.n_obs;
    WeightedGraph::from_symmetric_csr(graph.indptr, graph.indices, graph.data, n)
}

/// Compact a membership vector to contiguous ids in first-appearance order
/// (deterministic). Returns the compacted labels and community count.
/// Labels may exceed the vector length (aggregate base memberships carry
/// ids over a shrunken node set), so the remap is sized by the max
/// label, not the node count.
fn compact(membership: &[u32]) -> (Vec<u32>, usize) {
    let max_lab = membership.iter().copied().max().map_or(0, |m| m as usize);
    let mut remap = vec![u32::MAX; max_lab + 1];
    let mut next = 0_u32;
    let mut out = Vec::with_capacity(membership.len());
    for &lab in membership {
        let m = remap[lab as usize];
        let m = if m == u32::MAX {
            remap[lab as usize] = next;
            next += 1;
            next - 1
        } else {
            m
        };
        out.push(m);
    }
    (out, next as usize)
}

/// Relabel to contiguous ids ordered by community size descending, ties by
/// lowest first-appearance (scanpy's convention: cluster "0" is the
/// largest). Empty label slots (sparse ids) are dropped.
pub(crate) fn relabel_by_size(labels: &[u32]) -> (Vec<u32>, usize) {
    let n_comms = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut sizes = vec![0_usize; n_comms];
    let mut first_at = vec![usize::MAX; n_comms];
    for (i, &lab) in labels.iter().enumerate() {
        let c = lab as usize;
        sizes[c] += 1;
        if first_at[c] == usize::MAX {
            first_at[c] = i;
        }
    }
    let n_present = sizes.iter().filter(|&s| *s > 0).count();
    let mut order: Vec<usize> = (0..n_comms).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(first_at[a].cmp(&first_at[b])));
    let mut remap = vec![0_u32; n_comms];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new as u32;
    }
    let out = labels.iter().map(|&lab| remap[lab as usize]).collect();
    (out, n_present)
}

/// Minimum gain for a vertex move, **relative to the graph's total edge
/// weight**. An absolute epsilon does not work here: the move score
/// `w_to[c] − res·k_i·K_c` is in raw weight units, so on an aggregated level
/// its operands are of order `two_m`, and f64 rounding alone is
/// `≈ 2.2e-16 · two_m`. A fixed `1e-12` sits *below that noise floor*, so
/// two communities trade a vertex back and forth on rounding error, each
/// move looking "strictly better" while modularity does not move at all.
///
/// Measured on the heart_50k connectivity graph: an aggregated level of
/// **50 nodes** made 1,000,000+ moves with the queue grown to 17.2 million
/// entries, pinned at 100 % of one core, and never terminated. Scaling the
/// threshold by `two_m` puts it safely above rounding (2.2e-4 at that
/// level) and far below any real gain.
///
/// This is why the bug appeared only when `tl.leiden` switched from the
/// distances graph to connectivities: distance weights are large and
/// well separated, fuzzy connectivity weights live in [0, 1], so gains are
/// small and near-ties are everywhere.
const MOVE_REL_TOL: f64 = 1e-10;

/// Hard ceiling on moves per `fast_local_move` call, as a multiple of the vertex
/// count. The relative tolerance above removes the *known* cause of
/// non-termination; this guarantees termination regardless. Local moving is
/// a heuristic — stopping it early yields a valid partition that the outer
/// level loop can still improve — so a cap costs quality at worst, never
/// correctness. Never reached on any graph in the test suite or the
/// benchmark ladder; if it fires, something is wrong and the trace says so.
const MAX_MOVES_PER_VERTEX: usize = 100;

/// Fast local moving (Traag et al. 2019, Algorithm A.2; the CWTS
/// `FastLocalMovingAlgorithm`). The nodes are visited from a cyclic queue
/// that starts as a random permutation of all nodes. A node is taken out of
/// its cluster and moved to the candidate with the largest gain
/// `w(i, C) − res · k_i · K_C`; the candidates are one empty cluster (gain
/// 0, so a node can always leave) and the clusters of its neighbours, in the
/// order its edges reach them. Its own cluster wins every tie, which with
/// [`MOVE_REL_TOL`] also covers gains equal up to rounding. When a node
/// moves, its neighbours outside the new cluster that were stable rejoin the
/// queue. Self-loops carry no gain and are skipped.
///
/// `membership` ids must be below `wg.n` (singletons, or a compacted
/// partition); the result may leave gaps, which the caller compacts.
/// Returns the number of moves.
fn fast_local_move(
    wg: &WeightedGraph<'_>,
    membership: &mut [u32],
    res: f64,
    rng: &mut ChaCha8Rng,
) -> usize {
    let n = wg.n;
    if n <= 1 {
        return 0;
    }
    let move_eps = MOVE_REL_TOL * wg.two_m;
    let max_moves = MAX_MOVES_PER_VERTEX.saturating_mul(n);
    let mut cluster_w = vec![0.0_f64; n];
    let mut count = vec![0_u32; n];
    for i in 0..n {
        cluster_w[membership[i] as usize] += wg.strength[i];
        count[membership[i] as usize] += 1;
    }
    // Empty clusters as a stack whose top is the smallest free id.
    let mut unused: Vec<u32> = (0..n as u32)
        .rev()
        .filter(|&c| count[c as usize] == 0)
        .collect();

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let mut stable = vec![false; n];
    let mut n_unstable = n;
    let mut w_to = vec![0.0_f64; n];
    let mut candidates: Vec<u32> = Vec::new();
    let mut moves = 0_usize;
    let mut pos = 0_usize;

    while n_unstable > 0 {
        if moves >= max_moves {
            if std::env::var_os("POLARISEQ_TRACE").is_some() {
                eprintln!(
                    "[leiden] local moving hit the move cap: n={n} moves={moves} \
                     pending={n_unstable} two_m={:.3e}",
                    wg.two_m
                );
            }
            break;
        }
        let i = order[pos] as usize;
        let ki = wg.strength[i];
        let current = membership[i] as usize;

        cluster_w[current] -= ki;
        count[current] -= 1;
        if count[current] == 0 {
            unused.push(current as u32);
        }

        candidates.clear();
        candidates.push(*unused.last().expect("the node's removal leaves a free id"));
        let (s, e) = wg.row(i);
        for t in s..e {
            let j = wg.adj[t] as usize;
            if j == i {
                continue;
            }
            let c = membership[j];
            if w_to[c as usize] == 0.0 {
                candidates.push(c);
            }
            w_to[c as usize] += f64::from(wg.w[t]);
        }

        let mut best = current;
        let mut best_gain = w_to[current] - res * ki * cluster_w[current];
        for &c in &candidates {
            let c = c as usize;
            let gain = w_to[c] - res * ki * cluster_w[c];
            if gain > best_gain + move_eps {
                best = c;
                best_gain = gain;
            }
            w_to[c] = 0.0;
        }
        w_to[current] = 0.0;

        cluster_w[best] += ki;
        count[best] += 1;
        if unused.last().is_some_and(|&u| u as usize == best) {
            unused.pop();
        }
        stable[i] = true;
        n_unstable -= 1;

        if best != current {
            membership[i] = best as u32;
            moves += 1;
            for t in s..e {
                let j = wg.adj[t] as usize;
                if stable[j] && membership[j] as usize != best {
                    stable[j] = false;
                    n_unstable += 1;
                    order[(pos + n_unstable) % n] = j as u32;
                }
            }
        }
        pos = if pos + 1 < n { pos + 1 } else { 0 };
    }
    moves
}

/// Refinement (Traag et al. 2019, Algorithm A.3, `MergeNodesSubset`; the CWTS
/// `LocalMergingAlgorithm`), run inside every cluster of `clusters` at once.
/// Each cluster C starts as singletons, visited in random order. A node v
/// still in a singleton, and well connected to the rest of C
/// (`w(v, C∖v) ≥ res · k_v · (K_C − k_v)`), may join a sub-cluster S of a
/// neighbour in C that is itself well connected
/// (`w(S, C∖S) ≥ res · K_S · (K_C − K_S)`), or stay alone (gain 0). Among
/// the candidates with a non-negative gain Δ it picks one with probability
/// proportional to `exp(Δ / θ)`. The exponents are shifted by the largest
/// gain so they cannot overflow; this leaves the probabilities unchanged.
/// Only nodes that are still alone move, so sub-clusters only grow and
/// always stay connected.
///
/// Returns the refined partition, compacted, and its size.
fn refine(
    wg: &WeightedGraph<'_>,
    clusters: &[u32],
    n_clusters: usize,
    res: f64,
    rng: &mut ChaCha8Rng,
) -> (Vec<u32>, usize) {
    let n = wg.n;
    // Members of each cluster, in node order (counting sort).
    let mut offsets = vec![0_usize; n_clusters + 1];
    for &c in clusters {
        offsets[c as usize + 1] += 1;
    }
    for c in 0..n_clusters {
        offsets[c + 1] += offsets[c];
    }
    let mut members = vec![0_u32; n];
    let mut cursor = offsets.clone();
    for (i, &c) in clusters.iter().enumerate() {
        members[cursor[c as usize]] = i as u32;
        cursor[c as usize] += 1;
    }

    // Every node starts as its own sub-cluster, numbered by the node.
    let mut refined: Vec<u32> = (0..n as u32).collect();
    let mut sub_w: Vec<f64> = wg.strength.clone();
    let mut external = vec![0.0_f64; n];
    for i in 0..n {
        let (s, e) = wg.row(i);
        for t in s..e {
            let j = wg.adj[t] as usize;
            if j != i && clusters[j] == clusters[i] {
                external[i] += f64::from(wg.w[t]);
            }
        }
    }
    let mut non_singleton = vec![false; n];
    let mut w_to = vec![0.0_f64; n];
    let mut candidates: Vec<u32> = Vec::new();
    let mut gains: Vec<f64> = Vec::new();
    let mut cumulative: Vec<f64> = Vec::new();

    for c in 0..n_clusters {
        let part = &members[offsets[c]..offsets[c + 1]];
        if part.len() < 2 {
            continue;
        }
        let k_total: f64 = part.iter().map(|&v| wg.strength[v as usize]).sum();
        let mut visit = part.to_vec();
        visit.shuffle(rng);
        for &v in &visit {
            let v = v as usize;
            let kv = wg.strength[v];
            if non_singleton[v] || external[v] < res * sub_w[v] * (k_total - sub_w[v]) {
                continue;
            }
            sub_w[v] = 0.0;
            external[v] = 0.0;

            candidates.clear();
            candidates.push(v as u32);
            let (s, e) = wg.row(v);
            for t in s..e {
                let u = wg.adj[t] as usize;
                if u == v || clusters[u] as usize != c {
                    continue;
                }
                let r = refined[u];
                if w_to[r as usize] == 0.0 {
                    candidates.push(r);
                }
                w_to[r as usize] += f64::from(wg.w[t]);
            }

            gains.clear();
            let mut top = 0.0_f64; // staying alone always qualifies with gain 0
            for &r in &candidates {
                let r = r as usize;
                let well_connected = external[r] >= res * sub_w[r] * (k_total - sub_w[r]);
                let gain = w_to[r] - res * kv * sub_w[r];
                if well_connected && gain >= 0.0 {
                    gains.push(gain);
                    top = top.max(gain);
                } else {
                    gains.push(f64::NAN);
                }
                w_to[r] = 0.0;
            }
            cumulative.clear();
            let mut total = 0.0_f64;
            for &g in &gains {
                if !g.is_nan() {
                    total += ((g - top) / THETA).exp();
                }
                cumulative.push(total);
            }
            let draw = rng.random::<f64>() * total;
            let pick = cumulative
                .partition_point(|&x| x < draw)
                .min(candidates.len() - 1);
            let chosen = candidates[pick] as usize;

            sub_w[chosen] += kv;
            for t in s..e {
                let u = wg.adj[t] as usize;
                if u == v || clusters[u] as usize != c {
                    continue;
                }
                if refined[u] as usize == chosen {
                    external[chosen] -= f64::from(wg.w[t]);
                } else {
                    external[chosen] += f64::from(wg.w[t]);
                }
            }
            if chosen != v {
                refined[v] = chosen as u32;
                non_singleton[chosen] = true;
            }
        }
    }
    compact(&refined)
}

/// The reduced (aggregate) graph of a partition: one node per part, the
/// weight between two parts summed over the edges between them, and each
/// node's strength the sum of its members' strengths. Edges inside a part
/// are dropped (they carry no gain in local moving), so `two_m` is kept
/// exactly and needs no self-loops. Counting-sort scatter of the edges
/// between parts, then a sort and merge per row (parallel, deterministic).
fn reduce(wg: &WeightedGraph<'_>, part: &[u32], n_parts: usize) -> WeightedGraph<'static> {
    let mut strength = vec![0.0_f64; n_parts];
    let mut indptr = vec![0_u32; n_parts + 1];
    for i in 0..wg.n {
        let a = part[i] as usize;
        strength[a] += wg.strength[i];
        let (s, e) = wg.row(i);
        indptr[a + 1] += (s..e)
            .filter(|&t| part[wg.adj[t] as usize] as usize != a)
            .count() as u32;
    }
    for r in 0..n_parts {
        indptr[r + 1] += indptr[r];
    }
    let nnz = indptr[n_parts] as usize;
    let mut adj = vec![0_u32; nnz];
    let mut w = vec![0.0_f32; nnz];
    let mut cursor = indptr.clone();
    for i in 0..wg.n {
        let a = part[i] as usize;
        let (s, e) = wg.row(i);
        for t in s..e {
            let b = part[wg.adj[t] as usize];
            if b as usize != a {
                let p = cursor[a] as usize;
                adj[p] = b;
                w[p] = wg.w[t];
                cursor[a] += 1;
            }
        }
    }
    let rows: Vec<Vec<(u32, f32)>> = (0..n_parts)
        .into_par_iter()
        .map(|r| {
            let (s, e) = (indptr[r] as usize, indptr[r + 1] as usize);
            let mut row: Vec<(u32, f32)> = (s..e).map(|t| (adj[t], w[t])).collect();
            row.sort_unstable_by_key(|&(c, _)| c);
            let mut merged: Vec<(u32, f64)> = Vec::with_capacity(row.len());
            for &(col, x) in &row {
                match merged.last_mut() {
                    Some(last) if last.0 == col => last.1 += f64::from(x),
                    _ => merged.push((col, f64::from(x))),
                }
            }
            merged.into_iter().map(|(c, x)| (c, x as f32)).collect()
        })
        .collect();
    let mut new_indptr = vec![0_u32; n_parts + 1];
    for (r, row) in rows.iter().enumerate() {
        new_indptr[r + 1] = new_indptr[r] + row.len() as u32;
    }
    let mut new_adj = Vec::with_capacity(new_indptr[n_parts] as usize);
    let mut new_w = Vec::with_capacity(new_indptr[n_parts] as usize);
    for row in rows {
        for (c, x) in row {
            new_adj.push(c);
            new_w.push(x);
        }
    }
    WeightedGraph::from_csr_with_strength(new_indptr, new_adj, new_w, n_parts, strength)
}

/// One Leiden iteration from the partition in `clusters` (Traag et al. 2019,
/// Algorithm A.2; the CWTS `LeidenAlgorithm.improveClusteringOneIteration`):
/// local moving; unless every node is alone, refine, reduce the graph onto
/// the refined partition with each refined sub-cluster starting in its
/// cluster, and continue on the reduced graph. When refinement merges
/// nothing, the graph is reduced onto the clusters themselves, each starting
/// alone. The reduced graph's partition is then carried back to the nodes.
/// Returns whether local moving changed anything at any level.
fn improve(
    wg: &WeightedGraph<'_>,
    clusters: &mut Vec<u32>,
    n_clusters: &mut usize,
    res: f64,
    rng: &mut ChaCha8Rng,
    level: usize,
) -> bool {
    let mut update = fast_local_move(wg, clusters, res, rng) > 0;
    let (c, nc) = compact(clusters);
    *clusters = c;
    *n_clusters = nc;
    if nc >= wg.n || level + 1 >= MAX_LEVELS {
        return update;
    }
    let (refined, n_refined) = refine(wg, clusters, nc, res, rng);
    let (reduced, mut sub, mapping) = if n_refined < wg.n {
        let mut start = vec![0_u32; n_refined];
        for (i, &r) in refined.iter().enumerate() {
            start[r as usize] = clusters[i];
        }
        (reduce(wg, &refined, n_refined), start, refined)
    } else {
        let mapping = clusters.clone();
        (reduce(wg, &mapping, nc), (0..nc as u32).collect(), mapping)
    };
    let mut sub_n = 0;
    update |= improve(&reduced, &mut sub, &mut sub_n, res, rng, level + 1);
    for (i, m) in mapping.iter().enumerate() {
        clusters[i] = sub[*m as usize];
    }
    *n_clusters = sub_n;
    update
}

/// Leiden: [`OUTER_ITERATIONS`] iterations of [`improve`], each continuing
/// from the partition the previous one left, keeping the partition of
/// highest modularity; stops early when an iteration changes nothing. Then
/// [`coarsen_degenerate`]. Returns labels, community count, modularity.
pub(crate) fn leiden_partition(
    wg: &WeightedGraph<'_>,
    resolution: f64,
    seed: u64,
) -> (Vec<u32>, usize, f64) {
    if wg.two_m == 0.0 || wg.n == 0 {
        return (vec![0_u32; wg.n], wg.n.min(1), 0.0);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let res = resolution / wg.two_m;
    let mut clusters: Vec<u32> = (0..wg.n as u32).collect();
    let mut n_clusters = wg.n;
    let mut best: Option<(Vec<u32>, usize, f64)> = None;
    for it in 0..OUTER_ITERATIONS {
        let update = improve(wg, &mut clusters, &mut n_clusters, res, &mut rng, 0);
        let q = modularity(wg, &clusters, resolution);
        if std::env::var_os("POLARISEQ_TRACE").is_some() {
            eprintln!("[leiden] iteration {it}: k={n_clusters} Q={q:.6}");
        }
        if best.as_ref().is_none_or(|b| q > b.2 + 1e-12) {
            best = Some((clusters.clone(), n_clusters, q));
        }
        if !update {
            break;
        }
    }
    let (labels, n_comms, _q) = best.unwrap_or_else(|| (vec![0_u32; wg.n], 1, 0.0));
    let (labels, n_comms) = coarsen_degenerate(wg, labels, n_comms, resolution);
    let q = modularity(wg, &labels, resolution);
    (labels, n_comms, q)
}

/// Modularity with resolution γ on the weighted graph, in the per-community
/// form `Q = Σ_c [W_c/m − γ K_c²/(4m²)]`, algebraically identical to the
/// pair form `(1/2m) Σ_ij [w_ij − γ k_i k_j / 2m] δ(c_i, c_j)` (whose degree
/// term runs over all pairs within a community, not only edges).
/// `W_c` counts each internal undirected edge once and each self-loop once;
/// `K_c` is the community strength.
pub(crate) fn modularity(wg: &WeightedGraph<'_>, labels: &[u32], resolution: f64) -> f64 {
    if wg.two_m == 0.0 {
        return 0.0;
    }
    let n_comms = labels
        .iter()
        .copied()
        .max()
        .map_or(0_usize, |m| m as usize + 1);
    let mut directed_in = vec![0.0_f64; n_comms]; // internal row sums
    let mut self_w = vec![0.0_f64; n_comms]; // self-loop weights
    let mut k_c = vec![0.0_f64; n_comms];
    for i in 0..wg.n {
        let c = labels[i] as usize;
        k_c[c] += wg.strength[i];
        let (s, e) = wg.row(i);
        for t in s..e {
            let j = wg.adj[t] as usize;
            if labels[j] == c as u32 {
                directed_in[c] += f64::from(wg.w[t]);
                if j == i {
                    self_w[c] += f64::from(wg.w[t]);
                }
            }
        }
    }
    let m = wg.two_m / 2.0;
    let mut q = 0.0_f64;
    for c in 0..n_comms {
        let w_c = (directed_in[c] + self_w[c]) / 2.0;
        q += w_c / m - resolution * k_c[c] * k_c[c] / (4.0 * m * m);
    }
    q
}

/// Collapse modularity-degenerate communities. On real graphs the optimum
/// is surrounded by a flat region: merging or keeping 100-1000-node
/// fragments changes Q by ~1e-6 (measured on heart_100k: twelve sub-1000
/// clusters whose best merges were dQ ∈ [−5e-6, +3.4e-4]), so the level
/// dynamics — which only make strictly-improving moves — strand them while
/// igraph's trajectory happens to land on the coarse side. This pass
/// greedily merges the best community pair while its closed-form gain
/// `dQ = w(a,b)/m − γ K_a K_b / (2m²)` is positive (strictly improving
/// merges only; see the note in the body). Deterministic; k is small, so
/// the O(k³) scan is noise.
fn coarsen_degenerate(
    wg: &WeightedGraph<'_>,
    mut labels: Vec<u32>,
    n_comms: usize,
    resolution: f64,
) -> (Vec<u32>, usize) {
    // Only strictly-improving merges. This pass used to accept merges that
    // *lost* up to 1e-4 |Q|, to compensate for level dynamics that stranded
    // sub-1000-node fragments where igraph's trajectory landed coarser.
    //
    // That premise was established while `tl.leiden` was clustering the
    // distances graph. Once it moved to the connectivities (0fd1c0d) the
    // level procedure lands in the right range on its own, and the lossy
    // merges overshoot: measured against scanpy's igraph Leiden, the level
    // output was k=25 on pbmc10k (scanpy: 19-20) and this pass drove it to
    // 17. Worse, it did so on three unrelated datasets at once — pbmc10k,
    // pbmc20k and neuron10k all collapsed to exactly 17.
    //
    // The merges were also analytically pointless. Disabling them moves the
    // count back toward scanpy (8->9, 17->21, 17->21 against 9, 19, 22) and
    // leaves ARI essentially unchanged (0.8660->0.8664, 0.7343->0.7350,
    // 0.5973->0.6006) — the merged communities really were degenerate, so
    // collapsing them changed the cluster count without changing the
    // partition. Cosmetically harmful, analytically neutral: dropped.
    let eps = 0.0;
    let m = wg.two_m / 2.0;
    // Community strengths and pairwise weights (upper triangle).
    let mut k_arr = vec![0.0_f64; n_comms];
    let mut w_pairs = vec![0.0_f64; n_comms * n_comms];
    for i in 0..wg.n {
        let c = labels[i] as usize;
        k_arr[c] += wg.strength[i];
        let (s, e) = wg.row(i);
        for t in s..e {
            let j = wg.adj[t] as usize;
            if j != i && labels[j] != c as u32 {
                // both directions stored; accumulate once per pair
                if j > i {
                    let d = labels[j] as usize;
                    let (a, b) = (c.min(d), c.max(d));
                    w_pairs[a * n_comms + b] += f64::from(wg.w[t]);
                }
            }
        }
    }
    let mut alive: Vec<bool> = (0..n_comms).map(|_| true).collect();
    loop {
        let mut best_pair: Option<(usize, usize, f64)> = None;
        for a in 0..n_comms {
            if !alive[a] {
                continue;
            }
            for b in (a + 1)..n_comms {
                if !alive[b] {
                    continue;
                }
                let w_ab = w_pairs[a * n_comms + b];
                let dq = w_ab / m - resolution * k_arr[a] * k_arr[b] / (2.0 * m * m);
                if dq > -eps && best_pair.is_none_or(|(_, _, bq)| dq > bq) {
                    best_pair = Some((a, b, dq));
                }
            }
        }
        let Some((a, b, _dq)) = best_pair else {
            break;
        };
        // Merge a into b: sum strengths, fold a's pair weights into b's.
        k_arr[b] += k_arr[a];
        alive[a] = false;
        for c in 0..n_comms {
            let (lo, hi) = (a.min(c), a.max(c));
            let w_ac = w_pairs[lo * n_comms + hi];
            if w_ac != 0.0 && c != a {
                w_pairs[lo * n_comms + hi] = 0.0;
                let (lo, hi) = (b.min(c), b.max(c));
                w_pairs[lo * n_comms + hi] += w_ac;
            }
        }
        for i in 0..wg.n {
            if labels[i] as usize == a {
                labels[i] = b as u32;
            }
        }
    }
    compact(&labels)
}

/// Multi-level Louvain (Blondel et al. 2008) on the same machinery: local
/// moving and reduction, no refinement. The engine behind `fast_community`.
/// Stops when a level moves nothing, merges nothing, or moves fewer than
/// `convergence_fraction` of its nodes.
pub(crate) fn louvain_partition(
    wg: &WeightedGraph<'_>,
    resolution: f64,
    seed: u64,
    max_levels: usize,
    convergence_fraction: f64,
) -> Vec<u32> {
    let n = wg.n;
    if wg.two_m == 0.0 || n == 0 {
        return vec![0_u32; n];
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let res = resolution / wg.two_m;
    let mut graph: Cow<'_, WeightedGraph<'_>> = Cow::Borrowed(wg);
    let mut membership: Vec<u32> = (0..n as u32).collect();
    let mut mapping: Vec<u32> = (0..n as u32).collect();

    for _level in 0..max_levels.max(1) {
        let moves = fast_local_move(&graph, &mut membership, res, &mut rng);
        let (m_c, n_comms) = compact(&membership);
        membership = m_c;
        if moves == 0
            || n_comms >= graph.n
            || (moves as f64) < convergence_fraction * graph.n as f64
        {
            break;
        }
        let reduced = reduce(&graph, &membership, n_comms);
        for m in &mut mapping {
            *m = membership[*m as usize];
        }
        graph = Cow::Owned(reduced);
        membership = (0..n_comms as u32).collect();
    }

    mapping.iter().map(|&m| membership[m as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a WeightedGraph from undirected edge tuples on 0..n.
    fn wg_from_edges(n: usize, edges: &[(usize, usize, f32)]) -> WeightedGraph<'static> {
        let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
        for &(a, b, w) in edges {
            rows[a].push((b as u32, w));
            if a != b {
                rows[b].push((a as u32, w));
            }
        }
        let mut indptr = vec![0_u32; n + 1];
        let mut adj = Vec::new();
        let mut w = Vec::new();
        for (i, row) in rows.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(c, _)| c);
            for &(c, wt) in row.iter() {
                adj.push(c);
                w.push(wt);
            }
            indptr[i + 1] = adj.len() as u32;
        }
        WeightedGraph::from_symmetric_csr(indptr, adj, w, n)
    }

    /// A complete graph with large, equal weights.
    ///
    /// Note what this does *not* do: it does not reproduce the heart_50k
    /// ping-pong. That was verified by reintroducing the absolute 1e-12
    /// epsilon — these tests still pass. The real trigger needs an
    /// aggregated level's large self-loops (internal weight), which make
    /// `res·k_i·K_c` huge while `w_to` stays small, plus drift accumulated
    /// in `comm_strength` over a million incremental += / -= updates. None
    /// of that is reachable from a hand-built 50-node graph.
    ///
    /// What these tests do cover is that the *scaling* of the threshold is
    /// sane across three orders of magnitude of edge weight, which is the
    /// property that broke when `tl.leiden` switched graphs. The actual
    /// regression guard for the hang is two-fold: `MAX_MOVES_PER_VERTEX`,
    /// which makes non-termination impossible by construction, and
    /// `test_leiden.py::test_leiden_terminates_on_a_real_graph`, which
    /// runs the real dataset.
    fn degenerate_aggregate(n: usize, weight: f32) -> WeightedGraph<'static> {
        let mut edges = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                edges.push((a, b, weight));
            }
        }
        wg_from_edges(n, &edges)
    }

    #[test]
    fn fast_local_move_converges_on_a_symmetric_clique() {
        // A symmetric clique is the worst case for tie-breaking: every move
        // is an exact tie. Convergence must be near-linear, not quadratic,
        // and the safety cap must stay untouched.
        let n = 50;
        let wg = degenerate_aggregate(n, 1.0e6);
        assert!(
            wg.two_m > 1.0e9,
            "test graph must be large-weighted enough to swamp a 1e-12 \
             epsilon in rounding error; two_m = {}",
            wg.two_m
        );

        let mut membership: Vec<u32> = (0..n as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let moves = fast_local_move(&wg, &mut membership, 1.0 / wg.two_m, &mut rng);

        let cap = MAX_MOVES_PER_VERTEX * n;
        assert!(
            moves < cap,
            "local moving hit its safety cap ({moves} >= {cap}) — it is \
             ping-ponging on rounding noise, not converging"
        );
        // Converging on a symmetric clique means collapsing to one
        // community, and doing it in far fewer moves than the cap allows.
        assert!(
            moves <= 2 * n,
            "expected near-linear convergence, got {moves}"
        );
    }

    #[test]
    fn leiden_terminates_on_small_weighted_graphs() {
        // The same shape end to end, at the weight scale of UMAP fuzzy
        // connectivities ([0, 1]) and at the aggregate scale (large), so a
        // regression in either direction is caught.
        for &weight in &[0.05_f32, 1.0, 1.0e6] {
            let wg = degenerate_aggregate(40, weight);
            let (labels, k, q) = leiden_partition(&wg, 1.0, 0);
            assert_eq!(labels.len(), 40);
            assert!((1..=40).contains(&k), "weight {weight}: k = {k}");
            assert!(q.is_finite(), "weight {weight}: Q = {q}");
        }
    }

    fn two_cliques() -> WeightedGraph<'static> {
        wg_from_edges(
            6,
            &[
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 2, 1.0),
                (3, 4, 1.0),
                (3, 5, 1.0),
                (4, 5, 1.0),
                (2, 3, 0.05), // weak bridge
            ],
        )
    }

    #[test]
    fn two_cliques_split() {
        let wg = two_cliques();
        let (labels, n_comms, q) = leiden_partition(&wg, 1.0, 0);
        assert_eq!(n_comms, 2, "labels {labels:?}");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
        // Exact value for this partition: ΣW_c/m − ΣK_c²/(4m²) = 0.4917.
        assert!((q - 0.4917).abs() < 1e-3, "modularity {q}");
    }

    #[test]
    fn ring_of_cliques_one_each() {
        // 5 cliques of 4 nodes on a ring, single weak link between
        // neighbours — the canonical community structure.
        let mut edges = Vec::new();
        for c in 0..5 {
            let base = c * 4;
            for a in 0..4 {
                for b in (a + 1)..4 {
                    edges.push((base + a, base + b, 1.0));
                }
            }
            let next = ((c + 1) % 5) * 4;
            edges.push((base + 3, next, 0.05));
        }
        let wg = wg_from_edges(20, &edges);
        let (labels, n_comms, _q) = leiden_partition(&wg, 1.0, 0);
        assert_eq!(n_comms, 5, "labels {labels:?}");
        for c in 0..5 {
            let base = c * 4;
            let lab = labels[base];
            for v in 1..4 {
                assert_eq!(labels[base + v], lab, "clique {c} split");
            }
        }
    }

    #[test]
    fn modularity_matches_brute_force() {
        // Deterministic pseudo-random weighted graph.
        let mut x = 0x1234_5678_u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let n = 60_usize;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if next() % 7 == 0 {
                    edges.push((i, j, 0.1 + (next() % 100) as f32 / 100.0));
                }
            }
        }
        let wg = wg_from_edges(n, &edges);
        let (labels, n_comms, q) = leiden_partition(&wg, 1.0, 42);

        // Brute force: Q = Σ_c [W_c/m − γ K_c²/(4m²)] over the edge list.
        let m = wg.two_m / 2.0;
        let mut w_c = vec![0.0_f64; n_comms];
        let mut k_c = vec![0.0_f64; n_comms];
        for &(a, b, w) in &edges {
            if labels[a] == labels[b] {
                w_c[labels[a] as usize] += f64::from(w);
            }
        }
        for (i, &lab) in labels.iter().enumerate() {
            k_c[lab as usize] += wg.strength[i];
        }
        let brute: f64 = (0..n_comms)
            .map(|c| w_c[c] / m - k_c[c] * k_c[c] / (4.0 * m * m))
            .sum();

        assert!(
            (q - brute).abs() < 1e-10,
            "reported {q} vs brute force {brute}"
        );
    }

    #[test]
    fn communities_are_connected() {
        // Same pseudo-random graph as the modularity test.
        let mut x = 0x1234_5678_u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let n = 80_usize;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if next() % 6 == 0 {
                    edges.push((i, j, 0.1 + (next() % 100) as f32 / 100.0));
                }
            }
        }
        let wg = wg_from_edges(n, &edges);
        let (labels, n_comms, _) = leiden_partition(&wg, 1.0, 7);

        // BFS from the first node of each community must reach all members.
        for c in 0..n_comms {
            let seed_node = (0..n).find(|&i| labels[i] as usize == c).unwrap();
            let mut seen = vec![false; n];
            let mut stack = vec![seed_node];
            seen[seed_node] = true;
            while let Some(v) = stack.pop() {
                let (s, e) = wg.row(v);
                for t in s..e {
                    let j = wg.adj[t] as usize;
                    if !seen[j] && labels[j] as usize == c {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            for (i, &ok) in seen.iter().enumerate() {
                if labels[i] as usize == c {
                    assert!(ok, "community {c} is disconnected at node {i}");
                }
            }
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let wg = two_cliques();
        let (a, _, _) = leiden_partition(&wg, 1.0, 3);
        let (b, _, _) = leiden_partition(&wg, 1.0, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn relabel_by_size_orders_largest_first() {
        let labels = vec![7_u32, 7, 3, 3, 3, 5];
        let (out, n) = relabel_by_size(&labels);
        assert_eq!(n, 3);
        assert_eq!(out, vec![1, 1, 0, 0, 0, 2]);
    }

    #[test]
    fn louvain_matches_leiden_on_two_cliques() {
        let wg = two_cliques();
        let lab = louvain_partition(&wg, 1.0, 0, 10, 0.0);
        assert_eq!(lab[0], lab[1]);
        assert_eq!(lab[1], lab[2]);
        assert_ne!(lab[2], lab[3]);
    }

    #[test]
    fn reduce_keeps_strengths_and_two_m() {
        let wg = two_cliques();
        let part = vec![0_u32, 0, 0, 1, 1, 1];
        let red = reduce(&wg, &part, 2);
        assert!((red.two_m - wg.two_m).abs() < 1e-9);
        assert!((red.strength[0] - 6.05).abs() < 1e-6, "{:?}", red.strength);
        assert_eq!(&*red.adj, &[1, 0]);
        assert!((red.w[0] - 0.05).abs() < 1e-7);
    }

    #[test]
    fn refinement_stays_inside_clusters() {
        let mut x = 0x9e37_79b9_u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let n = 120_usize;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if next() % 5 == 0 {
                    edges.push((i, j, 0.1 + (next() % 100) as f32 / 100.0));
                }
            }
        }
        let wg = wg_from_edges(n, &edges);
        let clusters: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (refined, n_ref) = refine(&wg, &clusters, 4, 1.0 / wg.two_m, &mut rng);
        assert!(n_ref >= 4 && n_ref <= n);
        let mut parent = vec![u32::MAX; n_ref];
        for i in 0..n {
            let r = refined[i] as usize;
            assert!(
                parent[r] == u32::MAX || parent[r] == clusters[i],
                "sub-cluster spans clusters"
            );
            parent[r] = clusters[i];
        }
    }

    #[test]
    fn modularity_never_below_singletons_and_iterations_help() {
        // On a graph with clear structure the partition must beat both the
        // singleton and the one-community partitions.
        let mut edges = Vec::new();
        for c in 0..6 {
            let base = c * 10;
            for a in 0..10 {
                for b in (a + 1)..10 {
                    edges.push((base + a, base + b, 1.0));
                }
            }
            edges.push((base, ((c + 1) % 6) * 10 + 5, 0.2));
        }
        let wg = wg_from_edges(60, &edges);
        let (labels, k, q) = leiden_partition(&wg, 1.0, 5);
        assert_eq!(k, 6, "labels {labels:?}");
        let single = modularity(&wg, &(0..60).collect::<Vec<u32>>(), 1.0);
        let one = modularity(&wg, &vec![0_u32; 60], 1.0);
        assert!(q > single && q > one, "{q} vs {single}, {one}");
    }
}
