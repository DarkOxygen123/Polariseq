#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
// Single-character names (i, j, k, a, b, c, w, q, m) follow the algorithm's
// notation in Traag et al. (2019) and igraph's implementation.

//! Leiden community detection (Traag, Waltman & van Eck 2019) on CSR.
//!
//! One Leiden *level* is fast local moving (queue-based Louvain phase 1) →
//! refinement into well-connected sub-communities → aggregation onto the
//! refined partition. Levels repeat until nothing moves and refinement
//! splits nothing. The whole procedure runs twice starting from the
//! previous partition (igraph's `community_leiden(n_iterations=2)`
//! convention), keeping the best partition found.
//!
//! Everything is arrays: the neighbour-community accumulation uses a dense
//! scratch vector plus a touched list (no hashing), aggregation is a
//! counting-sort CSR build, and one `ChaCha8` stream drives the node order
//! and the refinement choices — same graph and seed ⇒ identical output.
//!
//! The local moving, the refinement and the two-iteration scheme are
//! structured as igraph's `igraph_community_leiden` (`src/community/leiden.c`,
//! GPL-2.0-or-later, Copyright (C) 2020-2025 The igraph development team).
//! Polariseq is therefore to be distributed under a licence compatible with
//! the GPL, version 3 (see `THIRD_PARTY_NOTICES.md`).

use std::borrow::Cow;
use std::collections::VecDeque;

use crate::backend::NeighborGraph;
use crate::umap;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Safety cap on Leiden levels per run (convergence usually takes 3–6).
const MAX_LEVELS: usize = 20;

/// Outer Leiden iterations — igraph's `n_iterations=2`. Measured on
/// heart_100k: further iterations to the fixed point (~9) buy +0.0002
/// modularity for ~3x the time, and the degenerate-coarsening pass makes
/// them redundant.
const MAX_OUTER_ITERATIONS: usize = 2;

/// Softmax temperature for refinement merge choices.
const THETA: f64 = 0.01;

/// Weighted undirected graph in CSR form: both directions stored, rows
/// sorted by column, self-loops (aggregate graphs) as a single entry.
/// Sums are f64; `strength[i]` counts self-loops twice and `two_m` is
/// Σ strength, the igraph convention.
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

/// Community strengths for a membership vector, sized to the graph.
fn strengths_for(wg: &WeightedGraph<'_>, membership: &[u32]) -> Vec<f64> {
    strengths_sized(wg, membership, wg.n)
}

/// Community strengths for a membership vector, in a buffer of `cap`
/// entries (local moving may hand out any community id below the cap).
fn strengths_sized(wg: &WeightedGraph<'_>, membership: &[u32], cap: usize) -> Vec<f64> {
    let mut k = vec![0.0_f64; cap];
    for i in 0..wg.n {
        k[membership[i] as usize] += wg.strength[i];
    }
    k
}

/// Compact a membership vector to contiguous ids in first-appearance order
/// (deterministic). Returns the compacted labels and community count.
/// Labels may exceed the vector length (aggregate base memberships carry
/// phase-1 ids over a shrunken node set), so the remap is sized by the max
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

/// Hard ceiling on moves per `local_move` call, as a multiple of the vertex
/// count. The relative tolerance above removes the *known* cause of
/// non-termination; this guarantees termination regardless. Local moving is
/// a heuristic — stopping it early yields a valid partition that the outer
/// level loop can still improve — so a cap costs quality at worst, never
/// correctness. Never reached on any graph in the test suite or the
/// benchmark ladder; if it fires, something is wrong and the trace says so.
const MAX_MOVES_PER_VERTEX: usize = 100;

/// Fast local moving (Louvain phase 1), structured exactly like igraph's
/// `leiden_fastmove_vertices`: a FIFO of unstable vertices seeded with all
/// vertices in random order; a vertex that moves re-queues its stable
/// neighbours outside the new community. Every vertex ALWAYS has an empty
/// community as a candidate (diff = 0) — the escape hatch that keeps the
/// dynamics from carving fragments (measured: without it, Q varied
/// 0.918–0.960 across seeds on the 20k mixture where igraph is stable at
/// 0.9599). The vertex is removed from its community before scoring, so the
/// "stay" diff uses the community strength without it. Ties keep the
/// earlier candidate (empty first, then communities in ascending id).
/// Returns the number of moves; `membership` ids stay within
/// `n_comms_cap` (compacted by the caller).
fn local_move(
    wg: &WeightedGraph<'_>,
    membership: &mut [u32],
    n_comms_cap: usize,
    comm_strength: &mut [f64],
    resolution: f64,
    rng: &mut ChaCha8Rng,
) -> usize {
    let n = wg.n;
    let res = resolution / wg.two_m; // igraph folds 1/2m into the resolution
                                     // Scale the move threshold to the graph, not to an absolute constant:
                                     // see MOVE_REL_TOL. `two_m` is 0 only for an edgeless graph, which the
                                     // caller has already handled.
    let move_eps = MOVE_REL_TOL * wg.two_m;
    let max_moves = MAX_MOVES_PER_VERTEX.saturating_mul(n.max(1));
    // A VecDeque, not a growing Vec: `in_queue` already prevents duplicates,
    // so at most `n` vertices are pending at once, but a Vec that is only
    // ever pushed to and indexed by a moving head never reclaims consumed
    // entries — measured at 17.2 million entries (69 MB) for a 50-node
    // graph. Popping from the front keeps this O(n).
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let mut queue: VecDeque<u32> = order.into_iter().collect();
    let mut in_queue = vec![true; n];
    let mut w_to = vec![0.0_f64; n_comms_cap];
    let mut touched: Vec<u32> = Vec::new();
    let mut count = vec![0_u32; n_comms_cap];
    for &c in membership.iter() {
        count[c as usize] += 1;
    }
    let mut empty: Vec<u32> = (0..n_comms_cap as u32)
        .filter(|&c| count[c as usize] == 0)
        .collect();
    let mut moves = 0_usize;

    while let Some(i) = queue.pop_front() {
        if moves >= max_moves {
            if std::env::var_os("POLARISEQ_TRACE").is_some() {
                eprintln!(
                    "[leiden] local_move hit the move cap: n={n} moves={moves} \
                     pending={} two_m={:.3e}",
                    queue.len(),
                    wg.two_m
                );
            }
            break;
        }
        let i = i as usize;
        in_queue[i] = false;
        let (s, e) = wg.row(i);
        let ki = wg.strength[i];
        let a = membership[i] as usize;

        // Remove i from its community while scoring.
        comm_strength[a] -= ki;
        count[a] -= 1;
        if count[a] == 0 {
            empty.push(a as u32);
        }

        for t in s..e {
            let j = wg.adj[t] as usize;
            if j == i {
                continue; // self-loops cancel in the move gain (igraph's u != v)
            }
            let c = membership[j];
            if w_to[c as usize] == 0.0 {
                touched.push(c);
            }
            w_to[c as usize] += f64::from(wg.w[t]);
        }
        touched.sort_unstable();

        // Candidates: the empty community (diff 0), then the touched
        // communities in ascending id. Stay = own community without i.
        let mut best = a;
        let mut best_score = w_to[a] - res * ki * comm_strength[a];
        if !empty.is_empty() && 0.0 > best_score + move_eps {
            best = empty[empty.len() - 1] as usize;
            best_score = 0.0;
        }
        for &c in &touched {
            let c = c as usize;
            let score = w_to[c] - res * ki * comm_strength[c];
            if score > best_score + move_eps {
                best_score = score;
                best = c;
            }
        }

        // Put i back into its chosen community.
        comm_strength[best] += ki;
        count[best] += 1;
        if best != a {
            if !empty.is_empty() && empty[empty.len() - 1] as usize == best {
                empty.pop();
            }
            membership[i] = best as u32;
            moves += 1;
            for t in s..e {
                let j = wg.adj[t] as usize;
                if membership[j] != best as u32 && !in_queue[j] {
                    in_queue[j] = true;
                    queue.push_back(j as u32);
                }
            }
        }

        for &c in &touched {
            w_to[c as usize] = 0.0;
        }
        touched.clear();
    }
    moves
}

/// Leiden refinement, structured like igraph's `leiden_merge_vertices`.
/// Within each phase-1 community C, vertices start as singletons and are
/// visited in seeded random order; a still-singleton vertex v may merge
/// into a neighbouring refined sub-community S only if BOTH are well
/// connected inside C:
///
/// - v: `w(v, C∖v) ≥ γ k_v (K_C − k_v) / 2m`
/// - S: `w(S, C∖S) ≥ γ K_S (K_C − K_S) / 2m` (tracked incrementally)
///
/// The target is drawn from a softmax over the non-negative move gains
/// (`exp(Δ/θ)`, θ = 0.01 — igraph's default `beta`; if no gain is
/// non-negative the vertex stays a singleton). Merging only into a
/// neighbour's community keeps every refined community connected by
/// construction.
fn refine(
    wg: &WeightedGraph<'_>,
    phase1: &[u32],
    n_p1: usize,
    resolution: f64,
    rng: &mut ChaCha8Rng,
) -> (Vec<u32>, usize) {
    let n = wg.n;
    let res = resolution / wg.two_m;
    // Bucket nodes by phase-1 community (counting sort, deterministic).
    let mut offsets = vec![0_u32; n_p1 + 1];
    for &c in phase1 {
        offsets[c as usize + 1] += 1;
    }
    for c in 0..n_p1 {
        offsets[c + 1] += offsets[c];
    }
    let mut order: Vec<u32> = vec![0; n];
    let mut cursor = offsets.clone();
    for (i, &c) in phase1.iter().enumerate() {
        order[cursor[c as usize] as usize] = i as u32;
        cursor[c as usize] += 1;
    }

    let mut refined = vec![u32::MAX; n];
    // Per refined community: strength, size, external weight w(S, C∖S),
    // non-singleton flag. ids are assigned sequentially.
    let mut k_ref: Vec<f64> = vec![0.0; n];
    let mut size: Vec<u32> = vec![0; n];
    let mut external: Vec<f64> = vec![0.0; n];
    let mut non_singleton = vec![false; n];
    let mut next_id = 0_u32;
    let mut w_to = vec![0.0_f64; n];
    let mut touched: Vec<u32> = Vec::new();

    for c in 0..n_p1 {
        let (s, e) = (offsets[c] as usize, offsets[c + 1] as usize);
        // Singleton init: external of {v} = w(v, C∖v); K_C = Σ k_v.
        let mut k_total = 0.0_f64;
        for &v in &order[s..e] {
            let vi = v as usize;
            let id = next_id;
            next_id += 1;
            refined[vi] = id;
            k_ref[id as usize] = wg.strength[vi];
            size[id as usize] = 1;
            let (vs, ve) = wg.row(vi);
            let mut ext = 0.0_f64;
            for t in vs..ve {
                let u = wg.adj[t] as usize;
                if u != vi && phase1[u] == c as u32 {
                    ext += f64::from(wg.w[t]);
                }
            }
            external[id as usize] = ext;
            k_total += wg.strength[vi];
        }
        let mut members: Vec<u32> = order[s..e].to_vec();
        members.shuffle(rng);
        for &v in &members {
            let vi = v as usize;
            let own = refined[vi];
            if non_singleton[own as usize] {
                continue; // only singletons merge
            }
            let kv = wg.strength[vi];
            // v must be well connected to the rest of C.
            if external[own as usize] < res * kv * (k_total - kv) {
                continue;
            }
            let (vs, ve) = wg.row(vi);
            // Gains towards each neighbouring refined community in C.
            touched.clear();
            for t in vs..ve {
                let u = wg.adj[t] as usize;
                if u == vi || phase1[u] != c as u32 {
                    continue;
                }
                let r = refined[u];
                if w_to[r as usize] == 0.0 {
                    touched.push(r);
                }
                w_to[r as usize] += f64::from(wg.w[t]);
            }
            // Softmax over non-negative gains of well-connected targets;
            // shift by the max so exp cannot overflow (igraph relies on
            // IEEE inf arithmetic; shifting is equivalent and portable).
            let mut cands: Vec<(u32, f64)> = Vec::with_capacity(touched.len());
            let mut dmax = f64::NEG_INFINITY;
            for &r in &touched {
                let ru = r as usize;
                let gain = w_to[ru] - res * kv * k_ref[ru];
                if gain >= 0.0 && external[ru] >= res * k_ref[ru] * (k_total - k_ref[ru]) {
                    cands.push((r, gain));
                    dmax = dmax.max(gain);
                }
                w_to[ru] = 0.0;
            }
            if !cands.is_empty() {
                let weights: Vec<f64> = cands
                    .iter()
                    .map(|&(_, g)| ((g - dmax) / THETA).exp())
                    .collect();
                let total: f64 = weights.iter().sum();
                let mut pick = rng.random::<f64>() * total;
                let mut chosen = cands[0].0;
                for (&(r, _), &wt) in cands.iter().zip(&weights) {
                    pick -= wt;
                    if pick <= 0.0 {
                        chosen = r;
                        break;
                    }
                }
                // Merge v into `chosen`: edges to its members become
                // internal, edges to the rest of C become external.
                let cu = chosen as usize;
                refined[vi] = chosen;
                k_ref[cu] += kv;
                size[cu] += 1;
                non_singleton[cu] = true;
                for t in vs..ve {
                    let u = wg.adj[t] as usize;
                    if u != vi && phase1[u] == c as u32 {
                        if refined[u] == chosen {
                            external[cu] -= f64::from(wg.w[t]);
                        } else {
                            external[cu] += f64::from(wg.w[t]);
                        }
                    }
                }
            }
        }
    }
    // Renumber to contiguous ids (dead singleton ids may remain).
    let (refined_c, n_ref) = compact(&refined);
    (refined_c, n_ref)
}

/// Aggregate onto the refined partition: nodes = refined communities,
/// weights summed between them, internal weight as a self-loop. Counting
/// sort + per-row sort/merge of duplicates (parallel over rows, ordered).
/// The phase-1 partition becomes the initial membership of the aggregate
/// nodes. Returns the quotient graph and that initial membership.
fn aggregate(
    wg: &WeightedGraph<'_>,
    refined: &[u32],
    n_ref: usize,
    phase1: &[u32],
) -> (WeightedGraph<'static>, Vec<u32>) {
    // Initial membership: any member's phase-1 label (they all agree).
    let mut base = vec![u32::MAX; n_ref];
    for (i, &r) in refined.iter().enumerate() {
        if base[r as usize] == u32::MAX {
            base[r as usize] = phase1[i];
        }
    }

    // Counting-sort scatter of every directed entry into its refined row.
    let mut indptr = vec![0_u32; n_ref + 1];
    for i in 0..wg.n {
        let (s, e) = wg.row(i);
        indptr[refined[i] as usize + 1] += (e - s) as u32;
    }
    for r in 0..n_ref {
        indptr[r + 1] += indptr[r];
    }
    let nnz = indptr[n_ref] as usize;
    let mut adj = vec![0_u32; nnz];
    let mut w = vec![0.0_f32; nnz];
    let mut cursor = indptr.clone();
    for i in 0..wg.n {
        let (s, e) = wg.row(i);
        let a = refined[i] as usize;
        for t in s..e {
            let pos = cursor[a] as usize;
            adj[pos] = refined[wg.adj[t] as usize];
            w[pos] = wg.w[t];
            cursor[a] += 1;
        }
    }

    // Sort + merge duplicates within each row (deterministic, f64 sums).
    let rows: Vec<Vec<(u32, f32)>> = (0..n_ref)
        .into_par_iter()
        .map(|r| {
            let (s, e) = (indptr[r] as usize, indptr[r + 1] as usize);
            let mut row: Vec<(u32, f32)> = (s..e).map(|t| (adj[t], w[t])).collect();
            row.sort_unstable_by_key(|&(c, _)| c);
            let mut merged: Vec<(u32, f32)> = Vec::with_capacity(row.len());
            for &(col, wt) in &row {
                match merged.last_mut() {
                    Some(last) if last.0 == col => {
                        last.1 = (f64::from(last.1) + f64::from(wt)) as f32;
                    }
                    _ => merged.push((col, wt)),
                }
            }
            merged
        })
        .collect();

    let mut new_indptr = vec![0_u32; n_ref + 1];
    for (r, row) in rows.iter().enumerate() {
        new_indptr[r + 1] = new_indptr[r] + row.len() as u32;
    }
    let mut new_adj = vec![0_u32; new_indptr[n_ref] as usize];
    let mut new_w = vec![0.0_f32; new_indptr[n_ref] as usize];
    for r in 0..n_ref {
        let s = new_indptr[r] as usize;
        for (t, &(col, wt)) in rows[r].iter().enumerate() {
            new_adj[s + t] = col;
            new_w[s + t] = wt;
        }
    }

    (
        WeightedGraph::from_symmetric_csr(new_indptr, new_adj, new_w, n_ref),
        base,
    )
}

/// Modularity with resolution γ on the weighted graph, in the per-community
/// form `Q = Σ_c [W_c/m − γ K_c²/(4m²)]` (algebraically identical to
/// igraph's `(1/2m) Σ_ij [w_ij − γ k_i k_j / 2m] δ(c_i, c_j)`, which sums
/// the degree term over *all* pairs within a community, not just edges).
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

/// One full Leiden run over the levels, starting from `start` (or
/// singletons). Returns the labels on the *original* graph nodes.
/// Level structure per igraph: local moving; stop when the partition has
/// one community per vertex; otherwise refine and aggregate — and when
/// refinement merges nothing, aggregate on the phase-1 partition anyway
/// (the pass that lets whole fragments merge on the quotient graph).
fn run_levels(
    wg: &WeightedGraph<'_>,
    start: Option<Vec<u32>>,
    resolution: f64,
    rng: &mut ChaCha8Rng,
) -> (Vec<u32>, usize) {
    let n = wg.n;
    // Borrow the input for the first level; own only the (smaller)
    // aggregated graphs. A clone here doubled the largest allocation of the
    // whole clustering: ~0.84 GB at 4M cells.
    let mut graph: Cow<'_, WeightedGraph<'_>> = Cow::Borrowed(wg);
    let mut membership: Vec<u32> = start.unwrap_or_else(|| (0..n as u32).collect());
    let mut comm_strength = strengths_for(&graph, &membership);
    let mut mapping: Vec<u32> = (0..n as u32).collect();
    let mut n_comms;

    for _level in 0..MAX_LEVELS {
        local_move(
            &graph,
            &mut membership,
            graph.n,
            &mut comm_strength,
            resolution,
            rng,
        );
        let (m_c, nc) = compact(&membership);
        membership = m_c;
        n_comms = nc;

        if n_comms >= graph.n {
            break; // fully contracted: one community per vertex
        }

        let (refined_raw, n_ref) = refine(&graph, &membership, n_comms, resolution, rng);
        let (refined, n_ref_eff) = if n_ref >= graph.n {
            (membership.clone(), n_comms)
        } else {
            (refined_raw, n_ref)
        };
        if n_ref_eff == graph.n {
            break; // nothing to aggregate
        }

        let (new_graph, init) = aggregate(&graph, &refined, n_ref_eff, &membership);
        for m in &mut mapping {
            *m = refined[*m as usize];
        }
        graph = Cow::Owned(new_graph);
        let (init_c, _n_init) = compact(&init);
        membership = init_c;
        comm_strength = strengths_for(&graph, &membership);
    }

    let mut labels: Vec<u32> = mapping.iter().map(|&m| membership[m as usize]).collect();
    let (labels_c, n_out) = compact(&labels);
    labels = labels_c;
    (labels, n_out)
}

/// Full Leiden: run the level procedure `MAX_OUTER_ITERATIONS` times
/// (igraph's `n_iterations=2`), each run starting from the previous
/// partition and keeping the best by modularity — then collapse
/// modularity-degenerate micro-communities with [`coarsen_degenerate`].
/// Returns labels, community count, modularity.
pub(crate) fn leiden_partition(
    wg: &WeightedGraph<'_>,
    resolution: f64,
    seed: u64,
) -> (Vec<u32>, usize, f64) {
    if wg.two_m == 0.0 || wg.n == 0 {
        return (vec![0_u32; wg.n], wg.n.min(1), 0.0);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut best: Option<(Vec<u32>, usize, f64)> = None;
    let mut start: Option<Vec<u32>> = None;
    for it in 0..MAX_OUTER_ITERATIONS {
        let (labels, n_comms) = run_levels(wg, start.clone(), resolution, &mut rng);
        let q = modularity(wg, &labels, resolution);
        if std::env::var_os("POLARISEQ_TRACE").is_some() {
            eprintln!("[leiden] outer {it}: k={n_comms} Q={q:.6}");
        }
        let improved = best.as_ref().is_none_or(|b| q > b.2 + 1e-12);
        if improved {
            best = Some((labels.clone(), n_comms, q));
        }
        if let Some(s) = &start {
            if *s == labels {
                break; // fixed point; further iterations are no-ops
            }
        }
        start = Some(labels);
    }
    let (labels, n_comms, _q) = best.unwrap_or_else(|| (vec![0_u32; wg.n], 1, 0.0));
    let (labels, n_comms) = coarsen_degenerate(wg, labels, n_comms, resolution);
    let q = modularity(wg, &labels, resolution);
    (labels, n_comms, q)
}

/// Collapse modularity-degenerate communities. On real graphs the optimum
/// is surrounded by a flat region: merging or keeping 100-1000-node
/// fragments changes Q by ~1e-6 (measured on heart_100k: twelve sub-1000
/// clusters whose best merges were dQ ∈ [−5e-6, +3.4e-4]), so the level
/// dynamics — which only make strictly-improving moves — strand them while
/// igraph's trajectory happens to land on the coarse side. This pass
/// greedily merges the best community pair while its closed-form gain
/// `dQ = w(a,b)/m − γ K_a K_b / (2m²)` exceeds `−ε` (ε = 1e-4·|Q|), each
/// merge costing at most ε of modularity. Deterministic; k is small, so
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

/// Multi-level Louvain on the same machinery (local moving + aggregation,
/// no refinement) — the engine behind `fast_community`.
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
    let mut graph: Cow<'_, WeightedGraph<'_>> = Cow::Borrowed(wg);
    let mut membership: Vec<u32> = (0..n as u32).collect();
    let mut comm_strength = strengths_for(&graph, &membership);
    let mut mapping: Vec<u32> = (0..n as u32).collect();

    for _level in 0..max_levels.max(1) {
        let moves = local_move(
            &graph,
            &mut membership,
            graph.n,
            &mut comm_strength,
            resolution,
            &mut rng,
        );
        let (m_c, n_comms) = compact(&membership);
        membership = m_c;
        if moves == 0
            || n_comms >= graph.n
            || (moves as f64) < convergence_fraction * graph.n as f64
        {
            break;
        }
        // Aggregate onto the partition itself: the initial membership of
        // aggregate nodes is the partition.
        let (new_graph, init) = aggregate(&graph, &membership, n_comms, &membership);
        if new_graph.n == graph.n {
            break; // nothing merged; next level would repeat
        }
        for m in &mut mapping {
            *m = membership[*m as usize];
        }
        graph = Cow::Owned(new_graph);
        let (init_c, _n_init) = compact(&init);
        membership = init_c;
        comm_strength = strengths_for(&graph, &membership);
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
    fn local_move_converges_on_a_symmetric_clique() {
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
        let mut comm_strength = strengths_for(&wg, &membership);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let moves = local_move(&wg, &mut membership, n, &mut comm_strength, 1.0, &mut rng);

        let cap = MAX_MOVES_PER_VERTEX * n;
        assert!(
            moves < cap,
            "local_move hit its safety cap ({moves} >= {cap}) — it is \
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
}
