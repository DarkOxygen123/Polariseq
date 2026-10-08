//! Approximate kNN via NN-Descent (Dong, Moses & Li 2011), built the way
//! pynndescent does it:
//!
//! 1. **Random-projection-tree forest initialisation.** Each tree splits the
//!    data with random hyperplanes down to leaves of ~`k` points; every leaf
//!    contributes its all-pairs distances. This puts every point next to
//!    genuinely close points before the first iteration — far better than a
//!    random start.
//! 2. **Local join with new/old flags.** Each neighbour is `new` until it has
//!    been joined once; each iteration samples `max_candidates` new and old
//!    neighbours per node — *including reverse neighbours* — and only computes
//!    new×new and new×old pairs. Pairs are pruned against the current worst
//!    distance of both endpoints before the distance is computed.
//! 3. **Bounded heaps, bounded memory.** One fixed-size max-heap per node
//!    (index, distance, flag) — no hash sets, no per-node allocation per
//!    iteration. Candidate generation and joins run over blocks of nodes,
//!    emitting updates into buckets by destination range that are applied in
//!    parallel with exclusive row access. Memory per iteration is bounded by
//!    the block size, not by `n`.
//! 4. **Deterministic.** All randomness comes from `splitmix(seed, ...)`
//!    keyed by node/slot/iteration, and updates are applied in block order,
//!    so the graph is identical whatever the thread count.
//! 5. Early termination when fewer than `delta · n · k` heap entries changed.
//!
//! The tree initialisation and the local join follow PyNNDescent
//! (BSD-2-Clause, Copyright (c) 2018 Leland McInnes), whose notice
//! `THIRD_PARTY_NOTICES.md` keeps.

#![allow(
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

use crate::backend::NeighborGraph;
use crate::simd;
use rayon::prelude::*;

/// Options for NN-Descent. Zero means "auto" for the sized fields.
#[derive(Debug, Clone)]
pub struct NndescentOptions {
    /// Number of nearest neighbours per node.
    pub k: usize,
    /// New/old candidates sampled per node per iteration (0 → `min(60, 2k)`).
    pub max_candidates: usize,
    /// Maximum iterations (0 → `max(5, round(log2 n))`).
    pub max_iterations: usize,
    /// Stop when fewer than `delta · n · k` neighbour entries changed.
    pub delta: f64,
    /// Random-projection trees for initialisation (0 → `min(48, 8 + √n/10)`).
    pub n_trees: usize,
    /// Leaf size of the trees (0 → `max(10, k)`).
    pub leaf_size: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for NndescentOptions {
    fn default() -> Self {
        Self {
            k: 15,
            max_candidates: 0,
            max_iterations: 0,
            delta: 0.001,
            n_trees: 0,
            leaf_size: 0,
            seed: 0,
        }
    }
}

// ── deterministic hashing RNG ────────────────────────────────────────────

#[inline]
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `f32` in `[0, 1)` from a keyed hash — reproducible and
/// thread-independent.
#[inline]
fn hash_unit(seed: u64, a: u64, b: u64) -> f32 {
    let h = splitmix(seed ^ splitmix(a) ^ splitmix(b.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    (h >> 40) as f32 / (1u64 << 24) as f32
}

#[inline]
fn hash_index(seed: u64, a: u64, b: u64, n: usize) -> usize {
    (splitmix(seed ^ splitmix(a) ^ splitmix(b.wrapping_add(0x1234_5678))) % n as u64) as usize
}

/// A small, stateful PRNG for the tree builder (one per tree).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = splitmix(self.0);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ── fixed-size flagged max-heaps, one row per node ───────────────────────

const EMPTY: u32 = u32::MAX;

/// `n` rows of `k` `(index, distance, flag)` entries, each row a max-heap on
/// distance (root = worst neighbour so far).
struct NeighborHeap {
    k: usize,
    idx: Vec<u32>,
    dist: Vec<f32>,
    flag: Vec<bool>,
}

impl NeighborHeap {
    fn new(n: usize, k: usize) -> Self {
        Self {
            k,
            idx: vec![EMPTY; n * k],
            dist: vec![f32::INFINITY; n * k],
            flag: vec![false; n * k],
        }
    }

    #[inline]
    fn worst(&self, i: usize) -> f32 {
        self.dist[i * self.k]
    }

    /// Row views for the bucketed parallel apply.
    fn rows_mut(
        &mut self,
        rows_per_bucket: usize,
    ) -> impl IndexedParallelIterator<Item = HeapRowsMut<'_>> {
        let k = self.k;
        let stride = rows_per_bucket * k;
        self.idx
            .par_chunks_mut(stride)
            .zip(self.dist.par_chunks_mut(stride))
            .zip(self.flag.par_chunks_mut(stride))
            .map(move |((idx, dist), flag)| HeapRowsMut { k, idx, dist, flag })
    }
}

/// A contiguous block of heap rows with exclusive access.
struct HeapRowsMut<'a> {
    k: usize,
    idx: &'a mut [u32],
    dist: &'a mut [f32],
    flag: &'a mut [bool],
}

impl HeapRowsMut<'_> {
    /// Insert `(j, d)` into local row `r` if it beats the worst entry and is
    /// not already present. Returns whether the row changed.
    fn push(&mut self, r: usize, j: u32, d: f32, flag: bool) -> bool {
        let k = self.k;
        let base = r * k;
        let (idx, dist, flg) = (
            &mut self.idx[base..base + k],
            &mut self.dist[base..base + k],
            &mut self.flag[base..base + k],
        );
        if d >= dist[0] {
            return false;
        }
        if idx.contains(&j) {
            return false;
        }
        // Replace the root and sift down (max-heap on dist).
        idx[0] = j;
        dist[0] = d;
        flg[0] = flag;
        let mut p = 0;
        loop {
            let (l, rr) = (2 * p + 1, 2 * p + 2);
            let mut swap = p;
            if l < k && dist[l] > dist[swap] {
                swap = l;
            }
            if rr < k && dist[rr] > dist[swap] {
                swap = rr;
            }
            if swap == p {
                break;
            }
            idx.swap(p, swap);
            dist.swap(p, swap);
            flg.swap(p, swap);
            p = swap;
        }
        true
    }
}

/// `n` rows of `m` `(priority, index)` entries, a max-heap on priority: keeps
/// the `m` candidates with the smallest random priorities — a uniform sample.
struct CandidateHeap {
    m: usize,
    idx: Vec<u32>,
    prio: Vec<f32>,
}

impl CandidateHeap {
    fn new(n: usize, m: usize) -> Self {
        Self {
            m,
            idx: vec![EMPTY; n * m],
            prio: vec![f32::INFINITY; n * m],
        }
    }

    fn reset(&mut self) {
        self.idx.par_iter_mut().for_each(|x| *x = EMPTY);
        self.prio.par_iter_mut().for_each(|x| *x = f32::INFINITY);
    }

    fn row(&self, i: usize) -> &[u32] {
        &self.idx[i * self.m..(i + 1) * self.m]
    }

    fn rows_mut(
        &mut self,
        rows_per_bucket: usize,
    ) -> impl IndexedParallelIterator<Item = (&mut [u32], &mut [f32])> {
        let stride = rows_per_bucket * self.m;
        self.idx
            .par_chunks_mut(stride)
            .zip(self.prio.par_chunks_mut(stride))
    }

    fn push_row(m: usize, idx: &mut [u32], prio: &mut [f32], r: usize, j: u32, p: f32) {
        let base = r * m;
        let (idx, prio) = (&mut idx[base..base + m], &mut prio[base..base + m]);
        if p >= prio[0] || idx.contains(&j) {
            return;
        }
        idx[0] = j;
        prio[0] = p;
        let mut q = 0;
        loop {
            let (l, rr) = (2 * q + 1, 2 * q + 2);
            let mut swap = q;
            if l < m && prio[l] > prio[swap] {
                swap = l;
            }
            if rr < m && prio[rr] > prio[swap] {
                swap = rr;
            }
            if swap == q {
                break;
            }
            idx.swap(q, swap);
            prio.swap(q, swap);
            q = swap;
        }
    }
}

// ── bucketed updates ─────────────────────────────────────────────────────

/// Destination-range buckets: bucket `b` owns rows `[b·rows, (b+1)·rows)`.
const N_BUCKETS: usize = 64;
/// Nodes whose candidates/joins are generated before an apply pass. Bounds
/// the update buffers: 4096 nodes × ~1300 candidate pairs × 2 × 12 B ≈ 125 MB
/// worst case (early iterations emit most pairs).
const BLOCK: usize = 4096;
/// Nodes per parallel generation task inside a block.
const GEN_CHUNK: usize = 512;

/// A neighbour candidate for `dest`; every push made through updates is
/// flagged `new`, so no flag field is carried (12 bytes, not 16).
#[derive(Clone, Copy)]
struct Update {
    dest: u32,
    nbr: u32,
    dist: f32,
}

#[derive(Clone, Copy)]
struct CandUpdate {
    dest: u32,
    nbr: u32,
    prio: f32,
    is_new: bool,
}

fn rows_per_bucket(n: usize) -> usize {
    n.div_ceil(N_BUCKETS).max(1)
}

/// Apply neighbour updates: `gen[chunk][bucket]` is consumed in chunk order
/// per bucket, each bucket by one task with exclusive rows. Returns the
/// number of heap entries that changed.
fn apply_updates(heap: &mut NeighborHeap, gen: &[Vec<Vec<Update>>], n: usize) -> usize {
    let rpb = rows_per_bucket(n);
    heap.rows_mut(rpb)
        .enumerate()
        .map(|(b, mut rows)| {
            let base = b * rpb;
            let mut changed = 0;
            for chunk in gen {
                for u in &chunk[b] {
                    if rows.push(u.dest as usize - base, u.nbr, u.dist, true) {
                        changed += 1;
                    }
                }
            }
            changed
        })
        .sum()
}

fn apply_candidates(
    cands: &mut CandidateHeap,
    gen: &[Vec<Vec<CandUpdate>>],
    n: usize,
    want_new: bool,
) {
    let rpb = rows_per_bucket(n);
    let m = cands.m;
    cands
        .rows_mut(rpb)
        .enumerate()
        .for_each(|(b, (idx, prio))| {
            let base = b * rpb;
            for chunk in gen {
                for u in &chunk[b] {
                    if u.is_new == want_new {
                        CandidateHeap::push_row(
                            m,
                            idx,
                            prio,
                            u.dest as usize - base,
                            u.nbr,
                            u.prio,
                        );
                    }
                }
            }
        });
}

// ── random projection trees ──────────────────────────────────────────────

/// A built tree: `ids` is a permutation of the input indices and `leaves`
/// are `[start, end)` ranges into it. No per-leaf allocation.
struct Tree {
    ids: Vec<u32>,
    leaves: Vec<(u32, u32)>,
}

/// Choose a random hyperplane through two distinct points of `ids`.
/// Returns `(normal, offset)`.
fn random_hyperplane(data: &[f32], d: usize, ids: &[u32], rng: &mut Rng) -> (Vec<f32>, f32) {
    let n = ids.len();
    let a = ids[rng.below(n)] as usize;
    let mut b = ids[rng.below(n)] as usize;
    let mut tries = 0;
    while b == a && tries < 8 {
        b = ids[rng.below(n)] as usize;
        tries += 1;
    }
    let (xa, xb) = (&data[a * d..(a + 1) * d], &data[b * d..(b + 1) * d]);
    let normal: Vec<f32> = xa.iter().zip(xb).map(|(p, q)| p - q).collect();
    let offset: f32 = -xa
        .iter()
        .zip(xb)
        .zip(&normal)
        .map(|((p, q), nv)| nv * 0.5 * (p + q))
        .sum::<f32>();
    (normal, offset)
}

/// Partition `ids` in place by `sides[i]` (true = left); returns the split
/// point. Stable order is irrelevant for a tree.
fn partition_in_place(ids: &mut [u32], sides: &[bool]) -> usize {
    let mut lo = 0;
    let mut hi = ids.len();
    // Two-pointer swap partition; `sides` is indexed by the *original*
    // position, so track positions through the swaps with a parallel array.
    let mut pos: Vec<u32> = (0..ids.len() as u32).collect();
    while lo < hi {
        if sides[pos[lo] as usize] {
            lo += 1;
        } else {
            hi -= 1;
            ids.swap(lo, hi);
            pos.swap(lo, hi);
        }
    }
    lo
}

/// Random split of a slice that could not be separated by a hyperplane.
fn random_split(ids: &mut [u32], rng: &mut Rng) -> usize {
    // Fisher–Yates on the slice, then cut in half.
    for i in (1..ids.len()).rev() {
        let j = rng.below(i + 1);
        ids.swap(i, j);
    }
    ids.len() / 2
}

/// Sequential recursion on a slice of ids, writing leaves as ranges
/// relative to `base`.
fn rp_tree_leaves(
    data: &[f32],
    d: usize,
    ids: &mut [u32],
    base: usize,
    leaf_size: usize,
    rng: &mut Rng,
    depth: usize,
    out: &mut Vec<(u32, u32)>,
) {
    let n = ids.len();
    if n <= leaf_size || depth > 100 {
        out.push((base as u32, (base + n) as u32));
        return;
    }
    let (normal, offset) = random_hyperplane(data, d, ids, rng);
    let sides: Vec<bool> = ids
        .iter()
        .map(|&i| {
            let x = &data[i as usize * d..(i as usize + 1) * d];
            let margin = simd::dot(&normal, x) + offset;
            if margin.abs() < 1e-8 {
                rng.next_u64() & 1 == 0
            } else {
                margin > 0.0
            }
        })
        .collect();
    let mut split = partition_in_place(ids, &sides);
    if split == 0 || split == n {
        split = random_split(ids, rng);
        if split == 0 || split == n {
            out.push((base as u32, (base + n) as u32));
            return;
        }
    }
    let (left, right) = ids.split_at_mut(split);
    rp_tree_leaves(data, d, left, base, leaf_size, rng, depth + 1, out);
    rp_tree_leaves(data, d, right, base + split, leaf_size, rng, depth + 1, out);
}

/// Parallel recursion for the top levels (rayon `join`); below ~8k points
/// the subtree's rows are gathered into a contiguous buffer (L2-resident)
/// and finished sequentially — the levels below then read memory
/// sequentially instead of missing cache on every point.
fn rp_tree_parallel(
    data: &[f32],
    d: usize,
    ids: &mut [u32],
    base: usize,
    leaf_size: usize,
    seed: u64,
    depth: usize,
) -> Vec<(u32, u32)> {
    let n = ids.len();
    if n < 8192 || depth >= 6 {
        let mut local = Vec::with_capacity(n * d);
        for &i in ids.iter() {
            local.extend_from_slice(&data[i as usize * d..(i as usize + 1) * d]);
        }
        let mut local_ids: Vec<u32> = (0..n as u32).collect();
        let mut out = Vec::new();
        rp_tree_leaves(
            &local,
            d,
            &mut local_ids,
            base,
            leaf_size,
            &mut Rng(seed),
            depth,
            &mut out,
        );
        // `local_ids` is now the permutation; apply it to `ids`.
        let snapshot: Vec<u32> = ids.to_vec();
        for (slot, &li) in ids.iter_mut().zip(&local_ids) {
            *slot = snapshot[li as usize];
        }
        return out;
    }
    let mut rng = Rng(seed);
    let (normal, offset) = random_hyperplane(data, d, ids, &mut rng);
    let sides: Vec<bool> = ids
        .par_iter()
        .map(|&i| {
            let x = &data[i as usize * d..(i as usize + 1) * d];
            let margin = simd::dot(&normal, x) + offset;
            if margin.abs() < 1e-8 {
                splitmix(seed ^ u64::from(i)) & 1 == 0
            } else {
                margin > 0.0
            }
        })
        .collect();
    let mut split = partition_in_place(ids, &sides);
    if split == 0 || split == n {
        split = random_split(ids, &mut rng);
    }
    let (left, right) = ids.split_at_mut(split);
    let (mut l, r) = rayon::join(
        || {
            rp_tree_parallel(
                data,
                d,
                left,
                base,
                leaf_size,
                splitmix(seed ^ 1),
                depth + 1,
            )
        },
        || {
            rp_tree_parallel(
                data,
                d,
                right,
                base + split,
                leaf_size,
                splitmix(seed ^ 2),
                depth + 1,
            )
        },
    );
    l.extend(r);
    l
}

fn build_tree(data: &[f32], d: usize, n: usize, leaf_size: usize, seed: u64) -> Tree {
    let mut ids: Vec<u32> = (0..n as u32).collect();
    let leaves = rp_tree_parallel(data, d, &mut ids, 0, leaf_size, seed, 0);
    Tree { ids, leaves }
}

// ── the algorithm ────────────────────────────────────────────────────────

/// Neighbours per node that [`nndescent`] returns for `n_obs` points: `k`,
/// or fewer when there are not `k` other points.
#[must_use]
pub fn effective_k(n_obs: usize, k: usize) -> usize {
    k.min(n_obs.saturating_sub(1)).max(1)
}

/// Build an approximate kNN graph via NN-Descent. Distances are Euclidean.
///
/// # Panics
/// Panics if `data` is empty or `k == 0`.
#[must_use]
pub fn nndescent(
    data: &[f32],
    n_obs: usize,
    n_dim: usize,
    opts: &NndescentOptions,
) -> NeighborGraph {
    assert!(n_obs > 0 && n_dim > 0, "empty data");
    assert!(opts.k > 0, "k must be >= 1");
    let k = effective_k(n_obs, opts.k);
    let mut indptr = vec![0_u32; n_obs + 1];
    let mut indices = vec![0_u32; n_obs * k];
    let mut dists = vec![0.0_f32; n_obs * k];
    let nnz = nndescent_into(
        data,
        n_obs,
        n_dim,
        opts,
        &mut indptr,
        &mut indices,
        &mut dists,
    );
    indices.truncate(nnz);
    dists.truncate(nnz);
    NeighborGraph {
        indptr,
        indices,
        data: dists,
        n_obs,
    }
}

/// [`nndescent`] with the graph written into the caller's arrays, which may be
/// memory-mapped files: `indptr` (`n_obs + 1`), and `indices` and `dists`
/// (`n_obs × effective_k(n_obs, k)` each, the most a graph can hold). Returns
/// the number of entries written, which is that size unless some point found
/// fewer than `k` neighbours; entries past it are untouched.
///
/// The search itself holds its working set in memory (the neighbour heaps, the
/// candidate lists); only the finished graph goes to the arrays.
///
/// # Panics
/// Panics if `data` is empty, `k == 0`, or an output array is too short.
pub fn nndescent_into(
    data: &[f32],
    n_obs: usize,
    n_dim: usize,
    opts: &NndescentOptions,
    indptr_out: &mut [u32],
    indices_out: &mut [u32],
    dists_out: &mut [f32],
) -> usize {
    assert!(n_obs > 0 && n_dim > 0, "empty data");
    assert!(opts.k > 0, "k must be >= 1");
    assert_eq!(
        indptr_out.len(),
        n_obs + 1,
        "indptr must have n_obs + 1 entries"
    );
    let cap = n_obs * effective_k(n_obs, opts.k);
    assert!(
        indices_out.len() >= cap && dists_out.len() >= cap,
        "the graph arrays are too short"
    );
    let n = n_obs;
    let d = n_dim;
    let k = opts.k.min(n - 1).max(1);
    // Auto defaults measured on heart_100k PCA embeddings (float64 exact
    // reference): 2k candidates / ~40 trees give recall@15 0.978 (p05 0.87)
    // in 1.2 s vs 0.957 (p05 0.80) at pynndescent's k / 21 trees.
    let max_cand = if opts.max_candidates > 0 {
        opts.max_candidates
    } else {
        (2 * k).min(60)
    }
    .max(1);
    let n_iters = if opts.max_iterations > 0 {
        opts.max_iterations
    } else {
        ((n as f64).log2().round() as usize).max(5)
    };
    let n_trees = if opts.n_trees > 0 {
        opts.n_trees
    } else {
        (8 + ((n as f64).sqrt() / 10.0).round() as usize).min(48)
    };
    let leaf_size = if opts.leaf_size > 0 {
        opts.leaf_size
    } else {
        k.max(10)
    };
    let seed = opts.seed;
    let rpb = rows_per_bucket(n);
    let row = |i: usize| &data[i * d..(i + 1) * d];

    let mut heap = NeighborHeap::new(n, k);
    let trace = std::env::var_os("POLARISEQ_TRACE").is_some();
    let t0 = std::time::Instant::now();
    let mut lap = t0;
    let tick = |label: &str, lap: &mut std::time::Instant| {
        if trace {
            eprintln!("[nndescent] {label}: {:.2}s", lap.elapsed().as_secs_f64());
            *lap = std::time::Instant::now();
        }
    };

    // ── 1. RP-tree forest initialisation ──────────────────────────────
    let forest: Vec<Tree> = (0..n_trees)
        .into_par_iter()
        .map(|t| build_tree(data, d, n, leaf_size, splitmix(seed ^ (t as u64 + 1))))
        .collect();
    tick(
        &format!("rp-forest build ({n_trees} trees, leaf {leaf_size})"),
        &mut lap,
    );
    for tree in &forest {
        // All pairs within each leaf → bucketed updates → apply.
        let ids = &tree.ids;
        let gen: Vec<Vec<Vec<Update>>> = tree
            .leaves
            .par_chunks(64)
            .map(|leaf_chunk| {
                let mut buckets: Vec<Vec<Update>> = (0..N_BUCKETS).map(|_| Vec::new()).collect();
                for &(a, b) in leaf_chunk {
                    let leaf = &ids[a as usize..b as usize];
                    for (i, &p) in leaf.iter().enumerate() {
                        for &q in &leaf[i + 1..] {
                            let dist = simd::sq_dist(row(p as usize), row(q as usize));
                            buckets[p as usize / rpb].push(Update {
                                dest: p,
                                nbr: q,
                                dist,
                            });
                            buckets[q as usize / rpb].push(Update {
                                dest: q,
                                nbr: p,
                                dist,
                            });
                        }
                    }
                }
                buckets
            })
            .collect();
        apply_updates(&mut heap, &gen, n);
    }
    drop(forest);
    tick("rp-forest leaf pairs + apply", &mut lap);

    // Fill any remaining empty slots with random neighbours (own row only).
    {
        let kk = k;
        heap.idx
            .par_chunks_mut(kk)
            .zip(heap.dist.par_chunks_mut(kk))
            .zip(heap.flag.par_chunks_mut(kk))
            .enumerate()
            .for_each(|(i, ((idx, dist), flag))| {
                let mut rows = HeapRowsMut {
                    k: kk,
                    idx,
                    dist,
                    flag,
                };
                let mut attempt = 0u64;
                while rows.idx.contains(&EMPTY) && attempt < 64 {
                    let mut j = hash_index(seed, i as u64, attempt, n);
                    if j == i {
                        j = (j + 1) % n;
                    }
                    let dist = simd::sq_dist(row(i), row(j));
                    rows.push(0, j as u32, dist, true);
                    attempt += 1;
                }
            });
    }

    // ── 2. NN-Descent iterations ──────────────────────────────────────
    let mut new_cand = CandidateHeap::new(n, max_cand);
    let mut old_cand = CandidateHeap::new(n, max_cand);
    let stop_at = (opts.delta * n as f64 * k as f64) as usize;

    tick("random fill", &mut lap);
    for iter in 0..n_iters {
        // 2a. Candidate sampling (forward + reverse) with random priorities.
        new_cand.reset();
        old_cand.reset();
        for block_start in (0..n).step_by(BLOCK) {
            let block_end = (block_start + BLOCK).min(n);
            let gen: Vec<Vec<Vec<CandUpdate>>> = (block_start..block_end)
                .into_par_iter()
                .step_by(GEN_CHUNK)
                .map(|c0| {
                    let mut buckets: Vec<Vec<CandUpdate>> =
                        (0..N_BUCKETS).map(|_| Vec::new()).collect();
                    for i in c0..(c0 + GEN_CHUNK).min(block_end) {
                        for s in 0..k {
                            let j = heap.idx[i * k + s];
                            if j == EMPTY {
                                continue;
                            }
                            let is_new = heap.flag[i * k + s];
                            let prio = hash_unit(seed, iter as u64, (i * k + s) as u64);
                            buckets[i / rpb].push(CandUpdate {
                                dest: i as u32,
                                nbr: j,
                                prio,
                                is_new,
                            });
                            buckets[j as usize / rpb].push(CandUpdate {
                                dest: j,
                                nbr: i as u32,
                                prio,
                                is_new,
                            });
                        }
                    }
                    buckets
                })
                .collect();
            apply_candidates(&mut new_cand, &gen, n, true);
            apply_candidates(&mut old_cand, &gen, n, false);
        }

        // 2b. Neighbours that made it into a node's *new* list are now old.
        {
            let nc = &new_cand;
            heap.idx
                .par_chunks(k)
                .zip(heap.flag.par_chunks_mut(k))
                .enumerate()
                .for_each(|(i, (idx, flag))| {
                    let cands = nc.row(i);
                    for s in 0..k {
                        if flag[s] && idx[s] != EMPTY && cands.contains(&idx[s]) {
                            flag[s] = false;
                        }
                    }
                });
        }

        tick(&format!("iter {iter}: candidates"), &mut lap);
        // 2c. Local join: new×new and new×old, pruned by both endpoints'
        //     current worst distance; applied per block.
        let thresholds: Vec<f32> = (0..n).map(|i| heap.worst(i)).collect();
        let mut changed = 0usize;
        for block_start in (0..n).step_by(BLOCK) {
            let block_end = (block_start + BLOCK).min(n);
            let (nc, oc, thr) = (&new_cand, &old_cand, &thresholds);
            let gen: Vec<Vec<Vec<Update>>> = (block_start..block_end)
                .into_par_iter()
                .step_by(GEN_CHUNK)
                .map(|c0| {
                    let mut buckets: Vec<Vec<Update>> =
                        (0..N_BUCKETS).map(|_| Vec::new()).collect();
                    let mut emit = |p: u32, q: u32, dist: f32| {
                        if dist < thr[p as usize] {
                            buckets[p as usize / rpb].push(Update {
                                dest: p,
                                nbr: q,
                                dist,
                            });
                        }
                        if dist < thr[q as usize] {
                            buckets[q as usize / rpb].push(Update {
                                dest: q,
                                nbr: p,
                                dist,
                            });
                        }
                    };
                    for i in c0..(c0 + GEN_CHUNK).min(block_end) {
                        let news = nc.row(i);
                        let olds = oc.row(i);
                        for (a, &p) in news.iter().enumerate() {
                            if p == EMPTY {
                                continue;
                            }
                            let xp = row(p as usize);
                            for &q in &news[a + 1..] {
                                if q == EMPTY || q == p {
                                    continue;
                                }
                                let dist = simd::sq_dist(xp, row(q as usize));
                                emit(p, q, dist);
                            }
                            for &q in olds {
                                if q == EMPTY || q == p {
                                    continue;
                                }
                                let dist = simd::sq_dist(xp, row(q as usize));
                                emit(p, q, dist);
                            }
                        }
                    }
                    buckets
                })
                .collect();
            changed += apply_updates(&mut heap, &gen, n);
        }

        tick(&format!("iter {iter}: join, {changed} changed"), &mut lap);
        if changed <= stop_at {
            break;
        }
    }

    // ── 3. Extract sorted rows, Euclidean distances ───────────────────
    indptr_out[0] = 0;
    let mut pos = 0_usize;
    let mut rowbuf: Vec<(u32, f32)> = Vec::with_capacity(k);
    for i in 0..n {
        rowbuf.clear();
        for s in 0..k {
            let j = heap.idx[i * k + s];
            if j != EMPTY {
                rowbuf.push((j, heap.dist[i * k + s]));
            }
        }
        rowbuf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(j, dist) in &rowbuf {
            indices_out[pos] = j;
            dists_out[pos] = dist.sqrt();
            pos += 1;
        }
        indptr_out[i + 1] = pos as u32;
    }

    tick("extract", &mut lap);
    if trace {
        eprintln!("[nndescent] total: {:.2}s", t0.elapsed().as_secs_f64());
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    fn make_clustered_data(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            let cluster = i % n_clusters;
            for j in 0..d {
                data[i * d + j] = cluster as f32 * 10.0 + rng.random::<f32>();
            }
        }
        data
    }

    fn exact_knn(data: &[f32], n: usize, d: usize, i: usize, k: usize) -> HashSet<u32> {
        let mut all: Vec<(u32, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    j as u32,
                    simd::sq_dist(&data[i * d..(i + 1) * d], &data[j * d..(j + 1) * d]),
                )
            })
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all.iter().take(k).map(|e| e.0).collect()
    }

    fn recall(
        graph: &NeighborGraph,
        data: &[f32],
        n: usize,
        d: usize,
        k: usize,
        queries: usize,
    ) -> f64 {
        let mut hit = 0;
        for i in 0..queries {
            let exact = exact_knn(data, n, d, i, k);
            let (s, e) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
            hit += graph.indices[s..e]
                .iter()
                .filter(|j| exact.contains(j))
                .count();
        }
        hit as f64 / (queries * k) as f64
    }

    /// The graph written into the caller's arrays is the graph returned.
    #[test]
    fn writing_into_caller_arrays_gives_the_same_graph() {
        let (n, d, k) = (1500, 6, 9);
        let data = make_clustered_data(n, d, 4, 3);
        let opts = NndescentOptions {
            k,
            seed: 5,
            ..Default::default()
        };
        let g = nndescent(&data, n, d, &opts);
        let cap = n * effective_k(n, k);
        // Buffers that are not clean: every entry the graph uses is overwritten.
        let (mut ip, mut ix, mut dx) = (vec![7_u32; n + 1], vec![9_u32; cap], vec![1.0_f32; cap]);
        let nnz = nndescent_into(&data, n, d, &opts, &mut ip, &mut ix, &mut dx);
        assert_eq!(nnz, g.indices.len());
        assert_eq!((&ip, &ix[..nnz]), (&g.indptr, &g.indices[..]));
        assert!(dx[..nnz]
            .iter()
            .zip(&g.data)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn graph_structure_is_valid() {
        let (n, d, k) = (2000, 8, 10);
        let data = make_clustered_data(n, d, 5, 1);
        let g = nndescent(
            &data,
            n,
            d,
            &NndescentOptions {
                k,
                seed: 1,
                ..Default::default()
            },
        );
        assert_eq!(g.n_obs, n);
        for i in 0..n {
            let (s, e) = (g.indptr[i] as usize, g.indptr[i + 1] as usize);
            assert_eq!(e - s, k, "node {i} has {} neighbours", e - s);
            let row = &g.indices[s..e];
            assert!(!row.contains(&(i as u32)), "self-loop at {i}");
            let uniq: HashSet<u32> = row.iter().copied().collect();
            assert_eq!(uniq.len(), k, "duplicate neighbour at {i}");
            for w in g.data[s..e].windows(2) {
                assert!(w[0] <= w[1], "distances not sorted at {i}");
            }
        }
    }

    #[test]
    fn recall_is_high_on_structured_data() {
        // Gaussian-ish blobs in 20-D, 6000 points: recall@15 vs exact.
        let (n, d, k) = (6000, 20, 15);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            let c = (i % 12) as f32;
            for j in 0..d {
                data[i * d + j] = c * 3.0 * ((j % 3) as f32) + rng.random::<f32>() * 4.0;
            }
        }
        let g = nndescent(
            &data,
            n,
            d,
            &NndescentOptions {
                k,
                seed: 4,
                ..Default::default()
            },
        );
        let r = recall(&g, &data, n, d, k, 300);
        assert!(r >= 0.95, "recall@15 = {r:.3}");
    }

    #[test]
    fn is_deterministic() {
        let (n, d, k) = (3000, 10, 12);
        let data = make_clustered_data(n, d, 7, 5);
        let opts = NndescentOptions {
            k,
            seed: 42,
            ..Default::default()
        };
        let g1 = nndescent(&data, n, d, &opts);
        let g2 = nndescent(&data, n, d, &opts);
        assert_eq!(g1.indices, g2.indices);
        assert_eq!(g1.data, g2.data);
    }

    #[test]
    fn tiny_inputs_work() {
        let data = make_clustered_data(5, 3, 2, 0);
        let g = nndescent(
            &data,
            5,
            3,
            &NndescentOptions {
                k: 10,
                ..Default::default()
            },
        );
        for i in 0..5 {
            let (s, e) = (g.indptr[i] as usize, g.indptr[i + 1] as usize);
            assert_eq!(e - s, 4);
        }
    }
}

#[cfg(test)]
mod forest_bench {
    //! `cargo test --release -p polariseq-core --lib -- nndescent::forest_bench --ignored --nocapture`
    use super::*;

    #[test]
    #[ignore = "manual microbenchmark"]
    fn rp_tree_build_time() {
        let (n, d) = (500_000usize, 50usize);
        let data: Vec<f32> = (0..n * d)
            .map(|i| (splitmix(i as u64) >> 40) as f32 / (1u64 << 24) as f32)
            .collect();
        let t = std::time::Instant::now();
        let tree = build_tree(&data, d, n, 15, 1);
        println!(
            "one tree: {:.3}s, {} leaves",
            t.elapsed().as_secs_f64(),
            tree.leaves.len()
        );
        let t = std::time::Instant::now();
        let forest: Vec<Tree> = (0..48)
            .into_par_iter()
            .map(|tr| build_tree(&data, d, n, 15, tr as u64 + 1))
            .collect();
        println!("forest of 48: {:.3}s", t.elapsed().as_secs_f64());
        let _ = forest;
    }
}
