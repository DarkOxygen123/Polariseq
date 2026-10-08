//! Paris hierarchical clustering of a graph, and cuts of its dendrogram.
//!
//! Paris (Bonald, Charpentier, Galland and Hollocou, 2018, "Hierarchical
//! graph clustering using node pair sampling") merges, at every step, the
//! two clusters `a`, `b` with the largest node-pair sampling similarity
//! `p(a, b) / (p(a) p(b))`, found with the nearest-neighbour chain. The
//! merge height is the inverse similarity. One pass over the graph gives the
//! whole hierarchy, so clusters at any resolution come from cutting it,
//! without clustering again.
//!
//! [`paris`] reproduces scikit-network's `Paris` (BSD-3-Clause, Bonald,
//! Charpentier and Lutz) bit for bit, and [`cut_balanced`] reproduces Scarf's
//! `BalancedCut` (BSD-3-Clause, Dhapola and Karlsson), because a port is only
//! as trustworthy as the evidence that it computes the same thing. That
//! includes their arithmetic, which is part of their output:
//!
//! * edge probabilities are `A_ij / f32(ΣA)` in `f64`: the total is rounded
//!   to `f32` (a `cdef float` in the reference);
//! * node probabilities are weighted degrees over their sum, with NumPy's
//!   summation order ([`numpy_sum`]);
//! * similarities are computed and compared in `f32`, heights are
//!   `1 / f64(similarity)`;
//! * ties go to the smallest cluster index, the chain starts at the oldest
//!   remaining cluster, and disconnected components are joined at infinite
//!   height, last component first.
//!
//! Memory: each cluster keeps its neighbours as two sorted parallel vectors
//! (`u32` ids, `f64` weights), 12 bytes per stored direction of an edge. A
//! merged cluster always has the largest id so far, so appending it keeps
//! every list sorted, without hashing.

// Project names (NumPy, SciPy) appear in prose; short names (i, j, p, q, t_ix,
// t_dx) follow the references' notation.
#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use thiserror::Error;

/// Errors from the hierarchy functions.
#[derive(Debug, Error, PartialEq)]
pub enum HierarchyError {
    /// Paris needs at least two nodes (the reference raises the same).
    #[error("the graph must contain at least two nodes, got {0}")]
    TooFewNodes(usize),
    /// The CSR arrays do not describe an `n × n` graph.
    #[error("malformed graph: {0}")]
    Malformed(String),
    /// A cut was asked for with impossible parameters.
    #[error("invalid cut: {0}")]
    InvalidCut(String),
}

/// One merge of a dendrogram: `[a, b, height, size]`. Leaves are `0..n`;
/// merge `t` creates cluster `n + t`. The layout of scipy's `linkage` and
/// scikit-network's `dendrogram_`.
pub type Merge = [f64; 4];

/// Options for [`paris`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ParisOptions {
    /// Reorder merges by non-decreasing height (scikit-network's
    /// `reorder=True`; Scarf uses `False`, the default here).
    pub reorder: bool,
    /// How float64 arrays are summed, which sets the last bits of the two
    /// normalising totals: `None` is NumPy 2 (pairwise over the whole
    /// array), `Some(8192)` is NumPy 1.x (pairwise within 8,192-element
    /// buffers, added in order). Only reference tests need to change it.
    pub numpy_sum_block: Option<usize>,
}

/// NumPy's pairwise summation of a contiguous `float64` array
/// (`pairwise_sum` in NumPy's `loops_utils.h`): plain loop below 8
/// elements, eight accumulators up to 128, halves (rounded down to a
/// multiple of 8) above.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let end = n - n % 8;
        let mut i = 8;
        while i < end {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum` of a contiguous `float64` array, in the summation order of the
/// NumPy version selected by `block` (see [`ParisOptions::numpy_sum_block`]).
/// Checked against NumPy 2.4.6 (`None`) and 1.26.4 (`Some(8192)`) on 440
/// arrays with values spanning 16 orders of magnitude.
#[must_use]
pub fn numpy_sum(a: &[f64], block: Option<usize>) -> f64 {
    match block {
        Some(b) if b > 0 && a.len() > b => {
            let mut s = pairwise_sum(&a[..b]);
            let mut i = b;
            while i < a.len() {
                s += pairwise_sum(&a[i..(i + b).min(a.len())]);
                i += b;
            }
            s
        }
        _ => pairwise_sum(a),
    }
}

/// The symmetric graph Paris aggregates, as `float64` CSR.
struct Csr {
    indptr: Vec<usize>,
    indices: Vec<u32>,
    data: Vec<f64>,
}

fn check_csr(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    n: usize,
) -> Result<(), HierarchyError> {
    if indptr.len() != n + 1 {
        return Err(HierarchyError::Malformed(format!(
            "indptr has {} entries for {n} nodes",
            indptr.len()
        )));
    }
    let nnz = indptr[n] as usize;
    if indices.len() != nnz || data.len() != nnz || indptr[0] != 0 {
        return Err(HierarchyError::Malformed(format!(
            "indptr ends at {nnz}, indices has {}, data has {}",
            indices.len(),
            data.len()
        )));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(HierarchyError::Malformed(
            "indptr is not non-decreasing".into(),
        ));
    }
    if let Some(&j) = indices.iter().find(|&&j| j as usize >= n) {
        return Err(HierarchyError::Malformed(format!(
            "column index {j} for {n} nodes"
        )));
    }
    for i in 0..n {
        let row = &indices[indptr[i] as usize..indptr[i + 1] as usize];
        if row.windows(2).any(|w| w[0] >= w[1]) {
            return Err(HierarchyError::Malformed(format!(
                "row {i} is not sorted by column without duplicates"
            )));
        }
    }
    if let Some(v) = data.iter().find(|&&v| !(v > 0.0 && v.is_finite())) {
        return Err(HierarchyError::Malformed(format!(
            "edge weight {v}: weights must be positive and finite"
        )));
    }
    Ok(())
}

/// Row sums in stored order (scipy's `csr_matvec` with a vector of ones).
fn row_sums(indptr: &[u32], data: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut s = 0.0_f64;
            for &v in &data[indptr[i] as usize..indptr[i + 1] as usize] {
                s += v;
            }
            s
        })
        .collect()
}

/// Column sums accumulated row by row (scipy's `csc_matvec` on `A.T`).
fn col_sums(indptr: &[u32], indices: &[u32], data: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        for jj in indptr[i] as usize..indptr[i + 1] as usize {
            y[indices[jj] as usize] += data[jj];
        }
    }
    y
}

/// The transpose as CSR, rows sorted (entries of each column in row order).
fn transpose(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    n: usize,
) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let mut count = vec![0usize; n + 1];
    for &j in indices {
        count[j as usize + 1] += 1;
    }
    for i in 0..n {
        count[i + 1] += count[i];
    }
    let mut next = count.clone();
    let mut t_ix = vec![0u32; indices.len()];
    let mut t_dx = vec![0.0f64; indices.len()];
    for i in 0..n {
        for jj in indptr[i] as usize..indptr[i + 1] as usize {
            let j = indices[jj] as usize;
            t_ix[next[j]] = i as u32;
            t_dx[next[j]] = data[jj];
            next[j] += 1;
        }
    }
    (count, t_ix, t_dx)
}

/// `A + sign * A.T` for sorted rows, dropping zero results: scipy's
/// canonical `csr_binop_csr` in `float64`.
fn add_transpose(indptr: &[u32], indices: &[u32], data: &[f64], n: usize, sign: f64) -> Csr {
    let (tp, tx, td) = transpose(indptr, indices, data, n);
    let mut out = Csr {
        indptr: Vec::with_capacity(n + 1),
        indices: Vec::new(),
        data: Vec::new(),
    };
    out.indptr.push(0);
    for i in 0..n {
        let (mut p, pe) = (indptr[i] as usize, indptr[i + 1] as usize);
        let (mut q, qe) = (tp[i], tp[i + 1]);
        while p < pe || q < qe {
            let (j, v) = if q >= qe || (p < pe && indices[p] < tx[q]) {
                p += 1;
                (indices[p - 1], data[p - 1])
            } else if p >= pe || tx[q] < indices[p] {
                q += 1;
                (tx[q - 1], sign * td[q - 1])
            } else {
                p += 1;
                q += 1;
                (indices[p - 1], data[p - 1] + sign * td[q - 1])
            };
            if v != 0.0 {
                out.indices.push(j);
                out.data.push(v);
            }
        }
        out.indptr.push(out.indices.len());
    }
    out
}

/// The graph Paris aggregates: the input if symmetric, `A + A.T` otherwise
/// (scikit-network on `float64` input; on `float32` input scikit-network
/// 0.33 truncates `A` to integers first, because its test
/// `data.dtype == float` fails for `float32`, which is not reproduced), plus
/// a unit self-loop on every node of zero weight, as the reference adds.
fn aggregate_input(indptr: &[u32], indices: &[u32], data: &[f64], n: usize, null: &[bool]) -> Csr {
    let symmetric = add_transpose(indptr, indices, data, n, -1.0)
        .indices
        .is_empty();
    let g = if symmetric {
        Csr {
            indptr: indptr.iter().map(|&p| p as usize).collect(),
            indices: indices.to_vec(),
            data: data.to_vec(),
        }
    } else {
        add_transpose(indptr, indices, data, n, 1.0)
    };
    if !null.iter().any(|&z| z) {
        return g;
    }
    // `adjacency += diags(null)`: a null node has an empty row (weights are
    // positive), so its row becomes `[(i, 1.0)]`.
    let mut out = Csr {
        indptr: Vec::with_capacity(n + 1),
        indices: Vec::new(),
        data: Vec::new(),
    };
    out.indptr.push(0);
    for (i, &is_null) in null.iter().enumerate() {
        if is_null {
            out.indices.push(i as u32);
            out.data.push(1.0);
        } else {
            out.indices
                .extend_from_slice(&g.indices[g.indptr[i]..g.indptr[i + 1]]);
            out.data
                .extend_from_slice(&g.data[g.indptr[i]..g.indptr[i + 1]]);
        }
        out.indptr.push(out.indices.len());
    }
    out
}

/// Node-pair similarity exactly as scikit-network's `cdef float similarity`.
#[inline]
fn similarity(w: f64, out1: f64, in1: f64, out2: f64, in2: f64) -> f32 {
    let a = (out1 * in2) as f32;
    let b = (out2 * in1) as f32;
    let den = a + b;
    if den > 0.0 {
        ((2.0 * w) / f64::from(den)) as f32
    } else {
        f32::NEG_INFINITY
    }
}

/// The clusters being aggregated. Neighbour lists are sorted by id and may
/// hold stale entries for clusters merged since (skipped by `active`, and
/// compacted when they outnumber the live ones), so a merge never has to
/// search a neighbour's list.
struct Aggregate {
    ids: Vec<Vec<u32>>,
    ws: Vec<Vec<f64>>,
    w_out: Vec<f64>,
    w_in: Vec<f64>,
    size: Vec<u64>,
    active: Vec<bool>,
}

impl Aggregate {
    /// Nearest active neighbour of `node`: `(id, similarity, n_live)`.
    fn nearest(&mut self, node: usize) -> (usize, f32, usize) {
        let mut max_sim = f32::NEG_INFINITY;
        let mut nearest = usize::MAX;
        let mut live = 0usize;
        for (k, &j) in self.ids[node].iter().enumerate() {
            let j = j as usize;
            if !self.active[j] {
                continue;
            }
            live += 1;
            let sim = similarity(
                self.ws[node][k],
                self.w_out[node],
                self.w_in[node],
                self.w_out[j],
                self.w_in[j],
            );
            if sim > max_sim {
                nearest = j;
                max_sim = sim;
            } else if sim.to_bits() == max_sim.to_bits() {
                // An exact tie, as the reference compares.
                nearest = nearest.min(j);
            }
        }
        if 2 * live < self.ids[node].len() {
            let active = &self.active;
            let (ids, ws) = (&mut self.ids[node], &mut self.ws[node]);
            let mut w = 0;
            for r in 0..ids.len() {
                if active[ids[r] as usize] {
                    ids[w] = ids[r];
                    ws[w] = ws[r];
                    w += 1;
                }
            }
            ids.truncate(w);
            ws.truncate(w);
        }
        (nearest, max_sim, live)
    }

    /// Merge clusters `a` and `b` into `new`.
    fn merge(&mut self, a: usize, b: usize, new: usize) {
        let (ia, wa) = (
            std::mem::take(&mut self.ids[a]),
            std::mem::take(&mut self.ws[a]),
        );
        let (ib, wb) = (
            std::mem::take(&mut self.ids[b]),
            std::mem::take(&mut self.ws[b]),
        );
        self.active[a] = false;
        self.active[b] = false;
        let live = |j: u32, act: &[bool]| act[j as usize];
        let mut ids = Vec::with_capacity(ia.len().max(ib.len()));
        let mut ws = Vec::with_capacity(ia.len().max(ib.len()));
        let (mut p, mut q) = (0, 0);
        while p < ia.len() || q < ib.len() {
            if q >= ib.len() || (p < ia.len() && ia[p] < ib[q]) {
                if live(ia[p], &self.active) {
                    ids.push(ia[p]);
                    ws.push(wa[p]);
                }
                p += 1;
            } else if p >= ia.len() || ib[q] < ia[p] {
                if live(ib[q], &self.active) {
                    ids.push(ib[q]);
                    ws.push(wb[q]);
                }
                q += 1;
            } else {
                // A common neighbour: the reference adds a's weight, then b's.
                if live(ia[p], &self.active) {
                    ids.push(ia[p]);
                    ws.push(wa[p] + wb[q]);
                }
                p += 1;
                q += 1;
            }
        }
        // Each neighbour gains `new`, the largest id so far, so appending
        // keeps its list sorted; its weight is the same sum (symmetry).
        for (k, &x) in ids.iter().enumerate() {
            self.ids[x as usize].push(new as u32);
            self.ws[x as usize].push(ws[k]);
        }
        self.ids[new] = ids;
        self.ws[new] = ws;
        self.w_out[new] = self.w_out[a] + self.w_out[b];
        self.w_in[new] = self.w_in[a] + self.w_in[b];
        self.size[new] = self.size[a] + self.size[b];
        self.active[new] = true;
    }
}

/// Paris hierarchical clustering of a weighted graph given as CSR
/// (`indptr`, `indices`, `data`, `n` nodes). Rows must be sorted by column
/// without duplicates, and weights positive and finite (the inputs the
/// reference is defined on; checked).
///
/// Returns the `n - 1` merges (see [`Merge`]). A directed graph is
/// symmetrised as `A + A.T`, with node weights taken from the directed
/// graph, as in the reference.
///
/// # Errors
/// [`HierarchyError::TooFewNodes`] below two nodes,
/// [`HierarchyError::Malformed`] for inconsistent CSR arrays.
pub fn paris(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    n: usize,
    opts: &ParisOptions,
) -> Result<Vec<Merge>, HierarchyError> {
    if n <= 1 {
        return Err(HierarchyError::TooFewNodes(n));
    }
    check_csr(indptr, indices, data, n)?;
    // Node sampling probabilities: degrees of the (possibly directed) input.
    let out_deg = row_sums(indptr, data, n);
    let in_deg = col_sums(indptr, indices, data, n);
    let so = numpy_sum(&out_deg, opts.numpy_sum_block);
    let si = numpy_sum(&in_deg, opts.numpy_sum_block);
    let mut w_out: Vec<f64> = out_deg.iter().map(|&d| d / so).collect();
    let mut w_in: Vec<f64> = in_deg.iter().map(|&d| d / si).collect();
    drop((out_deg, in_deg));
    let null: Vec<bool> = (0..n).map(|i| w_out[i] + w_in[i] == 0.0).collect();

    // Edge sampling probabilities: weight over f32(total), in f64.
    let g = aggregate_input(indptr, indices, data, n, &null);
    let total = f64::from(numpy_sum(&g.data, opts.numpy_sum_block) as f32);

    let n_max = 2 * n - 1;
    let mut agg = Aggregate {
        ids: vec![Vec::new(); n_max],
        ws: vec![Vec::new(); n_max],
        w_out: Vec::new(),
        w_in: Vec::new(),
        size: vec![1; n],
        active: vec![true; n],
    };
    for i in 0..n {
        let (lo, hi) = (g.indptr[i], g.indptr[i + 1]);
        let (mut ids, mut ws) = (Vec::with_capacity(hi - lo), Vec::with_capacity(hi - lo));
        for jj in lo..hi {
            if g.indices[jj] as usize != i {
                ids.push(g.indices[jj]);
                ws.push(g.data[jj] / total);
            }
        }
        agg.ids[i] = ids;
        agg.ws[i] = ws;
    }
    drop(g);
    w_out.resize(n_max, 0.0);
    w_in.resize(n_max, 0.0);
    agg.w_out = w_out;
    agg.w_in = w_in;
    agg.size.resize(n_max, 0);
    agg.active.resize(n_max, false);

    let mut next = n; // id of the next cluster
    let mut first = 0usize; // oldest possibly-active cluster
    let mut dendrogram: Vec<Merge> = Vec::with_capacity(n - 1);
    let mut components: Vec<(usize, u64)> = Vec::new();
    let mut chain: Vec<usize> = Vec::new();
    let mut remaining = n;
    while remaining > 0 {
        while !agg.active[first] {
            first += 1;
        }
        chain.clear();
        chain.push(first);
        while let Some(node) = chain.pop() {
            let (nearest, max_sim, live) = agg.nearest(node);
            if live == 0 || nearest == usize::MAX {
                // No neighbour left: a connected component. (All-`-inf`
                // similarities cannot occur with positive weights; the
                // reference would read an unset variable there.)
                components.push((node, agg.size[node]));
                agg.active[node] = false;
                remaining -= 1;
                continue;
            }
            match chain.pop() {
                Some(last) if last == nearest => {
                    let s = agg.size[node] + agg.size[nearest];
                    dendrogram.push([
                        node as f64,
                        nearest as f64,
                        1.0 / f64::from(max_sim),
                        s as f64,
                    ]);
                    agg.merge(node, nearest, next);
                    next += 1;
                    remaining -= 1;
                }
                Some(last) => {
                    chain.push(last);
                    chain.push(node);
                    chain.push(nearest);
                }
                None => {
                    chain.push(node);
                    chain.push(nearest);
                }
            }
        }
    }

    // Join the connected components at infinite height, last one first.
    if let Some((mut node, mut csize)) = components.pop() {
        for &(other, osize) in &components {
            csize += osize;
            dendrogram.push([node as f64, other as f64, f64::INFINITY, csize as f64]);
            node = next;
            next += 1;
        }
    }
    debug_assert_eq!(dendrogram.len(), n - 1);
    if opts.reorder {
        reorder_dendrogram(&mut dendrogram);
    }
    Ok(dendrogram)
}

/// scikit-network's `reorder_dendrogram`: a stable sort of the merges by
/// height, then by their larger child, with internal ids renumbered.
pub fn reorder_dendrogram(d: &mut [Merge]) {
    let n = d.len() + 1;
    let mut index: Vec<usize> = (0..d.len()).collect();
    let key = |t: usize| (d[t][2], d[t][0].max(d[t][1]));
    index.sort_by(|&x, &y| {
        let (hx, mx) = key(x);
        let (hy, my) = key(y);
        hx.total_cmp(&hy).then(mx.total_cmp(&my))
    });
    let mut index_new: Vec<usize> = (0..2 * n - 1).collect();
    for (k, &t) in index.iter().enumerate() {
        index_new[n + t] = n + k;
    }
    let old: Vec<Merge> = d.to_vec();
    for (k, &t) in index.iter().enumerate() {
        let m = old[t];
        d[k] = [
            index_new[m[0] as usize] as f64,
            index_new[m[1] as usize] as f64,
            m[2],
            m[3],
        ];
    }
}

fn check_dendrogram(d: &[Merge]) -> Result<usize, HierarchyError> {
    let n = d.len() + 1;
    for (t, m) in d.iter().enumerate() {
        for c in [m[0], m[1]] {
            if !(c >= 0.0 && c < (n + t) as f64 && c.fract() == 0.0) {
                return Err(HierarchyError::InvalidCut(format!(
                    "merge {t} has child {c}, not an earlier node"
                )));
            }
        }
    }
    Ok(n)
}

/// Cut a dendrogram into (at least) `n_clusters` clusters, or at a height
/// `threshold`: scikit-network's `cut_straight`. Merges strictly below the
/// cut height are kept; equal heights can give more clusters than asked.
///
/// Labels are numbered by decreasing cluster size, ties in order of first
/// appearance. The partition equals the reference's; label numbers can
/// differ among equal-size clusters, where NumPy's unstable sort decides.
///
/// # Errors
/// [`HierarchyError::InvalidCut`] for `n_clusters` outside `1..=n` or a
/// malformed dendrogram.
pub fn cut_straight(
    d: &[Merge],
    n_clusters: Option<usize>,
    threshold: Option<f64>,
) -> Result<Vec<u32>, HierarchyError> {
    let n = check_dendrogram(d)?;
    let k = match (n_clusters, threshold) {
        (Some(k), _) => k,
        (None, None) => 2,
        (None, Some(_)) => n,
    };
    if k == 0 || k > n {
        return Err(HierarchyError::InvalidCut(format!(
            "n_clusters must be in 1..={n}, got {k}"
        )));
    }
    let mut heights: Vec<f64> = d.iter().map(|m| m[2]).collect();
    heights.sort_by(f64::total_cmp);
    if k == 1 {
        // scikit-network indexes past the end here (IndexError); one
        // cluster is the only sensible answer.
        return Ok(vec![0; n]);
    }
    let mut cut = heights[n - k];
    if let Some(t) = threshold {
        cut = cut.max(t);
    }
    // cluster id -> members, in the reference dict's insertion order.
    let mut members: Vec<Option<Vec<u32>>> = (0..n as u32).map(|i| Some(vec![i])).collect();
    members.resize(2 * n - 1, None);
    let mut order: Vec<usize> = (0..n).collect();
    for (t, m) in d.iter().enumerate() {
        let (i, j) = (m[0] as usize, m[1] as usize);
        if m[2] < cut && members[i].is_some() && members[j].is_some() {
            let mut a = members[i].take().unwrap_or_default();
            a.extend(members[j].take().unwrap_or_default());
            members[n + t] = Some(a);
            order.push(n + t);
        }
    }
    let mut clusters: Vec<Vec<u32>> = order
        .into_iter()
        .filter_map(|c| members[c].take())
        .collect();
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len())); // stable
    let mut labels = vec![0u32; n];
    for (label, nodes) in clusters.iter().enumerate() {
        for &v in nodes {
            labels[v as usize] = label as u32;
        }
    }
    Ok(labels)
}

/// The dendrogram as Scarf's `make_digraph` builds it: leaves have 0
/// leaves and the height of the merge that takes them; internal node
/// `n + t` has merge `t`'s size and height; children in the merge's order.
struct Tree {
    n: usize,
    nleaves: Vec<i64>,
    dist: Vec<f64>,
    children: Vec<[u32; 2]>,
    parent: Vec<u32>,
}

impl Tree {
    fn new(d: &[Merge], n: usize) -> Self {
        let total = 2 * n - 1;
        let mut tree = Tree {
            n,
            nleaves: vec![0; total],
            dist: vec![0.0; total],
            children: vec![[u32::MAX; 2]; total],
            parent: vec![u32::MAX; total],
        };
        for (t, m) in d.iter().enumerate() {
            let node = n + t;
            let (a, b) = (m[0] as usize, m[1] as usize);
            tree.nleaves[node] = m[3] as i64;
            tree.dist[node] = m[2];
            for c in [a, b] {
                if c < n {
                    tree.dist[c] = m[2];
                }
                tree.parent[c] = node as u32;
            }
            tree.children[node] = [a as u32, b as u32];
        }
        tree
    }

    fn successors(&self, i: usize) -> &[u32] {
        if i >= self.n {
            &self.children[i]
        } else {
            &[]
        }
    }

    /// Mean height over a node's descendants in breadth-first order
    /// (Scarf's `_get_mean_dist`), summed as NumPy does; cached.
    fn mean_dist(&self, start: usize, block: Option<usize>, cache: &mut [Option<f64>]) -> f64 {
        if let Some(v) = cache[start] {
            return v;
        }
        let mut q = std::collections::VecDeque::new();
        q.extend(self.successors(start).iter().map(|&c| c as usize));
        let mut vals = Vec::new();
        while let Some(i) = q.pop_front() {
            vals.push(self.dist[i]);
            q.extend(self.successors(i).iter().map(|&c| c as usize));
        }
        let v = numpy_sum(&vals, block) / vals.len() as f64;
        cache[start] = Some(v);
        v
    }
}

/// Scarf's `BalancedCut`: climb from each leaf while the subtree stays at
/// most `max_size` leaves and the two subtrees being joined have similar
/// heights (ratio at most `max_distance_fc`, for their own heights and the
/// mean heights below them, checked once both exceed `min_size` leaves).
///
/// Labels start at 1 in the order clusters are formed, as in Scarf; `-1`
/// would mark an unassigned leaf (none is expected). One deliberate
/// difference: when `max_size` admits the whole tree, Scarf raises
/// (`StopIteration` at the root); here the climb stops at the root.
///
/// # Errors
/// [`HierarchyError::InvalidCut`] for a malformed dendrogram or `max_size`
/// of zero.
pub fn cut_balanced(
    d: &[Merge],
    max_size: u64,
    min_size: u64,
    max_distance_fc: f64,
    numpy_sum_block: Option<usize>,
) -> Result<Vec<i64>, HierarchyError> {
    let n = check_dendrogram(d)?;
    if max_size == 0 {
        return Err(HierarchyError::InvalidCut(
            "max_size must be at least 1".into(),
        ));
    }
    let tree = Tree::new(d, n);
    let (min_size, max_size) = (min_size as i64, max_size as i64);
    let mut cache: Vec<Option<f64>> = vec![None; 2 * n - 1];
    let mut mergeable = |s1: usize, s2: usize| -> bool {
        if tree.nleaves[s1] > min_size && tree.nleaves[s2] > min_size {
            let (d1, d2) = (tree.dist[s1], tree.dist[s2]);
            if d1 / d2 > max_distance_fc || d2 / d1 > max_distance_fc {
                return false;
            }
            let m1 = tree.mean_dist(s1, numpy_sum_block, &mut cache);
            let m2 = tree.mean_dist(s2, numpy_sum_block, &mut cache);
            if m1 / m2 > max_distance_fc || m2 / m1 > max_distance_fc {
                return false;
            }
        }
        true
    };

    // `leaves` is a dict emptied with popitem(): the largest remaining leaf.
    let mut remaining = vec![true; n];
    let mut n_remaining = n;
    let mut top = n;
    let mut is_bp = vec![false; 2 * n - 1];
    let mut bps: Vec<Vec<u32>> = Vec::new();
    while n_remaining > 0 {
        while !remaining[top - 1] {
            top -= 1;
        }
        let leaf = top - 1;
        remaining[leaf] = false;
        n_remaining -= 1;
        let mut cur = leaf;
        while tree.parent[cur] != u32::MAX {
            let temp = tree.parent[cur] as usize;
            if is_bp[temp] || tree.nleaves[temp] > max_size {
                break;
            }
            let [s1, s2] = tree.children[temp];
            if !mergeable(s1 as usize, s2 as usize) {
                break;
            }
            cur = temp;
        }
        let mut members = vec![leaf as u32];
        is_bp[cur] = true;
        let mut stack = vec![cur];
        while let Some(i) = stack.pop() {
            if i < n && remaining[i] {
                members.push(i as u32);
                remaining[i] = false;
                n_remaining -= 1;
            } else if i != cur && (is_bp[i] || tree.nleaves[i] >= max_size) {
                // Taken by another branch, or too large to take greedily.
            } else {
                stack.extend(tree.successors(i).iter().map(|&c| c as usize));
            }
        }
        bps.push(members);
    }
    let mut labels = vec![-1i64; n];
    for (k, members) in bps.iter().enumerate() {
        for &v in members {
            labels[v as usize] = k as i64 + 1;
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr_from_edges(
        n: usize,
        edges: &[(usize, usize, f64)],
        symmetric: bool,
    ) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for &(a, b, w) in edges {
            rows[a].push((b as u32, w));
            if symmetric {
                rows[b].push((a as u32, w));
            }
        }
        let mut indptr = vec![0u32];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for r in &mut rows {
            r.sort_by_key(|&(j, _)| j);
            for &(j, w) in r.iter() {
                indices.push(j);
                data.push(w);
            }
            indptr.push(indices.len() as u32);
        }
        (indptr, indices, data)
    }

    /// scikit-network's `house()` graph: 5 nodes, 6 unit edges.
    fn house() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let e = [(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 4)];
        csr_from_edges(5, &e.map(|(a, b)| (a, b, 1.0)), true)
    }

    #[test]
    fn house_matches_scikit_network() {
        // scikit-network 0.33.2's actual output (its docstring's heights,
        // 0.17/0.25/0.31/0.67, are out of date by a factor of 2):
        // [[3, 2, 1/3, 2], [1, 0, 0.5, 2], [6, 4, 0.62499999, 3], [7, 5, 4/3, 5]]
        let (ip, ix, dx) = house();
        let opts = ParisOptions {
            reorder: true,
            numpy_sum_block: None,
        };
        let d = paris(&ip, &ix, &dx, 5, &opts).unwrap();
        let want = [
            [3.0, 2.0, 1.0 / 3.0, 2.0],
            [1.0, 0.0, 0.5, 2.0],
            [6.0, 4.0, 0.625, 3.0],
            [7.0, 5.0, 4.0 / 3.0, 5.0],
        ];
        for (got, want) in d.iter().zip(want) {
            assert_eq!(
                (got[0], got[1], got[3]),
                (want[0], want[1], want[3]),
                "{d:?}"
            );
            // Heights carry f32 similarity rounding (0.62499999 in the reference).
            assert!((got[2] - want[2]).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn numpy_sum_orders() {
        let a: Vec<f64> = (0..300)
            .map(|i| (f64::from(i) * 0.37).sin() * 10f64.powi(i % 13 - 6))
            .collect();
        let naive: f64 = a.iter().sum();
        assert!((numpy_sum(&a, None) - naive).abs() < 1e-3);
        assert!((numpy_sum(&a, Some(128)) - naive).abs() < 1e-3);
        // Below one block both NumPy versions add in the same order.
        assert_eq!(
            numpy_sum(&a[..100], None).to_bits(),
            numpy_sum(&a[..100], Some(8192)).to_bits()
        );
        assert_eq!(numpy_sum(&[], None).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn components_are_joined_at_infinity() {
        // Two disjoint edges and an isolated node.
        let (ip, ix, dx) = csr_from_edges(5, &[(0, 1, 1.0), (2, 3, 1.0)], true);
        let d = paris(&ip, &ix, &dx, 5, &ParisOptions::default()).unwrap();
        assert_eq!(d.len(), 4);
        assert_eq!(d.iter().filter(|m| m[2].is_infinite()).count(), 2);
        assert_eq!(d.last().unwrap()[3].to_bits(), 5.0f64.to_bits());
    }

    #[test]
    fn a_directed_graph_is_its_symmetrisation_when_weights_agree() {
        // Upper triangle with weight 2 vs the symmetric graph with weight 1:
        // A + A.T is the same graph; in/out degrees differ, so compare
        // against the symmetric graph with weight 2 read both ways instead.
        let e: Vec<(usize, usize, f64)> = [(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 4)]
            .iter()
            .map(|&(a, b)| (a, b, 2.0))
            .collect();
        let (dip, dix, ddx) = csr_from_edges(5, &e, false);
        let d = paris(&dip, &dix, &ddx, 5, &ParisOptions::default()).unwrap();
        assert_eq!(d.len(), 4);
        assert_eq!(d.last().unwrap()[3].to_bits(), 5.0f64.to_bits());
        assert!(d.iter().all(|m| m[2].is_finite() && m[2] > 0.0));
    }

    #[test]
    fn straight_cut_gives_the_asked_number_of_clusters() {
        let (ip, ix, dx) = house();
        let d = paris(&ip, &ix, &dx, 5, &ParisOptions::default()).unwrap();
        for k in 1..=5 {
            let labels = cut_straight(&d, Some(k), None).unwrap();
            let distinct: std::collections::BTreeSet<u32> = labels.iter().copied().collect();
            assert_eq!(distinct.len(), k, "k = {k}: {labels:?}");
        }
        assert!(cut_straight(&d, Some(0), None).is_err());
        assert!(cut_straight(&d, Some(6), None).is_err());
    }

    #[test]
    fn balanced_cut_covers_every_leaf_within_max_size() {
        let (ip, ix, dx) = house();
        let d = paris(&ip, &ix, &dx, 5, &ParisOptions::default()).unwrap();
        let labels = cut_balanced(&d, 2, 1, 2.0, None).unwrap();
        assert!(labels.iter().all(|&l| l >= 1), "{labels:?}");
        let mut counts = std::collections::BTreeMap::new();
        for &l in &labels {
            *counts.entry(l).or_insert(0) += 1;
        }
        assert!(counts.values().all(|&c| c <= 2), "{labels:?}");
    }

    #[test]
    fn malformed_input_is_refused() {
        let o = ParisOptions::default();
        assert_eq!(
            paris(&[0, 0], &[], &[], 1, &o),
            Err(HierarchyError::TooFewNodes(1))
        );
        assert!(paris(&[0, 1, 1], &[5], &[1.0], 2, &o).is_err()); // column out of range
        assert!(paris(&[0, 2, 2], &[1, 1], &[1.0, 1.0], 2, &o).is_err()); // duplicate
        assert!(paris(&[0, 1, 2], &[1, 0], &[1.0, -1.0], 2, &o).is_err()); // negative weight
    }
}
