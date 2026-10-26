//! k-nearest-neighbor graph construction for the neighbor step of the pipeline.
//!
//! Given a cell embedding (e.g. `X_pca`, `n_obs × n_dim`), find the `k` nearest
//! neighbors of each cell under a metric (Euclidean by default, matching
//! scanpy's `method='umap'` default). Produces a sparse kNN graph in CSR-like
//! form: for each cell, its `k` neighbors and their distances.
//!
//! This module implements **exact brute-force kNN** with rayon parallelism over
//! query cells. For 1M+ cells this is O(n²·d) and should be replaced with
//! approximate kNN (HNSW / NN-Descent) — but brute force is the correct
//! reference and is fast enough up to ~50k cells.

use crate::backend::NeighborGraph;
use rayon::prelude::*;

/// Metric for the neighbor search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// Squared Euclidean distance (we keep the sqrt off for speed during the
    /// search, applying it only to the final k distances).
    #[default]
    Euclidean,
    /// Cosine distance (`1 - cosine_similarity`).
    Cosine,
}

impl Metric {
    /// Parse a metric from its string name (case-insensitive).
    ///
    /// See also: `std::str::FromStr` (this is the inherent convenience form).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "euclidean" => Some(Self::Euclidean),
            "cosine" => Some(Self::Cosine),
            _ => None,
        }
    }
}

/// Options for neighbor graph construction.
#[derive(Debug, Clone)]
pub struct NeighborsOptions {
    /// Number of nearest neighbors per cell (scanpy default: 15).
    pub n_neighbors: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Random seed (for randomized variants; unused by exact kNN).
    pub seed: u64,
}

impl Default for NeighborsOptions {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            seed: 0,
        }
    }
}

/// Errors raised by neighbor graph construction.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum NeighborsError {
    /// The embedding has zero rows or zero dimensions.
    #[error("embedding has zero rows ({nrows}) or dims ({ndims})")]
    EmptyInput {
        /// Number of rows.
        nrows: usize,
        /// Number of dimensions.
        ndims: usize,
    },
    /// `n_neighbors` is larger than the number of cells.
    #[error("n_neighbors ({requested}) exceeds n_obs ({n_obs})")]
    TooManyNeighbors {
        /// Requested neighbor count.
        requested: usize,
        /// Number of observations.
        n_obs: usize,
    },
    /// `n_neighbors` is zero.
    #[error("n_neighbors must be ≥ 1, got {0}")]
    ZeroNeighbors(usize),
}

/// A single neighbor: index and distance.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Neighbor {
    dist: f32,
    idx: u32,
}

/// Build the kNN graph from a row-major embedding.
///
/// Returns a [`NeighborGraph`] where each cell has up to `n_neighbors` edges
/// (excluding self). Distances are Euclidean (sqrt applied), matching scanpy's
/// `obsp['distances']`.
///
/// # Errors
/// Returns [`NeighborsError`] on invalid inputs.
pub fn neighbors(
    embedding: &[f32],
    n_obs: usize,
    n_dim: usize,
    opts: &NeighborsOptions,
) -> Result<NeighborGraph, NeighborsError> {
    if n_obs == 0 || n_dim == 0 {
        return Err(NeighborsError::EmptyInput {
            nrows: n_obs,
            ndims: n_dim,
        });
    }
    if opts.n_neighbors == 0 {
        return Err(NeighborsError::ZeroNeighbors(opts.n_neighbors));
    }
    // +1 because the search includes self; we drop self afterward.
    let k_with_self = opts.n_neighbors.min(n_obs);

    // For each query cell, find its k nearest neighbors (parallel over queries).
    let per_cell: Vec<Vec<Neighbor>> = (0..n_obs)
        .into_par_iter()
        .map(|i| knn_for_one(embedding, n_obs, n_dim, i, k_with_self, opts.metric))
        .collect();

    // Assemble CSR. Each cell contributes (k_with_self - 1) edges (self dropped).
    let edges_per_cell = k_with_self - 1;
    let n_edges = n_obs * edges_per_cell;
    let mut indptr = Vec::with_capacity(n_obs + 1);
    let mut indices = Vec::with_capacity(n_edges);
    let mut data = Vec::with_capacity(n_edges);
    indptr.push(0);
    for cell in &per_cell {
        for nb in cell.iter().skip(1) {
            // skip(1) drops self (index 0, distance 0)
            indices.push(nb.idx);
            data.push(nb.dist);
        }
        indptr.push(indices.len() as u32);
    }

    Ok(NeighborGraph {
        indptr,
        indices,
        data,
        n_obs,
    })
}

/// Find the k nearest neighbors of cell `query` by brute force, keeping a
/// bounded max-heap of size `k` (root = current worst). O(n·d + n·log k).
fn knn_for_one(
    embedding: &[f32],
    n_obs: usize,
    n_dim: usize,
    query: usize,
    k: usize,
    metric: Metric,
) -> Vec<Neighbor> {
    let qv = &embedding[query * n_dim..(query + 1) * n_dim];
    let mut heap: Vec<Neighbor> = Vec::with_capacity(k);

    let sift_down = |h: &mut [Neighbor], mut p: usize| {
        let len = h.len();
        loop {
            let (l, r) = (2 * p + 1, 2 * p + 2);
            let mut m = p;
            if l < len && h[l].dist > h[m].dist {
                m = l;
            }
            if r < len && h[r].dist > h[m].dist {
                m = r;
            }
            if m == p {
                break;
            }
            h.swap(p, m);
            p = m;
        }
    };

    for j in 0..n_obs {
        let other = &embedding[j * n_dim..(j + 1) * n_dim];
        let d = match metric {
            Metric::Euclidean => euclidean_sq(qv, other),
            Metric::Cosine => cosine_dist(qv, other),
        };
        if heap.len() < k {
            heap.push(Neighbor {
                dist: d,
                idx: j as u32,
            });
            // Sift up.
            let mut c = heap.len() - 1;
            while c > 0 {
                let parent = (c - 1) / 2;
                if heap[c].dist > heap[parent].dist {
                    heap.swap(c, parent);
                    c = parent;
                } else {
                    break;
                }
            }
        } else if d < heap[0].dist {
            heap[0] = Neighbor {
                dist: d,
                idx: j as u32,
            };
            sift_down(&mut heap, 0);
        }
    }
    // Sort ascending (closest first), apply sqrt for Euclidean.
    heap.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if metric == Metric::Euclidean {
        for n in &mut heap {
            n.dist = n.dist.sqrt();
        }
    }
    heap
}

/// Squared Euclidean distance between two equal-length slices.
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::sq_dist(a, b)
}

/// Cosine distance (1 - cosine similarity).
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot = crate::simd::dot(a, b);
    let na = crate::simd::dot(a, a);
    let nb = crate::simd::dot(b, b);
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        1.0 - dot / denom
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_nearest_in_simple_embedding() {
        // 4 cells in 2D:
        //   0: (0,0)  1: (1,0)  2: (10,10)  3: (0,1)
        // Cell 0's nearest: itself (0), then 1 (dist 1), 3 (dist 1), 2 (dist ~14).
        let emb = [0.0_f32, 0.0, 1.0, 0.0, 10.0, 10.0, 0.0, 1.0];
        let opts = NeighborsOptions {
            n_neighbors: 3,
            ..Default::default()
        };
        let g = neighbors(&emb, 4, 2, &opts).unwrap();
        assert_eq!(g.n_obs, 4);
        // Cell 0's neighbors (after dropping self): indices 1 and 3, both dist 1.0.
        let (start, end) = (g.indptr[0] as usize, g.indptr[1] as usize);
        let nbrs = &g.indices[start..end];
        let dists = &g.data[start..end];
        assert!(nbrs.contains(&1) && nbrs.contains(&3));
        assert!(dists.iter().all(|d| (*d - 1.0).abs() < 1e-5));
    }

    #[test]
    fn rejects_zero_neighbors() {
        let emb = [0.0_f32; 4];
        let opts = NeighborsOptions {
            n_neighbors: 0,
            ..Default::default()
        };
        let err = neighbors(&emb, 2, 2, &opts).unwrap_err();
        assert_eq!(err, NeighborsError::ZeroNeighbors(0));
    }

    #[test]
    fn rejects_empty_embedding() {
        let err = neighbors(&[], 0, 5, &NeighborsOptions::default()).unwrap_err();
        assert!(matches!(err, NeighborsError::EmptyInput { .. }));
    }

    #[test]
    fn neighbors_count_is_correct() {
        // 10 cells, 3 dims, ask for 5 neighbors each.
        let emb: Vec<f32> = (0..30).map(|i| i as f32 * 0.1).collect();
        let opts = NeighborsOptions {
            n_neighbors: 5,
            ..Default::default()
        };
        let g = neighbors(&emb, 10, 3, &opts).unwrap();
        for i in 0..10 {
            let (s, e) = (g.indptr[i] as usize, g.indptr[i + 1] as usize);
            assert_eq!(e - s, 4, "cell {i} should have 4 neighbors (5 incl self)");
        }
    }

    #[test]
    fn self_is_excluded_from_graph() {
        let emb = [0.0_f32, 0.0, 1.0, 0.0, 2.0, 0.0];
        let opts = NeighborsOptions {
            n_neighbors: 2,
            ..Default::default()
        };
        let g = neighbors(&emb, 3, 2, &opts).unwrap();
        for i in 0..3 {
            let (s, e) = (g.indptr[i] as usize, g.indptr[i + 1] as usize);
            for &idx in &g.indices[s..e] {
                assert_ne!(idx as usize, i, "self should not be in neighbor list");
            }
        }
    }

    #[test]
    fn cosine_metric_works() {
        // Two cells pointing the same direction (cos dist 0) vs opposite (cos dist 2).
        let emb = [1.0_f32, 0.0, 2.0, 0.0, -1.0, 0.0];
        let opts = NeighborsOptions {
            n_neighbors: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        let g = neighbors(&emb, 3, 2, &opts).unwrap();
        // Cell 0's nearest should be cell 1 (same direction, dist 0).
        let (s, _e) = (g.indptr[0] as usize, g.indptr[1] as usize);
        assert_eq!(g.indices[s], 1);
        assert!(
            g.data[s].abs() < 1e-5,
            "cosine dist to same-direction should be ~0"
        );
    }
}
