//! Trajectory analysis: pseudotime, diffusion maps, and PAGA.
//!
//! These methods reveal developmental progressions and lineage relationships
//! from single-cell data. All operate on the kNN graph we already compute.

// Graph code reads naturally with (i, j, k, s, e, w) — the textbook names.
#![allow(clippy::many_single_char_names)]

use crate::backend::NeighborGraph;
use faer::Mat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Errors from trajectory analysis.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum TrajectoryError {
    /// Empty graph.
    #[error("empty graph")]
    EmptyGraph,
    /// Root cell index out of bounds.
    #[error("root cell {root} out of bounds for {n_obs} cells")]
    RootOutOfBounds {
        /// Requested root.
        root: usize,
        /// Number of cells.
        n_obs: usize,
    },
}

/// Result of pseudotime computation.
#[derive(Debug, Clone)]
pub struct PseudotimeResult {
    /// Pseudotime value for each cell (0 = root, increasing = later).
    pub pseudotime: Vec<f32>,
    /// The root cell index used.
    pub root: usize,
    /// Number of cells reachable from root.
    pub n_reachable: usize,
}

/// Compute diffusion pseudotime via Dijkstra shortest path from a root cell.
///
/// Uses the symmetrized kNN graph with edge weights = distances. Cells closer
/// to the root have smaller pseudotime values.
///
/// # Errors
/// Returns [`TrajectoryError`] on invalid input.
pub fn pseudotime(graph: &NeighborGraph, root: usize) -> Result<PseudotimeResult, TrajectoryError> {
    let n = graph.n_obs;
    if n == 0 {
        return Err(TrajectoryError::EmptyGraph);
    }
    if root >= n {
        return Err(TrajectoryError::RootOutOfBounds { root, n_obs: n });
    }

    // Symmetrize the kNN graph (it's directed: cell i's neighbor j may not have i).
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        let (s, e) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
        for k in s..e {
            let j = graph.indices[k] as usize;
            let w = graph.data[k];
            adj[i].push((j, w));
            adj[j].push((i, w)); // add reverse edge
        }
    }

    // Dijkstra from root. Use f32::to_bits() as the priority key (IEEE 754
    // bit patterns preserve ordering for non-negative floats, which all
    // distances are).
    let mut dist = vec![f32::INFINITY; n];
    dist[root] = 0.0;
    let mut visited = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0.0_f32.to_bits(), root)));

    while let Some(Reverse((d_bits, u))) = heap.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        let d = f32::from_bits(d_bits);

        for &(v, w) in &adj[u] {
            let new_dist = d + w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                heap.push(Reverse((new_dist.to_bits(), v)));
            }
        }
    }

    // Normalize to [0, 1] for interpretability.
    let max_dist = dist
        .iter()
        .filter(|d| d.is_finite())
        .fold(0.0_f32, |a, b| a.max(*b));
    if max_dist > 0.0 {
        for d in &mut dist {
            if d.is_finite() {
                *d /= max_dist;
            }
        }
    }

    let n_reachable = dist.iter().filter(|d| d.is_finite()).count();

    Ok(PseudotimeResult {
        pseudotime: dist,
        root,
        n_reachable,
    })
}

/// Result of PAGA (partition-based graph abstraction).
#[derive(Debug, Clone)]
pub struct PagaResult {
    /// `n_clusters × n_clusters` connectivity matrix (row-major).
    pub connectivities: Vec<f32>,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Number of cells in each cluster.
    pub cluster_sizes: Vec<usize>,
}

/// Compute PAGA connectivities: cluster-level graph from the kNN graph.
///
/// For each pair of clusters, connectivity = fraction of kNN edges connecting
/// them, normalized by the total possible edges. Higher = more connected.
///
/// # Errors
/// Returns [`TrajectoryError`] on invalid input.
pub fn paga(
    graph: &NeighborGraph,
    labels: &[u32],
    n_clusters: usize,
) -> Result<PagaResult, TrajectoryError> {
    let n = graph.n_obs;
    if n == 0 {
        return Err(TrajectoryError::EmptyGraph);
    }
    if labels.len() != n {
        return Err(TrajectoryError::RootOutOfBounds {
            root: labels.len(),
            n_obs: n,
        });
    }

    let mut cluster_sizes = vec![0usize; n_clusters];
    for &l in labels {
        cluster_sizes[l as usize] += 1;
    }

    // Count edges between cluster pairs (from the directed kNN graph).
    let mut edge_counts = vec![0u32; n_clusters * n_clusters];
    for i in 0..n {
        let (s, e) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
        let ci = labels[i] as usize;
        for k in s..e {
            let cell_j = graph.indices[k] as usize;
            let cj = labels[cell_j] as usize;
            edge_counts[ci * n_clusters + cj] += 1;
        }
    }

    // Normalize: connectivity[i][j] = edges_ij / (size_i * avg_k)
    // A simpler approach: connectivity = edges_ij / min(size_i, size_j)
    let mut connectivities = vec![0.0_f32; n_clusters * n_clusters];
    for i in 0..n_clusters {
        for j in 0..n_clusters {
            if i == j {
                connectivities[i * n_clusters + j] = 1.0;
                continue;
            }
            let edges = edge_counts[i * n_clusters + j] + edge_counts[j * n_clusters + i];
            let denom = (cluster_sizes[i] + cluster_sizes[j]).max(1) as f32;
            connectivities[i * n_clusters + j] = edges as f32 / denom;
        }
    }

    Ok(PagaResult {
        connectivities,
        n_clusters,
        cluster_sizes,
    })
}

/// Result of diffusion map computation.
#[derive(Debug, Clone)]
pub struct DiffusionResult {
    /// `n_obs × n_components` diffusion map embedding.
    pub embedding: Vec<f32>,
    /// Number of components.
    pub n_components: usize,
    /// Eigenvalues (indicative of diffusion timescales).
    pub eigenvalues: Vec<f32>,
}

/// Compute diffusion maps from the kNN graph.
///
/// Builds the row-normalized transition matrix and finds its top eigenvectors.
/// These give a low-dimensional embedding that preserves diffusion geometry.
///
/// # Errors
/// Returns [`TrajectoryError`] on failure.
pub fn diffusion_map(
    graph: &NeighborGraph,
    n_components: usize,
) -> Result<DiffusionResult, TrajectoryError> {
    let n = graph.n_obs;
    if n == 0 {
        return Err(TrajectoryError::EmptyGraph);
    }

    // Symmetrize with Gaussian kernel weights (same as clustering.rs).
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        let (s, e) = (graph.indptr[i] as usize, graph.indptr[i + 1] as usize);
        if s == e {
            continue;
        }
        let dists: Vec<f64> = graph.data[s..e].iter().map(|d| f64::from(*d)).collect();
        let sigma = {
            let mut sorted = dists.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2].max(1e-12)
        };
        let sigma_sq = sigma * sigma;
        for (k, &j) in graph.indices[s..e].iter().enumerate() {
            let w = (-dists[k] * dists[k] / sigma_sq).exp();
            adj[i].push((j as usize, w));
            adj[j as usize].push((i, w));
        }
    }

    // Build dense transition matrix. This is O(n²) memory: fine up to ~10K
    // nodes; larger graphs need a sparse Lanczos path (not implemented yet).
    let mut degree = vec![0.0_f64; n];
    for i in 0..n {
        for &(_, w) in &adj[i] {
            degree[i] += w;
        }
    }

    // Transition matrix P = D^-1 * A (row-normalized).
    let mut p = vec![0.0_f64; n * n];
    for i in 0..n {
        let d = degree[i].max(1e-12);
        for &(j, w) in &adj[i] {
            p[i * n + j] = w / d;
        }
    }

    // Eigendecomposition of P (find top eigenvectors).
    // P is not symmetric, but P+P^T is (and has similar spectral properties
    // for our purposes). Alternatively, use the symmetrized D^-1/2 A D^-1/2.
    let mut sym = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = (p[i * n + j] + p[j * n + i]) / 2.0;
        }
    }

    // Use faer for eigendecomposition.
    let mat = Mat::<f64>::from_fn(n, n, |i, j| sym[i * n + j]);
    let eigen = mat
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| TrajectoryError::EmptyGraph)?;

    let eigenvalues_all = eigen.S().column_vector();
    let u = eigen.U();

    // Take the top n_components eigenvectors (largest eigenvalues).
    // faer returns eigenvalues in ascending order, so take from the end.
    let k = n_components.min(n);
    let mut embedding = vec![0.0_f32; n * k];
    let mut eigenvalues = Vec::with_capacity(k);

    for comp in 0..k {
        let eig_idx = n - 1 - comp; // from largest
        eigenvalues.push(eigenvalues_all[eig_idx] as f32);
        for i in 0..n {
            embedding[i * k + comp] = u[(i, eig_idx)] as f32;
        }
    }

    Ok(DiffusionResult {
        embedding,
        n_components: k,
        eigenvalues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_graph(n: usize) -> NeighborGraph {
        // Chain: 0-1-2-3-...-n-1 (each connected to next).
        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..n {
            if i > 0 {
                indices.push((i - 1) as u32);
                data.push(1.0);
            }
            if i < n - 1 {
                indices.push((i + 1) as u32);
                data.push(1.0);
            }
            indptr.push(indices.len() as u32);
        }
        NeighborGraph {
            indptr,
            indices,
            data,
            n_obs: n,
        }
    }

    #[test]
    fn pseudotime_linear_chain() {
        let graph = make_linear_graph(10);
        let result = pseudotime(&graph, 0).unwrap();
        // Cell 0 is root (pseudotime 0), cell 9 is farthest (pseudotime ~1).
        assert!(result.pseudotime[0].abs() < f32::EPSILON);
        assert!(
            result.pseudotime[9] > 0.9,
            "last cell should be far from root"
        );
        // Monotonically increasing along the chain.
        for i in 1..10 {
            assert!(
                result.pseudotime[i] >= result.pseudotime[i - 1],
                "pseudotime should increase along chain"
            );
        }
    }

    #[test]
    fn pseudotime_from_middle() {
        // 11 nodes (0-10); node 5 is the exact middle.
        let graph = make_linear_graph(11);
        let result = pseudotime(&graph, 5).unwrap();
        let d0 = result.pseudotime[0];
        let d10 = result.pseudotime[10];
        assert!(
            (d0 - d10).abs() < 0.01,
            "cells 0 and 10 should be equidistant from root 5: {d0} vs {d10}"
        );
    }

    #[test]
    fn paga_finds_cluster_connections() {
        // Two clusters: {0,1,2} and {3,4,5} with a bridge edge 2-3.
        let mut indptr = vec![0_u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let edges = vec![
            (0u32, 1u32),
            (0, 2),
            (1, 2), // cluster 0 clique
            (3u32, 4u32),
            (3, 5),
            (4, 5), // cluster 1 clique
            (2, 3), // bridge
        ];
        let mut adj: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for &(a, b) in &edges {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        for i in 0..6u32 {
            if let Some(nbrs) = adj.get(&i) {
                for &j in nbrs {
                    indices.push(j);
                    data.push(1.0);
                }
            }
            indptr.push(indices.len() as u32);
        }
        let graph = NeighborGraph {
            indptr,
            indices,
            data,
            n_obs: 6,
        };
        let labels = vec![0, 0, 0, 1, 1, 1];

        let result = paga(&graph, &labels, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        // Self-connectivity should be 1.0.
        assert!((result.connectivities[0] - 1.0).abs() < f32::EPSILON);
        assert!((result.connectivities[3] - 1.0).abs() < f32::EPSILON);
        // Cross-cluster connectivity should be > 0 (bridge exists).
        assert!(
            result.connectivities[1] > 0.0,
            "bridge should create connectivity"
        );
    }

    #[test]
    fn diffusion_map_produces_embedding() {
        let graph = make_linear_graph(20);
        let result = diffusion_map(&graph, 3).unwrap();
        assert_eq!(result.embedding.len(), 20 * 3);
        assert_eq!(result.n_components, 3);
        assert_eq!(result.eigenvalues.len(), 3);
    }
}
