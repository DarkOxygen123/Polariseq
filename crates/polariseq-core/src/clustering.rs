//! Community detection (Leiden clustering) on the neighbor graph.
//!
//! Given a kNN neighbor graph (from [`crate::neighbors`]), produce per-cell
//! cluster assignments via the Leiden algorithm (Traag, Waltman & van Eck
//! 2019), implemented in [`crate::leiden`] on CSR.
//!
//! Two graph representations are involved:
//! - **Distance graph** (`obsp['distances']`): the raw kNN distances.
//! - **Connectivity graph** (`obsp['connectivities']`): the UMAP fuzzy
//!   simplicial set — a symmetrised, weighted graph. Leiden runs on this,
//!   exactly as scanpy's Leiden runs on connectivities.
//!
//! Modularity is reported on the same weighted graph igraph receives (see
//! [`crate::umap::fuzzy_connectivities`]), so quality numbers are
//! comparable across implementations.

use crate::backend::NeighborGraph;
use crate::leiden as leiden_engine;

/// Options for Leiden clustering.
#[derive(Debug, Clone)]
pub struct LeidenOptions {
    /// Resolution parameter γ. Higher → more, smaller clusters; lower → fewer,
    /// larger clusters. scanpy default: 1.0.
    pub resolution: f64,
    /// Random seed.
    pub seed: u64,
    /// Safety cap on aggregation levels per run (Leiden converges long
    /// before this; kept from the old API).
    pub max_iterations: usize,
}

impl Default for LeidenOptions {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            seed: 0,
            max_iterations: 100,
        }
    }
}

/// The result of Leiden clustering: one community label per cell.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusteringResult {
    /// Community label for each cell (contiguous, size-descending so
    /// cluster 0 is the largest — scanpy's convention).
    pub labels: Vec<u32>,
    /// Number of distinct communities found.
    pub n_communities: usize,
    /// Modularity of the partition on the weighted connectivity graph.
    pub quality: f64,
    /// Undirected edge count of the graph the quality was computed on.
    pub n_edges: usize,
    /// Total edge weight of that graph (m).
    pub total_weight: f64,
}

/// Errors raised by clustering.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ClusteringError {
    /// The neighbor graph is empty.
    #[error("neighbor graph has 0 nodes")]
    Empty,
}

/// Run Leiden clustering on a neighbor graph.
///
/// Converts the distance graph to UMAP connectivities (the same graph
/// `tl.umap` uses), then runs Leiden with igraph's two-iteration scheme.
/// Deterministic for a given seed.
///
/// # Errors
/// Returns [`ClusteringError`] on failure.
pub fn leiden(
    graph: &NeighborGraph,
    opts: &LeidenOptions,
) -> Result<ClusteringResult, ClusteringError> {
    if graph.n_obs == 0 {
        return Err(ClusteringError::Empty);
    }
    Ok(partition(
        &leiden_engine::weighted_from_distances(graph),
        opts,
    ))
}

/// Leiden on a graph already weighted for clustering: a symmetric
/// connectivity CSR (both directions, rows sorted by column), as
/// `pp.neighbors` stores in `obsp['connectivities']` and Scanpy's
/// `sc.tl.leiden` clusters. The graph is used as given; [`leiden`] instead
/// takes kNN *distances* and derives this graph itself.
///
/// # Errors
/// Returns [`ClusteringError::Empty`] if the graph has no nodes.
pub fn leiden_connectivities(
    graph: NeighborGraph,
    opts: &LeidenOptions,
) -> Result<ClusteringResult, ClusteringError> {
    if graph.n_obs == 0 {
        return Err(ClusteringError::Empty);
    }
    Ok(partition(
        &leiden_engine::weighted_from_connectivities(graph),
        opts,
    ))
}

/// [`leiden_connectivities`] on a connectivity CSR the caller holds (both
/// directions, rows sorted by column), read in place: nothing is copied, so
/// the graph can be a memory-mapped file.
///
/// # Errors
/// Returns [`ClusteringError::Empty`] if the graph has no nodes.
pub fn leiden_connectivities_ref(
    indptr: &[u32],
    indices: &[u32],
    data: &[f32],
    n_obs: usize,
    opts: &LeidenOptions,
) -> Result<ClusteringResult, ClusteringError> {
    if n_obs == 0 {
        return Err(ClusteringError::Empty);
    }
    Ok(partition(
        &leiden_engine::WeightedGraph::from_symmetric_csr_ref(indptr, indices, data, n_obs),
        opts,
    ))
}

fn partition(wg: &leiden_engine::WeightedGraph<'_>, opts: &LeidenOptions) -> ClusteringResult {
    let (labels, _, quality) = leiden_engine::leiden_partition(wg, opts.resolution, opts.seed);
    let (labels, n_communities) = leiden_engine::relabel_by_size(&labels);
    ClusteringResult {
        labels,
        n_communities,
        quality,
        n_edges: wg.n_undirected,
        total_weight: wg.two_m / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clustering the caller's arrays in place gives the labels, quality and
    /// counts of clustering an owned copy of them.
    #[test]
    fn clustering_borrowed_arrays_equals_clustering_a_copy() {
        let mut edges = Vec::new();
        for c in 0..4_usize {
            for i in 0..30 {
                for j in (i + 1)..30 {
                    if (i * 7 + j * 3) % 5 != 0 {
                        edges.push((c * 30 + i, c * 30 + j, 0.5 + ((i + j) % 3) as f32 * 0.2));
                    }
                }
            }
            edges.push((c * 30, ((c + 1) % 4) * 30, 0.05));
        }
        let g = make_graph(120, &edges);
        let opts = LeidenOptions {
            resolution: 1.0,
            seed: 3,
            ..Default::default()
        };
        let by_ref = leiden_connectivities_ref(&g.indptr, &g.indices, &g.data, 120, &opts).unwrap();
        let owned = leiden_connectivities(g, &opts).unwrap();
        assert_eq!(by_ref.labels, owned.labels);
        assert_eq!(by_ref.quality.to_bits(), owned.quality.to_bits());
        assert_eq!(
            (by_ref.n_communities, by_ref.n_edges),
            (owned.n_communities, owned.n_edges)
        );
    }

    fn make_graph(n_obs: usize, edges: &[(usize, usize, f32)]) -> NeighborGraph {
        // Build a simple kNN-like CSR from an edge list (symmetric for clustering).
        use std::collections::BTreeMap;
        let mut adj: BTreeMap<usize, Vec<(u32, f32)>> = BTreeMap::new();
        for &(s, d, w) in edges {
            adj.entry(s).or_default().push((d as u32, w));
            adj.entry(d).or_default().push((s as u32, w));
        }
        let mut indptr = vec![0_u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n_obs {
            if let Some(nbrs) = adj.get(&i) {
                for &(idx, w) in nbrs {
                    indices.push(idx);
                    data.push(w);
                }
            }
            indptr.push(indices.len() as u32);
        }
        NeighborGraph {
            indptr,
            indices,
            data,
            n_obs,
        }
    }

    #[test]
    fn clusters_two_clear_groups() {
        // Two well-separated cliques: {0,1,2} and {3,4,5}, with one weak bridge.
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.1),
            (1, 2, 0.1),
            (3, 4, 0.1),
            (3, 5, 0.1),
            (4, 5, 0.1),
            (2, 3, 5.0), // weak bridge (large distance → weak fuzzy weight)
        ];
        let g = make_graph(6, &edges);
        let result = leiden(&g, &LeidenOptions::default()).unwrap();
        assert_eq!(result.n_communities, 2, "should find 2 clusters");
        // Cells 0,1,2 in one cluster; 3,4,5 in the other.
        let c0 = result.labels[0];
        assert_eq!(result.labels[1], c0);
        assert_eq!(result.labels[2], c0);
        let c1 = result.labels[3];
        assert_eq!(result.labels[4], c1);
        assert_eq!(result.labels[5], c1);
        assert_ne!(c0, c1);
    }

    /// Clustering the connectivity graph directly must give exactly what
    /// the distance path gives, since that path builds the same graph first.
    /// Before `leiden_connectivities`, `tl.leiden` handed the connectivities
    /// to the distance path, which re-weighted them as if they were
    /// distances.
    #[test]
    fn connectivities_path_equals_distance_path() {
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.2),
            (1, 2, 0.1),
            (3, 4, 0.1),
            (3, 5, 0.3),
            (4, 5, 0.1),
            (2, 3, 5.0),
            (6, 7, 0.1),
            (6, 8, 0.2),
            (7, 8, 0.15),
            (5, 6, 4.0),
        ];
        let g = make_graph(9, &edges);
        let (indptr, indices, data) = crate::umap::fuzzy_connectivities(&g);
        let conn = NeighborGraph {
            indptr,
            indices,
            data,
            n_obs: 9,
        };
        for seed in [0, 7] {
            let opts = LeidenOptions {
                seed,
                ..Default::default()
            };
            let a = leiden(&g, &opts).unwrap();
            let b = leiden_connectivities(conn.clone(), &opts).unwrap();
            assert_eq!(a.labels, b.labels);
            assert!((a.quality - b.quality).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_empty_graph() {
        let g = NeighborGraph {
            indptr: vec![0],
            indices: vec![],
            data: vec![],
            n_obs: 0,
        };
        let err = leiden(&g, &LeidenOptions::default()).unwrap_err();
        assert_eq!(err, ClusteringError::Empty);
    }

    #[test]
    fn single_cluster_for_fully_connected() {
        // A complete graph K4 — everything connects to everything.
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.1),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.1),
            (2, 3, 0.1),
        ];
        let g = make_graph(4, &edges);
        let result = leiden(&g, &LeidenOptions::default()).unwrap();
        assert_eq!(result.n_communities, 1, "K4 should be one cluster");
    }

    #[test]
    fn labels_are_contiguous_and_size_ordered() {
        // Three equally-sized pairs: {0,1} {2,3} {4,5}; equal sizes keep
        // first-appearance order after size sorting.
        let edges = vec![(0, 1, 0.1), (2, 3, 0.1), (4, 5, 0.1)];
        let g = make_graph(6, &edges);
        let result = leiden(&g, &LeidenOptions::default()).unwrap();
        let max_label = *result.labels.iter().max().unwrap();
        assert_eq!(
            max_label as usize + 1,
            result.n_communities,
            "labels should be 0..n_communities"
        );
        assert_eq!(result.labels, vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn deterministic_across_runs() {
        let edges = vec![
            (0, 1, 0.1),
            (0, 2, 0.1),
            (1, 2, 0.1),
            (3, 4, 0.1),
            (3, 5, 0.1),
            (4, 5, 0.1),
            (2, 3, 5.0),
        ];
        let g = make_graph(6, &edges);
        let a = leiden(
            &g,
            &LeidenOptions {
                seed: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let b = leiden(
            &g,
            &LeidenOptions {
                seed: 7,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a.labels, b.labels);
        assert!((a.quality - b.quality).abs() < 1e-12);
    }
}
